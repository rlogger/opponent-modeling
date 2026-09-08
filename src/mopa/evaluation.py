"""Matched real-environment evaluation of blue controllers against frozen red.

Shared by the simulator-backed planner (Gate 3), the learned TD-MPC controllers
(Gates 4-5), and the fixed-policy / random controls. Every controller sees the
same reset keys and step keys, faces the true frozen predator specialist, and
receives an opponent context chosen by ``context_mode``:

- ``online``       : ``c_t`` from the frozen causal encoder on the real history;
- ``zero``         : all-zero context;
- ``shuffled``     : the online context of a different episode in the batch;
- ``oracle``       : the true objective one-hot;
- ``wrong_oracle`` : a deliberately wrong objective one-hot.

With ``zero_s``, contexts are 8D completed state-action history latents and
oracle modes use the frozen model's train-only class prototypes, not one-hots.
The real opponent is always the specialist policy, never the learned decoder.
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from mopa.context import CONTEXT_DIM, CausalContextEncoder, derangement
from mopa.continuous_data import deterministic_specialist_action, markov_state
from tag_objectives import CONTINUOUS_ACTION_DIM, joint_action_dict
from tag_objectives.teams import freeze_tree

if TYPE_CHECKING:
    from mopa.zero_s import ZeroSOpponent

CONTEXT_MODES = ("online", "zero", "shuffled", "oracle", "wrong_oracle")

# blue_fn(state, obs_dict, context (B, C), carry, key, t) -> (action (B, 2), carry)
BlueController = Callable[[Any, dict, jax.Array, Any, jax.Array, int], tuple[jax.Array, Any]]

__all__ = [
    "CONTEXT_MODES",
    "BlueController",
    "mappo_prey_controller",
    "random_controller",
    "run_matched_episodes",
]


def mappo_prey_controller(params: Any, obs_width: int, prey_name: str) -> BlueController:
    act = jax.jit(lambda o: deterministic_specialist_action(params, o, obs_width))

    def blue(state, obs, context, carry, key, t):  # noqa: ANN001
        del state, context, key, t
        return act(obs[prey_name]), carry

    return blue


def random_controller(prey_name: str) -> BlueController:
    def blue(state, obs, context, carry, key, t):  # noqa: ANN001
        del state, context, t
        n = obs[prey_name].shape[0]
        return jax.random.uniform(key, (n, CONTINUOUS_ACTION_DIM), minval=-1.0, maxval=1.0), carry

    return blue


def run_matched_episodes(
    env: Any,
    red_params: Any,
    blue: BlueController,
    reset_keys: np.ndarray,
    step_seed: np.ndarray,
    *,
    horizon: int,
    context_mode: str,
    label: int,
    encoder: CausalContextEncoder | None = None,
    zero_s: ZeroSOpponent | None = None,
    shuffle_seed: int = 0,
    initial_carry: Any = None,
    record_positions: bool = False,
    record_transitions: bool = False,
) -> dict[str, Any]:
    """Run ``B`` matched episodes; return per-episode metrics and bound checks.

    ``record_transitions=True`` additionally returns the episodes in the
    continuous data contract (``state``, ``blue_action``, ``red_action``,
    ``blue_reward``, ``terminated_capture``, ``truncated_timeout``,
    ``valid_mask``, ``valid_length``, positions) so collected experience can be
    appended to a world-model replay.

    Recorded ``context`` has shape (B, horizon, C): the actual context supplied to
    each decision, before observing that step's opponent action.
    """
    if context_mode not in CONTEXT_MODES:
        raise ValueError(f"context_mode must be one of {CONTEXT_MODES}")
    if horizon < 1 or len(reset_keys) < 1:
        raise ValueError("positive horizon and a nonempty episode batch required")
    if encoder is not None and zero_s is not None:
        raise ValueError("provide one context encoder, not both")
    if context_mode in {"online", "shuffled"} and encoder is None and zero_s is None:
        raise ValueError("online/shuffled context requires a frozen encoder")
    if context_mode == "shuffled" and len(reset_keys) < 2:
        raise ValueError("shuffled context requires at least two episodes")
    pred_name, prey_name = env.adversaries[0], env.good_agents[0]
    pred_index, prey_index = env.agents.index(pred_name), env.agents.index(prey_name)
    obs_width = max(env.observation_space(a).shape[0] for a in env.agents)
    red_act = jax.jit(lambda o: deterministic_specialist_action(red_params, o, obs_width))
    step_fn = jax.jit(jax.vmap(env.step_env))
    state_fn = jax.jit(lambda s: markov_state(env, s))

    batch = len(reset_keys)
    obs, state = jax.vmap(env.reset)(jnp.asarray(reset_keys, jnp.uint32))
    step_seed_j = jnp.asarray(step_seed, jnp.uint32)
    done = np.zeros(batch, dtype=bool)
    ret = np.zeros(batch, dtype=np.float64)
    pred_lava = np.zeros(batch, dtype=np.float32)
    prey_lava = np.zeros(batch, dtype=np.float32)
    prey_hist = [np.asarray(state.p_pos[:, prey_index])]
    pred_hist = [np.asarray(state.p_pos[:, pred_index : pred_index + 1])]
    action_abs_max = 0.0
    acted_after_done = False
    carry = initial_carry
    shuffle = (
        derangement(np.arange(batch, dtype=np.int32), np.random.default_rng(shuffle_seed))
        if context_mode == "shuffled"
        else None
    )
    key = jax.random.PRNGKey(int(shuffle_seed) + 7)
    prototypes = zero_s.prototypes if zero_s is not None else np.eye(CONTEXT_DIM, dtype=np.float32)
    context_dim = prototypes.shape[-1]
    context_carry = zero_s.initial_context(batch) if zero_s is not None else None
    update_context = jax.jit(zero_s.update_context) if zero_s is not None else None
    controller_seconds = []
    contexts = []
    rec: dict[str, list[np.ndarray]] = {k: [] for k in ("state", "blue", "red", "reward", "term", "trunc", "valid")}
    if record_transitions:
        rec["state"].append(np.asarray(state_fn(state)))

    for t in range(horizon):
        active = ~done
        if context_mode == "zero":
            context = np.zeros((batch, context_dim), dtype=np.float32)
        elif context_mode == "oracle":
            context = np.repeat(prototypes[label][None], batch, axis=0)
        elif context_mode == "wrong_oracle":
            context = np.repeat(prototypes[(label + 1) % len(prototypes)][None], batch, axis=0)
        else:
            context = np.asarray(context_carry.context) if zero_s is not None else encoder.online_context(
                prey_hist, pred_hist, done, np.asarray(state.capture_t), horizon=horizon
            )
            if context_mode == "shuffled":
                context = context[shuffle]
        if record_transitions:
            contexts.append(np.asarray(context))
        key, k_blue = jax.random.split(key)
        start = time.perf_counter()
        blue_action, carry = blue(state, obs, jnp.asarray(context), carry, k_blue, t)
        blue_np = np.asarray(blue_action, dtype=np.float32)
        controller_seconds.append(time.perf_counter() - start)
        if blue_np.shape != (batch, CONTINUOUS_ACTION_DIM) or not np.isfinite(blue_np).all():
            raise ValueError("controller must return finite actions with shape (batch, 2)")
        if np.any(np.abs(blue_np) > 1.0 + 1e-6):
            raise ValueError("controller actions must lie in [-1, 1]")
        action_abs_max = max(action_abs_max, float(np.abs(blue_np).max(initial=0.0)))
        blue_np = np.where(active[:, None], blue_np, 0.0)
        red_action = jnp.where(jnp.asarray(active[:, None]), red_act(obs[pred_name]), 0.0)
        keys = jax.vmap(lambda k: jax.random.fold_in(k, t))(step_seed_j)
        actions = joint_action_dict(env, jnp.asarray(blue_np), red_action[:, None, :])
        new_obs, new_state, rew, dones, info = step_fn(keys, state, actions)
        if update_context is not None:
            # Consume the actual completed pair, including a terminating action.
            # Neither the next state nor any imagined action enters this history.
            context_carry = update_context(context_carry, state_fn(state), red_action, jnp.asarray(active))
        reward_np = np.asarray(rew[prey_name])
        ret += reward_np * active
        pred_lava += np.asarray(info["pred_lava"][:, pred_index]) * active
        prey_lava += np.asarray(info["prey_lava"][:, prey_index]) * active
        if record_transitions:
            capture_now = np.asarray(info["captured"][:, prey_index] > 0.5)
            ep_done = np.asarray(dones["__all__"]) | (t == horizon - 1)
            rec["blue"].append(np.where(active[:, None], blue_np, 0.0).astype(np.float32))
            rec["red"].append(np.where(active[:, None], np.asarray(red_action), 0.0).astype(np.float32))
            rec["reward"].append(np.where(active, reward_np, 0.0).astype(np.float32))
            rec["term"].append(active & capture_now)
            rec["trunc"].append(active & ep_done & ~capture_now)
            rec["valid"].append(active.copy())
        active_j = jnp.asarray(active)
        state = freeze_tree(active_j, new_state, state)
        obs = freeze_tree(active_j, new_obs, obs)
        done = done | np.asarray(dones["__all__"])
        prey_hist.append(np.asarray(state.p_pos[:, prey_index]))
        pred_hist.append(np.asarray(state.p_pos[:, pred_index : pred_index + 1]))
        if record_transitions:
            rec["state"].append(np.asarray(state_fn(state)))

    capture_t = np.asarray(state.capture_t)
    captured = capture_t >= 0
    out: dict[str, Any] = {
        "blue_return": ret.astype(np.float32),
        "captured": captured.astype(np.float32),
        "survival_time": np.where(captured, capture_t, horizon).astype(np.float32),
        "resources_collected": np.asarray(state.collected).sum(-1).astype(np.float32),
        "pred_lava_steps": pred_lava,
        "prey_lava_steps": prey_lava,
        "pred_coverage": np.asarray(state.visited).sum(-1).astype(np.float32),
        "blue_action_abs_max": action_abs_max,
        "acted_after_done": acted_after_done,
        "controller_seconds_per_batch": np.asarray(controller_seconds),
    }
    if record_positions or record_transitions:
        out["prey_pos"] = np.stack(prey_hist, axis=1)
        out["pred_pos"] = np.stack(pred_hist, axis=1)
    if record_transitions:
        valid = np.stack(rec["valid"], axis=1)
        out["transitions"] = {
            "state": np.stack(rec["state"], axis=1).astype(np.float32),
            "blue_action": np.stack(rec["blue"], axis=1),
            "red_action": np.stack(rec["red"], axis=1),
            "blue_reward": np.stack(rec["reward"], axis=1),
            "terminated_capture": np.stack(rec["term"], axis=1),
            "truncated_timeout": np.stack(rec["trunc"], axis=1),
            "valid_mask": valid,
            "valid_length": valid.sum(axis=1).astype(np.int32),
            "capture_t": capture_t.astype(np.int32),
            "prey_pos": out["prey_pos"].astype(np.float32),
            "pred_pos": out["pred_pos"].astype(np.float32),
            "context": np.stack(contexts, axis=1).astype(np.float32),
        }
    return out
