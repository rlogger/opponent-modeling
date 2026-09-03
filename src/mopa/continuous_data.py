"""Continuous joint-action trajectories from frozen MAPPO specialists (Gate 1).

Implements the handoff data contract: every episode stores a Markov-equivalent
state vector, both agents' exact observations, two-dimensional ``[-1, 1]``
blue and red actions, the blue reward, and separate ``terminated_capture`` /
``truncated_timeout`` flags, all keyed by the reset and step PRNG keys so the
episode can be replayed exactly through the simulator.

Fixed protocol decisions (handoff):

- blue is the controlled prey; red is the frozen predator specialist;
- the prey checkpoint family is fixed to ``capture`` (same checkpoint seed) so
  the objective label cannot identify three different prey policies;
- rollouts use each frozen specialist's deterministic policy mean
  ``tanh(mean)``; sampled rollouts are a separately declared arm;
- reset keys are matched across the three objective types.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from mopa.features import EP_LEN
from mopa.nets import ContinuousActor
from mopa.types import ContinuousTrajectoryDataset
from tag_objectives import (
    CONTINUOUS_ACTION_DIM,
    SimpleTagObjectivesMPE,
    joint_action_dict,
    make_env,
)
from tag_objectives.teams import freeze_tree

OBJECTIVE_TYPES: tuple[str, ...] = ("capture", "risk", "curious")
DEFAULT_CONTINUOUS_LOGDIR = Path("logs") / "MPE_simple_tag_v3_continuous"
ENV_NAME = "MPE_simple_tag_v3_continuous"
HIDDEN = 128

# Markov-equivalent state used as the world-model observation. Components that
# affect blue's transition or reward are all present; the predator's novelty
# grid (``visited``) is excluded because it enters only the predator's reward
# and is not an input to the frozen predator policy, so it cannot influence any
# transition or blue reward.
MARKOV_STATE_FIELDS: tuple[str, ...] = (
    "agent_positions",  # (P + 1) * 2
    "agent_velocities",  # (P + 1) * 2
    "resource_positions",  # num_resources * 2
    "resource_collected",  # num_resources
    "lava_positions",  # num_lava * 2
    "lava_radii",  # num_lava
    "time_fraction",  # step / max_steps
)

__all__ = [
    "DEFAULT_CONTINUOUS_LOGDIR",
    "ENV_NAME",
    "MARKOV_STATE_FIELDS",
    "OBJECTIVE_TYPES",
    "ContinuousTrajectoryDataset",
    "continuous_checkpoint_path",
    "continuous_objective_dataset",
    "deterministic_specialist_action",
    "family_behavior_summary",
    "load_continuous_actor_params",
    "markov_state",
    "markov_state_dim",
    "pad_obs",
    "replay_episodes",
    "rollout_continuous_checkpoint",
    "state_action_coverage",
    "validate_continuous_dataset",
]


# --------------------------------------------------------------------------- #
# Checkpoints and policies
# --------------------------------------------------------------------------- #
def continuous_checkpoint_path(
    logdir: Path | str, pred_type: str, team: str, seed_idx: int
) -> Path:
    alg = f"mappo_continuous_{pred_type}"
    return Path(logdir) / f"{alg}_{ENV_NAME}_{team}_actor_seed0_vmap{seed_idx}.safetensors"


def load_continuous_actor_params(path: Path | str) -> Any:
    from jaxmarl.wrappers.baselines import load_params

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"missing continuous specialist checkpoint: {path}")
    return load_params(str(path))


def pad_obs(obs: jax.Array, width: int) -> jax.Array:
    """Zero-pad the trailing observation axis to ``width`` (trainer convention)."""
    if obs.shape[-1] >= width:
        return obs
    pad = width - obs.shape[-1]
    return jnp.concatenate([obs, jnp.zeros(obs.shape[:-1] + (pad,), obs.dtype)], -1)


def deterministic_specialist_action(
    params: Any, obs: jax.Array, obs_width: int, hidden: int = HIDDEN
) -> jax.Array:
    """``tanh(mean)`` of a continuous MAPPO actor on (padded) observations."""
    actor = ContinuousActor(action_dim=CONTINUOUS_ACTION_DIM, hidden_dim=hidden)
    mean, _ = actor.apply(params, pad_obs(obs, obs_width))
    return jnp.tanh(mean)


# --------------------------------------------------------------------------- #
# Markov state
# --------------------------------------------------------------------------- #
def markov_state_dim(env: SimpleTagObjectivesMPE) -> int:
    n = env.num_agents
    return 4 * n + 3 * env.num_resources + 3 * env.num_lava + 1


def markov_state(env: SimpleTagObjectivesMPE, state: Any) -> jax.Array:
    """Flatten an ``ObjectiveState`` (optionally batched) into the Markov vector."""
    n = env.num_agents
    pos = jnp.asarray(state.p_pos)[..., :n, :]
    vel = jnp.asarray(state.p_vel)[..., :n, :]
    lead = pos.shape[:-2]
    parts = [
        pos.reshape(*lead, 2 * n),
        vel.reshape(*lead, 2 * n),
        jnp.asarray(state.resource_pos).reshape(*lead, 2 * env.num_resources),
        jnp.asarray(state.collected).astype(jnp.float32),
        jnp.asarray(state.lava_pos).reshape(*lead, 2 * env.num_lava),
        jnp.asarray(state.lava_rad),
        (jnp.asarray(state.step).astype(jnp.float32) / float(env.max_steps))[
            ..., None
        ],
    ]
    return jnp.concatenate(parts, axis=-1).astype(jnp.float32)


# --------------------------------------------------------------------------- #
# Rollouts
# --------------------------------------------------------------------------- #
def _step_keys(step_seed: jax.Array, t: int) -> jax.Array:
    return jax.vmap(lambda k: jax.random.fold_in(k, t))(step_seed)


def rollout_continuous_checkpoint(
    pred_type: str,
    seed_idx: int,
    num_eps: int,
    rng_key: jax.Array,
    *,
    logdir: Path | str = DEFAULT_CONTINUOUS_LOGDIR,
    num_steps: int = EP_LEN,
    prey_type: str = "capture",
    deterministic: bool = True,
) -> dict[str, np.ndarray]:
    """Roll one frozen predator specialist against the fixed prey family.

    Returns the per-episode arrays of :class:`ContinuousTrajectoryDataset`
    (without labels). ``rng_key`` seeds the reset keys, the per-episode step
    keys, and (if ``deterministic=False``) the policy noise.
    """
    env = make_env(pred_type, continuous=True)
    if env.num_adversaries != 1:
        raise ValueError("the continuous data contract stores one red action per step")
    pred_name = env.adversaries[0]
    prey_name = env.good_agents[0]
    pred_index = env.agents.index(pred_name)
    prey_index = env.agents.index(prey_name)
    obs_width = max(env.observation_space(a).shape[0] for a in env.agents)
    logdir = Path(logdir)
    pred_params = load_continuous_actor_params(
        continuous_checkpoint_path(logdir, pred_type, "pred", seed_idx)
    )
    prey_params = load_continuous_actor_params(
        continuous_checkpoint_path(logdir, prey_type, "prey", seed_idx)
    )
    actor = ContinuousActor(action_dim=CONTINUOUS_ACTION_DIM, hidden_dim=HIDDEN)

    rng_key, k_reset, k_step, k_policy = jax.random.split(rng_key, 4)
    reset_keys = jax.random.split(k_reset, num_eps)
    step_seed = jax.random.split(k_step, num_eps)
    obs, state = jax.vmap(env.reset)(reset_keys)

    @jax.jit
    def act(params, o, key):
        mean, log_std = actor.apply(params, pad_obs(o, obs_width))
        if deterministic:
            return jnp.tanh(mean)
        u = mean + jnp.exp(log_std) * jax.random.normal(key, mean.shape)
        return jnp.tanh(u)

    step_fn = jax.jit(jax.vmap(env.step_env))
    state_fn = jax.jit(lambda s: markov_state(env, s))

    states = [np.asarray(state_fn(state))]
    blue_obs = [np.asarray(obs[prey_name])]
    red_obs = [np.asarray(obs[pred_name])]
    prey_pos = [np.asarray(state.p_pos[:, prey_index])]
    pred_pos = [np.asarray(state.p_pos[:, pred_index : pred_index + 1])]
    blue_actions, red_actions, rewards = [], [], []
    terminated, truncated, valid = [], [], []

    done = jnp.zeros((num_eps,), dtype=bool)
    pred_lava_steps = jnp.zeros((num_eps,), dtype=jnp.float32)
    prey_lava_steps = jnp.zeros((num_eps,), dtype=jnp.float32)
    for t in range(num_steps):
        active = ~done
        k_policy, k_blue, k_red = jax.random.split(k_policy, 3)
        blue_a = act(prey_params, obs[prey_name], k_blue)
        red_a = act(pred_params, obs[pred_name], k_red)
        actions = joint_action_dict(env, blue_a, red_a[:, None, :])
        new_obs, new_state, rew, dones, info = step_fn(_step_keys(step_seed, t), state, actions)

        capture_now = info["captured"][:, prey_index] > 0.5
        # Truncation = env time limit or the recording horizon, without capture.
        episode_done = dones["__all__"] | (t == num_steps - 1)
        active_f = active.astype(jnp.float32)
        pred_lava_steps += info["pred_lava"][:, pred_index] * active_f
        prey_lava_steps += info["prey_lava"][:, prey_index] * active_f

        blue_actions.append(np.asarray(jnp.where(active[:, None], blue_a, 0.0)))
        red_actions.append(np.asarray(jnp.where(active[:, None], red_a, 0.0)))
        rewards.append(np.asarray(jnp.where(active, rew[prey_name], 0.0)))
        terminated.append(np.asarray(active & capture_now))
        truncated.append(np.asarray(active & episode_done & ~capture_now))
        valid.append(np.asarray(active))

        state = freeze_tree(active, new_state, state)
        obs = freeze_tree(active, new_obs, obs)
        done = done | episode_done

        states.append(np.asarray(state_fn(state)))
        blue_obs.append(np.asarray(obs[prey_name]))
        red_obs.append(np.asarray(obs[pred_name]))
        prey_pos.append(np.asarray(state.p_pos[:, prey_index]))
        pred_pos.append(np.asarray(state.p_pos[:, pred_index : pred_index + 1]))

    capture_t = np.asarray(state.capture_t).astype(np.int32)
    captured = capture_t >= 0
    valid_length = np.where(captured, capture_t, num_steps).astype(np.int32)
    valid_mask = np.stack(valid, axis=1)
    if not np.array_equal(valid_mask.sum(axis=1), valid_length):
        raise AssertionError("valid_mask does not match capture-derived valid_length")
    return dict(
        state=np.stack(states, axis=1).astype(np.float32),
        blue_observation=np.stack(blue_obs, axis=1).astype(np.float32),
        red_observation=np.stack(red_obs, axis=1).astype(np.float32),
        blue_action=np.stack(blue_actions, axis=1).astype(np.float32),
        red_action=np.stack(red_actions, axis=1).astype(np.float32),
        blue_reward=np.stack(rewards, axis=1).astype(np.float32),
        terminated_capture=np.stack(terminated, axis=1).astype(bool),
        truncated_timeout=np.stack(truncated, axis=1).astype(bool),
        valid_mask=valid_mask.astype(bool),
        causal_context=np.zeros((num_eps, num_steps + 1, 0), dtype=np.float32),
        environment_seed=np.asarray(reset_keys, dtype=np.uint32),
        step_seed=np.asarray(step_seed, dtype=np.uint32),
        prey_pos=np.stack(prey_pos, axis=1).astype(np.float32),
        pred_pos=np.stack(pred_pos, axis=1).astype(np.float32),
        lava_pos=np.asarray(state.lava_pos).astype(np.float32),
        lava_rad=np.asarray(state.lava_rad).astype(np.float32),
        capture_t=capture_t,
        captured=captured.astype(np.int32),
        survival_time=np.where(captured, capture_t, num_steps).astype(np.float32),
        pred_lava_steps=np.asarray(pred_lava_steps).astype(np.float32),
        prey_lava_steps=np.asarray(prey_lava_steps).astype(np.float32),
        resources_collected=np.asarray(
            jnp.sum(state.collected.astype(jnp.float32), axis=-1)
        ).astype(np.float32),
        pred_coverage=np.asarray(
            jnp.sum(state.visited.astype(jnp.float32), axis=-1)
        ).astype(np.float32),
        valid_length=valid_length,
    )


def continuous_objective_dataset(
    n_eps: int = 200,
    ckpt_seeds: Sequence[int] = (0, 1, 2),
    rng0: int = 0,
    num_steps: int | None = None,
    logdir: Path | str = DEFAULT_CONTINUOUS_LOGDIR,
    prey_type: str = "capture",
    deterministic: bool = True,
) -> ContinuousTrajectoryDataset:
    """Matched continuous rollouts for every objective type and checkpoint seed.

    The PRNG key for ``(type, seed)`` is ``PRNGKey(rng0 + seed)`` for every type,
    so the three objective types share reset keys and step keys exactly.
    """
    steps = int(num_steps if num_steps is not None else EP_LEN)
    rows: dict[str, list[np.ndarray]] = {}
    for lab, pred_type in enumerate(OBJECTIVE_TYPES):
        for seed in ckpt_seeds:
            d = rollout_continuous_checkpoint(
                pred_type,
                seed,
                n_eps,
                jax.random.PRNGKey(rng0 + seed),
                logdir=logdir,
                num_steps=steps,
                prey_type=prey_type,
                deterministic=deterministic,
            )
            n = len(d["state"])
            d["objective_label"] = np.full(n, lab, np.int32)
            d["checkpoint_seed"] = np.full(n, seed, np.int32)
            for key, value in d.items():
                rows.setdefault(key, []).append(value)
    stacked = {key: np.concatenate(vals) for key, vals in rows.items()}
    return ContinuousTrajectoryDataset(**stacked)


# --------------------------------------------------------------------------- #
# Validation: exact replay, saturation, coverage, behavioural separation
# --------------------------------------------------------------------------- #
def replay_episodes(
    ds: ContinuousTrajectoryDataset | dict[str, np.ndarray],
    indices: Sequence[int] | np.ndarray | None = None,
) -> dict[str, float]:
    """Re-step stored joint actions through the simulator and compare.

    Returns the maximum absolute deviation of the Markov state, both
    observations, and the blue reward over all valid transitions, plus the
    fraction of episodes whose replayed capture/timeout flags match exactly.
    """
    d = ds if isinstance(ds, dict) else ds.as_dict()
    labels = np.asarray(d["objective_label"])
    n_total = len(labels)
    idx = np.arange(n_total) if indices is None else np.asarray(indices, dtype=np.int64)
    horizon = int(d["blue_action"].shape[1])
    worst = {"state": 0.0, "blue_observation": 0.0, "red_observation": 0.0, "blue_reward": 0.0}
    flags_ok = 0
    for lab, pred_type in enumerate(OBJECTIVE_TYPES):
        rows = idx[labels[idx] == lab]
        if len(rows) == 0:
            continue
        env = make_env(pred_type, continuous=True)
        prey_name, pred_name = env.good_agents[0], env.adversaries[0]
        prey_index = env.agents.index(prey_name)
        step_fn = jax.jit(jax.vmap(env.step_env))
        state_fn = jax.jit(lambda s: markov_state(env, s))
        reset_keys = jnp.asarray(d["environment_seed"][rows], dtype=jnp.uint32)
        step_seed = jnp.asarray(d["step_seed"][rows], dtype=jnp.uint32)
        obs, state = jax.vmap(env.reset)(reset_keys)
        worst["state"] = max(
            worst["state"],
            float(np.max(np.abs(np.asarray(state_fn(state)) - d["state"][rows, 0]))),
        )
        valid = np.asarray(d["valid_mask"][rows])
        done = np.zeros(len(rows), dtype=bool)
        term_ok = np.ones(len(rows), dtype=bool)
        for t in range(horizon):
            active = ~done
            if not np.any(active):
                break
            blue_a = jnp.asarray(d["blue_action"][rows, t])
            red_a = jnp.asarray(d["red_action"][rows, t])
            actions = joint_action_dict(env, blue_a, red_a[:, None, :])
            new_obs, new_state, rew, dones, info = step_fn(_step_keys(step_seed, t), state, actions)
            act_rows = np.flatnonzero(active & valid[:, t])
            if len(act_rows):
                worst["state"] = max(
                    worst["state"],
                    float(
                        np.max(
                            np.abs(
                                np.asarray(state_fn(new_state))[act_rows]
                                - d["state"][rows[act_rows], t + 1]
                            )
                        )
                    ),
                )
                worst["blue_observation"] = max(
                    worst["blue_observation"],
                    float(
                        np.max(
                            np.abs(
                                np.asarray(new_obs[prey_name])[act_rows]
                                - d["blue_observation"][rows[act_rows], t + 1]
                            )
                        )
                    ),
                )
                worst["red_observation"] = max(
                    worst["red_observation"],
                    float(
                        np.max(
                            np.abs(
                                np.asarray(new_obs[pred_name])[act_rows]
                                - d["red_observation"][rows[act_rows], t + 1]
                            )
                        )
                    ),
                )
                worst["blue_reward"] = max(
                    worst["blue_reward"],
                    float(
                        np.max(
                            np.abs(
                                np.asarray(rew[prey_name])[act_rows]
                                - d["blue_reward"][rows[act_rows], t]
                            )
                        )
                    ),
                )
                cap = np.asarray(info["captured"][:, prey_index] > 0.5)
                ep_done = np.asarray(dones["__all__"]) | (t == horizon - 1)
                term_ok[act_rows] &= (
                    cap[act_rows] == d["terminated_capture"][rows[act_rows], t]
                ) & ((ep_done & ~cap)[act_rows] == d["truncated_timeout"][rows[act_rows], t])
            state = freeze_tree(jnp.asarray(active), new_state, state)
            obs = freeze_tree(jnp.asarray(active), new_obs, obs)
            done = done | np.asarray(dones["__all__"])
        flags_ok += int(term_ok.sum())
    return {
        **{f"max_abs_error_{k}": v for k, v in worst.items()},
        "termination_flags_match_fraction": flags_ok / max(len(idx), 1),
        "n_replayed": int(len(idx)),
    }


def state_action_coverage(
    ds: ContinuousTrajectoryDataset | dict[str, np.ndarray],
    *,
    saturation_threshold: float = 0.95,
    grid: int = 16,
    arena: float = 2.0,
) -> dict[str, Any]:
    """Action saturation and state coverage diagnostics before world-model fitting."""
    d = ds if isinstance(ds, dict) else ds.as_dict()
    valid = np.asarray(d["valid_mask"], dtype=bool)
    out: dict[str, Any] = {"n_valid_transitions": int(valid.sum())}
    for name in ("blue_action", "red_action"):
        a = np.asarray(d[name])[valid]
        out[name] = {
            "mean": a.mean(axis=0).tolist(),
            "std": a.std(axis=0).tolist(),
            "abs_mean": np.abs(a).mean(axis=0).tolist(),
            "saturation_fraction": (np.abs(a) > saturation_threshold).mean(axis=0).tolist(),
            "either_axis_saturated_fraction": float(
                (np.abs(a) > saturation_threshold).any(axis=1).mean()
            ),
            "min": a.min(axis=0).tolist(),
            "max": a.max(axis=0).tolist(),
        }
    # Blue and red position coverage of a grid over the arena.
    for name, pos in (("prey", d["prey_pos"]), ("pred", d["pred_pos"][:, :, 0])):
        p = np.asarray(pos)[:, :-1][valid]
        cells = np.clip(((p + arena) / (2 * arena) * grid).astype(int), 0, grid - 1)
        flat = cells[:, 0] * grid + cells[:, 1]
        out[f"{name}_grid_cells_visited_fraction"] = float(len(np.unique(flat)) / grid**2)
    out["capture_transition_fraction"] = float(
        np.asarray(d["terminated_capture"])[valid].mean()
    )
    out["timeout_transition_fraction"] = float(
        np.asarray(d["truncated_timeout"])[valid].mean()
    )
    return out


def family_behavior_summary(
    ds: ContinuousTrajectoryDataset | dict[str, np.ndarray],
) -> dict[str, Any]:
    """Per-objective behavioural metrics plus a leave-one-checkpoint-out probe.

    The probe is a multinomial logistic regression from the per-episode
    behaviour vector (capture, survival, lava steps, coverage, resources) to
    the objective label, trained on two checkpoint seeds and scored on the
    third. Chance is 1/3; clearly separable families score far above it.
    """
    d = ds if isinstance(ds, dict) else ds.as_dict()
    labels = np.asarray(d["objective_label"])
    ckpt = np.asarray(d["checkpoint_seed"])
    metric_names = (
        "captured",
        "survival_time",
        "pred_lava_steps",
        "pred_coverage",
        "resources_collected",
        "prey_lava_steps",
    )
    per_type: dict[str, Any] = {}
    for lab, name in enumerate(OBJECTIVE_TYPES):
        rows = labels == lab
        per_type[name] = {
            m: {
                "mean": float(np.mean(d[m][rows])),
                "std": float(np.std(d[m][rows])),
            }
            for m in metric_names
        }
        per_type[name]["n_episodes"] = int(rows.sum())
    features = np.stack([np.asarray(d[m], dtype=np.float64) for m in metric_names], -1)
    probe: dict[str, Any] = {"chance": 1.0 / len(OBJECTIVE_TYPES), "folds": []}
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        accs = []
        for heldout in np.unique(ckpt):
            train = ckpt != heldout
            scaler = StandardScaler().fit(features[train])
            clf = LogisticRegression(max_iter=2000).fit(
                scaler.transform(features[train]), labels[train]
            )
            acc = float(clf.score(scaler.transform(features[~train]), labels[~train]))
            accs.append(acc)
            probe["folds"].append({"heldout_checkpoint": int(heldout), "accuracy": acc})
        probe["mean_accuracy"] = float(np.mean(accs))
    except ImportError:  # pragma: no cover - scikit-learn is a core dependency
        probe["mean_accuracy"] = None
    return {"per_type": per_type, "behaviour_probe": probe, "metrics": list(metric_names)}


def validate_continuous_dataset(
    ds: ContinuousTrajectoryDataset | dict[str, np.ndarray],
    *,
    replay_indices: Sequence[int] | np.ndarray | None = None,
    replay_atol: float = 1e-5,
) -> dict[str, Any]:
    """Structural checks for the continuous data contract plus exact replay."""
    d = ds if isinstance(ds, dict) else ds.as_dict()
    n, t_plus, _ = d["state"].shape
    horizon = t_plus - 1
    checks = {
        "state": (n, t_plus, None, np.float32),
        "blue_observation": (n, t_plus, None, np.float32),
        "red_observation": (n, t_plus, None, np.float32),
        "blue_action": (n, horizon, CONTINUOUS_ACTION_DIM, np.float32),
        "red_action": (n, horizon, CONTINUOUS_ACTION_DIM, np.float32),
    }
    for name, (n_exp, t_exp, last, dtype) in checks.items():
        arr = np.asarray(d[name])
        if arr.dtype != dtype:
            raise ValueError(f"{name} must be {dtype.__name__}, got {arr.dtype}")
        if arr.shape[0] != n_exp or arr.shape[1] != t_exp or (last and arr.shape[2] != last):
            raise ValueError(f"{name} has shape {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must be finite")
    for name in ("blue_action", "red_action"):
        if np.any(np.abs(d[name]) > 1.0):
            raise ValueError(f"{name} leaves [-1, 1]")
    for name in ("terminated_capture", "truncated_timeout", "valid_mask"):
        arr = np.asarray(d[name])
        if arr.dtype != bool or arr.shape != (n, horizon):
            raise ValueError(f"{name} must be bool [N, T]")
    valid = np.asarray(d["valid_mask"])
    lengths = valid.sum(axis=1)
    if not np.array_equal(valid, np.arange(horizon)[None, :] < lengths[:, None]):
        raise ValueError("valid_mask must be a contiguous prefix")
    if not np.array_equal(lengths, np.asarray(d["valid_length"])):
        raise ValueError("valid_length must equal the valid_mask prefix length")
    term = np.asarray(d["terminated_capture"])
    trunc = np.asarray(d["truncated_timeout"])
    if np.any(term & trunc):
        raise ValueError("a transition cannot be both terminated and truncated")
    if np.any((term | trunc) & ~valid):
        raise ValueError("termination flags must lie inside the valid prefix")
    ends = term | trunc
    if not np.array_equal(ends.sum(axis=1), np.ones(n, dtype=int)):
        raise ValueError("every episode must end with exactly one terminal/truncation flag")
    end_index = np.argmax(ends, axis=1)
    if not np.array_equal(end_index, lengths - 1):
        raise ValueError("the terminal flag must sit at the last valid transition")
    if not np.array_equal(term.any(axis=1), np.asarray(d["captured"]).astype(bool)):
        raise ValueError("terminated_capture disagrees with captured")
    if np.asarray(d["blue_reward"]).shape != (n, horizon):
        raise ValueError("blue_reward must be [N, T]")
    if np.any(np.abs(np.asarray(d["blue_reward"])[~valid]) > 0):
        raise ValueError("padded rewards must be zero")
    labels = np.asarray(d["objective_label"])
    ckpt = np.asarray(d["checkpoint_seed"])
    env_seed = np.asarray(d["environment_seed"])
    if env_seed.shape != (n, 2) or env_seed.dtype != np.uint32:
        raise ValueError("environment_seed must be uint32 [N, 2]")
    # Matched resets: identical reset keys across the three labels per checkpoint.
    for seed in np.unique(ckpt):
        groups = [env_seed[(ckpt == seed) & (labels == lab)] for lab in range(3)]
        if any(len(g) != len(groups[0]) for g in groups):
            raise ValueError("objective groups must have equal sizes per checkpoint")
        if any(not np.array_equal(g, groups[0]) for g in groups[1:]):
            raise ValueError("reset keys must match across objective types")
    replay = replay_episodes(d, replay_indices)
    worst = max(v for k, v in replay.items() if k.startswith("max_abs_error_"))
    if worst > replay_atol or replay["termination_flags_match_fraction"] < 1.0:
        raise ValueError(f"exact replay failed: {replay}")
    return {
        "n_episodes": int(n),
        "horizon": int(horizon),
        "state_dim": int(d["state"].shape[-1]),
        "blue_observation_dim": int(d["blue_observation"].shape[-1]),
        "red_observation_dim": int(d["red_observation"].shape[-1]),
        "context_dim": int(d["causal_context"].shape[-1]),
        "n_valid_transitions": int(valid.sum()),
        "capture_rate": float(np.asarray(d["captured"]).mean()),
        "replay": replay,
    }
