"""Objective-typed specialist rollouts from MAPPO checkpoints.

Rolls out capture / risk / curious actors greedily with ``step_env`` (no
auto-reset), freezing finished episodes so each trajectory describes the first
episode only.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from mopa.features import EP_LEN, occupancy, standardize, window
from mopa.nets import ActorLogits
from mopa.types import ObjectiveDataset, ObjectiveObservationDataset
from tag_objectives import SimpleTagObjectivesMPE
from tag_objectives.teams import freeze_tree

OBJECTIVE_TYPES: tuple[str, ...] = ("capture", "risk", "curious")
DEFAULT_LOGDIR = Path("logs") / "MPE_simple_tag_v3"
HIDDEN = 128

__all__ = [
    "EP_LEN",
    "OBJECTIVE_TYPES",
    "ObjectiveDataset",
    "objective_dataset",
    "occupancy",
    "rollout_one_checkpoint",
    "standardize",
    "window",
]


def _pad_obs(obs, width: int):
    if obs.shape[-1] >= width:
        return obs
    pad = width - obs.shape[-1]
    return jnp.concatenate([obs, jnp.zeros(obs.shape[:-1] + (pad,))], axis=-1)


def _params_path(logdir: Path, pred_type: str, team: str, seed_idx: int) -> Path:
    alg = f"mappo_objectives_{pred_type}"
    return (
        logdir
        / f"{alg}_MPE_simple_tag_v3_{team}_actor_seed0_vmap{seed_idx}.safetensors"
    )


def rollout_one_checkpoint(
    pred_type: str,
    seed_idx: int,
    num_eps: int,
    rng_key,
    *,
    logdir: Path | str = DEFAULT_LOGDIR,
    num_steps: int = EP_LEN,
    num_adversaries: int = 1,
    prey_type: str | None = None,
) -> dict[str, np.ndarray]:
    """Roll out one MAPPO checkpoint greedily for ``num_eps`` first episodes."""
    from jaxmarl.wrappers.baselines import load_params

    logdir = Path(logdir)
    env = SimpleTagObjectivesMPE(
        pred_type=pred_type, num_adversaries=num_adversaries
    )
    preds = [a for a in env.agents if a.startswith("adversary")]
    prey_agents = [a for a in env.agents if a.startswith("agent")]
    teams = {"pred": preds, "prey": prey_agents}
    prey_name = prey_agents[0]
    pred_indices = [env.agents.index(name) for name in preds]
    prey_idx = env.agents.index(prey_name)
    action_dim = env.action_space(preds[0]).n

    obs_sizes = {a: env.observation_space(a).shape[0] for a in env.agents}
    max_obs = max(obs_sizes.values())
    nets = {t: ActorLogits(action_dim=action_dim, hidden_dim=HIDDEN) for t in teams}
    prey_checkpoint_type = pred_type if prey_type is None else prey_type
    params = {
        "pred": load_params(str(_params_path(logdir, pred_type, "pred", seed_idx))),
        "prey": load_params(
            str(_params_path(logdir, prey_checkpoint_type, "prey", seed_idx))
        ),
    }

    rng_key, k_reset = jax.random.split(rng_key)
    reset_keys = jax.random.split(k_reset, num_eps)
    obs, state = jax.vmap(env.reset)(reset_keys)

    prey_pos = [np.asarray(state.p_pos[:, prey_idx])]
    pred_pos = [np.asarray(state.p_pos[:, pred_indices])]
    pred_obs = [np.stack([np.asarray(obs[name]) for name in preds], axis=1)]
    prey_acts, pred_acts = [], []

    done = jnp.zeros((num_eps,), dtype=bool)
    pred_lava_steps = jnp.zeros((num_eps,), dtype=jnp.float32)
    prey_lava_steps = jnp.zeros((num_eps,), dtype=jnp.float32)

    for _ in range(num_steps):
        active = ~done
        all_actions = {}
        for team, agents in teams.items():
            for agent in agents:
                logits = nets[team].apply(params[team], _pad_obs(obs[agent], max_obs))
                all_actions[agent] = jnp.argmax(logits, axis=-1).astype(jnp.int32)

        rng_key, k_step = jax.random.split(rng_key)
        step_keys = jax.random.split(k_step, num_eps)
        new_obs, new_state, rewards, dones, info = jax.vmap(env.step_env)(
            step_keys, state, all_actions
        )

        pred_lava_steps += jnp.sum(
            info["pred_lava"][:, pred_indices], axis=-1
        ) * active.astype(jnp.float32)
        prey_lava_steps += info["prey_lava"][:, prey_idx] * active.astype(jnp.float32)

        state = freeze_tree(active, new_state, state)
        obs = freeze_tree(active, new_obs, obs)
        done = done | dones["__all__"]

        prey_pos.append(np.asarray(state.p_pos[:, prey_idx]))
        pred_pos.append(np.asarray(state.p_pos[:, pred_indices]))
        pred_obs.append(np.stack([np.asarray(obs[name]) for name in preds], axis=1))
        prey_acts.append(np.asarray(jnp.where(active, all_actions[prey_name], 0)))
        pred_acts.append(
            np.stack(
                [
                    np.asarray(jnp.where(active, all_actions[name], 0))
                    for name in preds
                ],
                axis=-1,
            )
        )

    capture_t = np.asarray(state.capture_t)
    captured = capture_t >= 0
    survival_time = np.where(captured, capture_t, num_steps).astype(np.float32)
    return dict(
        positions=np.stack(prey_pos, axis=1).astype(np.float32),
        pred_positions=np.stack(pred_pos, axis=1).astype(np.float32),
        pred_observations=np.stack(pred_obs, axis=1).astype(np.float32),
        lava_pos=np.asarray(state.lava_pos).astype(np.float32),
        lava_rad=np.asarray(state.lava_rad).astype(np.float32),
        actions=np.stack(prey_acts, axis=1).astype(np.int32),
        pred_actions=np.stack(pred_acts, axis=1).astype(np.int32),
        capture_t=capture_t.astype(np.int32),
        captured=captured.astype(bool),
        survival_time=survival_time,
        pred_lava_steps=np.asarray(pred_lava_steps).astype(np.float32),
        prey_lava_steps=np.asarray(prey_lava_steps).astype(np.float32),
        resources_collected=np.asarray(
            jnp.sum(state.collected.astype(jnp.float32), axis=-1)
        ).astype(np.float32),
        pred_coverage=np.asarray(
            jnp.sum(state.visited.astype(jnp.float32), axis=-1)
        ).astype(np.float32),
        env_seed=np.asarray(reset_keys, dtype=np.uint32),
        valid_length=np.where(captured, capture_t, num_steps).astype(np.int32),
    )


def objective_dataset(
    n_eps: int = 200,
    ckpt_seeds: Sequence[int] = (0, 1, 2),
    rng0: int = 0,
    num_steps: int | None = None,
    logdir: Path | str = DEFAULT_LOGDIR,
    num_adversaries: int = 1,
    prey_type: str | None = "capture",
) -> ObjectiveDataset:
    """Roll out objective-typed predators against one declared prey policy.

    ``prey_type=None`` reproduces matched co-training pairs and is useful only
    as a confounded control. The default fixes the prey checkpoint family to
    ``capture`` so objective labels cannot identify three different prey
    policies.
    """
    steps = int(num_steps if num_steps is not None else EP_LEN)
    rows: dict[str, list[Any]] = {
        k: []
        for k in (
            "prey_pos",
            "pred_pos",
            "pred_obs",
            "lava_pos",
            "lava_rad",
            "prey_act",
            "pred_act",
            "capture_t",
            "captured",
            "survival_time",
            "pred_lava_steps",
            "prey_lava_steps",
            "resources_collected",
            "pred_coverage",
            "label",
            "ckpt_seed",
            "env_seed",
            "valid_length",
        )
    }

    for lab, pred_type in enumerate(OBJECTIVE_TYPES):
        for seed in ckpt_seeds:
            d = rollout_one_checkpoint(
                pred_type,
                seed,
                n_eps,
                # Match reset/transition randomness across objective labels so
                # the label cannot be inferred from a different environment
                # seed distribution.
                jax.random.PRNGKey(rng0 + seed),
                logdir=logdir,
                num_steps=steps,
                num_adversaries=num_adversaries,
                prey_type=prey_type,
            )
            n = len(d["positions"])
            rows["prey_pos"].append(d["positions"])
            rows["pred_pos"].append(d["pred_positions"])
            rows["pred_obs"].append(d["pred_observations"])
            rows["lava_pos"].append(d["lava_pos"])
            rows["lava_rad"].append(d["lava_rad"])
            rows["prey_act"].append(d["actions"])
            rows["pred_act"].append(d["pred_actions"])
            rows["capture_t"].append(d["capture_t"])
            rows["captured"].append(d["captured"].astype(np.int32))
            rows["survival_time"].append(d["survival_time"])
            rows["pred_lava_steps"].append(d["pred_lava_steps"])
            rows["prey_lava_steps"].append(d["prey_lava_steps"])
            rows["resources_collected"].append(d["resources_collected"])
            rows["pred_coverage"].append(d["pred_coverage"])
            rows["label"].append(np.full(n, lab, np.int32))
            rows["ckpt_seed"].append(np.full(n, seed, np.int32))
            rows["env_seed"].append(d["env_seed"])
            rows["valid_length"].append(d["valid_length"])

    float_keys = {
        "prey_pos",
        "pred_pos",
        "pred_obs",
        "lava_pos",
        "lava_rad",
        "survival_time",
        "pred_lava_steps",
        "prey_lava_steps",
        "resources_collected",
        "pred_coverage",
    }
    stacked = {
        key: np.concatenate(vals).astype(
            np.float32
            if key in float_keys
            else np.uint32
            if key == "env_seed"
            else np.int32
        )
        for key, vals in rows.items()
    }
    return ObjectiveObservationDataset(**stacked)
