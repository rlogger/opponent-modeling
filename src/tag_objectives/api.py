"""Research facade for the objective-typed predator environment.

    from tag_objectives import make_env, evaluate_policy, random_policy

    env = make_env("curious", num_adversaries=3)
    metrics = evaluate_policy(
        env, random_policy(env), n_eps=128, key=jax.random.PRNGKey(0)
    )
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from tag_objectives.actions import (
    CONTINUOUS_ACTION_DIM,
    clip_action,
    from_mpe_action,
    joint_action_dict,
    to_mpe_action,
)
from tag_objectives.objectives import (
    OBJECTIVES,
    ObjectiveSpec,
    SimpleTagObjectivesMPE,
    register_objective,
)
from tag_objectives.teams import freeze_tree
from tag_objectives.types import EpisodeMetrics

PolicyFn = Callable[[Mapping[str, Any], Any], Mapping[str, Any]]

__all__ = [
    "CONTINUOUS_ACTION_DIM",
    "OBJECTIVES",
    "EpisodeMetrics",
    "ObjectiveSpec",
    "PolicyFn",
    "SimpleTagObjectivesMPE",
    "clip_action",
    "evaluate_policy",
    "from_mpe_action",
    "joint_action_dict",
    "list_objectives",
    "make_env",
    "random_policy",
    "register_objective",
    "to_mpe_action",
]


def list_objectives() -> list[str]:
    """Names accepted by :func:`make_env`."""
    return sorted(OBJECTIVES)


def make_env(
    objective: str = "capture", *, continuous: bool = False, **overrides: Any
) -> SimpleTagObjectivesMPE:
    """Build the environment for any registered objective.

    ``continuous=True`` selects JaxMARL's continuous MPE action space. Callers
    then pass two-dimensional ``[-1, 1]^2`` actions through
    :func:`to_mpe_action` / :func:`joint_action_dict` at the env boundary.
    """
    if continuous:
        overrides.setdefault("action_type", "Continuous")
    return SimpleTagObjectivesMPE(pred_type=objective, **overrides)


def random_policy(env: SimpleTagObjectivesMPE) -> PolicyFn:
    """Uniform-random joint policy in :func:`evaluate_policy`'s signature.

    Continuous environments receive uniform ``[-1, 1]^2`` actions converted at
    the boundary; discrete environments receive uniform action indices.
    """
    if env.continuous_actions:

        def policy(obs: Mapping[str, Any], rng: Any) -> dict[str, Any]:
            acts: dict[str, Any] = {}
            for a in env.agents:
                rng, k = jax.random.split(rng)
                n = obs[a].shape[0]
                acts[a] = to_mpe_action(
                    jax.random.uniform(
                        k, (n, CONTINUOUS_ACTION_DIM), minval=-1.0, maxval=1.0
                    )
                )
            return acts

        return policy

    dims = {a: int(env.action_space(a).n) for a in env.agents}

    def policy(obs: Mapping[str, Any], rng: Any) -> dict[str, Any]:
        acts: dict[str, Any] = {}
        for a in env.agents:
            rng, k = jax.random.split(rng)
            n = obs[a].shape[0]
            acts[a] = jax.random.randint(k, (n,), 0, dims[a])
        return acts

    return policy


def evaluate_policy(
    env: SimpleTagObjectivesMPE,
    policy: PolicyFn,
    n_eps: int = 128,
    key: Any | None = None,
    num_steps: int | None = None,
) -> EpisodeMetrics:
    """Roll ``policy`` for ``n_eps`` episodes; return the standard metric panel.

    ``policy`` maps ``(obs_dict, rng) -> action_dict`` with batch size ``n_eps``.
    Episodes freeze at first capture so metrics describe the first episode only.
    """
    key = jax.random.PRNGKey(0) if key is None else key
    T = int(num_steps or env.max_steps)
    P = env.num_adversaries
    prey_idx = P

    key, kr = jax.random.split(key)
    obs, state = jax.vmap(env.reset)(jax.random.split(kr, n_eps))

    done = jnp.zeros((n_eps,), dtype=bool)
    pred_lava = jnp.zeros((n_eps,))
    prey_lava = jnp.zeros((n_eps,))

    for _ in range(T):
        active = ~done
        key, kp, ks = jax.random.split(key, 3)
        raw = policy(obs, kp)
        if env.continuous_actions:
            acts = {a: jnp.asarray(v, dtype=jnp.float32) for a, v in raw.items()}
        else:
            acts = {a: v.astype(jnp.int32) for a, v in raw.items()}
        new_obs, new_state, _, dones, info = jax.vmap(env.step_env)(
            jax.random.split(ks, n_eps), state, acts
        )
        af = active.astype(jnp.float32)
        pred_lava += jnp.sum(info["pred_lava"][:, :P], axis=-1) * af
        prey_lava += info["prey_lava"][:, prey_idx] * af
        state = freeze_tree(active, new_state, state)
        obs = freeze_tree(active, new_obs, obs)
        done = done | dones["__all__"]

    capture_t = np.asarray(state.capture_t)
    captured = capture_t >= 0
    res = np.asarray(state.collected)
    res_lava_d = np.linalg.norm(
        np.asarray(state.resource_pos)[:, :, None, :]
        - np.asarray(state.lava_pos)[:, None, :, :],
        axis=-1,
    )
    near = np.any(res_lava_d < 1.25 * np.asarray(state.lava_rad)[:, None, :], axis=-1)
    return EpisodeMetrics(
        capture_rate=captured.astype(np.float32),
        survival_time=np.where(captured, capture_t, T).astype(np.float32),
        resources_collected=res.sum(-1).astype(np.float32),
        near_lava_collected=(res & near).sum(-1).astype(np.float32),
        pred_lava_steps=np.asarray(pred_lava, dtype=np.float32),
        prey_lava_steps=np.asarray(prey_lava, dtype=np.float32),
        pred_coverage=np.asarray(state.visited).sum(-1).astype(np.float32),
    )
