"""Simulator-backed MPPI planner for the blue prey (Gate 3).

Validates planning before any physics is learned. The imagined dynamics are the
exact JaxMARL environment (``step_env`` vmapped over candidates), so world-model
error is zero and the only sources of error are the planner and the opponent
model. MPPI optimizes a blue action sequence only:

    for every candidate and imagined step:
        v = pi_red(imagined red observation, c)     # red is never optimized
        state <- step_env(state, (u, v))
        add blue reward until capture
    terminal value is zero (planner-only smoke); execute the first action

The opponent context ``c`` is held fixed inside the short imagined horizon and
recomputed from the real history before every real step.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from mopa.bc_continuous import ContinuousBCPolicy
from mopa.continuous_data import deterministic_specialist_action
from tag_objectives import CONTINUOUS_ACTION_DIM, joint_action_dict

RedPolicy = Callable[[jax.Array, jax.Array], jax.Array]

__all__ = [
    "MPPIConfig",
    "RedPolicy",
    "imagined_returns",
    "make_bc_red_policy",
    "make_true_red_policy",
    "mppi_plan",
]


@dataclass(frozen=True)
class MPPIConfig:
    """Planner hyper-parameters (all static under ``jit``)."""

    horizon: int = 10
    population_size: int = 128
    num_elites: int = 16
    mppi_iterations: int = 3
    min_plan_std: float = 0.05
    max_plan_std: float = 1.0
    temperature: float = 1.0
    discount: float = 0.99

    def __post_init__(self) -> None:
        if not 0 < self.num_elites <= self.population_size:
            raise ValueError("num_elites must be in (0, population_size]")
        if self.horizon < 1 or self.mppi_iterations < 1:
            raise ValueError("horizon and mppi_iterations must be positive")
        if not 0.0 < self.min_plan_std <= self.max_plan_std:
            raise ValueError("plan std bounds must satisfy 0 < min <= max")


def make_true_red_policy(params: Any, obs_width: int) -> RedPolicy:
    """The frozen specialist's deterministic mean; ignores the context input."""

    def policy(red_obs: jax.Array, context: jax.Array) -> jax.Array:
        del context
        return deterministic_specialist_action(params, red_obs, obs_width)

    return policy


def make_bc_red_policy(policy: ContinuousBCPolicy) -> RedPolicy:
    """Continuous BC on ``[red_observation, context]``."""

    def red(red_obs: jax.Array, context: jax.Array) -> jax.Array:
        return policy.act(jnp.concatenate([red_obs, context], axis=-1))

    return red


def imagined_returns(
    env: Any,
    state: Any,
    actions: jax.Array,
    context: jax.Array,
    red_policy: RedPolicy,
    keys: jax.Array,
    *,
    discount: float,
) -> jax.Array:
    """Discounted blue return of ``actions`` ``(K, H, 2)`` from batched ``state``.

    ``state`` is a pytree batched over ``K`` candidates, ``context`` is ``(K, C)``,
    ``keys`` is ``(H, K, 2)``. Accumulation stops at the first capture/timeout.
    """
    prey_name = env.good_agents[0]
    pred_name = env.adversaries[0]
    horizon = actions.shape[-2]
    k = actions.shape[0]
    step = jax.vmap(env.step_env)
    obs_fn = jax.vmap(env.get_obs)

    def body(carry, inputs):
        state, alive, disc, total = carry
        blue, key = inputs
        obs = obs_fn(state)
        red = red_policy(obs[pred_name], context)
        _, new_state, rew, dones, _ = step(key, state, joint_action_dict(env, blue, red[:, None, :]))
        total = total + disc * rew[prey_name] * alive
        alive = alive & ~dones["__all__"]
        disc = disc * discount
        return (new_state, alive, disc, total), None

    init = (state, jnp.ones((k,), dtype=bool), jnp.ones(()), jnp.zeros((k,)))
    (_, _, _, total), _ = jax.lax.scan(
        body, init, (jnp.swapaxes(actions, 0, 1), keys), length=horizon
    )
    return total


def _replicate(tree: Any, n: int) -> Any:
    return jax.tree_util.tree_map(lambda x: jnp.repeat(x[None], n, axis=0), tree)


def _plan_single(
    env: Any,
    red_policy: RedPolicy,
    cfg: MPPIConfig,
    state: Any,
    context: jax.Array,
    prev_mean: jax.Array,
    key: jax.Array,
    deterministic: bool,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    """MPPI for one (unbatched) environment state; mirrors ``mopa.mppi.plan``."""
    k, h, a = cfg.population_size, cfg.horizon, CONTINUOUS_ACTION_DIM
    states = _replicate(state, k)
    contexts = jnp.repeat(context[None], k, axis=0)
    key, noise_key, *value_keys = jax.random.split(key, 2 + cfg.mppi_iterations)
    noise = jax.random.normal(noise_key, (k, cfg.mppi_iterations, h, a))
    mean = jnp.zeros((h, a)).at[:-1].set(prev_mean[1:])
    std = jnp.full((h, a), cfg.max_plan_std)
    elite_values = jnp.zeros((cfg.num_elites,))
    elite_actions = jnp.zeros((cfg.num_elites, h, a))
    score = jnp.full((cfg.num_elites,), 1.0 / cfg.num_elites)
    for i in range(cfg.mppi_iterations):
        actions = jnp.clip(mean[None] + std[None] * noise[:, i], -1.0, 1.0)
        step_keys = jax.random.split(value_keys[i], h * k).reshape(h, k, 2)
        values = imagined_returns(
            env, states, actions, contexts, red_policy, step_keys, discount=cfg.discount
        )
        elite_values, elite_inds = jax.lax.top_k(values, cfg.num_elites)
        elite_actions = actions[elite_inds]
        score = jax.nn.softmax(cfg.temperature * elite_values)
        mean = jnp.sum(score[:, None, None] * elite_actions, axis=0)
        std = jnp.sqrt(
            jnp.sum(score[:, None, None] * (elite_actions - mean[None]) ** 2, axis=0) + 1e-6
        ).clip(cfg.min_plan_std, cfg.max_plan_std)
    if deterministic:
        idx = jnp.argmax(elite_values)
    else:
        key, pick = jax.random.split(key)
        idx = jax.random.categorical(pick, jnp.log(score))
    action = elite_actions[idx, 0]
    return jnp.clip(action, -1.0, 1.0), (mean, std)


def mppi_plan(
    env: Any,
    red_policy: RedPolicy,
    cfg: MPPIConfig,
    state: Any,
    context: jax.Array,
    prev_mean: jax.Array,
    key: jax.Array,
    *,
    deterministic: bool = True,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    """Batched MPPI: ``state`` batched over ``B`` episodes, ``context`` ``(B, C)``.

    ``prev_mean`` ``(B, H, 2)`` warm-starts the mean (shifted by one step);
    pass zeros at episode start. Returns the first blue action ``(B, 2)`` and
    the final ``(mean, std)`` each ``(B, H, 2)``.
    """
    batch = context.shape[0]
    keys = jax.random.split(key, batch)
    fn = partial(_plan_single, env, red_policy, cfg, deterministic=deterministic)
    return jax.vmap(fn)(state, context, prev_mean, keys)


def jit_planner(
    env: Any, red_policy: RedPolicy, cfg: MPPIConfig, *, deterministic: bool = True
) -> Callable:
    """Compile ``mppi_plan`` for one environment / opponent model / config."""
    return jax.jit(
        lambda state, context, prev_mean, key: mppi_plan(
            env, red_policy, cfg, state, context, prev_mean, key, deterministic=deterministic
        )
    )


def zero_plan(batch: int, cfg: MPPIConfig) -> np.ndarray:
    return np.zeros((batch, cfg.horizon, CONTINUOUS_ACTION_DIM), dtype=np.float32)
