"""MPPI planner for TD-MPC2 (Gate 0 source port).

Ported from ``TDMPC2.plan`` and ``TDMPC2.estimate_value`` in
``tdmpc2_jax/tdmpc2.py`` of ShaneFlandermeyer/tdmpc2-jax (MIT), pinned commit
``5b05ff452424896d709848e1f249bd67e269b8a1``. The methods are expressed as
module-level functions taking the :class:`mopa.tdmpc.TDMPC2` PyTree as their
first argument; ``TDMPC2.plan`` / ``TDMPC2.estimate_value`` delegate here.

The algorithm, PRNG splitting order, and numerics are unchanged. The planner
optimizes the single agent's own action sequence only (Equation 1 rollouts,
``x_next = d(x, u)``). No opponent generation is present at this gate.
"""
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Optional, Tuple

import jax
import jax.numpy as jnp

if TYPE_CHECKING:  # pragma: no cover
    from mopa.tdmpc import TDMPC2

PRNGKey = jax.Array

__all__ = ["estimate_value", "plan"]


@partial(jax.jit, static_argnames=("horizon", "deterministic", "train"))
def plan(
    agent: "TDMPC2",
    x: jax.Array,
    horizon: int,
    prev_plan: Optional[Tuple[jax.Array, jax.Array]] = None,
    deterministic: bool = False,
    train: bool = False,
    *,
    key: PRNGKey,
) -> Tuple[jax.Array, Tuple[jax.Array, jax.Array]]:
    batch_shape = x.shape[:-1]
    actions = jnp.zeros(
        (*batch_shape, agent.population_size, horizon, agent.model.action_dim)
    )

    ###########################################################
    # Policy prior samples
    ###########################################################
    if agent.policy_prior_samples > 0:
        key, *prior_noise_keys = jax.random.split(key, 1 + horizon)
        policy_actions = jnp.zeros(
            (
                *batch_shape,
                agent.policy_prior_samples,
                horizon,
                agent.model.action_dim,
            )
        )
        x_t = x[..., None, :].repeat(agent.policy_prior_samples, axis=-2)
        for t in range(horizon):
            policy_actions = policy_actions.at[..., t, :].set(
                agent.model.sample_actions(
                    x=x_t,
                    deterministic=False,
                    params=agent.model.policy_model.params,
                    key=prior_noise_keys[t],
                )[0]
            )
            if t < horizon - 1:  # Don't need for the last time step
                x_t = agent.model.next(
                    x=x_t,
                    a=policy_actions[..., t, :],
                    params=agent.model.dynamics_model.params,
                )

        actions = actions.at[..., : agent.policy_prior_samples, :, :].set(
            policy_actions
        )

    ###########################################################
    # MPPI planning
    ###########################################################
    x_t = x[..., None, :].repeat(agent.population_size, axis=-2)
    key, mppi_noise_key, *value_keys = jax.random.split(
        key, 2 + agent.mppi_iterations
    )
    noise = jax.random.normal(
        mppi_noise_key,
        shape=(
            *batch_shape,
            agent.population_size - agent.policy_prior_samples,
            agent.mppi_iterations,
            horizon,
            agent.model.action_dim,
        ),
    )
    # Initialize population state
    mean = jnp.zeros((*batch_shape, horizon, agent.model.action_dim))
    std = jnp.full((*batch_shape, horizon, agent.model.action_dim), agent.max_plan_std)
    if prev_plan is not None:
        mean = mean.at[..., :-1, :].set(prev_plan[0][..., 1:, :])

    for i in range(agent.mppi_iterations):
        actions = (
            actions.at[..., agent.policy_prior_samples :, :, :]
            .set(mean[..., None, :, :] + std[..., None, :, :] * noise[..., i, :, :])
            .clip(-1, 1)
        )

        # Compute elites
        values = estimate_value(
            agent, x=x_t, actions=actions, horizon=horizon, key=value_keys[i]
        )
        elite_values, elite_inds = jax.lax.top_k(values, agent.num_elites)
        elite_actions = jnp.take_along_axis(
            actions, elite_inds[..., None, None], axis=-3
        )

        # Update population distribution
        score = jax.nn.softmax(agent.temperature * elite_values)
        mean = jnp.sum(score[..., None, None] * elite_actions, axis=-3)
        std = jnp.sqrt(
            jnp.sum(
                score[..., None, None] * (elite_actions - mean[..., None, :, :]) ** 2,
                axis=-3,
            )
            + 1e-6
        ).clip(agent.min_plan_std, agent.max_plan_std)

    # Sample final action
    if deterministic:  # Use best trajectory
        action_ind = jnp.argmax(elite_values, axis=-1)
    else:  # Sample from elites
        key, final_action_key = jax.random.split(key)
        action_ind = jax.random.categorical(
            final_action_key, logits=jnp.log(score), shape=batch_shape
        )
    action = jnp.take_along_axis(
        elite_actions, action_ind[..., None, None, None], axis=-3
    ).squeeze(-3)
    if train:
        key, final_noise_key = jax.random.split(key)
        final_action = action[..., 0, :] + std[..., 0, :] * jax.random.normal(
            final_noise_key, shape=batch_shape + (agent.model.action_dim,)
        )
    else:
        final_action = action[..., 0, :]

    return final_action.clip(-1, 1), (mean, std)


@partial(jax.jit, static_argnames=("horizon",))
def estimate_value(
    agent: "TDMPC2",
    x: jax.Array,
    actions: jax.Array,
    horizon: int,
    key: PRNGKey,
) -> jax.Array:
    G, discount = 0.0, 1.0
    for t in range(horizon):
        reward, _ = agent.model.reward(
            x=x, a=actions[..., t, :], params=agent.model.reward_model.params
        )
        x = agent.model.next(
            x=x, a=actions[..., t, :], params=agent.model.dynamics_model.params
        )
        G += discount * reward
        discount *= agent.discount

        if agent.model.predict_continues:
            continues = (
                jax.nn.sigmoid(
                    agent.model.continue_model.apply_fn(
                        {"params": agent.model.continue_model.params}, x
                    )
                ).squeeze(-1)
                > 0.5
            )
            discount *= continues

    action_key, Q_key = jax.random.split(key, 2)
    next_action = agent.model.sample_actions(
        x=x,
        deterministic=False,
        params=agent.model.policy_model.params,
        key=action_key,
    )[0]

    Qs, _ = agent.model.Q(
        x=x, a=next_action, params=agent.model.value_model.params, key=Q_key
    )
    Q = Qs.mean(axis=0)
    return G + discount * Q
