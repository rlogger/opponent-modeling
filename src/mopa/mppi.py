"""MPPI planner for TD-MPC2 (Gate 0 source port + Gate 4 opponent generation).

Ported from ``TDMPC2.plan`` and ``TDMPC2.estimate_value`` in
``tdmpc2_jax/tdmpc2.py`` of ShaneFlandermeyer/tdmpc2-jax (MIT), pinned commit
``5b05ff452424896d709848e1f249bd67e269b8a1``. The methods are expressed as
module-level functions taking the :class:`mopa.tdmpc.TDMPC2` PyTree as their
first argument; ``TDMPC2.plan`` / ``TDMPC2.estimate_value`` delegate here.

MPPI optimizes the blue action sequence only. Local additions:

- an opponent context ``c`` (held fixed inside the imagined horizon) that is
  concatenated to policy-prior and Q inputs in ``conditioned``/``factored``
  modes and to dynamics/reward/continue inputs in ``conditioned`` mode;
- in ``factored`` mode a red action ``v = red(x_t, c)`` is generated before
  every imagined transition with an explicitly split PRNG key (the first
  experiment uses the deterministic red mean, so the key is reserved but
  unused); the planner never optimizes or selects ``v``;
- the continuation head is queried on the same transition ``(x_t, a_t)`` that
  it was trained on and gates the discount after that transition.

With ``opponent_mode="implicit"``, ``context_dim=0`` and
``predict_continues=False`` the computation and PRNG consumption are the
unchanged upstream algorithm.
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


def _repeat_context(context: jax.Array, n: int) -> jax.Array:
    return context[..., None, :].repeat(n, axis=-2)


@partial(jax.jit, static_argnames=("horizon", "deterministic", "train"))
def plan(
    agent: "TDMPC2",
    x: jax.Array,
    horizon: int,
    context: jax.Array,
    prev_plan: Optional[Tuple[jax.Array, jax.Array]] = None,
    deterministic: bool = False,
    train: bool = False,
    *,
    key: PRNGKey,
) -> Tuple[jax.Array, Tuple[jax.Array, jax.Array]]:
    model = agent.model
    batch_shape = x.shape[:-1]
    actions = jnp.zeros(
        (*batch_shape, agent.population_size, horizon, model.action_dim)
    )

    ###########################################################
    # Policy prior samples
    ###########################################################
    if agent.policy_prior_samples > 0:
        key, *prior_noise_keys = jax.random.split(key, 1 + horizon)
        policy_actions = jnp.zeros(
            (*batch_shape, agent.policy_prior_samples, horizon, model.action_dim)
        )
        x_t = x[..., None, :].repeat(agent.policy_prior_samples, axis=-2)
        c_t = _repeat_context(context, agent.policy_prior_samples)
        for t in range(horizon):
            policy_actions = policy_actions.at[..., t, :].set(
                model.sample_actions(
                    x=model.policy_inputs(x_t, c_t),
                    deterministic=False,
                    params=model.policy_model.params,
                    key=prior_noise_keys[t],
                )[0]
            )
            if t < horizon - 1:  # Don't need for the last time step
                v_t = (
                    model.red_action(x_t, c_t, model.red_model.params)
                    if model.opponent_mode == "factored"
                    else None
                )
                x_t = model.next(
                    x=x_t,
                    a=model.transition_inputs(policy_actions[..., t, :], c_t, v_t),
                    params=model.dynamics_model.params,
                )

        actions = actions.at[..., : agent.policy_prior_samples, :, :].set(
            policy_actions
        )

    ###########################################################
    # MPPI planning
    ###########################################################
    x_t = x[..., None, :].repeat(agent.population_size, axis=-2)
    c_pop = _repeat_context(context, agent.population_size)
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
            model.action_dim,
        ),
    )
    # Initialize population state
    mean = jnp.zeros((*batch_shape, horizon, model.action_dim))
    std = jnp.full((*batch_shape, horizon, model.action_dim), agent.max_plan_std)
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
            agent, x=x_t, actions=actions, context=c_pop, horizon=horizon, key=value_keys[i]
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
            final_noise_key, shape=batch_shape + (model.action_dim,)
        )
    else:
        final_action = action[..., 0, :]

    return final_action.clip(-1, 1), (mean, std)


@partial(jax.jit, static_argnames=("horizon",))
def estimate_value(
    agent: "TDMPC2",
    x: jax.Array,
    actions: jax.Array,
    context: jax.Array,
    horizon: int,
    key: PRNGKey,
) -> jax.Array:
    """Discounted imagined return of blue ``actions`` ``(..., horizon, action_dim)``.

    ``context`` ``(..., context_dim)`` is held fixed. Red actions (factored
    mode) are generated per step from the current imagined latent; one PRNG key
    is split off per step for them even though the deterministic mean is used.
    """
    model = agent.model
    G, discount = 0.0, 1.0
    if model.opponent_mode == "factored":
        # Explicit per-step red keys (reserved; the first experiment uses the
        # deterministic red mean). Splitting only here keeps the implicit
        # path's PRNG consumption identical to upstream.
        key, red_key = jax.random.split(key)
        red_keys = jax.random.split(red_key, horizon)
    for t in range(horizon):
        u_t = actions[..., t, :]
        if model.opponent_mode == "factored":
            _ = red_keys[t]
            v_t = model.red_action(x, context, model.red_model.params)
        else:
            v_t = None
        a_t = model.transition_inputs(u_t, context, v_t)
        reward, _ = model.reward(x=x, a=a_t, params=model.reward_model.params)
        if model.predict_continues:
            continues = (
                jax.nn.sigmoid(
                    model.continue_logits(x, a_t, model.continue_model.params)
                )
                > 0.5
            )
        x = model.next(x=x, a=a_t, params=model.dynamics_model.params)
        G += discount * reward
        discount *= agent.discount
        if model.predict_continues:
            discount *= continues

    action_key, Q_key = jax.random.split(key, 2)
    next_action = model.sample_actions(
        x=model.policy_inputs(x, context),
        deterministic=False,
        params=model.policy_model.params,
        key=action_key,
    )[0]

    Qs, _ = model.Q(
        x=x,
        a=model.value_inputs(next_action, context),
        params=model.value_model.params,
        key=Q_key,
    )
    Q = Qs.mean(axis=0)
    return G + discount * Q
