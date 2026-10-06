"""Planner information boundaries, numerical stability and analytic contracts."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")
pytest.importorskip("optax")
pytest.importorskip("distrax")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402

from mopa import mppi  # noqa: E402
from mopa.tdmpc import (  # noqa: E402
    build_identity_encoder,
    create_agent,
    load_config,
    mish,
)


def _agent(mode="factored", context_dim=3):
    cfg = load_config(profile="smoke")
    cfg.update(opponent_mode=mode, context_dim=context_dim)
    cfg["encoder"].update(encoder_dim=8)
    cfg["world_model"].update(latent_dim=8, hidden_dim=8, num_value_nets=2, num_bins=11)
    cfg["factored"].update(red_hidden_dim=8)
    cfg["tdmpc2"].update(horizon=2, batch_size=2, population_size=4, num_elites=2, policy_prior_samples=1)
    return create_agent(cfg, 4, key=jax.random.PRNGKey(0))


def _batch(context_dim=3):
    rng = np.random.default_rng(8)
    return dict(
        observations=jnp.asarray(rng.normal(size=(2, 2, 4)), jnp.float32),
        next_observations=jnp.asarray(rng.normal(size=(2, 2, 4)), jnp.float32),
        actions=jnp.zeros((2, 2, 2)), red_actions=jnp.ones((2, 2, 2)) * 0.2,
        rewards=jnp.ones((2, 2)), terminated=jnp.zeros((2, 2), bool),
        truncated=jnp.array([[False, False], [True, True]]),
        context=jnp.zeros((2, 2, context_dim)), next_context=jnp.ones((2, 2, context_dim)),
    )


@pytest.mark.parametrize("mode,field", [("factored", "red_actions"), ("factored", "context"),
                                       ("factored", "next_context"), ("conditioned", "context"),
                                       ("conditioned", "next_context")])
def test_required_opponent_information_is_not_imputed(mode, field):
    agent = _agent(mode)
    batch = _batch()
    del batch[field]
    with pytest.raises(ValueError, match=field + " is required"):
        agent.update(**batch, key=jax.random.PRNGKey(1))


@pytest.mark.parametrize("field", ["actions", "rewards", "terminated", "truncated", "observations",
                                   "next_observations", "red_actions", "context", "next_context"])
def test_update_rejects_misaligned_sequence_shapes(field):
    batch = _batch()
    batch[field] = batch[field][:1]
    with pytest.raises(ValueError, match="shape|align"):
        _agent().update(**batch, key=jax.random.PRNGKey(1))


@pytest.mark.parametrize("mode", ["implicit", "factored"])
def test_history_free_updates_do_not_require_context(mode):
    agent = _agent(mode, context_dim=0)
    batch = _batch(0)
    del batch["context"], batch["next_context"]
    if mode == "implicit":
        del batch["red_actions"]
    trained, info = agent.update(**batch, key=jax.random.PRNGKey(1))
    assert bool(info["world_gradients_finite"]) and bool(info["policy_gradients_finite"])
    assert int(trained.model.value_model.step) == 1


def test_mish_extreme_forward_and_backward_are_finite_and_normal_scale_preserved():
    extreme = jnp.array([-1e4, -100., -20., 0., 20., 100., 1e4])
    assert np.isfinite(np.asarray(mish(extreme))).all()
    derivative = jax.vmap(jax.grad(mish))(extreme)
    assert np.isfinite(np.asarray(derivative)).all()
    assert float(derivative[-1]) == pytest.approx(1.0)
    normal = jnp.linspace(-8., 8., 100)
    old = normal * jnp.tanh(jnp.log(1 + jnp.exp(normal)))
    np.testing.assert_allclose(mish(normal), old, rtol=2e-5, atol=1e-6)


@pytest.mark.parametrize("field", ["mean", "std"])
@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_identity_encoder_rejects_nonfinite_scaling(field, bad):
    mean, std = np.zeros(4, np.float32), np.ones(4, np.float32)
    (mean if field == "mean" else std)[0] = bad
    with pytest.raises(ValueError, match="finite"):
        build_identity_encoder(mean, std, key=jax.random.PRNGKey(0))


def test_frozen_opponent_parameters_and_target_ema_ownership():
    agent = _agent()
    red = agent.model.red_model
    freeze = optax.set_to_zero()
    agent = agent.replace(model=agent.model.replace(
        red_model=red.replace(tx=freeze, opt_state=freeze.init(red.params))
    ), red_loss_scale=0.0)
    trained, info = agent.update(**_batch(), key=jax.random.PRNGKey(9))
    for old, new in zip(jax.tree.leaves(red.params), jax.tree.leaves(trained.model.red_model.params)):
        np.testing.assert_array_equal(old, new)
    for old, target, online in zip(jax.tree.leaves(agent.model.target_value_model.params),
                                   jax.tree.leaves(trained.model.target_value_model.params),
                                   jax.tree.leaves(trained.model.value_model.params)):
        np.testing.assert_allclose(target, (1 - agent.tau) * old + agent.tau * online, atol=1e-7)
    assert bool(info["world_gradients_finite"]) and bool(info["policy_gradients_finite"])


@pytest.mark.parametrize("horizon,stop_at,expected", [(1, None, 19.0), (3, None, 64.36), (3, 2., 2.8)])
def test_imagination_uses_each_predicted_state_and_counts_terminal_q_once(horizon, stop_at, expected):
    seen_red_states = []

    def red_action(x, context, params):
        seen_red_states.append(float(x[0, 0]))
        return jnp.concatenate([x, jnp.zeros_like(x)], axis=-1)

    model = SimpleNamespace(
        opponent_mode="factored", predict_continues=stop_at is not None,
        red_model=SimpleNamespace(params=None), reward_model=SimpleNamespace(params=None),
        dynamics_model=SimpleNamespace(params=None), continue_model=SimpleNamespace(params=None),
        policy_model=SimpleNamespace(params=None), value_model=SimpleNamespace(params=None),
        red_action=red_action,
        transition_inputs=lambda u, c, v: jnp.concatenate([u, v], axis=-1),
        reward=lambda x, a, params: (x[..., 0], None),
        next=lambda x, a, params: x + a[..., 2:3],
        continue_logits=lambda x, a, params: jnp.where(x[..., 0] < stop_at, 10., -10.),
        policy_inputs=lambda x, c: x,
        value_inputs=lambda u, c: u,
        sample_actions=lambda x, deterministic, params, key: (jnp.zeros((*x.shape[:-1], 2)),),
        Q=lambda x, a, params, key: (jnp.stack([10 * x[..., 0]] * 2), None),
    )
    # Execute the actual planner body with analytic heads, without tracing the
    # Python diagnostic spy. This isolates recurrence and discount arithmetic.
    value = mppi.estimate_value.__wrapped__(
        SimpleNamespace(model=model, discount=0.9), jnp.ones((1, 1)),
        jnp.zeros((1, horizon, 2)), jnp.zeros((1, 0)), horizon, jax.random.PRNGKey(1)
    )
    np.testing.assert_allclose(value, [expected], atol=1e-5)
    assert seen_red_states == [float(2**t) for t in range(horizon)]
