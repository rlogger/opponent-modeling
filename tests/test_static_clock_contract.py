"""P11 optional static/clock contract; unchanged game and legacy defaults."""
import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import serialization, struct

from mopa import mppi
from mopa.tdmpc import check_update, create_agent, load_config

CONTRACT = "objective_static_clock_v1"
STATIC = np.r_[8:40, 56:65]
DYNAMIC = np.r_[0:8, 40:56]


def config(contract=CONTRACT):
    cfg = load_config(profile="smoke")
    cfg.update(opponent_mode="implicit", context_dim=0)
    cfg["encoder"]["type"] = "identity"
    cfg["world_model"].update(hidden_dim=16, transition_contract=contract)
    return cfg


@pytest.fixture(scope="module")
def agent():
    mean = np.linspace(-.4, .7, 66, dtype=np.float32)
    std = np.linspace(.2, 2., 66, dtype=np.float32)
    model = create_agent(config(), 66, key=jax.random.PRNGKey(0), obs_mean=mean, obs_std=std)
    params = copy.deepcopy(model.model.dynamics_model.params)
    params["layers_2"]["bias"] = jnp.linspace(.01, .66, 66)
    return model.replace(model=model.model.replace(
        dynamics_model=model.model.dynamics_model.replace(params=params)))


def encoded_states(model, steps):
    raw = np.zeros((len(steps), 66), np.float32)
    raw[:, 2] = 2.4  # Legal outside-arena position; no new wall.
    raw[:, 8:40] = np.linspace(-2, 2, 32)
    raw[:, 56:65] = np.linspace(.1, .5, 9)
    raw[:, 65] = np.asarray(steps) / 100
    return jnp.asarray((raw - np.asarray(model.state_mean)) / np.asarray(model.state_std))


def test_static_copy_dynamic_preservation_clock_and_jit_vmap(agent):
    model = agent.model
    x = encoded_states(model, np.arange(101))
    a = jnp.zeros((101, 2))
    raw_prediction = x + model.dynamics_model.apply_fn(
        {"params": model.dynamics_model.params}, jnp.concatenate([x, a], -1))
    actual = model.next(x, a, model.dynamics_model.params)
    np.testing.assert_array_equal(actual[:, STATIC], x[:, STATIC])
    np.testing.assert_array_equal(actual[:-1, DYNAMIC], raw_prediction[:-1, DYNAMIC])
    np.testing.assert_array_equal(actual[-1], x[-1])
    physical = np.asarray(actual) * np.asarray(model.state_std) + np.asarray(model.state_mean)
    np.testing.assert_allclose(physical[:, 65], np.minimum(np.arange(101) + 1, 100) / 100, atol=8e-8)
    assert (physical[:-1, 2] > 2.4).all()
    np.testing.assert_array_equal(model.known_continuation(actual), np.arange(101) < 99)
    by_row = jax.jit(jax.vmap(lambda xx, aa: model.next(xx, aa, model.dynamics_model.params)))(x, a)
    np.testing.assert_array_equal(actual, by_row)


def test_gradient_only_removes_static_clock_residual_outputs(agent):
    model = agent.model
    x = encoded_states(model, [20])[0]
    def evaluate(params):
        return model.next(x, jnp.array([.2, -.3]), params).sum()
    gradients = jax.grad(evaluate)(model.dynamics_model.params)
    assert all(np.isfinite(g).all() for g in jax.tree.leaves(gradients))
    final_bias = np.asarray(gradients["layers_2"]["bias"])
    np.testing.assert_array_equal(final_bias[np.r_[STATIC, 65]], 0.)
    np.testing.assert_array_equal(final_bias[DYNAMIC], 1.)


@struct.dataclass
class DummyState:
    params: object = None


@struct.dataclass
class AnalyticModel:
    """Independent closed-form reward/value fixture around real MPPI scoring."""
    transition_contract: str = struct.field(pytree_node=False, default=CONTRACT)
    opponent_mode: str = struct.field(pytree_node=False, default="implicit")
    predict_continues: bool = struct.field(pytree_node=False, default=True)
    stop: bool = struct.field(pytree_node=False, default=False)
    poison_after_timeout: bool = struct.field(pytree_node=False, default=False)
    reward_model: DummyState = DummyState()
    continue_model: DummyState = DummyState()
    dynamics_model: DummyState = DummyState()
    policy_model: DummyState = DummyState()
    value_model: DummyState = DummyState()

    def known_continuation(self, x):
        return x[..., 0] < 100

    def next(self, x, a, params):
        return jnp.minimum(x + 1, 100)

    def reward(self, x, a, params):
        reward = jnp.full(x.shape[:-1], 2.)
        if self.poison_after_timeout:
            reward = jnp.where(x[..., 0] >= 100, jnp.nan, reward)
        return reward, None

    def continue_logits(self, x, a, params):
        return jnp.full(x.shape[:-1], -2. if self.stop else 2.)

    def transition_inputs(self, u, context, red):
        return u

    def policy_inputs(self, x, context):
        return x

    def value_inputs(self, action, context):
        return action

    def sample_actions(self, x, deterministic, params, key):
        return (jnp.zeros((*x.shape[:-1], 2)),)

    def Q(self, x, a, params, key):
        value = jnp.full((2, *x.shape[:-1]), 7.)
        if self.poison_after_timeout:
            value = jnp.where(x[..., 0] >= 100, jnp.nan, value)
        return value, None


@struct.dataclass
class AnalyticAgent:
    model: AnalyticModel
    discount: float = .9


@pytest.mark.parametrize("predict_continues", [True, False])
@pytest.mark.parametrize("poison", [True, False])
def test_final_reward_counts_but_timeout_future_and_q_do_not(predict_continues, poison):
    model = AnalyticModel(predict_continues=predict_continues, poison_after_timeout=poison)
    x = jnp.array([[0.], [98.], [99.], [100.]])
    actual = mppi.estimate_value(AnalyticAgent(model), x, jnp.zeros((4, 3, 2)),
                                 jnp.zeros((4, 0)), 3, jax.random.PRNGKey(0))
    np.testing.assert_allclose(actual, [2 + .9 * 2 + .9**2 * 2 + .9**3 * 7, 3.8, 2, 0], atol=1e-6)


def test_learned_capture_stop_remains_independent():
    actual = mppi.estimate_value(AnalyticAgent(AnalyticModel(stop=True)), jnp.array([[20.]]),
                                 jnp.zeros((1, 3, 2)), jnp.zeros((1, 0)), 3,
                                 jax.random.PRNGKey(1))
    np.testing.assert_array_equal(actual, [2.])


@pytest.mark.parametrize("change", ["mlp", "width", "unknown", "normalization"])
def test_contract_rejects_wrong_configuration(change):
    cfg, width = config(), 66
    mean, std = np.zeros(width), np.ones(width)
    if change == "mlp":
        cfg["encoder"]["type"] = "mlp"
    elif change == "width":
        width, mean, std = 65, mean[:65], std[:65]
    elif change == "unknown":
        cfg["world_model"]["transition_contract"] = "unversioned"
    else:
        std[0] = 0
    with pytest.raises(ValueError):
        create_agent(cfg, width, key=jax.random.PRNGKey(0), obs_mean=mean, obs_std=std)


def test_serialization_round_trip_and_explicit_legacy_default(agent):
    restored = serialization.from_bytes(agent, serialization.to_bytes(agent))
    x = encoded_states(agent.model, [98])
    np.testing.assert_array_equal(restored.model.next(x, jnp.zeros((1, 2)), restored.model.dynamics_model.params),
                                  agent.model.next(x, jnp.zeros((1, 2)), agent.model.dynamics_model.params))
    cfg = config("none")
    kwargs = dict(key=jax.random.PRNGKey(7), obs_mean=np.zeros(66), obs_std=np.ones(66))
    explicit = create_agent(cfg, 66, **kwargs)
    del cfg["world_model"]["transition_contract"]
    default = create_agent(cfg, 66, **kwargs)
    assert serialization.to_bytes(explicit) == serialization.to_bytes(default)
    legacy = serialization.from_bytes(default, serialization.to_bytes(explicit))
    raw = jnp.linspace(-.1, .3, 66)
    for first, second in zip(jax.tree.leaves(legacy.act(raw, key=jax.random.PRNGKey(8))),
                             jax.tree.leaves(explicit.act(raw, key=jax.random.PRNGKey(8)))):
        np.testing.assert_array_equal(first, second)
    assert legacy.model.transition_contract == "none" and legacy.model.state_mean == ()


def test_update_reports_wrong_timeout_targets_without_rewriting(agent):
    horizon, batch = 3, 2
    raw = jnp.zeros((horizon, batch, 66)).at[..., 65].set(jnp.array([.98, .99, 1.])[:, None])
    following = raw.at[..., 65].set(jnp.array([.99, 1., 1.])[:, None])
    kwargs = dict(observations=raw, next_observations=following,
                  actions=jnp.zeros((horizon, batch, 2)), rewards=jnp.ones((horizon, batch)),
                  terminated=jnp.zeros((horizon, batch), bool),
                  truncated=jnp.zeros((horizon, batch), bool).at[1].set(True), key=jax.random.PRNGKey(9))
    _, info = agent.update(**kwargs)
    assert not bool(info["physical_timeout_targets_valid"])
    with pytest.raises(ValueError, match="physical timeout"):
        check_update(info, 1)
    # Final padded row has an intentionally wrong label; it is excluded.
    kwargs["terminated"] = kwargs["terminated"].at[1].set(True)
    updated, valid_info = agent.update(**kwargs)
    check_update(valid_info, 2)
    assert bool(valid_info["physical_timeout_targets_valid"])
    assert int(updated.model.dynamics_model.step) == int(agent.model.dynamics_model.step) + 1
