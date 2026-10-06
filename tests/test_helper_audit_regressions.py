"""Regression contracts for class mapping, validation, terminal states and RNG keys."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mopa.belief import softmax_belief_from_logits
from mopa.metrics import calibration_metrics, temperature_scale_logits
from mopa.sim_planner import MPPIConfig, imagined_returns, mppi_plan
from tag_objectives import make_env


def test_belief_hard_returns_declared_noncontiguous_classes():
    belief = softmax_belief_from_logits(np.array([[0., 10.], [10., 0.]]), classes=np.array([5, 9]))
    np.testing.assert_array_equal(belief.hard, [9, 5])


@pytest.mark.parametrize("labels", [np.array([[0], [1]]), np.array([0.1, 1.9]), np.array(0), np.array([False, True]), np.array([2**64-1, 1], np.uint64)])
def test_probability_and_temperature_metrics_reject_malformed_labels(labels):
    probabilities = np.array([[.9, .1], [.2, .8]])
    with pytest.raises(ValueError):
        calibration_metrics(probabilities, labels)
    with pytest.raises(ValueError):
        temperature_scale_logits(np.log(probabilities), labels, grid_size=3)


def test_probability_metrics_keep_correct_integer_label_result():
    result = calibration_metrics(np.array([[.9, .1], [.2, .8]]), np.array([0, 1], np.uint32))
    assert result["nll"] == pytest.approx(-np.log([.9, .8]).mean())
    assert sum(result["reliability"]["count"]) == 2


def zero_red(observation, context):
    del context
    return jnp.zeros((observation.shape[0], 2))


@pytest.mark.parametrize("terminal", ["capture", "timeout"])
def test_simulator_planner_terminal_start_has_no_future_reward(terminal):
    env = make_env("capture", continuous=True)
    _, state = env.reset(jax.random.PRNGKey(1))
    if terminal == "capture":
        state = state.replace(p_pos=state.p_pos.at[1].set(state.p_pos[0]), capture_t=jnp.array(1), step=jnp.array(1))
    else:
        state = state.replace(p_pos=state.p_pos.at[1].set(jnp.array([3., 3.])), step=jnp.array(env.max_steps))
    state = state.replace(done=jnp.ones_like(state.done, dtype=bool))
    states = jax.tree.map(lambda value: value[None], state)
    keys = jax.random.split(jax.random.PRNGKey(2), (2, 1))
    returns = imagined_returns(env, states, jnp.zeros((1, 2, 2)), jnp.zeros((1, 0)), zero_red, keys, discount=1.)
    np.testing.assert_array_equal(returns, 0)


def test_simulator_planner_typed_and_legacy_keys_agree():
    env = make_env("capture", continuous=True)
    _, state = env.reset(jax.random.PRNGKey(1))
    states = jax.tree.map(lambda value: value[None], state)
    cfg = MPPIConfig(horizon=2, population_size=2, num_elites=1, mppi_iterations=1)
    arguments = (env, zero_red, cfg, states, jnp.zeros((1, 0)), jnp.zeros((1, 2, 2)))
    legacy = mppi_plan(*arguments, jax.random.PRNGKey(2))
    typed = mppi_plan(*arguments, jax.random.key(2))
    for a, b in zip(jax.tree.leaves(legacy), jax.tree.leaves(typed), strict=True):
        np.testing.assert_array_equal(a, b)
