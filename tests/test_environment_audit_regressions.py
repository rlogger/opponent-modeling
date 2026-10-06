"""Numerical regressions for audited environment reporting contracts."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("jaxmarl")

from tag_objectives import SimpleTagObjectivesMPE, evaluate_policy  # noqa: E402


@pytest.mark.parametrize("recording_steps,expected", [(1, 1), (2, 2), (5, 2)])
def test_uncaptured_survival_counts_actual_transitions(recording_steps, expected):
    """A timeout stops survival; longer recording requests cannot extend it."""
    env = SimpleTagObjectivesMPE(max_steps=2)

    def stationary(obs, key):
        del key
        return {
            agent: jnp.zeros(value.shape[0], dtype=jnp.int32)
            for agent, value in obs.items()
        }

    result = evaluate_policy(
        env,
        stationary,
        n_eps=4,
        key=jax.random.PRNGKey(7),
        num_steps=recording_steps,
    )
    np.testing.assert_array_equal(result.capture_rate, np.zeros(4))
    np.testing.assert_array_equal(result.survival_time, np.full(4, expected))


def test_capture_survival_counts_first_capture_transition():
    """The same counter preserves early capture rather than recording length."""

    class CapturedAtReset(SimpleTagObjectivesMPE):
        def reset(self, key):
            _, state = super().reset(key)
            positions = state.p_pos.at[0].set(jnp.zeros(2)).at[1].set(jnp.zeros(2))
            state = state.replace(p_pos=positions, p_vel=jnp.zeros_like(state.p_vel))
            return self.get_obs(state), state

    env = CapturedAtReset(max_steps=5)

    def stationary(obs, key):
        del key
        return {
            agent: jnp.zeros(value.shape[0], dtype=jnp.int32)
            for agent, value in obs.items()
        }

    result = evaluate_policy(
        env, stationary, n_eps=2, key=jax.random.PRNGKey(3), num_steps=5
    )
    np.testing.assert_array_equal(result.capture_rate, np.ones(2))
    np.testing.assert_array_equal(result.survival_time, np.ones(2))
