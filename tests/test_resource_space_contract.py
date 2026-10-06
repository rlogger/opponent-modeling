"""Declared spaces must match actual resource-environment observations."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("jaxmarl")

from tag_objectives import SimpleTagObjectivesMPE, SimpleTagResourcesMPE  # noqa: E402


@pytest.mark.parametrize(
    "predators,landmarks,resources", [(1, 0, 4), (3, 2, 4), (5, 3, 7), (2, 1, 1)]
)
@pytest.mark.parametrize("action_type", ["Discrete", "Continuous"])
def test_resource_spaces_match_reset_and_transition_arrays(
    predators, landmarks, resources, action_type
):
    env = SimpleTagResourcesMPE(
        num_adversaries=predators,
        num_good_agents=1,
        num_obs=landmarks,
        num_resources=resources,
        action_type=action_type,
    )
    obs, state = env.reset(jax.random.PRNGKey(17))
    actions = {
        agent: jnp.zeros(5) if action_type == "Continuous" else jnp.int32(0)
        for agent in env.agents
    }
    following, _, _, _, _ = env.step_env(jax.random.PRNGKey(18), state, actions)
    for sample in (obs, following):
        for agent in env.agents:
            assert sample[agent].shape == env.observation_space(agent).shape
            assert np.isfinite(sample[agent]).all()
    # Resource visibility is unchanged: prey alone gets position/status fields.
    common = 4 + 2 * landmarks + 2 * predators
    assert obs[env.adversaries[0]].shape == (common + 2,)
    assert obs[env.good_agents[0]].shape == (common + 3 * resources,)


def test_resource_spaces_match_vectorized_reset():
    env = SimpleTagResourcesMPE(num_adversaries=1, num_good_agents=1, num_obs=0)
    obs, _ = jax.jit(jax.vmap(env.reset))(jax.random.split(jax.random.PRNGKey(2), 3))
    for agent in env.agents:
        assert obs[agent].shape == (3,) + env.observation_space(agent).shape


def test_objective_subclass_keeps_its_distinct_nearest_geometry_schema():
    env = SimpleTagObjectivesMPE()
    obs, _ = env.reset(jax.random.PRNGKey(7))
    assert obs[env.adversaries[0]].shape == env.observation_space(env.adversaries[0]).shape == (17,)
    assert obs[env.good_agents[0]].shape == env.observation_space(env.good_agents[0]).shape == (35,)
