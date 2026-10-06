"""Game-rule and JaxMARL interface checks, independent of learned policies."""

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("jaxmarl")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jaxmarl.wrappers.baselines import LogWrapper  # noqa: E402

from spatial_blotto import SpatialBlotto  # noqa: E402


def _still(env):
    return {a: jnp.zeros(2) for a in env.agents}


def _assert_tree_equal(actual, expected):
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("team_size", [2, 3])
def test_reset_is_seeded_symmetric_and_observations_fit_spaces(team_size):
    env = SpatialBlotto(team_size=team_size)
    obs, state = env.reset(jax.random.PRNGKey(0))
    _assert_tree_equal((obs, state), env.reset(jax.random.PRNGKey(0)))
    assert not np.array_equal(state.p_pos, env.reset(jax.random.PRNGKey(1))[1].p_pos)
    np.testing.assert_allclose(
        state.p_pos[team_size:], state.p_pos[:team_size] * np.array([-1, 1])
    )
    assert set(obs) == set(env.agents)
    assert set(env.agent_classes) == {"red", "blue"}
    assert env.get_state(state).shape == (env.state_size,)
    for agent in env.agents:
        assert obs[agent].shape == env.observation_space(agent).shape
        assert env.observation_space(agent).contains(obs[agent])
        assert env.action_space(agent).shape == (2,)
    np.testing.assert_array_equal(env.zone_counts(state), np.zeros((2, 3)))


@pytest.mark.parametrize("reward_mode,expected", [("ownership", [1, 2]), ("zero_sum", [-1, 1])])
def test_slide_allocation_scores_and_team_exchange(reward_mode, expected):
    env = SpatialBlotto(reward_mode=reward_mode)
    _, state = env.reset(jax.random.PRNGKey(0))
    # Source example: red (2,1,0), blue (0,2,1).
    state = state.replace(p_pos=env.zone_centers[jnp.array([0, 0, 1, 1, 1, 2])])
    _, next_state, rewards, _, info = env.step_env(jax.random.PRNGKey(1), state, _still(env))
    np.testing.assert_array_equal(info["zone_counts"], [[2, 1, 0], [0, 2, 1]])
    np.testing.assert_array_equal(info["zone_owners"], [1, -1, -1])
    np.testing.assert_array_equal(info["team_rewards"], expected)
    np.testing.assert_array_equal(next_state.team_scores, [1, 2])
    assert all(float(rewards[a]) == expected[0] for a in env.red_agents)
    assert all(float(rewards[a]) == expected[1] for a in env.blue_agents)

    exchanged = state.replace(p_pos=jnp.roll(state.p_pos, env.team_size, axis=0))
    _, _, _, _, swapped_info = env.step_env(jax.random.PRNGKey(1), exchanged, _still(env))
    np.testing.assert_array_equal(swapped_info["team_rewards"], expected[::-1])
    np.testing.assert_array_equal(swapped_info["zone_owners"], -info["zone_owners"])


@pytest.mark.parametrize("team_size", [2, 3])
@pytest.mark.parametrize("reward_mode", ["ownership", "zero_sum"])
def test_tied_and_empty_zones_are_neutral(team_size, reward_mode):
    env = SpatialBlotto(team_size=team_size, reward_mode=reward_mode)
    _, state = env.reset(jax.random.PRNGKey(0))
    # Two contested zones and one empty zone in 2v2; three ties in 3v3.
    positions = jnp.tile(env.zone_centers[:team_size], (2, 1))
    state = state.replace(p_pos=positions)
    _, _, rewards, _, info = env.step_env(jax.random.PRNGKey(1), state, _still(env))
    np.testing.assert_array_equal(info["zone_owners"], [0, 0, 0])
    assert all(float(value) == 0 for value in rewards.values())


def test_scoring_follows_movement_and_boundary_counts():
    env = SpatialBlotto(team_size=2, zone_radius=0.25, dt=0.125)
    _, state = env.reset(jax.random.PRNGKey(0))
    # The first unit moves from outside A onto its exact, inclusive boundary.
    state = state.replace(p_pos=state.p_pos.at[0].set(env.zone_centers[0] + jnp.array([0.375, 0])))
    assert int(env.zone_counts(state)[0, 0]) == 0
    actions = _still(env)
    actions[env.red_agents[0]] = jnp.array([-1.0, 0])
    _, next_state, rewards, _, info = env.step_env(jax.random.PRNGKey(1), state, actions)
    assert int(info["zone_counts"][0, 0]) == 1
    assert float(rewards[env.red_agents[0]]) == 1
    np.testing.assert_allclose(next_state.p_pos[0], env.zone_centers[0] + np.array([0.25, 0]))


def test_speed_limit_action_clipping_and_wall_stop():
    env = SpatialBlotto(team_size=2)
    _, state = env.reset(jax.random.PRNGKey(0))
    state = state.replace(p_pos=jnp.zeros_like(state.p_pos).at[-1].set(jnp.array([env.arena, 0])))
    actions = {a: jnp.array([5.0, 5.0]) for a in env.agents}
    actions[env.agents[-1]] = jnp.array([1.0, 0])
    obs, next_state, _, _, _ = env.step_env(jax.random.PRNGKey(1), state, actions)
    np.testing.assert_allclose(next_state.p_vel[0], np.array([1, 1]) / np.sqrt(2), rtol=1e-6)
    np.testing.assert_array_equal(next_state.p_pos[-1], state.p_pos[-1])
    np.testing.assert_array_equal(next_state.p_vel[-1], [0, 0])
    assert np.all(np.linalg.norm(next_state.p_vel, axis=-1) <= env.max_speed + 1e-6)
    assert all(bool(env.observation_space(a).contains(obs[a])) for a in env.agents)


def test_axis_motion_stays_inside_declared_observation_space():
    env = SpatialBlotto()
    _, state = env.reset(jax.random.PRNGKey(0))
    actions = {a: jnp.array([1.0, 0.0]) for a in env.agents}
    for _ in range(30):
        obs, state, _, _, _ = env.step_env(jax.random.PRNGKey(1), state, actions)
        assert all(bool(env.observation_space(a).contains(obs[a])) for a in env.agents)


def test_terminal_state_absorbs_and_step_autoresets_with_terminal_info():
    env = SpatialBlotto(max_steps=1)
    _, state = env.reset(jax.random.PRNGKey(0))
    state = state.replace(p_pos=env.zone_centers[jnp.array([0, 0, 1, 1, 1, 2])])
    obs, terminal, rewards, dones, info = env.step_env(jax.random.PRNGKey(1), state, _still(env))
    assert all(bool(done) for done in dones.values())
    assert info["terminated"] and not info["truncated"]
    _, still_terminal, later_rewards, _, _ = env.step_env(jax.random.PRNGKey(2), terminal, _still(env))
    _assert_tree_equal(still_terminal, terminal)
    assert all(float(value) == 0 for value in later_rewards.values())

    reset_obs, reset_state = env.reset(jax.random.PRNGKey(3))
    actual_obs, actual_state, actual_rewards, actual_dones, actual_info = env.step(
        jax.random.PRNGKey(1), state, _still(env), reset_state=reset_state
    )
    _assert_tree_equal((actual_obs, actual_state), (reset_obs, reset_state))
    _assert_tree_equal((actual_rewards, actual_dones), (rewards, dones))
    _assert_tree_equal(actual_info["terminal_observation"], obs)
    np.testing.assert_array_equal(actual_info["terminal_state"], env.get_state(terminal))


@pytest.mark.parametrize("team_size", [2, 3])
def test_jit_vmap_scan_and_generic_log_wrapper(team_size):
    env = SpatialBlotto(team_size=team_size, max_steps=3)
    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    _, states = jax.jit(jax.vmap(env.reset))(keys)
    actions = {a: jnp.zeros((4, 2)) for a in env.agents}

    def advance(current, key):
        _, following, _, done, _ = jax.vmap(env.step)(jax.random.split(key, 4), current, actions)
        return following, done["__all__"]

    final, dones = jax.jit(lambda s: jax.lax.scan(advance, s, keys))(states)
    np.testing.assert_array_equal(dones[:, 0], [False, False, True, False])
    np.testing.assert_array_equal(final.step, np.ones(4))

    wrapped = LogWrapper(env)
    _, wrapped_state = wrapped.reset(keys[0])
    for key in keys[:3]:
        _, wrapped_state, _, _, info = wrapped.step(key, wrapped_state, _still(env))
    np.testing.assert_array_equal(info["returned_episode_lengths"], np.full(env.num_agents, 3))
    assert np.all(info["returned_episode"])


@pytest.mark.parametrize("kwargs", [
    {"team_size": 0}, {"team_size": True}, {"max_steps": 0},
    {"dt": float("nan")}, {"max_speed": 0}, {"zone_radius": 0.7},
    {"reward_mode": "unknown"},
])
def test_invalid_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        SpatialBlotto(**kwargs)
