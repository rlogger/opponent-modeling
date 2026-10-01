"""Game-rule and JaxMARL checks against independent scalar NumPy rules.

No learned policies, production scoring helpers, or production movement helpers
are used to construct the differential-test expectations.
"""

import itertools
import math

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
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_array_equal(a, b)


def _reference_counts(positions, team_size, radius):
    centers = [
        (0.0, 0.75),
        (-math.sqrt(3) * 0.375, -0.375),
        (math.sqrt(3) * 0.375, -0.375),
    ]
    counts = np.zeros((2, 3), dtype=np.int32)
    for unit, position in enumerate(positions):
        for zone, center in enumerate(centers):
            distance_squared = sum(
                (float(position[k]) - center[k]) ** 2 for k in range(2)
            )
            if distance_squared <= radius * radius:
                counts[unit // team_size, zone] += 1
    return counts


def _reference_step(positions, actions, *, team_size, dt, max_speed, radius):
    """Scalar implementation of the stated equations, calculated in float64."""
    following = np.empty_like(positions, dtype=np.float64)
    for unit in range(len(positions)):
        command = [min(1.0, max(-1.0, float(x))) for x in actions[unit]]
        length = max(1.0, math.sqrt(command[0] ** 2 + command[1] ** 2))
        for coordinate in range(2):
            movement = dt * max_speed * command[coordinate] / length
            following[unit, coordinate] = min(
                1.5, max(-1.5, float(positions[unit, coordinate]) + movement)
            )
    velocity = (following - positions) / dt
    counts = _reference_counts(following, team_size, radius)
    owners = np.array(
        [
            1
            if counts[0, zone] > counts[1, zone]
            else -1
            if counts[1, zone] > counts[0, zone]
            else 0
            for zone in range(3)
        ],
        dtype=np.int32,
    )
    scores = np.array([list(owners).count(1), list(owners).count(-1)], dtype=np.int32)
    return following, velocity, counts, owners, scores


@pytest.mark.parametrize("team_size", [1, 2, 3, 5])
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
    assert env.get_state(state).dtype == jnp.float32
    assert env.state_space().shape == (env.state_size,)
    assert bool(env.state_space().contains(env.get_state(state)))
    assert state.team_scores.dtype == jnp.int32
    assert state.step.dtype == jnp.int32
    assert state.done.dtype == jnp.bool_
    for agent in env.agents:
        assert obs[agent].shape == env.observation_space(agent).shape
        assert obs[agent].dtype == jnp.float32
        assert env.observation_space(agent).contains(obs[agent])
        assert env.action_space(agent).shape == (2,)
    np.testing.assert_array_equal(env.zone_counts(state), np.zeros((2, 3)))


def test_fixed_geometry_has_equal_disjoint_circles_inside_the_arena():
    env = SpatialBlotto()
    expected = np.array(
        [[0, 0.75], [-math.sqrt(3) * 0.375, -0.375], [math.sqrt(3) * 0.375, -0.375]]
    )
    np.testing.assert_allclose(env.zone_centers, expected, rtol=1e-7)
    assert env.num_zones == 3
    assert env.num_agents == 6
    assert env.agent_classes == {
        "red": ("red_0", "red_1", "red_2"),
        "blue": ("blue_0", "blue_1", "blue_2"),
    }
    for left, right in itertools.combinations(expected, 2):
        assert np.linalg.norm(left - right) > 2 * env.zone_radius
    assert np.all(np.abs(expected) + env.zone_radius < env.arena)


@pytest.mark.parametrize("reward_mode", ["ownership", "zero_sum"])
def test_every_default_budget_allocation_pair_matches_strict_majority(reward_mode):
    """All 10 allocations of three units, crossed with all 10 opponent allocations."""
    env = SpatialBlotto(reward_mode=reward_mode)
    allocations = [a for a in itertools.product(range(4), repeat=3) if sum(a) == 3]
    pairs = list(itertools.product(allocations, repeat=2))
    assert len(pairs) == 100
    states = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(44), len(pairs)))[
        1
    ]
    positions = [
        np.concatenate(
            [np.repeat(np.asarray(env.zone_centers), a, axis=0) for a in pair]
        )
        for pair in pairs
    ]
    states = states.replace(p_pos=jnp.asarray(positions))
    actions = {a: jnp.zeros((len(pairs), 2)) for a in env.agents}
    _, following, rewards, dones, info = jax.jit(jax.vmap(env.step_env))(
        jax.random.split(jax.random.PRNGKey(45), len(pairs)), states, actions
    )
    counts = np.asarray(pairs, dtype=np.int32)
    owners = np.sign(counts[:, 0] - counts[:, 1])
    scores = np.stack([(owners == 1).sum(axis=1), (owners == -1).sum(axis=1)], axis=1)
    expected_rewards = (
        scores if reward_mode == "ownership" else scores - scores[:, ::-1]
    )
    np.testing.assert_array_equal(info["zone_counts"], counts)
    np.testing.assert_array_equal(info["zone_owners"], owners)
    np.testing.assert_array_equal(info["ownership_scores"], scores)
    np.testing.assert_array_equal(following.team_scores, scores)
    np.testing.assert_array_equal(info["team_rewards"], expected_rewards)
    assert not np.asarray(dones["__all__"]).any()
    assert np.all(np.sum(scores, axis=1) <= 3)
    for unit, agent in enumerate(env.agents):
        np.testing.assert_array_equal(
            rewards[agent], expected_rewards[:, unit // env.team_size]
        )


@pytest.mark.parametrize(
    "team_size,reward_mode", [(1, "ownership"), (3, "ownership"), (5, "zero_sum")]
)
def test_randomized_transitions_match_independent_scalar_numpy(team_size, reward_mode):
    env = SpatialBlotto(
        team_size=team_size, dt=0.19, max_speed=1.7, reward_mode=reward_mode
    )
    count = 48
    rng = np.random.default_rng(1729)
    keys = jax.random.split(jax.random.PRNGKey(5), count)
    _, states = jax.vmap(env.reset)(keys)
    positions = rng.uniform(-1.5, 1.5, (count, env.num_agents, 2)).astype(np.float32)
    # Include dense overlapping teams near zones, sparse random fields, and walls.
    positions[: count // 2] = np.asarray(env.zone_centers)[
        rng.integers(0, 3, (count // 2, env.num_agents))
    ] + rng.uniform(-0.3, 0.3, (count // 2, env.num_agents, 2))
    positions[-1, :, 0] = 1.5
    commands = rng.uniform(-3, 3, positions.shape).astype(np.float32)
    old_scores = rng.integers(0, 15, (count, 2), dtype=np.int32)
    old_velocities = rng.uniform(-1, 1, positions.shape).astype(np.float32)
    states = states.replace(
        p_pos=jnp.asarray(positions),
        p_vel=jnp.asarray(old_velocities),
        step=jnp.full(count, 10, dtype=jnp.int32),
        team_scores=jnp.asarray(old_scores),
    )
    actions = {
        agent: jnp.asarray(commands[:, unit]) for unit, agent in enumerate(env.agents)
    }
    obs, following, rewards, dones, info = jax.jit(jax.vmap(env.step_env))(
        keys, states, actions
    )
    expected = [
        _reference_step(
            p,
            a,
            team_size=team_size,
            dt=env.dt,
            max_speed=env.max_speed,
            radius=env.zone_radius,
        )
        for p, a in zip(positions, commands, strict=True)
    ]
    next_positions, velocities, counts, owners, scores = map(
        np.asarray, zip(*expected, strict=True)
    )
    np.testing.assert_allclose(following.p_pos, next_positions, rtol=1e-6, atol=2e-7)
    np.testing.assert_allclose(following.p_vel, velocities, rtol=2e-5, atol=1e-6)
    np.testing.assert_array_equal(info["zone_counts"], counts)
    np.testing.assert_array_equal(info["zone_owners"], owners)
    np.testing.assert_array_equal(info["ownership_scores"], scores)
    np.testing.assert_array_equal(following.team_scores, old_scores + scores)
    np.testing.assert_array_equal(following.step, np.full(count, 11))
    expected_rewards = (
        scores if reward_mode == "ownership" else scores - scores[:, ::-1]
    )
    np.testing.assert_array_equal(info["team_rewards"], expected_rewards)
    assert not np.asarray(dones["__all__"]).any()
    assert np.all(np.sum(counts, axis=-1) <= team_size)
    assert np.all(np.sum(scores, axis=-1) <= 3)
    assert np.all(np.linalg.norm(following.p_vel, axis=-1) <= env.max_speed + 2e-6)
    assert np.all(np.abs(following.p_pos) <= env.arena)
    for unit, agent in enumerate(env.agents):
        np.testing.assert_array_equal(
            rewards[agent], expected_rewards[:, unit // team_size]
        )
        assert obs[agent].shape == (count, env.obs_size)
        assert bool(env.observation_space(agent).contains(obs[agent]))
    vectors = jax.vmap(env.get_state)(following)
    assert vectors.shape == (count, env.state_size)
    assert bool(env.state_space().contains(vectors))


@pytest.mark.parametrize(
    "reward_mode,expected", [("ownership", [1, 2]), ("zero_sum", [-1, 1])]
)
def test_slide_allocation_scores_and_team_exchange(reward_mode, expected):
    env = SpatialBlotto(reward_mode=reward_mode)
    _, state = env.reset(jax.random.PRNGKey(0))
    # Source example: red (2,1,0), blue (0,2,1).
    state = state.replace(p_pos=env.zone_centers[jnp.array([0, 0, 1, 1, 1, 2])])
    _, next_state, rewards, _, info = env.step_env(
        jax.random.PRNGKey(1), state, _still(env)
    )
    np.testing.assert_array_equal(info["zone_counts"], [[2, 1, 0], [0, 2, 1]])
    np.testing.assert_array_equal(info["zone_owners"], [1, -1, -1])
    np.testing.assert_array_equal(info["team_rewards"], expected)
    np.testing.assert_array_equal(next_state.team_scores, [1, 2])
    assert all(float(rewards[a]) == expected[0] for a in env.red_agents)
    assert all(float(rewards[a]) == expected[1] for a in env.blue_agents)

    exchanged = state.replace(p_pos=jnp.roll(state.p_pos, env.team_size, axis=0))
    _, _, _, _, swapped_info = env.step_env(
        jax.random.PRNGKey(1), exchanged, _still(env)
    )
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
    state = state.replace(
        p_pos=state.p_pos.at[0].set(env.zone_centers[0] + jnp.array([0.375, 0]))
    )
    assert int(env.zone_counts(state)[0, 0]) == 0
    actions = _still(env)
    actions[env.red_agents[0]] = jnp.array([-1.0, 0])
    _, next_state, rewards, _, info = env.step_env(
        jax.random.PRNGKey(1), state, actions
    )
    assert int(info["zone_counts"][0, 0]) == 1
    assert float(rewards[env.red_agents[0]]) == 1
    np.testing.assert_allclose(
        next_state.p_pos[0], env.zone_centers[0] + np.array([0.25, 0])
    )


def test_speed_limit_action_clipping_and_wall_stop():
    env = SpatialBlotto(team_size=2)
    _, state = env.reset(jax.random.PRNGKey(0))
    state = state.replace(
        p_pos=jnp.zeros_like(state.p_pos).at[-1].set(jnp.array([env.arena, 0]))
    )
    actions = {a: jnp.array([5.0, 5.0]) for a in env.agents}
    actions[env.agents[-1]] = jnp.array([1.0, 0])
    obs, next_state, _, _, _ = env.step_env(jax.random.PRNGKey(1), state, actions)
    np.testing.assert_allclose(
        next_state.p_vel[0], np.array([1, 1]) / np.sqrt(2), rtol=1e-6
    )
    np.testing.assert_array_equal(next_state.p_pos[-1], state.p_pos[-1])
    np.testing.assert_array_equal(next_state.p_vel[-1], [0, 0])
    assert np.all(np.linalg.norm(next_state.p_vel, axis=-1) <= env.max_speed + 1e-6)
    assert all(bool(env.observation_space(a).contains(obs[a])) for a in env.agents)


def test_clipping_precedes_disk_projection_and_velocity_has_no_inertia():
    env = SpatialBlotto(team_size=1, dt=0.25, max_speed=2)
    _, state = env.reset(jax.random.PRNGKey(3))
    state = state.replace(p_pos=jnp.zeros((2, 2)), p_vel=jnp.ones((2, 2)))
    actions = {"red_0": jnp.array([2, 0.5]), "blue_0": jnp.zeros(2)}
    _, following, _, _, _ = env.step_env(jax.random.PRNGKey(4), state, actions)
    expected = 2 * np.array([1, 0.5]) / math.sqrt(1.25)
    np.testing.assert_allclose(following.p_vel[0], expected, rtol=1e-7)
    np.testing.assert_allclose(following.p_pos[0], expected * 0.25, rtol=1e-7)
    np.testing.assert_array_equal(following.p_vel[1], np.zeros(2))
    np.testing.assert_array_equal(following.p_pos[1], np.zeros(2))


def test_wall_sliding_corner_clipping_and_overlapping_units():
    env = SpatialBlotto(team_size=2, dt=0.25)
    _, state = env.reset(jax.random.PRNGKey(3))
    positions = jnp.array([[1.5, 0], [1.5, 1.5], [-1.5, -1.5], [-1.5, -1.5]])
    state = state.replace(p_pos=positions)
    actions = dict(
        zip(
            env.agents,
            [
                jnp.array([1, 1]),
                jnp.array([1, 1]),
                jnp.array([-1, -1]),
                jnp.array([-1, -1]),
            ],
            strict=True,
        )
    )
    _, following, _, _, _ = env.step_env(jax.random.PRNGKey(4), state, actions)
    np.testing.assert_allclose(following.p_vel[0], [0, 1 / math.sqrt(2)], rtol=1e-6)
    np.testing.assert_array_equal(following.p_pos[1:], positions[1:])
    np.testing.assert_array_equal(following.p_vel[1:], np.zeros((3, 2)))


def test_boundary_membership_is_inclusive_and_excludes_next_float_outside():
    env = SpatialBlotto(team_size=3, zone_radius=0.25)
    _, state = env.reset(jax.random.PRNGKey(7))
    radius = np.float32(0.25)
    offsets = np.array(
        [
            np.nextafter(radius, np.float32(0)),
            radius,
            np.nextafter(radius, np.float32(np.inf)),
        ]
    )
    state = state.replace(
        p_pos=state.p_pos.at[:3].set(
            jnp.stack((jnp.asarray(offsets), jnp.full(3, 0.75)), axis=1)
        )
    )
    np.testing.assert_array_equal(env.zone_counts(state), [[2, 0, 0], [0, 0, 0]])


def test_tiny_radius_does_not_count_outside_points_through_squared_underflow():
    env = SpatialBlotto(zone_radius=1e-30)
    _, state = env.reset(jax.random.PRNGKey(7))
    # All are representable float32 positions, but their squared x distances
    # underflow. The third unit is ten orders of magnitude outside the circle.
    state = state.replace(
        p_pos=state.p_pos.at[:3].set(
            jnp.array([[0, 0.75], [1e-30, 0.75], [1e-20, 0.75]], dtype=jnp.float32)
        )
    )
    np.testing.assert_array_equal(env.zone_counts(state), [[2, 0, 0], [0, 0, 0]])


def test_radius_near_tangency_requires_a_float32_separation_margin():
    tangent = math.sqrt(3) * 0.375
    with pytest.raises(ValueError, match="float32 margin"):
        SpatialBlotto(zone_radius=np.nextafter(tangent, 0))
    margin = 8 * np.finfo(np.float32).eps
    with pytest.raises(ValueError, match="float32 margin"):
        SpatialBlotto(zone_radius=tangent - margin / 2)
    env = SpatialBlotto(zone_radius=tangent - 2 * margin)
    _, state = env.reset(jax.random.PRNGKey(17))
    midpoints = jnp.stack(
        [
            (env.zone_centers[a] + env.zone_centers[b]) / 2
            for a, b in itertools.combinations(range(3), 2)
        ]
    )
    state = state.replace(p_pos=jnp.concatenate((midpoints, midpoints)))
    np.testing.assert_array_equal(env.zone_counts(state), np.zeros((2, 3)))


def test_post_movement_scoring_handles_simultaneous_entry_and_exit():
    env = SpatialBlotto(team_size=1, zone_radius=0.25, dt=0.25)
    _, state = env.reset(jax.random.PRNGKey(7))
    state = state.replace(p_pos=jnp.array([[0.125, 0.75], [0.375, 0.75]]))
    np.testing.assert_array_equal(env.zone_owners(state), [1, 0, 0])
    actions = {"red_0": jnp.array([1.0, 0]), "blue_0": jnp.array([-1.0, 0])}
    _, _, rewards, _, info = env.step_env(jax.random.PRNGKey(8), state, actions)
    np.testing.assert_array_equal(info["zone_owners"], [-1, 0, 0])
    assert rewards == {"red_0": 0.0, "blue_0": 1.0}


@pytest.mark.parametrize("reward_mode", ["ownership", "zero_sum"])
def test_transition_respects_team_exchange_mirror_and_teammate_permutation(reward_mode):
    env = SpatialBlotto(reward_mode=reward_mode, dt=0.125)
    key = jax.random.PRNGKey(32)
    _, state = env.reset(key)
    state = state.replace(
        p_pos=env.zone_centers[jnp.array([0, 0, 1, 1, 1, 2])],
        team_scores=jnp.array([7, 4], dtype=jnp.int32),
        step=jnp.array(4, dtype=jnp.int32),
    )
    commands = jnp.array(
        [[0.1, 0.2], [-0.3, 0.1], [0.25, -0.4], [0.2, 0.15], [-0.1, -0.2], [0.3, 0.2]]
    )

    def advance(s, a):
        return env.step_env(key, s, {agent: a[i] for i, agent in enumerate(env.agents)})

    original = advance(state, commands)
    _, following, _, _, info = original
    exchange = jnp.array([3, 4, 5, 0, 1, 2])
    exchanged = advance(
        state.replace(
            p_pos=state.p_pos[exchange],
            p_vel=state.p_vel[exchange],
            team_scores=state.team_scores[::-1],
        ),
        commands[exchange],
    )
    np.testing.assert_allclose(exchanged[1].p_pos, following.p_pos[exchange])
    np.testing.assert_array_equal(exchanged[1].team_scores, following.team_scores[::-1])
    np.testing.assert_array_equal(exchanged[4]["zone_owners"], -info["zone_owners"])
    np.testing.assert_array_equal(
        exchanged[4]["team_rewards"], info["team_rewards"][::-1]
    )

    mirror = jnp.array([-1.0, 1.0])
    mirrored = advance(
        state.replace(p_pos=state.p_pos * mirror, p_vel=state.p_vel * mirror),
        commands * mirror,
    )
    np.testing.assert_allclose(mirrored[1].p_pos, following.p_pos * mirror)
    np.testing.assert_array_equal(
        mirrored[4]["zone_counts"], info["zone_counts"][:, jnp.array([0, 2, 1])]
    )
    np.testing.assert_array_equal(mirrored[4]["team_rewards"], info["team_rewards"])

    permutation = jnp.array([2, 0, 1, 4, 5, 3])
    permuted = advance(
        state.replace(p_pos=state.p_pos[permutation], p_vel=state.p_vel[permutation]),
        commands[permutation],
    )
    np.testing.assert_allclose(permuted[1].p_pos, following.p_pos[permutation])
    np.testing.assert_array_equal(permuted[4]["zone_counts"], info["zone_counts"])
    np.testing.assert_array_equal(permuted[4]["team_rewards"], info["team_rewards"])


def test_transition_is_key_independent_and_uses_agent_keys_not_dict_order():
    env = SpatialBlotto()
    key = jax.random.PRNGKey(2)
    _, state = env.reset(key)
    actions = {a: jnp.array([0.1 * i, -0.1 * i]) for i, a in enumerate(env.agents)}
    ordered = env.step_env(key, state, actions)
    reversed_keys = env.step_env(
        jax.random.PRNGKey(9), state, dict(reversed(list(actions.items())))
    )
    _assert_tree_equal(ordered, reversed_keys)


def test_observation_layout_and_centralized_state_are_explicit():
    env = SpatialBlotto(max_steps=20)
    _, state = env.reset(jax.random.PRNGKey(9))
    state = state.replace(
        p_pos=env.zone_centers[jnp.array([0, 0, 1, 1, 1, 2])],
        p_vel=jnp.asarray(np.arange(12).reshape(6, 2) / 20, dtype=jnp.float32),
        step=jnp.array(5, dtype=jnp.int32),
        team_scores=jnp.array([4, 8], dtype=jnp.int32),
    )
    obs = env.get_obs(state)
    positions, velocities = np.asarray(state.p_pos), np.asarray(state.p_vel)
    for unit, agent in enumerate(env.agents):
        own_team = range(0, 3) if unit < 3 else range(3, 6)
        opponents = range(3, 6) if unit < 3 else range(0, 3)
        others = [i for i in own_team if i != unit] + list(opponents)
        expected = np.concatenate(
            [
                positions[unit] / 1.5,
                velocities[unit],
                ((positions[others] - positions[unit]) / 3).ravel(),
                velocities[others].ravel(),
                ((np.asarray(env.zone_centers) - positions[unit]) / 3).ravel(),
                np.array([1, -1, -1]) * (1 if unit < 3 else -1),
                [0.25],
                np.eye(3)[unit % 3],
            ]
        )
        np.testing.assert_allclose(obs[agent], expected, rtol=1e-7)
    expected_state = np.concatenate(
        [(positions / 1.5).ravel(), velocities.ravel(), [0.25]]
    )
    np.testing.assert_allclose(env.get_state(state), expected_state, rtol=1e-7)
    changed_scores = state.replace(team_scores=jnp.array([999, 999], dtype=jnp.int32))
    _assert_tree_equal(env.get_obs(state), env.get_obs(changed_scores))
    np.testing.assert_array_equal(env.get_state(state), env.get_state(changed_scores))


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
    obs, terminal, rewards, dones, info = env.step_env(
        jax.random.PRNGKey(1), state, _still(env)
    )
    assert all(bool(done) for done in dones.values())
    assert info["terminated"] and not info["truncated"]
    still_obs, still_terminal, later_rewards, later_dones, later_info = env.step_env(
        jax.random.PRNGKey(2), terminal, {a: jnp.ones(2) for a in env.agents}
    )
    _assert_tree_equal(still_terminal, terminal)
    _assert_tree_equal(still_obs, obs)
    _assert_tree_equal(later_dones, dones)
    np.testing.assert_array_equal(later_info["ownership_scores"], [0, 0])
    np.testing.assert_array_equal(later_info["team_scores"], terminal.team_scores)
    assert all(float(value) == 0 for value in later_rewards.values())

    reset_obs, reset_state = env.reset(jax.random.PRNGKey(3))
    actual_obs, actual_state, actual_rewards, actual_dones, actual_info = env.step(
        jax.random.PRNGKey(1), state, _still(env), reset_state=reset_state
    )
    _assert_tree_equal((actual_obs, actual_state), (reset_obs, reset_state))
    _assert_tree_equal((actual_rewards, actual_dones), (rewards, dones))
    _assert_tree_equal(actual_info["terminal_observation"], obs)
    np.testing.assert_array_equal(
        actual_info["terminal_state"], env.get_state(terminal)
    )

    key = jax.random.PRNGKey(81)
    reset_key = jax.random.split(key)[1]
    actual_obs, actual_state, _, actual_dones, info = env.step(key, state, _still(env))
    _assert_tree_equal((actual_obs, actual_state), env.reset(reset_key))
    assert actual_dones["__all__"]
    _assert_tree_equal(info["terminal_observation"], obs)


@pytest.mark.parametrize("reward_mode", ["ownership", "zero_sum"])
def test_finite_horizon_cumulative_scores_and_reward_bounds(reward_mode):
    env = SpatialBlotto(max_steps=7, reward_mode=reward_mode)
    key = jax.random.PRNGKey(51)
    _, state = env.reset(key)
    state = state.replace(p_pos=env.zone_centers[jnp.array([0, 0, 1, 1, 1, 2])])
    for step in range(1, 8):
        obs, state, rewards, dones, info = env.step_env(key, state, _still(env))
        assert int(state.step) == step
        assert bool(dones["__all__"]) == (step == 7)
        assert all(bool(done) == (step == 7) for done in dones.values())
        assert bool(info["terminated"]) == (step == 7)
        assert not bool(info["truncated"])
        np.testing.assert_array_equal(state.team_scores, [step, step * 2])
        assert state.team_scores.dtype == jnp.int32
        assert int(jnp.sum(state.team_scores)) <= env.num_zones * step
        lower = 0 if reward_mode == "ownership" else -3
        assert all(lower <= float(r) <= 3 for r in rewards.values())
        assert all(r.dtype == jnp.float32 for r in rewards.values())
        assert all(bool(env.observation_space(a).contains(o)) for a, o in obs.items())
        assert bool(env.state_space().contains(env.get_state(state)))


def test_largest_supported_horizon_has_exact_scores_without_overflow():
    limit = 2**24
    env = SpatialBlotto(max_steps=int(limit))
    key = jax.random.PRNGKey(10)
    _, state = env.reset(key)
    state = state.replace(
        p_pos=state.p_pos.at[:3].set(env.zone_centers),
        step=jnp.array(limit - 1, dtype=jnp.int32),
        team_scores=jnp.array([3 * (limit - 1), 0], dtype=jnp.int32),
    )
    _, terminal, _, dones, _ = env.step_env(key, state, _still(env))
    np.testing.assert_array_equal(terminal.team_scores, [3 * limit, 0])
    assert int(terminal.step) == limit
    assert bool(dones["__all__"])
    # A stationary unit one step before the horizon must observe different time
    # from the terminal state, even at the largest supported task horizon.
    assert float(env.get_state(state)[-1]) < float(env.get_state(terminal)[-1])
    _, following, rewards, _, _ = env.step_env(key, terminal, _still(env))
    _assert_tree_equal(following, terminal)
    assert all(float(r) == 0 for r in rewards.values())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dt": float(2**63)},
        {"max_speed": float(2**63)},
        {"dt": float(np.finfo(np.float32).tiny)},
        {"max_speed": float(np.finfo(np.float32).tiny), "dt": 1.0},
    ],
)
def test_extreme_supported_numeric_parameters_produce_finite_bounded_states(kwargs):
    env = SpatialBlotto(team_size=1, **kwargs)
    key = jax.random.PRNGKey(14)
    _, state = env.reset(key)
    state = state.replace(p_pos=jnp.zeros_like(state.p_pos))
    obs, following, _, _, _ = env.step_env(
        key, state, {agent: jnp.array([1.0, 0]) for agent in env.agents}
    )
    assert all(np.isfinite(leaf).all() for leaf in jax.tree.leaves((obs, following)))
    assert all(
        bool(env.observation_space(agent).contains(value))
        for agent, value in obs.items()
    )
    assert bool(env.state_space().contains(env.get_state(following)))
    assert np.all(np.abs(following.p_pos) <= env.arena)
    np.testing.assert_allclose(
        following.p_pos[:, 0], min(env.dt * env.max_speed, env.arena), rtol=1e-6, atol=0
    )


def test_vmap_mixed_active_and_absorbing_states_are_independent():
    env = SpatialBlotto(max_steps=2)
    keys = jax.random.split(jax.random.PRNGKey(3), 3)
    _, states = jax.vmap(env.reset)(keys)
    states = states.replace(
        step=jnp.array([0, 1, 2], dtype=jnp.int32),
        done=jnp.array([False, False, True]),
        team_scores=jnp.array([[0, 0], [1, 2], [2, 4]], dtype=jnp.int32),
    )
    actions = {a: jnp.ones((3, 2)) for a in env.agents}
    _, following, rewards, dones, _ = jax.jit(jax.vmap(env.step_env))(
        keys, states, actions
    )
    np.testing.assert_array_equal(following.step, [1, 2, 2])
    np.testing.assert_array_equal(dones["__all__"], [False, True, True])
    for field in ("p_pos", "p_vel", "team_scores"):
        np.testing.assert_array_equal(
            getattr(following, field)[-1], getattr(states, field)[-1]
        )
    assert all(float(r[-1]) == 0 for r in rewards.values())


@pytest.mark.parametrize("team_size", [2, 3])
def test_jit_vmap_scan_and_generic_log_wrapper(team_size):
    env = SpatialBlotto(team_size=team_size, max_steps=3)
    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    _, states = jax.jit(jax.vmap(env.reset))(keys)
    actions = {a: jnp.zeros((4, 2)) for a in env.agents}

    def advance(current, key):
        _, following, _, done, _ = jax.vmap(env.step)(
            jax.random.split(key, 4), current, actions
        )
        return following, done["__all__"]

    final, dones = jax.jit(lambda s: jax.lax.scan(advance, s, keys))(states)
    np.testing.assert_array_equal(dones[:, 0], [False, False, True, False])
    np.testing.assert_array_equal(final.step, np.ones(4))

    wrapped = LogWrapper(env)
    _, wrapped_state = wrapped.reset(keys[0])
    for key in keys[:3]:
        _, wrapped_state, _, _, info = wrapped.step(key, wrapped_state, _still(env))
    np.testing.assert_array_equal(
        info["returned_episode_lengths"], np.full(env.num_agents, 3)
    )
    assert np.all(info["returned_episode"])


def test_log_wrapper_tracks_shared_returns_in_agent_order_under_jit():
    env = SpatialBlotto(max_steps=3, reward_mode="zero_sum")
    wrapper = LogWrapper(env)
    key = jax.random.PRNGKey(15)
    _, state = wrapper.reset(key)
    state = state.replace(
        env_state=state.env_state.replace(
            p_pos=env.zone_centers[jnp.array([0, 0, 1, 1, 1, 2])]
        )
    )
    for step in range(3):
        _, state, _, _, info = jax.jit(wrapper.step)(key, state, _still(env))
        np.testing.assert_array_equal(info["returned_episode"], np.full(6, step == 2))
    np.testing.assert_array_equal(
        info["returned_episode_returns"], [-3, -3, -3, 3, 3, 3]
    )
    np.testing.assert_array_equal(info["returned_episode_lengths"], np.full(6, 3))
    np.testing.assert_array_equal(state.episode_returns, np.zeros(6))
    assert int(state.env_state.step) == 0
    assert not bool(state.env_state.done)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"team_size": 0},
        {"team_size": True},
        {"team_size": -1},
        {"team_size": 1.5},
        {"team_size": "3"},
        {"max_steps": 0},
        {"max_steps": True},
        {"max_steps": 2.5},
        {"max_steps": 16777217},
        {"max_steps": "200"},
        {"dt": float("nan")},
        {"dt": 0},
        {"dt": -0.1},
        {"dt": float("inf")},
        {"dt": True},
        {"dt": "0.1"},
        {"dt": None},
        {"dt": 1e-50},
        {"dt": 1e38},
        {"max_speed": 0},
        {"max_speed": -1},
        {"max_speed": float("nan")},
        {"max_speed": float("inf")},
        {"max_speed": True},
        {"max_speed": 1e40},
        {"max_speed": 1e38},
        {"zone_radius": 0},
        {"zone_radius": -1},
        {"zone_radius": float("nan")},
        {"zone_radius": 0.7},
        {"zone_radius": math.sqrt(3) * 0.375},
        {"dt": 1e30, "max_speed": 1e30},
        {"dt": 1e10, "max_speed": 1e10},
        {"dt": 1e-30, "max_speed": 1e-30},
        {"reward_mode": "unknown"},
    ],
)
def test_invalid_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        SpatialBlotto(**kwargs)


def test_numpy_numeric_configuration_and_host_action_validation():
    env = SpatialBlotto(
        team_size=np.int64(2), max_steps=np.int32(3), dt=np.float32(0.1)
    )
    actions = {a: [2, -4] for a in env.agents}
    assert env.validate_actions(actions) is None
    assert isinstance(env.team_size, int)
    assert isinstance(env.max_steps, int)
    with pytest.raises(NotImplementedError, match="continuous Box"):
        env.get_avail_actions(None)


@pytest.mark.parametrize(
    "bad_value",
    [
        [np.nan, 0],
        [np.inf, 0],
        [-np.inf, 0],
        [1e100, 0],
        [1 + 2j, 0],
        [True, False],
        ["0", "1"],
        0,
        [0],
        [0, 0, 0],
        [[0, 0]],
    ],
)
def test_host_action_validation_rejects_nonfinite_or_malformed_actions(bad_value):
    env = SpatialBlotto(team_size=1)
    actions = _still(env)
    actions["red_0"] = bad_value
    with pytest.raises(ValueError, match="action for red_0"):
        env.validate_actions(actions)


@pytest.mark.parametrize(
    "kind",
    [
        "missing",
        "extra",
        "not_mapping",
        "scalar",
        "wide",
        "matrix",
        "complex",
        "boolean",
    ],
)
def test_compiled_transition_rejects_bad_action_structure(kind):
    env = SpatialBlotto(team_size=1)
    key = jax.random.PRNGKey(1)
    _, state = env.reset(key)
    actions = _still(env)
    if kind == "missing":
        del actions["red_0"]
    elif kind == "extra":
        actions["unknown"] = jnp.zeros(2)
    elif kind == "not_mapping":
        actions = jnp.zeros((2, 2))
    else:
        actions["red_0"] = {
            "scalar": jnp.array(0),
            "wide": jnp.zeros(3),
            "matrix": jnp.zeros((1, 2)),
            "complex": jnp.array([1 + 1j, 0]),
            "boolean": jnp.array([True, False]),
        }[kind]
    with pytest.raises(ValueError):
        env.step_env(key, state, actions)
    with pytest.raises(ValueError):
        env.validate_actions(actions)


@pytest.mark.parametrize(
    "field,value",
    [
        ("p_pos", jnp.zeros((5, 2))),
        ("p_vel", jnp.zeros((6, 3))),
        ("team_scores", jnp.zeros(3, dtype=jnp.int32)),
        ("step", jnp.zeros(1, dtype=jnp.int32)),
        ("done", jnp.zeros(1, dtype=jnp.bool_)),
    ],
)
def test_wrong_state_shapes_fail_before_broadcasting(field, value):
    env = SpatialBlotto()
    key = jax.random.PRNGKey(1)
    _, state = env.reset(key)
    state = state.replace(**{field: value})
    with pytest.raises(ValueError, match=f"state.{field} must have shape"):
        env.step_env(key, state, _still(env))
    with pytest.raises(ValueError, match=f"state.{field} must have shape"):
        env.get_obs(state)
    with pytest.raises(ValueError, match=f"state.{field} must have shape"):
        env.get_state(state)
