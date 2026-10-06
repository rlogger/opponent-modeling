"""Independent payoff, path, visibility, and transformation checks for baselines."""

from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("jaxmarl")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from spatial_blotto import SpatialBlotto  # noqa: E402
from spatial_blotto.allocations import (  # noqa: E402
    allocation_scores,
    enumerate_allocations,
    validate_allocation,
)
from spatial_blotto.controllers import make_controller  # noqa: E402


def _counts(assignments):
    return np.bincount(np.asarray(assignments), minlength=3)


def _score(own, other, mode):
    # Direct game definition, independent of the controller's scoring utility.
    owners = np.sign(np.asarray(own) - np.asarray(other))
    return int(np.sum(owners == 1) - (mode == "zero_sum") * np.sum(owners == -1))


@pytest.mark.parametrize("n", [1, 2, 3, 6])
def test_allocations_exhaust_budget_without_duplicates(n):
    actual = np.asarray(enumerate_allocations(n))
    expected = {row for row in product(range(n + 1), repeat=3) if sum(row) == n}
    assert {tuple(row) for row in actual} == expected
    assert len(actual) == len(expected) == (n + 1) * (n + 2) // 2
    assert np.issubdtype(actual.dtype, np.integer)


@pytest.mark.parametrize("allocation", [None, 3, (), (1, 2), (1, 1, 1, 0),
                                         (-1, 2, 2), (1.0, 1, 1), (True, 1, 1),
                                         (1, 0, 0), (2, 2, 0)])
def test_invalid_allocations_rejected(allocation):
    with pytest.raises(ValueError):
        validate_allocation(allocation, 3)


@pytest.mark.parametrize("budget", [0, -1, True, 2.0])
def test_invalid_budget_rejected(budget):
    with pytest.raises(ValueError):
        enumerate_allocations(budget)


@pytest.mark.parametrize("kwargs", [
    {"team": "green"}, {"policy": "magic"}, {"policy": "fixed"},
    {"allocation": (1, 1, 1)}, {"period": 0}, {"period": True}, {"period": 1.5},
])
def test_invalid_controller_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        make_controller(SpatialBlotto(), **({"team": "red"} | kwargs))


def test_exponential_search_limit_is_explicit():
    env = SpatialBlotto(team_size=7)
    with pytest.raises(ValueError, match="team_size <= 6"):
        make_controller(env, "red")


@pytest.mark.parametrize("invalid", [0, [0], [0, 0], [0, 0, 0, 0], np.zeros((2, 1))])
@pytest.mark.parametrize("side", ["allocations", "opponent_counts"])
def test_allocation_scores_require_three_zones_even_under_jit(invalid, side):
    args = {"allocations": jnp.array([1, 1, 1]), "opponent_counts": jnp.array([0, 2, 1])}
    args[side] = jnp.asarray(invalid)
    with pytest.raises(ValueError, match=f"{side}.*three zones"):
        jax.jit(allocation_scores)(**args)


def test_allocation_scores_broadcast_only_leading_dimensions():
    own = jnp.array([[[2, 1, 0]], [[1, 1, 1]]])
    opponents = jnp.array([[0, 0, 0], [2, 0, 1]])
    actual = jax.jit(allocation_scores)(own, opponents)
    np.testing.assert_array_equal(actual, [[2, 1], [3, 1]])
    assert actual.shape == (2, 2)


def test_cyclic_period_fits_int32_counter_and_largest_value_is_jittable():
    env = SpatialBlotto(team_size=1)
    for period in (2**31, 2**63, np.uint64(2**63)):
        with pytest.raises(ValueError, match="int32"):
            make_controller(env, "red", "cyclic", period=period)
    controller = make_controller(env, "red", "cyclic", period=np.int64(2**31 - 1))
    _, state = env.reset(jax.random.PRNGKey(0))
    select = jax.jit(controller.target_zones)
    for step, expected in [(2**31 - 2, 0), (2**31 - 1, 1)]:
        np.testing.assert_array_equal(select(state.replace(step=jnp.int32(step))), [expected])


def test_tiny_representable_distance_produces_bounded_direction():
    env = SpatialBlotto(team_size=1, dt=1e-30)
    controller = make_controller(env, "red", "fixed", allocation=(1, 0, 0))
    _, state = env.reset(jax.random.PRNGKey(0))
    state = state.replace(p_pos=state.p_pos.at[0].set(jnp.array([1e-20, 0.75])))
    # Squaring 1e-20 underflows on float32 JAX backends, although the distance
    # itself is normal and the unit must head left at the speed limit.
    actual = jax.jit(controller)(state)
    np.testing.assert_array_equal(actual, [[-1, 0]])
    assert np.isfinite(actual).all()


@pytest.mark.parametrize("n", [2, 3])
@pytest.mark.parametrize("mode", ["ownership", "zero_sum"])
@pytest.mark.parametrize("team", ["red", "blue"])
def test_reactive_matches_independent_exhaustive_static_oracle(n, mode, team):
    env = SpatialBlotto(team_size=n, reward_mode=mode)
    _, initial = env.reset(jax.random.PRNGKey(17))
    controller = make_controller(env, team, "reactive")
    select = jax.jit(controller.target_zones)
    centers = np.asarray(env.zone_centers)
    own_index = int(team == "blue")
    own_slice = slice(own_index * n, (own_index + 1) * n)
    opponent_slice = slice((1 - own_index) * n, (2 - own_index) * n)
    own_positions = np.asarray(initial.p_pos[own_slice])
    possible_allocations = [row for row in product(range(n + 1), repeat=3) if sum(row) == n]
    # Include partial budgets because opponents may currently be in transit.
    for opponent in product(range(n + 1), repeat=3):
        if sum(opponent) > n:
            continue
        opponent_positions = [centers[z] for z, count in enumerate(opponent) for _ in range(count)]
        opponent_positions += [np.array([env.arena, env.arena])] * (n - sum(opponent))
        state = initial.replace(
            p_pos=initial.p_pos.at[opponent_slice].set(jnp.asarray(opponent_positions))
        )
        selected = np.asarray(select(state))
        optimal_score = max(_score(row, opponent, mode) for row in possible_allocations)
        assert _score(_counts(selected), opponent, mode) == optimal_score
        # Independently enumerate physical unit assignments to certify travel tie-break.
        costs = [
            sum(np.linalg.norm(centers[z] - own_positions[i]) for i, z in enumerate(row))
            for row in product(range(3), repeat=n)
            if _score(_counts(row), opponent, mode) == optimal_score
        ]
        selected_cost = np.linalg.norm(centers[selected] - own_positions, axis=-1).sum()
        assert selected_cost == pytest.approx(min(costs), abs=1e-6)
        # Real environment scoring at the hypothetical destinations agrees too.
        arrived = state.replace(p_pos=state.p_pos.at[own_slice].set(env.zone_centers[selected]))
        actions = {a: jnp.zeros(2) for a in env.agents}
        _, _, _, _, info = env.step_env(jax.random.PRNGKey(0), arrived, actions)
        assert int(info["team_rewards"][own_index]) == optimal_score


def test_ownership_and_difference_objectives_change_the_response():
    allocations = jnp.array([[0, 1, 2], [1, 1, 1]])
    np.testing.assert_array_equal(allocation_scores(allocations, [1, 0, 0]), [2, 2])
    np.testing.assert_array_equal(allocation_scores(allocations, [1, 0, 0], "zero_sum"), [1, 2])
    for mode, expected in [("ownership", [0, 1, 2]), ("zero_sum", [1, 1, 1])]:
        env = SpatialBlotto(reward_mode=mode)
        _, state = env.reset(jax.random.PRNGKey(0))
        state = state.replace(p_pos=jnp.concatenate((
            env.zone_centers[jnp.array([1, 2, 2])],
            env.zone_centers[:1], jnp.full((2, 2), env.arena),
        )))
        np.testing.assert_array_equal(
            _counts(make_controller(env, "red", "reactive").target_zones(state)), expected
        )
    with pytest.raises(ValueError):
        allocation_scores(allocations, [1, 0, 0], "unknown")


def test_static_three_unit_pure_equilibria_follow_full_best_response_enumeration():
    # The claim concerns only simultaneous static allocations, not moving agents.
    allocations = [row for row in product(range(4), repeat=3) if sum(row) == 3]
    concentrated = {(3, 0, 0), (0, 3, 0), (0, 0, 3)}
    balanced = (1, 1, 1)
    for mode in ["ownership", "zero_sum"]:
        equilibria = {
            (red, blue) for red in allocations for blue in allocations
            if _score(red, blue, mode) == max(_score(a, blue, mode) for a in allocations)
            and _score(blue, red, mode) == max(_score(a, red, mode) for a in allocations)
        }
        expected = {(balanced, balanced)} if mode == "zero_sum" else (
            {(allocation, balanced) for allocation in concentrated}
            | {(balanced, allocation) for allocation in concentrated}
        )
        assert equilibria == expected


@pytest.mark.parametrize("n", [2, 3])
@pytest.mark.parametrize("mode", ["ownership", "zero_sum"])
def test_reactive_converges_to_static_optimum_against_stationary_opponents(n, mode):
    env = SpatialBlotto(team_size=n, reward_mode=mode, max_steps=50)
    _, initial = env.reset(jax.random.PRNGKey(10))
    opponent = (n - 1, 1, 0)
    opponent_zones = jnp.array([0] * (n - 1) + [1])
    initial = initial.replace(p_pos=initial.p_pos.at[n:].set(env.zone_centers[opponent_zones]))
    controller = make_controller(env, "red", "reactive")

    def advance(state, _):
        actions = controller(state)
        joint = {a: actions[i] for i, a in enumerate(env.red_agents)}
        joint.update({a: jnp.zeros(2) for a in env.blue_agents})
        _, following, _, _, info = env.step_env(jax.random.PRNGKey(0), state, joint)
        return following, (actions, info["team_rewards"][0])

    final, (actions, rewards) = jax.jit(
        lambda state: jax.lax.scan(advance, state, None, length=40)
    )(initial)
    allocations = [row for row in product(range(n + 1), repeat=3) if sum(row) == n]
    optimum = max(_score(row, opponent, mode) for row in allocations)
    np.testing.assert_array_equal(rewards[-10:], np.full(10, optimum))
    np.testing.assert_array_equal(actions[-10:], np.zeros((10, n, 2)))
    np.testing.assert_array_equal(
        final.p_pos[:n], env.zone_centers[controller.target_zones(final)]
    )


@pytest.mark.parametrize("n", [2, 3])
def test_fixed_routes_are_shortest_bounded_and_converge_without_overshoot(n):
    env = SpatialBlotto(team_size=n, max_steps=80, dt=0.17, max_speed=0.8)
    _, state = env.reset(jax.random.PRNGKey(11))
    allocation = (n - 1, 1, 0)
    controller = make_controller(env, "red", "fixed", allocation=allocation)
    selected = np.asarray(controller.target_zones(state))
    targets = np.asarray(env.zone_centers[selected])
    initial = np.asarray(state.p_pos[:n])
    for _ in range(35):
        current = np.asarray(state.p_pos[:n])
        actions = controller(state)
        assert np.all(np.linalg.norm(actions, axis=-1) <= 1 + 1e-6)
        np.testing.assert_array_equal(controller.target_zones(state), selected)
        both = {a: actions[i] for i, a in enumerate(env.red_agents)}
        both.update({a: jnp.zeros(2) for a in env.blue_agents})
        _, state, _, _, _ = env.step_env(jax.random.PRNGKey(0), state, both)
        remaining = np.linalg.norm(targets - current, axis=-1)
        new_remaining = np.linalg.norm(targets - np.asarray(state.p_pos[:n]), axis=-1)
        np.testing.assert_allclose(
            new_remaining, np.maximum(remaining - env.dt * env.max_speed, 0), atol=3e-7
        )
        # Each complete trajectory stays on its original straight segment.
        moved = np.asarray(state.p_pos[:n]) - initial
        path = targets - initial
        cross = moved[:, 0] * path[:, 1] - moved[:, 1] * path[:, 0]
        np.testing.assert_allclose(cross, 0, atol=4e-7)
    np.testing.assert_allclose(state.p_pos[:n], targets, atol=1e-7)
    np.testing.assert_array_equal(controller(state), np.zeros((n, 2)))
    np.testing.assert_array_equal(env.zone_counts(state)[0], allocation)
    np.testing.assert_array_equal(controller(state.replace(done=jnp.array(True))), np.zeros((n, 2)))


@pytest.mark.parametrize("n", [2, 3])
@pytest.mark.parametrize("policy", ["balanced", "cyclic", "reactive"])
def test_seeded_mirrors_and_cyclic_schedule(n, policy):
    env = SpatialBlotto(team_size=n)
    red = make_controller(env, "red", policy, period=7)
    blue = make_controller(env, "blue", policy, period=7)
    mirror_zones = np.array([0, 2, 1])
    for seed in [0, 19]:
        _, state = env.reset(jax.random.PRNGKey(seed))
        for step in [0, 6, 7, 14, 21]:
            state = state.replace(step=jnp.array(step))
            np.testing.assert_array_equal(
                blue.target_zones(state), mirror_zones[np.asarray(red.target_zones(state))]
            )
            np.testing.assert_allclose(blue(state), red(state) * np.array([-1, 1]), atol=1e-7)
            if policy == "cyclic":
                np.testing.assert_array_equal(
                    _counts(red.target_zones(state)), np.roll((n - 1, 1, 0), step // 7)
                )
            elif policy == "balanced":
                counts = _counts(red.target_zones(state))
                assert max(counts) - min(counts) <= 1


def test_reactive_uses_current_physical_observation_only():
    env = SpatialBlotto()
    _, state = env.reset(jax.random.PRNGKey(4))
    controller = make_controller(env, "red", "reactive")
    expected = controller(state)
    changed_logs = state.replace(
        p_vel=jnp.full_like(state.p_vel, 0.9), team_scores=jnp.array([100, 200]),
        step=jnp.array(77),
    )
    np.testing.assert_array_equal(controller(changed_logs), expected)
    # A minimal view has no velocity, scores, goals, future actions, or policy state.
    observation_only = SimpleNamespace(p_pos=state.p_pos, done=state.done)
    np.testing.assert_array_equal(controller(observation_only), expected)


@pytest.mark.parametrize("n", [2, 3])
@pytest.mark.parametrize("policy", ["fixed", "balanced", "cyclic", "reactive"])
def test_controllers_jit_vmap_and_scan(n, policy):
    env = SpatialBlotto(team_size=n, max_steps=5)
    options = {"allocation": (n - 1, 1, 0)} if policy == "fixed" else {}
    controller = make_controller(env, "red", policy, period=2, **options)
    keys = jax.random.split(jax.random.PRNGKey(8), 3)
    _, states = jax.vmap(env.reset)(keys)
    expected = jnp.stack([controller(jax.tree.map(lambda x: x[i], states)) for i in range(3)])
    actual = jax.jit(jax.vmap(controller))(states)
    np.testing.assert_allclose(actual, expected, atol=1e-7)

    def advance(current, _):
        actions = controller(current)
        joint = {a: actions[i] for i, a in enumerate(env.red_agents)}
        joint.update({a: jnp.zeros(2) for a in env.blue_agents})
        _, following, _, _, _ = env.step_env(keys[0], current, joint)
        return following, actions

    initial = jax.tree.map(lambda x: x[0], states)
    final, actions = jax.jit(lambda s: jax.lax.scan(advance, s, None, length=7))(initial)
    assert final.step == 5
    assert actions.shape == (7, n, 2)
    np.testing.assert_array_equal(actions[5:], np.zeros((2, n, 2)))
