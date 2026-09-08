"""Exact main-source contract plus original marl-opp-aware numerical goldens.

Source hashes are from opponent-modeling@8c24db3. Numerical goldens were obtained
from marl-opp-aware@aecbab5, whose mechanics match main under the locked runtime.
Neither check needs a sibling checkout. See third_party/marl-opp-aware.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("jaxmarl")

from mopa.continuous_data import markov_state  # noqa: E402
from mopa.zero_s import zero_s_features  # noqa: E402
from tag_objectives import SimpleTagObjectivesMPE, to_mpe_action  # noqa: E402


@pytest.mark.parametrize("filename,expected", [
    ("objectives.py",
     "71332590fc61490bfd6271ab72da01f41df952963c10d86b577035ac74796f84"),
    ("resources.py",
     "72910059f63c01613b4ad3803236497723724e3c16525d9b21ca51b920b0d4d4"),
])
def test_environment_source_is_byte_identical_to_main_8c24db3(filename, expected):
    source = Path(__file__).resolve().parents[1] / "src" / "tag_objectives" / filename
    assert hashlib.sha256(source.read_bytes()).hexdigest() == expected


# Seed 7 from original aecbab5, JAX 0.4.38 / JaxMARL 0.1.0, CPU float32.
_RESET_POSITION = [[-0.6217032075, 1.3794082403], [0.3454952240, -1.1978396177]]
_RED_OBSERVATION = [
    0., 0., -0.6217032075, 1.3794082403, 0.9671984315, -2.5772478580, 0., 0.,
    -0.2877776027, -1.3285959959, 0.5685236454,
    1.4536459446, 0.1017869711, 0.4115721583,
    2.3834028244, -2.8793802261, 0.5398736000,
]
_BLUE_OBSERVATION = [
    0., 0., 0.3454952240, -1.1978396177, -0.9671984315, 2.5772478580,
    -1.1749749184, 0.2164343596, 0.8482060432, 1.0552576780,
    1.1997420788, -0.6298596859, -1.1114547253, 1.0837388039,
    -0.8949157596, 1.5284360647, -1.8166180849, 0.6955589056,
    -2.0597839355, 0.1410368681, -1.6930646896, 1.4060286283,
    -1.7749364376, 1.3753862381, 1.5016751289, 1.8601130247,
    1.4162044525, -0.3021322489, 0.5398736000,
    -1.2549760342, 1.2486518621, 0.5685236454,
    0.4864474535, 2.6790347099, 0.4115721583,
]


@pytest.mark.parametrize("action_type", ["Discrete", "Continuous"])
def test_original_reset_observation_order_and_checkpoint_dimensions(action_type):
    env = SimpleTagObjectivesMPE(action_type=action_type)
    obs, state = env.reset(jax.random.PRNGKey(7))
    assert (env.num_adversaries, env.num_good_agents, env.num_landmarks) == (1, 1, 0)
    assert (env.arena, env.num_resources, env.num_lava, env.max_steps) == (2., 16, 3, 100)
    assert (env.collect_radius, env.collect_reward, env.min_start_dist) == (0.15, 5., 2.2)
    assert (env.lava_penalty, env.base_lava_penalty, env.prey_lava_penalty) == (100., 0., 0.)
    assert (env.m_nearest_resources, env.n_nearest_lava, env.grid_size) == (10, 3, 16)
    assert (env.lava_radius_min, env.lava_radius_max, env.frac_resources_near_lava) == (0.35, 0.60, 0.5)
    assert (env.novelty_bonus, env.capture_bonus, env.dense_chase_coef) == (0.5, 10., 0.)
    assert env.fixed_obstacle_positions is None and (env.dt, env.damping) == (0.1, 0.25)
    np.testing.assert_allclose(env.max_speed, [1.4, 1.3], atol=1e-7)
    np.testing.assert_allclose(env.rad, [0.075, 0.05], atol=1e-7)
    np.testing.assert_array_equal(env.accel, [3., 4.])
    np.testing.assert_allclose(state.p_pos, _RESET_POSITION, atol=2e-6)
    np.testing.assert_allclose(obs["adversary_0"], _RED_OBSERVATION, atol=2e-6)
    np.testing.assert_allclose(obs["agent_0"], _BLUE_OBSERVATION, atol=2e-6)
    assert obs["adversary_0"].shape == (17,) and obs["agent_0"].shape == (35,)
    assert not np.asarray(state.collected).any() and int(state.capture_t) == -1
    encoded = markov_state(env, state)
    assert encoded.shape == (66,)
    np.testing.assert_allclose(encoded[:4], np.asarray(_RESET_POSITION).ravel(), atol=2e-6)
    np.testing.assert_allclose(
        zero_s_features(encoded),
        [0.3454952240, -1.1978396177, -0.6217032075, 1.3794082403, 0., 0., 0., 0.],
        atol=2e-6,
    )


@pytest.mark.parametrize("objective,pred_reward", [
    ("capture", -1.5790890455), ("risk", 0.), ("curious", 0.),
])
def test_original_twelve_step_continuous_golden(objective, pred_reward):
    env = SimpleTagObjectivesMPE(pred_type=objective, action_type="Continuous")
    _, state = env.reset(jax.random.PRNGKey(7))
    actions = {
        "adversary_0": to_mpe_action(jnp.array([0.75, -0.25])),
        "agent_0": to_mpe_action(jnp.array([-0.4, 0.6])),
    }
    for step in range(12):
        obs, state, reward, done, info = env.step_env(
            jax.random.PRNGKey(100 + step), state, actions
        )
    np.testing.assert_allclose(
        state.p_pos, [[0.1097003296, 1.1356071234], [-0.1746139675, -0.4176757634]],
        atol=2e-6,
    )
    np.testing.assert_allclose(
        state.p_vel, [[0.8714913130, -0.2904970646], [-0.6197271943, 0.9295907021]],
        atol=2e-6,
    )
    np.testing.assert_allclose(obs["adversary_0"][4:6], [-0.2843143046, -1.5532828569], atol=2e-6)
    np.testing.assert_allclose(obs["agent_0"][4:6], [0.2843143046, 1.5532828569], atol=2e-6)
    np.testing.assert_allclose([reward["adversary_0"], reward["agent_0"]], [pred_reward, 0.], atol=2e-6)
    assert int(state.step) == 12 and not bool(done["__all__"])
    assert int(state.capture_t) == -1 and not np.asarray(state.collected).any()
    np.testing.assert_array_equal(info["captured"], [0., 0.])


def test_two_dimensional_adapter_preserves_original_continuous_forces():
    env = SimpleTagObjectivesMPE(action_type="Continuous")
    for force in ([0., 0.], [-1., 0.], [1., 0.], [0., -1.], [0., 1.], [0.35, -0.8]):
        mapped = to_mpe_action(jnp.array(force))
        decoded, _ = env.action_decoder(env.agent_range, jnp.stack([mapped, mapped]))
        np.testing.assert_allclose(decoded, np.array(force)[None] * np.array([3., 4.])[:, None], atol=1e-6)
        assert mapped.shape == (5,) and np.all((np.asarray(mapped) >= 0) & (np.asarray(mapped) <= 1))


def test_original_boundary_is_a_penalty_not_a_new_physical_wall():
    env = SimpleTagObjectivesMPE(action_type="Continuous")
    _, state = env.reset(jax.random.PRNGKey(7))
    state = state.replace(
        p_pos=jnp.array([[-1.2, 0.], [2.4, 0.]]),
        p_vel=jnp.array([[0., 0.], [1., 0.]]),
        resource_pos=jnp.full_like(state.resource_pos, -1.8),
    )
    actions = {agent: to_mpe_action(jnp.zeros(2)) for agent in env.agents}
    _, after, rewards, done, _ = env.step_env(jax.random.PRNGKey(99), state, actions)
    np.testing.assert_allclose(after.p_pos[1], [2.5, 0.], atol=1e-6)
    np.testing.assert_allclose(after.p_vel[1], [0.75, 0.], atol=1e-6)
    np.testing.assert_allclose(rewards["agent_0"], -np.exp(0.5), atol=1e-6)
    assert not bool(done["__all__"])
