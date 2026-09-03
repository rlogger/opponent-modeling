import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("jaxmarl")

from tag_objectives import (  # noqa: E402
    ObjectiveSpec,
    SimpleTagObjectivesMPE,
    evaluate_policy,
    from_mpe_action,
    joint_action_dict,
    make_env,
    random_policy,
    register_objective,
    to_mpe_action,
)


def _noop(env):
    return {a: jnp.int32(0) for a in env.agents}


def _quiet_state(env, state):
    return state.replace(
        p_pos=state.p_pos.at[0].set(jnp.array([0.0, 0.0])).at[1].set(jnp.array([1.2, 0.0])),
        p_vel=jnp.zeros_like(state.p_vel),
        resource_pos=jnp.full_like(state.resource_pos, 1.8),
        collected=jnp.zeros_like(state.collected),
        lava_pos=jnp.array([[0.0, 0.0], [1.5, 1.5], [-1.5, 1.5]], dtype=jnp.float32),
        lava_rad=jnp.array([0.4, 0.3, 0.3], dtype=jnp.float32),
        visited=jnp.zeros_like(state.visited),
    )


def test_objective_obs_shapes_are_static():
    env = SimpleTagObjectivesMPE()
    obs, _ = env.reset(jax.random.PRNGKey(0))

    assert obs[env.adversaries[0]].shape == env.observation_space(env.adversaries[0]).shape
    assert obs[env.good_agents[0]].shape == env.observation_space(env.good_agents[0]).shape
    assert obs[env.adversaries[0]].shape == (17,)
    assert obs[env.good_agents[0]].shape == (35,)


def test_capture_sets_done_and_records_first_capture_step():
    env = SimpleTagObjectivesMPE()
    _, state = env.reset(jax.random.PRNGKey(0))
    p_pos = state.p_pos.at[0].set(jnp.array([0.0, 0.0]))
    p_pos = p_pos.at[1].set(jnp.array([0.05, 0.0]))
    state = state.replace(p_pos=p_pos, p_vel=jnp.zeros_like(state.p_vel))

    _, state2, _, dones, info = env.step_env(jax.random.PRNGKey(1), state, _noop(env))

    assert bool(dones["__all__"])
    assert int(state2.capture_t) == 1
    assert float(info["captured"][0]) == 1.0


def test_lava_penalizes_every_predator_type_risk_strongest():
    kwargs = dict(lava_penalty=5.0, base_lava_penalty=1.0)
    risk = SimpleTagObjectivesMPE(pred_type="risk", **kwargs)
    capture = SimpleTagObjectivesMPE(pred_type="capture", **kwargs)
    _, state = risk.reset(jax.random.PRNGKey(0))
    state = _quiet_state(risk, state)
    out_state = state.replace(p_pos=state.p_pos.at[0].set(jnp.array([2.4, 0.0])))

    _, _, risk_reward, _, info = risk.step_env(jax.random.PRNGKey(1), state, _noop(risk))
    _, _, capture_in, _, _ = capture.step_env(jax.random.PRNGKey(1), state, _noop(capture))
    _, _, capture_out, _, _ = capture.step_env(jax.random.PRNGKey(1), out_state, _noop(capture))

    pred = risk.adversaries[0]
    assert float(info["pred_lava"][0]) == 1.0
    assert np.isclose(float(risk_reward[pred]), -5.0, atol=1e-4)
    assert np.isclose(float(capture_in[pred]) - float(capture_out[pred]), -1.0, atol=1e-4)
    assert float(capture_in[pred]) > float(risk_reward[pred])


def test_prey_pays_lava_penalty_per_step():
    env = SimpleTagObjectivesMPE(pred_type="risk", prey_lava_penalty=1.0)
    _, state = env.reset(jax.random.PRNGKey(0))
    state = _quiet_state(env, state)
    prey_idx = env.num_adversaries
    state = state.replace(p_pos=state.p_pos.at[0].set(jnp.array([-1.2, 0.0])))

    in_lava = state.replace(p_pos=state.p_pos.at[prey_idx].set(jnp.array([0.0, 0.0])))
    out_lava = state.replace(p_pos=state.p_pos.at[prey_idx].set(jnp.array([0.8, 0.0])))
    _, _, r_in, _, info_in = env.step_env(jax.random.PRNGKey(1), in_lava, _noop(env))
    _, _, r_out, _, info_out = env.step_env(jax.random.PRNGKey(1), out_lava, _noop(env))

    prey = env.good_agents[0]
    assert float(info_in["prey_lava"][prey_idx]) == 1.0
    assert float(info_out["prey_lava"][prey_idx]) == 0.0
    assert np.isclose(float(r_in[prey]) - float(r_out[prey]), -1.0, atol=1e-4)


def test_curious_novelty_bonus_fires_once_per_grid_cell():
    env = SimpleTagObjectivesMPE(pred_type="curious", novelty_bonus=0.5)
    _, state = env.reset(jax.random.PRNGKey(0))
    state = _quiet_state(env, state)

    _, state2, r1, _, info1 = env.step_env(jax.random.PRNGKey(1), state, _noop(env))
    _, _, r2, _, info2 = env.step_env(jax.random.PRNGKey(2), state2, _noop(env))

    pred = env.adversaries[0]
    assert float(info1["pred_new_cell"][0]) == 1.0
    assert float(info2["pred_new_cell"][0]) == 0.0
    assert np.isclose(float(r1[pred]) - float(r2[pred]), 0.5, atol=1e-4)


def test_team_novelty_credits_only_first_predator_in_shared_cell():
    env = SimpleTagObjectivesMPE(
        pred_type=("curious", "curious", "curious"),
        num_adversaries=3,
        novelty_bonus=0.5,
    )
    _, st = env.reset(jax.random.PRNGKey(0))
    cell_pos = jnp.array([0.0, 0.0])
    prey_idx = env.num_adversaries
    st = st.replace(
        p_pos=(
            st.p_pos.at[0]
            .set(cell_pos)
            .at[1]
            .set(cell_pos)
            .at[2]
            .set(jnp.array([0.8, 0.8]))
            .at[prey_idx]
            .set(jnp.array([1.2, 0.0]))
        ),
        p_vel=jnp.zeros_like(st.p_vel),
        visited=jnp.zeros_like(st.visited),
        resource_pos=jnp.full_like(st.resource_pos, 1.8),
    )
    _, _, _, _, info = env.step_env(jax.random.PRNGKey(1), st, _noop(env))
    assert float(info["pred_new_cell"][0]) == 1.0
    assert float(info["pred_new_cell"][1]) == 0.0
    assert float(info["pred_new_cell"][2]) == 1.0


def test_custom_info_is_agent_shaped_for_trainer_logging():
    env = SimpleTagObjectivesMPE()
    _, state = env.reset(jax.random.PRNGKey(0))
    _, _, _, _, info = env.step_env(jax.random.PRNGKey(1), state, _noop(env))

    for key in (
        "resources_collected",
        "new_resources_collected",
        "pred_lava",
        "prey_lava",
        "captured",
        "capture_t",
        "pred_new_cell",
        "pred_coverage",
    ):
        assert info[key].shape == (env.num_agents,), key


def test_resources_cluster_near_lava_at_reset():
    env = SimpleTagObjectivesMPE()
    _, state = env.reset(jax.random.PRNGKey(3))
    d = jnp.linalg.norm(
        state.resource_pos[:, None, :] - state.lava_pos[None, :, :], axis=-1
    )
    near = jnp.any(d < 1.5 * state.lava_rad[None, :], axis=-1)
    assert int(near.sum()) >= env.n_near_lava_resources - 2


def test_near_lava_collected_info_counts_only_near_lava_resources():
    env = SimpleTagObjectivesMPE()
    _, state = env.reset(jax.random.PRNGKey(0))
    state = _quiet_state(env, state)
    state = state.replace(
        resource_pos=state.resource_pos.at[0]
        .set(jnp.array([0.2, 0.0]))
        .at[1]
        .set(jnp.array([1.8, -1.8])),
        collected=state.collected.at[0].set(True).at[1].set(True),
    )
    _, _, _, _, info = env.step_env(jax.random.PRNGKey(1), state, _noop(env))
    prey_idx = env.num_adversaries
    assert float(info["near_lava_collected"][prey_idx]) == 1.0


def _contrived(env, st):
    return st.replace(
        p_pos=st.p_pos.at[0].set(jnp.array([0.0, 0.0])).at[1].set(jnp.array([0.9, 0.0])),
        p_vel=jnp.zeros_like(st.p_vel),
        lava_pos=jnp.array([[0.05, 0.0], [1.5, 1.5], [-1.5, -1.5]], dtype=jnp.float32),
        lava_rad=jnp.array([0.4, 0.35, 0.35], dtype=jnp.float32),
        visited=jnp.zeros_like(st.visited),
    )


def test_golden_rewards_are_stable_across_refactors():
    golds = {"capture": -1.9, "risk": -5.0, "curious": -0.5}
    for t, gold in golds.items():
        env = SimpleTagObjectivesMPE(
            pred_type=t, base_lava_penalty=1.0, lava_penalty=5.0
        )
        _, st = env.reset(jax.random.PRNGKey(0))
        st = _contrived(env, st)
        acts = {a: jnp.int32(0) for a in env.agents}
        _, _, rew, _, _ = env.step_env(jax.random.PRNGKey(1), st, acts)
        assert abs(float(rew[env.adversaries[0]]) - gold) < 1e-5, t


def test_registered_custom_objective_builds_and_rewards():
    register_objective(
        "test_ambusher",
        lambda env: ObjectiveSpec(
            capture=env.capture_bonus, lava=-env.base_lava_penalty, still=0.05
        ),
    )
    env = SimpleTagObjectivesMPE(pred_type="test_ambusher", base_lava_penalty=1.0)
    _, st = env.reset(jax.random.PRNGKey(0))
    st = _contrived(env, st)
    _, _, rew, _, _ = env.step_env(
        jax.random.PRNGKey(1), st, {a: jnp.int32(0) for a in env.agents}
    )
    assert abs(float(rew[env.adversaries[0]]) - (-1.0)) < 1e-5


def test_multi_predator_shapes_and_any_capture_termination():
    env = SimpleTagObjectivesMPE(pred_type="curious", num_adversaries=3)
    obs, st = env.reset(jax.random.PRNGKey(2))
    for a in env.agents:
        assert obs[a].shape == env.observation_space(a).shape
    prey_idx = env.num_adversaries
    st = st.replace(p_pos=st.p_pos.at[2].set(st.p_pos[prey_idx]))
    _, st2, rew, dones, info = env.step_env(
        jax.random.PRNGKey(3), st, {a: jnp.int32(0) for a in env.agents}
    )
    assert bool(dones["__all__"])
    assert int(st2.capture_t) >= 0
    assert len([a for a in rew if a.startswith("adversary")]) == 3


def test_reset_and_step_are_deterministic_given_key():
    env = SimpleTagObjectivesMPE(pred_type="risk")
    o1, s1 = env.reset(jax.random.PRNGKey(9))
    o2, s2 = env.reset(jax.random.PRNGKey(9))
    assert bool(jnp.all(s1.p_pos == s2.p_pos) & jnp.all(s1.lava_pos == s2.lava_pos))
    acts = {a: jnp.int32(2) for a in env.agents}
    _, n1, r1, _, _ = env.step_env(jax.random.PRNGKey(4), s1, acts)
    _, n2, r2, _, _ = env.step_env(jax.random.PRNGKey(4), s2, acts)
    assert bool(jnp.all(n1.p_pos == n2.p_pos))
    assert all(float(r1[a]) == float(r2[a]) for a in env.agents)


def test_evaluate_policy_facade_returns_standard_metrics():
    env = make_env("capture", num_adversaries=2)
    m = evaluate_policy(
        env, random_policy(env), n_eps=4, key=jax.random.PRNGKey(0), num_steps=5
    )
    expected = {
        "capture_rate",
        "survival_time",
        "resources_collected",
        "near_lava_collected",
        "pred_lava_steps",
        "prey_lava_steps",
        "pred_coverage",
    }
    assert set(m) == expected
    assert all(v.shape == (4,) for v in m.values())


def test_mixed_team_each_predator_gets_its_own_objective():
    env = SimpleTagObjectivesMPE(
        pred_type=("capture", "risk", "curious"),
        num_adversaries=3,
        base_lava_penalty=1.0,
        lava_penalty=5.0,
    )
    _, st = env.reset(jax.random.PRNGKey(0))
    prey_idx = env.num_adversaries
    pp = (
        st.p_pos.at[0]
        .set(jnp.array([0.0, 0.0]))
        .at[1]
        .set(jnp.array([0.3, 0.0]))
        .at[2]
        .set(jnp.array([0.0, 0.3]))
        .at[prey_idx]
        .set(jnp.array([0.9, 0.0]))
    )
    st = st.replace(
        p_pos=pp,
        p_vel=jnp.zeros_like(st.p_vel),
        lava_pos=jnp.array([[0.05, 0.0], [1.5, 1.5], [-1.5, -1.5]], dtype=jnp.float32),
        lava_rad=jnp.array([0.4, 0.35, 0.35], dtype=jnp.float32),
        visited=jnp.zeros_like(st.visited),
    )
    _, _, rew, _, _ = env.step_env(
        jax.random.PRNGKey(1), st, {a: jnp.int32(0) for a in env.agents}
    )
    got = {t: float(rew[a]) for t, a in zip(env.pred_types, env.adversaries)}
    assert abs(got["capture"] - (-1.9)) < 0.05
    assert abs(got["risk"] - (-5.0)) < 1e-3
    assert abs(got["curious"] - (-0.5)) < 1e-3


def test_prey_is_not_penalized_for_lava_by_default():
    env = SimpleTagObjectivesMPE(pred_type="risk")
    assert env.prey_lava_penalty == 0.0
    _, st = env.reset(jax.random.PRNGKey(0))
    st = _contrived(env, st)
    prey_idx = env.num_adversaries
    st = st.replace(
        p_pos=st.p_pos.at[prey_idx].set(jnp.array([1.5, 1.5])),
        resource_pos=jnp.full_like(st.resource_pos, -1.9),
    )
    _, _, rew, _, _ = env.step_env(
        jax.random.PRNGKey(1), st, {a: jnp.int32(0) for a in env.agents}
    )
    assert abs(float(rew[env.good_agents[0]])) < 1e-4


def test_mixed_team_rejects_wrong_length():
    with pytest.raises(ValueError):
        SimpleTagObjectivesMPE(pred_type=("capture", "risk"), num_adversaries=3)


# --------------------------------------------------------------------------- #
# Continuous action adapter (Gate 1 boundary contract)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "action, expected",
    [
        ([0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]),
        ([1.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0]),
        ([-1.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]),
        ([0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 1.0]),
        ([0.0, -1.0], [0.0, 0.0, 0.0, 1.0, 0.0]),
        ([0.5, -0.25], [0.0, 0.0, 0.5, 0.25, 0.0]),
        ([-0.7, 0.7], [0.0, 0.7, 0.0, 0.0, 0.7]),
        ([1.0, 1.0], [0.0, 0.0, 1.0, 0.0, 1.0]),
        ([-1.0, -1.0], [0.0, 1.0, 0.0, 1.0, 0.0]),
    ],
)
def test_to_mpe_action_zero_axes_diagonals_and_bounds(action, expected):
    out = np.asarray(to_mpe_action(jnp.asarray(action)))
    np.testing.assert_allclose(out, np.asarray(expected, dtype=np.float32), atol=1e-7)
    assert out.shape == (5,)
    assert (out >= 0.0).all() and (out <= 1.0).all()
    np.testing.assert_allclose(np.asarray(from_mpe_action(out)), action, atol=1e-7)


def test_adapter_roundtrip_on_batches_under_jit_and_vmap():
    rng = np.random.default_rng(0)
    a = rng.uniform(-1.0, 1.0, size=(6, 4, 2)).astype(np.float32)
    out = np.asarray(to_mpe_action(jnp.asarray(a)))
    assert out.shape == (6, 4, 5)
    np.testing.assert_array_equal(out[..., 0], 0.0)
    assert (out >= 0.0).all() and (out <= 1.0).all()
    np.testing.assert_allclose(np.asarray(from_mpe_action(out)), a, atol=1e-7)

    jitted = jax.jit(lambda x: from_mpe_action(to_mpe_action(x)))
    np.testing.assert_allclose(np.asarray(jitted(jnp.asarray(a))), a, atol=1e-7)
    vmapped = jax.vmap(jax.vmap(lambda x: from_mpe_action(to_mpe_action(x))))
    np.testing.assert_allclose(np.asarray(vmapped(jnp.asarray(a))), a, atol=1e-7)
    with pytest.raises(ValueError):
        to_mpe_action(jnp.zeros((3,)))
    with pytest.raises(ValueError):
        from_mpe_action(jnp.zeros((2,)))


def test_env_decodes_adapter_output_to_the_original_force_times_accel():
    env = make_env("capture", continuous=True)
    assert env.continuous_actions
    assert env.action_space(env.agents[0]).shape == (5,)
    rng = np.random.default_rng(1)
    a = jnp.asarray(rng.uniform(-1.0, 1.0, size=(env.num_agents, 2)), dtype=jnp.float32)
    mpe = to_mpe_action(a)
    u, _ = env._decode_continuous_action(env.agent_range, mpe)
    expected = np.asarray(a) * np.asarray(env.accel[: env.num_agents])[:, None]
    np.testing.assert_allclose(np.asarray(u), expected, atol=1e-6)
    # joint_action_dict routes blue/red actions to the right agent names.
    acts = joint_action_dict(env, a[env.num_adversaries], a[: env.num_adversaries])
    assert set(acts) == set(env.agents)
    for i, name in enumerate(env.agents):
        np.testing.assert_allclose(np.asarray(acts[name]), np.asarray(mpe[i]), atol=1e-7)


def test_continuous_env_steps_and_matches_discrete_axis_moves():
    """A unit continuous action along +x equals the discrete '+x' action (2)."""
    disc = make_env("capture")
    cont = make_env("capture", continuous=True)
    _, s0 = disc.reset(jax.random.PRNGKey(5))
    _, c0 = cont.reset(jax.random.PRNGKey(5))
    assert bool(jnp.all(s0.p_pos == c0.p_pos))
    d_acts = {a: jnp.int32(2) for a in disc.agents}
    c_acts = {a: to_mpe_action(jnp.asarray([1.0, 0.0])) for a in cont.agents}
    _, s1, r1, _, _ = disc.step_env(jax.random.PRNGKey(6), s0, d_acts)
    _, c1, r2, _, _ = cont.step_env(jax.random.PRNGKey(6), c0, c_acts)
    np.testing.assert_allclose(np.asarray(s1.p_pos), np.asarray(c1.p_pos), atol=1e-6)
    np.testing.assert_allclose(np.asarray(s1.p_vel), np.asarray(c1.p_vel), atol=1e-6)
    assert all(abs(float(r1[a]) - float(r2[a])) < 1e-5 for a in disc.agents)
    # And the facade's random policy / evaluator run end to end.
    m = evaluate_policy(
        cont, random_policy(cont), n_eps=3, key=jax.random.PRNGKey(0), num_steps=4
    )
    assert all(v.shape == (3,) for v in m.values())
    with pytest.raises(ValueError):
        SimpleTagObjectivesMPE(action_type="Bogus")


def test_current_lava_default_constants():
    env = SimpleTagObjectivesMPE(pred_type="risk")
    assert env.lava_penalty == 100.0
    assert env.base_lava_penalty == 0.0
    assert env.prey_lava_penalty == 0.0
    _, st = env.reset(jax.random.PRNGKey(0))
    st = _contrived(env, st)
    _, _, rew, _, _ = env.step_env(
        jax.random.PRNGKey(1), st, {a: jnp.int32(0) for a in env.agents}
    )
    assert abs(float(rew[env.adversaries[0]]) - (-100.0)) < 1e-3
    env2 = SimpleTagObjectivesMPE(pred_type="capture")
    _, st2 = env2.reset(jax.random.PRNGKey(0))
    st2 = _contrived(env2, st2)
    _, _, rew2, _, _ = env2.step_env(
        jax.random.PRNGKey(1), st2, {a: jnp.int32(0) for a in env2.agents}
    )
    assert abs(float(rew2[env2.adversaries[0]]) - (-0.9)) < 1e-3
