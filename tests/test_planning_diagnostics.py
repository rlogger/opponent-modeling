"""Analytic contracts and numerical frozen-diagnostic integration checks."""
import numpy as np
import pytest


@pytest.fixture(scope="module")
def diagnostic():
    from mopa import planning_diagnostics
    return planning_diagnostics


def test_reward_on_terminal_transition_counts_but_later_padding_does_not(diagnostic):
    reward = np.array([[2., -10., 999.], [2., 3., 4.]])
    valid = np.array([[True, True, False], [True, True, True]])
    np.testing.assert_allclose(diagnostic.discounted_rewards(reward, valid, .5), [-3., 4.5])


def test_ranking_ties_and_negative_selected_outcome_are_not_positive_evidence(diagnostic):
    result = diagnostic.ranking_agreement([2., 2., 0.], [-3., 1., 1.])
    assert result["informative_pairs"] == 2 and result["true_tied_pairs"] == 1
    assert result["pairwise_agreement"] == .25
    assert result["selected_realized_return"] == -3. and result["selection_regret"] == 4.
    assert diagnostic.ranking_agreement([2., 1.], [0., 0.])["pairwise_agreement"] is None


def test_invariants_ignore_legal_outside_arena_and_retain_model_violations(diagnostic):
    states = np.zeros((2, 4, 66))
    states[..., 62:65] = .4
    states[..., 65] = np.arange(4) / 100
    states[..., :4] = 1.5  # positions outside arena incur a reward; they are legal.
    valid = np.array([[True, True, False], [True, True, True]])
    result = diagnostic.physical_invariants(states, valid)
    assert set(result["counts"].values()) == {0}
    states[0, 1, 8] = .2
    states[1, 2, 40] = .5
    states[0, 3, 65] = 100.  # ignored because this transition is after predicted stop.
    result = diagnostic.physical_invariants(states, valid)
    assert result["counts"]["static_geometry"] == 1
    assert result["counts"]["nonbinary_collected"] == 1
    assert result["counts"]["decreasing_collected"] == 1
    assert result["counts"]["wrong_clock_increment"] == 0


def test_finite_and_tail_targets_are_reported_separately(diagnostic):
    states = np.zeros((2, 2, 66))
    states[:, :, 62:65] = .4
    states[:, 1, 65] = .01
    states[..., 2] = 1.
    actual = dict(state=states, valid=np.ones((2, 1), bool), red_action=np.zeros((2, 1, 2)),
                  reward=np.array([[1.], [2.]]), captured=np.zeros((2, 1), bool), continues=np.ones((2, 1), bool))
    predicted = dict(state=states.copy(), valid=actual["valid"], red_action=actual["red_action"],
                     reward=actual["reward"], continues_probability=np.ones((2, 1)),
                     finite_return=np.array([1., 2.]), tail=np.array([9., 0.]), planner_return=np.array([10., 2.]))
    result = diagnostic.compare_candidates(predicted, actual, np.ones(66), .9,
        capture_distance=.125, arena=1.25, actual_tail=np.array([0., 10.]))
    assert result["finite_return_mse"] == 0.
    assert result["ranking_finite_horizon"]["pairwise_agreement"] == 1.
    assert result["ranking_full_finite_task"]["pairwise_agreement"] == 0.
    assert result["discounted_tail_mse"] == 81.


def test_invalid_prediction_vectors_fail_closed(diagnostic):
    with pytest.raises(ValueError, match="nonfinite"):
        diagnostic.ranking_agreement([float("nan"), 1], [1, 2])
    with pytest.raises(ValueError, match="aligned"):
        diagnostic.discounted_rewards([[1, 2]], [[True]], .9)


@pytest.mark.parametrize("clock", [None, .98, .99, 1.])
@pytest.mark.parametrize("mode", ["implicit", "factored"])
def test_candidate_diagnostic_preserves_planner_key_and_timeout_values(diagnostic, clock, mode):
    import jax
    import jax.numpy as jnp

    from mopa import mppi
    from mopa.tdmpc import create_agent, load_config

    cfg = load_config(profile="smoke")
    cfg.update(opponent_mode=mode, context_dim=0)
    cfg["encoder"]["type"] = "identity"
    cfg["world_model"].update(hidden_dim=16, predict_continues=True)
    if clock is not None:
        cfg["world_model"]["transition_contract"] = "objective_static_clock_v1"
    mean, std = np.zeros(66, np.float32), np.ones(66, np.float32)
    agent = nontrivial_test_agent(create_agent(cfg, 66, key=jax.random.PRNGKey(2), obs_mean=mean, obs_std=std))
    initial = jnp.linspace(-.2, .4, 66)
    if clock is not None:
        initial = initial.at[65].set(clock)
    actions = jax.random.uniform(jax.random.PRNGKey(7), (8, 3, 2), minval=-1, maxval=1)
    context = jnp.zeros((0,))
    key = jax.random.PRNGKey(19)
    red_source = (lambda raw, c, time: agent.model.red_action(raw, c, agent.model.red_model.params)) if mode == "factored" else None
    observed = diagnostic.imagined_candidates(agent, initial, actions, context, mean, std, red_source, key=key)
    expected = mppi.estimate_value(agent, jnp.broadcast_to(initial, (8, 66)), actions,
                                   jnp.zeros((8, 0)), 3, key)
    if clock == 1.:
        np.testing.assert_array_equal(expected, 0.)
    else:
        assert np.ptp(expected) > 1e-5
    if clock is not None:
        assert int(observed["valid"].sum()) == 8 * round((1 - clock) * 100)
    np.testing.assert_allclose(observed["planner_return"], expected, rtol=2e-5, atol=2e-4)


@pytest.fixture(scope="module")
def frozen_components():
    """Random initialized networks exercise contracts without any fitting."""
    import jax
    import jax.numpy as jnp

    from mopa.action_decoder import (
        ActionDecoderConfig,
        FrozenActionEncoder,
        MLPHead,
        SeqGaussian,
    )
    from mopa.nets import ContinuousActor
    from mopa.tdmpc import create_agent, load_config
    from mopa.zero_s import ZeroSOpponent
    from tag_objectives import make_env

    config = ActionDecoderConfig(action_type="continuous", hid=8, lat=2, window=4)
    encoder = FrozenActionEncoder(
        SeqGaussian(lat=2, hid=8).init(jax.random.PRNGKey(1), jnp.zeros((1, 4, 10)), jnp.ones((1, 4), bool)),
        np.zeros(8, np.float32), np.ones(8, np.float32), config)
    opponent = ZeroSOpponent(encoder, MLPHead(out=2, hid=8).init(jax.random.PRNGKey(2), jnp.zeros((1, 10))),
                             np.zeros((3, 2), np.float32))
    cfg = load_config(profile="smoke")
    cfg.update(opponent_mode="factored", context_dim=2)
    cfg["encoder"]["type"] = "identity"
    cfg["world_model"].update(hidden_dim=16, predict_continues=True)
    cfg["tdmpc2"].update(population_size=8, policy_prior_samples=2, num_elites=2, mppi_iterations=2)
    mean, std = np.zeros(66, np.float32), np.ones(66, np.float32)
    agent = opponent.attach(create_agent(cfg, 66, key=jax.random.PRNGKey(3), obs_mean=mean, obs_std=std), mean, std)
    env = make_env("capture", continuous=True)
    red = ContinuousActor(action_dim=2, hidden_dim=128).init(jax.random.PRNGKey(4), jnp.zeros((1, 35)))
    return env, red, opponent, agent, mean, std


def assert_tree_equal(left, right):
    import jax
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for one, two in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        np.testing.assert_array_equal(one, two)


def nontrivial_test_agent(agent):
    """Seeded synthetic nonzero heads, so score/key parity is not a zero check."""
    import jax
    import jax.numpy as jnp

    def output_head(state, width, seed, scale, continue_bias=None):
        leaves, structure = jax.tree.flatten(state.params)
        changed = []
        for index, value in enumerate(leaves):
            if value.ndim >= 2 and value.shape[-1] == width:
                value = scale * jax.random.normal(jax.random.fold_in(jax.random.PRNGKey(seed), index), value.shape)
            if continue_bias is not None and value.shape == (1,):
                value = jnp.full_like(value, continue_bias)
            changed.append(value)
        return state.replace(params=jax.tree.unflatten(structure, changed))

    model = agent.model
    return agent.replace(model=model.replace(
        dynamics_model=output_head(model.dynamics_model, 66, 101, .01),
        reward_model=output_head(model.reward_model, 101, 102, .05),
        value_model=output_head(model.value_model, 101, 103, .05),
        continue_model=output_head(model.continue_model, 1, 104, .001, continue_bias=2.)))


@pytest.mark.parametrize("deterministic,train", [(True, False), (False, False), (False, True)])
def test_mppi_observer_does_not_change_actions_warm_start_or_selected_value(frozen_components, deterministic, train):
    import jax
    import jax.numpy as jnp

    from mopa import mppi
    from mopa.continuous_data import markov_state

    env, _, _, agent, _, _ = frozen_components
    agent = nontrivial_test_agent(agent)
    _, state = env.reset(jax.random.PRNGKey(5))
    x = markov_state(env, state)[None]
    context = jnp.array([[.3, -.2]])
    key = jax.random.PRNGKey(6)
    original = mppi.plan(agent, x, 3, context, deterministic=deterministic, train=train, key=key)
    action, plan, observed = mppi.plan(agent, x, 3, context, deterministic=deterministic, train=train,
                                      key=key, return_diagnostics=True)
    assert_tree_equal(original, (action, plan))
    # The observed population is exactly the one scored by the deployed planner.
    values = mppi.estimate_value(agent, jnp.repeat(x[:, None], 8, axis=1), observed["candidate_actions"],
                                jnp.repeat(context[:, None], 8, axis=1), 3, observed["estimate_value_key"])
    # Standalone versus nested JIT evaluation can round float32 reductions
    # differently; the actual action and warm-start outputs remain bitwise equal.
    np.testing.assert_allclose(values, observed["candidate_values"], atol=1e-6, rtol=0)
    assert float(jnp.ptp(values)) > 1e-5 and float(jnp.max(jnp.abs(values))) > 1e-4
    changed_key_values = mppi.estimate_value(agent, jnp.repeat(x[:, None], 8, axis=1), observed["candidate_actions"],
        jnp.repeat(context[:, None], 8, axis=1), 3, jax.random.fold_in(observed["estimate_value_key"], 999))
    assert not np.allclose(values, changed_key_values, atol=1e-6, rtol=0)
    if deterministic:
        selected = int(observed["selected_population_index"][0])
        np.testing.assert_array_equal(action[0], observed["candidate_actions"][0, selected, 0])
    second = mppi.plan(agent, x, 3, context, original[1], deterministic=deterministic, train=train, key=key)
    second_observed = mppi.plan(agent, x, 3, context, plan, deterministic=deterministic, train=train,
                                key=key, return_diagnostics=True)
    assert_tree_equal(second, second_observed[:2])


@pytest.mark.parametrize("horizon", [1, 3, 10])
def test_learned_diagnostic_values_reproduce_original_population_scores(diagnostic, frozen_components, horizon):
    import jax
    import jax.numpy as jnp

    from mopa import mppi
    from mopa.continuous_data import markov_state

    env, _, opponent, agent, mean, std = frozen_components
    agent = nontrivial_test_agent(agent)
    _, state = env.reset(jax.random.PRNGKey(8))
    raw = markov_state(env, state)
    context = jnp.array([[.3, -.2]])
    _, _, observed = mppi.plan(agent, raw[None], horizon, context, deterministic=True,
                               key=jax.random.PRNGKey(9), return_diagnostics=True)
    result = diagnostic.imagined_candidates(agent, raw, observed["candidate_actions"][0], context[0], mean, std,
        lambda physical, conditioning, time: opponent.actions(physical, conditioning), key=observed["estimate_value_key"])
    assert np.ptp(result["planner_return"]) > 1e-5 and np.any(np.abs(result["tail"]) > 1e-4)
    np.testing.assert_allclose(result["planner_return"], observed["candidate_values"][0], atol=2e-4, rtol=2e-5)


def test_complete_state_and_context_round_trip_then_independent_simulator_replay(diagnostic, frozen_components):
    import flax.serialization
    import jax
    import jax.numpy as jnp

    from mopa.continuous_data import deterministic_specialist_action, markov_state
    from tag_objectives import joint_action_dict

    env, red, opponent, agent, _, _ = frozen_components
    prefix = diagnostic.frozen_prefix(env, agent, opponent, red, jax.random.PRNGKey(17),
                                     jax.random.PRNGKey(18), 3, jax.random.PRNGKey(19))
    assert prefix["valid_length"] == 3
    state, carry = prefix["complete_state"], prefix["context_carry"]
    restored_state = jax.tree.map(jnp.asarray, flax.serialization.from_bytes(state, flax.serialization.to_bytes(state)))
    restored_carry = jax.tree.map(jnp.asarray, flax.serialization.from_bytes(carry, flax.serialization.to_bytes(carry)))
    assert_tree_equal(restored_state, state)
    assert_tree_equal(restored_carry, carry)
    assert int(carry.observed[0]) == 3
    reference = opponent.context(prefix["state"][None], prefix["red_action"][None], np.array([3]))
    np.testing.assert_allclose(reference[0, -1], carry.context[0], atol=2e-6, rtol=0)
    actions = np.array([[[.2, -.1], [-.4, .3], [.1, -.3]]], np.float32)
    step_seed = jax.random.PRNGKey(20)
    branch = diagnostic.simulator_candidates(env, restored_state, actions, red, step_seed)
    # Independently written unbatched loop, preserving the complete simulator
    # state. This deliberately does not call the diagnostic branch function.
    current = restored_state
    for t in range(3):
        obs = env.get_obs(current)
        true_red = deterministic_specialist_action(red, obs[env.adversaries[0]][None], 35)
        predicted_obs_red = diagnostic.specialist_on_imagined_observation(env, restored_state,
                                                                         markov_state(env, current)[None], red)
        np.testing.assert_allclose(predicted_obs_red, true_red, atol=1e-6, rtol=0)
        commands = {name: value[0] for name, value in joint_action_dict(env, jnp.asarray(actions[:, t]), true_red[:, None]).items()}
        _, following, reward, _, info = env.step_env(jax.random.fold_in(step_seed, int(state.step) + t), current, commands)
        np.testing.assert_allclose(branch["state"][0, t + 1], markov_state(env, following), atol=1e-6, rtol=0)
        np.testing.assert_allclose(branch["reward"][0, t], reward[env.good_agents[0]], atol=1e-6, rtol=0)
        np.testing.assert_array_equal(branch["captured"][0, t], info["captured"][1])
        current = following
    assert_tree_equal(jax.tree.map(lambda value: value[0], branch["complete_final_state"]), current)
    repeated = diagnostic.simulator_candidates(env, state, actions, red, step_seed)
    assert_tree_equal(branch, repeated)


def test_terminal_and_timeout_masks_count_only_physical_transitions(diagnostic, frozen_components):
    import jax
    import jax.numpy as jnp

    from mopa.continuous_data import markov_state

    env, red, _, agent, _, _ = frozen_components
    _, state = env.reset(jax.random.PRNGKey(24))
    actions = np.zeros((2, 3, 2), np.float32)
    terminal = state.replace(done=jnp.ones_like(state.done, dtype=bool), capture_t=jnp.asarray(2, dtype=state.capture_t.dtype))
    stopped = diagnostic.simulator_candidates(env, terminal, actions, red, jax.random.PRNGKey(25))
    assert not stopped["valid"].any() and not stopped["reward"].any() and not stopped["continues"].any()
    np.testing.assert_array_equal(stopped["state"], np.broadcast_to(markov_state(env, terminal), (2, 4, 66)))
    timeout = state.replace(step=jnp.asarray(99, dtype=state.step.dtype))
    timed = diagnostic.simulator_candidates(env, timeout, actions, red, jax.random.PRNGKey(25))
    np.testing.assert_array_equal(timed["valid"], [[True, False, False], [True, False, False]])
    assert not timed["continues"].any() and not timed["reward"][:, 1:].any()
    tail = diagnostic.simulator_policy_tail(agent, env, timed["complete_final_state"], red,
        jnp.zeros(2), jax.random.PRNGKey(25), 102, 3, key=jax.random.PRNGKey(26))
    assert not tail["valid"].any() and not tail["return"].any()


@pytest.mark.parametrize("remaining_steps", [9, 99])
def test_full_remainder_commands_follow_frozen_deterministic_prior(diagnostic, frozen_components, remaining_steps):
    import jax
    import jax.numpy as jnp

    from mopa.continuous_data import markov_state

    env, _, _, agent, mean, std = frozen_components
    _, state = env.reset(jax.random.PRNGKey(31))
    raw = markov_state(env, state)
    context = jnp.array([.2, -.1])
    key = jax.random.PRNGKey(32)
    # A declared constant drift makes the recurrence observable without fitting.
    # The default zero residual would leave x fixed and miss stale-state bugs.
    dynamic_params = jax.tree.map(lambda value: value.at[:4].set(.03) if value.shape == (66,) else value,
                                  agent.model.dynamics_model.params)
    agent = agent.replace(model=agent.model.replace(dynamics_model=agent.model.dynamics_model.replace(params=dynamic_params)))
    result = diagnostic.imagined_policy_commands(agent, raw, context, mean, std, remaining_steps, key=key)
    assert result["status"] == "complete" and result["requested_steps"] == result["generated_steps"] == remaining_steps
    assert result["blue_action"].shape == (remaining_steps, 2) and result["state"].shape == (remaining_steps + 1, 66)
    assert not np.array_equal(result["blue_action"][0], result["blue_action"][-1])
    model, x = agent.model, raw[None]
    alive = True
    for t in range(remaining_steps):
        blue = model.sample_actions(model.policy_inputs(x, context[None]), model.policy_model.params,
                                    deterministic=True, key=jax.random.fold_in(key, t))[0]
        red = model.red_action(x, context[None], model.red_model.params)
        transition = model.transition_inputs(blue, context[None], red)
        continuation = jax.nn.sigmoid(model.continue_logits(x, transition, model.continue_model.params))
        x = model.next(x, transition, model.dynamics_model.params)
        np.testing.assert_array_equal(result["blue_action"][t], blue[0])
        np.testing.assert_array_equal(result["state"][t + 1], x[0])
        assert result["valid"][t] == alive
        alive = alive and bool(continuation[0] > .5)
    different_key = diagnostic.imagined_policy_commands(agent, raw, context, mean, std, remaining_steps, key=jax.random.PRNGKey(33))
    # Mean actions do not use sampling noise; this is not a policy-sample sweep.
    for field in ("blue_action", "state", "red_action", "valid"):
        np.testing.assert_array_equal(result[field], different_key[field])


def test_full_remainder_reports_nonfinite_generation_as_failed(diagnostic, frozen_components):
    import jax
    import jax.numpy as jnp

    from mopa.continuous_data import markov_state

    env, _, _, agent, mean, std = frozen_components
    _, state = env.reset(jax.random.PRNGKey(34))
    bad_dynamics = agent.model.dynamics_model.replace(params=jax.tree.map(lambda value: jnp.full_like(value, jnp.nan), agent.model.dynamics_model.params))
    bad_agent = agent.replace(model=agent.model.replace(dynamics_model=bad_dynamics))
    result = diagnostic.imagined_policy_commands(bad_agent, markov_state(env, state), jnp.zeros(2), mean, std, 10,
                                                  key=jax.random.PRNGKey(35))
    assert result["status"] == "failed_nonfinite_generation" and result["generated_steps"] == 0
    assert result["requested_steps"] == 10 and result["blue_action"].shape == (0, 2)
    assert result["state"].shape == (1, 66)  # no fabricated continuation


def test_full_remainder_counterfactuals_share_commands_without_simulator_feedback(frozen_components, tmp_path):
    import importlib.util
    from pathlib import Path
    from types import SimpleNamespace

    import jax
    import jax.numpy as jnp

    path = Path(__file__).resolve().parents[1] / "scripts/run_frozen_diagnostics.py"
    spec = importlib.util.spec_from_file_location("frozen_diagnostic_driver", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    env, red, opponent, agent, mean, std = frozen_components
    _, initial = env.reset(jax.random.PRNGKey(36))
    initial = initial.replace(step=jnp.asarray(97, dtype=initial.step.dtype))
    # Changing only the alternative BC source cannot change generated commands,
    # learned predictions, or the real simulator reference for those commands.
    for name, sign in (("positive_bc", 1.), ("negative_bc", -1.)):
        bc = SimpleNamespace(actions=lambda raw, s=sign: jnp.full((len(raw), 2), s))
        result = module.full_remainder_scene(tmp_path / name, env, agent, opponent, bc, mean, std, red,
                                             initial, np.zeros(2, np.float32), jax.random.PRNGKey(37), jax.random.PRNGKey(38))
        assert result["status"] == "complete" and result["generated_steps"] == 3
        assert set(result["methods"]) == {"learned", "state_only_BC", "actual_specialist_on_imagined_observation"}
        assert result["methods"]["learned"]["ranking_finite_horizon"]["status"].startswith("one supplied")
    for name in ("commands.npz", "actual.npz", "learned.npz"):
        with np.load(tmp_path / "positive_bc" / name) as one, np.load(tmp_path / "negative_bc" / name) as two:
            for field in one.files:
                np.testing.assert_array_equal(one[field], two[field])
    with np.load(tmp_path / "positive_bc/state_only_BC.npz") as one, np.load(tmp_path / "negative_bc/state_only_BC.npz") as two:
        assert not np.array_equal(one["red_action"], two["red_action"])
        # Zero-initialized residual dynamics can legitimately ignore this
        # action intervention. No positive state-separation outcome is required.
