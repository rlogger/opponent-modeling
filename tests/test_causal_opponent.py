"""Information boundaries and math of separately versioned causal predictors."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import serialization

from mopa.causal_opponent import (
    CausalOpponent,
    CausalOpponentConfig,
    encoder_parameter_count,
    evaluate_prediction,
    fit_causal_opponent,
    gaussian_kl,
    history_batch,
    match_encoder_capacity,
    opponent_features,
    sample_training_indices,
)


def data():
    rng = np.random.default_rng(21)
    state = rng.normal(size=(6, 6, 66)).astype(np.float32)
    action = np.tanh(rng.normal(size=(6, 5, 2))).astype(np.float32)
    lengths = np.array([5, 3, 4, 5, 3, 4])
    return state, action, lengths


def config(method, **kwargs):
    cfg = CausalOpponentConfig(method=method, history=0 if method == "bc" else 3,
                               hid=8, steps=0, batch=4, latent_dim=3,
                               beta=0.01 if "vae" in method else 0,
                               sample_training="vae" in method)
    return replace(cfg, **kwargs)


@pytest.mark.parametrize("method", ["bc", "deterministic_history", "mlp_vae", "recurrent_vae"])
def test_future_perturbation_and_streaming_match_offline(method):
    x, a, sizes = data()
    model, _ = fit_causal_opponent(x, a, sizes, np.array([0, 1, 2]), jax.random.PRNGKey(0), config=config(method))
    context = model.context(x, a, sizes)
    changed_x, changed_a = x.copy(), a.copy()
    changed_x[:, 2:] += 20
    changed_a[:, 2:] *= -1
    changed_context = model.context(changed_x, changed_a, sizes)
    np.testing.assert_array_equal(context[:, :3], changed_context[:, :3])
    features = model.features(x[:, :-1])
    for t in range(a.shape[1] + 1):
        ep = jnp.arange(len(sizes))
        times = jnp.minimum(t, jnp.asarray(sizes))
        window, mask = history_batch(features, jnp.asarray(a), ep, times, model.config.history)
        mean, _ = model.posterior(window, mask)
        np.testing.assert_allclose(context[:, t], mean, atol=2e-6)
    np.testing.assert_array_equal(context[:, 0], 0)
    for i, size in enumerate(sizes):
        np.testing.assert_array_equal(context[i, size:], np.broadcast_to(context[i, size], context[i, size:].shape))


def test_history_zero_is_memoryless_and_no_history_prior():
    x, a, sizes = data()
    model, _ = fit_causal_opponent(x, a, sizes, np.array([0, 1]), jax.random.PRNGKey(1), config=config("mlp_vae", history=0))
    np.testing.assert_array_equal(model.context(x, a, sizes), 0)
    c = model.initial_context(2)
    mean, logvar = model.posterior(c.window, c.mask)
    np.testing.assert_array_equal(mean, 0)
    np.testing.assert_array_equal(logvar, 0)


def test_sampler_measure_on_unequal_episode_lengths():
    train, lengths = jnp.array([1, 3]), jnp.array([4, 1, 4, 9])
    for measure, expected in [("episode_uniform", 0.5), ("transition_uniform", 0.1)]:
        episodes, times = sample_training_indices(train, lengths, 20000, jax.random.key(51), jax.random.key(52), measure)
        assert abs(float(jnp.mean(episodes == 1)) - expected) < 0.015
        assert set(np.unique(episodes)) == {1, 3}
        assert np.all(np.asarray(times) >= 0) and np.all(times < lengths[episodes])
        counts = np.bincount(np.asarray(times[episodes == 3]), minlength=9)
        assert (counts.max() - counts.min()) / counts.mean() < 0.2


@pytest.mark.parametrize("method", ["mlp_vae", "recurrent_vae", "deterministic_history"])
@pytest.mark.parametrize("limit", [0, 1, 8])
def test_separate_history_length_control_matches_streaming_and_excludes_older_pairs(method, limit, tmp_path):
    rng = np.random.default_rng(8)
    x = rng.normal(size=(3, 13, 66)).astype(np.float32)
    a = np.tanh(rng.normal(size=(3, 12, 2))).astype(np.float32)
    lengths = np.array([12, 9, 3])
    cfg = config(method, history=8, max_history=limit)
    model, _ = fit_causal_opponent(x, a, lengths, np.array([0, 1]), jax.random.key(8), config=cfg)
    contexts = model.context(x, a, lengths)
    features = model.features(x[:, :-1])
    for t in (0, 1, 3, 8, 9, 12):
        times = jnp.minimum(t, jnp.asarray(lengths))
        h, mask = history_batch(features, jnp.asarray(a), jnp.arange(3), times, 8, max_history=limit)
        means, _ = model.posterior(h, mask)
        np.testing.assert_allclose(contexts[:, t], means, atol=2e-6)
        np.testing.assert_array_equal(mask.sum(axis=(1, 2)), np.minimum(times, limit))
    changed_x, changed_a = x.copy(), a.copy()
    changed_x[0, :12-limit] += 20
    changed_a[0, :12-limit] *= -1
    np.testing.assert_allclose(model.context(changed_x, changed_a, lengths)[0, 12], contexts[0, 12], atol=2e-6)
    if limit == 0:
        np.testing.assert_array_equal(contexts, 0)
    path = tmp_path / "limited.msgpack"
    model.save(path)
    loaded = CausalOpponent.load(path)
    assert loaded.config.max_history == limit
    np.testing.assert_array_equal(loaded.context(x, a, lengths), contexts)


def test_context_accepts_empty_quota_rows_but_fitting_rejects_them():
    x, a, sizes = data()
    model, _ = fit_causal_opponent(x, a, sizes, np.array([0, 1]), jax.random.PRNGKey(1), config=config("recurrent_vae"))
    changed = sizes.copy()
    changed[:3] = [0, 1, 5]
    contexts = model.context(x, a, changed)
    np.testing.assert_array_equal(contexts[0], 0)
    np.testing.assert_array_equal(contexts[1, 1:], np.broadcast_to(contexts[1, 1], contexts[1, 1:].shape))
    with pytest.raises(ValueError, match="lengths"):
        fit_causal_opponent(x, a, changed, np.array([0, 1]), jax.random.PRNGKey(1), config=config("bc"))


def test_padded_sentinels_and_training_fold_only_normalization():
    x, a, sizes = data()
    valid = np.arange(5)[None] < sizes[:, None]
    x[:, :-1][~valid] = np.nan
    a[~valid] = np.nan
    train = np.array([0, 1, 2])
    first, _ = fit_causal_opponent(x, a, sizes, train, jax.random.PRNGKey(4), config=config("recurrent_vae"))
    x[3:, :, :] = 400
    second, _ = fit_causal_opponent(x, a, sizes, train, jax.random.PRNGKey(4), config=config("recurrent_vae"))
    np.testing.assert_array_equal(first.state_mean, second.state_mean)
    np.testing.assert_array_equal(first.state_std, second.state_std)
    for p, q in zip(jax.tree.leaves(first.params), jax.tree.leaves(second.params)):
        np.testing.assert_array_equal(p, q)
    assert np.isfinite(first.context(x, a, sizes)).all()


def test_expert_geometry_matches_real_observation_and_agent_aliasing():
    from mopa.continuous_data import markov_state
    from tag_objectives import make_env

    env = make_env("risk", action_type="Continuous", num_adversaries=1)
    obs, state = env.reset(jax.random.PRNGKey(9))
    vector = markov_state(env, state)
    features = opponent_features(vector, "expert17_v1")
    np.testing.assert_allclose(features[-9:], obs[env.adversaries[0]][-9:], atol=1e-6)
    changed = vector.at[56:62].add(0.5)
    np.testing.assert_array_equal(opponent_features(vector, "agent8_v1"), opponent_features(changed, "agent8_v1"))
    assert not np.allclose(features, opponent_features(changed, "expert17_v1"))


def test_gaussian_kl_known_values_and_objective_history_boundary():
    assert float(gaussian_kl(jnp.zeros(3), jnp.zeros(3))) == 0
    assert float(gaussian_kl(jnp.array([1., 2.]), jnp.zeros(2))) == pytest.approx(2.5)
    f = jnp.arange(5, dtype=jnp.float32)[None, :, None]
    a = jnp.zeros((1, 5, 2))
    before, mask = history_batch(f, a, jnp.array([0]), jnp.array([2]), 3)
    np.testing.assert_array_equal(before[0, 0, :, 0], [0, 1, 0])
    np.testing.assert_array_equal(mask, [[[True, True, False], [False, False, False]]])


def test_window_pooling_matches_mean_and_independent_sample_variance():
    x, a, lengths = data()
    model, _ = fit_causal_opponent(x, a, lengths, np.arange(3), jax.random.PRNGKey(3), config=config("recurrent_vae", history=2))
    features = model.features(x[:, :-1])
    history, mask = history_batch(features, jnp.asarray(a), jnp.array([0]), jnp.array([5]), 2)
    mu, lv = model.posterior(history, mask)
    window_values = [model.posterior(history[:, i], mask[:, i]) for i in range(3)]
    np.testing.assert_allclose(mu, np.mean([m for m, _ in window_values], axis=0), atol=1e-6)
    np.testing.assert_allclose(np.exp(lv), np.sum([np.exp(v) for _, v in window_values], axis=0)/9, atol=1e-6)


def test_capacity_is_matched_before_fitting_and_count_is_exact():
    x, a, lengths = data()
    recurrent = config("recurrent_vae")
    mlp, report = match_encoder_capacity(config("mlp_vae"), recurrent)
    assert mlp.hid == recurrent.hid and mlp.lat == recurrent.lat
    for cfg in [recurrent, mlp]:
        model, _ = fit_causal_opponent(x, a, lengths, np.arange(3), jax.random.PRNGKey(3), config=cfg)
        actual = sum(int(v.size) for v in jax.tree.leaves(model.params["encoder"]))
        assert actual == encoder_parameter_count(cfg)
    assert report["parameter_gap"] == encoder_parameter_count(mlp)-encoder_parameter_count(recurrent)


@pytest.mark.parametrize("method", ["bc", "deterministic_history", "mlp_vae", "recurrent_vae"])
def test_small_update_roundtrip_and_bounded_predictions(method, tmp_path):
    x, a, sizes = data()
    cfg = config(method, steps=2)
    training = tmp_path / "training.msgpack"
    model, history = fit_causal_opponent(x, a, sizes, np.array([0, 1, 2]), jax.random.key(31), config=cfg, labels=np.array([0, 1, 2, 0, 1, 2]), training_state_path=training)
    assert len(history) == 2 and np.isfinite(history[-1]["loss"])
    path = tmp_path / "predictor.msgpack"
    model.save(path)
    restored = CausalOpponent.load(path)
    saved = serialization.msgpack_restore(training.read_bytes())
    assert saved["step"] == 2
    optimizer_state = serialization.from_state_dict(optax.adam(cfg.learning_rate).init(restored.params), saved["optimizer"])
    assert int(optimizer_state[0].count) == 2
    context = model.context(x, a, sizes)
    predicted = model.actions(x[:, 0], context[:, 0])
    np.testing.assert_array_equal(predicted, restored.actions(x[:, 0], context[:, 0]))
    assert np.max(np.abs(predicted)) <= 1
    metrics, raw = evaluate_prediction(restored, x, a, sizes, np.array([3, 4, 5]), np.array([0, 1, 2, 0, 1, 2]), samples=3)
    assert set(metrics["per_objective"]) == {"0", "1", "2"}
    assert raw["indices"].tolist() == [3, 4, 5]
    valid = raw["donor_available"]
    recipients, times = np.where(valid)
    donors = raw["donor_episode_index"][valid]
    assert np.all(raw["donor_label"][valid] != raw["labels"][recipients])
    assert np.all(sizes[donors] >= times)
    np.testing.assert_array_equal(raw["donor_time"][valid], times)
    np.testing.assert_array_equal(raw["donor_completed_pairs"][valid], times)


def test_cross_type_intervention_without_donor_is_excluded():
    x, a, sizes = data()
    model, _ = fit_causal_opponent(x, a, sizes, np.array([0, 1]), jax.random.key(11), config=config("bc"))
    metrics, raw = evaluate_prediction(model, x, a, sizes, np.array([0, 1]), np.zeros(6, int), samples=2)
    assert not raw["donor_available"].any()
    assert metrics["per_objective"]["0"]["cross_type_context_mse"] is None
    assert metrics["per_objective"]["0"]["cross_type_context_excluded_transitions"] == int(sizes[:2].sum())


def test_runner_rejects_config_or_data_outside_frozen_protocol():
    import importlib.util
    from dataclasses import asdict
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("causal_runner", Path(__file__).parents[1] / "scripts/run_causal_prediction.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    cfg = config("recurrent_vae")
    binding = {"checkpoint_families": {"train": [0, 1], "validation": [], "test": [2]},
               "evaluation_split": "test", "evaluation_samples": 32,
               "dataset_sha256": "a" * 64, "dataset_sidecar_sha256": "b" * 64, "splits_sha256": "d" * 64}
    protocol = {"causal_prediction": {"main": {"configurations": [asdict(cfg)], "seeds": [0], **binding}}}
    runner.validate_protocol_run(protocol, "main", cfg, 0, binding)
    with pytest.raises(ValueError, match="splits_sha256"):
        runner.validate_protocol_run(protocol, "main", cfg, 0, dict(binding, splits_sha256="e" * 64))
    with pytest.raises(ValueError, match="configuration"):
        runner.validate_protocol_run(protocol, "main", replace(cfg, steps=cfg.steps + 1), 0, binding)
    with pytest.raises(ValueError, match="dataset_sha256"):
        runner.validate_protocol_run(protocol, "main", cfg, 0, dict(binding, dataset_sha256="c" * 64))
    with pytest.raises(ValueError, match="checkpoint_families"):
        runner.validate_protocol_run(protocol, "main", cfg, 0, dict(binding, checkpoint_families={"train": [0], "validation": [1], "test": [2]}))


def test_rejects_duplicate_indices_and_bad_config():
    x, a, sizes = data()
    with pytest.raises(ValueError, match="unique"):
        fit_causal_opponent(x, a, sizes, np.array([0, 0]), jax.random.PRNGKey(0), config=config("bc"))
    with pytest.raises(ValueError, match="BC requires"):
        config("bc", history=1)
    with pytest.raises(ValueError, match="beta=0"):
        config("deterministic_history", beta=0.1)


@pytest.mark.parametrize("method", ["bc", "recurrent_vae"])
def test_world_adapter_freeze_and_agent_checkpoint(method):
    from mopa.tdmpc import create_agent, load_config
    from mopa.tdmpc_data import SequenceReplay

    x, a, lengths = data()
    model, _ = fit_causal_opponent(x, a, lengths, np.array([0, 1, 2]), jax.random.PRNGKey(1), config=config(method))
    cfg = load_config(profile="smoke")
    cfg.update(opponent_mode="factored", context_dim=model.config.lat)
    cfg["encoder"]["type"] = "identity"
    cfg["world_model"]["hidden_dim"] = 16
    mean, std = np.zeros(66, np.float32), np.ones(66, np.float32)
    base = create_agent(cfg, 66, key=jax.random.PRNGKey(1), obs_mean=mean, obs_std=std)
    with pytest.raises(ValueError, match="does not match"):
        model.attach(base, mean+1, std)
    agent = model.attach(base, mean, std)
    context = model.context(x, a, lengths)
    raw, c = jnp.asarray(x[:2, 1]), jnp.asarray(context[:2, 1])
    np.testing.assert_allclose(agent.model.red_action(raw, c, agent.model.red_model.params), model.actions(raw, c), atol=1e-6)
    dataset = dict(state=x, blue_action=a*.3, red_action=a, valid_length=lengths,
                   blue_reward=np.zeros((6, 5), np.float32), terminated_capture=np.zeros((6, 5), bool),
                   truncated_timeout=np.arange(5)[None] == lengths[:, None]-1)
    replay = SequenceReplay.from_dataset(dataset, np.arange(3), agent.horizon, context)
    batch = replay.sample(np.random.default_rng(9), agent.batch_size)
    updated, info = agent.update(**batch, key=jax.random.PRNGKey(8))
    assert np.isfinite(info["total_loss"])
    for before, after in zip(jax.tree.leaves(agent.model.red_model.params), jax.tree.leaves(updated.model.red_model.params), strict=True):
        np.testing.assert_array_equal(before, after)
    restored = serialization.from_bytes(agent, serialization.to_bytes(updated))
    for expected, actual in zip(jax.tree.leaves(updated), jax.tree.leaves(restored), strict=True):
        np.testing.assert_array_equal(expected, actual)
