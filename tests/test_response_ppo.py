"""Equal-information blue-only PPO: fixed inputs, exact density, updates and IO."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from mopa.continuous import tanh_gaussian_log_prob  # noqa: E402
from mopa.response_ppo import (  # noqa: E402
    ResponsePPO,
    create_response,
    update_response,
    warm_start_response,
)


def _agent(context_dim=8, seed=0):
    return create_response(np.zeros(66), np.ones(66), context_dim=context_dim, seed=seed, hidden=16)


def _batch(agent, n=19):
    rng = np.random.default_rng(4)
    obs = rng.normal(size=(n, 66)).astype(np.float32)
    context = rng.normal(size=(n, agent.context_dim)).astype(np.float32)
    sampled = agent.sample(obs, context, jax.random.PRNGKey(3))
    return {
        "observations": obs, "context": context,
        **{k: np.asarray(v) for k, v in sampled.items()},
        "returns": np.asarray(sampled["old_values"]) + 2,
        "advantages": np.linspace(-1, 1, n, dtype=np.float32),
        "valid_mask": np.ones(n, bool),
    }


def _different(a, b):
    return any(not np.array_equal(x, y) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True))


@pytest.mark.parametrize("context_dim", [0, 8])
def test_inputs_actions_and_recorded_density(context_dim):
    agent = _agent(context_dim)
    batch = _batch(agent)
    features = agent.features(batch["observations"], batch["context"])
    np.testing.assert_array_equal(features[:, :66], batch["observations"])
    np.testing.assert_array_equal(features[:, 66:], batch["context"])
    mean, log_std = agent.actor.apply_fn(agent.actor.params, features)
    np.testing.assert_allclose(batch["log_probs"], tanh_gaussian_log_prob(mean, log_std, batch["pre_tanh"]), atol=1e-6)
    np.testing.assert_allclose(batch["old_values"], agent.critic.apply_fn(agent.critic.params, features), atol=1e-6)
    deterministic = jax.jit(lambda a, x, c: a.act(x, c, jax.random.PRNGKey(1)))(agent, batch["observations"], batch["context"])
    np.testing.assert_allclose(deterministic, np.tanh(mean), atol=1e-6)
    assert deterministic.shape == batch["actions"].shape == (19, 2)
    assert np.max(np.abs(batch["actions"])) <= 1
    np.testing.assert_array_equal(_batch(_agent(context_dim))["actions"], batch["actions"])


def test_state_and_history_are_shared_features_without_new_information():
    agent = _agent().replace(state_mean=jnp.full(66, 2.), state_std=jnp.full(66, 3.))
    obs = np.full((2, 66), 5., np.float32)
    context = np.arange(16, dtype=np.float32).reshape(2, 8)
    np.testing.assert_array_equal(agent.features(obs, context), np.concatenate((np.ones_like(obs), context), -1))
    no_history = _agent(0)
    with pytest.raises(ValueError, match="matching"):
        no_history.features(obs, context)
    with pytest.raises(ValueError, match="matching"):
        agent.features(obs[:, :65], context)


def test_ppo_updates_actor_value_but_not_input_statistics():
    agent = _agent()
    updated, metrics = update_response(agent, _batch(agent), jax.random.PRNGKey(7), epochs=2, minibatch_size=8)
    assert metrics["valid_samples"] == 19 and metrics["gradient_updates"] == 6
    assert _different(agent.actor.params, updated.actor.params)
    assert _different(agent.critic.params, updated.critic.params)
    np.testing.assert_array_equal(agent.state_mean, updated.state_mean)
    np.testing.assert_array_equal(agent.state_std, updated.state_std)
    assert all(np.isfinite(v) for v in metrics.values())
    assert int(updated.actor.step) == int(updated.critic.step) == 6


def test_masked_rows_cannot_affect_training_including_nan_padding():
    agent = _agent()
    clean = _batch(agent)
    dirty = {name: np.concatenate((value, np.full_like(value[:2], np.nan)), axis=0) for name, value in clean.items() if name != "valid_mask"}
    dirty["valid_mask"] = np.r_[np.ones(19, bool), np.zeros(2, bool)]
    first, m1 = update_response(agent, clean, jax.random.PRNGKey(6), epochs=1, minibatch_size=8)
    second, m2 = update_response(agent, dirty, jax.random.PRNGKey(6), epochs=1, minibatch_size=8)
    for left, right in zip(jax.tree.leaves(first), jax.tree.leaves(second), strict=True):
        np.testing.assert_array_equal(left, right)
    assert m1 == m2


def test_ppo_rejects_missing_pre_tanh_and_inconsistent_actions():
    agent = _agent()
    batch = _batch(agent)
    missing = {k: v for k, v in batch.items() if k != "pre_tanh"}
    with pytest.raises(ValueError, match="pre_tanh"):
        update_response(agent, missing, jax.random.PRNGKey(0))
    with pytest.raises(ValueError, match="tanh"):
        update_response(agent, {**batch, "actions": np.zeros_like(batch["actions"])}, jax.random.PRNGKey(0))
    with pytest.raises(ValueError, match="valid_mask"):
        update_response(agent, {**batch, "valid_mask": np.zeros(19, bool)}, jax.random.PRNGKey(0))


def test_warm_start_learns_demonstrations_and_real_return_targets():
    agent = _agent(0)
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(64, 66)).astype(np.float32)
    context = np.empty((64, 0), np.float32)
    actions = np.tile(np.array([[0.25, -0.4]], np.float32), (64, 1))
    returns = np.full(64, 2., np.float32)
    key = jax.random.PRNGKey(0)
    before_action = np.square(np.asarray(agent.act(obs, context, key)) - actions).mean()
    before_value = np.square(np.asarray(agent.value(obs, context)) - returns).mean()
    trained, metrics = warm_start_response(agent, {
        "observations": obs, "context": context, "actions": actions, "returns": returns,
    }, key, steps=400, minibatch_size=32)
    assert metrics["gradient_updates"] == 400
    assert np.square(np.asarray(trained.act(obs, context, key)) - actions).mean() < before_action * .15
    assert np.square(np.asarray(trained.value(obs, context)) - returns).mean() < before_value * .15


def test_serialization_keeps_optimizer_and_exact_continuation(tmp_path):
    agent, _ = update_response(_agent(), _batch(_agent()), jax.random.PRNGKey(8), epochs=1)
    path = tmp_path / "response.msgpack"
    agent.save(path)
    loaded = ResponsePPO.load(path)
    for left, right in zip(jax.tree.leaves(agent), jax.tree.leaves(loaded), strict=True):
        np.testing.assert_array_equal(left, right)
    batch = _batch(agent)
    first, _ = update_response(agent, batch, jax.random.PRNGKey(9), epochs=1)
    second, _ = update_response(loaded, batch, jax.random.PRNGKey(9), epochs=1)
    for left, right in zip(jax.tree.leaves(first), jax.tree.leaves(second), strict=True):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("kwargs", [{"context_dim": 3}, {"hidden": 0}, {"learning_rate": 0}, {"clip_epsilon": 2}, {"entropy_coefficient": -1}])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        create_response(np.zeros(66), np.ones(66), **kwargs)


def test_invalid_normalization():
    with pytest.raises(ValueError, match="66D"):
        create_response(np.zeros(66), np.zeros(66))


@pytest.fixture(scope="module")
def benchmark_driver():
    pytest.importorskip("jaxmarl")
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_control_benchmark.py"
    spec = importlib.util.spec_from_file_location("benchmark_ppo_wiring", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_benchmark_gae_terminal_budget_and_padding(benchmark_driver):
    reward = np.array([[1., 2., 0.], [1., 2., 0.]])
    values = np.array([[3., 4., 5., 100.], [3., 4., 5., 100.]])
    valid = np.array([[1, 1, 0], [1, 1, 0]], bool)
    terminal = np.array([[0, 1, 0], [0, 0, 0]], bool)
    advantage, returns = benchmark_driver.advantage_targets(reward, values, terminal, valid, discount=.9, gae_lambda=1.)
    # Capture cannot bootstrap; the administrative quota cut must bootstrap V(s_2).
    np.testing.assert_allclose(returns, [[2.8, 2., 0.], [6.85, 6.5, 0.]], atol=1e-6)
    np.testing.assert_array_equal(advantage[:, 2], 0)


@pytest.mark.parametrize("context_dim", [0, 8])
def test_real_environment_ppo_chain_preserves_samples_context_and_quota(benchmark_driver, tmp_path, context_dim):
    from mopa.action_decoder import ActionDecoderConfig, fit_action_decoder_vae
    from mopa.evaluation import run_matched_episodes
    from mopa.nets import ContinuousActor
    from mopa.zero_s import ZeroSOpponent, zero_s_features
    from tag_objectives import make_env

    opponent = None
    if context_dim:
        rng = np.random.default_rng(90)
        state = rng.normal(size=(3, 5, 66)).astype(np.float32)
        action = np.tanh(state[:, :-1, :2])
        fit = fit_action_decoder_vae(
            np.asarray(zero_s_features(state[:, :-1])), action, np.full(3, 4),
            np.arange(3), jax.random.PRNGKey(91),
            config=ActionDecoderConfig(action_type="continuous", hid=8, window=3, steps=1, batch=8),
        )
        opponent = ZeroSOpponent.from_fit(fit, np.arange(3), np.arange(3))
    env = make_env("capture", continuous=True, max_steps=7)
    red = ContinuousActor(hidden_dim=128)
    width = max(env.observation_space(a).shape[0] for a in env.agents)
    red_params = red.init(jax.random.PRNGKey(92), jnp.zeros((1, width)))
    frozen_red = jax.tree.map(np.array, red_params)
    agent = _agent(context_dim)
    controller = benchmark_driver.Controller(agent, env, training=True)
    result = run_matched_episodes(
        env, red_params, controller,
        np.asarray(jax.random.split(jax.random.PRNGKey(93), 2)),
        np.asarray(jax.random.split(jax.random.PRNGKey(94), 2)),
        horizon=9, context_mode="online" if opponent else "zero", label=0,
        zero_s=opponent, context_width=None if opponent else 0,
        max_transitions=5, record_transitions=True,
    )
    tr = result["transitions"]
    assert int(tr["valid_mask"].sum()) == 5
    np.testing.assert_array_equal(tr["valid_length"], [3, 2])
    replay_context = benchmark_driver.replay_context(tr)
    assert replay_context.shape == (2, 10, context_dim)
    if opponent:
        independently_encoded = opponent.context(tr["state"], tr["red_action"], tr["valid_length"])
        np.testing.assert_allclose(replay_context, independently_encoded, atol=2e-6)
    else:
        assert replay_context.shape[-1] == 0
    batch = benchmark_driver.ppo_batch(agent, tr, controller.samples)
    selected = batch["valid_mask"]
    np.testing.assert_allclose(batch["actions"][selected], np.tanh(batch["pre_tanh"][selected]), atol=1e-6)
    features = agent.features(batch["observations"], batch["context"])
    means, stds = agent.actor.apply_fn(agent.actor.params, features)
    log_probs = tanh_gaussian_log_prob(means, stds, batch["pre_tanh"])
    np.testing.assert_allclose(batch["log_probs"][selected], log_probs[selected], atol=2e-6)
    np.testing.assert_allclose(batch["old_values"][selected], agent.value(batch["observations"], batch["context"])[selected], atol=2e-6)
    values = np.asarray(agent.value(tr["state"], replay_context))
    for episode, length in enumerate(tr["valid_length"]):
        t = length - 1
        assert tr["truncated_timeout"][episode, t]
        expected = tr["blue_reward"][episode, t] + .99 * values[episode, t + 1]
        np.testing.assert_allclose(batch["returns"][episode, t], expected, atol=2e-6)
    updated, metrics = update_response(agent, batch, jax.random.PRNGKey(95), epochs=1, minibatch_size=4)
    assert metrics["valid_samples"] == 5 and _different(agent.actor.params, updated.actor.params)
    for original, after in zip(jax.tree.leaves(frozen_red), jax.tree.leaves(red_params), strict=True):
        np.testing.assert_array_equal(original, after)
    updated.save(tmp_path / "agent.msgpack")
    loaded = ResponsePPO.load(tmp_path / "agent.msgpack")
    np.testing.assert_array_equal(
        updated.act(tr["state"], replay_context, jax.random.PRNGKey(96)),
        loaded.act(tr["state"], replay_context, jax.random.PRNGKey(96)),
    )


def test_benchmark_collection_uses_only_training_groups_and_exact_quotas(benchmark_driver, tmp_path):
    from mopa.nets import ContinuousActor
    from tag_objectives import make_env

    env = make_env("capture", continuous=True)
    width = max(env.observation_space(a).shape[0] for a in env.agents)
    red = ContinuousActor(hidden_dim=128)
    params = {(checkpoint, label): red.init(jax.random.PRNGKey(100 + 3 * checkpoint + label), jnp.zeros((1, width)))
              for checkpoint in (0, 1) for label in range(3)}
    # Deliberately no checkpoint 2: collection must not even attempt its lookup.
    trajectories, batches, logs = benchmark_driver.collect(
        SimpleNamespace(transitions_per_group=3, batch_episodes=2),
        _agent(0), None, 0, 0, tmp_path, params,
    )
    assert len(trajectories) == len(batches) == len(logs) == 6
    assert sum(record["valid_transitions"] for record in logs) == 18
    for tr, batch, record in zip(trajectories, batches, logs, strict=True):
        assert int(tr["valid_mask"].sum()) == 3
        assert batch["context"].shape[-1] == 0
        path = tmp_path / record["file"]
        assert benchmark_driver.file_sha256(path) == record["sha256"]
        with np.load(path) as saved:
            assert set(saved["checkpoint_seed"]) <= {0, 1}
            np.testing.assert_array_equal(saved["valid_mask"], tr["valid_mask"])
