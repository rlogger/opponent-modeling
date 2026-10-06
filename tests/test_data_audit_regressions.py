"""Data integrity, population calibration and terminal-context audit regressions."""
from __future__ import annotations

from types import SimpleNamespace
from typing import NamedTuple

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("jaxmarl")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from mopa import continuous_data  # noqa: E402
from mopa.evaluation import run_matched_episodes  # noqa: E402
from mopa.nets import ContinuousActor  # noqa: E402
from mopa.tdmpc_data import termination_calibration  # noqa: E402
from tag_objectives import make_env  # noqa: E402


@pytest.fixture(scope="module")
def actor_params():
    env = make_env("capture", continuous=True)
    width = max(env.observation_space(a).shape[0] for a in env.agents)
    return ContinuousActor(action_dim=2, hidden_dim=128).init(
        jax.random.PRNGKey(7), jnp.zeros((1, width))
    )


@pytest.fixture
def tiny_data(monkeypatch, actor_params):
    monkeypatch.setattr(continuous_data, "load_continuous_actor_params", lambda _: actor_params)
    return continuous_data.continuous_objective_dataset(
        n_eps=2, ckpt_seeds=(0,), rng0=42, num_steps=3
    ).as_dict()


@pytest.mark.parametrize("field", ["blue_reward", "causal_context", "survival_time"])
@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_dataset_rejects_nonfinite_fields_before_replay(tiny_data, field, bad_value):
    data = dict(tiny_data)
    if field == "causal_context":
        data[field] = np.zeros((*data["state"].shape[:2], 1), np.float32)
    else:
        data[field] = data[field].copy()
    data[field].flat[0] = bad_value
    with pytest.raises(ValueError, match=field):
        continuous_data.validate_continuous_dataset(data, replay_indices=[0])


def test_replay_rejects_nan_reward_and_checks_initial_observations(tiny_data):
    clean = continuous_data.validate_continuous_dataset(tiny_data)
    assert clean["replay"]["max_abs_error_blue_reward"] == 0.0
    for field in ("blue_observation", "red_observation"):
        bad = dict(tiny_data)
        bad[field] = bad[field].copy()
        bad[field][0, 0, 0] += 1.0
        with pytest.raises(ValueError, match="exact replay failed"):
            continuous_data.validate_continuous_dataset(bad, replay_indices=[0])
    bad = dict(tiny_data)
    bad["blue_reward"] = bad["blue_reward"].copy()
    bad["blue_reward"][0, 0] = np.nan
    with pytest.raises(ValueError, match="blue_reward"):
        continuous_data.replay_episodes(bad, indices=[0])


def _calibration_fixture(n_capture, population=1000):
    capture = np.zeros((population, 1), bool)
    capture[:n_capture] = True
    replay = SimpleNamespace(
        state=np.zeros((population, 2, 1), np.float32), valid_length=np.ones(population, int),
        terminated=capture, blue_action=np.zeros((population, 1, 2), np.float32),
        red_action=np.zeros((population, 1, 2), np.float32),
        context=np.zeros((population, 2, 0), np.float32),
    )
    model = SimpleNamespace(
        predict_continues=True, encoder=SimpleNamespace(params=None),
        continue_model=SimpleNamespace(params=None),
        encode=lambda x, params, key: x,
        transition_inputs=lambda action, context, red: action,
        continue_logits=lambda x, action, params: jnp.full((len(x),), np.log(9.0)),
    )
    return SimpleNamespace(model=model), replay


@pytest.mark.parametrize("n_capture", [0, 1, 50, 999, 1000])
def test_stratified_calibration_preserves_population_prevalence(n_capture):
    agent, replay = _calibration_fixture(n_capture)
    full = termination_calibration(agent, replay, max_samples=2000)
    sampled = termination_calibration(agent, replay, max_samples=10)
    assert sampled["n_samples"] == 10
    assert sampled["population_n_capture_transitions"] == n_capture
    for metric in ("brier", "ece_10_bins", "hard_threshold_accuracy"):
        assert sampled[metric] == pytest.approx(full[metric], abs=1e-7)
    assert sampled["brier"] == pytest.approx(0.01 + 0.8 * n_capture / 1000, abs=1e-7)
    if 0 < n_capture < 1000:
        assert 0 < sampled["n_capture_transitions"] < 10


class _Carry(NamedTuple):
    context: jax.Array


class _CounterOpponent:
    prototypes = np.array([[10., 10., 10.], [20., 20., 20.], [30., 30., 30.]], np.float32)

    def initial_context(self, batch):
        return _Carry(jnp.zeros((batch, 3)))

    def update_context(self, carry, state, action, active):
        del state, action
        return _Carry(carry.context + active[:, None] * (jnp.arange(len(active)) + 1)[:, None])


class _LegacyCounter:
    def online_context(self, prey_history, pred_history, done, capture_t, *, horizon):
        del pred_history
        completed = len(prey_history) - 1
        if completed == 0:
            return np.zeros((len(done), 3), np.float32)
        lengths = np.clip(np.where(done, capture_t, completed), 1, horizon)
        return np.repeat((lengths * (np.arange(len(done)) + 1))[:, None], 3, axis=1).astype(np.float32)


@pytest.mark.parametrize("encoder_kind", ["streaming", "legacy"])
@pytest.mark.parametrize("mode", ["zero", "oracle", "wrong_oracle", "online", "shuffled"])
@pytest.mark.parametrize("stop", ["horizon", "timeout", "quota"])
def test_bootstrap_context_preserves_intervention_and_last_causal_prefix(actor_params, encoder_kind, mode, stop):
    env = make_env("capture", continuous=True, max_steps=2 if stop == "timeout" else 100)
    kwargs = {"zero_s": _CounterOpponent()} if encoder_kind == "streaming" else {"encoder": _LegacyCounter()}

    def blue(state, obs, context, carry, key, t):
        del state, obs, key, t
        return jnp.zeros((len(context), 2)), carry

    result = run_matched_episodes(
        env, actor_params, blue,
        np.asarray(jax.random.split(jax.random.PRNGKey(42), 2)),
        np.asarray(jax.random.split(jax.random.PRNGKey(43), 2)),
        horizon=4, context_mode=mode, label=1, record_transitions=True,
        max_transitions=3 if stop == "quota" else None, **kwargs,
    )
    trace = result["transitions"]
    lengths = trace["valid_length"]
    assert lengths.tolist() == {"horizon": [4, 4], "timeout": [2, 2], "quota": [2, 1]}[stop]
    np.testing.assert_array_equal(result["survival_time"], lengths)
    prototypes = _CounterOpponent.prototypes if encoder_kind == "streaming" else np.eye(3)
    if mode == "zero":
        expected = np.zeros((2, 3))
    elif mode in {"oracle", "wrong_oracle"}:
        expected = np.repeat(prototypes[1 if mode == "oracle" else 2][None], 2, axis=0)
    else:
        counts = lengths * np.array([1, 2]) if mode == "online" else np.minimum(lengths, lengths[::-1]) * np.array([2, 1])
        expected = np.repeat(counts[:, None], 3, axis=1)
    np.testing.assert_array_equal(trace["final_context"], expected)
    for episode, length in enumerate(lengths):
        if length < 4:
            np.testing.assert_array_equal(trace["context"][episode, length:], np.repeat(expected[episode][None], 4 - length, axis=0))
    assert trace["truncated_timeout"][np.arange(2), lengths - 1].all()
