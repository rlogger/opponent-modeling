"""Matched-control orchestration: causal inputs, exact quotas, and split guards."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mopa.action_decoder import ActionDecoderConfig, fit_action_decoder_vae
from mopa.evaluation import run_matched_episodes
from mopa.nets import ContinuousActor
from mopa.tdmpc_data import SequenceReplay
from mopa.zero_s import ZeroSOpponent, zero_s_features
from tag_objectives import make_env


@pytest.fixture(scope="module")
def driver():
    path = Path(__file__).resolve().parents[1] / "scripts/run_control_benchmark.py"
    spec = importlib.util.spec_from_file_location("control_benchmark_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def env_and_red():
    env = make_env("capture", continuous=True)
    width = max(env.observation_space(name).shape[0] for name in env.agents)
    actor = ContinuousActor(action_dim=2, hidden_dim=128)
    red = jax.tree.map(jnp.zeros_like, actor.init(jax.random.PRNGKey(2), jnp.zeros((1, width))))
    return env, red


@pytest.fixture(scope="module")
def opponent():
    rng = np.random.default_rng(7)
    state = rng.normal(size=(3, 5, 66)).astype(np.float32)
    actions = np.tanh(state[:, :-1, :2]).astype(np.float32)
    fit = fit_action_decoder_vae(
        np.asarray(zero_s_features(state[:, :-1])), actions, np.full(3, 4), np.arange(3),
        jax.random.PRNGKey(1),
        config=ActionDecoderConfig(action_type="continuous", hid=8, steps=1, batch=8),
    )
    return ZeroSOpponent.from_fit(fit, np.arange(3), np.arange(3))


def test_keyed_resets_match_arms_and_heldout_objectives(driver):
    first = driver.keys(1, 2, 4, 3, 8)
    repeated = driver.keys(1, 2, 4, 3, 8)
    for actual, expected in zip(first[:2], repeated[:2], strict=True):
        np.testing.assert_array_equal(actual, expected)
    for arguments in ((2, 2, 4, 3), (1, 3, 4, 3), (1, 2, 5, 3), (1, 2, 4, 4)):
        assert not np.array_equal(driver.keys(*arguments, 8)[0], first[0])
    heldout = driver.keys(1, 0, 0, 0, 24, evaluation=True)
    for group in range(3):
        np.testing.assert_array_equal(driver.keys(1, 0, group, 0, 24, evaluation=True)[0], heldout[0])
    assert not np.array_equal(driver.keys(1, 0, 0, 0, 24)[0], heldout[0])


def test_gae_bootstraps_truncation_but_not_capture_or_padding(driver):
    reward = np.array([[1, 2, 999], [1, 2, 999], [999, 999, 999]], np.float32)
    value = np.array([[10, 20, 30, 999], [10, 20, 30, 999], [7, 8, 9, 10]], np.float32)
    valid = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], bool)
    terminal = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]], bool)
    advantage, returns = driver.advantage_targets(reward, value, terminal, valid,
                                                  discount=0.5, gae_lambda=1)
    # Capture: last target=2. Budget/time cut: last target=2+.5*30=17.
    np.testing.assert_allclose(advantage, [[-8, -18, 0], [-0.5, -3, 0], [0, 0, 0]])
    np.testing.assert_allclose(returns, [[2, 2, 0], [9.5, 17, 0], [0, 0, 0]])
    changed = value.copy()
    changed[:, 3] = -1e6
    changed[0, 2] = -1e6  # terminal next value must not enter the capture target
    altered, _ = driver.advantage_targets(reward, changed, terminal, valid,
                                           discount=0.5, gae_lambda=1)
    np.testing.assert_array_equal(altered, advantage)
    with pytest.raises(ValueError, match="unaligned"):
        driver.advantage_targets(reward, value[:, :-1], terminal, valid)


def _assert_stopped_trace(trace, quota):
    lengths, valid = trace["valid_length"], trace["valid_mask"]
    assert int(lengths.sum()) == int(valid.sum()) == quota
    np.testing.assert_array_equal(valid, np.arange(valid.shape[1])[None] < lengths[:, None])
    done = trace["terminated_capture"] | trace["truncated_timeout"]
    np.testing.assert_array_equal(done.sum(axis=1), lengths > 0)
    assert not (trace["terminated_capture"] & trace["truncated_timeout"]).any()
    for row, length in enumerate(lengths):
        if length:
            assert done[row, length - 1]
        np.testing.assert_array_equal(trace["blue_action"][row, length:], 0)
        np.testing.assert_array_equal(trace["red_action"][row, length:], 0)
        np.testing.assert_array_equal(trace["blue_reward"][row, length:], 0)
        np.testing.assert_array_equal(trace["state"][row, length:], np.broadcast_to(
            trace["state"][row, length], trace["state"][row, length:].shape))


@pytest.mark.parametrize("quota", [2, 7])
def test_actual_transition_quota_zero_context_and_zero_length_rows(driver, env_and_red, quota):
    env, red = env_and_red
    calls = []

    def blue(state, obs, context, carry, key, t):
        del state, obs, key
        assert context.shape == (4, 0)
        calls.append(t)
        return jnp.broadcast_to(jnp.array([0.2, -0.1]), (4, 2)), carry

    reset, step, _ = driver.keys(0, 0, 0, 0, 4)
    result = run_matched_episodes(env, red, blue, reset, step, horizon=6,
                                 context_mode="zero", context_width=0, label=0,
                                 max_transitions=quota, record_transitions=True)
    tr = result["transitions"]
    _assert_stopped_trace(tr, quota)
    np.testing.assert_array_equal(tr["valid_length"], [1, 1, 0, 0] if quota == 2 else [2, 2, 2, 1])
    assert calls == [0] if quota == 2 else calls == [0, 1]
    assert tr["context"].shape == (4, 6, 0)
    assert tr["final_context"].shape == (4, 0)
    assert not np.array_equal(tr["state"][0, 0], tr["state"][0, 1])
    # Empty episode rows remain auditable but have zero replay probability.
    replay = SequenceReplay.from_dataset(tr, np.arange(4), 3, driver.replay_context(tr))
    sampled = replay.sample(np.random.default_rng(3), 16)
    assert sampled["context"].shape == (3, 16, 0)
    assert np.isfinite(np.asarray(sampled["observations"])).all()


@pytest.mark.parametrize("options", [
    {"context_width": -1}, {"max_transitions": 0}, {"max_transitions": -1},
])
def test_collector_rejects_invalid_context_width_and_transition_quota(driver, env_and_red, options):
    env, red = env_and_red
    reset, step, _ = driver.keys(0, 0, 0, 0, 4)

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid collector input must fail before acting")

    with pytest.raises(ValueError):
        run_matched_episodes(env, red, forbidden, reset, step, horizon=6,
                             context_mode="zero", label=0, **options)


@pytest.mark.parametrize("quota", [2, 7])
def test_budget_cut_0s_context_matches_actual_causal_prefix(driver, env_and_red, opponent, quota):
    env, red = env_and_red

    def blue(state, obs, context, carry, key, t):
        del state, obs, context, key, t
        return jnp.broadcast_to(jnp.array([0.2, -0.1]), (4, 2)), carry

    reset, step, _ = driver.keys(0, 0, 0, 0, 4)
    result = run_matched_episodes(env, red, blue, reset, step, horizon=6,
                                 context_mode="online", label=0, zero_s=opponent,
                                 max_transitions=quota, record_transitions=True)
    tr = result["transitions"]
    _assert_stopped_trace(tr, quota)
    positive = tr["valid_length"] > 0
    reference = opponent.context(tr["state"][positive], tr["red_action"][positive],
                                 tr["valid_length"][positive])
    observed = driver.replay_context(tr)
    np.testing.assert_allclose(observed[positive], reference, atol=2e-6)
    np.testing.assert_array_equal(observed[~positive], 0)
    for row, length in enumerate(tr["valid_length"]):
        np.testing.assert_array_equal(observed[row, length:], np.broadcast_to(
            observed[row, length], observed[row, length:].shape))


def test_collector_uses_exact_per_group_quotas_and_only_training_opponents(driver, env_and_red, monkeypatch, tmp_path):
    env, red = env_and_red

    class FastController:
        is_ppo = False

        def __init__(self, *args, **kwargs):
            self.samples = {}

        def __call__(self, state, obs, context, carry, key, t):
            del obs, key, t
            assert context.shape[-1] == 0
            return jnp.full((state.p_pos.shape[0], 2), 0.1), carry

    monkeypatch.setattr(driver, "Controller", FastController)
    monkeypatch.setattr(driver, "make_env", lambda *a, **k: env)
    args = SimpleNamespace(transitions_per_group=2, batch_episodes=4)
    # No held-out key exists: consulting checkpoint 2 fails immediately.
    params = {(checkpoint, label): red for checkpoint in (0, 1) for label in range(3)}
    trajectories, ppo_data, logs = driver.collect(args, object(), None, 0, 0, tmp_path, params)
    assert len(trajectories) == len(logs) == 6 and not ppo_data
    assert sum(log["valid_transitions"] for log in logs) == 12
    assert {log["checkpoint"] for log in logs} == {0, 1}
    for trace, log in zip(trajectories, logs, strict=True):
        _assert_stopped_trace(trace, 2)
        path = tmp_path / log["file"]
        assert driver.file_sha256(path) == log["sha256"]
        with np.load(path, allow_pickle=False) as saved:
            assert np.isin(saved["checkpoint_seed"], [0, 1]).all()


def test_main_filters_heldout_before_statistics_or_encoder_fitting(driver, monkeypatch, tmp_path):
    checkpoint = np.array([0, 1, 2, 0, 1, 2])
    state = np.ones((6, 3, 66), np.float32)
    state[checkpoint == 2] = np.nan  # any train-time leakage is immediately visible
    data = {"state": state, "valid_mask": np.ones((6, 2), bool), "checkpoint_seed": checkpoint}
    dataset = SimpleNamespace(checkpoint_seed=checkpoint, as_dict=lambda: data)
    monkeypatch.setattr(driver, "load_continuous_dataset", lambda path: dataset)
    monkeypatch.setattr(driver, "specialist_binding", lambda args: {})
    monkeypatch.setattr(driver, "load_continuous_actor_params", lambda path: {})
    seen = []

    def prepare(args, seed, selected, mean, std):
        assert np.isfinite(selected["state"]).all() and np.isfinite(mean).all() and np.isfinite(std).all()
        np.testing.assert_array_equal(selected["checkpoint_seed"], [0, 1, 0, 1])
        seen.append(seed)
        return None

    monkeypatch.setattr(driver, "prepare", prepare)
    assert driver.main(["--execute", "--phase", "prepare", "--out", str(tmp_path), "--seeds", "0,1"]) == 0
    assert seen == [0, 1]


def test_default_invocation_does_not_execute_experiments(driver, monkeypatch, tmp_path, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("implementation-only invocation must not access experiment data")
    monkeypatch.setattr(driver, "load_continuous_dataset", forbidden)
    monkeypatch.setattr(driver, "summarize", forbidden)
    out = tmp_path / "not-created"
    assert driver.main(["--out", str(out)]) == 0
    assert not out.exists()
    assert "Implementation-only" in capsys.readouterr().out


def test_prepare_rejects_changed_source_before_loading_artifacts(driver, monkeypatch, tmp_path):
    out = tmp_path / "seed_0" / "shared"
    out.mkdir(parents=True)
    (out / "manifest.json").write_text(json.dumps({
        "dataset": "dataset", "code": {"source": "old"}, "seed": 0,
        "encoder_steps": 1, "smoke": True, "artifacts": {},
    }))
    monkeypatch.setattr(driver, "file_sha256", lambda path: "dataset")
    monkeypatch.setattr(driver, "code_hashes", lambda: {"source": "changed"})
    args = SimpleNamespace(out=tmp_path, dataset=tmp_path / "data.npz", encoder_steps=1, smoke=True,
                           specialists={})
    with pytest.raises(ValueError, match="configuration/source changed"):
        driver.prepare(args, 0, {}, np.zeros(66), np.ones(66))


def test_specialist_binding_rejects_changed_checkpoint(driver, monkeypatch, tmp_path):
    args = SimpleNamespace(dataset=tmp_path / "dataset.npz", logdir=tmp_path)
    checkpoints = [
        {"seed": seed, "objective": objective, "team": "pred", "sha256": f"{seed}:{objective}"}
        for seed in (0, 1, 2) for objective in driver.OBJECTIVE_TYPES
    ]
    args.dataset.with_suffix(".manifest.json").write_text(json.dumps({"source_checkpoints": checkpoints}))
    monkeypatch.setattr(driver, "continuous_checkpoint_path", lambda logdir, objective, team, seed: tmp_path / f"{seed}:{objective}")
    monkeypatch.setattr(driver, "file_sha256", lambda path: path.name)
    bound = driver.specialist_binding(args)
    assert len(bound) == 9 and bound["2:risk"]["sha256"] == "2:risk"
    monkeypatch.setattr(driver, "file_sha256", lambda path: "changed" if path.name == "2:risk" else path.name)
    with pytest.raises(ValueError, match="provenance"):
        driver.specialist_binding(args)


def _resume_fixture(driver, tmp_path, opponent):
    args = SimpleNamespace(out=tmp_path, dataset=tmp_path / "dataset.npz", smoke=True,
                           offline_updates=1, rounds=1, updates_per_round=1,
                           transitions_per_group=2, batch_episodes=4, eval_episodes=2,
                           specialists={"0:capture": {"sha256": "specialist"}})
    out = tmp_path / "seed_0" / "0s"
    out.mkdir(parents=True)
    (out / "agent.msgpack").write_bytes(b"synthetic checkpoint")
    data = dict(state=np.zeros((3, 5, 66), np.float32),
                blue_action=np.zeros((3, 4, 2), np.float32), red_action=np.zeros((3, 4, 2), np.float32),
                blue_reward=np.zeros((3, 4), np.float32),
                terminated_capture=np.zeros((3, 4), bool),
                truncated_timeout=np.tile([False, False, False, True], (3, 1)),
                valid_mask=np.ones((3, 4), bool), valid_length=np.full(3, 4))
    budget = {name: getattr(args, name) for name in (
        "offline_updates", "rounds", "updates_per_round", "transitions_per_group",
        "batch_episodes", "eval_episodes", "smoke")}
    manifest = {"arm": "0s", "seed": 0, "code": {"source": "current"},
                "dataset": "sha", "dataset_path": str(args.dataset.resolve()),
                "dataset_manifest_path": str(args.dataset.with_suffix(".manifest.json").resolve()),
                "dataset_manifest": "sha", "specialists": args.specialists,
                "shared_manifest": "sha", "budget": budget, "agent_sha256": "sha",
                "replay_rng": np.random.default_rng(0).bit_generator.state,
                "update_key": [0, 1], "rounds": []}
    shared = (opponent, None, np.zeros((3, 5, 8), np.float32))
    return args, out, data, manifest, shared


@pytest.mark.parametrize("field", ["code", "specialists", "dataset_manifest"])
def test_controller_resume_rejects_source_or_specialist_changes(driver, opponent, monkeypatch, tmp_path, field):
    args, out, data, manifest, shared = _resume_fixture(driver, tmp_path, opponent)
    manifest[field] = "changed"
    (out / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(driver, "file_sha256", lambda path: "sha")
    monkeypatch.setattr(driver, "code_hashes", lambda: {"source": "current"})
    with pytest.raises(ValueError, match="source/config-mismatched resume"):
        driver.train_and_evaluate(args, 0, "0s", data, np.zeros(66), np.ones(66), shared, {})


def test_resume_rejects_decoder_not_equal_to_frozen_shared_source(driver, opponent, monkeypatch, tmp_path):
    args, out, data, manifest, shared = _resume_fixture(driver, tmp_path, opponent)
    (out / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(driver, "file_sha256", lambda path: "sha")
    monkeypatch.setattr(driver, "code_hashes", lambda: {"source": "current"})

    def changed_decoder(template, payload):
        del payload
        red = template.model.red_model
        changed = red.replace(params=jax.tree.map(lambda value: value + 1, red.params))
        return template.replace(model=template.model.replace(red_model=changed))

    monkeypatch.setattr(driver.flax.serialization, "from_bytes", changed_decoder)
    with pytest.raises(RuntimeError, match="frozen shared opponent"):
        driver.train_and_evaluate(args, 0, "0s", data, np.zeros(66), np.ones(66), shared, {})
