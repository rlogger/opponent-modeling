"""Saved 0s Equation 3 checkpoints compose with causal, real-environment MPC."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("jaxmarl")
pytest.importorskip("distrax")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from mopa.action_decoder import (  # noqa: E402
    ActionDecoderConfig,
    fit_action_decoder_vae,
)
from mopa.evaluation import run_matched_episodes  # noqa: E402
from mopa.nets import ContinuousActor  # noqa: E402
from mopa.tdmpc import create_agent, load_config  # noqa: E402
from mopa.zero_s import ZeroSOpponent, zero_s_features  # noqa: E402
from tag_objectives import make_env  # noqa: E402


@pytest.fixture(scope="module")
def driver():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_tdmpc.py"
    spec = importlib.util.spec_from_file_location("run_tdmpc_zero_s_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tiny():
    rng = np.random.default_rng(72)
    states = rng.normal(size=(6, 7, 66)).astype(np.float32)
    actions = np.tanh(states[:, :-1, :2]).astype(np.float32)
    train = np.arange(3)
    fit = fit_action_decoder_vae(
        np.asarray(zero_s_features(states[:, :-1])), actions, np.full(6, 6),
        train, jax.random.PRNGKey(2),
        config=ActionDecoderConfig(action_type="continuous", hid=8, window=3, steps=2, batch=8),
    )
    opponent = ZeroSOpponent.from_fit(fit, np.tile(np.arange(3), 2), train)
    cfg = load_config(profile="smoke")
    cfg.update(opponent_mode="factored", context_dim=8)
    cfg["encoder"]["type"] = "identity"
    cfg["world_model"]["hidden_dim"] = 16
    cfg["tdmpc2"].update(population_size=8, policy_prior_samples=2, num_elites=2, mppi_iterations=1)
    mean = rng.normal(size=66).astype(np.float32)
    std = rng.uniform(0.5, 2.0, size=66).astype(np.float32)
    agent = create_agent(cfg, 66, key=jax.random.PRNGKey(3), obs_mean=mean, obs_std=std)
    return SimpleNamespace(opponent=opponent, cfg=cfg, mean=mean, std=std,
                           states=states, agent=opponent.attach(agent, mean, std))


@pytest.fixture
def saved_run(driver, tiny, tmp_path):
    run = tmp_path / "zero_s"
    run.mkdir()
    (run / "config.json").write_text(json.dumps({"world_model": tiny.cfg}))
    np.savez(run / "state_stats.npz", mean=tiny.mean, std=tiny.std)
    tiny.opponent.save(run / "opponent.msgpack")
    driver.save_agent(tiny.agent, run / "agent.msgpack")
    artifacts = ("config.json", "state_stats.npz", "opponent.msgpack", "agent.msgpack")
    manifest = {
        "source_0s_commit": "synthetic-test-fixture", "seed": 3,
        "heldout_checkpoint": 2,
        "artifacts": {name: driver.file_sha256(run / name) for name in artifacts},
    }
    (run / "manifest.json").write_text(json.dumps(manifest))
    return run


def test_saved_0s_template_restores_decoder_before_agent_weights(driver, tiny, saved_run):
    restored, manifest, stats = driver.build_template(saved_run)
    assert manifest["features"] == "markov" and manifest["state_dim"] == 66
    assert manifest["context_source"] == "zero_s"
    assert restored.red_loss_scale == 0.0
    raw, context = jnp.asarray(tiny.states[:2, 0]), jnp.asarray(tiny.opponent.prototypes[:2])
    x = restored.model.encode(raw, restored.model.encoder.params, jax.random.PRNGKey(0))
    np.testing.assert_allclose(
        restored.model.red_action(x, context, restored.model.red_model.params),
        tiny.opponent.actions(raw, context), atol=1e-6,
    )
    kwargs = dict(context=context, key=jax.random.PRNGKey(9), deterministic=True)
    expected, expected_plan = tiny.agent.act(raw, **kwargs)
    actual, actual_plan = restored.act(raw, **kwargs)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual_plan, expected_plan)
    assert actual.shape == (2, 2) and np.isfinite(actual).all()
    assert np.max(np.abs(actual)) <= 1
    for before, after in zip(jax.tree.leaves(tiny.agent), jax.tree.leaves(restored), strict=True):
        np.testing.assert_array_equal(after, before)
    stats.close()


@pytest.mark.parametrize("name", ["config.json", "state_stats.npz", "opponent.msgpack", "agent.msgpack"])
def test_saved_0s_rejects_changed_artifact(driver, saved_run, name):
    path = saved_run / name
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match=f"artifact.*{name}"):
        driver.build_template(saved_run)


def _specialist(env):
    width = max(env.observation_space(name).shape[0] for name in env.agents)
    actor = ContinuousActor(action_dim=2, hidden_dim=128)
    params = actor.init(jax.random.PRNGKey(4), jnp.zeros((1, width)))
    params["params"]["Dense_2"]["bias"] = jnp.array([0.3, -0.4])
    return params


def test_real_loop_context_is_observed_only_and_freezes_after_timeout(tiny):
    env = make_env("capture", continuous=True, max_steps=5)
    contexts = []

    def spy(state, obs, context, carry, key, t):
        del state, obs, key, t
        contexts.append(np.asarray(context))
        return jnp.broadcast_to(jnp.array([0.1, -0.1]), (2, 2)), carry

    out = run_matched_episodes(
        env, _specialist(env), spy,
        np.asarray(jax.random.split(jax.random.PRNGKey(42), 2)),
        np.asarray(jax.random.split(jax.random.PRNGKey(43), 2)),
        horizon=8, context_mode="online", label=2, zero_s=tiny.opponent,
        record_transitions=True,
    )
    trace = out["transitions"]
    assert trace["state"].shape == (2, 9, 66)
    assert trace["context"].shape == (2, 8, 8)
    assert trace["blue_action"].shape == trace["red_action"].shape == (2, 8, 2)
    np.testing.assert_array_equal(trace["context"], np.stack(contexts, axis=1))
    np.testing.assert_array_equal(trace["context"][:, 0], 0)
    offline = tiny.opponent.context(trace["state"], trace["red_action"], trace["valid_length"])
    np.testing.assert_allclose(trace["context"], offline[:, :-1], atol=2e-6)
    assert not np.allclose(trace["context"][:, 1], 0)
    np.testing.assert_array_equal(trace["valid_length"], [5, 5])
    assert trace["truncated_timeout"][:, 4].all()
    assert not trace["terminated_capture"].any()
    for ep, length in enumerate(trace["valid_length"]):
        assert trace["valid_mask"][ep, :length].all()
        assert not trace["valid_mask"][ep, length:].any()
        np.testing.assert_array_equal(trace["blue_action"][ep, length:], 0)
        np.testing.assert_array_equal(trace["red_action"][ep, length:], 0)
        np.testing.assert_allclose(trace["state"][ep, length:],
                                   np.broadcast_to(trace["state"][ep, length], trace["state"][ep, length:].shape))
        np.testing.assert_allclose(trace["context"][ep, length:],
                                   np.broadcast_to(offline[ep, length], trace["context"][ep, length:].shape), atol=2e-6)
    assert np.isfinite(trace["state"]).all() and np.isfinite(out["blue_return"]).all()
    assert np.max(np.abs(trace["red_action"])) <= 1
    assert out["blue_action_abs_max"] <= 1


def test_saved_0s_mpc_controller_steps_real_continuous_environment(driver, tiny, saved_run):
    agent, _, stats = driver.build_template(saved_run)
    env = make_env("capture", continuous=True)
    before = [np.asarray(x).copy() for x in jax.tree.leaves(agent)]
    out = run_matched_episodes(
        env, _specialist(env), driver.tdmpc_controller(agent, env, "markov"),
        np.asarray(jax.random.split(jax.random.PRNGKey(44), 1)),
        np.asarray(jax.random.split(jax.random.PRNGKey(45), 1)),
        horizon=2, context_mode="online", label=0, zero_s=tiny.opponent,
        record_transitions=True,
    )
    assert out["transitions"]["state"].shape == (1, 3, 66)
    assert out["transitions"]["context"].shape == (1, 2, 8)
    assert np.isfinite(out["blue_return"]).all() and out["blue_action_abs_max"] <= 1
    for initial, after in zip(before, jax.tree.leaves(agent), strict=True):
        np.testing.assert_array_equal(after, initial)
    stats.close()


def test_0s_cli_defaults_to_online_without_legacy_bc(driver, tiny, saved_run, monkeypatch, tmp_path):
    dataset = tmp_path / "dataset.npz"
    dataset.write_bytes(b"synthetic dataset")
    manifest = json.loads((saved_run / "manifest.json").read_text())
    manifest["dataset"] = {"sha256": driver.file_sha256(dataset)}
    manifest["specialist_checkpoints"] = [
        {"type": name, "sha256": driver.file_sha256(dataset)} for name in driver.OBJECTIVE_TYPES
    ]
    (saved_run / "manifest.json").write_text(json.dumps(manifest))
    n = len(driver.OBJECTIVE_TYPES)
    ds = SimpleNamespace(
        checkpoint_seed=np.full(n, 2), objective_label=np.arange(n),
        environment_seed=np.zeros((n, 2), np.uint32),
        step_seed=np.ones((n, 2), np.uint32), blue_action=np.zeros((n, 2, 2), np.float32),
    )

    def forbid(*args, **kwargs):
        raise AssertionError("0s must not read legacy BC/context-encoder artifacts")

    env = SimpleNamespace(good_agents=["prey"], agents=["red", "prey"],
                          observation_space=lambda name: SimpleNamespace(shape=(17,)))
    monkeypatch.setattr(driver, "load_context", forbid)
    monkeypatch.setattr(driver.CausalContextEncoder, "load", forbid)
    monkeypatch.setattr(driver, "load_continuous_dataset", lambda path: ds)
    monkeypatch.setattr(driver, "make_env", lambda *args, **kwargs: env)
    monkeypatch.setattr(driver, "tdmpc_controller", lambda *args: object())
    monkeypatch.setattr(driver, "continuous_checkpoint_path", lambda *args: dataset)
    monkeypatch.setattr(driver, "load_continuous_actor_params", lambda path: object())
    calls = []

    def episodes(env, red, blue, reset, steps, **kwargs):
        assert kwargs["context_mode"] == "online" and kwargs["encoder"] is None
        assert kwargs["zero_s"].encoder.config.lat == 8
        assert kwargs["horizon"] == 2
        calls.append(kwargs["label"])
        return {**{metric: np.zeros(len(reset), np.float32) for metric in driver.METRICS},
                "blue_action_abs_max": 0.5}

    monkeypatch.setattr(driver, "run_matched_episodes", episodes)
    missing_bc = tmp_path / "missing_bc"
    assert driver.main(["evaluate", str(saved_run), "--dataset", str(dataset),
                        "--n-eps", "1", "--bc-artifacts", str(missing_bc)]) == 0
    result = json.loads((saved_run / "closed_loop" / "evaluation.json").read_text())
    assert result["opponent_model"] == "frozen_zero_s" and result["context_dim"] == 8
    assert len(result["runs"]) == n and calls == list(range(n))
    assert not missing_bc.exists()


def _training_dataset(tiny):
    states = np.concatenate([tiny.states, tiny.states[:3]], axis=0)
    n, t = len(states), states.shape[1] - 1
    data = {"state": states, "blue_action": np.zeros((n, t, 2), np.float32),
            "red_action": np.tanh(states[:, :-1, :2]), "blue_reward": np.ones((n, t), np.float32),
            "valid_length": np.full(n, t, np.int32), "valid_mask": np.ones((n, t), bool),
            "terminated_capture": np.zeros((n, t), bool), "truncated_timeout": np.zeros((n, t), bool),
            "checkpoint_seed": np.repeat([0, 1, 2], 3), "objective_label": np.tile([0, 1, 2], 3)}
    data["truncated_timeout"][:, -1] = True
    return SimpleNamespace(**data, as_dict=lambda: data)


def test_0s_online_collector_uses_only_training_specialists_and_records_causal_replay(driver, tiny, monkeypatch, tmp_path):
    ds = _training_dataset(tiny)
    context = tiny.opponent.context(ds.state, ds.red_action, ds.valid_length)
    replay = driver.SequenceReplay.from_dataset(ds.as_dict(), np.arange(6), 3, context)
    before = replay.n_episodes
    env = make_env("capture", continuous=True, max_steps=4)
    params = _specialist(env)
    loaded = []

    def checkpoint_path(logdir, pred_type, team, ckpt):
        assert team == "pred" and ckpt in {0, 1}
        loaded.append((pred_type, ckpt))
        path = tmp_path / f"{pred_type}_{ckpt}.checkpoint"
        path.write_bytes(b"synthetic specialist")
        return path

    def controller(*args, **kwargs):
        assert kwargs["explore"]
        return lambda state, obs, ctx, carry, key, t: (jnp.zeros((len(ctx), 2)), carry)

    monkeypatch.setattr(driver, "make_env", lambda *args, **kwargs: env)
    monkeypatch.setattr(driver, "TDMPCController", controller)
    monkeypatch.setattr(driver, "continuous_checkpoint_path", checkpoint_path)
    monkeypatch.setattr(driver, "load_continuous_actor_params", lambda path: params)
    result = driver.collect_online_round(
        tiny.agent, ds, replay, None, heldout=2, features="markov", context_source="zero_s",
        mode="factored", episodes_per_group=1, logdir=tmp_path, rng=np.random.default_rng(23),
        horizon=6, zero_s=tiny.opponent, record_dir=tmp_path / "recordings",
    )
    assert set(loaded) == {(name, ckpt) for name in driver.OBJECTIVE_TYPES for ckpt in (0, 1)}
    assert replay.n_episodes == before + 6 and result["n_episodes"] == 6
    for i, group in enumerate(result["groups"]):
        recording = group["transitions"]
        assert driver.file_sha256(Path(recording["path"])) == recording["sha256"]
        with np.load(recording["path"]) as trace:
            assert trace["checkpoint_seed"][0] in {0, 1}
            assert trace["environment_seed"].shape == trace["step_seed"].shape == (1, 2)
            assert trace["replay_context"].shape == (1, 7, 8)
            np.testing.assert_allclose(trace["context"], trace["replay_context"][:, :-1], atol=2e-6)
            np.testing.assert_array_equal(replay.context[before + i], trace["replay_context"][0])


def test_0s_adaptation_updates_controller_not_opponent_and_resumes_replay(driver, tiny, saved_run, monkeypatch, tmp_path):
    ds = _training_dataset(tiny)
    dataset = tmp_path / "dataset.npz"
    dataset.write_bytes(b"synthetic dataset")
    parent = json.loads((saved_run / "manifest.json").read_text())
    parent["dataset"] = {"sha256": driver.file_sha256(dataset)}
    (saved_run / "manifest.json").write_text(json.dumps(parent))
    parent_hashes = {name: driver.file_sha256(saved_run / name) for name in parent["artifacts"]}
    # Held-out observations must never enter even the context computation.
    ds.state[ds.checkpoint_seed == 2] = np.nan
    monkeypatch.setattr(driver, "load_continuous_dataset", lambda path: ds)
    lengths_before_collection = []

    def collect(agent, data, replay, encoder, **kwargs):
        assert encoder is None and kwargs["heldout"] == 2 and kwargs["zero_s"] is not None
        lengths_before_collection.append(replay.n_episodes)
        trace = {k: v[:1] for k, v in data.as_dict().items()}
        context = kwargs["zero_s"].context(trace["state"], trace["red_action"], trace["valid_length"])
        assert np.isfinite(replay.context).all()
        replay.append(trace, context)
        directory = kwargs["record_dir"]
        directory.mkdir()
        path = directory / "training.npz"
        np.savez_compressed(path, **trace, replay_context=context)
        return {"n_transitions": 6, "n_episodes": 1,
                "groups": [{"blue_return_mean": 6.0, "checkpoint": 0,
                            "transitions": {"path": str(path), "sha256": driver.file_sha256(path)}}]}

    monkeypatch.setattr(driver, "collect_online_round", collect)
    first, second = tmp_path / "adapted", tmp_path / "resumed"
    for source, out in ((saved_run, first), (first, second)):
        assert driver.main(["adapt-0s", str(source), "--dataset", str(dataset), "--out", str(out),
                            "--rounds", "1", "--episodes-per-group", "1", "--updates-per-round", "1"]) == 0
    assert lengths_before_collection == [6, 7]
    restored, manifest, stats = driver.build_template(second)
    for name in ("config.json", "state_stats.npz", "opponent.msgpack"):
        assert driver.file_sha256(second / name) == parent_hashes[name]
    assert driver.file_sha256(second / "agent.msgpack") != parent_hashes["agent.msgpack"]
    for old, new in zip(jax.tree.leaves(tiny.agent.model.red_model.params),
                        jax.tree.leaves(restored.model.red_model.params), strict=True):
        np.testing.assert_array_equal(old, new)
    online = manifest["online_adaptation"]
    assert online["updates_completed"] == 2 and len(online["rounds"]) == 2
    assert online["online_transitions"] == 12 and online["online_fraction"] == 0.25
    assert online["training_checkpoints"] == [0, 1]
    assert manifest["parent_run"]["agent_sha256"] == driver.file_sha256(first / "agent.msgpack")
    assert {name: driver.file_sha256(saved_run / name) for name in parent["artifacts"]} == parent_hashes
    with pytest.raises(ValueError, match="new or empty"):
        driver.main(["adapt-0s", str(saved_run), "--dataset", str(dataset), "--out", str(first)])
    dataset.write_bytes(b"changed dataset")
    with pytest.raises(ValueError, match="dataset does not match"):
        driver.main(["adapt-0s", str(saved_run), "--dataset", str(dataset), "--out", str(tmp_path / "bad")])
    stats.close()
