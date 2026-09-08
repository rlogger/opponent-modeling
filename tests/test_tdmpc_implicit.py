"""Equation 1 driver defaults and independence from opponent-context artifacts."""
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


@pytest.fixture(scope="module")
def driver():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_tdmpc.py"
    spec = importlib.util.spec_from_file_location("run_tdmpc_implicit_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _forbid_encoder_load(*args, **kwargs):
    raise AssertionError("Equation 1 must not load an opponent-context encoder")


def test_training_defaults_to_equation_one(driver):
    args = driver.build_parser().parse_args(["train"])
    assert args.mode == "implicit"
    assert args.context_source is None  # Resolved to "none" for Equation 1.
    assert args.profile == "equation1"
    assert args.features == "relative"


def test_zero_context_does_not_read_bc_artifacts(driver, monkeypatch, tmp_path):
    monkeypatch.setattr(driver.CausalContextEncoder, "load", _forbid_encoder_load)
    data = {"state": np.zeros((2, 4, 66), dtype=np.float32)}
    ds = SimpleNamespace(as_dict=lambda: data)
    artifacts = tmp_path / "missing_bc_artifacts"

    encoder, path, context = driver.load_context(
        ds, artifacts, heldout=2, bc_seed=0, source="zero"
    )

    assert encoder is None
    assert path is None
    assert context.shape == (2, 4, driver.CONTEXT_DIM)
    assert context.dtype == np.float32
    np.testing.assert_array_equal(context, np.zeros_like(context))
    assert not artifacts.exists()


def test_implicit_training_does_not_require_bc_artifacts(driver, monkeypatch, tmp_path):
    """Exercise training setup and manifest creation without compiling a model."""
    monkeypatch.setattr(driver.CausalContextEncoder, "load", _forbid_encoder_load)
    data = {"state": np.zeros((2, 4, 66), dtype=np.float32)}
    ds = SimpleNamespace(
        as_dict=lambda: data,
        state=data["state"],
        checkpoint_seed=np.asarray([0, 2]),
        valid_mask=np.ones((2, 3), dtype=bool),
    )
    monkeypatch.setattr(driver, "load_continuous_dataset", lambda path: ds)
    agent = SimpleNamespace(horizon=3, batch_size=1)
    metric_names = (
        "total_loss", "consistency_loss", "reward_loss", "value_loss",
        "continue_loss", "red_loss", "policy_loss",
    )
    agent.update = lambda **batch: (agent, dict.fromkeys(metric_names, 0.1))

    def create_agent(cfg, state_dim, **kwargs):
        assert cfg["opponent_mode"] == "implicit"
        assert cfg["context_dim"] == 0
        return agent

    def make_replay(data, episodes, horizon, context, **kwargs):
        assert context.shape[-1] == 0
        return SimpleNamespace(
            n_transitions=3, n_episodes=1, sample=lambda rng, batch_size: {}
        )

    monkeypatch.setattr(driver, "create_agent", create_agent)
    monkeypatch.setattr(driver.SequenceReplay, "from_dataset", make_replay)
    monkeypatch.setattr(driver, "save_agent", lambda agent, path: path.write_bytes(b"mock agent"))
    monkeypatch.setattr(driver, "multistep_model_error", lambda *a, **kw: {"per_horizon": {}})
    monkeypatch.setattr(driver, "reward_calibration", lambda *a, **kw: {})
    monkeypatch.setattr(driver, "termination_calibration", lambda *a, **kw: None)
    dataset = tmp_path / "dataset.npz"
    dataset.write_bytes(b"mock dataset")
    artifacts = tmp_path / "missing_bc_artifacts"
    out = tmp_path / "runs"
    assert driver.main([
        "train", "--dataset", str(dataset), "--out", str(out),
        "--bc-artifacts", str(artifacts), "--updates", "1", "--log-every", "1",
    ]) == 0
    manifests = list(out.glob("*/manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text())
    assert manifest["mode"] == "implicit"
    assert manifest["context_source"] == "none"
    assert manifest["config"]["context_dim"] == 0
    assert manifest["context_encoder"] is None
    assert not artifacts.exists()


def test_implicit_evaluation_does_not_require_bc_artifacts(driver, monkeypatch, tmp_path):
    """Exercise evaluation setup/output with cheap stand-ins for environment rollouts."""
    monkeypatch.setattr(driver.CausalContextEncoder, "load", _forbid_encoder_load)
    n = len(driver.OBJECTIVE_TYPES)
    data = {"state": np.zeros((n, 3, 66), dtype=np.float32)}
    ds = SimpleNamespace(
        as_dict=lambda: data,
        checkpoint_seed=np.full(n, 2),
        objective_label=np.arange(n),
        environment_seed=np.zeros((n, 2), dtype=np.uint32),
        step_seed=np.ones((n, 2), dtype=np.uint32),
        blue_action=np.zeros((n, 2, 2), dtype=np.float32),
    )
    manifest = {
        "mode": "implicit",
        "encoder": "mlp",
        "features": "relative",
        "heldout_checkpoint": 2,
        "seed": 0,
        "context_source": "none",
        "context_encoder": None,
        "config": {
            "context_dim": 0,
            "tdmpc2": {
                "horizon": 3,
                "population_size": 512,
                "policy_prior_samples": 24,
                "num_elites": 64,
                "mppi_iterations": 6,
            },
        },
    }
    run = tmp_path / "implicit_run"
    run.mkdir()
    (run / "agent.msgpack").write_bytes(b"mock agent")
    (run / "manifest.json").write_text(json.dumps(manifest))
    env = SimpleNamespace(
        good_agents=["prey"],
        agents=["pred", "prey"],
        observation_space=lambda name: SimpleNamespace(shape=(2,)),
    )
    monkeypatch.setattr(driver, "build_template", lambda path: (object(), manifest, {}))
    monkeypatch.setattr(driver, "load_continuous_dataset", lambda path: ds)
    monkeypatch.setattr(driver, "make_env", lambda *args, **kwargs: env)
    monkeypatch.setattr(driver, "tdmpc_controller", lambda *args: object())
    monkeypatch.setattr(driver, "load_continuous_actor_params", lambda path: object())
    monkeypatch.setattr(driver, "factored_invariance_checks", lambda *args: None)
    calls = []

    def fake_episodes(env, red, blue, reset_keys, step_keys, **kwargs):
        assert kwargs["context_mode"] == "zero"
        assert kwargs["encoder"] is None
        calls.append(kwargs["label"])
        return {
            **{metric: np.zeros(len(reset_keys), dtype=np.float32) for metric in driver.METRICS},
            "blue_action_abs_max": 0.0,
        }

    monkeypatch.setattr(driver, "run_matched_episodes", fake_episodes)
    artifacts = tmp_path / "missing_bc_artifacts"
    assert driver.main([
        "evaluate", str(run), "--n-eps", "1", "--bc-artifacts", str(artifacts)
    ]) == 0
    result = json.loads((run / "evaluation.json").read_text())
    assert result["mode"] == "implicit"
    assert len(result["runs"]) == n
    assert calls == list(range(n))
    assert not artifacts.exists()


def test_implicit_online_collection_needs_no_encoder(driver, monkeypatch, tmp_path):
    monkeypatch.setattr(driver.CausalContextEncoder, "load", _forbid_encoder_load)
    monkeypatch.setattr(driver, "make_env", lambda *a, **kw: object())
    monkeypatch.setattr(driver, "TDMPCController", lambda *a, **kw: object())
    monkeypatch.setattr(
        driver, "continuous_checkpoint_path",
        lambda logdir, objective, team, seed: (objective, team, seed),
    )
    loaded, appended = [], []

    def load_actor(path):
        loaded.append(path)
        return object()

    def fake_episodes(env, red, blue, reset_keys, step_keys, **kwargs):
        assert kwargs["context_mode"] == "zero"
        assert kwargs["encoder"] is None
        assert kwargs["record_transitions"] is True
        return {
            "transitions": {"valid_length": np.asarray([2], dtype=np.int32)},
            "blue_return": np.zeros(1),
            "captured": np.zeros(1),
            "resources_collected": np.zeros(1),
        }

    def append(transitions, context, *, feature_map):
        assert context.shape == (1, 3, 0)
        assert feature_map == "relative"
        appended.append(transitions)

    monkeypatch.setattr(driver, "load_continuous_actor_params", load_actor)
    monkeypatch.setattr(driver, "run_matched_episodes", fake_episodes)
    replay = SimpleNamespace(context=np.zeros((1, 3, 0), np.float32), append=append)
    summary = driver.collect_online_round(
        object(), SimpleNamespace(checkpoint_seed=np.asarray([0, 1, 2])), replay, None,
        heldout=2, features="relative", context_source="none", mode="implicit",
        episodes_per_group=1, logdir=tmp_path / "logs",
        rng=np.random.default_rng(0), horizon=2,
    )
    expected = [(objective, "pred", seed) for seed in (0, 1) for objective in driver.OBJECTIVE_TYPES]
    assert loaded == expected
    assert len(appended) == len(expected)
    assert summary["n_episodes"] == len(expected)
    assert summary["n_transitions"] == 2 * len(expected)
