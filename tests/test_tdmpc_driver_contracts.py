"""Reject confounded fit aggregation and mislabeled replay collection."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture(scope="module")
def driver():
    path = Path(__file__).resolve().parents[1] / "scripts/run_tdmpc.py"
    spec = importlib.util.spec_from_file_location("tdmpc_driver_contracts", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bound_run(driver, root, mode="implicit", seed=0):
    root.mkdir()
    (root / "agent.msgpack").write_bytes(b"weights")
    (root / "state_stats.npz").write_bytes(b"normalization")
    manifest = {"mode": mode, "encoder": "identity", "seed": seed,
                "dataset": {"sha256": "dataset"}, "features": "markov",
                "heldout_checkpoint": 2, "updates": 10, "git_sha": "code", "profile": "smoke",
                "config": {"opponent_mode": mode, "context_dim": 0 if mode == "implicit" else 3,
                           "tdmpc2": {"discount": .99}},
                "state_stats_sha256": driver.file_sha256(root / "state_stats.npz"),
                "context_encoder": {"sha256": "encoder"}, "context_source": "causal",
                "specialist_checkpoints": [{"objective": kind, "team": "pred", "seed": 2, "sha256": kind}
                                           for kind in driver.OBJECTIVE_TYPES],
                "executable_source": {"training.py": "hash"}, "final_replay_transitions": 100}
    (root / "manifest.json").write_text(json.dumps(manifest))
    evaluation = {"mode": mode, "encoder": "identity", "seed": seed, "planner": {"horizon": 3},
                  "agent_sha256": driver.file_sha256(root / "agent.msgpack"),
                  "manifest_sha256": driver.file_sha256(root / "manifest.json"), "runs": [],
                  "executable_source": {"evaluation.py": "hash"}, "context_modes": ["zero"],
                  "specialist_checkpoints": [{"type": kind, "seed": 2, "sha256": kind}
                                             for kind in driver.OBJECTIVE_TYPES]}
    traces = {}
    for kind in driver.OBJECTIVE_TYPES:
        traces[f"{kind}__reset_keys"] = np.asarray([[1, 2], [3, 4]])
        traces[f"{kind}__step_keys"] = np.asarray([[5, 6], [7, 8]])
        row = {"opponent": kind, "controller": "tdmpc", "context_mode": "zero", "n_episodes": 2}
        for metric in driver.METRICS:
            traces[f"{kind}__tdmpc__zero__{metric}"] = np.asarray([1., 3.])
            row[metric] = {"mean": 2.}
        evaluation["runs"].append(row)
    np.savez(root / "evaluation_per_episode.npz", **traces)
    evaluation["per_episode_sha256"] = driver.file_sha256(root / "evaluation_per_episode.npz")
    return root, evaluation, manifest, traces


def validate(driver, records):
    driver.validate_comparison_inputs(*zip(*records))


def test_matched_modes_accept_but_copied_fit_is_not_new_seed(driver, tmp_path):
    one = bound_run(driver, tmp_path / "one")
    other_mode = bound_run(driver, tmp_path / "other", mode="factored")
    validate(driver, [one, other_mode])
    copied = bound_run(driver, tmp_path / "copy")
    with pytest.raises(ValueError, match="duplicate independent-fit"):
        validate(driver, [one, copied])


@pytest.mark.parametrize("corruption", ["features", "budget", "reset", "raw_metric", "agent", "manifest", "missing_reset"])
def test_comparison_rejects_incompatible_or_unbound_evidence(driver, tmp_path, corruption):
    one = bound_run(driver, tmp_path / "one")
    two = bound_run(driver, tmp_path / "two", mode="factored")
    path, evaluation, manifest, traces = two
    if corruption == "features":
        manifest["features"] = "relative"
    elif corruption == "budget":
        manifest["updates"] += 1
    elif corruption == "reset":
        traces["capture__reset_keys"][0, 0] += 1
    elif corruption == "raw_metric":
        traces["capture__tdmpc__zero__blue_return"][0] += 1
    elif corruption == "missing_reset":
        del traces["capture__reset_keys"]
    elif corruption == "agent":
        (path / "agent.msgpack").write_bytes(b"tampered")
    else:
        (path / "manifest.json").write_text("{}")
    with pytest.raises(ValueError):
        validate(driver, [one, two])


@pytest.mark.parametrize("corruption", ["discount", "normalization", "specialist", "actual_budget",
                                      "empty", "missing_objective", "duplicate_reset", "unpaired_seed",
                                      "trace_file", "eval_source", "context", "missing_specialists", "missing_source"])
def test_independent_review_counterexamples_are_rejected(driver, tmp_path, corruption):
    one = bound_run(driver, tmp_path / "one", mode="conditioned")
    two = bound_run(driver, tmp_path / "two", mode="factored")
    path, ev, manifest, traces = two
    if corruption == "discount":
        manifest["config"]["tdmpc2"]["discount"] = .5
    elif corruption == "normalization":
        (path / "state_stats.npz").write_bytes(b"changed")
    elif corruption == "specialist":
        manifest["specialist_checkpoints"][0]["sha256"] = "different"
    elif corruption == "actual_budget":
        manifest["final_replay_transitions"] = 1
    elif corruption == "empty":
        ev["runs"] = []
    elif corruption == "missing_objective":
        ev["runs"].pop()
    elif corruption == "duplicate_reset":
        traces["capture__reset_keys"][1] = traces["capture__reset_keys"][0]
    elif corruption == "unpaired_seed":
        ev["seed"] = manifest["seed"] = 2
    elif corruption == "trace_file":
        (path / "evaluation_per_episode.npz").write_bytes(b"changed")
    elif corruption == "missing_specialists":
        del ev["specialist_checkpoints"]
    elif corruption == "missing_source":
        manifest["executable_source"] = {}
    elif corruption == "eval_source":
        ev["executable_source"]["evaluation.py"] = "changed"
    else:
        manifest["context_encoder"]["sha256"] = "different"
    # Bind mutations so validation must inspect semantics, not merely stale JSON.
    (path / "manifest.json").write_text(json.dumps(manifest))
    ev["manifest_sha256"] = driver.file_sha256(path / "manifest.json")
    with pytest.raises(ValueError):
        validate(driver, [one, two])


@pytest.mark.parametrize("field,value", [("world_gradients_finite", False),
                                       ("policy_gradients_finite", False), ("total_loss", np.nan)])
def test_each_update_rejects_nonfinite_before_optimizer_masking(driver, field, value):
    info = dict(world_gradients_finite=True, policy_gradients_finite=True, total_loss=1.)
    info[field] = value
    with pytest.raises(FloatingPointError):
        driver.check_update(info, 2)


def test_specialist_replacement_rejected_before_model_loading(driver, tmp_path, monkeypatch):
    checkpoint = tmp_path / "specialist"
    checkpoint.write_bytes(b"original")
    dataset = tmp_path / "dataset.npz"
    sidecar = dataset.with_suffix(".manifest.json")
    sidecar.write_text(json.dumps({"source_checkpoints": [{"objective": "capture", "team": "pred",
        "seed": 0, "sha256": driver.file_sha256(checkpoint)}]}))
    monkeypatch.setattr(driver, "continuous_checkpoint_path", lambda *args: checkpoint)
    driver.checkpoint_bindings(dataset, tmp_path)
    checkpoint.write_bytes(b"replacement")
    with pytest.raises(ValueError, match="specialist differs"):
        driver.checkpoint_bindings(dataset, tmp_path)


@pytest.mark.parametrize("source,expected", [("zero", "zero"), ("none", "zero"), ("oracle", "oracle")])
def test_collection_uses_requested_intervention_and_stores_it(driver, monkeypatch, tmp_path, source, expected):
    monkeypatch.setattr(driver, "make_env", lambda *a, **k: object())
    monkeypatch.setattr(driver, "TDMPCController", lambda *a, **k: object())
    monkeypatch.setattr(driver, "load_continuous_actor_params", lambda *a: object())
    observed = []

    def episodes(*args, **kwargs):
        assert kwargs["context_mode"] == expected
        assert kwargs["encoder"] is None
        context = (np.eye(driver.CONTEXT_DIM)[kwargs["label"]] if expected == "oracle"
                   else np.zeros(driver.CONTEXT_DIM))
        observed.append(context)
        return {"transitions": {"context": np.broadcast_to(context, (1, 2, driver.CONTEXT_DIM)),
                                "final_context": context[None], "valid_length": np.asarray([2])},
                "blue_return": np.zeros(1), "captured": np.zeros(1), "resources_collected": np.zeros(1)}

    stored = []
    monkeypatch.setattr(driver, "run_matched_episodes", episodes)
    replay = SimpleNamespace(context=np.zeros((1, 3, driver.CONTEXT_DIM)),
                             append=lambda tr, ctx, **kw: stored.append(ctx))
    driver.collect_online_round(object(), SimpleNamespace(checkpoint_seed=np.asarray([0, 2])),
        replay, object(), heldout=2, features="markov", context_source=source, mode="factored",
        episodes_per_group=1, logdir=tmp_path, rng=np.random.default_rng(3), horizon=2)
    for actual, saved in zip(observed, stored):
        np.testing.assert_array_equal(saved, np.broadcast_to(actual, saved.shape))


@pytest.mark.parametrize("artifact", ["agent.msgpack", "state_stats.npz"])
def test_generic_restore_rejects_changed_artifact_before_deserialization(driver, tmp_path, artifact):
    for name in ("agent.msgpack", "state_stats.npz"):
        (tmp_path / name).write_bytes(b"original")
    (tmp_path / "manifest.json").write_text(json.dumps({"schema_version": 2,
        "agent_sha256": driver.file_sha256(tmp_path / "agent.msgpack"),
        "state_stats_sha256": driver.file_sha256(tmp_path / "state_stats.npz")}))
    (tmp_path / artifact).write_bytes(b"changed")
    with pytest.raises(ValueError, match="manifest"):
        driver.build_template(tmp_path)
