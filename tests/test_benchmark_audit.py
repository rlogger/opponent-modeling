"""File-level matched benchmark audit, independent of training implementations."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from mopa.benchmark_report import audit_benchmark

ARMS, SEEDS, OBJECTIVES = ("implicit", "bc", "0s", "ppo", "ppo_z"), (0, 1), ("capture", "risk", "curious")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value))


def trajectory(path, keys, *, checkpoint=0, label=0, reward=1.):
    n = len(keys)
    state = np.zeros((n, 2, 66), np.float32)
    state[:, 1, 40] = 1.
    np.savez(path, environment_seed=np.asarray(keys), checkpoint_seed=np.full(n, checkpoint),
             objective_label=np.full(n, label), state=state, valid_length=np.ones(n, np.int32),
             valid_mask=np.ones((n, 1), bool), blue_reward=np.full((n, 1), reward),
             terminated_capture=np.zeros((n, 1), bool), truncated_timeout=np.ones((n, 1), bool))


@pytest.fixture
def artifacts(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("# bound source\n")
    dataset = tmp_path / "dataset.npz"
    np.savez(dataset, checkpoint_seed=[0, 1, 2], valid_mask=np.ones((3, 1), bool),
             environment_seed=[[100, 0], [100, 1], [100, 2]])
    specialists, checkpoint_rows = {}, []
    for checkpoint in (0, 1, 2):
        for objective in OBJECTIVES:
            path = tmp_path / f"specialist_{checkpoint}_{objective}"
            path.write_bytes(f"{checkpoint}_{objective}".encode())
            specialists[f"{checkpoint}:{objective}"] = {"path": str(path), "sha256": sha(path)}
            checkpoint_rows.append(dict(seed=checkpoint, objective=objective, team="pred", sha256=sha(path)))
    dataset_manifest = tmp_path / "dataset.manifest.json"
    save(dataset_manifest, {"source_checkpoints": checkpoint_rows})
    binding = dict(code={str(source): sha(source)}, dataset=sha(dataset), dataset_path=str(dataset),
                   dataset_manifest=sha(dataset_manifest), dataset_manifest_path=str(dataset_manifest), specialists=specialists)
    budget = dict(smoke=True, rounds=1, transitions_per_group=1, eval_episodes=2,
                  offline_updates=2, updates_per_round=2, batch_episodes=2)
    for seed in SEEDS:
        shared = tmp_path / f"seed_{seed}" / "shared"
        shared.mkdir(parents=True)
        for name in ("0s.msgpack", "bc.npz", "context.npz", "stats.npz"):
            (shared / name).write_bytes(f"shared_{seed}_{name}".encode())
        save(shared / "manifest.json", {**binding, "seed": seed, "smoke": True, "encoder_steps": 2,
                                        "training_checkpoints": [0, 1],
                                        "artifacts": {p.name: sha(p) for p in shared.iterdir()}})
        for arm in ARMS:
            out = tmp_path / f"seed_{seed}" / arm
            out.mkdir()
            (out / "agent.msgpack").write_bytes(f"{arm}_{seed}".encode())
            logs = []
            for checkpoint in (0, 1):
                for label, objective in enumerate(OBJECTIVES):
                    path = out / f"train_{checkpoint}_{objective}.npz"
                    trajectory(path, [[200 + seed, checkpoint * 3 + label]], checkpoint=checkpoint, label=label)
                    logs.append(dict(file=path.name, sha256=sha(path), checkpoint=checkpoint,
                                     objective=objective, valid_transitions=1))
            records, traces = [], {}
            for objective in OBJECTIVES:
                path = out / f"heldout_{objective}.npz"
                keys = [[300 + seed, i] for i in range(2)]
                trajectory(path, keys, checkpoint=2, reward=seed + 1.)
                traces[path.name] = sha(path)
                records.append(dict(arm=arm, seed=seed, objective=objective, episode_keys=keys,
                                    metrics=dict(blue_return=[seed + 1.] * 2, captured=[0, 0], resources_collected=[1, 1])))
            save(out / "evaluation.json", records)
            save(out / "manifest.json", {**binding, "arm": arm, "seed": seed, "status": "complete",
                "config": {"method": arm},
                "budget": budget, "train_checkpoints": [0, 1], "heldout_checkpoint": 2,
                "offline_transitions": 2, "online_transitions": 6,
                "shared_manifest": sha(shared / "manifest.json"), "agent_sha256": sha(out / "agent.msgpack"),
                "rounds": [dict(round=0, data=logs, online_transitions=6)],
                "evaluation_sha256": sha(out / "evaluation.json"), "evaluation_traces": traces})
    return tmp_path


def test_complete_file_audit_reconciles_counts_and_metrics(artifacts):
    result = audit_benchmark(artifacts, ARMS, SEEDS, smoke=True)
    assert result["passed"] is True and result["controllers"] == 10
    assert result["online_transitions_total"] == 60 and result["traces"] == 90
    assert result["offline_transitions_per_controller"] == 2
    json.dumps(result)


@pytest.mark.parametrize("field", ["code", "dataset", "dataset_manifest", "specialists", "budget"])
def test_mismatched_provenance_or_budget_cannot_be_aggregated(artifacts, field):
    path = artifacts / "seed_1" / "ppo_z" / "manifest.json"
    manifest = json.loads(path.read_text())
    if field == "budget":
        manifest[field]["offline_updates"] += 1
    elif isinstance(manifest[field], dict):
        manifest[field] = {}
    else:
        manifest[field] = "changed"
    save(path, manifest)
    with pytest.raises(ValueError, match="mismatch"):
        audit_benchmark(artifacts, ARMS, SEEDS, smoke=True)


@pytest.mark.parametrize("target", ["agent", "shared", "specialist", "dataset"])
def test_changed_dependency_or_checkpoint_is_rejected(artifacts, target):
    path = {"agent": artifacts / "seed_0" / "0s" / "agent.msgpack",
            "shared": artifacts / "seed_0" / "shared" / "0s.msgpack",
            "specialist": artifacts / "specialist_2_capture", "dataset": artifacts / "dataset.npz"}[target]
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        audit_benchmark(artifacts, ARMS, SEEDS, smoke=True)


def test_rehashed_evaluation_still_must_match_trace_rewards(artifacts):
    out = artifacts / "seed_1" / "bc"
    records = json.loads((out / "evaluation.json").read_text())
    records[0]["metrics"]["blue_return"][0] += 5
    save(out / "evaluation.json", records)
    manifest = json.loads((out / "manifest.json").read_text())
    manifest["evaluation_sha256"] = sha(out / "evaluation.json")
    save(out / "manifest.json", manifest)
    with pytest.raises(ValueError, match="metrics differ"):
        audit_benchmark(artifacts, ARMS, SEEDS, smoke=True)


def test_missing_group_quota_and_smoke_mislabel_are_rejected(artifacts):
    with pytest.raises(ValueError, match="training seeds"):
        audit_benchmark(artifacts, ARMS, SEEDS)
    with pytest.raises(ValueError, match="smoke/scientific"):
        audit_benchmark(artifacts, ARMS, (0, 1, 2))
    path = artifacts / "seed_0" / "implicit" / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["rounds"][0]["data"].pop()
    save(path, manifest)
    with pytest.raises(ValueError, match="group interaction quota"):
        audit_benchmark(artifacts, ARMS, SEEDS, smoke=True)
