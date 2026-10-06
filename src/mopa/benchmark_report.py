"""Matched controller comparisons with training-seed, not episode, uncertainty.

Each record is ``{arm, seed, objective, episode_keys, metrics}``; ``metrics``
contains one-dimensional ``blue_return``, ``captured`` and
``resources_collected`` arrays. Reset keys must be ordered identically across
all arms and objectives within a training seed. Seeds may use different keys,
but every record must have the same number of episodes.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

METRICS = ("blue_return", "captured", "resources_collected")
DEFAULT_COMPARISONS = (
    ("0s", "bc"), ("0s", "implicit"), ("0s", "ppo_z"),
    ("bc", "ppo"), ("implicit", "ppo"),
)


def audit_benchmark(root, expected_arms, expected_seeds, *, smoke=False):
    """Read-only provenance/quota/trace audit before publishing an aggregate.

    This checks recorded artifacts, not simulator replay or causal-encoder
    recomputation. The caller must separately test those numerical contracts.
    """
    root, repository = Path(root), Path(__file__).resolve().parents[2]
    if not smoke and set(expected_seeds) != {0, 1, 2}:
        raise ValueError("scientific benchmark requires training seeds 0, 1, 2")
    digests, reference, shared_by_seed, configs_by_arm = {}, None, {}, {}
    train_keys, test_keys, result_records = set(), set(), []
    total_online, trace_count = 0, 0

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    def verify(path, expected):
        path = Path(path).resolve()
        if path not in digests:
            digests[path] = hashlib.sha256(path.read_bytes()).hexdigest()
        require(digests[path] == expected, f"artifact hash mismatch: {path}")

    def read(path):
        return json.loads(Path(path).read_text())

    def key_set(keys):
        return {tuple(k) for k in np.asarray(keys).tolist()}

    def trace(path, expected):
        nonlocal trace_count
        verify(path, expected)
        with np.load(path, allow_pickle=False) as source:
            tr = dict(source)
        valid, lengths = np.asarray(tr["valid_mask"], bool), np.asarray(tr["valid_length"])
        require(valid.ndim == 2 and lengths.shape == (len(valid),), "trace validity shapes differ")
        require(np.all((lengths >= 0) & (lengths <= valid.shape[1])), "invalid trace lengths")
        require(np.array_equal(valid, np.arange(valid.shape[1])[None] < lengths[:, None]), "non-prefix valid trace")
        term, trunc = tr["terminated_capture"], tr["truncated_timeout"]
        require(term.shape == trunc.shape == valid.shape, "trace termination shapes differ")
        require(not np.any(term & trunc) and not np.any((term | trunc) & ~valid), "invalid terminal flags")
        expected_done = valid & (np.arange(valid.shape[1])[None] == lengths[:, None] - 1)
        require(np.array_equal(term | trunc, expected_done), "termination must mark last valid transition")
        require(tr["blue_reward"].shape == valid.shape and np.isfinite(tr["blue_reward"][valid]).all(), "invalid trace rewards")
        require(tr["environment_seed"].shape == (len(valid), 2), "invalid trace reset keys")
        require(tr["state"].shape == (len(valid), valid.shape[1] + 1, 66), "invalid trace state shape")
        trace_count += 1
        return tr, valid, lengths

    for seed in expected_seeds:
        shared_path = root / f"seed_{seed}" / "shared" / "manifest.json"
        shared = read(shared_path)
        for name, sha in shared["artifacts"].items():
            verify(shared_path.parent / name, sha)
        require(shared["seed"] == seed and shared["training_checkpoints"] == [0, 1], "shared training split changed")
        for arm in expected_arms:
            out = root / f"seed_{seed}" / arm
            manifest = read(out / "manifest.json")
            require(manifest["arm"] == arm and manifest["seed"] == seed and manifest["status"] == "complete", "incomplete or mislabeled controller")
            configuration = {k: v for k, v in manifest["config"].items() if k != "seed"}
            require(configuration == configs_by_arm.setdefault(arm, configuration), "per-arm model configuration mismatch across seeds")
            require(manifest["train_checkpoints"] == [0, 1] and manifest["heldout_checkpoint"] == 2, "training/held-out split changed")
            budget = manifest["budget"]
            require(budget["smoke"] is smoke, "mixed smoke/scientific benchmark")
            if not smoke:
                fixed = dict(offline_updates=2000, rounds=6, updates_per_round=1000,
                             transitions_per_group=600, batch_episodes=8, eval_episodes=24)
                require(all(budget[k] == v for k, v in fixed.items()), "scientific budget differs from predeclared protocol")
                require(shared["encoder_steps"] == 1500, "scientific opponent fitting budget changed")
                require(manifest["dataset"] == "928181027a9e5e86b106190b1487cda90c09e2cc5df08b569f7b93ba8350b5f5", "scientific dataset differs from protocol")
                fixed_sources = {
                    "src/tag_objectives/objectives.py": "71332590fc61490bfd6271ab72da01f41df952963c10d86b577035ac74796f84",
                    "src/tag_objectives/resources.py": "72910059f63c01613b4ad3803236497723724e3c16525d9b21ca51b920b0d4d4",
                }
                require(all(manifest["code"].get(k) == v for k, v in fixed_sources.items()), "scientific environment differs from main-source pin")
            binding = {k: manifest[k] for k in ("code", "dataset", "dataset_manifest", "specialists", "budget")}
            if reference is None:
                reference = binding
                require(bool(manifest["code"]), "missing source hashes")
                for name, sha in manifest["code"].items():
                    verify(repository / name, sha)
                verify(manifest["dataset_path"], manifest["dataset"])
                verify(manifest["dataset_manifest_path"], manifest["dataset_manifest"])
                provenance = read(manifest["dataset_manifest_path"])
                source_pred = {f"{r['seed']}:{r['objective']}": r["sha256"]
                               for r in provenance["source_checkpoints"] if r["team"] == "pred"}
                required_sources = {f"{c}:{o}" for c in (0, 1, 2) for o in ("capture", "risk", "curious")}
                require(set(manifest["specialists"]) == required_sources, "missing frozen specialist checkpoints")
                for key, checkpoint in manifest["specialists"].items():
                    require(source_pred.get(key) == checkpoint["sha256"], "specialist differs from dataset provenance")
                    verify(checkpoint["path"], checkpoint["sha256"])
                with np.load(manifest["dataset_path"], allow_pickle=False) as data:
                    training = np.isin(data["checkpoint_seed"], [0, 1])
                    require(set(np.unique(data["checkpoint_seed"])) == {0, 1, 2}, "invalid offline checkpoint split")
                    offline_count = int(data["valid_mask"][training].sum())
                    train_keys.update(key_set(data["environment_seed"][training]))
                require(smoke or offline_count == 86_899, "scientific offline budget changed")
            require(binding == reference, "source/data/specialist/budget mismatch across controllers")
            require(manifest["offline_transitions"] == offline_count, "offline transition count mismatch")
            require(all(shared[k] == manifest[k] for k in ("code", "dataset", "dataset_manifest", "specialists")), "shared artifacts have different provenance")
            require(shared["smoke"] is smoke, "shared artifacts have different smoke mode")
            verify(shared_path, manifest["shared_manifest"])
            shared_by_seed[seed] = manifest["shared_manifest"]
            verify(out / "agent.msgpack", manifest["agent_sha256"])
            require(len(manifest["rounds"]) == budget["rounds"], "incomplete training rounds")
            running, seen_recordings = 0, set()
            for index, row in enumerate(manifest["rounds"]):
                require(row["round"] == index, "unordered training rounds")
                counts = {(c, o): 0 for c in (0, 1) for o in ("capture", "risk", "curious")}
                for recording in row["data"]:
                    recording_path = (out / recording["file"]).resolve()
                    require(recording_path not in seen_recordings, "duplicate training recording")
                    seen_recordings.add(recording_path)
                    group = (recording["checkpoint"], recording["objective"])
                    require(group in counts, "held-out or unknown opponent in training recordings")
                    tr, valid, lengths = trace(recording_path, recording["sha256"])
                    require(np.all(tr["checkpoint_seed"] == group[0]), "training checkpoint labels disagree")
                    require(np.all(tr["objective_label"] == ("capture", "risk", "curious").index(group[1])), "training objective labels disagree")
                    require(int(valid.sum()) == recording["valid_transitions"], "recorded transition count mismatch")
                    counts[group] += int(valid.sum())
                    train_keys.update(key_set(tr["environment_seed"][lengths > 0]))
                require(all(n == budget["transitions_per_group"] for n in counts.values()), "unequal group interaction quota")
                running += sum(counts.values())
                require(row["online_transitions"] == running, "round interaction total mismatch")
            require(manifest["online_transitions"] == running, "controller interaction total mismatch")
            total_online += running
            verify(out / "evaluation.json", manifest["evaluation_sha256"])
            evaluations = read(out / "evaluation.json")
            require({r["objective"] for r in evaluations} == {"capture", "risk", "curious"} and len(evaluations) == 3, "incomplete objective evaluation")
            require(set(manifest["evaluation_traces"]) == {f"heldout_{o}.npz" for o in ("capture", "risk", "curious")}, "incomplete held-out recordings")
            for record in evaluations:
                require(record["arm"] == arm and record["seed"] == seed, "evaluation identity mismatch")
                name = f"heldout_{record['objective']}.npz"
                tr, valid, lengths = trace(out / name, manifest["evaluation_traces"][name])
                require(len(lengths) == budget["eval_episodes"] and np.all(lengths > 0), "held-out episode count mismatch")
                require(np.array_equal(tr["environment_seed"], record["episode_keys"]), "evaluation reset keys mismatch")
                actual = {"blue_return": np.where(valid, tr["blue_reward"], 0).sum(axis=1, dtype=np.float64),
                          "captured": tr["terminated_capture"].any(axis=1).astype(float),
                          "resources_collected": tr["state"][np.arange(len(lengths)), lengths, 40:56].sum(axis=1)}
                require(all(np.allclose(actual[k], record["metrics"][k], atol=2e-5, rtol=1e-6) for k in METRICS), "evaluation metrics differ from recorded transitions")
                test_keys.update(key_set(tr["environment_seed"]))
            result_records.extend(evaluations)
    require(not train_keys & test_keys, "training and evaluation reset keys overlap")
    # Also validate ordered matching and complete arm/seed/objective coverage.
    summarize_benchmark(result_records, expected_arms, expected_seeds, comparisons=[], bootstrap_samples=2)
    return {"passed": True, "controllers": len(expected_arms) * len(expected_seeds),
            "verified_files": len(digests), "traces": trace_count, "online_transitions_total": total_online,
            "offline_transitions_per_controller": offline_count, "shared_manifest_by_seed": shared_by_seed,
            "scope": "artifact identity, matched provenance/budgets, trace flags and metric reconciliation; not simulator or context replay"}


def summarize_benchmark(
    records: Sequence[Mapping[str, Any]],
    expected_arms: Sequence[str],
    expected_seeds: Sequence[int],
    *,
    expected_objectives: Sequence[str] = ("capture", "risk", "curious"),
    comparisons: Sequence[tuple[str, str]] | None = None,
    bootstrap_seed: int = 0,
    bootstrap_samples: int = 10_000,
) -> dict[str, Any]:
    """Validate a complete matched design and return JSON-serializable results.

    Overall results weight objectives equally. SD is sample SD across training
    seed means. Percentile bootstrap intervals resample training seeds, then
    paired reset clusters within each selected seed. One reset draw is shared
    across objectives, arms, and metrics, preserving their dependencies.
    With only three seeds, these intervals are descriptive and unstable.
    """
    arms, seeds = list(expected_arms), list(expected_seeds)
    objective_order = sorted(expected_objectives)
    if not arms or any(not isinstance(a, str) or not a for a in arms) or len(set(arms)) != len(arms):
        raise ValueError("expected_arms must contain distinct nonempty names")
    if len(seeds) < 2 or any(type(s) is not int for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("expected_seeds must contain at least two distinct integers")
    if not objective_order or any(not isinstance(o, str) or not o for o in objective_order) or len(set(objective_order)) != len(objective_order):
        raise ValueError("expected_objectives must contain distinct nonempty names")
    if type(bootstrap_samples) is not int or bootstrap_samples < 2:
        raise ValueError("bootstrap_samples must be an integer >= 2")
    pairs = list(DEFAULT_COMPARISONS if comparisons is None else comparisons)
    if len(set(pairs)) != len(pairs) or any(a not in arms or b not in arms or a == b for a, b in pairs):
        raise ValueError("comparisons must be distinct pairs of expected arms")
    if not records:
        raise ValueError("no benchmark records")

    table: dict[tuple[str, int, str], dict[str, np.ndarray]] = {}
    keys_by_seed: dict[int, list[str]] = {}
    step_keys_by_seed: dict[int, np.ndarray] = {}
    has_step_keys = ["step_keys" in record for record in records]
    if any(has_step_keys) and not all(has_step_keys):
        raise ValueError("simulator step keys must be supplied for every record or none")
    n_episodes: int | None = None
    for record in records:
        arm, seed, objective = record["arm"], record["seed"], record["objective"]
        if arm not in arms or type(seed) is not int or seed not in seeds:
            raise ValueError(f"unexpected arm or training seed: {arm!r}, {seed!r}")
        if objective not in objective_order:
            raise ValueError(f"unexpected objective: {objective!r}")
        identity = (arm, seed, objective)
        if identity in table:
            raise ValueError(f"duplicate record: {identity}")
        raw_keys = np.asarray(record["episode_keys"])
        if raw_keys.ndim not in (1, 2) or len(raw_keys) == 0 or (raw_keys.ndim == 2 and raw_keys.shape[1] == 0):
            raise ValueError("episode_keys must be a nonempty vector or matrix")
        try:
            keys = [json.dumps(k, sort_keys=True, allow_nan=False) for k in raw_keys.tolist()]
        except (TypeError, ValueError) as error:
            raise ValueError("episode keys must be finite JSON values") from error
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate reset keys within a record")
        if n_episodes is None:
            n_episodes = len(keys)
        if len(keys) != n_episodes:
            raise ValueError("episode counts differ between records")
        if seed in keys_by_seed and keys != keys_by_seed[seed]:
            raise ValueError(f"ordered reset keys differ within training seed {seed}")
        keys_by_seed[seed] = keys
        if all(has_step_keys):
            step_keys = np.asarray(record["step_keys"])
            if step_keys.shape != (n_episodes, 2) or step_keys.dtype.kind not in "iu" or np.any(step_keys < 0):
                raise ValueError("step_keys must contain one unsigned RNG key pair per episode")
            if seed in step_keys_by_seed and not np.array_equal(step_keys, step_keys_by_seed[seed]):
                raise ValueError(f"ordered simulator step keys differ within training seed {seed}")
            step_keys_by_seed[seed] = step_keys
        values = {}
        for name in METRICS:
            if name not in record["metrics"]:
                raise ValueError(f"missing metric {name} in {identity}")
            value = np.asarray(record["metrics"][name], dtype=np.float64)
            if value.shape != (n_episodes,) or not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite with one value per episode")
            if name == "captured" and not np.isin(value, (0, 1)).all():
                raise ValueError("captured must contain binary episode outcomes")
            if name == "resources_collected" and ((value < 0).any() or (value != np.floor(value)).any()):
                raise ValueError("resources_collected must contain nonnegative counts")
            values[name] = value
        table[identity] = values
    required = {(a, s, o) for a in arms for s in seeds for o in objective_order}
    missing = required - table.keys()
    if missing:
        raise ValueError(f"incomplete benchmark; missing records: {sorted(missing)}")

    # Shared draws keep every paired contrast and metric comparison matched.
    rng = np.random.default_rng(bootstrap_seed)
    seed_draw = rng.integers(len(seeds), size=(bootstrap_samples, len(seeds)))
    reset_draw = rng.integers(n_episodes, size=(bootstrap_samples, len(seeds), n_episodes))

    def summary(values: np.ndarray) -> dict[str, Any]:
        # values: training seed x objective x matched reset.
        seed_means = values.mean(axis=-1)
        boot = np.empty((bootstrap_samples, len(objective_order)))
        for start in range(0, bootstrap_samples, 128):
            stop = min(start + 128, bootstrap_samples)
            selected = values[seed_draw[start:stop]]
            paired = np.take_along_axis(selected, reset_draw[start:stop, :, None, :], axis=-1)
            boot[start:stop] = paired.mean(axis=(1, 3))

        def entry(per_seed: np.ndarray, draws: np.ndarray) -> dict[str, Any]:
            return {
                "mean": float(per_seed.mean()),
                "seed_std": float(per_seed.std(ddof=1)),
                "seed_means": {str(s): float(v) for s, v in zip(seeds, per_seed, strict=True)},
                "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
            }

        return {
            "overall": entry(seed_means.mean(axis=1), boot.mean(axis=1)),
            "by_objective": {o: entry(seed_means[:, i], boot[:, i]) for i, o in enumerate(objective_order)},
        }

    arrays = {
        arm: {metric: np.asarray([[table[arm, seed, o][metric] for o in objective_order]
                                  for seed in seeds]) for metric in METRICS}
        for arm in arms
    }
    results = {arm: {m: summary(arrays[arm][m]) for m in METRICS} for arm in arms}
    contrasts = {
        f"{left}_minus_{right}": {
            "left": left, "right": right,
            "metrics": {m: summary(arrays[left][m] - arrays[right][m]) for m in METRICS},
        }
        for left, right in pairs
    }
    return {
        "schema_version": 1,
        "arms": arms,
        "training_seeds": seeds,
        "objectives": objective_order,
        "episodes_per_seed_objective": n_episodes,
        "uncertainty": {
            "standard_deviation": "sample SD across training-seed means; not episode SD",
            "bootstrap": "percentile hierarchical bootstrap: training seeds then paired reset clusters; objectives equally weighted",
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_samples": bootstrap_samples,
            "limited_training_seeds": len(seeds) < 5,
            "warning": "Intervals are unstable with few training seeds, especially N=3; not proof of SOTA or seed robustness.",
            "difference_direction": "left minus right; larger return/resources is better, smaller capture rate is better",
        },
        "results": results,
        "comparisons": contrasts,
    }
