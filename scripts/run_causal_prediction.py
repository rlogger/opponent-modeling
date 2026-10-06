#!/usr/bin/env python3
"""Run one predeclared causal prediction arm; never resume/overwrite artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import jax
import numpy as np

from mopa.causal_opponent import (
    FEATURES,
    METHODS,
    OBJECTIVES,
    SCHEMA,
    CausalOpponentConfig,
    evaluate_prediction,
    fit_causal_opponent,
    match_encoder_capacity,
    validate_indices,
)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, content):
    Path(path).write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")


def validate_protocol_run(protocol, stage, config, seed, binding):
    """Reject an unrelated protocol or any unfrozen scientific run setting."""
    frozen = protocol["causal_prediction"][stage]
    if asdict(config) not in frozen["configurations"] or seed not in frozen["seeds"]:
        raise ValueError("configuration or seed is not frozen in causal-prediction protocol")
    for name in ("checkpoint_families", "evaluation_split", "evaluation_samples", "dataset_sha256", "dataset_sidecar_sha256", "splits_sha256"):
        if binding[name] != frozen[name]:
            raise ValueError(f"{name} differs from frozen causal-prediction protocol")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True, help="JSON train/validation/test episode-index arrays")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--spec-commit", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--stage", choices=("pilot", "main"), required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--features", choices=FEATURES, default="expert17_v1")
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--max-history", type=int, help="Separate history-length ablation; omitted uses all past pairs")
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--encoder-hidden", type=int)
    parser.add_argument("--match-recurrent-capacity", action="store_true")
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--free-bits", type=float, default=0.2)
    parser.add_argument("--sample-training", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--objective", choices=OBJECTIVES, default="past_next_action")
    parser.add_argument("--sampler", choices=("episode_uniform", "transition_uniform"), default="episode_uniform")
    parser.add_argument("--evaluation-samples", type=int, default=32)
    parser.add_argument("--evaluation-split", choices=("validation", "test", "none"), default="validation")
    args = parser.parse_args()
    cfg = CausalOpponentConfig(method=args.method, feature_schema=args.features, history=args.history, max_history=args.max_history,
                               latent_dim=args.latent_dim, hid=args.hidden, encoder_hid=args.encoder_hidden, steps=args.steps,
                               batch=args.batch, learning_rate=args.learning_rate, beta=args.beta, free_bits=args.free_bits,
                               sample_training=args.sample_training, objective=args.objective, sampler=args.sampler)
    capacity = None
    if args.match_recurrent_capacity:
        from dataclasses import replace

        reference = replace(cfg, method="recurrent_vae", encoder_hid=None)
        cfg, capacity = match_encoder_capacity(cfg, reference)
    root = Path(__file__).resolve().parents[1]
    code_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if code_commit != args.code_commit or len(args.spec_commit) != 40:
        raise ValueError("exact executable HEAD and full specification commit required")
    if subprocess.check_output(["git", "diff", "--name-only", "HEAD", "--", "src", "scripts"], cwd=root, text=True).strip():
        raise ValueError("executable source has uncommitted modifications")
    if subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "--", "src", "scripts"], cwd=root, text=True).strip():
        raise ValueError("executable source has untracked additions")
    if args.output.exists():
        raise FileExistsError("new campaign output directory required")
    splits = json.loads(args.splits.read_text())
    with np.load(args.dataset, allow_pickle=False) as raw:
        state, action, lengths = raw["state"], raw["red_action"], raw["valid_length"]
        labels, families = raw["objective_label"], raw["checkpoint_seed"]
    indices = {name: (np.empty(0, np.int32) if name != "train" and not len(splits[name]) else validate_indices(splits[name], len(state), name)) for name in ("train", "validation", "test")}
    if args.evaluation_split != "none" and not len(indices[args.evaluation_split]):
        raise ValueError("selected evaluation split is empty")
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if np.intersect1d(indices[left], indices[right]).size:
            raise ValueError("episode split overlap")
        if np.intersect1d(families[indices[left]], families[indices[right]]).size:
            raise ValueError("checkpoint-family split overlap")
    # Read and hash the pre-existing protocol before any fit.
    protocol_text = args.protocol.read_text()
    if not protocol_text.strip():
        raise ValueError("empty protocol")
    protocol = json.loads(protocol_text)
    if not protocol.get("protocol_id") or "version" not in protocol:
        raise ValueError("protocol ID and version required")
    spec_root = next((p for p in args.protocol.resolve().parents if (p / ".git").exists()), None)
    if spec_root is None:
        raise ValueError("protocol must reside in the specification Git checkout")
    protocol_relative = args.protocol.resolve().relative_to(spec_root)
    committed_protocol = subprocess.check_output(["git", "show", f"{args.spec_commit}:{protocol_relative}"], cwd=spec_root)
    if committed_protocol != args.protocol.read_bytes():
        raise ValueError("protocol differs from specification commit")
    sidecar_path = args.dataset.with_suffix(".manifest.json")
    sidecar = json.loads(sidecar_path.read_text())
    sources = sidecar.get("source_checkpoints", [])
    if not sources or any(len(source.get("sha256", "")) != 64 for source in sources):
        raise ValueError("dataset sidecar must bind source checkpoints")
    binding = {"checkpoint_families": {k: sorted(np.unique(families[v]).tolist()) for k, v in indices.items()},
               "evaluation_split": args.evaluation_split, "evaluation_samples": args.evaluation_samples,
               "dataset_sha256": sha256(args.dataset), "dataset_sidecar_sha256": sha256(sidecar_path),
               "splits_sha256": sha256(args.splits)}
    validate_protocol_run(protocol, args.stage, cfg, args.seed, binding)
    manifest = {
        "schema": SCHEMA, "status": "running", "stage": args.stage, "seed": args.seed, "config": asdict(cfg),
        "capacity_match": capacity,
        "specification": {"repository": "rlogger/marl-private", "commit": args.spec_commit,
                          "protocol_path": str(protocol_relative), "protocol_sha256": sha256(args.protocol),
                          "protocol_id": protocol["protocol_id"], "protocol_version": protocol["version"]},
        "implementation": {"repository": "rlogger/opponent-modeling", "commit": code_commit},
        "dataset": {"path": str(args.dataset.resolve()), "sha256": binding["dataset_sha256"],
                    "sidecar_path": str(sidecar_path.resolve()), "sidecar_sha256": binding["dataset_sidecar_sha256"],
                    "source_checkpoints": sources},
        "splits": {"path": str(args.splits.resolve()), "sha256": sha256(args.splits),
                   "families": binding["checkpoint_families"],
                   "episodes": {k: len(v) for k, v in indices.items()}},
        "evaluation_split": args.evaluation_split, "evaluation_samples": args.evaluation_samples,
        "runtime": {"python": platform.python_version(), "jax": jax.__version__, "numpy": np.__version__, "devices": [str(d) for d in jax.devices()]},
        "lockfile_sha256": sha256(root / "uv.lock"),
        "source_hashes": {str(p.relative_to(root)): sha256(p) for p in sorted((root / "src").rglob("*.py")) + [Path(__file__)]},
        "claims": "New comparison; not reproduction of the unidentified collaborator report. Sample spread alone is not calibrated strategy uncertainty.",
    }
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "manifest.json", manifest)
    start = time.monotonic()
    try:
        model, history = fit_causal_opponent(state, action, lengths, indices["train"], jax.random.PRNGKey(args.seed), config=cfg, labels=labels,
                                           training_state_path=args.output / "training_state.msgpack")
        fit_seconds = time.monotonic() - start
        model.save(args.output / "opponent.msgpack")
        if args.evaluation_split == "none":
            metrics = {"evaluation": "not_requested", "purpose": "training runtime profile only"}
        else:
            metrics, traces = evaluate_prediction(model, state, action, lengths, indices[args.evaluation_split], labels,
                                                 sample_seed=args.seed + 100000, samples=args.evaluation_samples)
            np.savez_compressed(args.output / "prediction_traces.npz", **traces)
        write_json(args.output / "training_history.json", history)
        write_json(args.output / "metrics.json", metrics)
        watched = [(args.dataset, manifest["dataset"]["sha256"]), (sidecar_path, manifest["dataset"]["sidecar_sha256"]),
                   (args.splits, manifest["splits"]["sha256"]), (args.protocol, manifest["specification"]["protocol_sha256"]),
                   (root / "uv.lock", manifest["lockfile_sha256"])]
        watched.extend((root / name, digest) for name, digest in manifest["source_hashes"].items())
        if any(sha256(path) != digest for path, digest in watched):
            raise RuntimeError("source, dataset or protocol changed during fitting/evaluation")
        if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip() != code_commit:
            raise RuntimeError("executable commit changed during fitting/evaluation")
        manifest.update(status="completed", fit_seconds=fit_seconds, total_seconds=time.monotonic()-start,
                        parameter_count=sum(int(np.asarray(v).size) for v in jax.tree.leaves(model.params)),
                        artifacts={p.name: sha256(p) for p in args.output.iterdir() if p.name != "manifest.json"})
    except Exception as error:
        manifest.update(status="failed", error_type=type(error).__name__, error=str(error), total_seconds=time.monotonic()-start)
        write_json(args.output / "manifest.json", manifest)
        raise
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps({"output": str(args.output), "status": manifest["status"], "fit_seconds": fit_seconds}))


if __name__ == "__main__":
    main()
