"""Fresh matched discrete rollouts and pinned 0s source/port comparison.

Run from the repository root with its locked environment. Source math is read
from the pinned checkout, never edited; --collect-only exposes the same fresh
dataset to the independent upstream strategy suite.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import jax
import numpy as np
from flax.serialization import to_bytes

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
PIN = "7bae281091a96fc83cadad3671bef070bd371675"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def collect():
    from mopa.data import (
        DEFAULT_LOGDIR,
        OBJECTIVE_TYPES,
        _params_path,
        objective_dataset,
    )

    start = time.perf_counter()
    print("Collecting 1800 fresh matched-start discrete episodes", flush=True)
    data = objective_dataset(n_eps=200, ckpt_seeds=(0, 1, 2), rng0=0,
                             num_steps=100, prey_type="capture").as_dict()
    # Upstream expects one observation per action, not the terminal observation.
    data["pred_obs"] = data["pred_obs"][:, :-1]
    for seed in (0, 1, 2):
        groups = [np.flatnonzero((data["label"] == label) & (data["ckpt_seed"] == seed))
                  for label in range(3)]
        for other in groups[1:]:
            for key in ("env_seed", "lava_pos", "lava_rad"):
                np.testing.assert_array_equal(data[key][groups[0]], data[key][other])
            for key in ("prey_pos", "pred_pos"):
                np.testing.assert_array_equal(data[key][groups[0], 0], data[key][other, 0])
    target = OUT / "dataset.npz"
    np.savez_compressed(target, **data)
    checkpoints = []
    for objective, team in [(x, "pred") for x in OBJECTIVE_TYPES] + [("capture", "prey")]:
        for seed in (0, 1, 2):
            path = _params_path(DEFAULT_LOGDIR, objective, team, seed)
            checkpoints.append({"path": str(path), "sha256": sha(path)})
    source = ["src/tag_objectives/objectives.py", "src/tag_objectives/resources.py",
              "src/mopa/data.py", "src/mopa/action_decoder.py", "src/mopa/features.py"]
    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": str(target), "dataset_sha256": sha(target),
        "collection_seconds": time.perf_counter() - start,
        "episodes": 1800, "episodes_per_objective_checkpoint": 200,
        "checkpoint_seeds": [0, 1, 2], "environment_rng0": 0,
        "prey_family": "capture", "horizon": 100,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_worktree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
        "source_hashes": {p: sha(ROOT / p) for p in source},
        "checkpoints": checkpoints,
        "schema": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in data.items()},
        "adaptation": "Drop final pred_obs frame only to match upstream pre-action observation schema.",
        "matched_reset_validation": "passed exact equality across objectives within each checkpoint seed",
        "specialists_retrained": False,
        "environment_changed": False,
        "objective_metrics": {
            objective: {k: float(data[k][data["label"] == i].mean())
                        for k in ("captured", "survival_time", "pred_lava_steps", "pred_coverage", "resources_collected")}
            for i, objective in enumerate(OBJECTIVE_TYPES)
        },
    }
    write_json(OUT / "dataset_manifest.json", manifest)
    print(f"DATASET_READY {target} sha256={manifest['dataset_sha256']} seconds={manifest['collection_seconds']:.1f}", flush=True)


def fit(source):
    from mopa.action_decoder import (
        ActionDecoderConfig,
        decode_action_decoder,
        fit_action_decoder_vae,
        gather_windows,
        pool_episode_prefix_latents,
    )
    from mopa.features import predator_sequence_features

    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if actual != PIN:
        raise ValueError(f"upstream pin mismatch: {actual}")
    sys.path.insert(0, str(source / "encoding_viz"))
    import eval_metrics
    import strategies
    import traj_data

    traj = traj_data.load(OUT / "dataset.npz")
    split = traj_data.checkpoint_split(traj, 2)
    raw = dict(np.load(OUT / "dataset.npz"))
    cfg = ActionDecoderConfig()

    def report(z, windows, episode, accuracy):
        result = eval_metrics.latent_report(z, traj.label, split.train_idx, split.val_idx)
        result["unit_probe"] = eval_metrics.unit_probe(windows, traj.label, episode, split.train_idx, split.val_idx)
        result["decoder_action_accuracy"] = accuracy
        result["decoder_accuracy_scope"] = "all valid windows including train and held-out, same as source"
        return result

    print("Fitting pinned source 0s: 1500 updates, key1000", flush=True)
    captured_params = {}

    def capture_return(frame, event, arg):
        if event == "return" and frame.f_code.co_name == "fit_action_decoder_vae":
            captured_params.update(frame.f_locals["params"])

    start = time.perf_counter()
    sys.setprofile(capture_return)
    try:
        upstream = strategies.strategy_0s_action_decoder_vae_short(traj, split, strategies.Config(seed=0))
    finally:
        sys.setprofile(None)
    source_report = report(upstream.z, *upstream.step_latents, upstream.extra["decoder_action_accuracy"])
    source_report["seconds"] = time.perf_counter() - start
    (OUT / "source_0s.msgpack").write_bytes(to_bytes(captured_params))
    np.savez_compressed(OUT / "source_latents.npz", episode_latents=upstream.z,
                        window_latents=upstream.step_latents[0], window_episode=upstream.step_latents[1])
    write_json(OUT / "source_history.json", upstream.history)
    print(f"SOURCE_COMPLETE {json.dumps(source_report)}", flush=True)

    all_reports = {"source_legacy_forward": source_report}
    fits = [("integrated_legacy_forward", 0, traj.state)] + [
        (f"integrated_causal_seed{seed}", seed,
         predator_sequence_features(raw["prey_pos"], raw["pred_pos"], traj.lengths, velocity_mode="causal_past"))
        for seed in (0, 1, 2)]
    for name, seed, state in fits:
        start = time.perf_counter()
        print(f"Fitting {name}: 1500 updates, key{1000 + seed}", flush=True)
        fitted = fit_action_decoder_vae(state, traj.action, traj.lengths, split.train_idx,
                                        jax.random.PRNGKey(1000 + seed), config=cfg)
        current = report(fitted.episode_latents, fitted.window_latents,
                         fitted.encoding.windows.episode, fitted.decoder_action_accuracy)
        windows = fitted.encoding.windows
        step_state = gather_windows(((state - fitted.encoder.state_mean) / fitted.encoder.state_std)
                                    * traj.mask[..., None], windows)
        z = np.broadcast_to(fitted.window_latents[:, None], (*windows.mask.shape, cfg.lat))
        prediction = np.asarray(decode_action_decoder(fitted.decoder_params, step_state, z, cfg)).argmax(-1)
        target = gather_windows(traj.action, windows)
        holdout_mask = windows.mask & np.isin(windows.episode, split.val_idx)[:, None]
        current["heldout_decoder_accuracy"] = float((prediction == target)[holdout_mask].mean())
        current["prefix_probe"] = []
        for prefix in (2, 5, 8, 10, 20, 40, 70, 100):
            available = np.minimum(traj.lengths, prefix)
            zp = pool_episode_prefix_latents(fitted.prefix_latents, windows, available, traj.n_eps)
            zt, zs = eval_metrics.standardize(zp[split.train_idx], zp)
            clf = eval_metrics.LogisticRegression(max_iter=2000).fit(zt, traj.label[split.train_idx])
            current["prefix_probe"].append({"steps": prefix, "heldout_probe": float(
                eval_metrics.accuracy_score(traj.label[split.val_idx], clf.predict(zs[split.val_idx])))})
        current["seconds"] = time.perf_counter() - start
        current["logical_seed"] = seed
        current["jax_key_seed"] = 1000 + seed
        current["causal_features"] = "causal" in name
        current["episode_latent_scope"] = "post-hoc full valid episode; not an online next-action claim"
        current["prefix_scope"] = "representation after observing k state/action pairs; use before next action only"
        np.savez_compressed(OUT / f"{name}_latents.npz", episode_latents=fitted.episode_latents,
                            window_latents=fitted.window_latents, prefix_latents=fitted.prefix_latents,
                            window_episode=windows.episode, label=traj.label, checkpoint_seed=traj.ckpt_seed,
                            train_idx=split.train_idx, val_idx=split.val_idx,
                            state_mean=fitted.encoder.state_mean, state_std=fitted.encoder.state_std)
        (OUT / f"{name}.msgpack").write_bytes(to_bytes({"e": fitted.encoder.params, "d": fitted.decoder_params}))
        write_json(OUT / f"{name}_history.json", fitted.history)
        all_reports[name] = current
        if name == "integrated_legacy_forward":
            parity = {"episode_latent_max_abs": float(np.max(np.abs(upstream.z - fitted.episode_latents))),
                      "window_latent_max_abs": float(np.max(np.abs(upstream.step_latents[0] - fitted.window_latents))),
                      "decoder_accuracy_abs": abs(upstream.extra["decoder_action_accuracy"] - fitted.decoder_action_accuracy),
                      "history_max_abs": max(abs(a[k] - b[k]) for a, b in zip(upstream.history, fitted.history)
                                              for k in ("loss", "action_mse", "kl"))}
            for value in parity.values():
                assert value <= 1e-5, parity
            all_reports["parity"] = parity
        write_json(OUT / "metrics.partial.json", all_reports)
        print(f"FIT_COMPLETE {name} probe={current['heldout_probe']:.4f} ari={current['heldout_ari']:.4f} unit={current['unit_probe']:.4f} accuracy={current['decoder_action_accuracy']:.4f} seconds={current['seconds']:.1f}", flush=True)

    names = ("heldout_probe", "heldout_ari", "unit_probe", "decoder_action_accuracy", "heldout_decoder_accuracy")
    all_reports["causal_three_seed_summary"] = {
        name: {"mean": float(np.mean(values)), "std": float(np.std(values)), "values": values}
        for name in names
        for values in [[all_reports[f"integrated_causal_seed{seed}"][name] for seed in (0, 1, 2)]]
    }
    all_reports["controls"] = {"length_probe": eval_metrics.control_probe(traj.survival_time, traj.label, split.train_idx, split.val_idx), "chance": 1 / 3}
    all_reports["protocol"] = {"config": asdict(cfg), "heldout_checkpoint": 2, "source_pin": PIN,
                                "dataset_sha256": sha(OUT / "dataset.npz"),
                                "scoring_seed": 0, "std_definition": "population std across encoder seeds; not confidence interval",
                                "source_hashes": {str(p.relative_to(source)): sha(p) for p in (source / "encoding_viz").glob("*.py")}}
    write_json(OUT / "metrics.json", all_reports)
    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version, "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ("jax", "jaxlib", "flax", "optax", "numpy", "scikit-learn")},
        "driver_sha256": sha(Path(__file__)),
        "artifacts": {path.name: sha(path) for path in OUT.iterdir()
                      if path.suffix in {".npz", ".msgpack"} or path.name == "metrics.json"},
        "weight_capture": "Source trainer params captured at function return with sys.setprofile; no source edits or numerical changes.",
        "source_0s_default_key": 1000,
        "parity_absolute_tolerance": 1e-5,
    }
    write_json(OUT / "fit_manifest.json", manifest)
    print(f"DONE {OUT / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/private/tmp/shashank-encoding-audit.5AsKUv/repo"))
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--fit-only", action="store_true")
    args = parser.parse_args()
    if args.collect_only and args.fit_only:
        parser.error("choose at most one stage")
    if not args.fit_only:
        collect()
    if not args.collect_only:
        fit(args.source)
