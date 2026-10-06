#!/usr/bin/env python3
"""Inspect three actual world-model modes with the same frozen 0s prototypes.

Default: restore compatible checkpoints and export imagined paths. Explicit
--fit-missing fits only missing implicit/conditioned models on saved training
episodes; it never trains 0s/MAPPO, steps an environment, or computes probes.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import flax.serialization
import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mopa.manifest import file_sha256  # noqa: E402
from mopa.tdmpc import check_update, create_agent  # noqa: E402
from mopa.tdmpc_data import SequenceReplay  # noqa: E402
from mopa.zero_s import ZeroSOpponent, strategy_prototypes  # noqa: E402

TYPES = ("capture", "risk", "curious")
MODES = ("implicit", "conditioned", "factored")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_source(source, dataset):
    manifest = json.loads((source / "manifest.json").read_text())
    for name in ("agent.msgpack", "opponent.msgpack", "config.json", "latents.npz", "state_stats.npz", "training_history.json"):
        if file_sha256(source / name) != manifest["artifacts"][name]:
            raise ValueError(f"Original 0s artifact hash mismatch: {name}")
    if file_sha256(dataset) != manifest["dataset"]["sha256"]:
        raise ValueError("Dataset differs from the frozen 0s training dataset")
    with np.load(dataset, allow_pickle=False) as archive:
        data = {k: archive[k] for k in ("state", "blue_action", "red_action", "blue_reward",
                                       "terminated_capture", "truncated_timeout", "valid_length",
                                       "valid_mask", "objective_label", "checkpoint_seed")}
    with np.load(source / "latents.npz", allow_pickle=False) as archive:
        latent = {k: archive[k] for k in ("episode", "context", "prototypes", "labels", "train", "heldout",
                                         "checkpoint_seed", "pca_components", "pca_mean", "latent_mean", "latent_scale")}
    with np.load(source / "state_stats.npz", allow_pickle=False) as archive:
        mean, std = archive["mean"], archive["std"]
    opponent = ZeroSOpponent.load(source / "opponent.msgpack")
    np.testing.assert_array_equal(latent["labels"], data["objective_label"])
    np.testing.assert_array_equal(latent["checkpoint_seed"], data["checkpoint_seed"])
    heldout = manifest["heldout_checkpoint"]
    np.testing.assert_array_equal(latent["train"], np.flatnonzero(data["checkpoint_seed"] != heldout))
    np.testing.assert_array_equal(latent["heldout"], np.flatnonzero(data["checkpoint_seed"] == heldout))
    means = strategy_prototypes(latent["episode"], latent["labels"], latent["train"])
    np.testing.assert_allclose(means, opponent.prototypes, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(means, latent["prototypes"], rtol=1e-6, atol=1e-6)
    if latent["context"].shape != (*data["state"].shape[:2], 8):
        raise ValueError("Expected the matching frozen 8D causal 0s context cache")
    config = json.loads((source / "config.json").read_text())["world_model"]
    updates = json.loads((source / "training_history.json").read_text())["world_model"][-1]["step"]
    return manifest, data, latent, opponent, mean, std, config, updates


def load_or_fit(mode, out, source, manifest, data, latent, opponent, mean, std, base_config, updates, fit_missing):
    config = copy.deepcopy(base_config)
    config["opponent_mode"] = mode
    config["context_dim"] = 0 if mode == "implicit" else 8
    seed = int(manifest["seed"])
    agent = create_agent(config, 66, key=jax.random.PRNGKey(seed), obs_mean=mean, obs_std=std)
    if mode == "factored":
        agent = opponent.attach(agent, mean, std)
        agent = flax.serialization.from_bytes(agent, (source / "agent.msgpack").read_bytes())
        return agent, {"source": str(source.resolve()), "agent_sha256": manifest["artifacts"]["agent.msgpack"],
                       "updates": updates, "mode": mode, "reused": True}
    directory = out / mode
    model_path, binding_path = directory / "agent.msgpack", directory / "binding.json"
    binding = {"mode": mode, "config": config, "updates": updates, "seed": seed,
               "dataset_sha256": manifest["dataset"]["sha256"],
               "frozen_opponent_sha256": manifest["artifacts"]["opponent.msgpack"],
               "latents_sha256": manifest["artifacts"]["latents.npz"],
               "train_episodes": latent["train"].tolist(),
               "code": {name: file_sha256(ROOT / name) for name in
                        ("src/mopa/tdmpc.py", "src/mopa/tdmpc_data.py", "src/mopa/world_model.py", "scripts/inspect_world_models.py")
                        if (ROOT / name).is_file()}}
    if model_path.exists():
        saved = json.loads(binding_path.read_text())
        if saved["binding"] != binding or saved["agent_sha256"] != file_sha256(model_path):
            raise ValueError(f"Incompatible saved {mode}; use a new output directory, never relabel weights")
        return flax.serialization.from_bytes(agent, model_path.read_bytes()), saved
    if not fit_missing:
        raise ValueError(f"Missing {mode}+0s-compatible checkpoint: explicitly use --fit-missing to fit it")
    if directory.exists() and any(directory.iterdir()):
        raise ValueError(f"Refusing to overwrite an incomplete model directory: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    context = latent["context"] if mode != "implicit" else latent["context"][..., :0]
    replay = SequenceReplay.from_dataset(data, latent["train"], agent.horizon, context, feature_map="markov")
    rng, key = np.random.default_rng(seed), jax.random.PRNGKey(10000 + seed)
    print(f"Fit {mode}: matching the source world-model budget; frozen 0s; training episodes only", flush=True)
    history = []
    for step in range(1, updates + 1):
        key, update_key = jax.random.split(key)
        agent, info = agent.update(**replay.sample(rng, agent.batch_size), key=update_key)
        check_update(info, step)
        if step == 1 or step % 100 == 0 or step == updates:
            row = {k: float(np.asarray(v)) for k, v in info.items() if np.asarray(v).ndim == 0}
            if not np.isfinite(list(row.values())).all():
                raise RuntimeError(f"Nonfinite {mode} update")
            history.append({"step": step, **row})
            print(f"{mode}: update {step}/{updates}", flush=True)
    model_path.write_bytes(flax.serialization.to_bytes(agent))
    saved = {"binding": binding, "agent_sha256": file_sha256(model_path), "training_history": history}
    write_json(binding_path, saved)
    return agent, saved


def scene_indices(data, heldout, per_type=3):
    """First/middle/last held-out examples by type; never select by prediction."""
    selected = []
    for label in range(3):
        rows = heldout[data["objective_label"][heldout] == label]
        if len(rows) < per_type:
            raise ValueError("Insufficient held-out scenes")
        selected.extend(rows[np.linspace(0, len(rows) - 1, per_type, dtype=int)].tolist())
    return selected


def display_paths(result):
    states = result["state"][:, 0]
    continuation = result["continuation"][:, 0]
    red = result.get("red_action")
    return [{"xy": np.round(states[:, k, :4].astype(float), 4).tolist(),
             "red": None if red is None else np.round(red[:, 0, k].astype(float), 4).tolist(),
             "stop": int(np.flatnonzero(continuation[:, k] < 0.5)[0] + 1)
                     if np.any(continuation[:, k] < 0.5) else len(states)} for k in range(3)]


def export_view(args, source_info):
    from mopa.world_model_inspection import rollout_prototypes

    manifest, data, latent, opponent, mean, std, config, updates = source_info
    trained = {}
    provenance = {}
    for mode in MODES:
        trained[mode], provenance[mode] = load_or_fit(
            mode, args.out, args.source, manifest, data, latent, opponent, mean, std, config, updates, args.fit_missing)
    scenes, arrays = [], {}
    for i, episode in enumerate(scene_indices(data, latent["heldout"])):
        horizon = min(args.horizon, int(data["valid_length"][episode]))
        initial = data["state"][episode, 0:1]
        blue = data["blue_action"][episode, :horizon, None]
        red = data["red_action"][episode, :horizon, None]
        scene = {"episode": episode, "objective": TYPES[int(data["objective_label"][episode])],
                 "state0": initial[0].astype(float).tolist(), "blue": blue[:, 0].astype(float).tolist(),
                 "reference": data["state"][episode, :horizon + 1, :4].astype(float).tolist(), "models": []}
        for mode in MODES:
            print(f"Imagine {mode}: saved scene {episode}, fixed blue actions and three fixed means", flush=True)
            result = rollout_prototypes(trained[mode], initial, blue, opponent.prototypes, mean, std)
            item = {"mode": mode, "paths": display_paths(result)}
            arrays.update({f"scene{i}_{mode}_{k}": v for k, v in result.items() if v is not None})
            if mode == "implicit":
                np.testing.assert_array_equal(result["state"][:, :, 0], result["state"][:, :, 1])
                np.testing.assert_array_equal(result["state"][:, :, 1], result["state"][:, :, 2])
            if mode == "factored":
                clamped = rollout_prototypes(trained[mode], initial, blue, opponent.prototypes, mean, std, clamped_red_actions=red)
                for name in ("state", "continuation"):
                    np.testing.assert_array_equal(clamped[name][:, :, 0], clamped[name][:, :, 1])
                    np.testing.assert_array_equal(clamped[name][:, :, 1], clamped[name][:, :, 2])
                item["clamped_paths"] = display_paths(clamped)
                arrays.update({f"scene{i}_clamped_{k}": v for k, v in clamped.items() if v is not None})
            scene["models"].append(item)
        scenes.append(scene)
    def project(z):
        standardized = (z - latent["latent_mean"]) / latent["latent_scale"]
        return (standardized - latent["pca_mean"]) @ latent["pca_components"].T
    xy = project(latent["episode"])
    payload = {"prototypes": opponent.prototypes.tolist(), "prototype_pc": project(opponent.prototypes).tolist(),
               "latent_cloud": [{"point": np.round(xy[e].astype(float), 4).tolist(), "label": int(latent["labels"][e]), "episode": int(e)}
                                for e in latent["train"]], "scenes": scenes}
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "rollouts.npz", **arrays, prototypes=opponent.prototypes,
                        scenes=np.asarray([s["episode"] for s in scenes]), train_episodes=latent["train"])
    template = Path(__file__).with_name("world_model_inspector.html")
    encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    fragment = template.read_text().replace("__WORLD_MODEL_DATA__", encoded)
    if len(fragment.encode()) >= 1_000_000:
        raise ValueError("Visualization exceeds inline size limit")
    (args.out / "world-model-behaviors.html").write_text(fragment)
    write_json(args.out / "world-model-data.json", payload)
    write_json(args.out / "manifest.json", {
        "source": str(args.source.resolve()), "source_manifest_sha256": file_sha256(args.source / "manifest.json"),
        "dataset_sha256": manifest["dataset"]["sha256"], "model_provenance": provenance,
        "selection": "First, middle, last held-out episode per objective; no prediction-based selection",
        "protocol": "Same raw initial state and recorded blue actions across three equations and three train-only 0s means",
        "encoder_retrained": False, "environment_rollouts": False, "mpc_planning": False,
        "equation1_swap_invariance": True, "equation3_joint_action_clamp_invariance": True,
        "limitations": ["Known-type mean intervention, not online strategy inference",
                        "Predicted paths, not simulator ground truth for latent swaps",
                        "Fixed-horizon predictions beyond model stop are extrapolation",
                        "Matching source offline update budget is not control-performance evidence"],
        "code": {str(p.relative_to(ROOT)): file_sha256(p) for p in
                 (Path(__file__), ROOT / "src/mopa/world_model_inspection.py", ROOT / "src/mopa/zero_s.py", template)},
        "artifacts": {name: file_sha256(args.out / name) for name in ("rollouts.npz", "world-model-behaviors.html", "world-model-data.json")}})
    print(f"World-model visualization: {args.out / 'world-model-behaviors.html'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "experiments/continuous_0s")
    parser.add_argument("--dataset", type=Path, default=ROOT / "experiments/main_env_20260908/continuous_data/dataset.npz")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/world_model_inspection")
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--fit-missing", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.horizon <= 100:
        parser.error("horizon must lie between 1 and 100")
    export_view(args, load_source(args.source, args.dataset))


if __name__ == "__main__":
    main()
