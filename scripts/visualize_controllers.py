#!/usr/bin/env python3
"""Export saved matched controller traces for inspection; never run a policy.

Only NumPy is needed. PCA is a descriptive projection of this saved cohort,
not an encoder, probe, clustering benchmark, or out-of-sample evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

OBJECTIVES = ("capture", "risk", "curious")
CONTROLLERS = (("mappo", "mappo_prey", "zero"), ("tdmpc", "tdmpc", "online"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_trace(path):
    """Validate the one-red/one-blue Markov recording contract, not rewards."""
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in (
            "state", "blue_action", "red_action", "valid_mask", "valid_length",
            "terminated_capture", "truncated_timeout", "context", "dataset_episode",
            "objective_label", "checkpoint_seed", "environment_seed", "step_seed",
        )}
    state, mask, lengths = data["state"], data["valid_mask"], data["valid_length"]
    if state.ndim != 3 or state.shape[-1] != 66 or mask.shape != (state.shape[0], state.shape[1] - 1):
        raise ValueError("Expected recorded state (episodes, steps+1, 66) and matching mask")
    batch, horizon = mask.shape
    if (lengths.shape != (batch,) or lengths.dtype.kind not in "iu"
            or not np.all((lengths >= 1) & (lengths <= horizon))):
        raise ValueError("Invalid valid_length")
    expected_mask = np.arange(horizon)[None] < lengths[:, None]
    if not np.array_equal(mask, expected_mask):
        raise ValueError("valid_mask must be a contiguous prefix matching valid_length")
    for key in ("blue_action", "red_action"):
        action = data[key]
        if action.shape != (batch, horizon, 2) or not np.isfinite(action[expected_mask]).all():
            raise ValueError(f"Invalid {key}")
        if np.any(np.abs(action[expected_mask]) > 1 + 1e-6):
            raise ValueError(f"Unbounded {key}")
    context = data["context"]
    if context.ndim != 3 or context.shape[:2] != mask.shape or not np.isfinite(context[expected_mask]).all():
        raise ValueError("Invalid decision-time context")
    for key in ("terminated_capture", "truncated_timeout"):
        if data[key].shape != mask.shape or data[key].dtype.kind != "b":
            raise ValueError(f"Invalid {key}")
    for key in ("dataset_episode", "objective_label", "checkpoint_seed", "environment_seed", "step_seed"):
        shape = (batch, 2) if key in {"environment_seed", "step_seed"} else (batch,)
        if data[key].shape != shape or data[key].dtype.kind not in "iu":
            raise ValueError(f"Invalid {key}")
    if len(np.unique(data["dataset_episode"])) != batch:
        raise ValueError("Duplicate dataset_episode")
    terminal = data["terminated_capture"] | data["truncated_timeout"]
    last = np.arange(horizon)[None] == lengths[:, None] - 1
    if terminal.shape != mask.shape or not np.array_equal(terminal, last):
        raise ValueError("Termination/truncation must mark only the last valid action")
    if np.any(data["terminated_capture"] & data["truncated_timeout"]):
        raise ValueError("Capture and timeout cannot both be true")
    for i, length in enumerate(lengths):
        if not np.isfinite(state[i, :length + 1]).all():
            raise ValueError("Nonfinite valid state")
    return data


def trajectory_features(positions, samples=32):
    """Joint displacement paths, interpolated over completed-episode phase.

    Remove each agent's initial position; retain arena distance units and
    orientation. Time normalization deliberately discards absolute duration.
    This summary uses the whole episode and must never be fed to an online actor.
    """
    positions = np.asarray(positions, dtype=np.float64)
    if (positions.ndim != 2 or positions.shape[1] != 4 or len(positions) < 2
            or not np.isfinite(positions).all() or not isinstance(samples, int) or samples < 2):
        raise ValueError("Expected at least two joint-position frames")
    phase = np.linspace(0, 1, len(positions))
    sampled = np.stack([np.interp(np.linspace(0, 1, samples), phase, positions[:, j])
                        for j in range(4)], axis=-1)
    return (sampled - positions[0]).reshape(-1)


def pca(values):
    """Label-free, centered, unscaled descriptive PCA with stable axis signs."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 2 or not np.isfinite(values).all():
        raise ValueError("PCA requires finite 2D data with at least two rows and columns")
    mean = values.mean(axis=0)
    _, _, vectors = np.linalg.svd(values - mean, full_matrices=False)
    components = vectors[:2].copy()
    for row in components:
        if row[np.argmax(np.abs(row))] < 0:
            row *= -1
    return (values - mean) @ components.T, mean, components


def rounded(values):
    return np.round(np.asarray(values, dtype=np.float64), 3).tolist()


def build_bundle(evaluation):
    evaluation = Path(evaluation)
    report = json.loads(evaluation.read_text())
    traces, inputs, maps, rows, descriptors = {}, {}, [], [], []
    inputs[evaluation.name] = digest(evaluation)
    for objective in OBJECTIVES:
        for short, controller, context_mode in CONTROLLERS:
            matching = [r for r in report["runs"] if (r["opponent"], r["controller"], r["context_mode"])
                        == (objective, controller, context_mode)]
            if len(matching) != 1:
                raise ValueError(f"Need exactly one recorded {objective}/{controller}/{context_mode}")
            name = matching[0]["recordings"]["transitions"]
            if Path(name).name != name:
                raise ValueError("Recording must be a file in the evaluation directory")
            path = evaluation.parent / name
            traces[objective, short] = load_trace(path)
            inputs[name] = digest(path)
        first, second = traces[objective, "mappo"], traces[objective, "tdmpc"]
        for key in ("environment_seed", "step_seed", "dataset_episode", "objective_label", "checkpoint_seed"):
            if not np.array_equal(first[key], second[key]):
                raise ValueError(f"Cannot pair {objective}: mismatched {key}")
        if not np.array_equal(first["state"][:, 0], second["state"][:, 0]):
            raise ValueError("Paired controllers must share the entire initial state")
        if not np.all(first["objective_label"] == OBJECTIVES.index(objective)):
            raise ValueError("Objective labels do not match the recording name")
        for i in range(len(first["valid_length"])):
            initial = first["state"][i, 0]
            map_id = len(maps)
            maps.append({"resources": rounded(initial[8:40].reshape(-1, 2)),
                         "lava": rounded(np.column_stack((initial[56:62].reshape(-1, 2), initial[62:65])))})
            for short, _, _ in CONTROLLERS:
                data = traces[objective, short]
                length = int(data["valid_length"][i])
                state = data["state"][i, :length + 1]
                rows.append({"controller": short, "objective": objective, "pair": i,
                             "episode": int(data["dataset_episode"][i]), "map": map_id,
                             "length": length, "checkpoint": int(data["checkpoint_seed"][i]),
                             "stop": "capture" if data["terminated_capture"][i, length - 1] else "timeout",
                             "xy": rounded(state[:, :4]),
                             "blue": rounded(data["blue_action"][i, :length]),
                             "red": rounded(data["red_action"][i, :length]),
                             "collected": ((state[:, 40:56] > 0.5).astype(np.int64) @ (1 << np.arange(16))).tolist()})
                descriptors.append(trajectory_features(state[:, :4]))
    coords, mean, components = pca(descriptors)
    for row, point in zip(rows, coords):
        row["pc"] = rounded(point)

    # A separate projection of the actual pre-action opponent contexts. These
    # coordinates are not comparable with trajectory PCA or MAPPO activations.
    contexts = [traces[row["objective"], "tdmpc"]["context"][row["pair"], :row["length"]]
                for row in rows if row["controller"] == "tdmpc"]
    if any(c.shape[-1] != 8 for c in contexts):
        raise ValueError("This inspector expects the recorded eight-dimensional 0s context")
    _, zmean, zcomponents = pca(np.concatenate(contexts))
    for row, context in zip((r for r in rows if r["controller"] == "tdmpc"), contexts):
        row["zpc"] = rounded((context - zmean) @ zcomponents.T)
    metadata = {"evaluation": str(evaluation.resolve()), "input_sha256": inputs,
                "checkpoint_artifacts": report["checkpoint_artifacts"],
                "prey_control_checkpoint": report["prey_control_checkpoint"],
                "planner": report["planner"], "world_model_mode": report["mode"],
                "state_encoder": report["encoder"], "opponent_model": report["opponent_model"],
                "source_git_sha": report["git_sha"],
                "source_git_dirty": report["git_dirty"],
                "selection": "All saved episodes in each of the six matched recordings; no outcome selection",
                "trajectory_pca": {"mean": mean.tolist(), "components": components.tolist(),
                                   "features": "32 phase-resampled joint displacement frames, flattened; centered, not standardized",
                                   "fit": "All displayed completed episodes; labels excluded; descriptive only"},
                "context_pca": {"mean": zmean.tolist(), "components": zcomponents.tolist(),
                                "fit": "All valid pre-action TD-MPC 0s contexts; labels excluded; descriptive only"},
                "limitations": ["Same reset, separate realized trajectories; not same-state action counterfactuals",
                                "MAPPO local observations versus TD-MPC full state/history; not equal-information evidence",
                                "No planner candidates or actor hidden activations recorded",
                                "PCA neighborhoods do not establish learned strategies; no clusters or performance metrics fitted"],
                "policy_inference": False, "new_rollouts": False, "retraining": False}
    return {"episodes": rows, "maps": maps}, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, default=Path("experiments/main_env_20260908/controller/evaluation.json"))
    parser.add_argument("--out", type=Path, default=Path("experiments/controller_inspection"))
    args = parser.parse_args()
    bundle, manifest = build_bundle(args.evaluation)
    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out / "trajectories.json"
    output.write_text(json.dumps(bundle, separators=(",", ":"), allow_nan=False) + "\n")
    manifest["source_sha256"] = digest(__file__)
    manifest["output_sha256"] = digest(output)
    template_path = Path(__file__).with_name("controller_inspector.html")
    template = template_path.read_text()
    fragment = template.replace("__TRAJECTORY_DATA__", output.read_text().replace("<", "\\u003c"))
    if len(fragment.encode()) >= 1_000_000:
        raise ValueError("Inline inspector exceeds 1 MB; reduce display precision or cohort explicitly")
    (args.out / "controller-trajectories.html").write_text(fragment)
    manifest["template_sha256"] = digest(template_path)
    manifest["fragment_sha256"] = digest(args.out / "controller-trajectories.html")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(f"Saved trajectory inspection data: {output}. No inference, rollouts, training, or performance evaluation.")


if __name__ == "__main__":
    main()
