#!/usr/bin/env python3
"""Generate or repackage a provenance-bound three-objective trajectory bundle."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mopa.data import OBJECTIVE_TYPES, objective_dataset  # noqa: E402
from mopa.manifest import validate_manifest  # noqa: E402
from mopa.trajectory_export import (  # noqa: E402
    LABEL_NAMES,
    dataset_schema,
    file_sha256,
    load_objective_dataset,
    reconstruct_resources,
    render_representative_figure,
    save_dataset_atomic,
    select_representative_group,
    validate_objective_dataset,
)

ENVIRONMENT_HORIZON_CAP = 100
BUNDLE_SCHEMA = "objective-trajectory-export/v1"
LEGACY_BUNDLE_FILENAMES = frozenset(
    {
        "trajectories.npz",
        "representative_trajectories.png",
        "representative_trajectories.pdf",
        "metadata.json",
        "source_manifest.json",
    }
)
BUNDLE_FILENAMES = LEGACY_BUNDLE_FILENAMES | {"renderer_sources.json"}
RENDERER_SOURCE_FILES = (
    Path("scripts/export_trajectory_bundle.py"),
    Path("src/mopa/trajectory_export.py"),
)


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds or len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError(
            "checkpoint seeds must be non-empty and unique"
        )
    return seeds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir", type=Path, default=Path("artifacts/trajectory_share")
    )
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--source-manifest", type=Path, default=None)
    parser.add_argument("--logdir", type=Path, default=Path("logs/MPE_simple_tag_v3"))
    parser.add_argument("--n-eps", type=int, default=200)
    parser.add_argument("--ckpt-seeds", type=_parse_seeds, default=(0, 1, 2))
    parser.add_argument("--rollout-seed", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument(
        "--prey-objective",
        choices=OBJECTIVE_TYPES,
        default="capture",
        help="One fixed prey checkpoint family; matched prey is unsupported.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()

    return {
        "sha": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(run("status", "--porcelain")),
        "environment_dirty": bool(
            run("status", "--porcelain", "--", "src/tag_objectives")
        ),
        "renderer_dirty": bool(
            run(
                "status",
                "--porcelain",
                "--",
                *(str(path) for path in RENDERER_SOURCE_FILES),
            )
        ),
    }


def _checkpoint_path(logdir: Path, objective: str, team: str, seed: int) -> Path:
    algorithm = f"mappo_objectives_{objective}"
    return logdir / (
        f"{algorithm}_MPE_simple_tag_v3_{team}_actor_seed0_vmap{seed}.safetensors"
    )


def _checkpoint_records(
    logdir: Path, seeds: tuple[int, ...], prey_objective: str
) -> list[dict[str, Any]]:
    requested = [
        (objective, "pred", seed) for objective in OBJECTIVE_TYPES for seed in seeds
    ] + [(prey_objective, "prey", seed) for seed in seeds]
    records = []
    for objective, team, seed in requested:
        path = _checkpoint_path(logdir, objective, team, seed)
        if not path.is_file():
            raise FileNotFoundError(f"missing checkpoint: {path}")
        records.append(
            {
                "objective": objective,
                "team": team,
                "seed": seed,
                "filename": path.name,
                "sha256": file_sha256(path),
            }
        )
    return records


def _load_verified_source(
    dataset_path: Path,
    manifest_path: Path,
    renderer_git: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest)
    if manifest.get("run_kind") != "full":
        raise ValueError("source manifest must record run_kind='full'")
    if manifest["git_sha"] != renderer_git["sha"]:
        raise ValueError(
            "check out the source manifest Git SHA before reconstructing scene data"
        )
    if renderer_git["environment_dirty"]:
        raise ValueError(
            "tag_objectives has uncommitted changes; scene replay is unsafe"
        )
    for distribution in ("jax", "jaxlib"):
        current = importlib.metadata.version(distribution)
        source = manifest.get("dependency_pins", {}).get(distribution)
        if source != current:
            raise ValueError(
                f"source {distribution} version {source!r} does not match {current!r}"
            )
    config = manifest.get("config", {})
    if config.get("synthetic") is not False:
        raise ValueError("synthetic data cannot be exported as a share bundle")
    if config.get("prey_objective") == "matched":
        raise ValueError("matched-prey datasets are confounded and cannot be shared")
    source_horizon = config.get("num_steps")
    if (
        not isinstance(source_horizon, int)
        or isinstance(source_horizon, bool)
        or not 1 <= source_horizon <= ENVIRONMENT_HORIZON_CAP
    ):
        raise ValueError(
            "source num_steps must be between 1 and the environment's "
            f"{ENVIRONMENT_HORIZON_CAP}-step cap"
        )
    data_stage = manifest.get("stages", {}).get("data", {})
    if data_stage.get("status") != "full_run_finished":
        raise ValueError("source manifest does not record a finished full data run")
    dataset_hash = file_sha256(dataset_path)
    artifact_hashes = {
        artifact.get("sha256")
        for artifact in manifest.get("artifacts", [])
        if artifact.get("kind") == "dataset_cache"
    }
    if dataset_hash not in artifact_hashes:
        raise ValueError("dataset SHA-256 is not bound by the source manifest")
    checkpoints = _verified_checkpoint_records(manifest)
    return manifest, checkpoints


def _verified_checkpoint_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate and canonicalize the specialist/fixed-control checkpoint matrix."""
    config = manifest["config"]
    seeds = tuple(int(seed) for seed in config["ckpt_seeds"])
    prey_objective = str(config["prey_objective"])
    expected_order = [
        (objective, "pred", seed)
        for objective in OBJECTIVE_TYPES
        for seed in seeds
    ] + [(prey_objective, "prey", seed) for seed in seeds]
    indexed: dict[tuple[str, str, int], dict[str, Any]] = {}
    for checkpoint in manifest.get("checkpoints", []):
        if not isinstance(checkpoint, dict):
            raise ValueError("every source checkpoint must be a mapping")
        required = {"alg", "team", "seed", "path", "sha256"}
        missing = required - set(checkpoint)
        if missing:
            raise ValueError(
                f"source checkpoint is missing fields: {sorted(missing)}"
            )
        algorithm = checkpoint["alg"]
        if not isinstance(algorithm, str) or not algorithm.startswith(
            "mappo_objectives_"
        ):
            raise ValueError("source checkpoint has an invalid objective algorithm")
        seed = checkpoint["seed"]
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("source checkpoint seed must be an integer")
        team = checkpoint["team"]
        if team not in {"pred", "prey"}:
            raise ValueError("source checkpoint team must be 'pred' or 'prey'")
        path_value = checkpoint["path"]
        if not isinstance(path_value, str) or not path_value:
            raise ValueError("source checkpoint path must be a non-empty string")
        key = (algorithm.removeprefix("mappo_objectives_"), team, seed)
        if key in indexed:
            raise ValueError(f"duplicate source checkpoint matrix entry: {key}")
        checkpoint_path = Path(path_value)
        resolved_path = (
            checkpoint_path if checkpoint_path.is_absolute() else ROOT / checkpoint_path
        )
        if not resolved_path.is_file():
            raise FileNotFoundError(f"source checkpoint is unavailable: {resolved_path}")
        if file_sha256(resolved_path) != checkpoint["sha256"]:
            raise ValueError(f"source checkpoint SHA-256 mismatch: {resolved_path}")
        indexed[key] = {
            "objective": key[0],
            "team": key[1],
            "seed": key[2],
            "filename": checkpoint_path.name,
            "sha256": checkpoint["sha256"],
        }
    if set(indexed) != set(expected_order):
        missing = sorted(set(expected_order) - set(indexed))
        unexpected = sorted(set(indexed) - set(expected_order))
        raise ValueError(
            "source checkpoint matrix does not match the required specialists and "
            f"fixed prey controls; missing={missing}, unexpected={unexpected}"
        )
    return [indexed[key] for key in expected_order]


def _validate_manifest_dataset_binding(manifest: dict[str, Any], dataset) -> None:
    """Bind every cached episode to the full pipeline manifest."""
    data = dataset.as_dict()
    config = manifest["config"]
    seeds = tuple(int(seed) for seed in config["ckpt_seeds"])
    expected_episodes = 3 * len(seeds) * int(config["n_eps"])
    if len(data["label"]) != expected_episodes:
        raise ValueError(
            f"dataset has {len(data['label'])} episodes; manifest expects {expected_episodes}"
        )
    if int(data["prey_act"].shape[1]) != int(config.get("num_steps", -1)):
        raise ValueError("dataset horizon does not match source manifest")
    if set(np.unique(data["ckpt_seed"]).tolist()) != set(seeds):
        raise ValueError("dataset checkpoint seeds do not match source manifest")
    for label in range(3):
        for seed in seeds:
            count = int(np.sum((data["label"] == label) & (data["ckpt_seed"] == seed)))
            if count != int(config["n_eps"]):
                raise ValueError(
                    f"label {label}, checkpoint {seed} has {count} episodes; "
                    f"expected {config['n_eps']}"
                )

    records = manifest["split"]["episodes"]
    if len(records) != expected_episodes:
        raise ValueError("source split record count does not match dataset")
    seen: set[int] = set()
    for record in records:
        index = int(record.get("index", -1))
        if index < 0 or index >= expected_episodes or index in seen:
            raise ValueError("source split has invalid or duplicate episode indices")
        seen.add(index)
        expected_key = [int(value) for value in data["env_seed"][index]]
        if (
            int(record["strategy_label"]) != int(data["label"][index])
            or int(record["checkpoint_seed"]) != int(data["ckpt_seed"][index])
            or record["environment_key"] != expected_key
            or int(record["valid_length"]) != int(data["valid_length"][index])
        ):
            raise ValueError(
                f"source split record does not match dataset episode {index}"
            )


def _bundle_paths(root: Path) -> dict[str, Path]:
    return {
        "dataset": root / "trajectories.npz",
        "png": root / "representative_trajectories.png",
        "pdf": root / "representative_trajectories.pdf",
        "metadata": root / "metadata.json",
        "source_manifest": root / "source_manifest.json",
        "renderer_sources": root / "renderer_sources.json",
    }


def _assert_safe_bundle_target(target: Path) -> None:
    """Reject symlinks and repository/home/temp roots as publication targets."""
    if target.is_symlink():
        raise ValueError(f"bundle target cannot be a symlink: {target}")
    resolved = target.resolve()
    repository = ROOT.resolve()
    forbidden = {
        repository,
        *repository.parents,
        Path.home().resolve(),
        Path(tempfile.gettempdir()).resolve(),
    }
    if resolved in forbidden:
        raise ValueError(f"refusing broad bundle target: {resolved}")


def _validate_bundle_directory(path: Path, *, allow_legacy: bool) -> None:
    """Require an exact file allowlist and a parseable bundle marker."""
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"bundle directory must be a real directory: {path}")
    names = {entry.name for entry in path.iterdir()}
    accepted = {BUNDLE_FILENAMES}
    if allow_legacy:
        accepted.add(LEGACY_BUNDLE_FILENAMES)
    if names not in accepted:
        raise ValueError(
            f"refusing to replace unrecognized directory {path}; found {sorted(names)}"
        )
    for name in names:
        if not (path / name).is_file() or (path / name).is_symlink():
            raise ValueError(f"bundle member must be a regular file: {path / name}")
    try:
        metadata = json.loads((path / "metadata.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"bundle marker is unreadable: {path / 'metadata.json'}") from error
    if metadata.get("schema") != BUNDLE_SCHEMA:
        raise ValueError(f"bundle marker has an unsupported schema: {path}")


def _publish_bundle(staging: Path, target: Path) -> None:
    """Swap a fully built directory into place, with rollback on failure."""
    _assert_safe_bundle_target(target)
    _validate_bundle_directory(staging, allow_legacy=False)
    backup: Path | None = None
    if target.exists():
        if not target.is_dir():
            raise ValueError(f"bundle target is not a directory: {target}")
        if any(target.iterdir()):
            _validate_bundle_directory(target, allow_legacy=True)
            backup = target.parent / f".{target.name}.backup-{uuid.uuid4().hex}"
            os.replace(target, backup)
        else:
            target.rmdir()
    try:
        os.replace(staging, target)
    except Exception:
        if backup is not None and not target.exists():
            os.replace(backup, target)
        raise
    if backup is not None:
        shutil.rmtree(backup, ignore_errors=True)


def _versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0], "numpy": np.__version__}
    for distribution in ("jax", "jaxmarl", "matplotlib"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _renderer_source_snapshot(renderer_git: dict[str, Any]) -> dict[str, Any]:
    files = []
    for relative_path in RENDERER_SOURCE_FILES:
        path = ROOT / relative_path
        files.append(
            {
                "path": str(relative_path),
                "sha256": file_sha256(path),
                "content": path.read_text(),
            }
        )
    return {
        "schema": "objective-trajectory-renderer-sources/v1",
        "git_sha": renderer_git["sha"],
        "renderer_dirty": renderer_git["renderer_dirty"],
        "files": files,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if (args.dataset is None) != (args.source_manifest is None):
        raise ValueError("--dataset and --source-manifest must be supplied together")
    if args.n_eps < 1 or not 1 <= args.num_steps <= ENVIRONMENT_HORIZON_CAP:
        raise ValueError(
            "n-eps must be positive and num-steps must be between 1 and the "
            f"environment's {ENVIRONMENT_HORIZON_CAP}-step cap"
        )

    target = args.out_dir.expanduser().resolve()
    _assert_safe_bundle_target(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not target.is_dir():
            raise ValueError(f"bundle target is not a directory: {target}")
        if any(target.iterdir()) and not args.overwrite:
            raise FileExistsError(
                f"refusing to replace non-empty bundle directory: {target}"
            )
        if any(target.iterdir()):
            _validate_bundle_directory(target, allow_legacy=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
    )
    outputs = _bundle_paths(staging)

    try:
        renderer_git = _git_state()
        source_manifest = None
        if args.dataset is None:
            if renderer_git["dirty"]:
                raise ValueError("fresh generation requires a clean Git worktree")
            if len(args.ckpt_seeds) < 3:
                raise ValueError("share generation requires at least three checkpoints")
            checkpoints = _checkpoint_records(
                args.logdir, args.ckpt_seeds, args.prey_objective
            )
            dataset = objective_dataset(
                n_eps=args.n_eps,
                ckpt_seeds=args.ckpt_seeds,
                rng0=args.rollout_seed,
                num_steps=args.num_steps,
                logdir=args.logdir,
                prey_type=args.prey_objective,
            )
            save_dataset_atomic(outputs["dataset"], dataset)
            source = {
                "mode": "fresh_generation",
                "git_sha": renderer_git["sha"],
                "git_dirty": False,
                "config": {
                    "n_eps": args.n_eps,
                    "ckpt_seeds": list(args.ckpt_seeds),
                    "rollout_seed": args.rollout_seed,
                    "num_steps": args.num_steps,
                    "prey_objective": args.prey_objective,
                    "synthetic": False,
                },
            }
            _write_json(outputs["source_manifest"], source)
            source_manifest_info = {
                "filename": outputs["source_manifest"].name,
                "sha256": file_sha256(outputs["source_manifest"]),
            }
        else:
            source_manifest, checkpoints = _load_verified_source(
                args.dataset, args.source_manifest, renderer_git
            )
            shutil.copyfile(args.dataset, outputs["dataset"])
            shutil.copyfile(args.source_manifest, outputs["source_manifest"])
            dataset = load_objective_dataset(outputs["dataset"])
            source = {
                "mode": "verified_pipeline_cache",
                "git_sha": source_manifest["git_sha"],
                "git_dirty": source_manifest["git_dirty"],
                "config": source_manifest["config"],
            }
            source_manifest_info = {
                "filename": outputs["source_manifest"].name,
                "sha256": file_sha256(outputs["source_manifest"]),
            }

        validation = validate_objective_dataset(dataset)
        if source_manifest is not None:
            _validate_manifest_dataset_binding(source_manifest, dataset)
        selection = select_representative_group(dataset)
        resources = reconstruct_resources(dataset, selection)
        prey_objective = str(source["config"]["prey_objective"])
        figure_info = render_representative_figure(
            dataset,
            selection,
            outputs["png"],
            outputs["pdf"],
            resource_pos=resources,
            prey_objective=prey_objective,
        )
        _write_json(outputs["renderer_sources"], _renderer_source_snapshot(renderer_git))
        metadata = {
            "schema": BUNDLE_SCHEMA,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "generator": {
                "script": "scripts/export_trajectory_bundle.py",
                "argv": list(sys.argv[1:] if argv is None else argv),
                "renderer_git": renderer_git,
                "versions": _versions(),
            },
            "source": source,
            "source_manifest": source_manifest_info,
            "rollout_protocol": {
                "objectives": list(OBJECTIVE_TYPES),
                "label_mapping": {
                    str(key): value for key, value in LABEL_NAMES.items()
                },
                "prey_policy_control": (f"fixed {prey_objective} checkpoint family"),
                "action_selection": "greedy argmax",
                "termination": "first capture or horizon",
                "valid_prefix": ("positions[:valid_length+1], actions[:valid_length]"),
                "matched_reset_keys_across_labels": True,
                "scene_reconstruction": (
                    "exact source Git SHA and JAX/JAXlib versions; clean "
                    "tag_objectives tree; recorded reset geometry rechecked; exact "
                    "renderer/exporter sources embedded in renderer_sources.json"
                ),
            },
            "validation": validation,
            "array_schema": dataset_schema(dataset),
            "checkpoints": checkpoints,
            "representative_selection": selection,
            "artifacts": {
                "dataset": {
                    "filename": outputs["dataset"].name,
                    "bytes": outputs["dataset"].stat().st_size,
                    "sha256": file_sha256(outputs["dataset"]),
                },
                "figure_png": {
                    "filename": outputs["png"].name,
                    "bytes": outputs["png"].stat().st_size,
                    "sha256": file_sha256(outputs["png"]),
                    **figure_info,
                },
                "figure_pdf": {
                    "filename": outputs["pdf"].name,
                    "bytes": outputs["pdf"].stat().st_size,
                    "sha256": file_sha256(outputs["pdf"]),
                },
                "renderer_sources": {
                    "filename": outputs["renderer_sources"].name,
                    "bytes": outputs["renderer_sources"].stat().st_size,
                    "sha256": file_sha256(outputs["renderer_sources"]),
                },
            },
            "claim_boundary": (
                "These are behavior rollouts from trained specialists. The bundle "
                "does not by itself establish representation recovery or downstream "
                "scientific success."
            ),
        }
        _write_json(outputs["metadata"], metadata)
        _publish_bundle(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    print(f"Wrote trajectory bundle to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
