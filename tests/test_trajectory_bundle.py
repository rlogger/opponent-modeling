"""Fail-closed and atomic-publication checks for the share-bundle CLI."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _export_module():
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "export_trajectory_bundle.py"
    )
    spec = importlib.util.spec_from_file_location("trajectory_bundle_export", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_marked_bundle(
    exporter, path: Path, *, token: str, legacy: bool = False
) -> dict[str, bytes]:
    path.mkdir()
    names = (
        exporter.LEGACY_BUNDLE_FILENAMES if legacy else exporter.BUNDLE_FILENAMES
    )
    for name in names:
        value = f"{token}:{name}\n"
        if name == "metadata.json":
            value = json.dumps({"schema": exporter.BUNDLE_SCHEMA, "token": token})
        (path / name).write_text(value)
    return {entry.name: entry.read_bytes() for entry in path.iterdir()}


def test_source_mode_rejects_an_invalid_pipeline_manifest(tmp_path: Path):
    exporter = _export_module()
    dataset = tmp_path / "dataset.npz"
    dataset.write_bytes(b"not reached")
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"schema_version": 2}\n')

    with pytest.raises(ValueError, match="manifest missing keys"):
        exporter._load_verified_source(
            dataset,
            manifest,
            {
                "sha": "a" * 40,
                "branch": "test",
                "dirty": False,
                "environment_dirty": False,
            },
        )


def test_failed_overwrite_preserves_the_previous_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    exporter = _export_module()
    target = tmp_path / "bundle"
    previous = _write_marked_bundle(exporter, target, token="old", legacy=True)
    dataset = tmp_path / "dataset.npz"
    manifest = tmp_path / "manifest.json"
    dataset.write_bytes(b"invalid")
    manifest.write_text("{}\n")

    monkeypatch.setattr(
        exporter,
        "_git_state",
        lambda: {
            "sha": "a" * 40,
            "branch": "test",
            "dirty": True,
            "environment_dirty": False,
            "renderer_dirty": True,
        },
    )

    def fail_source(*_args, **_kwargs):
        raise ValueError("bad replacement")

    monkeypatch.setattr(exporter, "_load_verified_source", fail_source)
    with pytest.raises(ValueError, match="bad replacement"):
        exporter.main(
            [
                "--dataset",
                str(dataset),
                "--source-manifest",
                str(manifest),
                "--out-dir",
                str(target),
                "--overwrite",
            ]
        )

    assert {entry.name: entry.read_bytes() for entry in target.iterdir()} == previous
    assert not list(tmp_path.glob(".bundle.staging-*"))


def test_publish_swaps_complete_directories(tmp_path: Path):
    exporter = _export_module()
    target = tmp_path / "bundle"
    _write_marked_bundle(exporter, target, token="old", legacy=True)
    staging = tmp_path / ".bundle.staging-test"
    expected = _write_marked_bundle(exporter, staging, token="new")

    exporter._publish_bundle(staging, target)

    assert {entry.name: entry.read_bytes() for entry in target.iterdir()} == expected
    assert not list(tmp_path.glob(".bundle.backup-*"))


def test_publish_rejects_an_unrecognized_existing_directory(tmp_path: Path):
    exporter = _export_module()
    target = tmp_path / "unrelated"
    target.mkdir()
    marker = target / "keep.txt"
    marker.write_text("do not delete\n")
    staging = tmp_path / ".unrelated.staging-test"
    _write_marked_bundle(exporter, staging, token="new")

    with pytest.raises(ValueError, match="unrecognized directory"):
        exporter._publish_bundle(staging, target)

    assert marker.read_text() == "do not delete\n"


def test_broad_targets_and_horizons_above_environment_cap_are_rejected(
    tmp_path: Path,
):
    exporter = _export_module()
    with pytest.raises(ValueError, match="broad bundle target"):
        exporter._assert_safe_bundle_target(exporter.ROOT)
    with pytest.raises(ValueError, match="100-step cap"):
        exporter.main(
            ["--num-steps", "101", "--out-dir", str(tmp_path / "bundle")]
        )


def test_checkpoint_matrix_requires_all_specialists_and_fixed_prey(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    exporter = _export_module()
    monkeypatch.setattr(exporter, "ROOT", tmp_path)
    config = {"ckpt_seeds": [0, 1, 2], "prey_objective": "capture"}
    checkpoints = []
    requested = [
        (objective, "pred", seed)
        for objective in exporter.OBJECTIVE_TYPES
        for seed in config["ckpt_seeds"]
    ] + [("capture", "prey", seed) for seed in config["ckpt_seeds"]]
    for objective, team, seed in requested:
        path = Path("logs") / f"{objective}-{team}-{seed}.safetensors"
        absolute = tmp_path / path
        absolute.parent.mkdir(exist_ok=True)
        absolute.write_bytes(f"{objective}:{team}:{seed}".encode())
        checkpoints.append(
            {
                "alg": f"mappo_objectives_{objective}",
                "team": team,
                "seed": seed,
                "path": str(path),
                "sha256": exporter.file_sha256(absolute),
            }
        )
    manifest = {"config": config, "checkpoints": checkpoints}
    assert len(exporter._verified_checkpoint_records(manifest)) == 12

    manifest["checkpoints"][-1] = dict(manifest["checkpoints"][0])
    with pytest.raises(ValueError, match="duplicate source checkpoint"):
        exporter._verified_checkpoint_records(manifest)
