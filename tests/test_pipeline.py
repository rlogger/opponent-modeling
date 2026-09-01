"""End-to-end checks for the Part 1 experiment driver."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")
pytest.importorskip("optax")

from mopa.manifest import validate_manifest


def _driver_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_part1.py"
    spec = importlib.util.spec_from_file_location("run_part1_pipeline", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_causal_sample_latents_never_select_current_or_future_steps():
    driver = _driver_module()
    latents = np.zeros((2, 6, 2), dtype=np.float32)
    latents[..., 0] = np.arange(6)[None, :]
    latents[..., 1] = 100.0 + np.arange(6)[None, :]

    selected = driver._causal_sample_latents(
        latents,
        episode_ids=np.array([0, 1, 0]),
        timesteps=np.array([1, 3, 5]),
    )

    np.testing.assert_array_equal(selected[:, 0], [0.0, 2.0, 4.0])
    np.testing.assert_array_equal(selected[:, 1], [100.0, 102.0, 104.0])


def test_synthetic_smoke_runs_the_complete_pipeline(tmp_path):
    driver = _driver_module()
    out = tmp_path / "part1_synthetic.json"

    result = driver.main(
        [
            "--synthetic",
            "--run-kind",
            "full",
            "--n-eps",
            "3",
            "--ckpt-seeds",
            "0,1",
            "--encoder-seeds",
            "0",
            "--bc-seeds",
            "0",
            "--ctx",
            "2",
            "--prefixes",
            "2,4",
            "--lat",
            "2",
            "--hid",
            "8",
            "--encoder-steps",
            "1",
            "--bc-steps",
            "1",
            "--calibration-bins",
            "3",
            "--out",
            str(out),
        ]
    )

    assert result == 0
    manifest = json.loads(out.read_text())
    validate_manifest(manifest)
    assert manifest["schema_version"] == 2
    assert manifest["run_kind"] == "smoke"
    assert manifest["checkpoints"] == []
    assert manifest["stages"]["cpl_preferences"]["status"] == "not_run"
    assert all(
        stage["status"] == "smoke_passed"
        for name, stage in manifest["stages"].items()
        if name != "cpl_preferences"
    )
    assert manifest["split"]["shared_across_encoder_and_bc_seeds"] is True
    assert len(manifest["split"]["episodes"]) == 18
    assert all(
        len(episode["environment_key"]) == 2
        and episode["valid_length"] > 0
        for episode in manifest["split"]["episodes"]
    )
    labels_by_environment_key: dict[tuple[int, int], set[int]] = {}
    for episode in manifest["split"]["episodes"]:
        key = tuple(episode["environment_key"])
        labels_by_environment_key.setdefault(key, set()).add(
            episode["strategy_label"]
        )
    assert all(labels == {0, 1, 2} for labels in labels_by_environment_key.values())

    metrics = manifest["metrics"]
    expected_models = {
        "gru_jepa",
        "fixed_window_jepa",
        "beta_vae",
        "random_projection",
        "sa_short_seq_action_decoder_vae",
        "supervised_oracle",
    }
    assert set(metrics["representations"]) == expected_models
    for name in expected_models - {"supervised_oracle"}:
        report = metrics["representations"][name]
        assert len(report["heldout_probe"]["runs"]) == 1
        assert len(report["heldout_gmm_ari"]["runs"]) == 1
        reliability = report["calibration"]["reliability"][0]["bins"]
        assert sum(reliability["count"]) == manifest["split"]["n_val"]

    assert metrics["anytime"]["requested_prefixes"] == [2, 4]
    assert set(metrics["anytime"]["gru_jepa"]) == {"2", "4"}
    action_decoder = metrics["representations"][
        "sa_short_seq_action_decoder_vae"
    ]
    assert action_decoder["state_feature_mode"] == "causal_past"
    assert action_decoder["contains_current_actions"] is True
    assert action_decoder["contains_future_state"] is False
    assert action_decoder["temporal_scope"] == "full_valid_episode_post_hoc"
    assert len(action_decoder["window_unit_probe"]["runs"]) == 1
    assert len(action_decoder["decoder_action_accuracy"]["runs"]) == 1
    assert metrics["bc"]["point_z"]["conditioning"].endswith("t_minus_1")
    assert metrics["bc"]["point_z"]["contains_future_episode_information"] is False
    assert metrics["sequential_belief_mixture"]["belief_timing"] == (
        "predictive_before_observed_action"
    )
    assert 0.0 <= metrics["sequential_belief_mixture"]["top1"] <= 1.0
    assert metrics["sequential_belief_mixture"]["nll"] >= 0.0
    assert (
        metrics["sequential_belief_mixture"]["posterior_strategy_belief"]
        ["mean_entropy"]
        >= 0.0
    )
    assert set(metrics["environment"]["heldout"]) == {
        "capture",
        "risk",
        "curious",
    }


def test_full_run_rejects_underpowered_or_confounded_settings(tmp_path):
    driver = _driver_module()

    assert driver.main(
        [
            "--run-kind",
            "full",
            "--ckpt-seeds",
            "0",
            "--encoder-seeds",
            "0",
            "--bc-seeds",
            "0",
            "--out",
            str(tmp_path / "underpowered.json"),
        ]
    ) == 2
    assert driver.main(
        [
            "--run-kind",
            "full",
            "--prey-objective",
            "matched",
            "--out",
            str(tmp_path / "confounded.json"),
        ]
    ) == 2
    assert driver.main(
        [
            "--run-kind",
            "full",
            "--action-decoder-state-mode",
            "legacy_forward",
            "--out",
            str(tmp_path / "leaky.json"),
        ]
    ) == 2
    assert driver.main(
        [
            "--run-kind",
            "full",
            "--action-decoder-steps",
            "0",
            "--out",
            str(tmp_path / "zero_steps.json"),
        ]
    ) == 2
