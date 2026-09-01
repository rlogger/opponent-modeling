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
        episode_ids=np.array([0, 0, 1, 0]),
        timesteps=np.array([0, 1, 3, 5]),
    )

    np.testing.assert_array_equal(selected[0], np.zeros(2))
    np.testing.assert_array_equal(selected[1:], latents[[0, 1, 0], [0, 2, 4]])


def test_split_local_derangement_is_deterministic_and_never_crosses_split():
    driver = _driver_module()
    validation = np.array([False] * 8 + [True] * 4)
    checkpoints = np.repeat(np.arange(3), 4)

    permutation = driver._split_local_episode_derangement(
        validation, seed=7, checkpoint_ids=checkpoints
    )

    np.testing.assert_array_equal(validation[permutation], validation)
    np.testing.assert_array_equal(checkpoints[permutation], checkpoints)
    assert np.all(permutation != np.arange(len(validation)))
    np.testing.assert_array_equal(
        driver._split_local_episode_derangement(
            validation, seed=7, checkpoint_ids=checkpoints
        ),
        permutation,
    )


def test_dataset_cache_preserves_exact_observation_subclass(tmp_path):
    from mopa.types import ObjectiveObservationDataset

    driver = _driver_module()
    expected = driver._synthetic_objective_dataset(2, (0,), 8)
    cache = tmp_path / "observed.npz"
    np.savez_compressed(cache, **expected.as_dict())
    args = driver.build_parser().parse_args(["--dataset-cache", str(cache)])

    loaded, source = driver._load_or_create_dataset(args, (0,), (2, 4))

    assert source == "cache"
    assert type(loaded) is ObjectiveObservationDataset
    np.testing.assert_array_equal(loaded.pred_obs, expected.pred_obs)


def test_bc_requires_paired_seed_lists_and_exact_observations(tmp_path, capsys):
    driver = _driver_module()
    assert driver.main(
        [
            "--synthetic",
            "--encoder-seeds",
            "0,1",
            "--bc-seeds",
            "0",
            "--out",
            str(tmp_path / "unpaired.json"),
        ]
    ) == 2
    assert "equal length" in capsys.readouterr().err

    observed = driver._synthetic_objective_dataset(2, (0,), 8)
    legacy_cache = tmp_path / "legacy.npz"
    np.savez_compressed(
        legacy_cache,
        **{
            name: value
            for name, value in observed.as_dict().items()
            if name != "pred_obs"
        },
    )
    assert driver.main(
        [
            "--synthetic",
            "--dataset-cache",
            str(legacy_cache),
            "--encoder-seeds",
            "0",
            "--bc-seeds",
            "0",
            "--out",
            str(tmp_path / "legacy.json"),
        ]
    ) == 2
    assert "requires exact pred_obs" in capsys.readouterr().err


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
    bc = metrics["bc"]
    assert bc["condition_dim"] == 3
    assert bc["feature_schema"] == "exact_predator_observation_plus_condition"
    assert bc["real_z"]["conditioning"].endswith("t_minus_1")
    assert bc["contains_future_episode_information"] is False
    assert bc["initial_latent"] == "zero_at_t0"
    expected_samples = sum(
        episode["valid_length"] for episode in manifest["split"]["episodes"]
    )
    for arm in ("no_z", "real_z", "shuffled_z", "oracle"):
        assert len(bc[arm]["per_seed"]) == 1
        run = bc[arm]["per_seed"][0]
        assert run["n_train"] + run["n_val"] == expected_samples
        assert run["encoder_seed"] == run["bc_seed"] == 0
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
