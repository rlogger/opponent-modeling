"""Saved-trace inspection only: synthetic arrays, no model or environment imports."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def exporter():
    path = Path(__file__).resolve().parents[1] / "scripts/visualize_controllers.py"
    spec = importlib.util.spec_from_file_location("controller_visualization_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def trace(label=0, controller="mappo", *, nan_padding=True):
    batch, horizon = 2, 5
    lengths = np.array([2, 4], np.int32)
    mask = np.arange(horizon)[None] < lengths[:, None]
    state = np.zeros((batch, horizon + 1, 66), np.float32)
    state[:, :, 8:40] = np.linspace(-1, 1, 32)
    state[:, :, 56:62] = [-0.8, 0.4, 0.3, -0.6, 0.6, 0.8]
    state[:, :, 62:65] = [0.2, 0.3, 0.25]
    action = np.zeros((batch, horizon, 2), np.float32)
    red = np.zeros_like(action)
    context = np.zeros((batch, horizon, 8 if controller == "tdmpc" else 3), np.float32)
    terminated, truncated = np.zeros_like(mask), np.zeros_like(mask)
    for row, length in enumerate(lengths):
        start = np.array([-0.7 + row * 0.2, 0.1, 0.8, -0.2])
        direction = np.array([0.03, 0.02, -0.01, 0.04])
        if controller == "tdmpc":
            direction = np.array([-0.02, 0.04, 0.03, 0.01])
        for step in range(horizon + 1):
            state[row, step, :4] = start + direction * step * (1 + label + row * 0.3)
            state[row, step, 65] = step / 100
        state[row, 1:, 40] = 1
        action[row, :length] = [0.3, -0.2]
        red[row, :length] = [-0.1, 0.4]
        if controller == "tdmpc":
            context[row, :length] = np.arange(length)[:, None] * np.linspace(0.1, 0.8, 8)
        state[row, length + 1:] = np.nan if nan_padding else state[row, length]
        action[row, length:] = np.nan if nan_padding else 0
        red[row, length:] = np.nan if nan_padding else 0
        context[row, length:] = np.nan if nan_padding else 0
        (terminated if row == 0 else truncated)[row, length - 1] = True
    return dict(state=state, blue_action=action, red_action=red, valid_mask=mask,
                valid_length=lengths, terminated_capture=terminated, truncated_timeout=truncated,
                context=context, dataset_episode=np.array([label * 100, label * 100 + 1]),
                objective_label=np.full(batch, label), checkpoint_seed=np.full(batch, 2),
                environment_seed=np.array([[100, 200], [101, 201]], np.uint32),
                step_seed=np.array([[300, 400], [301, 401]], np.uint32))


def write_cohort(directory, *, nan_padding=True):
    directory.mkdir(parents=True, exist_ok=True)
    runs = []
    for label, objective in enumerate(("capture", "risk", "curious")):
        for short, controller, context in (("mappo", "mappo_prey", "zero"), ("tdmpc", "tdmpc", "online")):
            name = f"{objective}__{controller}__{context}.npz"
            np.savez(directory / name, **trace(label, short, nan_padding=nan_padding))
            runs.append({"opponent": objective, "controller": controller,
                         "context_mode": context, "recordings": {"transitions": name}})
    report = {"runs": runs, "checkpoint_artifacts": {"agent.msgpack": "saved-controller"},
              "prey_control_checkpoint": {"sha256": "saved-mappo"}, "planner": {"horizon": 3},
              "mode": "factored", "encoder": "identity", "opponent_model": "frozen_zero_s",
              "git_sha": "synthetic-source", "git_dirty": False}
    path = directory / "evaluation.json"
    path.write_text(json.dumps(report))
    return path


def test_load_trace_accepts_valid_prefix_and_ignores_padding(exporter, tmp_path):
    values = trace()
    path = tmp_path / "trace.npz"
    np.savez(path, **values)
    loaded = exporter.load_trace(path)
    np.testing.assert_array_equal(loaded["valid_length"], [2, 4])
    assert np.isnan(loaded["state"][0, 3:]).all()  # padding is absent data
    assert loaded["state"].shape == (2, 6, 66)


@pytest.mark.parametrize("damage", [
    "noncontiguous", "zero_length", "float_length", "blue_bounds", "red_bounds",
    "nonfinite_state", "nonfinite_context", "flag_shape", "both_flags", "key_shape",
])
def test_load_trace_rejects_invalid_recorded_contract(exporter, tmp_path, damage):
    values = trace()
    if damage == "noncontiguous":
        values["valid_mask"][0, :3] = [True, False, True]
    elif damage == "zero_length":
        values["valid_length"][0] = 0
    elif damage == "float_length":
        values["valid_length"] = values["valid_length"].astype(float)
    elif damage in {"blue_bounds", "red_bounds"}:
        values["blue_action" if damage == "blue_bounds" else "red_action"][0, 0, 0] = 1.01
    elif damage == "nonfinite_state":
        values["state"][0, 1, 0] = np.nan
    elif damage == "nonfinite_context":
        values["context"][0, 1, 0] = np.nan
    elif damage == "flag_shape":
        values["truncated_timeout"] |= values["terminated_capture"]
        values["terminated_capture"] = np.zeros((2, 1), bool)
    elif damage == "both_flags":
        values["truncated_timeout"][0, 1] = True
    elif damage == "key_shape":
        values["environment_seed"] = values["environment_seed"][:, :1]
    path = tmp_path / "invalid.npz"
    np.savez(path, **values)
    with pytest.raises(ValueError):
        exporter.load_trace(path)


@pytest.mark.parametrize("key", ["environment_seed", "step_seed", "dataset_episode", "checkpoint_seed"])
def test_pairing_requires_matching_episode_provenance(exporter, tmp_path, key):
    report = write_cohort(tmp_path)
    path = tmp_path / "capture__tdmpc__online.npz"
    values = trace(controller="tdmpc")
    values[key].flat[0] += 1
    np.savez(path, **values)
    with pytest.raises(ValueError, match=key):
        exporter.build_bundle(report)


def test_pairing_requires_full_initial_state_not_only_agent_positions(exporter, tmp_path):
    report = write_cohort(tmp_path)
    values = trace(controller="tdmpc")
    values["state"][0, 0, 39] += 0.5  # only a resource coordinate differs
    np.savez(tmp_path / "capture__tdmpc__online.npz", **values)
    with pytest.raises(ValueError, match="entire initial state"):
        exporter.build_bundle(report)


def test_bundle_uses_all_saved_episodes_and_only_valid_frames(exporter, tmp_path):
    clean, clean_metadata = exporter.build_bundle(write_cohort(tmp_path / "clean", nan_padding=False))
    padded, padded_metadata = exporter.build_bundle(write_cohort(tmp_path / "padded", nan_padding=True))
    assert clean == padded
    assert len(padded["episodes"]) == 12 and len(padded["maps"]) == 6
    assert {row["controller"] for row in padded["episodes"]} == {"mappo", "tdmpc"}
    for row in padded["episodes"]:
        assert len(row["xy"]) == len(row["collected"]) == row["length"] + 1
        assert len(row["blue"]) == len(row["red"]) == row["length"]
        assert len(row["pc"]) == 2
        assert row["stop"] == ("capture" if row["pair"] == 0 else "timeout")
        if row["controller"] == "tdmpc":
            assert len(row["zpc"]) == row["length"]
        else:
            assert "zpc" not in row
    json.dumps(padded, allow_nan=False)
    for key in ("trajectory_pca", "context_pca"):
        assert clean_metadata[key] == padded_metadata[key]
    assert not padded_metadata["policy_inference"]
    assert not padded_metadata["new_rollouts"]
    assert not padded_metadata["retraining"]


def test_joint_trajectory_features_are_translation_invariant(exporter):
    positions = np.array([[0, 1, 2, 3], [1, 3, 4, 2], [3, 4, 5, 4]], np.float64)
    features = exporter.trajectory_features(positions)
    translated = exporter.trajectory_features(positions + [10, -20, 30, -40])
    assert features.shape == (128,)
    np.testing.assert_allclose(translated, features, atol=1e-14)
    np.testing.assert_allclose(exporter.trajectory_features(positions * 2), features * 2, atol=1e-14)
    np.testing.assert_array_equal(features[:4], 0)


def test_pca_is_centered_label_free_and_deterministic(exporter):
    values = np.random.default_rng(3).normal(size=(9, 5)) * [1, 8, 2, 0.5, 3]
    coords, mean, components = exporter.pca(values)
    again = exporter.pca(values.copy())
    np.testing.assert_allclose(mean, values.mean(0), atol=1e-14)
    np.testing.assert_allclose(coords.mean(0), 0, atol=1e-14)
    np.testing.assert_allclose(components @ components.T, np.eye(2), atol=1e-14)
    np.testing.assert_allclose(coords, (values - mean) @ components.T, atol=1e-14)
    for actual, expected in zip(again, (coords, mean, components), strict=True):
        np.testing.assert_array_equal(actual, expected)
    for component in components:
        assert component[np.argmax(np.abs(component))] >= 0


@pytest.mark.parametrize("invalid", [np.ones((3,)), np.ones((1, 4)), np.full((3, 4), np.nan)])
def test_pca_rejects_malformed_or_nonfinite_input(exporter, invalid):
    with pytest.raises(ValueError):
        exporter.pca(invalid)
