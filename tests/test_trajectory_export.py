"""Tests for controlled trajectory validation and representative selection."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mopa.trajectory_export import (
    dataset_schema,
    load_objective_dataset,
    render_representative_figure,
    resource_visibility,
    select_representative_group,
    validate_objective_dataset,
)
from mopa.types import ObjectiveDataset, ObjectiveObservationDataset


def _toy_dataset(order: np.ndarray | None = None) -> ObjectiveDataset:
    groups = 3
    labels = 3
    n = groups * labels
    horizon = 4
    prey_pos = np.zeros((n, horizon + 1, 2), dtype=np.float32)
    pred_pos = np.zeros((n, horizon + 1, 1, 2), dtype=np.float32)
    lava_pos = np.zeros((n, 1, 2), dtype=np.float32)
    lava_rad = np.full((n, 1), 0.2, dtype=np.float32)
    prey_act = np.zeros((n, horizon), dtype=np.int32)
    pred_act = np.zeros((n, horizon, 1), dtype=np.int32)
    capture_t = np.full(n, -1, dtype=np.int32)
    captured = np.zeros(n, dtype=np.int32)
    survival = np.full(n, horizon, dtype=np.float32)
    pred_lava = np.zeros(n, dtype=np.float32)
    prey_lava = np.zeros(n, dtype=np.float32)
    resources = np.zeros(n, dtype=np.float32)
    coverage = np.zeros(n, dtype=np.float32)
    label_values = np.zeros(n, dtype=np.int32)
    checkpoint = np.zeros(n, dtype=np.int32)
    env_seed = np.zeros((n, 2), dtype=np.uint32)
    valid_length = np.full(n, horizon, dtype=np.int32)

    index = 0
    for group in range(groups):
        initial_prey = np.array([0.2 * group, -0.1 * group], dtype=np.float32)
        initial_pred = np.array([-0.2 * group, 0.1 * group], dtype=np.float32)
        lava = np.array([0.3 * group, -0.25], dtype=np.float32)
        for label in range(labels):
            label_values[index] = label
            checkpoint[index] = group
            env_seed[index] = [group + 10, group + 20]
            prey_pos[index] = initial_prey
            pred_pos[index, :, 0] = initial_pred
            lava_pos[index, 0] = lava
            coverage[index] = float(group + label)
            pred_lava[index] = float(abs(group - 1) * label)
            if group != 1 and label == 0:
                captured[index] = 1
                capture_t[index] = 2
                survival[index] = 2
                valid_length[index] = 2
                prey_pos[index, 2:] = prey_pos[index, 2]
                pred_pos[index, 2:] = pred_pos[index, 2]
            index += 1

    values = {
        "prey_pos": prey_pos,
        "pred_pos": pred_pos,
        "lava_pos": lava_pos,
        "lava_rad": lava_rad,
        "prey_act": prey_act,
        "pred_act": pred_act,
        "capture_t": capture_t,
        "captured": captured,
        "survival_time": survival,
        "pred_lava_steps": pred_lava,
        "prey_lava_steps": prey_lava,
        "resources_collected": resources,
        "pred_coverage": coverage,
        "label": label_values,
        "ckpt_seed": checkpoint,
        "env_seed": env_seed,
        "valid_length": valid_length,
    }
    if order is not None:
        values = {name: value[order] for name, value in values.items()}
    return ObjectiveDataset(**values)


def test_validation_and_selection_are_matched_and_row_order_invariant():
    dataset = _toy_dataset()
    summary = validate_objective_dataset(dataset)
    assert summary == {
        "episodes": 9,
        "horizon": 4,
        "predators": 1,
        "matched_groups": 3,
        "episodes_per_label": {"0": 3, "1": 3, "2": 3},
    }
    selection = select_representative_group(dataset)

    order = np.array([8, 2, 4, 1, 7, 0, 5, 6, 3])
    permuted = select_representative_group(_toy_dataset(order))
    assert selection["group_key"] == permuted["group_key"]
    assert selection["score"] == permuted["score"]


def test_exact_predator_observation_schema_and_legacy_loading(tmp_path: Path):
    legacy = _toy_dataset()
    legacy_path = tmp_path / "legacy.npz"
    np.savez_compressed(legacy_path, **legacy.as_dict())

    loaded_legacy = load_objective_dataset(legacy_path)
    assert type(loaded_legacy) is ObjectiveDataset

    values = legacy.as_dict()
    pred_obs = np.zeros((9, 5, 1, 17), dtype=np.float32)
    values["pred_obs"] = pred_obs
    observed = ObjectiveObservationDataset(**values)
    observed_path = tmp_path / "observed.npz"
    np.savez_compressed(observed_path, **observed.as_dict())

    loaded = load_objective_dataset(observed_path)
    assert isinstance(loaded, ObjectiveObservationDataset)
    np.testing.assert_array_equal(loaded.pred_obs, pred_obs)
    summary = validate_objective_dataset(loaded)
    assert summary["predator_observation_dim"] == 17
    assert dataset_schema(loaded)["pred_obs"] == {
        "shape": [9, 5, 1, 17],
        "dtype": "float32",
        "axes": [
            "episode",
            "time_including_initial",
            "predator",
            "observation_feature",
        ],
    }


def test_validation_rejects_malformed_or_unfrozen_predator_observations():
    values = _toy_dataset().as_dict()
    values["pred_obs"] = np.zeros((9, 4, 1, 17), dtype=np.float32)
    with np.testing.assert_raises_regex(ValueError, "pred_obs has shape"):
        validate_objective_dataset(values)

    values["pred_obs"] = np.zeros((9, 5, 1, 17), dtype=np.float32)
    values["pred_obs"][0, 3, 0, 0] = 1.0
    with np.testing.assert_raises_regex(
        ValueError, "predator observation tail is not frozen"
    ):
        validate_objective_dataset(values)


def test_validation_rejects_label_correlated_layouts():
    dataset = _toy_dataset()
    dataset.lava_pos[1, 0, 0] += 0.1
    try:
        validate_objective_dataset(dataset)
    except ValueError as error:
        assert "lava positions differs across labels" in str(error)
    else:
        raise AssertionError("mismatched layouts must be rejected")


def test_headless_figure_uses_only_valid_position_prefixes(tmp_path: Path):
    dataset = _toy_dataset()
    selection = select_representative_group(dataset)
    png = tmp_path / "figure.png"
    pdf = tmp_path / "figure.pdf"
    info = render_representative_figure(
        dataset,
        selection,
        png,
        pdf,
        resource_pos=np.array([[0.8, -0.8], [-0.8, 0.8]], dtype=np.float32),
        prey_objective="capture",
        dpi=80,
    )
    assert png.stat().st_size > 0
    assert pdf.stat().st_size > 0
    assert info["common_abs_axis_limit"] > 2.0
    for label, index in selection["indices"].items():
        assert info["line_point_counts"][label] == dataset.valid_length[index] + 1
        assert info["visible_resource_counts"][label] == 2


def test_resource_visibility_matches_collection_timing():
    resources = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    prey = np.array(
        [
            [0.0, 0.0],  # Reset proximity does not collect.
            [0.15, 0.0],  # The environment uses a strict radius.
            [0.149, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ],
        dtype=np.float32,
    )
    visible = resource_visibility(
        prey, resources, 0.15, expected_collected=2
    )
    np.testing.assert_array_equal(
        visible,
        [
            [True, True],
            [True, True],
            [False, True],
            [False, False],
            [False, False],
        ],
    )


def test_resource_visibility_rejects_a_wrong_final_count():
    with np.testing.assert_raises_regex(ValueError, "implies 1 collections"):
        resource_visibility(
            np.array([[2.0, 0.0], [0.0, 0.0]], dtype=np.float32),
            np.array([[0.0, 0.0]], dtype=np.float32),
            0.15,
            expected_collected=0,
        )


def test_validation_rejects_malformed_lava_rank_and_dtype():
    dataset = _toy_dataset()
    values = dataset.as_dict()
    values["lava_pos"] = values["lava_pos"][:, 0]
    values["lava_rad"] = values["lava_rad"][:, 0]
    try:
        validate_objective_dataset(values)
    except ValueError as error:
        assert "lava_pos must have shape" in str(error)
    else:
        raise AssertionError("rank-two lava positions must be rejected")

    values = _toy_dataset().as_dict()
    values["prey_pos"] = values["prey_pos"].astype(np.float64)
    try:
        validate_objective_dataset(values)
    except ValueError as error:
        assert "prey_pos must use float32" in str(error)
    else:
        raise AssertionError("unexpected dataset dtypes must be rejected")
