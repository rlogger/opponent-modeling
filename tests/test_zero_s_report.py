"""Regression checks for the interpretation of continuous 0s diagnostics."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def report():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_0s_world_model.py"
    spec = importlib.util.spec_from_file_location("zero_s_report_driver", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_action_error_weights_episodes_then_types_equally(report):
    # Capture has two episodes of unequal lengths; the other types have one.
    # Large padded errors must contribute nothing.
    errors = np.array([[0, 900, 900], [6, 6, 6], [2, 2, 900], [8, 8, 8]])
    mask = np.array([[1, 0, 0], [1, 1, 1], [1, 1, 0], [1, 1, 1]], dtype=bool)
    labels = np.array([0, 0, 1, 2])

    result = report.balanced_action_error(errors, mask, labels)

    assert result["per_type"] == {"capture": 3.0, "risk": 2.0, "curious": 8.0}
    assert result["macro_episode_mse"] == pytest.approx(13 / 3)
    assert result["macro_episode_mse"] != pytest.approx(errors[mask].mean())


def test_prototype_match_compares_rows_for_each_fixed_target_column(report):
    # Column minima are capture/capture/curious; row minima would incorrectly
    # produce risk/curious/curious and a different success count.
    matrix = np.array([[4, 1, 8], [9, 3, 2], [5, 6, 0]], dtype=np.float32)

    result = report.prototype_match_summary(matrix)

    assert result["best_prototype_per_specialist"] == ["capture", "capture", "curious"]
    assert result["correct_best_prototypes"] == 2


def test_average_diagonal_advantage_does_not_imply_three_correct_types(report):
    # A favorable aggregate diagonal can coexist with every target preferring
    # the risk prototype. The report must count actual per-target winners.
    matrix = np.array([[3, 4, 6], [2, 1, 4], [4, 5, 5]], dtype=np.float32)
    assert np.diag(matrix).mean() < matrix[~np.eye(3, dtype=bool)].mean()

    result = report.prototype_match_summary(matrix)

    assert result["best_prototype_per_specialist"] == ["risk", "risk", "risk"]
    assert result["correct_best_prototypes"] == 1
