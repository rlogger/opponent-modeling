"""Integer narrowing cannot turn invalid action labels into valid actions."""
import numpy as np
import pytest

from mopa.bc import bc_metrics_from_logits, fit_bc


@pytest.mark.parametrize("value", [2**32, -(2**32), float(2**32), float(-(2**32)), np.uint64(2**64 - 1)])
def test_discrete_action_bounds_are_checked_before_int32_conversion(value):
    targets = np.array([value])
    with pytest.raises(ValueError, match="within n_actions"):
        fit_bc(np.ones((1, 2), np.float32), targets, 0, steps=0)
    with pytest.raises(ValueError, match="outside the action vocabulary"):
        bc_metrics_from_logits(np.array([[1., 0., 0., 0., 0.]]), targets)

def test_valid_action_extremes_remain_valid():
    metrics = bc_metrics_from_logits(np.array([[2.,0.,0.,0.,0.], [0.,0.,0.,0.,2.]]), np.array([0,4], np.int64))
    assert metrics["accuracy"] == 1.0
