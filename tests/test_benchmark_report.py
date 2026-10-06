"""Aggregation must not turn episode replication into training-seed evidence."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from mopa.benchmark_report import DEFAULT_COMPARISONS, summarize_benchmark

ARMS = ("0s", "bc", "implicit", "ppo_z", "ppo")
SEEDS = (0, 1, 2)


def records():
    return [
        {
            "arm": arm, "seed": seed, "objective": objective,
            "episode_keys": [[10, i] for i in range(4)],
            "metrics": {
                "blue_return": np.array([-3., -1., 1., 3.]) + 10 * seed + offset + 100 * objective_index,
                "captured": [0, 1, 0, 1],
                "resources_collected": [1, 2, 3, 4],
            },
        }
        for arm, offset in zip(ARMS, (5, 3, 0, 4, 1), strict=True)
        for seed in SEEDS
        for objective_index, objective in enumerate(("capture", "risk", "curious"))
    ]


def summarize(rows, **kwargs):
    return summarize_benchmark(rows, ARMS, SEEDS, bootstrap_samples=128, **kwargs)


def test_seed_standard_deviation_and_equal_objective_weight():
    result = summarize(records())
    item = result["results"]["0s"]["blue_return"]["overall"]
    assert item["seed_means"] == {"0": 105., "1": 115., "2": 125.}
    assert item["mean"] == 115. and item["seed_std"] == 10.
    assert result["results"]["0s"]["blue_return"]["by_objective"]["capture"]["mean"] == 15.
    assert result["uncertainty"]["limited_training_seeds"] is True
    json.dumps(result, allow_nan=False)


def test_paired_reset_bootstrap_preserves_constant_arm_difference():
    result = summarize(records())
    assert len(result["comparisons"]) == len(DEFAULT_COMPARISONS)
    contrast = result["comparisons"]["0s_minus_bc"]["metrics"]["blue_return"]
    assert contrast["overall"]["mean"] == 2.
    assert contrast["overall"]["seed_std"] == 0.
    assert contrast["overall"]["ci95"] == [2., 2.]
    for objective in contrast["by_objective"].values():
        assert objective["ci95"] == [2., 2.]


def test_bootstrap_repeatable_and_record_order_independent():
    rows = records()
    expected = summarize(rows, bootstrap_seed=7)
    np.random.default_rng(5).shuffle(rows)
    assert summarize(rows, bootstrap_seed=7) == expected
    assert summarize(rows, bootstrap_seed=8) != expected


@pytest.mark.parametrize("failure", ["missing", "duplicate", "keys", "counts", "nan", "metric", "capture", "resources", "seed", "arm", "repeated_keys"])
def test_reject_incomplete_or_unmatched_records(failure):
    rows = records()
    if failure == "missing":
        rows.pop()
    elif failure == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif failure == "keys":
        rows[0]["episode_keys"].reverse()
    elif failure == "counts":
        rows[0]["episode_keys"].pop()
    elif failure == "nan":
        rows[0]["metrics"]["blue_return"][0] = np.nan
    elif failure == "metric":
        del rows[0]["metrics"]["captured"]
    elif failure == "capture":
        rows[0]["metrics"]["captured"][0] = 0.5
    elif failure == "resources":
        rows[0]["metrics"]["resources_collected"][0] = -1
    elif failure == "seed":
        rows[0]["seed"] = 99
    elif failure == "arm":
        rows[0]["arm"] = "unexpected"
    elif failure == "repeated_keys":
        rows[0]["episode_keys"][0] = rows[0]["episode_keys"][1]
    with pytest.raises(ValueError):
        summarize(rows)


def test_reject_missing_entire_training_seed_and_invalid_expectations():
    with pytest.raises(ValueError, match="missing records"):
        summarize([r for r in records() if r["seed"] != 2])
    with pytest.raises(ValueError, match="missing records"):
        summarize([r for r in records() if r["objective"] != "risk"])
    with pytest.raises(ValueError, match="two distinct"):
        summarize_benchmark(records(), ARMS, [0])
    with pytest.raises(ValueError, match="distinct pairs"):
        summarize(records(), comparisons=[("0s", "absent")])


def test_custom_subset_and_different_reset_keys_between_seeds_are_explicit():
    rows = [r for r in records() if r["arm"] in ("bc", "implicit")]
    for row in rows:
        row["episode_keys"] = [[row["seed"], i] for i in range(4)]
    result = summarize_benchmark(
        rows, ["bc", "implicit"], SEEDS,
        comparisons=[("bc", "implicit")], bootstrap_samples=64,
    )
    assert result["comparisons"]["bc_minus_implicit"]["metrics"]["blue_return"]["overall"]["mean"] == 3.


def test_bootstrap_keeps_objectives_in_same_reset_cluster():
    rows = records()
    for row in rows:
        sign = {"capture": 1., "risk": -1., "curious": 0.}[row["objective"]]
        row["metrics"]["blue_return"] = sign * np.array([-3., -1., 1., 3.])
    result = summarize(rows)
    for arm in ARMS:
        assert result["results"][arm]["blue_return"]["overall"]["ci95"] == [0., 0.]
