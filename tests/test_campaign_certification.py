"""Reporting cannot silently change contrasts, pairings, or failed criteria."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from mopa.benchmark_report import summarize_benchmark

ARMS = ["implicit", "bc", "0s", "ppo", "ppo_z"]
SEEDS = [0, 1, 2]


def records():
    return [{"arm": arm, "seed": seed, "objective": objective,
             "episode_keys": [[seed, 1], [seed, 2]],
             "step_keys": [[seed + 100, 1], [seed + 100, 2]],
             "metrics": {"blue_return": np.array([1., 3.]) + seed + offset,
                         "captured": [0, 1], "resources_collected": [1, 1]}}
            for arm, offset in zip(ARMS, [0, 1, -2, 0, 0], strict=True)
            for seed in SEEDS for objective in ["capture", "risk", "curious"]]


@pytest.fixture(scope="module")
def campaign():
    path = Path(__file__).resolve().parents[1] / "scripts/run_resl_campaign.py"
    spec = importlib.util.spec_from_file_location("resl_campaign_certification", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("corruption", ["different_steps", "one_missing_steps", "bad_key_shape", "missing_arm", "duplicate_fit"])
def test_incomplete_or_unpaired_campaign_cannot_be_aggregated(corruption):
    rows = records()
    if corruption == "different_steps":
        rows[0]["step_keys"][0][1] += 10
    elif corruption == "one_missing_steps":
        del rows[0]["step_keys"]
    elif corruption == "bad_key_shape":
        rows[0]["step_keys"] = [1, 2]
    elif corruption == "missing_arm":
        rows = [row for row in rows if row["arm"] != "bc"]
    else:
        rows = [row for row in rows if row["seed"] != 2]
        rows += [row.copy() for row in rows if row["seed"] == 1]
    with pytest.raises(ValueError):
        summarize_benchmark(rows, ARMS, SEEDS, comparisons=[("0s", "bc")], bootstrap_samples=32)


def test_predeclared_contrasts_and_negative_results_are_preserved(campaign):
    statistics = {"primary_architecture_comparisons": ["0s minus bc", "bc minus implicit"],
                  "practically_meaningful_return_difference": 5.}
    binding = {"configuration": {"arms": ARMS}, "statistics": statistics, "seeds": SEEDS}
    result = campaign.report_comparisons(records(), binding)
    assert list(result["comparisons"]) == ["0s_minus_bc", "bc_minus_implicit"]
    negative = result["practical_return_comparisons"]["0s_minus_bc"]
    assert negative["mean_difference"] == -3.
    assert negative["status"] == "failed mean-difference criterion"
    assert result["practical_return_comparisons"]["bc_minus_implicit"]["status"] == "failed mean-difference criterion"
    interval = result["comparisons"]["0s_minus_bc"]["metrics"]["blue_return"]["overall"]["ci95"]
    assert interval == [-3., -3.]
    assert result["predeclared_statistics"] == statistics


@pytest.mark.parametrize("names,threshold", [([], 5), (["0s minus absent"], 5),
    (["0s minus bc", "0s minus bc"], 5), (["0s minus bc"], float("nan")), (["0s minus bc"], 0)])
def test_invalid_or_absent_analysis_contract_is_rejected(campaign, names, threshold):
    with pytest.raises(ValueError):
        campaign.protocol_comparisons({"primary_architecture_comparisons": names,
            "practically_meaningful_return_difference": threshold}, ARMS, "architecture")
