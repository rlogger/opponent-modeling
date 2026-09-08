"""Verify the main-source rerun using the existing exact simulator audit."""
from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np

from mopa.manifest import file_sha256

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    driver = ROOT / "experiments/original_env_20260908/verify.py"
    spec = importlib.util.spec_from_file_location("original_audit", driver)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    result = {"passed": False, "main_commit": "8c24db3c7deff8bd32610b04c3f1ebc08bf5e429",
              "backup_commit": "180571beed82326a0d11da8fad20c1ebb1886841",
              "verification_sources": {str(p.relative_to(ROOT)): file_sha256(p)
                                       for p in (Path(__file__), driver)}}
    try:
        result["main_source"] = {}
        for name in ("objectives.py", "resources.py"):
            path = ROOT / "src/tag_objectives" / name
            expected = subprocess.check_output(["git", "show", f"{result['main_commit']}:{path.relative_to(ROOT)}"], cwd=ROOT)
            assert path.read_bytes() == expected
            result["main_source"][name] = file_sha256(path)
        actual_backup = subprocess.check_output(["git", "rev-parse", "new-env"], cwd=ROOT, text=True).strip()
        assert actual_backup == result["backup_commit"]
        old = ROOT / "experiments/original_env_20260908"
        fresh_path = HERE / "continuous_data/dataset.npz"
        old_path = old / "data/dataset.npz"
        with np.load(fresh_path) as fresh, np.load(old_path) as previous:
            assert set(fresh.files) == set(previous.files)
            for key in fresh.files:
                np.testing.assert_array_equal(fresh[key], previous[key])
            result["continuous_data"] = {"arrays_identical": len(fresh.files),
                                         "episodes": len(fresh["checkpoint_seed"]),
                                         "fresh_sha256": file_sha256(fresh_path),
                                         "previous_sha256": file_sha256(old_path)}
        result["controller_audit"] = audit.audit_evaluation(HERE / "controller", {})
        evaluation = json.loads((HERE / "controller/evaluation.json").read_text())
        comparisons = []
        for row in evaluation["runs"]:
            folder = "final" if row["controller"] == "tdmpc" else "baseline"
            filename = row["recordings"]["transitions"]
            with np.load(HERE / "controller" / filename) as fresh, np.load(old / folder / filename) as previous:
                assert set(fresh.files) == set(previous.files)
                for key in fresh.files:
                    np.testing.assert_array_equal(fresh[key], previous[key])
                comparisons.append({"opponent": row["opponent"], "controller": row["controller"],
                                    "all_arrays_identical": True, "return_mean": row["blue_return"]["mean"],
                                    "fresh_sha256": file_sha256(HERE / "controller" / filename),
                                    "previous_sha256": file_sha256(old / folder / filename)})
        result["previous_controller_comparison"] = comparisons
        result["passed"] = True
    finally:
        (HERE / "verification.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"PASS: exact main source, identical continuous dataset and {len(comparisons)} matched controller arms")


if __name__ == "__main__":
    main()
