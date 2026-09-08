"""Fresh native-main data and 0s fits, reusing the existing comparison driver."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SOURCE = Path("/private/tmp/shashank-encoding-audit.5AsKUv/repo")
MAIN = "8c24db3"
CORE = ("src/tag_objectives/objectives.py", "src/tag_objectives/resources.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__ == "__main__":
    existing = [p.name for p in OUT.iterdir()
                if p.name not in {"run.py", "run.log", "__pycache__"}]
    if existing:
        raise SystemExit(f"Refusing to overwrite nonempty experiment output: {existing}")
    started = time.perf_counter()
    main_commit = subprocess.check_output(
        ["git", "rev-parse", MAIN], cwd=ROOT, text=True).strip()
    for relative in CORE:
        pinned = subprocess.check_output(["git", "show", f"{main_commit}:{relative}"], cwd=ROOT)
        assert (ROOT / relative).read_bytes() == pinned, f"Not exact main source: {relative}"
    previous = ROOT / "experiments/shashank_comparison_20260908/discrete"
    spec = importlib.util.spec_from_file_location("existing_comparison", previous / "run.py")
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    driver.OUT = OUT
    driver.collect()
    old = np.load(previous / "dataset.npz")
    fresh = np.load(OUT / "dataset.npz")
    assert set(old.files) == set(fresh.files)
    comparison = {name: bool(np.array_equal(fresh[name], old[name])) for name in fresh.files}
    provenance = {
        "environment_main_commit": main_commit,
        "environment_source_hashes": {name: sha(ROOT / name) for name in CORE},
        "previous_dataset_sha256": sha(previous / "dataset.npz"),
        "fresh_dataset_sha256": sha(OUT / "dataset.npz"),
        "dataset_arrays_exact_equal": comparison,
        "all_dataset_arrays_exact_equal": all(comparison.values()),
        "valid_transitions": int(fresh["valid_length"].sum()),
        "training_valid_transitions": int(fresh["valid_length"][fresh["ckpt_seed"] != 2].sum()),
        "existing_driver_sha256": sha(previous / "run.py"),
        "wrapper_sha256": sha(Path(__file__)),
        "claim_boundary": "Fresh native discrete specialists and representation fits; not TD-MPC control.",
    }
    driver.write_json(OUT / "main_provenance.json", provenance)
    print("FRESH_DATA_COMPARISON " + json.dumps(provenance), flush=True)
    driver.fit(SOURCE)
    current_metrics = json.loads((OUT / "metrics.json").read_text())
    old_metrics = json.loads((previous / "metrics.json").read_text())
    names = ("heldout_probe", "heldout_ari", "unit_probe", "decoder_action_accuracy")
    provenance["metric_deltas_vs_previous"] = {
        fit: {name: current_metrics[fit][name] - old_metrics[fit][name] for name in names}
        for fit in ("source_legacy_forward", "integrated_legacy_forward",
                    "integrated_causal_seed0", "integrated_causal_seed1", "integrated_causal_seed2")
    }
    provenance["total_collection_and_fit_seconds"] = time.perf_counter() - started
    driver.write_json(OUT / "main_provenance.json", provenance)
    print("COMPLETE " + json.dumps(provenance), flush=True)
