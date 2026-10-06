"""A06 driver rejects invented completion/provenance and retains failed runs."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture(scope="module")
def driver():
    path = Path(__file__).resolve().parents[1] / "scripts/run_opponent_updates.py"
    spec = importlib.util.spec_from_file_location("opponent_updates_driver_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def trace(length=3, physical_time=1., captured=False):
    state = np.zeros((1, 101, 66), np.float32)
    state[0, length, -1] = physical_time
    valid = np.arange(100)[None] < length
    term, trunc = np.zeros((1, 100), bool), np.zeros((1, 100), bool)
    (term if captured else trunc)[0, length - 1] = True
    return {"state": state, "valid_mask": valid, "valid_length": np.array([length]),
            "terminated_capture": term, "truncated_timeout": trunc}


def test_administrative_cut_is_not_a_completed_replay_episode(driver):
    with pytest.raises(ValueError, match="administrative"):
        driver.check_complete(trace(3, .03), 100)
    driver.check_complete(trace(100, 1.), 100)
    driver.check_complete(trace(3, .03, captured=True), 100)


def test_noncontiguous_validity_and_multiple_endpoints_fail(driver):
    bad = trace()
    bad["valid_mask"][0, 1] = False
    with pytest.raises(ValueError, match="contiguous"):
        driver.check_complete(bad, 100)
    bad = trace()
    bad["terminated_capture"][0, 0] = True
    with pytest.raises(ValueError, match="one physical"):
        driver.check_complete(bad, 100)


def test_disjoint_keys_and_full_state_identity_both_required(driver):
    keys, states = {(0, 1)}, {"old"}
    with pytest.raises(ValueError, match="overlaps"):
        driver.add_identities([[0, 1]], ["new"], keys, states)
    with pytest.raises(ValueError, match="overlaps"):
        driver.add_identities([[0, 2]], ["old"], keys, states)
    driver.add_identities([[0, 2]], ["new"], keys, states)
    assert keys == {(0, 1), (0, 2)} and states == {"old", "new"}


def test_metric_reconstruction_weights_episodes_and_stops_padding(driver):
    target = np.zeros((2, 3, 2))
    prediction = np.array([[[1., 0.], [1., 0.], [1., 0.]], [[2., 0.], [1000., 0.], [1000., 0.]]])
    result = driver.raw_errors(prediction, target, np.array([3, 1]), np.array([0, 1]))
    assert result["episode_mean_vector_mse"] == 2.5
    assert result["transition_mean_vector_mse"] == 1.75
    assert result["by_objective"] == {"capture": 1., "risk": 4.}


def test_source_or_artifact_drift_is_rejected(driver, tmp_path):
    p = tmp_path / "checkpoint"
    p.write_bytes(b"original")
    bound = {str(p): driver.file_sha256(p)}
    driver.check_files(bound)
    p.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        driver.check_files(bound)


def test_failed_run_preserves_manifest_and_refuses_overwrite(driver, tmp_path):
    args = SimpleNamespace(output=tmp_path / "run", protocol=tmp_path / "absent.json",
                           spec_commit="a" * 40, code_commit="b" * 40)
    with pytest.raises(ValueError, match="committed private protocol"):
        driver.run(args)
    manifest = json.loads((args.output / "manifest.json").read_text())
    assert manifest["status"] == "failed" and "ValueError" in manifest["error"]
    before = (args.output / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        driver.run(args)
    assert (args.output / "manifest.json").read_bytes() == before
