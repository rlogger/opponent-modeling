"""End-to-end smoke of the continuous programme's drivers on tiny synthetic checkpoints.

Runs dataset generation -> continuous BC -> simulator planner -> TD-MPC
train / evaluate / compare with random specialist actors and minimal budgets.
It checks that the scripts compose and write valid, hash-bound artifacts; it
is infrastructure evidence only.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("jaxmarl")
pytest.importorskip("distrax")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from mopa.continuous_data import (  # noqa: E402
    OBJECTIVE_TYPES,
    continuous_checkpoint_path,
)
from mopa.nets import ContinuousActor  # noqa: E402
from tag_objectives import make_env  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _load(script: str):
    path = ROOT / "scripts" / script
    spec = importlib.util.spec_from_file_location(script.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_checkpoints(logdir: Path, seeds=(0, 1)):
    from jaxmarl.wrappers.baselines import save_params

    env = make_env("capture", continuous=True)
    width = max(env.observation_space(a).shape[0] for a in env.agents)
    actor = ContinuousActor(action_dim=2, hidden_dim=128)
    logdir.mkdir(parents=True, exist_ok=True)
    k = 0
    for seed in seeds:
        for team in ("pred", "prey"):
            for objective in OBJECTIVE_TYPES:
                params = actor.init(jax.random.PRNGKey(k), jnp.zeros((1, width)))
                params["params"]["Dense_2"]["bias"] = jnp.asarray([0.5, -0.3]) * ((k % 3) - 1)
                save_params(params, str(continuous_checkpoint_path(logdir, objective, team, seed)))
                k += 1


def test_continuous_drivers_compose(tmp_path):
    logdir = tmp_path / "logs"
    _write_checkpoints(logdir)
    data_dir = tmp_path / "data"
    make_ds = _load("make_continuous_dataset.py")
    assert make_ds.main(
        [
            "--artifact-dir", str(data_dir), "--logdir", str(logdir),
            "--n-eps", "3", "--num-steps", "8", "--ckpt-seeds", "0,1",
        ]
    ) == 0
    report = json.loads((data_dir / "report.json").read_text())
    assert report["summary"]["replay"]["termination_flags_match_fraction"] == 1.0

    bc_dir = tmp_path / "bc"
    run_bc = _load("run_bc_continuous.py")
    assert run_bc.main(
        [
            "--artifact-dir", str(bc_dir), "--dataset", str(data_dir / "dataset.npz"),
            "--logdir", str(logdir), "--seeds", "0", "--encoder-steps", "2",
            "--bc-steps", "2", "--ctx", "2", "--closed-loop-eps", "2",
        ]
    ) == 0
    bc = json.loads((bc_dir / "results.json").read_text())
    assert set(bc["summary"]["offline"]) == {"no_c", "real_c", "shuffled_c", "oracle"}
    assert (bc_dir / "fold_1" / "seed_0" / "context_encoder.npz").is_file()

    sim_dir = tmp_path / "sim"
    run_sim = _load("run_sim_planner.py")
    assert run_sim.main(
        [
            "--artifact-dir", str(sim_dir), "--dataset", str(data_dir / "dataset.npz"),
            "--bc-artifacts", str(bc_dir), "--logdir", str(logdir), "--heldout", "1",
            "--n-eps", "2", "--population", "8", "--elites", "2", "--iterations", "1",
            "--horizon", "3", "--controllers", "mappo_prey,random,planner_true_red,planner_bc_real_online",
        ]
    ) == 0
    sim = json.loads((sim_dir / "results.json").read_text())
    assert sim["claim_gates"]["actions_within_bounds"] is True

    td_root = tmp_path / "tdmpc"
    run_td = _load("run_tdmpc.py")
    for mode in ("implicit", "factored"):
        online = ["--online-rounds", "1", "--online-episodes", "1", "--updates-per-round", "2", "--logdir", str(logdir)] if mode == "factored" else []
        assert run_td.main(
            [
                "train", "--mode", mode, "--encoder", "identity", "--profile", "smoke",
                "--features", "relative", "--heldout", "1", "--seed", "0", "--updates", "3",
                "--log-every", "1", "--dataset", str(data_dir / "dataset.npz"),
                "--bc-artifacts", str(bc_dir), "--out", str(td_root), *online,
            ]
        ) == 0
    runs = sorted(p for p in td_root.iterdir() if p.is_dir())
    assert len(runs) == 2
    by_mode = {json.loads((p / "manifest.json").read_text())["mode"]: p for p in runs}
    manifest = json.loads((by_mode["implicit"] / "manifest.json").read_text())
    assert manifest["evaluation"]["heldout"]["model_error"]["per_horizon"]["1"]["n_starts"] > 0
    assert manifest["features"] == "relative" and manifest["state_dim"] == 111
    factored_manifest = json.loads((by_mode["factored"] / "manifest.json").read_text())
    assert factored_manifest["online"]["rounds"] == 1
    # 3 opponents x 1 training checkpoint (checkpoint 1 is held out) x 1 episode.
    assert factored_manifest["online"]["log"][0]["n_episodes"] == 3
    assert factored_manifest["final_replay_episodes"] > factored_manifest["train_episodes"]
    for run in runs:
        contexts = "zero" if run == by_mode["implicit"] else "zero,online"
        assert run_td.main(
            [
                "evaluate", str(run), "--dataset", str(data_dir / "dataset.npz"),
                "--bc-artifacts", str(bc_dir), "--logdir", str(logdir), "--n-eps", "2",
                "--context-modes", contexts, "--controls",
            ]
        ) == 0
        ev = json.loads((run / "evaluation.json").read_text())
        assert all(r["blue_action_abs_max"] <= 1.0 + 1e-6 for r in ev["runs"])
    assert run_td.main(["compare", "--root", str(td_root)]) == 0
    cmp_ = json.loads((td_root / "comparison.json").read_text())
    assert set(cmp_["modes"]) == {"implicit", "factored"}
    inv = cmp_["factored_invariance"]
    assert len(inv) == 1 and inv[0]["physics_reward_termination_context_invariant"] is True
    assert np.isfinite(cmp_["summary"]["factored__identity"]["rows"]["capture__tdmpc__zero"]["blue_return"]["mean"])
    # Gate 4 model-quality scalars are carried from the manifests into the record.
    mq = cmp_["summary"]["factored__identity"]["model_quality"]
    assert mq["n_seeds"] == 1
    assert mq["heldout"]["model_error"]["1"]["ratio_model_over_persistence"]["n"] == 1
    assert mq["heldout"]["reward"]["explained_variance"] is not None
    # The smoke profile has no continuation head, so no termination block is aggregated.
    assert ("termination" in mq["heldout"]) == (
        factored_manifest["evaluation"]["heldout"]["termination_calibration"] is not None
    )
    assert len(mq["online_rounds"]) == 1 and set(mq["online_rounds"][0]["per_opponent"]) == set(OBJECTIVE_TYPES)
    assert cmp_["summary"]["implicit__identity"]["model_quality"]["online_rounds"] == []
    assert cmp_["summary"]["factored__identity"]["model_quality_per_seed"][0]["total_updates"] == 5
