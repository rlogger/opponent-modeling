"""Focused protocol checks for the four-arm BC experiment driver."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mopa.bc import build_observation_samples_with_time, evaluate_bc, fit_bc


def _driver():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_bc.py"
    spec = importlib.util.spec_from_file_location("run_bc_experiment", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _toy_dataset() -> dict[str, np.ndarray]:
    n, horizon, obs_dim = 12, 4, 3
    labels = np.tile(np.arange(3, dtype=np.int32), 4)
    checkpoints = np.repeat(np.arange(3, dtype=np.int32), 4)
    reset = np.tile(np.arange(4, dtype=np.uint32), 3)
    env_seed = np.stack([checkpoints.astype(np.uint32), reset], axis=-1)
    pred_obs = np.zeros((n, horizon + 1, 1, obs_dim), dtype=np.float32)
    pred_obs[..., 0, 0] = labels[:, None]
    pred_obs[..., 0, 1] = np.arange(horizon + 1, dtype=np.float32)
    pred_act = np.broadcast_to(labels[:, None, None], (n, horizon, 1)).copy()
    time = np.arange(horizon + 1, dtype=np.float32)
    prey_pos = np.stack([time, -time], axis=-1)[None].repeat(n, axis=0)
    pred_pos = (prey_pos + 0.1)[:, :, None, :]
    return {
        "pred_obs": pred_obs,
        "pred_act": pred_act,
        "prey_act": np.zeros((n, horizon), dtype=np.int32),
        "prey_pos": prey_pos.astype(np.float32),
        "pred_pos": pred_pos.astype(np.float32),
        "valid_length": np.full(n, horizon, dtype=np.int32),
        "label": labels,
        "ckpt_seed": checkpoints,
        "env_seed": env_seed,
    }


def test_split_local_derangement_has_no_fixed_points_or_split_crossing():
    driver = _driver()
    validation = np.array([False] * 6 + [True] * 6)

    permutation = driver.split_local_derangement(validation, seed=7)

    np.testing.assert_array_equal(validation[permutation], validation)
    assert np.all(permutation != np.arange(len(validation)))
    np.testing.assert_array_equal(
        driver.split_local_derangement(validation, seed=7), permutation
    )


def test_derangement_also_stays_inside_checkpoint_when_requested():
    driver = _driver()
    validation = np.array([False] * 8 + [True] * 4)
    checkpoints = np.repeat(np.arange(3), 4)

    permutation = driver.split_local_derangement(
        validation, seed=8, checkpoint_ids=checkpoints
    )

    np.testing.assert_array_equal(validation[permutation], validation)
    np.testing.assert_array_equal(checkpoints[permutation], checkpoints)
    assert np.all(permutation != np.arange(len(validation)))


def test_causal_latents_use_zero_at_t0_and_previous_completed_transition():
    driver = _driver()
    latent = np.arange(2 * 4 * 2, dtype=np.float32).reshape(2, 4, 2)

    selected = driver.causal_sample_latents(
        latent,
        episode_ids=np.array([0, 0, 1, 1]),
        timesteps=np.array([0, 2, 1, 3]),
    )

    np.testing.assert_array_equal(selected[0], np.zeros(2))
    np.testing.assert_array_equal(selected[1], latent[0, 1])
    np.testing.assert_array_equal(selected[2], latent[1, 0])
    np.testing.assert_array_equal(selected[3], latent[1, 2])


def test_four_conditioning_arms_have_one_common_width():
    driver = _driver()
    latent = np.arange(3 * 3 * 2, dtype=np.float32).reshape(3, 3, 2)
    labels = np.array([0, 1, 2], dtype=np.int32)
    episodes = np.array([0, 1, 2], dtype=np.int32)
    timesteps = np.array([0, 1, 2], dtype=np.int32)
    permutation = np.array([1, 2, 0], dtype=np.int32)

    arms = driver.conditioning_arms(latent, labels, episodes, timesteps, permutation)

    assert tuple(arms) == driver.ARMS
    assert all(values.shape == (3, 3) for values in arms.values())
    assert np.all(arms["no_z"] == 0.0)
    np.testing.assert_array_equal(arms["real_z"][0], np.zeros(3))
    np.testing.assert_array_equal(arms["real_z"][1, :2], latent[1, 0])
    np.testing.assert_array_equal(arms["shuffled_z"][1, :2], latent[2, 0])
    np.testing.assert_array_equal(arms["oracle"], np.eye(3, dtype=np.float32))


def test_train_only_sequence_scaling_ignores_heldout_shift():
    driver = _driver()
    sequence = np.zeros((4, 3, 2), dtype=np.float32)
    sequence[:3] = np.arange(18, dtype=np.float32).reshape(3, 3, 2)
    sequence[3] = 10_000.0
    lengths = np.array([3, 3, 2, 3], dtype=np.int32)

    scaled, mean, std = driver._scale_sequence_train_only(
        sequence, lengths, np.array([0, 1, 2])
    )

    train_mask = np.arange(3)[None, :] < lengths[:3, None]
    expected = sequence[:3][train_mask]
    np.testing.assert_allclose(mean, expected.mean(axis=0))
    np.testing.assert_allclose(std, expected.std(axis=0) + 1e-6)
    assert np.all(scaled[2, 2] == 0.0)


def test_four_arm_offline_orchestration_smoke(tmp_path):
    driver = _driver()
    data = _toy_dataset()
    obs, actions, episodes, timesteps, _ = build_observation_samples_with_time(data)
    validation_episodes = data["ckpt_seed"] == 2
    permutation = driver.split_local_derangement(validation_episodes, seed=3)
    latent = np.zeros((len(data["label"]), 4, 2), dtype=np.float32)
    latent[..., 0] = data["label"][:, None]
    conditions = driver.conditioning_arms(
        latent,
        data["label"],
        episodes,
        timesteps,
        permutation,
    )
    sample_validation = validation_episodes[episodes]

    for arm in driver.ARMS:
        features = np.concatenate([obs, conditions[arm]], axis=-1)
        policy = fit_bc(
            features[~sample_validation],
            actions[~sample_validation],
            rng_seed=0,
            steps=1,
            metadata={"arm": arm},
        )
        path = tmp_path / f"{arm}.npz"
        policy.save(path)
        metrics = evaluate_bc(
            policy.load(path),
            features[sample_validation],
            actions[sample_validation],
            episode_ids=episodes[sample_validation],
            strategy_labels=data["label"][episodes[sample_validation]],
        )
        assert np.isfinite(metrics["episode_strategy_macro_nll"])
        assert metrics["n_strategies"] == 3


def test_summary_averages_seeds_inside_each_fold_and_pairs_against_no_z():
    driver = _driver()
    folds = []
    for fold_index in range(3):
        seeds = []
        for seed_index in range(2):
            baseline_nll = 2.0 + fold_index + seed_index
            seeds.append(
                {
                    "offline": {
                        arm: {
                            "episode_strategy_macro_nll": baseline_nll
                            - (
                                0.25
                                if arm == "real_z"
                                else 0.15
                                if arm == "oracle"
                                else 0.0
                            ),
                            "episode_strategy_macro_accuracy": 0.5,
                            "action_balanced_accuracy": 0.5,
                        }
                        for arm in driver.ARMS
                    },
                    "closed_loop": {
                        arm: {
                            "macro": {
                                metric: (
                                    0.6 + (0.1 if arm == "real_z" else 0.0)
                                    if direction == "higher"
                                    else 1.0 - (0.1 if arm == "real_z" else 0.0)
                                )
                                for metric, direction in driver.CLOSED_LOOP_METRICS.items()
                            }
                        }
                        for arm in driver.ARMS
                    },
                }
            )
        folds.append({"seeds": seeds})

    summary = driver._summaries(folds)

    assert summary["offline"]["no_z"]["episode_strategy_macro_nll"]["fold_means"] == [
        2.5,
        3.5,
        4.5,
    ]
    paired = summary["paired_vs_no_z"]["offline"]["real_z"][
        "episode_strategy_macro_nll"
    ]
    assert paired["delta_mean"] == -0.25
    assert paired["all_folds_better"] is True
    assert summary["claim_gates"]["offline_controls_pass"] is True
    assert summary["claim_gates"]["real_z_primary_joint_gate"] is True


def test_dataset_cache_is_bound_to_code_config_and_checkpoints(
    tmp_path, monkeypatch
):
    driver = _driver()
    cache = tmp_path / "dataset.npz"
    args = driver.build_parser().parse_args(
        [
            "--dataset-cache",
            str(cache),
            "--n-eps",
            "2",
            "--num-steps",
            "4",
        ]
    )
    data = _toy_dataset()
    monkeypatch.setattr(
        driver,
        "objective_dataset",
        lambda **_: SimpleNamespace(as_dict=lambda: data),
    )
    checkpoints = [
        {
            "objective": "capture",
            "team": "pred",
            "seed": 0,
            "sha256": "a" * 64,
        }
    ]

    loaded, source, manifest = driver._load_or_create_dataset(
        args,
        run_git_sha="b" * 40,
        run_git_dirty=False,
        checkpoints=checkpoints,
    )
    assert source == "fresh_rollout"
    assert manifest.is_file()
    np.testing.assert_array_equal(loaded["pred_obs"], data["pred_obs"])

    _, source, _ = driver._load_or_create_dataset(
        args,
        run_git_sha="b" * 40,
        run_git_dirty=False,
        checkpoints=checkpoints,
    )
    assert source == "cache"

    changed = [{**checkpoints[0], "sha256": "c" * 64}]
    with pytest.raises(ValueError, match="provenance does not match"):
        driver._load_or_create_dataset(
            args,
            run_git_sha="b" * 40,
            run_git_dirty=False,
            checkpoints=changed,
        )
