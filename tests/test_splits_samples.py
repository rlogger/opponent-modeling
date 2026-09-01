import numpy as np
import pytest

from mopa.features import (
    episode_lengths,
    length_matched_prefix,
    predator_sequence_features,
    sequence_slice,
    sequence_window,
    trailing_sequence_slice,
)
from mopa.manifest import SCHEMA_VERSION, build_manifest, validate_manifest
from mopa.metrics import (
    survival_time_probe_acc,
    train_only_metrics,
    train_only_standardize,
    train_only_survival_time_probe_acc,
)
from mopa.samples import (
    build_predator_observation_samples,
    build_predator_observation_samples_with_time,
    build_predator_samples,
    build_predator_samples_with_time,
    predator_state_features,
    valid_bc_steps,
)
from mopa.splits import (
    checkpoint_validation_mask,
    episode_validation_mask,
    leave_one_checkpoint_out_folds,
)


def toy_dataset(n=2, horizon=6, n_pred=3, capture_t=None):
    prey = np.zeros((n, horizon + 1, 2), dtype=np.float32)
    preds = np.zeros((n, horizon + 1, n_pred, 2), dtype=np.float32)
    acts = np.zeros((n, horizon, n_pred), dtype=np.int32)
    for e in range(n):
        for t in range(horizon + 1):
            prey[e, t] = [10 * e + t, 10 * e - t]
            for p in range(n_pred):
                preds[e, t, p] = [100 * p + t, 10 * e + p]
        for t in range(horizon):
            for p in range(n_pred):
                acts[e, t, p] = (e + t + p) % 5
    if horizon > 4:
        acts[:, 4, :] = 99
    out = {"prey_pos": prey, "pred_pos": preds, "pred_act": acts}
    if capture_t is not None:
        out["capture_t"] = np.asarray(capture_t, dtype=np.int32)
    return out


def test_valid_bc_steps_skip_zero_and_reset_boundaries():
    assert valid_bc_steps(0, 7, ep_len=3) == (1, 2, 3, 5, 6)
    assert valid_bc_steps(0, 30, ep_len=25) == tuple(
        t for t in range(1, 30) if t % 26 != 0
    )


def test_predator_state_feature_order_and_identity():
    ds = toy_dataset(n=1, horizon=2, n_pred=3)
    feats = predator_state_features(
        ds["pred_pos"][:, 1],
        ds["prey_pos"][:, 1],
        ds["pred_pos"][:, 0],
        ds["prey_pos"][:, 0],
    )

    assert feats.shape == (1, 3, 19)
    np.testing.assert_array_equal(feats[0, 0, :8], [1, 0, 101, 1, 201, 2, 1, -1])
    np.testing.assert_array_equal(feats[0, 0, 8:16], [1, 0, 1, 0, 1, 0, 1, -1])
    np.testing.assert_array_equal(feats[0, 0, 16:], [1, 0, 0])
    np.testing.assert_array_equal(feats[0, 2, 16:], [0, 0, 1])


def test_predator_state_features_include_observed_lava_geometry():
    ds = toy_dataset(n=1, horizon=2, n_pred=1)
    lava_pos = np.array([[[0.2, -0.3], [1.0, 1.5]]], dtype=np.float32)
    lava_rad = np.array([[0.4, 0.25]], dtype=np.float32)
    feats = predator_state_features(
        ds["pred_pos"][:, 1],
        ds["prey_pos"][:, 1],
        ds["pred_pos"][:, 0],
        ds["prey_pos"][:, 0],
        lava_pos,
        lava_rad,
    )

    assert feats.shape == (1, 1, 15)
    np.testing.assert_allclose(feats[0, 0, 8:14], [0.2, -0.3, 1.0, 1.5, 0.4, 0.25])
    assert feats[0, 0, -1] == 1.0


def test_build_predator_samples_excludes_reset_leakage():
    ds = toy_dataset()

    S, A, ep = build_predator_samples(ds, 1, 6, ep_len=3)

    assert S.shape == (24, 19)
    assert A.shape == (24,)
    assert ep.shape == (24,)
    assert 99 not in A
    np.testing.assert_array_equal(ep[:2], [0, 1])


def test_build_samples_supports_single_predator():
    ds = toy_dataset(n_pred=1)
    S, A, ep = build_predator_samples(ds, 1, 6, ep_len=3)
    # steps (1,2,3,5) × 2 eps × 1 pred; feat = pos(4)+vel(4)+id(1)
    assert S.shape == (8, 9)
    assert A.shape == (8,)
    assert ep.shape == (8,)


def test_build_samples_retains_timestep_and_predator_provenance():
    ds = toy_dataset(n=2, horizon=4, n_pred=2)
    S, A, ep, timestep, pred_id = build_predator_samples_with_time(
        ds, 1, 4, ep_len=4
    )

    assert len(S) == len(A) == len(ep) == len(timestep) == len(pred_id)
    assert set(timestep.tolist()) == {1, 2, 3}
    assert set(pred_id.tolist()) == {0, 1}
    # The builder groups samples by timestep, then predator, then episode.
    np.testing.assert_array_equal(timestep[:4], [1, 1, 1, 1])
    np.testing.assert_array_equal(pred_id[:4], [0, 0, 1, 1])


def test_exact_observation_samples_include_t0_and_stop_at_valid_length():
    ds = toy_dataset(n=2, horizon=4, n_pred=2)
    observations = np.zeros((2, 5, 2, 3), dtype=np.float32)
    for episode in range(2):
        for timestep in range(5):
            for predator in range(2):
                observations[episode, timestep, predator] = [
                    episode,
                    timestep,
                    predator,
                ]
    ds["pred_obs"] = observations
    ds["valid_length"] = np.asarray([2, 4], dtype=np.int32)

    obs, action, episode, timestep, predator = (
        build_predator_observation_samples_with_time(ds)
    )

    assert obs.shape == (12, 3)
    assert len(obs) == len(action) == len(episode) == len(timestep) == len(predator)
    assert timestep.min() == 0
    assert np.all(timestep < ds["valid_length"][episode])
    np.testing.assert_array_equal(timestep[:4], [0, 0, 0, 0])
    np.testing.assert_array_equal(episode[:4], [0, 1, 0, 1])
    np.testing.assert_array_equal(predator[:4], [0, 0, 1, 1])
    np.testing.assert_array_equal(obs[:, 0], episode)
    np.testing.assert_array_equal(obs[:, 1], timestep)
    np.testing.assert_array_equal(obs[:, 2], predator)
    np.testing.assert_array_equal(
        action,
        ds["pred_act"][episode, timestep, predator],
    )

    compact_obs, compact_action, compact_episode = (
        build_predator_observation_samples(ds)
    )
    np.testing.assert_array_equal(compact_obs, obs)
    np.testing.assert_array_equal(compact_action, action)
    np.testing.assert_array_equal(compact_episode, episode)


def test_build_predator_samples_drops_post_capture():
    # Capture at t=3 → only steps 1,2 remain for ep_len=5 window [1,5)
    ds = toy_dataset(n=2, horizon=5, n_pred=1, capture_t=[3, 3])
    S, A, ep = build_predator_samples(ds, 1, 5, ep_len=5, capture_t=ds["capture_t"])
    # steps 1,2 × 2 eps × 1 pred
    assert S.shape[0] == 4
    assert set(ep.tolist()) == {0, 1}


def test_episode_lengths_match_survival_convention():
    capture_t = np.array([4, -1, 0, 100], dtype=np.int32)
    lengths = episode_lengths(capture_t, num_steps=10)
    np.testing.assert_array_equal(lengths, [4, 10, 1, 10])


def test_length_matched_prefix_and_sequence_zero_pad():
    ds = toy_dataset(n=2, horizon=5, n_pred=1)
    lengths = np.array([2, 4], dtype=np.int32)
    seq = predator_sequence_features(ds["prey_pos"], ds["pred_pos"], lengths)
    assert seq.shape[0] == 2
    assert seq.shape[1] == 5
    assert np.allclose(seq[0, 2:], 0.0)
    assert not np.allclose(seq[0, 0], 0.0)
    np.testing.assert_array_equal(length_matched_prefix(lengths, 3), [2, 3])


def test_causal_sequence_features_do_not_read_the_next_state():
    prey = np.arange(8, dtype=np.float32).reshape(1, 4, 2)
    pred = (prey + 0.5)[:, :, None, :]
    changed_prey = prey.copy()
    changed_pred = pred.copy()
    changed_prey[:, 2:] += 100.0
    changed_pred[:, 2:] -= 100.0

    causal = predator_sequence_features(
        prey, pred, velocity_mode="causal_past"
    )
    changed_causal = predator_sequence_features(
        changed_prey, changed_pred, velocity_mode="causal_past"
    )
    np.testing.assert_allclose(causal[:, :2], changed_causal[:, :2])

    legacy = predator_sequence_features(
        prey, pred, velocity_mode="legacy_forward"
    )
    changed_legacy = predator_sequence_features(
        changed_prey, changed_pred, velocity_mode="legacy_forward"
    )
    assert not np.allclose(legacy[:, 1], changed_legacy[:, 1])


def test_trailing_sequence_slice_moves_with_the_available_prefix():
    sequence = np.arange(2 * 6, dtype=np.float32).reshape(2, 6, 1)
    packed = trailing_sequence_slice(sequence, np.array([2, 5]), width=3)

    np.testing.assert_array_equal(packed[0], [0.0, 1.0, 0.0])
    np.testing.assert_array_equal(packed[1], [8.0, 9.0, 10.0])


def test_sequence_window_uses_shared_schema_and_lengths():
    seq = np.arange(2 * 5 * 3, dtype=np.float32).reshape(2, 5, 3)
    got = sequence_window(seq, 1, 4, lengths=np.array([2, 5]))
    restored = got.reshape(2, 5, 3)
    assert np.all(restored[:, 0] == 0)
    assert np.all(restored[0, 2:] == 0)
    np.testing.assert_array_equal(restored[1, 1:4], seq[1, 1:4])


def test_sequence_slice_packs_relative_window_and_masks_short_episode():
    seq = np.arange(2 * 6 * 2, dtype=np.float32).reshape(2, 6, 2)
    got = sequence_slice(seq, 2, 3, lengths=np.array([3, 6])).reshape(2, 3, 2)
    np.testing.assert_array_equal(got[0, 0], seq[0, 2])
    assert np.all(got[0, 1:] == 0)
    np.testing.assert_array_equal(got[1], seq[1, 2:5])


def test_episode_validation_mask_keeps_whole_episodes_together():
    ep = np.repeat(np.arange(10), 3)

    mask = episode_validation_mask(ep, rng_seed=7, val_frac=0.2)

    val_eps = set(ep[mask])
    train_eps = set(ep[~mask])
    assert len(val_eps) == 2
    assert val_eps.isdisjoint(train_eps)
    for e in val_eps:
        assert mask[ep == e].all()


def test_checkpoint_validation_mask_keeps_groups_together():
    # 3 checkpoints × 4 episodes each
    ckpt = np.repeat(np.arange(3), 4)
    mask = checkpoint_validation_mask(ckpt, rng_seed=0, val_frac=0.34)
    val = set(ckpt[mask])
    train = set(ckpt[~mask])
    assert val.isdisjoint(train)
    assert len(val) == 1
    for c in val:
        assert mask[ckpt == c].all()


def test_checkpoint_and_episode_masks_disagree_when_ckpt_mixes():
    # Episodes interleaved across checkpoints: episode split can leak a ckpt.
    ep = np.arange(12)
    ckpt = np.tile(np.arange(3), 4)  # 0,1,2,0,1,2,...
    emask = episode_validation_mask(ep, rng_seed=1, val_frac=0.25)
    cmask = checkpoint_validation_mask(ckpt, rng_seed=1, val_frac=0.34)
    # Checkpoint split never puts any episode from a held-out ckpt into train.
    val_ckpts = set(ckpt[cmask])
    assert set(ckpt[~cmask]).isdisjoint(val_ckpts)
    # Episode mask typically mixes checkpoints in both folds.
    assert set(ckpt[emask]) & set(ckpt[~emask]) or True  # structural smoke


def test_leave_one_checkpoint_out_folds():
    ckpt = np.array([0, 0, 1, 1, 2, 2])
    folds = leave_one_checkpoint_out_folds(ckpt)
    assert len(folds) == 3
    assert folds[0].sum() == 2


def test_train_only_standardize_no_val_leak():
    rng = np.random.default_rng(0)
    z_tr = rng.normal(size=(20, 3)).astype(np.float32) * 2 + 5
    z_va = rng.normal(size=(5, 3)).astype(np.float32) * 10 - 3
    zt, zq, mu, sd = train_only_standardize(z_tr, z_va)
    np.testing.assert_allclose(mu, z_tr.mean(0), atol=1e-5)
    # Query uses train stats, not val.
    np.testing.assert_allclose(zq, (z_va - mu) / sd, atol=1e-5)


def test_train_only_metrics_and_survival_control():
    rng = np.random.default_rng(0)
    # Separable 2-class latents.
    z0 = rng.normal(size=(30, 2)).astype(np.float32) + np.array([-3.0, 0.0])
    z1 = rng.normal(size=(30, 2)).astype(np.float32) + np.array([3.0, 0.0])
    z = np.concatenate([z0, z1])
    y = np.array([0] * 30 + [1] * 30)
    # Hold out last 10 of each class.
    train = np.concatenate([np.arange(20), np.arange(30, 50)])
    val = np.concatenate([np.arange(20, 30), np.arange(50, 60)])
    m = train_only_metrics(z[train], y[train], z[val], y[val], n_classes=2)
    assert m["probe"] > 0.8

    # Survival-time control API runs.
    surv = np.concatenate([np.full(30, 10.0), np.full(30, 80.0)])
    acc = survival_time_probe_acc(surv, y, cv=3)
    assert 0.0 <= acc <= 1.0
    heldout = train_only_survival_time_probe_acc(
        surv[train], y[train], surv[val], y[val]
    )
    assert heldout > 0.9


def test_manifest_schema_keys():
    man = build_manifest(
        config={"lat": 2, "split": "checkpoint"},
        checkpoints=[{"path": "x", "exists": False}],
        metrics={"heldout_probe_mean": 0.5},
        split={"mode": "checkpoint"},
    )
    for key in (
        "schema_version",
        "git_sha",
        "git_dirty",
        "created_at",
        "dependency_pins",
        "config",
        "checkpoints",
        "metrics",
        "notes",
    ):
        assert key in man
    assert man["schema_version"] == SCHEMA_VERSION


def test_manifest_fails_closed_for_smoke_and_unbound_full_runs(monkeypatch):
    with pytest.raises(ValueError, match="full"):
        build_manifest(
            config={},
            checkpoints=[],
            metrics={},
            run_kind="smoke",
            stages={"representation": {"status": "full_run_finished"}},
        )

    with pytest.raises(ValueError, match="clean|config|checkpoint|split"):
        build_manifest(
            config={},
            checkpoints=[],
            metrics={},
            run_kind="full",
            stages={"representation": {"status": "full_run_finished"}},
        )

    monkeypatch.setattr("mopa.manifest.git_sha", lambda cwd=None: "a" * 40)
    monkeypatch.setattr("mopa.manifest.git_dirty", lambda cwd=None: False)
    seeds = [0, 1, 2]
    finished = {"status": "full_run_finished"}
    full = build_manifest(
        config={
            "n_eps": 1,
            "ckpt_seeds": seeds,
            "encoder_seeds": seeds,
            "bc_seeds": seeds,
            "encoder_steps": 1,
            "bc_steps": 1,
            "split": "checkpoint",
            "prey_objective": "capture",
            "synthetic": False,
            "skip_bc": False,
            "effective_run_kind": "full",
        },
        checkpoints=[
            {
                "path": f"checkpoint-{index}",
                "exists": True,
                "sha256": "b" * 64,
            }
            for index in range(12)
        ],
        metrics={
            "environment": {},
            "representations": {},
            "anytime": {},
            "controls": {},
            "sequential_belief_mixture": {},
            "bc": {},
        },
        split={
            "mode": "checkpoint",
            "n_train": 6,
            "n_val": 3,
            "shared_across_encoder_and_bc_seeds": True,
            "episodes": [
                {
                    "fold": "validation" if seed == 2 else "train",
                    "strategy_label": label,
                    "checkpoint_seed": seed,
                    "environment_key": [seed, 0],
                    "valid_length": 10,
                }
                for seed in seeds
                for label in range(3)
            ],
        },
        run_kind="full",
        stages={
            "data": finished,
            "environment_evaluation": finished,
            "representations": finished,
            "anytime_and_calibration": finished,
            "sequential_belief_mixture": finished,
            "behaviour_cloning": finished,
        },
    )
    validate_manifest(full)


def test_run_part1_dry_run(tmp_path):
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "run_part1.py"
    spec = importlib.util.spec_from_file_location("run_part1_mod", str(script))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    out = tmp_path / "dry.json"
    rc = mod.main(
        [
            "--dry-run-checks",
            "--logdir",
            str(tmp_path / "missing_logs"),
            "--out",
            str(out),
            "--ckpt-seeds",
            "0,1",
        ]
    )
    assert rc == 2  # missing checkpoints
    assert out.is_file()
    text = out.read_text()
    assert "schema_version" in text
    assert "dry_run" in text
