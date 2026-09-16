"""Focused checks for the isolated tiny-set reconstruction diagnostic."""

import importlib.util
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture(scope="module")
def diagnostic():
    path = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_0s_reconstruction.py"
    spec = importlib.util.spec_from_file_location("reconstruction_diagnostic_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def episode_metadata():
    # Only three full windows per training checkpoint belong to behavior zero.
    lengths = np.array([8, 16, 24, 7, 8, 16, 24, 8, 16, 24, 32, 1])
    labels = np.array([0] * 10 + [1, 0])
    checkpoints = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0])
    return lengths, labels, checkpoints


@pytest.mark.parametrize("checkpoint", [0, 1])
def test_selection_is_repeatable_aligned_and_checkpoint_local(
    diagnostic, episode_metadata, checkpoint
):
    lengths, labels, checkpoints = episode_metadata
    episode, start = diagnostic.select_windows(
        lengths, labels, checkpoints, 0, 3, 8, 17, checkpoint=checkpoint
    )
    repeat = diagnostic.select_windows(
        lengths, labels, checkpoints, 0, 3, 8, 17, checkpoint=checkpoint
    )
    np.testing.assert_array_equal(episode, repeat[0])
    np.testing.assert_array_equal(start, repeat[1])
    assert len(np.unique(episode)) == 3
    assert np.all(labels[episode] == 0)
    assert np.all(checkpoints[episode] == checkpoint)
    assert np.all(start % 8 == 0)
    assert np.all(start >= 0)
    assert np.all(start + 8 <= lengths[episode])

    # The same episode/time indices must address state and target action.
    marker = np.arange(len(lengths))[:, None] * 100 + np.arange(33)[None, :]
    time_idx = start[:, None] + np.arange(8)
    states = marker[..., None]
    actions = -marker[..., None]
    selected_state = states[episode[:, None], time_idx]
    selected_action = actions[episode[:, None], time_idx]
    np.testing.assert_array_equal(selected_state, -selected_action)
    np.testing.assert_array_equal(
        selected_state[..., 0] // 100,
        episode[:, None] + np.zeros((3, 8), int),
    )


def test_selection_default_excludes_other_checkpoints_and_rejects_shortfall(
    diagnostic, episode_metadata
):
    lengths, labels, checkpoints = episode_metadata
    episode, _ = diagnostic.select_windows(lengths, labels, checkpoints, 0, 3, 8, 0)
    np.testing.assert_array_equal(np.sort(episode), [0, 1, 2])
    # Neither checkpoint one/two, a different behavior, nor a short episode may
    # silently supply the fourth requested window.
    with pytest.raises(ValueError, match="eligible"):
        diagnostic.select_windows(lengths, labels, checkpoints, 0, 4, 8, 0)
    with pytest.raises(ValueError):
        diagnostic.select_windows(lengths, labels, checkpoints, 0, 1, 8, 0)
    with pytest.raises(ValueError, match="checkpoint"):
        diagnostic.select_windows(lengths, labels, checkpoints, 0, 3, 8, 0, checkpoint=2)


def test_bc_initialization_matches_zero_latent_decoder(diagnostic):
    rng = np.random.default_rng(9)
    state = jnp.asarray(rng.normal(size=(3, 4, 8)), dtype=jnp.float32)
    action = jnp.tanh(state[..., :2])
    cfg = diagnostic.ActionDecoderConfig(
        action_type="continuous", lat=2, hid=8, window=4, steps=1
    )
    params, bc_params = diagnostic.initialize(state, action, cfg, 3)
    repeated, _ = diagnostic.initialize(state, action, cfg, 3)
    for left, right in zip(
        jax.tree_util.tree_leaves(params), jax.tree_util.tree_leaves(repeated)
    ):
        np.testing.assert_array_equal(left, right)
    latent = jnp.zeros((*state.shape[:2], cfg.lat), dtype=state.dtype)
    vae_zero = diagnostic.decode_action_decoder(params["d"], state, latent, cfg)
    bc = jnp.tanh(diagnostic.MLPHead(out=2, hid=cfg.hid).apply(bc_params, state))
    np.testing.assert_allclose(bc, vae_zero, atol=2e-6, rtol=2e-6)
    assert params["d"]["params"]["MLPTrunk_0"]["Dense_0"]["kernel"].shape == (10, 8)
    assert bc_params["params"]["MLPTrunk_0"]["Dense_0"]["kernel"].shape == (8, 8)


@pytest.mark.parametrize("mode", ["vae", "stochastic_no_kl", "deterministic_no_kl", "bc"])
def test_tiny_full_batch_training_is_finite_bounded_and_reduces_error(diagnostic, mode):
    state = np.random.default_rng(7).normal(size=(4, 4, 8)).astype(np.float32)
    action = np.tanh(0.4 * state[..., :2] + np.array([0.5, -0.3], np.float32))
    cfg = diagnostic.ActionDecoderConfig(
        action_type="continuous", lat=2, hid=8, window=4,
        steps=80, learning_rate=5e-3,
    )
    result, prediction = diagnostic.train_arm(state, action, cfg, 0, mode, log_every=40)
    assert prediction.shape == action.shape
    assert np.isfinite(prediction).all()
    assert np.all(np.abs(prediction) <= 1)
    assert result["final_mse"] < result["initial_mse"]
    assert [row["step"] for row in result["history"]] == [0, 40, 80]
    assert all(np.isfinite(row["mean_latent_mse"]) for row in result["history"])
    np.testing.assert_allclose(
        result["final_mse"], np.sum((prediction - action) ** 2, axis=-1).mean(),
        rtol=2e-6, atol=1e-7,
    )
    if mode in {"vae", "stochastic_no_kl"}:
        assert np.isfinite(result["sampled_mse_32_draws"])
        assert np.isfinite(result["raw_kl"])
        assert result["floored_kl"] >= cfg.lat * cfg.free_bits - 1e-6
    else:
        assert "sampled_mse_32_draws" not in result


def test_main_uses_only_selected_training_pairs_and_shared_normalization(
    diagnostic, tmp_path, monkeypatch
):
    rng = np.random.default_rng(11)
    labels = np.repeat(np.arange(3), 4)
    checkpoints = np.tile([0, 0, 1, 2], 3)
    lengths = np.full(12, 16)
    state = rng.normal(size=(12, 17, 66)).astype(np.float32)
    action = np.tanh(rng.normal(size=(12, 16, 2))).astype(np.float32)
    # Poison all other checkpoints: accidental selection or global normalization
    # must fail before any model fitting.
    state[checkpoints != 0] = np.nan
    action[checkpoints != 0] = np.nan
    dataset = tmp_path / "dataset.npz"
    np.savez_compressed(
        dataset, valid_length=lengths, objective_label=labels,
        checkpoint_seed=checkpoints, state=state, red_action=action,
    )
    calls = []

    def fake_train(normalized, target, cfg, seed, mode):
        calls.append((mode, normalized.copy(), target.copy()))
        assert np.isfinite(normalized).all() and np.isfinite(target).all()
        return {
            "final_mse": 0.0,
            "training_loop_seconds_including_update_compile": 0.0,
        }, target.copy()

    monkeypatch.setattr(diagnostic, "train_arm", fake_train)
    monkeypatch.setattr(diagnostic, "plots", lambda *_: None)
    monkeypatch.setattr(diagnostic, "file_sha256", lambda _: "test-hash")
    monkeypatch.setattr(diagnostic, "git_sha", lambda _: "test-commit")
    monkeypatch.setattr(diagnostic, "git_dirty", lambda _: False)
    monkeypatch.setattr(diagnostic, "package_versions", lambda: {})
    out = tmp_path / "diagnostic"
    monkeypatch.setattr(sys, "argv", [
        "diagnose_0s_reconstruction.py", "--dataset", str(dataset),
        "--out", str(out), "--windows", "2", "--steps", "1",
    ])
    diagnostic.main()
    result = json.loads((out / "results.json").read_text())
    assert len(calls) == 12
    with np.load(out / "predictions.npz", allow_pickle=False) as saved:
        for k, behavior in enumerate(diagnostic.BEHAVIORS):
            selected = result["selection"][behavior]
            episode = np.asarray(selected["episode"])
            times = np.asarray(selected["start"])[:, None] + np.arange(8)
            assert np.all(checkpoints[episode] == 0)
            expected_state = np.asarray(
                diagnostic.zero_s_features(state[episode[:, None], times])
            )
            expected_action = action[episode[:, None], times]
            mean, std = expected_state.mean((0, 1)), expected_state.std((0, 1)) + 1e-6
            np.testing.assert_allclose(saved[f"{behavior}_state"], expected_state)
            np.testing.assert_allclose(saved[f"{behavior}_action"], expected_action)
            np.testing.assert_allclose(saved[f"{behavior}_mean"], mean)
            np.testing.assert_allclose(saved[f"{behavior}_std"], std)
            for call, mode in zip(calls[k * 4:(k + 1) * 4], diagnostic.MODES):
                assert call[0] == mode
                np.testing.assert_allclose(call[1], (expected_state - mean) / std)
                np.testing.assert_array_equal(call[2], expected_action)
