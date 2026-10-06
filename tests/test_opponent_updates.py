"""A06: real-episode replay changes weights using the original saved optimizer."""
from pathlib import Path

import jax
import numpy as np
import pytest
from flax import serialization

from mopa.action_decoder import ActionDecoderConfig, fit_action_decoder_vae
from mopa.opponent_updates import OpponentReplay, update_zero_s_from_replay
from mopa.zero_s import zero_s_features


def _episode(seed, length):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(length+1, 66)).astype(np.float32), np.tanh(rng.normal(size=(length, 2))).astype(np.float32)


def _append(replay, seed, length, origin="initial", episode_id=None):
    x, a = _episode(seed, length)
    replay.append(x, a, episode_id=str(seed) if episode_id is None else episode_id,
                  origin=origin, source_ref=f"fixture:{seed}", completed=True, real=True)


@pytest.fixture
def initial(tmp_path):
    replay = OpponentReplay(3)
    _append(replay, 11, 5)
    _append(replay, 12, 3)
    x, a, lengths = replay.arrays()
    cfg = ActionDecoderConfig(window=3, hid=4, lat=2, steps=2, batch=2, action_type="continuous")
    checkpoint = tmp_path / "initial.msgpack"
    fit_action_decoder_vae(np.asarray(zero_s_features(x[:, :-1])), a, lengths, np.arange(2),
                           jax.random.key(5), config=cfg, training_state_path=checkpoint)
    evaluation = OpponentReplay(2)
    _append(evaluation, 21, 4)
    _append(evaluation, 22, 2)
    return replay, evaluation, checkpoint


def test_replay_preserves_episodes_and_declares_actual_window_sampling_mix():
    replay = OpponentReplay(2)
    _append(replay, 1, 2)
    _append(replay, 2, 5, "online")
    x, a, lengths = replay.arrays()
    np.testing.assert_array_equal(lengths, [2, 5])
    np.testing.assert_array_equal(a[0, 2:], 0)
    np.testing.assert_array_equal(x[0, 3:], np.broadcast_to(x[0, 2], x[0, 3:].shape))
    report = replay.summary(3)
    assert report["windows_by_origin"] == {"initial": 1, "online": 2}
    assert report["window_sampling_probability_by_origin"] == {"initial": 1/3, "online": 2/3}
    _append(replay, 3, 1, "online")
    assert [e["episode_id"] for e in replay.summary(3)["episodes"]] == ["2", "3"]
    with pytest.raises(ValueError, match="new nonempty"):
        _append(replay, 1, 2)  # Previously evicted IDs cannot masquerade as fresh data.


@pytest.mark.parametrize("completed,real", [(False, True), (True, False)])
def test_replay_rejects_incomplete_or_imagined_episodes(completed, real):
    replay = OpponentReplay(2)
    x, a = _episode(2, 3)
    with pytest.raises(ValueError, match="completed real"):
        replay.append(x, a, episode_id="x", origin="online", source_ref="test", completed=completed, real=real)


def test_online_update_continues_optimizer_and_changes_actual_weights(initial, tmp_path):
    replay, evaluation, checkpoint = initial
    _append(replay, 13, 4, "online")
    output = tmp_path / "continued.msgpack"
    saved_before = checkpoint.read_bytes()
    fit, report, traces = update_zero_s_from_replay(replay, evaluation, checkpoint, updates=2, training_state_path=output)
    old = serialization.msgpack_restore(saved_before)
    new = serialization.msgpack_restore(output.read_bytes())
    assert checkpoint.read_bytes() == saved_before
    assert new["step"] == 4 and new["anneal_steps"] == old["anneal_steps"] == 2
    assert int(new["optimizer"]["0"]["count"]) == 4
    assert [row["step"] for row in fit.history] == [2, 3]
    assert all(row["beta"] == 1 for row in fit.history)
    np.testing.assert_array_equal(new["state_mean"], old["state_mean"])
    np.testing.assert_array_equal(new["state_std"], old["state_std"])
    assert report["encoder_parameter_change_l2"] > 0
    assert report["decoder_parameter_change_l2"] > 0
    assert report["data_mix"]["episodes_by_origin"] == {"initial": 2, "online": 1}
    assert np.isfinite(report["prediction_mse_change"])
    assert not np.array_equal(traces["before_prediction"], traces["after_prediction"])
    mask = np.arange(traces["target"].shape[1])[None] < traces["valid_length"][:, None]
    per_episode = np.sum(np.where(mask, np.sum((traces["after_prediction"]-traces["target"])**2, axis=-1), 0), axis=1) / traces["valid_length"]
    np.testing.assert_allclose(report["after"]["episode_values"], per_episode)
    with pytest.raises(FileExistsError):
        update_zero_s_from_replay(replay, evaluation, checkpoint, updates=2, training_state_path=output)


def test_new_observations_change_weights_but_evaluation_targets_do_not(initial, tmp_path):
    _, evaluation, checkpoint = initial
    # This diagnostic retains only the new episode, so every sampled window
    # must contain online observations even with only two optimizer steps.
    replay = OpponentReplay(1)
    _append(replay, 13, 4, "online")
    first, _, _ = update_zero_s_from_replay(replay, evaluation, checkpoint, updates=2, training_state_path=tmp_path/"one.msgpack")
    changed_evaluation = OpponentReplay(2)
    _append(changed_evaluation, 41, 4)
    _append(changed_evaluation, 42, 2)
    repeated, _, _ = update_zero_s_from_replay(replay, changed_evaluation, checkpoint, updates=2, training_state_path=tmp_path/"two.msgpack")
    for a, b in zip(jax.tree.leaves(first.encoder.params), jax.tree.leaves(repeated.encoder.params), strict=True):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(jax.tree.leaves(first.decoder_params), jax.tree.leaves(repeated.decoder_params), strict=True):
        np.testing.assert_array_equal(a, b)
    changed_replay = OpponentReplay(1)
    _append(changed_replay, 53, 4, "online")
    different, _, _ = update_zero_s_from_replay(changed_replay, evaluation, checkpoint, updates=2, training_state_path=tmp_path/"three.msgpack")
    assert any(not np.array_equal(a, b) for a, b in zip(jax.tree.leaves(first.decoder_params), jax.tree.leaves(different.decoder_params), strict=True))


def test_online_update_rejects_evaluation_overlap_and_absent_online_data(initial, tmp_path):
    replay, evaluation, checkpoint = initial
    output = tmp_path/"not-written.msgpack"
    with pytest.raises(ValueError, match="newly observed"):
        update_zero_s_from_replay(replay, evaluation, checkpoint, updates=1, training_state_path=output)
    _append(replay, 13, 4, "online")
    with pytest.raises(ValueError, match="overlap"):
        update_zero_s_from_replay(replay, replay, checkpoint, updates=1, training_state_path=output)
    renamed_duplicate = OpponentReplay(1)
    _append(renamed_duplicate, 13, 4, episode_id="renamed")
    with pytest.raises(ValueError, match="overlap"):
        update_zero_s_from_replay(replay, renamed_duplicate, checkpoint, updates=1, training_state_path=output)
    assert not Path(output).exists()
