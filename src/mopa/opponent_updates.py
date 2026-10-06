"""Replay-driven opponent weight updates requested in the September 10 notes.

This separate path continues the original 0s reconstruction objective. It does
not change frozen-opponent controller comparisons. Callers supply completed
real episodes and must independently certify their simulator/source provenance.
"""
from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

import jax
import numpy as np
from flax import serialization

from mopa.action_decoder import (
    ActionDecoderConfig,
    FrozenActionEncoder,
    fit_action_decoder_vae,
)
from mopa.zero_s import ZeroSOpponent, zero_s_features


def _episode_hash(state, action):
    digest = hashlib.sha256()
    for value in (state, action):
        value = np.ascontiguousarray(value, dtype=np.float32)
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class _Episode:
    episode_id: str
    origin: str
    source_ref: str
    state: np.ndarray
    red_action: np.ndarray
    sha256: str


class OpponentReplay:
    """Finite FIFO of complete real episodes, with explicit initial/online mix.

    Sampling remains the original uniform-with-replacement sampling of windows
    inside the retained episodes. No imagined or held-out data are added by
    this class. ``source_ref`` identifies the source artifact and episode; its
    truth is a caller responsibility, not proved by an array shape check.
    """

    def __init__(self, capacity_episodes: int):
        if not isinstance(capacity_episodes, int) or isinstance(capacity_episodes, bool) or capacity_episodes < 1:
            raise ValueError("capacity_episodes must be a positive integer")
        self.capacity_episodes = capacity_episodes
        self._episodes: list[_Episode] = []
        self._seen_ids: set[str] = set()

    def append(self, state, red_action, *, episode_id, origin, source_ref, completed, real):
        if not completed or not real:
            raise ValueError("opponent replay accepts only completed real episodes")
        if not isinstance(episode_id, str) or not episode_id or episode_id in self._seen_ids:
            raise ValueError("a new nonempty episode_id is required")
        if origin not in {"initial", "online"} or not isinstance(source_ref, str) or not source_ref:
            raise ValueError("origin must be initial/online with a nonempty source_ref")
        state, action = np.asarray(state, np.float32), np.asarray(red_action, np.float32)
        if action.ndim != 2 or action.shape[1] != 2 or not len(action) or state.shape != (len(action) + 1, 66):
            raise ValueError("expected complete state(T+1,66) and red_action(T,2)")
        if not np.isfinite(state).all() or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
            raise ValueError("finite states and bounded actions required")
        state, action = state.copy(), action.copy()
        state.setflags(write=False)
        action.setflags(write=False)
        self._episodes.append(_Episode(episode_id, origin, source_ref, state, action, _episode_hash(state, action)))
        self._seen_ids.add(episode_id)
        self._episodes = self._episodes[-self.capacity_episodes:]

    def arrays(self):
        if not self._episodes:
            raise ValueError("nonempty opponent replay required")
        lengths = np.array([len(e.red_action) for e in self._episodes], np.int32)
        horizon = int(lengths.max())
        state = np.zeros((len(lengths), horizon + 1, 66), np.float32)
        action = np.zeros((len(lengths), horizon, 2), np.float32)
        for i, episode in enumerate(self._episodes):
            length = lengths[i]
            state[i, :length+1] = episode.state
            state[i, length+1:] = episode.state[-1]
            action[i, :length] = episode.red_action
        return state, action, lengths

    def summary(self, window: int):
        if window < 1:
            raise ValueError("positive reconstruction window required")
        self.arrays()  # Reject empty buffers before constructing a sampling claim.
        episodes, transitions, windows = Counter(), Counter(), Counter()
        for item in self._episodes:
            episodes[item.origin] += 1
            transitions[item.origin] += len(item.red_action)
            windows[item.origin] += (len(item.red_action) + window - 1) // window
        total = sum(windows.values())
        return {
            "capacity_episodes": self.capacity_episodes,
            "sampling": "original 0s uniform training windows with replacement",
            "episodes_by_origin": dict(episodes), "transitions_by_origin": dict(transitions),
            "windows_by_origin": dict(windows),
            "window_sampling_probability_by_origin": {k: v/total for k, v in windows.items()},
            "episodes": [{"episode_id": e.episode_id, "origin": e.origin,
                          "source_ref": e.source_ref, "state_action_sha256": e.sha256}
                         for e in self._episodes],
        }


def _prediction_error(opponent, dataset):
    state, action, lengths = dataset.arrays()
    context = opponent.context(state, action, lengths)[:, :-1]
    prediction = np.asarray(opponent.actions(state[:, :-1], context))
    error = np.sum((prediction - action)**2, axis=-1)
    mask = np.arange(action.shape[1])[None] < lengths[:, None]
    error = np.where(mask, error, 0.0)
    if not np.isfinite(error).all():
        raise FloatingPointError("nonfinite opponent evaluation")
    per_episode = error.sum(-1) / lengths
    return {"episode_mean_vector_mse": float(per_episode.mean()),
            "transition_mean_vector_mse": float(error.sum()/lengths.sum()),
            "episode_values": per_episode.tolist()}, prediction


def update_zero_s_from_replay(
    replay: OpponentReplay, evaluation: OpponentReplay, training_state,
    *, updates: int, training_state_path: str | Path,
):
    """Continue original optimizer/RNG and evaluate actual encoder/decoder change.

    ``evaluation`` is disjoint, fixed real data and never enters fitting. This
    returns both positive and negative changes without a success gate. The
    caller must bind the returned report/checkpoint to the governing protocol
    and both repository revisions before publishing an experiment.
    """
    if not isinstance(updates, int) or isinstance(updates, bool) or updates < 1:
        raise ValueError("updates must be a positive integer")
    destination = Path(training_state_path)
    if destination.exists():
        raise FileExistsError("new opponent training checkpoint path required")
    if isinstance(training_state, (str, Path)):
        initial_bytes = Path(training_state).read_bytes()
        initial = serialization.msgpack_restore(initial_bytes)
    else:
        initial = training_state
        initial_bytes = serialization.msgpack_serialize(initial)
    if not isinstance(initial, dict) or initial.get("schema") != "action_decoder_training_v1":
        raise ValueError("original 0s training checkpoint required")
    cfg = replace(ActionDecoderConfig(**initial["config"]), steps=updates)
    if cfg.action_type != "continuous" or cfg.action_dim != 2:
        raise ValueError("online opponent updates require continuous 2D actions")
    train_ids = {e.episode_id for e in replay._episodes}
    eval_ids = {e.episode_id for e in evaluation._episodes}
    train_hashes = {e.sha256 for e in replay._episodes}
    if train_ids & eval_ids or train_hashes & {e.sha256 for e in evaluation._episodes}:
        raise ValueError("evaluation episodes overlap opponent replay")
    state, action, lengths = replay.arrays()
    summary = replay.summary(cfg.window)
    if not summary["episodes_by_origin"].get("online", 0):
        raise ValueError("at least one newly observed online episode is required")
    before = ZeroSOpponent(
        FrozenActionEncoder(initial["params"]["e"], initial["state_mean"], initial["state_std"], cfg),
        initial["params"]["d"], np.zeros((3, cfg.lat), np.float32),
    )
    before_metrics, before_prediction = _prediction_error(before, evaluation)
    fit = fit_action_decoder_vae(
        np.asarray(zero_s_features(state[:, :-1])), action, lengths, np.arange(len(lengths)),
        jax.random.key(0), config=cfg, initial_training_state=initial,
        training_state_path=destination,
    )
    after = ZeroSOpponent(fit.encoder, fit.decoder_params, np.zeros((3, cfg.lat), np.float32))
    after_metrics, after_prediction = _prediction_error(after, evaluation)
    changes = {}
    for name, old, new in (("encoder", before.encoder.params, after.encoder.params),
                           ("decoder", before.decoder_params, after.decoder_params)):
        deltas = [np.asarray(n, np.float64)-np.asarray(o, np.float64)
                  for o, n in zip(jax.tree.leaves(old), jax.tree.leaves(new), strict=True)]
        changes[name + "_parameter_change_l2"] = float(np.sqrt(sum(np.sum(d*d) for d in deltas)))
    report = {
        "objective": "original 0s continuous reconstruction with existing window KL",
        "initial_checkpoint_sha256": hashlib.sha256(initial_bytes).hexdigest(),
        "checkpoint_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "initial_update_step": int(initial["step"]), "final_update_step": int(initial["step"])+updates,
        "normalization": "retained initial training statistics", "data_mix": summary,
        "evaluation": evaluation.summary(cfg.window), "before": before_metrics, "after": after_metrics,
        "prediction_mse_change": after_metrics["episode_mean_vector_mse"]-before_metrics["episode_mean_vector_mse"],
        **changes,
    }
    traces = {"before_prediction": before_prediction, "after_prediction": after_prediction,
              "target": evaluation.arrays()[1], "valid_length": evaluation.arrays()[2]}
    return fit, report, traces
