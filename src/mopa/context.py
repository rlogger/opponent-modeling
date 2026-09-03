"""Frozen causal opponent-context encoder ``c_t = g(H_<t)``.

Wraps the GRU-JEPA trajectory encoder (``mopa.encoders``) with the exact timing
contract from the handoff: ``c_0`` is a zero vector and ``c_t`` may use only
transitions completed before the action at time ``t``. Positions through time
``t`` are available before acting at ``t``, so ``c_t`` is the prefix latent of
the first ``t`` feature steps (``legacy_forward`` velocity of step ``t - 1`` is
``p[t] - p[t - 1]``). The encoder is trained once per fold on training
episodes and then frozen for BC, planning, and world-model experiments.

Also hosts the split-local episode derangement used by every ``shuffled``
control arm so the discrete and continuous experiments share one definition.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import numpy as np
from flax.traverse_util import flatten_dict, unflatten_dict

from mopa.encoders import encode_jepa_gru, train_jepa_gru_with_params
from mopa.features import predator_sequence_features

CONTEXT_DIM = 3

__all__ = [
    "CONTEXT_DIM",
    "CausalContextEncoder",
    "causal_sample_context",
    "derangement",
    "pad_context",
    "split_local_derangement",
    "train_context_encoder",
]


def pad_context(values: np.ndarray, width: int = CONTEXT_DIM) -> np.ndarray:
    """Right-pad ``(N, d <= width)`` context rows with zeros to ``(N, width)``."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] > width:
        raise ValueError(f"context must have at most {width} columns")
    out = np.zeros((len(values), width), dtype=np.float32)
    out[:, : values.shape[1]] = values
    return out


def causal_sample_context(
    prefix_latents: np.ndarray,
    episode_ids: np.ndarray,
    timesteps: np.ndarray,
    width: int = CONTEXT_DIM,
) -> np.ndarray:
    """Attach only the latent from transitions completed before each action."""
    latent = np.asarray(prefix_latents, dtype=np.float32)
    episodes = np.asarray(episode_ids, dtype=np.int32)
    time = np.asarray(timesteps, dtype=np.int32)
    if latent.ndim != 3 or episodes.shape != time.shape:
        raise ValueError("latents and sample provenance do not align")
    out = np.zeros((len(time), latent.shape[-1]), dtype=np.float32)
    observed = time > 0
    out[observed] = latent[episodes[observed], time[observed] - 1]
    return pad_context(out, width)


def derangement(indices: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Map every supplied episode to another episode without fixed points."""
    ids = np.asarray(indices, dtype=np.int32)
    if len(ids) < 2:
        raise ValueError("a shuffled-context split needs at least two episodes")
    order = rng.permutation(ids)
    mapping = np.empty(int(ids.max()) + 1, dtype=np.int32)
    mapping[order] = np.roll(order, 1)
    if np.any(mapping[ids] == ids):
        raise AssertionError("failed to construct an episode derangement")
    return mapping


def split_local_derangement(
    validation_episodes: np.ndarray,
    seed: int,
    checkpoint_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Derange episodes without crossing a split or checkpoint when supplied."""
    validation = np.asarray(validation_episodes, dtype=bool)
    checkpoints = (
        np.zeros(len(validation), dtype=np.int32)
        if checkpoint_ids is None
        else np.asarray(checkpoint_ids, dtype=np.int32)
    )
    if checkpoints.shape != validation.shape:
        raise ValueError("checkpoint IDs must align with the episode split")
    rng = np.random.default_rng(seed)
    output = np.empty(len(validation), dtype=np.int32)
    for split_value in (False, True):
        for checkpoint in np.unique(checkpoints[validation == split_value]):
            ids = np.flatnonzero((validation == split_value) & (checkpoints == checkpoint))
            local = derangement(ids, rng)
            output[ids] = local[ids]
    return output


@dataclass(frozen=True)
class CausalContextEncoder:
    """Frozen GRU-JEPA encoder plus its train-only sequence scaling."""

    params: Any
    sequence_mean: np.ndarray
    sequence_std: np.ndarray
    latent_dim: int
    hidden_dim: int
    context_dim: int = CONTEXT_DIM
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.latent_dim > self.context_dim:
            raise ValueError("latent_dim cannot exceed context_dim")
        mean = np.asarray(self.sequence_mean, dtype=np.float32)
        std = np.asarray(self.sequence_std, dtype=np.float32)
        if mean.shape != std.shape or mean.ndim != 1 or np.any(std <= 0):
            raise ValueError("sequence scaling must be positive vectors of one shape")
        object.__setattr__(self, "sequence_mean", mean)
        object.__setattr__(self, "sequence_std", std)

    # -- features -----------------------------------------------------------
    def _scaled_sequence(
        self, prey_pos: np.ndarray, pred_pos: np.ndarray, lengths: np.ndarray
    ) -> np.ndarray:
        seq = predator_sequence_features(
            np.asarray(prey_pos, np.float32),
            np.asarray(pred_pos, np.float32),
            lengths,
            velocity_mode="legacy_forward",
        )
        horizon = seq.shape[1]
        valid = np.arange(horizon)[None, :] < np.asarray(lengths)[:, None]
        scaled = ((seq - self.sequence_mean) / self.sequence_std).astype(np.float32)
        return scaled * valid[..., None].astype(np.float32)

    def prefix_latents(
        self, prey_pos: np.ndarray, pred_pos: np.ndarray, lengths: np.ndarray
    ) -> np.ndarray:
        """``(N, T, latent_dim)``; entry ``[:, k - 1]`` encodes the first ``k`` steps."""
        lengths = np.clip(np.asarray(lengths, dtype=np.int32), 1, None)
        return encode_jepa_gru(
            self.params,
            self._scaled_sequence(prey_pos, pred_pos, lengths),
            lengths,
            lat=self.latent_dim,
            hid=self.hidden_dim,
        )

    def causal_context(
        self, prey_pos: np.ndarray, pred_pos: np.ndarray, lengths: np.ndarray
    ) -> np.ndarray:
        """``(N, T + 1, context_dim)`` with ``c_0 = 0`` and ``c_t = pad(z[t - 1])``.

        Entries beyond an episode's valid length repeat the last valid context
        (the encoder freezes its hidden state past ``lengths``).
        """
        z = self.prefix_latents(prey_pos, pred_pos, lengths)
        n, horizon, _ = z.shape
        out = np.zeros((n, horizon + 1, self.context_dim), dtype=np.float32)
        out[:, 1:, : self.latent_dim] = z
        return out

    def online_context(
        self,
        prey_history: list[np.ndarray],
        pred_history: list[np.ndarray],
        done: np.ndarray,
        capture_t: np.ndarray,
        *,
        horizon: int,
    ) -> np.ndarray:
        """Context before the next action from positions observed so far.

        ``prey_history`` has ``completed + 1`` arrays ``(N, 2)``; ``pred_history``
        the matching ``(N, P, 2)`` arrays. Uses only completed transitions.
        """
        completed = len(prey_history) - 1
        batch = len(done)
        if completed == 0:
            return np.zeros((batch, self.context_dim), dtype=np.float32)
        prey = np.zeros((batch, horizon + 1, 2), dtype=np.float32)
        pred = np.zeros((batch, horizon + 1) + pred_history[0].shape[1:], dtype=np.float32)
        prey[:, : completed + 1] = np.stack(prey_history, axis=1)
        pred[:, : completed + 1] = np.stack(pred_history, axis=1)
        lengths = np.where(done, capture_t, completed).astype(np.int32)
        lengths = np.clip(lengths, 1, horizon)
        z = self.prefix_latents(prey, pred, lengths)
        return pad_context(z[np.arange(batch), lengths - 1], self.context_dim)

    # -- persistence ----------------------------------------------------------
    def save(self, path: Path | str) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        flat = flatten_dict(jax.tree_util.tree_map(np.asarray, self.params))
        arrays: dict[str, np.ndarray] = {
            "sequence_mean": self.sequence_mean,
            "sequence_std": self.sequence_std,
        }
        records = []
        for i, key in enumerate(sorted(flat)):
            arrays[f"parameter_{i}"] = np.asarray(flat[key])
            records.append({"path": list(key), "array": f"parameter_{i}"})
        manifest = {
            "format": "mopa-causal-context-encoder/v1",
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "context_dim": self.context_dim,
            "metadata": dict(self.metadata or {}),
            "parameters": records,
        }
        arrays["manifest"] = np.asarray(json.dumps(manifest, sort_keys=True))
        with destination.open("wb") as handle:
            np.savez_compressed(handle, **arrays)

    @classmethod
    def load(cls, path: Path | str) -> "CausalContextEncoder":
        with np.load(Path(path), allow_pickle=False) as payload:
            manifest = json.loads(str(payload["manifest"].item()))
            if manifest.get("format") != "mopa-causal-context-encoder/v1":
                raise ValueError("unsupported context encoder artifact")
            flat = {
                tuple(r["path"]): np.array(payload[r["array"]], copy=True)
                for r in manifest["parameters"]
            }
            mean = np.array(payload["sequence_mean"], copy=True)
            std = np.array(payload["sequence_std"], copy=True)
        return cls(
            params=unflatten_dict(flat),
            sequence_mean=mean,
            sequence_std=std,
            latent_dim=int(manifest["latent_dim"]),
            hidden_dim=int(manifest["hidden_dim"]),
            context_dim=int(manifest["context_dim"]),
            metadata=manifest.get("metadata", {}),
        )


def train_context_encoder(
    prey_pos: np.ndarray,
    pred_pos: np.ndarray,
    lengths: np.ndarray,
    train_episodes: np.ndarray,
    seed: int,
    *,
    latent_dim: int = 2,
    hidden_dim: int = 32,
    steps: int = 5000,
    context_dim: int = CONTEXT_DIM,
    metadata: dict[str, Any] | None = None,
) -> CausalContextEncoder:
    """Fit the GRU-JEPA on training episodes only and return the frozen encoder."""
    lengths = np.asarray(lengths, dtype=np.int32)
    train_episodes = np.asarray(train_episodes)
    raw = predator_sequence_features(
        np.asarray(prey_pos, np.float32),
        np.asarray(pred_pos, np.float32),
        lengths,
        velocity_mode="legacy_forward",
    )
    valid = np.arange(raw.shape[1])[None, :] < lengths[:, None]
    train_values = raw[train_episodes][valid[train_episodes]]
    mean = train_values.mean(axis=0).astype(np.float32)
    std = (train_values.std(axis=0) + 1e-6).astype(np.float32)
    scaled = ((raw - mean) / std).astype(np.float32) * valid[..., None]
    _, params, _ = train_jepa_gru_with_params(
        scaled[train_episodes],
        lengths[train_episodes],
        jax.random.PRNGKey(seed),
        lat=latent_dim,
        hid=hidden_dim,
        steps=steps,
    )
    details = dict(metadata or {})
    details.update({"training_seed": int(seed), "training_steps": int(steps)})
    return CausalContextEncoder(
        params=params,
        sequence_mean=mean,
        sequence_std=std,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        context_dim=context_dim,
        metadata=details,
    )
