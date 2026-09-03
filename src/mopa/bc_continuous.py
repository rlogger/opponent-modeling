"""Continuous opponent behaviour cloning (Gate 2 vanilla baseline).

    v_hat = tanh(MLP(red_observation [, context]))
    loss  = mean_squared_error(v_hat, v)

The deterministic MSE head is the declared vanilla baseline. The policy is a
Flax ``struct.PyTreeNode`` so its parameters and input normalization live in an
explicit JAX PyTree and can be called from jitted planners and world-model
rollouts unchanged (handoff Gate 4 requirement).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from flax.linen.initializers import orthogonal
from flax.traverse_util import flatten_dict, unflatten_dict

from mopa.bc import BC_BATCH, BC_HID, BC_STEPS
from tag_objectives.actions import CONTINUOUS_ACTION_DIM

__all__ = [
    "ContinuousBCNet",
    "ContinuousBCPolicy",
    "continuous_bc_metrics",
    "fit_continuous_bc",
]


class ContinuousBCNet(nn.Module):
    """Two-layer MLP with a tanh-bounded ``action_dim`` head."""

    action_dim: int = CONTINUOUS_ACTION_DIM
    hidden_size: int = BC_HID

    @nn.compact
    def __call__(self, x):  # noqa: ANN001
        x = nn.relu(nn.Dense(self.hidden_size, kernel_init=orthogonal(np.sqrt(2)))(x))
        x = nn.relu(nn.Dense(self.hidden_size, kernel_init=orthogonal(np.sqrt(2)))(x))
        return jnp.tanh(nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(x))


class ContinuousBCPolicy(struct.PyTreeNode):
    """Frozen continuous BC policy as an explicit JAX PyTree.

    ``params`` / ``mean`` / ``std`` are pytree leaves; the architecture and
    JSON metadata are static. ``act`` is pure ``jax.numpy`` and jit/vmap safe.
    """

    params: Any
    mean: jax.Array
    std: jax.Array
    input_size: int = struct.field(pytree_node=False)
    action_dim: int = struct.field(pytree_node=False, default=CONTINUOUS_ACTION_DIM)
    hidden_size: int = struct.field(pytree_node=False, default=BC_HID)
    metadata_json: str = struct.field(pytree_node=False, default="{}")

    @property
    def metadata(self) -> dict[str, Any]:
        return json.loads(self.metadata_json)

    def act(self, features: jax.Array) -> jax.Array:
        """Deterministic bounded action ``tanh(MLP((x - mean) / std))``."""
        x = (jnp.asarray(features, jnp.float32) - self.mean) / self.std
        net = ContinuousBCNet(action_dim=self.action_dim, hidden_size=self.hidden_size)
        return net.apply({"params": self.params}, x)

    def predict(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        if values.ndim < 1 or values.shape[-1] != self.input_size:
            raise ValueError(f"features must end with input size {self.input_size}")
        if not np.all(np.isfinite(values)):
            raise ValueError("features must be finite")
        return np.asarray(jax.jit(self.act)(jnp.asarray(values)), dtype=np.float32)

    def save(self, path: Path | str) -> None:
        destination = Path(path)
        if destination.suffix != ".npz":
            raise ValueError("continuous BC artifacts must use a .npz suffix")
        destination.parent.mkdir(parents=True, exist_ok=True)
        flat = flatten_dict(jax.tree_util.tree_map(np.asarray, self.params))
        arrays: dict[str, np.ndarray] = {
            "mean": np.asarray(self.mean),
            "std": np.asarray(self.std),
        }
        records = []
        for i, key in enumerate(sorted(flat)):
            arrays[f"parameter_{i}"] = np.asarray(flat[key])
            records.append({"path": list(key), "array": f"parameter_{i}"})
        manifest = {
            "format": "mopa-continuous-bc-policy/v1",
            "input_size": self.input_size,
            "action_dim": self.action_dim,
            "hidden_size": self.hidden_size,
            "metadata": self.metadata,
            "parameters": records,
        }
        arrays["manifest"] = np.asarray(json.dumps(manifest, sort_keys=True))
        with destination.open("wb") as handle:
            np.savez_compressed(handle, **arrays)

    @classmethod
    def load(cls, path: Path | str) -> "ContinuousBCPolicy":
        with np.load(Path(path), allow_pickle=False) as payload:
            manifest = json.loads(str(payload["manifest"].item()))
            if manifest.get("format") != "mopa-continuous-bc-policy/v1":
                raise ValueError("unsupported continuous BC artifact format")
            flat = {
                tuple(r["path"]): jnp.asarray(payload[r["array"]])
                for r in manifest["parameters"]
            }
            mean = jnp.asarray(payload["mean"])
            std = jnp.asarray(payload["std"])
        return cls(
            params=unflatten_dict(flat),
            mean=mean,
            std=std,
            input_size=int(manifest["input_size"]),
            action_dim=int(manifest["action_dim"]),
            hidden_size=int(manifest["hidden_size"]),
            metadata_json=json.dumps(manifest.get("metadata", {}), sort_keys=True),
        )


def fit_continuous_bc(
    features: np.ndarray,
    actions: np.ndarray,
    rng_seed: int,
    *,
    steps: int = BC_STEPS,
    batch_size: int = BC_BATCH,
    learning_rate: float = 1e-3,
    hidden_size: int = BC_HID,
    metadata: dict[str, Any] | None = None,
) -> ContinuousBCPolicy:
    """Fit ``tanh(MLP(x))`` to bounded actions with mean squared error."""
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(actions, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] < 1 or len(x) == 0:
        raise ValueError("features must have shape (samples, features)")
    if y.ndim != 2 or y.shape[0] != len(x) or y.shape[1] != CONTINUOUS_ACTION_DIM:
        raise ValueError("actions must have shape (samples, 2)")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("features and actions must be finite")
    if np.any(np.abs(y) > 1.0):
        raise ValueError("actions must lie in [-1, 1]")
    if steps < 0:
        raise ValueError("steps cannot be negative")

    mean = x.mean(axis=0).astype(np.float32)
    std = (x.std(axis=0) + 1e-6).astype(np.float32)
    normalized = ((x - mean) / std).astype(np.float32)
    net = ContinuousBCNet(action_dim=CONTINUOUS_ACTION_DIM, hidden_size=hidden_size)
    key = jax.random.PRNGKey(rng_seed)
    key, init_key = jax.random.split(key)
    params = net.init(init_key, normalized[:1])["params"]
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    x_jax = jnp.asarray(normalized)
    y_jax = jnp.asarray(y)

    def loss_fn(p, bx, by):  # noqa: ANN001
        pred = net.apply({"params": p}, bx)
        return jnp.mean(jnp.sum((pred - by) ** 2, axis=-1))

    @jax.jit
    def update(p, state, idx):  # noqa: ANN001
        grads = jax.grad(loss_fn)(p, x_jax[idx], y_jax[idx])
        updates, state = optimizer.update(grads, state)
        return optax.apply_updates(p, updates), state

    n = len(x)
    for _ in range(steps):
        key, bk = jax.random.split(key)
        idx = jax.random.randint(bk, (min(batch_size, n),), 0, n)
        params, opt_state = update(params, opt_state, idx)

    details = dict(metadata or {})
    details.update(
        {
            "training_seed": int(rng_seed),
            "training_steps": int(steps),
            "loss": "mean_squared_error",
            "head": "tanh",
        }
    )
    return ContinuousBCPolicy(
        params=params,
        mean=jnp.asarray(mean),
        std=jnp.asarray(std),
        input_size=int(x.shape[1]),
        action_dim=CONTINUOUS_ACTION_DIM,
        hidden_size=hidden_size,
        metadata_json=json.dumps(details, sort_keys=True),
    )


def continuous_bc_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    *,
    episode_ids: np.ndarray | None = None,
    strategy_labels: np.ndarray | None = None,
    agreement_tolerance: float = 0.25,
    saturation_threshold: float = 0.95,
) -> dict[str, Any]:
    """Pooled, episode-macro, and strategy-macro continuous action metrics.

    ``mse`` is the mean over samples of the squared L2 error summed over the
    two action dimensions (matches the training loss); ``mae`` averages the
    absolute error over samples and dimensions; ``direction_cosine`` averages
    the cosine between predicted and target vectors over samples whose target
    norm exceeds ``1e-3``; ``agreement`` is the fraction of samples with L2
    error at most ``agreement_tolerance``.
    """
    pred = np.asarray(predictions, dtype=np.float64)
    tgt = np.asarray(targets, dtype=np.float64)
    if pred.shape != tgt.shape or pred.ndim != 2 or pred.shape[1] != CONTINUOUS_ACTION_DIM:
        raise ValueError("predictions and targets must both have shape (samples, 2)")
    if len(pred) == 0:
        raise ValueError("metrics need at least one sample")
    if not np.all(np.isfinite(pred)) or not np.all(np.isfinite(tgt)):
        raise ValueError("predictions and targets must be finite")
    sq = np.sum((pred - tgt) ** 2, axis=-1)
    abs_err = np.mean(np.abs(pred - tgt), axis=-1)
    l2 = np.sqrt(sq)
    tn = np.linalg.norm(tgt, axis=-1)
    pn = np.linalg.norm(pred, axis=-1)
    moving = tn > 1e-3
    cosine = np.full(len(pred), np.nan)
    cosine[moving] = np.sum(pred[moving] * tgt[moving], axis=-1) / (
        pn[moving] * tn[moving] + 1e-8
    )
    agree = l2 <= agreement_tolerance
    result: dict[str, Any] = {
        "mse": float(sq.mean()),
        "rmse": float(np.sqrt(sq.mean())),
        "mae": float(abs_err.mean()),
        "direction_cosine": float(np.nanmean(cosine)) if moving.any() else None,
        "agreement": float(agree.mean()),
        "agreement_tolerance": float(agreement_tolerance),
        "prediction_saturation": float((np.abs(pred) > saturation_threshold).any(1).mean()),
        "target_saturation": float((np.abs(tgt) > saturation_threshold).any(1).mean()),
        "n_samples": int(len(pred)),
    }
    if episode_ids is None:
        if strategy_labels is not None:
            raise ValueError("strategy_labels require episode_ids")
        return result
    episodes = np.asarray(episode_ids)
    if episodes.shape != (len(pred),):
        raise ValueError("episode_ids must align with samples")
    uniq = np.unique(episodes)
    ep_mse = np.asarray([sq[episodes == e].mean() for e in uniq])
    ep_cos = np.asarray([np.nanmean(cosine[episodes == e]) for e in uniq])
    ep_agree = np.asarray([agree[episodes == e].mean() for e in uniq])
    result.update(
        {
            "episode_macro_mse": float(ep_mse.mean()),
            "episode_macro_direction_cosine": float(np.nanmean(ep_cos)),
            "episode_macro_agreement": float(ep_agree.mean()),
            "n_episodes": int(len(uniq)),
        }
    )
    if strategy_labels is None:
        return result
    labels = np.asarray(strategy_labels)
    if labels.shape != (len(pred),):
        raise ValueError("strategy_labels must align with samples")
    ep_label = []
    for e in uniq:
        vals = np.unique(labels[episodes == e])
        if len(vals) != 1:
            raise ValueError("each episode must have exactly one strategy label")
        ep_label.append(vals[0])
    ep_label = np.asarray(ep_label)
    per_strategy: dict[str, Any] = {}
    s_mse, s_cos, s_agree = [], [], []
    for label in np.unique(ep_label):
        sel = ep_label == label
        m, c, a = float(ep_mse[sel].mean()), float(np.nanmean(ep_cos[sel])), float(ep_agree[sel].mean())
        s_mse.append(m)
        s_cos.append(c)
        s_agree.append(a)
        key = label.item() if isinstance(label, np.generic) else label
        per_strategy[str(key)] = {
            "episode_macro_mse": m,
            "episode_macro_direction_cosine": c,
            "episode_macro_agreement": a,
            "n_episodes": int(sel.sum()),
            "n_samples": int((labels == label).sum()),
        }
    result.update(
        {
            "episode_strategy_macro_mse": float(np.mean(s_mse)),
            "episode_strategy_macro_direction_cosine": float(np.mean(s_cos)),
            "episode_strategy_macro_agreement": float(np.mean(s_agree)),
            "per_strategy": per_strategy,
            "n_strategies": int(len(per_strategy)),
        }
    )
    return result
