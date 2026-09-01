"""Reusable behaviour cloning for the predator team."""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import freeze, unfreeze
from flax.linen.initializers import orthogonal
from flax.traverse_util import flatten_dict, unflatten_dict
from numpy.typing import NDArray

from mopa.features import EP_LEN
from mopa.samples import (
    build_predator_observation_samples,
    build_predator_observation_samples_with_time,
    build_predator_samples,
    build_predator_samples_with_time,
)
from mopa.splits import episode_validation_mask, group_validation_mask
from mopa.types import BCRunStats

BC_HID = 128
BC_STEPS = 4000
BC_BATCH = 256

Arrayf = NDArray[np.floating[Any]]
Arrayi = NDArray[np.integer[Any]]

__all__ = [
    "BCNet",
    "BCPolicy",
    "BCRunStats",
    "BC_BATCH",
    "BC_HID",
    "BC_STEPS",
    "bc_comparison",
    "bc_metrics_from_logits",
    "build_observation_samples",
    "build_observation_samples_with_time",
    "build_samples",
    "build_samples_with_time",
    "evaluate_bc",
    "fit_bc",
    "train_eval_bc",
    "train_eval_bc_metrics",
]


class BCNet(nn.Module):
    """Two-layer MLP mapping state (+ optional latent) to discrete action logits."""

    n_actions: int = 5
    hidden_size: int = BC_HID

    @nn.compact
    def __call__(self, x):  # noqa: ANN001
        x = nn.relu(
            nn.Dense(self.hidden_size, kernel_init=orthogonal(np.sqrt(2)))(x)
        )
        x = nn.relu(
            nn.Dense(self.hidden_size, kernel_init=orthogonal(np.sqrt(2)))(x)
        )
        return nn.Dense(self.n_actions, kernel_init=orthogonal(0.01))(x)


@dataclass(frozen=True, slots=True)
class BCPolicy:
    """Frozen MLP parameters and train-only input normalization."""

    params: Any
    mean: Arrayf
    std: Arrayf
    input_size: int
    n_actions: int
    hidden_size: int = BC_HID
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.input_size < 1 or self.n_actions < 2 or self.hidden_size < 1:
            raise ValueError("BC policy dimensions must be positive")
        mean = np.array(self.mean, dtype=np.float32, copy=True)
        std = np.array(self.std, dtype=np.float32, copy=True)
        if mean.shape != (self.input_size,) or std.shape != mean.shape:
            raise ValueError("BC normalization must match input_size")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
            raise ValueError("BC normalization must be finite")
        if np.any(std <= 0.0):
            raise ValueError("BC standard deviations must be positive")
        details = {} if self.metadata is None else dict(self.metadata)
        try:
            json.dumps(details, sort_keys=True)
        except (TypeError, ValueError) as error:
            raise ValueError("BC metadata must be JSON-serializable") from error
        mean.setflags(write=False)
        std.setflags(write=False)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)
        params = jax.tree_util.tree_map(jnp.asarray, self.params)
        object.__setattr__(self, "params", freeze(params))
        object.__setattr__(self, "metadata", MappingProxyType(details))

    def _inputs(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        if values.ndim < 1 or values.shape[-1] != self.input_size:
            raise ValueError(
                f"BC features must end with input size {self.input_size}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("BC features must contain only finite values")
        return ((values - self.mean) / self.std).astype(np.float32)

    def logits(self, features: np.ndarray) -> np.ndarray:
        """Return categorical action logits without changing RNG state."""
        model = BCNet(n_actions=self.n_actions, hidden_size=self.hidden_size)
        return np.asarray(
            model.apply(self.params, jnp.asarray(self._inputs(features))),
            dtype=np.float32,
        )

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        """Return normalized action probabilities."""
        return np.asarray(
            jax.nn.softmax(jnp.asarray(self.logits(features)), axis=-1),
            dtype=np.float32,
        )

    def greedy_action(self, features: np.ndarray) -> np.ndarray:
        """Return deterministic maximum-logit actions."""
        return np.asarray(np.argmax(self.logits(features), axis=-1), dtype=np.int32)

    def sample_action(
        self,
        features: np.ndarray,
        rng_key: jax.Array | int,
    ) -> np.ndarray:
        """Sample actions from an explicit, caller-owned JAX key."""
        key = (
            jax.random.PRNGKey(rng_key)
            if isinstance(rng_key, (int, np.integer))
            else rng_key
        )
        return np.asarray(
            jax.random.categorical(key, jnp.asarray(self.logits(features)), axis=-1),
            dtype=np.int32,
        )

    def save(self, path: Path | str) -> None:
        """Save one compact, pickle-free ``.npz`` policy artifact."""
        destination = Path(path)
        if destination.suffix != ".npz":
            raise ValueError("BC policy artifacts must use a .npz suffix")
        destination.parent.mkdir(parents=True, exist_ok=True)
        flattened = flatten_dict(unfreeze(self.params))
        arrays: dict[str, np.ndarray] = {
            "mean": np.asarray(self.mean),
            "std": np.asarray(self.std),
        }
        parameter_records: list[dict[str, Any]] = []
        for index, key in enumerate(sorted(flattened)):
            array_name = f"parameter_{index}"
            arrays[array_name] = np.asarray(flattened[key])
            parameter_records.append({"path": list(key), "array": array_name})
        manifest = {
            "format": "mopa-bc-policy/v1",
            "input_size": self.input_size,
            "n_actions": self.n_actions,
            "hidden_size": self.hidden_size,
            "metadata": dict(self.metadata),
            "parameters": parameter_records,
        }
        arrays["manifest"] = np.asarray(json.dumps(manifest, sort_keys=True))
        with destination.open("wb") as handle:
            np.savez_compressed(handle, **arrays)

    @classmethod
    def load(cls, path: Path | str) -> "BCPolicy":
        """Load a policy written by :meth:`save` with pickle disabled."""
        source = Path(path)
        with np.load(source, allow_pickle=False) as payload:
            manifest = json.loads(str(payload["manifest"].item()))
            if manifest.get("format") != "mopa-bc-policy/v1":
                raise ValueError("unsupported BC policy artifact format")
            flattened = {
                tuple(record["path"]): np.array(payload[record["array"]], copy=True)
                for record in manifest["parameters"]
            }
            mean = np.array(payload["mean"], copy=True)
            std = np.array(payload["std"], copy=True)
        return cls(
            params=freeze(unflatten_dict(flattened)),
            mean=mean,
            std=std,
            input_size=int(manifest["input_size"]),
            n_actions=int(manifest["n_actions"]),
            hidden_size=int(manifest["hidden_size"]),
            metadata=manifest.get("metadata", {}),
        )


def build_observation_samples(
    ds: Mapping[str, np.ndarray],
    t0: int = 0,
    t_max: int | None = None,
) -> tuple[Arrayf, Arrayi, Arrayi]:
    """Build exact pre-action observation samples, including ``t = 0``."""
    if t_max is None:
        t_max = int(ds["pred_act"].shape[1])
    return build_predator_observation_samples(dict(ds), t0=t0, t1=t_max)


def build_observation_samples_with_time(
    ds: Mapping[str, np.ndarray],
    t0: int = 0,
    t_max: int | None = None,
) -> tuple[Arrayf, Arrayi, Arrayi, Arrayi, Arrayi]:
    """Build exact observation samples plus timestep and predator identity."""
    if t_max is None:
        t_max = int(ds["pred_act"].shape[1])
    return build_predator_observation_samples_with_time(
        dict(ds), t0=t0, t1=t_max
    )


def build_samples(
    ds: Mapping[str, np.ndarray],
    ctx: int,
    ep_len: int = EP_LEN,
    t_max: int | None = None,
) -> tuple[Arrayf, Arrayi, Arrayi]:
    """Build ``(state, action, episode_id)`` samples for every predator.

    State = absolute positions of all agents + velocity proxy + predator id
    one-hot. Valid steps: ``t in [ctx, t_max)`` excluding the auto-reset
    boundary and post-capture frozen frames when ``capture_t`` is present.
    """
    if t_max is None:
        t_max = min(ep_len, int(ds["pred_act"].shape[1]))
    capture_t = ds["capture_t"] if "capture_t" in ds else None
    return build_predator_samples(
        dict(ds), ctx, t_max, ep_len=ep_len, capture_t=capture_t
    )


def build_samples_with_time(
    ds: Mapping[str, np.ndarray],
    ctx: int,
    ep_len: int = EP_LEN,
    t_max: int | None = None,
) -> tuple[Arrayf, Arrayi, Arrayi, Arrayi, Arrayi]:
    """Build samples plus ``(timestep, predator_id)`` causal provenance."""
    if t_max is None:
        t_max = min(ep_len, int(ds["pred_act"].shape[1]))
    capture_t = ds["capture_t"] if "capture_t" in ds else None
    return build_predator_samples_with_time(
        dict(ds), ctx, t_max, ep_len=ep_len, capture_t=capture_t
    )


def fit_bc(
    features: Arrayf,
    actions: Arrayi,
    rng_seed: int,
    *,
    steps: int = BC_STEPS,
    n_actions: int = 5,
    metadata: Mapping[str, Any] | None = None,
) -> BCPolicy:
    """Fit one vanilla MLP and return a reusable frozen policy."""
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(actions)
    if x.ndim != 2 or x.shape[1] < 1:
        raise ValueError("BC features must have shape (samples, features)")
    if y.ndim != 1 or len(y) != len(x):
        raise ValueError("BC actions must be a sample-aligned vector")
    if len(x) == 0:
        raise ValueError("BC training requires at least one sample")
    if not np.all(np.isfinite(x)):
        raise ValueError("BC features must contain only finite values")
    if not np.issubdtype(y.dtype, np.integer):
        if not np.all(np.equal(y, np.floor(y))):
            raise ValueError("BC actions must contain integers")
    y = y.astype(np.int32)
    if n_actions < 2 or np.any(y < 0) or np.any(y >= n_actions):
        raise ValueError("BC actions must lie within n_actions")
    if steps < 0:
        raise ValueError("BC steps cannot be negative")

    mean = x.mean(axis=0).astype(np.float32)
    std = (x.std(axis=0) + 1e-6).astype(np.float32)
    normalized = ((x - mean) / std).astype(np.float32)
    model = BCNet(n_actions=n_actions, hidden_size=BC_HID)
    key = jax.random.PRNGKey(rng_seed)
    key, init_key = jax.random.split(key)
    params = model.init(init_key, normalized[:1])
    optimizer = optax.adam(1e-3)
    optimizer_state = optimizer.init(params)
    x_jax = jnp.asarray(normalized)
    y_jax = jnp.asarray(y)

    def loss_fn(model_params, batch_x, batch_y):  # noqa: ANN001
        logits = model.apply(model_params, batch_x)
        return optax.softmax_cross_entropy_with_integer_labels(
            logits, batch_y
        ).mean()

    @jax.jit
    def update(model_params, state, indices):  # noqa: ANN001
        gradient = jax.grad(loss_fn)(
            model_params, x_jax[indices], y_jax[indices]
        )
        updates, state = optimizer.update(gradient, state)
        return optax.apply_updates(model_params, updates), state

    for _ in range(steps):
        key, batch_key = jax.random.split(key)
        indices = jax.random.randint(
            batch_key,
            (min(BC_BATCH, len(x)),),
            0,
            len(x),
        )
        params, optimizer_state = update(params, optimizer_state, indices)

    details = {} if metadata is None else dict(metadata)
    details.update(
        {
            "training_seed": int(rng_seed),
            "training_steps": int(steps),
        }
    )
    return BCPolicy(
        params=params,
        mean=mean,
        std=std,
        input_size=int(x.shape[1]),
        n_actions=int(n_actions),
        hidden_size=BC_HID,
        metadata=details,
    )


def bc_metrics_from_logits(
    logits: np.ndarray,
    actions: np.ndarray,
    *,
    episode_ids: np.ndarray | None = None,
    strategy_labels: np.ndarray | None = None,
) -> dict[str, Any]:
    scores = np.asarray(logits, dtype=np.float64)
    target = np.asarray(actions)
    if scores.ndim != 2 or scores.shape[1] < 2:
        raise ValueError("BC logits must have shape (samples, actions)")
    if target.ndim != 1 or len(target) != len(scores) or len(target) == 0:
        raise ValueError("BC targets must be a non-empty aligned vector")
    if not np.issubdtype(target.dtype, np.integer):
        if not np.all(np.equal(target, np.floor(target))):
            raise ValueError("BC targets must contain integers")
    target = target.astype(np.int32)
    if np.any(target < 0) or np.any(target >= scores.shape[1]):
        raise ValueError("BC target is outside the action vocabulary")
    if not np.all(np.isfinite(scores)):
        raise ValueError("BC logits must be finite")

    shifted = scores - scores.max(axis=-1, keepdims=True)
    log_probabilities = shifted - np.log(
        np.exp(shifted).sum(axis=-1, keepdims=True)
    )
    per_sample_nll = -log_probabilities[np.arange(len(target)), target]
    correct = np.argmax(scores, axis=-1) == target
    per_action_accuracy = {
        str(int(action)): float(correct[target == action].mean())
        for action in np.unique(target)
    }
    result: dict[str, Any] = {
        "accuracy": float(correct.mean()),
        "nll": float(per_sample_nll.mean()),
        "action_balanced_accuracy": float(
            np.mean(list(per_action_accuracy.values()))
        ),
        "per_action_accuracy": per_action_accuracy,
        "n_samples": int(len(target)),
    }

    if episode_ids is None:
        if strategy_labels is not None:
            raise ValueError("strategy_labels require episode_ids")
        return result
    episodes = np.asarray(episode_ids)
    if episodes.ndim != 1 or len(episodes) != len(target):
        raise ValueError("episode_ids must align with BC samples")
    unique_episodes = np.unique(episodes)
    episode_nll = np.asarray(
        [per_sample_nll[episodes == episode].mean() for episode in unique_episodes]
    )
    episode_accuracy = np.asarray(
        [correct[episodes == episode].mean() for episode in unique_episodes]
    )
    result.update(
        {
            "episode_macro_nll": float(episode_nll.mean()),
            "episode_macro_accuracy": float(episode_accuracy.mean()),
            "n_episodes": int(len(unique_episodes)),
        }
    )

    if strategy_labels is None:
        return result
    labels = np.asarray(strategy_labels)
    if labels.ndim != 1 or len(labels) != len(target):
        raise ValueError("strategy_labels must align with BC samples")
    episode_labels: list[Any] = []
    for episode in unique_episodes:
        values = np.unique(labels[episodes == episode])
        if len(values) != 1:
            raise ValueError("each episode must have exactly one strategy label")
        episode_labels.append(values[0])
    episode_labels_array = np.asarray(episode_labels)
    per_strategy: dict[str, dict[str, float | int]] = {}
    strategy_nll: list[float] = []
    strategy_accuracy: list[float] = []
    for label in np.unique(episode_labels_array):
        selected_episodes = episode_labels_array == label
        sample_rows = labels == label
        label_nll = float(episode_nll[selected_episodes].mean())
        label_accuracy = float(episode_accuracy[selected_episodes].mean())
        strategy_nll.append(label_nll)
        strategy_accuracy.append(label_accuracy)
        label_value = label.item() if isinstance(label, np.generic) else label
        per_strategy[str(label_value)] = {
            "episode_macro_nll": label_nll,
            "episode_macro_accuracy": label_accuracy,
            "n_episodes": int(selected_episodes.sum()),
            "n_samples": int(sample_rows.sum()),
        }
    result.update(
        {
            "episode_strategy_macro_nll": float(np.mean(strategy_nll)),
            "episode_strategy_macro_accuracy": float(
                np.mean(strategy_accuracy)
            ),
            "per_strategy": per_strategy,
            "n_strategies": int(len(per_strategy)),
        }
    )
    return result


def evaluate_bc(
    policy: BCPolicy,
    features: Arrayf,
    actions: Arrayi,
    *,
    episode_ids: Arrayi | None = None,
    strategy_labels: Arrayi | None = None,
) -> dict[str, Any]:
    """Evaluate a frozen policy with pooled and episode-macro metrics."""
    return bc_metrics_from_logits(
        policy.logits(features),
        actions,
        episode_ids=episode_ids,
        strategy_labels=strategy_labels,
    )


def train_eval_bc(
    S: Arrayf,
    A: Arrayi,
    ep: Arrayi,
    rng_seed: int,
    val_frac: float = 0.2,
    steps: int = BC_STEPS,
    group_ids: Arrayi | None = None,
    split_seed: int | None = None,
    validation_mask: NDArray[np.bool_] | None = None,
) -> float:
    """Train a BC net and return held-out action accuracy.

    By default splits by episode. When ``group_ids`` is provided (e.g.
    checkpoint seeds broadcast to sample rows), holds out whole groups.
    ``split_seed`` defaults to ``rng_seed`` for backward compatibility.
    """
    return float(
        train_eval_bc_metrics(
            S,
            A,
            ep,
            rng_seed,
            val_frac=val_frac,
            steps=steps,
            group_ids=group_ids,
            split_seed=split_seed,
            validation_mask=validation_mask,
        )["accuracy"]
    )


def train_eval_bc_metrics(
    S: Arrayf,
    A: Arrayi,
    ep: Arrayi,
    rng_seed: int,
    val_frac: float = 0.2,
    steps: int = BC_STEPS,
    group_ids: Arrayi | None = None,
    split_seed: int | None = None,
    validation_mask: NDArray[np.bool_] | None = None,
) -> dict[str, Any]:
    """Train BC once and return held-out accuracy and action NLL.

    ``validation_mask`` is the preferred experiment-driver interface: it is a
    sample-aligned mask derived from the single manifest split and therefore
    stays identical across model initialization seeds.  The legacy split
    arguments remain for small standalone calls.
    """
    if validation_mask is not None:
        vmask = np.asarray(validation_mask, dtype=bool)
        if vmask.shape != (len(ep),):
            raise ValueError("validation_mask must align with samples")
    else:
        split_rng = rng_seed if split_seed is None else split_seed
        if group_ids is None:
            vmask = episode_validation_mask(ep, rng_seed=split_rng, val_frac=val_frac)
        else:
            if len(group_ids) != len(ep):
                raise ValueError("group_ids must align with samples")
            vmask = group_validation_mask(
                group_ids, rng_seed=split_rng, val_frac=val_frac
            )
    Str, Atr, Sva, Ava = S[~vmask], A[~vmask], S[vmask], A[vmask]
    if len(Str) == 0 or len(Sva) == 0:
        raise ValueError("BC split must contain both train and validation samples")
    policy = fit_bc(Str, Atr, rng_seed, steps=steps)
    metrics = evaluate_bc(policy, Sva, Ava, episode_ids=np.asarray(ep)[vmask])
    metrics.update({"n_train": int(len(Str)), "n_val": int(len(Sva))})
    return metrics


def bc_comparison(
    ds: Mapping[str, np.ndarray],
    z_dict: Mapping[str, np.ndarray | None],
    ctx: int,
    seeds: Sequence[int] = (0, 1, 2),
    group_ids: Arrayi | None = None,
    split: str = "episode",
    split_seed: int = 0,
) -> dict[str, BCRunStats]:
    """Run BC for each conditioning variant.

    ``z_dict`` maps variant name → per-episode conditioning array ``(N, d)``,
    or ``None`` for the unconditioned baseline.

    ``split="checkpoint"`` uses ``ds['ckpt_seed']`` (or provided ``group_ids``)
    broadcast onto sample rows so held-out checkpoints never appear in train.
    """
    S0, A, ep = build_samples(ds, ctx)
    sample_groups = None
    if split == "checkpoint":
        if group_ids is None:
            if "ckpt_seed" not in ds:
                raise ValueError("checkpoint split requires ckpt_seed on the dataset")
            group_ids = np.asarray(ds["ckpt_seed"])
        sample_groups = np.asarray(group_ids, dtype=np.int32)[ep]
    elif split != "episode":
        raise ValueError(f"unknown split={split!r}")

    results: dict[str, BCRunStats] = {}
    for name, z in z_dict.items():
        if z is None:
            S = S0
        else:
            # Append raw z; train_eval_bc standardizes from the train fold only
            # so val episodes never influence feature scale.
            S = np.concatenate([S0, np.asarray(z, dtype=np.float32)[ep]], -1)
        runs = tuple(
            train_eval_bc(
                S,
                A,
                ep,
                s,
                group_ids=sample_groups,
                split_seed=split_seed,
            )
            for s in seeds
        )
        v = np.asarray(runs)
        results[name] = BCRunStats(mean=float(v.mean()), std=float(v.std()), runs=runs)
        suffix = f",{name}" if z is not None else ""
        print(f"  BC pi(a|s{suffix}) : {v.mean():.4f} +/- {v.std():.4f}")
    return results
