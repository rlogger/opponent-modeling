"""Versioned, information-matched opponent predictors for continuous 1v1 tag.

At decision t the decoder receives f(x_t); the encoder receives completed
(f(x_s), v_s), s < t, in nonoverlapping windows. Completed windows and the
current partial window are pooled exactly as in the historical 0s adapter.
Training/deployment use the same ordering, right padding and masks.
Reconstruction remains in the separately retained original 0s implementation.

Loss = episode-equal squared vector-action error + annealed free-bits window KL.
The squared error sums the two action coordinates. This is a regularized
prediction objective, not an assertion of a calibrated strategy posterior.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isqrt
from pathlib import Path
from typing import Any, NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
from flax.training.train_state import TrainState

from mopa.action_decoder import GRUCore, MLPHead
from mopa.zero_s import zero_s_features

SCHEMA = "causal_opponent_v1"
METHODS = ("bc", "deterministic_history", "mlp_vae", "recurrent_vae")
FEATURES = ("agent8_v1", "expert17_v1")
OBJECTIVES = ("past_next_action",)


@dataclass(frozen=True)
class CausalOpponentConfig:
    method: str = "recurrent_vae"
    feature_schema: str = "expert17_v1"
    history: int = 8
    max_history: int | None = None
    latent_dim: int = 8
    hid: int = 64
    encoder_hid: int | None = None
    steps: int = 1000
    batch: int = 128
    learning_rate: float = 1e-3
    beta: float = 1.0
    free_bits: float = 0.2
    sample_training: bool = True
    objective: str = "past_next_action"
    sampler: str = "episode_uniform"

    def __post_init__(self):
        if self.method not in METHODS or self.feature_schema not in FEATURES:
            raise ValueError("unsupported method or feature schema")
        if self.objective not in OBJECTIVES:
            raise ValueError("unsupported training objective")
        if self.sampler not in {"episode_uniform", "transition_uniform"}:
            raise ValueError("unsupported training sample measure")
        for name in ("history", "steps", "latent_dim", "hid", "batch"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer")
        if self.history < 0 or self.steps < 0 or min(self.latent_dim, self.hid, self.batch) < 1:
            raise ValueError("invalid history, steps, width or batch")
        if self.max_history is not None and (not isinstance(self.max_history, int) or isinstance(self.max_history, bool) or not 0 <= self.max_history <= self.history):
            raise ValueError("max_history must be None or between zero and window width")
        if self.encoder_hid is not None and (not isinstance(self.encoder_hid, int) or isinstance(self.encoder_hid, bool) or self.encoder_hid < 1):
            raise ValueError("encoder_hid must be a positive integer")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not np.isfinite(self.beta) or self.beta < 0 or not np.isfinite(self.free_bits) or self.free_bits < 0:
            raise ValueError("beta and free_bits must be finite and nonnegative")
        if self.method == "bc" and self.history != 0:
            raise ValueError("state-only BC requires history=0")
        if not self.variational and (self.beta != 0 or self.sample_training):
            raise ValueError("BC/deterministic predictors require beta=0 and sample_training=False")

    @property
    def variational(self):
        return self.method in {"mlp_vae", "recurrent_vae"}

    @property
    def lat(self):
        return 0 if self.method == "bc" else self.latent_dim

    @property
    def feature_dim(self):
        return 8 if self.feature_schema == "agent8_v1" else 17

    @property
    def encoder_width(self):
        return self.hid if self.encoder_hid is None else self.encoder_hid


def encoder_parameter_count(config):
    """Exact count for these declared architectures, before any fitting."""
    c, h = config, config.encoder_width
    if c.history == 0 or c.method == "bc":
        return 0
    heads = (2 if c.variational else 1) * c.lat
    if c.method == "recurrent_vae":
        return 6*h*h + (c.feature_dim + 10 + heads)*h + heads
    return h*h + (c.history*(c.feature_dim+3) + 2 + heads)*h + heads


def match_encoder_capacity(config, reference_config):
    """Choose the closest MLP width to a specified recurrent parameter budget.

    This deterministic pre-fit choice keeps decoder width and latent dimension
    fixed. It reports, rather than conceals, the unavoidable integer-width gap.
    """
    from dataclasses import replace

    if config.method not in {"mlp_vae", "deterministic_history"} or not config.history:
        raise ValueError("capacity matching requires a nonempty MLP history encoder")
    if (config.feature_schema, config.history, config.max_history, config.lat, config.hid) != (reference_config.feature_schema, reference_config.history, reference_config.max_history, reference_config.lat, reference_config.hid):
        raise ValueError("match features, window, latent and decoder widths first")
    target = encoder_parameter_count(reference_config)
    width = min(range(1, isqrt(target) + 2), key=lambda h: (abs(encoder_parameter_count(replace(config, encoder_hid=h)) - target), h))
    selected = replace(config, encoder_hid=width)
    count = encoder_parameter_count(selected)
    return selected, {"reference_encoder_parameters": target, "encoder_parameters": count,
                      "encoder_hidden_size": width, "parameter_gap": count-target,
                      "relative_parameter_gap": (count-target)/max(target, 1)}


def opponent_features(state: Any, schema: str) -> jax.Array:
    """Agent coordinates plus expert-visible lava, excluding hidden resources.

    expert17 is information-equivalent to the red observation for the fixed
    66D schema (1v1, no landmarks, 16 resources, three visible lava disks).
    Absolute blue position is redundant given red position and relative blue.
    """
    x = jnp.asarray(state, jnp.float32)
    agent = zero_s_features(x)
    if schema == "agent8_v1":
        return agent
    if schema != "expert17_v1":
        raise ValueError("unsupported feature schema")
    relative = x[..., 56:62].reshape(x.shape[:-1] + (3, 2)) - x[..., None, :2]
    _, index = jax.lax.top_k(-jnp.linalg.norm(relative, axis=-1), 3)
    ordered = jnp.take_along_axis(relative, index[..., None], axis=-2)
    radii = jnp.take_along_axis(x[..., 62:65], index, axis=-1)
    geometry = jnp.concatenate([ordered, radii[..., None]], axis=-1)
    return jnp.concatenate([agent, geometry.reshape(x.shape[:-1] + (9,))], axis=-1)


class HistoryEncoder(nn.Module):
    config: CausalOpponentConfig

    @nn.compact
    def __call__(self, history, mask):
        cfg = self.config
        shape = (history.shape[0], cfg.lat)
        if cfg.history == 0 or cfg.method == "bc":
            return jnp.zeros(shape), jnp.zeros(shape)
        # Each independent window is right padded as in the original 0s.
        # Mask before arithmetic: NaN padding is absent data, never 0*NaN.
        history = jnp.where(mask[..., None], history, 0.0)
        if cfg.method == "recurrent_vae":
            recurrent = GRUCore(hid=cfg.encoder_width)(history, mask)
            last = jnp.maximum(mask.sum(-1) - 1, 0)
            hidden = recurrent[jnp.arange(len(history)), last]
        else:
            hidden = jnp.concatenate([history.reshape(history.shape[0], -1), mask], axis=-1)
            hidden = nn.relu(nn.Dense(cfg.encoder_width)(hidden))
            hidden = nn.relu(nn.Dense(cfg.encoder_width)(hidden))
        mean = nn.Dense(cfg.lat)(hidden)
        logvar = nn.Dense(cfg.lat)(hidden) if cfg.variational else jnp.zeros_like(mean)
        present = jnp.any(mask, axis=-1, keepdims=True)
        return jnp.where(present, mean, 0.0), jnp.where(present, logvar, 0.0)


def gaussian_kl(mean, logvar):
    """Summed latent KL for each example; no free-bit floor or batch coupling."""
    return 0.5 * jnp.sum(mean**2 + jnp.exp(logvar) - 1.0 - logvar, axis=-1)


def pooled_posterior(params, config, window, mask):
    """0s mean pooling; sampled training averages independent window latents.

    This Gaussian describes that arithmetic average, not a Bayesian posterior
    obtained by combining independent evidence about one strategy.
    """
    batch = window.shape[0]
    if config.history == 0:
        zeros = jnp.zeros((batch, config.lat))
        return zeros, zeros, zeros[:, None], zeros[:, None], jnp.zeros((batch, 1), bool)
    if window.ndim == 3:
        window, mask = window[:, None], mask[:, None]
    n_windows = window.shape[1]
    mu, lv = HistoryEncoder(config).apply(params, window.reshape(batch*n_windows, config.history, -1), mask.reshape(batch*n_windows, config.history))
    mu, lv = mu.reshape(batch, n_windows, -1), lv.reshape(batch, n_windows, -1)
    present = jnp.any(mask, axis=-1)
    count = present.sum(-1, keepdims=True)
    safe = jnp.maximum(count, 1)
    pooled_mu = jnp.sum(jnp.where(present[..., None], mu, 0.0), axis=1) / safe
    variance = jnp.sum(jnp.where(present[..., None], jnp.exp(lv), 0.0), axis=1) / safe**2
    variance = jnp.where(count > 0, variance, 1.0)
    return pooled_mu, jnp.log(variance), mu, lv, present


def history_batch(features, actions, episodes, times, width, *, max_history=None):
    """Gather all available nonoverlapping windows, padding only their tails.

    Target times must be valid transitions; only strictly earlier pairs enter.
    """
    end = times
    n_windows = (features.shape[1] + width - 1) // width if width else 0
    if max_history is not None and width:
        # A separate finite-history comparison with identical window width.
        # Right pad the last permitted pairs; no older pair enters the encoder.
        count = jnp.minimum(end, max_history)
        indices = jnp.maximum(end - max_history, 0)[:, None] + jnp.arange(width)[None]
        valid = jnp.arange(width)[None] < count[:, None]
        safe = jnp.minimum(indices, features.shape[1] - 1)
        joined = jnp.concatenate([features[episodes[:, None], safe], actions[episodes[:, None], safe]], axis=-1)
        return jnp.where(valid[..., None], joined, 0.0)[:, None], valid[:, None]
    indices = jnp.arange(n_windows * width).reshape(n_windows, width)
    valid = indices[None] < end[:, None, None]
    safe = jnp.minimum(indices, features.shape[1] - 1)
    joined = jnp.concatenate([features[episodes[:, None, None], safe[None]], actions[episodes[:, None, None], safe[None]]], axis=-1)
    return jnp.where(valid[..., None], joined, 0.0), valid


class CausalCarry(NamedTuple):
    window: jax.Array
    mask: jax.Array
    observed: jax.Array
    context: jax.Array
    completed_sum: jax.Array
    completed_variance: jax.Array


@dataclass(frozen=True)
class CausalOpponent:
    config: CausalOpponentConfig
    params: Any
    state_mean: np.ndarray
    state_std: np.ndarray
    prototypes: np.ndarray

    def __post_init__(self):
        dim = self.config.feature_dim
        mean, std = np.asarray(self.state_mean), np.asarray(self.state_std)
        if mean.shape != (dim,) or std.shape != (dim,):
            raise ValueError("normalization does not match versioned feature schema")
        if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
            raise ValueError("normalization must be finite with positive scale")
        if np.asarray(self.prototypes).shape != (3, self.config.lat) or not np.isfinite(self.prototypes).all():
            raise ValueError("three finite diagnostic prototypes required")
        if set(self.params) != {"encoder", "decoder"}:
            raise ValueError("encoder and decoder parameters required")
        if any(not np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(self.params)):
            raise ValueError("nonfinite parameters")

    @property
    def encoder(self):
        """Adapter compatibility: config.lat and training statistics are frozen."""
        return self

    def features(self, state):
        return (opponent_features(state, self.config.feature_schema) - self.state_mean) / self.state_std

    def posterior(self, window, mask):
        return pooled_posterior(self.params["encoder"], self.config, window, mask)[:2]

    def actions(self, state, context):
        features = self.features(state)
        value = MLPHead(out=2, hid=self.config.hid).apply(
            self.params["decoder"], jnp.concatenate([context, features], axis=-1),
        )
        return jnp.tanh(value)

    def initial_context(self, batch_size):
        if batch_size < 1:
            raise ValueError("positive batch size required")
        cfg = self.config
        return CausalCarry(
            jnp.zeros((batch_size, cfg.history, cfg.feature_dim + 2)),
            jnp.zeros((batch_size, cfg.history), bool),
            jnp.zeros(batch_size, jnp.int32),
            jnp.zeros((batch_size, cfg.lat)),
            jnp.zeros((batch_size, cfg.lat)),
            jnp.zeros((batch_size, cfg.lat)),
        )

    def update_context(self, carry, state, red_action, active):
        b = carry.observed.shape[0]
        if state.shape != (b, 66) or red_action.shape != (b, 2) or active.shape != (b,):
            raise ValueError("expected state(B,66), red_action(B,2), active(B)")
        pair = jnp.concatenate([self.features(state), red_action], axis=-1)
        if self.config.history and self.config.max_history is not None:
            width, limit = self.config.history, self.config.max_history
            count = jnp.minimum(carry.observed, limit)
            full = count == limit
            shifted = jnp.concatenate([carry.window[:, 1:], jnp.zeros_like(carry.window[:, :1])], axis=1)
            window = jnp.where(full[:, None, None], shifted, carry.window)
            offset = jnp.maximum(jnp.minimum(count, limit - 1), 0)
            window = window.at[jnp.arange(b), offset].set(pair)
            mask = jnp.arange(width)[None] < jnp.minimum(carry.observed + 1, limit)[:, None]
            window = jnp.where(mask[..., None], window, 0.0)
            context, _ = HistoryEncoder(self.config).apply(self.params["encoder"], window, mask)
            completed_sum, completed_variance = carry.completed_sum, carry.completed_variance
        elif self.config.history:
            width = self.config.history
            offset = carry.observed % width
            window = jnp.where(offset[:, None, None] == 0, 0.0, carry.window)
            window = window.at[jnp.arange(b), offset].set(pair)
            mask = jnp.arange(width)[None] <= offset[:, None]
            mu, lv = HistoryEncoder(self.config).apply(self.params["encoder"], window, mask)
            count = (carry.observed // width + 1)[:, None]
            context = (carry.completed_sum + mu) / count
            complete = (offset + 1 == width)[:, None]
            completed_sum = carry.completed_sum + jnp.where(complete, mu, 0.0)
            completed_variance = carry.completed_variance + jnp.where(complete, jnp.exp(lv), 0.0)
        else:
            window, mask = carry.window, carry.mask
            context = carry.context
            completed_sum, completed_variance = carry.completed_sum, carry.completed_variance
        candidate = CausalCarry(window, mask, carry.observed + 1, context, completed_sum, completed_variance)
        return jax.tree.map(lambda new, old: jnp.where(active.reshape((b,) + (1,) * (new.ndim - 1)), new, old), candidate, carry)

    def context(self, state, red_action, lengths):
        state, red_action, lengths = validate_dataset(state, red_action, lengths, allow_empty=True)
        carry = self.initial_context(len(lengths))
        update = jax.jit(self.update_context)
        contexts = [np.asarray(carry.context)]
        for t in range(red_action.shape[1]):
            carry = update(carry, jnp.asarray(state[:, t]), jnp.asarray(red_action[:, t]), jnp.asarray(t < lengths))
            contexts.append(np.asarray(carry.context))
        return np.stack(contexts, axis=1)

    def attach(self, agent, obs_mean, obs_std):
        model = agent.model
        cfg = self.config
        if model.opponent_mode != "factored" or model.encoder_type != "identity":
            raise ValueError("causal opponent needs factored identity world model")
        if (model.context_dim, model.latent_dim, model.action_dim) != (cfg.lat, 66, 2):
            raise ValueError("world/opponent dimension mismatch")
        mean, std = np.asarray(obs_mean, np.float32), np.asarray(obs_std, np.float32)
        if mean.shape != (66,) or std.shape != (66,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
            raise ValueError("invalid world normalization")
        encoded = model.encode(jnp.asarray(mean), model.encoder.params, jax.random.PRNGKey(0))
        scale = model.encode(jnp.asarray(mean + std), model.encoder.params, jax.random.PRNGKey(0))
        if not np.allclose(encoded, 0, atol=1e-5) or not np.allclose(scale, 1, atol=1e-4):
            raise ValueError("world normalization does not match identity encoder")

        def apply(variables, inputs):
            raw = inputs[..., :66] * std + mean
            features = self.features(raw)
            values = MLPHead(out=2, hid=cfg.hid).apply(variables, jnp.concatenate([inputs[..., 66:], features], axis=-1))
            return jnp.tanh(values)

        red = TrainState.create(apply_fn=apply, params=self.params["decoder"]["params"], tx=optax.set_to_zero())
        return agent.replace(model=model.replace(red_model=red), red_loss_scale=0.0)

    def save(self, path):
        payload = {"schema": SCHEMA, "config": asdict(self.config), "params": serialization.to_state_dict(self.params),
                   "state_mean": self.state_mean, "state_std": self.state_std, "prototypes": self.prototypes}
        Path(path).write_bytes(serialization.msgpack_serialize(payload))

    @classmethod
    def load(cls, path):
        raw = serialization.msgpack_restore(Path(path).read_bytes())
        if raw.get("schema") != SCHEMA:
            raise ValueError("unsupported causal opponent artifact schema")
        return cls(CausalOpponentConfig(**raw["config"]), raw["params"], raw["state_mean"], raw["state_std"], raw["prototypes"])


def validate_dataset(state, action, lengths, *, allow_empty=False):
    state, action, lengths = np.asarray(state, np.float32), np.asarray(action, np.float32), np.asarray(lengths)
    if action.ndim != 3 or action.shape[-1] != 2 or state.shape != (action.shape[0], action.shape[1] + 1, 66):
        raise ValueError("expected state(N,T+1,66), action(N,T,2)")
    if lengths.dtype.kind not in "iu" or lengths.shape != (len(state),) or not len(lengths):
        raise ValueError("nonempty integer lengths required")
    if np.any(lengths < (0 if allow_empty else 1)) or np.any(lengths > action.shape[1]):
        raise ValueError("lengths outside trajectory bounds")
    valid = np.arange(action.shape[1])[None, :] < lengths[:, None]
    if not np.isfinite(state[:, :-1][valid]).all() or not np.isfinite(action[valid]).all() or np.any(np.abs(action[valid]) > 1):
        raise ValueError("valid states/actions must be finite; action bounds [-1,1]")
    # Terminal state is unused by opponent action fitting; keep it separate.
    return np.where(np.pad(valid, ((0, 0), (0, 1)))[..., None], state, 0.0), np.where(valid[..., None], action, 0.0), lengths.astype(np.int32)


def validate_indices(indices, n, name="train"):
    idx = np.asarray(indices)
    if idx.ndim != 1 or not len(idx) or idx.dtype.kind not in "iu" or np.any(idx < 0) or np.any(idx >= n) or len(np.unique(idx)) != len(idx):
        raise ValueError(f"{name} indices must be unique in-bounds nonempty integers")
    return idx.astype(np.int32)


def sample_training_indices(train, lengths, batch, episode_key, time_key, sampler):
    """E3 control: uniform episodes then steps, or uniform valid transitions."""
    if sampler == "episode_uniform":
        episodes = train[jax.random.randint(episode_key, (batch,), 0, len(train))]
        return episodes, jax.random.randint(time_key, (batch,), 0, lengths[episodes])
    if sampler != "transition_uniform":
        raise ValueError("unsupported training sample measure")
    cumulative = jnp.cumsum(lengths[train])
    selected = jax.random.randint(episode_key, (batch,), 0, cumulative[-1])
    positions = jnp.searchsorted(cumulative, selected, side="right")
    start = jnp.concatenate([jnp.zeros(1, cumulative.dtype), cumulative[:-1]])
    return train[positions], selected - start[positions]


def fit_causal_opponent(state, red_action, lengths, train_indices, rng, *, config, labels=None, training_state_path=None):
    """Fit using explicit train indices; other episodes never affect updates.

    Default updates sample episodes uniformly then valid timesteps uniformly.
    The explicit transition-uniform control changes only this measure. A fixed
    checkpoint is returned; held-out data never select steps or normalization.
    """
    state, action, lengths = validate_dataset(state, red_action, lengths)
    train = validate_indices(train_indices, len(state))
    cfg = config
    raw = np.asarray(opponent_features(state[:, :-1], cfg.feature_schema))
    valid = np.arange(action.shape[1])[None, :] < lengths[:, None]
    rows = raw[train][valid[train]]
    mean, std = rows.mean(0).astype(np.float32), np.maximum(rows.std(0), 1e-6).astype(np.float32)
    features = jnp.asarray(np.where(valid[..., None], (raw - mean) / std, 0.0))
    actions = jnp.asarray(action)
    encoder, decoder = HistoryEncoder(cfg), MLPHead(out=2, hid=cfg.hid)
    rng, ekey, dkey = jax.random.split(rng, 3)
    params = {"encoder": encoder.init(ekey, jnp.zeros((1, cfg.history, cfg.feature_dim + 2)), jnp.zeros((1, cfg.history), bool)),
              "decoder": decoder.init(dkey, jnp.zeros((1, cfg.lat + cfg.feature_dim)))}
    optimizer = optax.adam(cfg.learning_rate)
    opt_state = optimizer.init(params)
    train_jax, lengths_jax = jnp.asarray(train), jnp.asarray(lengths)

    def loss_fn(p, episodes, times, key, beta):
        h, mask = history_batch(features, actions, episodes, times, cfg.history, max_history=cfg.max_history)
        mu, logvar, window_mu, window_lv, present = pooled_posterior(p["encoder"], cfg, h, mask)
        latent = mu + jnp.exp(logvar / 2) * jax.random.normal(key, mu.shape) if cfg.sample_training else mu
        prediction = jnp.tanh(decoder.apply(p["decoder"], jnp.concatenate([latent, features[episodes, times]], axis=-1)))
        mse = jnp.mean(jnp.sum((prediction - actions[episodes, times])**2, axis=-1))
        per_dim = -0.5 * (1 + window_lv - window_mu**2 - jnp.exp(window_lv))
        per_dim = jnp.sum(jnp.where(present[..., None], per_dim, 0.0), axis=(0, 1)) / jnp.maximum(present.sum(), 1)
        kl = jnp.sum(per_dim) if cfg.variational else jnp.asarray(0.0)
        penalty = jnp.sum(jnp.maximum(per_dim, cfg.free_bits)) if cfg.variational else jnp.asarray(0.0)
        return mse + beta * penalty, (mse, kl, penalty)

    @jax.jit
    def update(p, opt_state, key, beta):
        ek, tk, sk = jax.random.split(key, 3)
        episodes, times = sample_training_indices(train_jax, lengths_jax, cfg.batch, ek, tk, cfg.sampler)
        (loss, aux), grad = jax.value_and_grad(loss_fn, has_aux=True)(p, episodes, times, sk, beta)
        updates, opt_state = optimizer.update(grad, opt_state, p)
        new_params = optax.apply_updates(p, updates)
        finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(v)) for v in jax.tree.leaves((loss, aux, grad, opt_state, new_params))]))
        return new_params, opt_state, loss, aux, finite

    history = []
    for step in range(cfg.steps):
        rng, key = jax.random.split(rng)
        beta = cfg.beta * min(1.0, step / max(1, cfg.steps // 2))
        params, opt_state, loss, (mse, kl, penalty), finite = update(params, opt_state, key, beta)
        if not bool(finite):
            raise FloatingPointError(f"nonfinite causal-predictor update at step {step + 1}")
        if step % max(1, cfg.steps // 100) == 0 or step + 1 == cfg.steps:
            metrics = {"step": step + 1, "loss": float(loss), "vector_action_mse": float(mse), "kl_sum": float(kl), "floored_kl": float(penalty), "beta": float(beta)}
            if not np.isfinite(list(metrics.values())).all():
                raise FloatingPointError("nonfinite causal-predictor training loss")
            history.append(metrics)
    model = CausalOpponent(cfg, params, mean, std, np.zeros((3, cfg.lat), np.float32))
    if labels is not None:
        labels = np.asarray(labels)
        if labels.shape != (len(state),) or set(np.unique(labels[train])) != {0, 1, 2}:
            raise ValueError("training labels must cover three objectives for diagnostic prototypes")
        context = model.context(state[train], action[train], lengths[train])
        last = context[np.arange(len(train)), lengths[train]]
        prototypes = np.stack([last[labels[train] == k].mean(0) for k in range(3)])
        model = CausalOpponent(cfg, params, mean, std, prototypes)
    if training_state_path is not None:
        payload = {"schema": SCHEMA + "_training", "config": asdict(cfg), "step": cfg.steps,
                   "params": serialization.to_state_dict(params), "optimizer": serialization.to_state_dict(opt_state),
                   "rng": np.asarray(jax.random.key_data(rng)), "rng_implementation": str(jax.random.key_impl(rng)),
                   "train_indices": train, "state_mean": mean, "state_std": std}
        Path(training_state_path).write_bytes(serialization.msgpack_serialize(payload))
    return model, history


def evaluate_prediction(model, state, action, lengths, indices, labels, *, sample_seed=0, samples=32):
    """Equal-episode, per-type past-only error and explicitly named interventions.

    Sample spread and empirical coverage are descriptive diagnostics, not a
    claim that the latent posterior is calibrated. Normal observation noise is
    not added; intervals reflect latent samples only.
    """
    state, action, lengths = validate_dataset(state, action, lengths)
    idx = validate_indices(indices, len(state), "evaluation")
    labels = np.asarray(labels)
    if labels.shape != (len(state),) or not set(np.unique(labels[idx])).issubset({0, 1, 2}):
        raise ValueError("valid objective labels required")
    if samples < 2:
        raise ValueError("at least two latent samples required")
    x, a, sizes, y = state[idx], action[idx], lengths[idx], labels[idx]
    contexts = model.context(x, a, sizes)[:, :-1]
    valid = np.arange(a.shape[1])[None, :] < sizes[:, None]
    predict = jax.jit(model.actions)
    outputs = {"mean": np.asarray(predict(x[:, :-1], contexts)), "zero": np.asarray(predict(x[:, :-1], jnp.zeros_like(contexts)))}
    # EVAL-05: different-type donor with exactly t completed pairs, never a
    # stopped donor's shorter history. Availability/exclusions remain explicit.
    donor = np.full(valid.shape, -1, np.int32)
    generator = np.random.default_rng(sample_seed)
    for t in range(a.shape[1]):
        order = generator.permutation(len(x))
        for e in np.flatnonzero(valid[:, t]):
            eligible = order[(y[order] != y[e]) & (sizes[order] >= t)]
            if len(eligible):
                donor[e, t] = eligible[e % len(eligible)]
    available = donor >= 0
    donor_context = contexts[np.maximum(donor, 0), np.arange(a.shape[1])[None]]
    donor_context = np.where(available[..., None], donor_context, 0.0)
    outputs["cross_type_context"] = np.asarray(predict(x[:, :-1], donor_context))
    result = {"schema": SCHEMA, "metric": "sum of two squared action errors; equal episodes within objective", "episodes": len(idx), "per_objective": {}}
    for k in np.unique(y):
        selected = y == k
        metrics = {}
        for name, values in outputs.items():
            included = valid & available if name == "cross_type_context" else valid
            counts = included.sum(axis=1)
            errors = np.sum((values - a)**2, axis=-1)
            per_episode = np.sum(np.where(included, errors, 0.0), axis=1) / np.maximum(counts, 1)
            retained = selected & (counts > 0)
            metrics[name + "_mse"] = float(per_episode[retained].mean()) if retained.any() else None
            metrics[name + "_transition_weighted_mse"] = float(errors[selected][included[selected]].mean()) if included[selected].any() else None
            metrics[name + "_episode_values"] = [float(value) if count else None for value, count in zip(per_episode[selected], counts[selected])]
            metrics[name + "_included_transitions"] = int(counts[selected].sum())
            metrics[name + "_excluded_transitions"] = int((valid[selected] & ~included[selected]).sum())
        result["per_objective"][str(int(k))] = metrics
    # Sample only valid examples in bounded chunks, preserving all transitions.
    ep, time = np.where(valid)
    features = model.features(x[:, :-1])
    errors, coverages, spreads, kls = [], [], [], []
    key = jax.random.PRNGKey(sample_seed)
    for start in range(0, len(ep), 256):
        e, t = jnp.asarray(ep[start:start+256]), jnp.asarray(time[start:start+256])
        h, mask = history_batch(features, jnp.asarray(a), e, t, model.config.history, max_history=model.config.max_history)
        mu, logvar = model.posterior(h, mask)
        key, subkey = jax.random.split(key)
        z = mu[None] + jnp.exp(logvar[None] / 2) * jax.random.normal(subkey, (samples,) + mu.shape) if model.config.variational else jnp.broadcast_to(mu, (samples,) + mu.shape)
        predictions = jax.vmap(lambda c: model.actions(x[e, t], c))(z)
        pred = np.asarray(predictions)
        target = a[np.asarray(e), np.asarray(t)]
        lo, hi = np.quantile(pred, [0.05, 0.95], axis=0)
        errors.extend(np.mean(np.sum((pred-target[None])**2, axis=-1), axis=0).tolist())
        coverages.extend(np.mean((target >= lo) & (target <= hi), axis=-1).tolist())
        spreads.extend(np.mean(pred.var(axis=0), axis=-1).tolist())
        kls.extend(np.asarray(gaussian_kl(mu, logvar) if model.config.variational else jnp.zeros(len(mu))).tolist())
    diagnostics = {"sample_expected_vector_mse": np.asarray(errors), "latent_only_90pct_marginal_coverage": np.asarray(coverages), "latent_action_variance": np.asarray(spreads), "posterior_kl_sum": np.asarray(kls)}
    for k in np.unique(y):
        for name, values in diagnostics.items():
            episode_values = np.bincount(ep, weights=values, minlength=len(x)) / sizes
            result["per_objective"][str(int(k))][name] = (
                None if name == "posterior_kl_sum" and not model.config.variational
                else float(episode_values[y == k].mean())
            )
    traces = {"indices": idx, "lengths": sizes, "contexts": contexts,
              "prediction_mean": outputs["mean"], "prediction_zero": outputs["zero"],
              "prediction_cross_type_context": outputs["cross_type_context"], "target": a, "labels": y,
              "donor_available": available, "donor_episode_index": np.where(available, idx[np.maximum(donor, 0)], -1),
              "donor_label": np.where(available, y[np.maximum(donor, 0)], -1),
              "donor_time": np.where(available, np.arange(a.shape[1])[None], -1),
              "donor_completed_pairs": np.where(available, np.arange(a.shape[1])[None], -1),
              "valid_episode": ep, "valid_time": time, **diagnostics}
    return result, traces
