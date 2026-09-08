"""Short-window conditional-action VAE for opponent trajectories.

This is the reusable core of the ``0s`` encoder: a masked GRU encodes each
non-overlapping ``[state, action]`` window, while a stepwise MLP predicts the
action at each valid step from the window latent and that step's state.  Window
posterior means are averaged to obtain one episode representation.

Objective labels are deliberately absent from every API in this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import orthogonal


@dataclass(frozen=True)
class ActionDecoderConfig:
    """Legacy ``0s`` architecture and optimization defaults."""

    lat: int = 8
    hid: int = 64
    window: int = 8
    steps: int = 1500
    batch: int = 128
    n_actions: int = 5
    learning_rate: float = 1e-3
    beta_max: float = 1.0
    free_bits: float = 0.2
    action_type: str = "discrete"
    action_dim: int = 2


@dataclass(frozen=True)
class WindowIndex:
    """Non-overlapping, padded window layout for episode-major arrays."""

    episode: np.ndarray
    start: np.ndarray
    mask: np.ndarray
    lengths: np.ndarray
    width: int

    def __len__(self) -> int:
        return len(self.episode)


@dataclass(frozen=True)
class FrozenActionEncoder:
    """Frozen posterior-mean encoder and its train-fold state statistics."""

    params: Any
    state_mean: np.ndarray
    state_std: np.ndarray
    config: ActionDecoderConfig


@dataclass(frozen=True)
class ActionDecoderEncoding:
    """Window-prefix, final-window, and pooled episode representations."""

    windows: WindowIndex
    prefix_latents: np.ndarray
    window_latents: np.ndarray
    episode_latents: np.ndarray


@dataclass(frozen=True)
class ActionDecoderFit:
    """Trained ``0s`` model plus its frozen representations."""

    encoder: FrozenActionEncoder
    decoder_params: Any
    encoding: ActionDecoderEncoding
    history: tuple[dict[str, float | int], ...]
    decoder_action_accuracy: float | None
    decoder_action_mse: float | None = None

    @property
    def episode_latents(self) -> np.ndarray:
        return self.encoding.episode_latents

    @property
    def window_latents(self) -> np.ndarray:
        return self.encoding.window_latents

    @property
    def prefix_latents(self) -> np.ndarray:
        return self.encoding.prefix_latents


def episode_mask(lengths: np.ndarray, horizon: int) -> np.ndarray:
    """Return ``(N, T)`` validity mask for positive, bounded lengths."""

    lengths = np.asarray(lengths, dtype=np.int32)
    if lengths.ndim != 1:
        raise ValueError("lengths must be one-dimensional")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if np.any(lengths < 1) or np.any(lengths > horizon):
        raise ValueError(f"lengths must lie in [1, {horizon}]")
    return np.arange(horizon, dtype=np.int32)[None, :] < lengths[:, None]


def make_windows(lengths: np.ndarray, width: int = 8) -> WindowIndex:
    """Cut every episode into consecutive windows, padding only the final one."""

    lengths = np.asarray(lengths, dtype=np.int32)
    if lengths.ndim != 1:
        raise ValueError("lengths must be one-dimensional")
    if len(lengths) == 0:
        raise ValueError("at least one episode is required")
    if np.any(lengths < 1):
        raise ValueError("lengths must be positive")
    if width < 2:
        raise ValueError("window width must be at least 2")

    counts = np.maximum(1, np.ceil(lengths / width).astype(np.int32))
    episode = np.repeat(np.arange(len(lengths), dtype=np.int32), counts)
    starts = np.concatenate(
        [np.arange(count, dtype=np.int32) * width for count in counts]
    )
    offsets = np.arange(width, dtype=np.int32)[None, :]
    mask = starts[:, None] + offsets < lengths[episode, None]
    return WindowIndex(
        episode=episode,
        start=starts.astype(np.int32),
        mask=mask,
        lengths=mask.sum(axis=1).astype(np.int32),
        width=int(width),
    )


def gather_windows(values: np.ndarray, windows: WindowIndex) -> np.ndarray:
    """Gather ``(N, T, ...)`` values into zero-padded ``(M, W, ...)`` windows."""

    values = np.asarray(values)
    if values.ndim < 2:
        raise ValueError("values must have episode and time dimensions")
    if values.shape[0] <= int(windows.episode.max(initial=-1)):
        raise ValueError("window episode index is out of bounds")

    time = windows.start[:, None] + np.arange(windows.width)[None, :]
    time = np.minimum(time, values.shape[1] - 1)
    out = values[windows.episode[:, None], time]
    keep = windows.mask.reshape(windows.mask.shape + (1,) * (out.ndim - 2))
    return out * keep.astype(out.dtype)


def pool_window_latents(
    window_latents: np.ndarray,
    episode: np.ndarray,
    n_episodes: int,
) -> np.ndarray:
    """Mean-pool one latent per window into one latent per source episode."""

    rows = np.asarray(window_latents, dtype=np.float32)
    episode = np.asarray(episode, dtype=np.int32)
    if rows.ndim != 2:
        raise ValueError("window_latents must have shape (windows, latent_dim)")
    if episode.shape != (len(rows),):
        raise ValueError("episode indices must align with window_latents")
    if n_episodes < 1:
        raise ValueError("n_episodes must be positive")
    if np.any(episode < 0) or np.any(episode >= n_episodes):
        raise ValueError("episode index is out of bounds")

    pooled = np.zeros((n_episodes, rows.shape[-1]), dtype=np.float32)
    count = np.bincount(episode, minlength=n_episodes).astype(np.float32)[:, None]
    np.add.at(pooled, episode, rows)
    return pooled / np.maximum(count, 1.0)


def pool_episode_prefix_latents(
    prefix_latents: np.ndarray,
    windows: WindowIndex,
    observed_lengths: np.ndarray,
    n_episodes: int,
) -> np.ndarray:
    """Pool the window latents available at each episode's observed prefix.

    Fully observed earlier windows contribute their final posterior mean.  The
    currently active window contributes the mean at its last observed step.
    Episodes with an observed length of zero receive an all-zero vector.
    """

    z = np.asarray(prefix_latents, dtype=np.float32)
    observed = np.asarray(observed_lengths, dtype=np.int32)
    if z.ndim != 3 or z.shape[:2] != windows.mask.shape:
        raise ValueError("prefix_latents must have shape (windows, width, latent_dim)")
    if observed.shape != (n_episodes,):
        raise ValueError("observed_lengths must have one value per episode")
    if np.any(observed < 0):
        raise ValueError("observed_lengths cannot be negative")

    available = np.clip(
        observed[windows.episode] - windows.start,
        0,
        windows.lengths,
    ).astype(np.int32)
    include = available > 0
    selected = z[
        np.arange(len(windows)),
        np.maximum(available - 1, 0),
    ]
    if not np.any(include):
        return np.zeros((n_episodes, z.shape[-1]), dtype=np.float32)
    return pool_window_latents(
        selected[include], windows.episode[include], n_episodes
    )


class MLPTrunk(nn.Module):
    hid: int = 64

    @nn.compact
    def __call__(self, x):
        for _ in range(2):
            x = nn.Dense(self.hid, kernel_init=orthogonal(np.sqrt(2)))(x)
            x = nn.relu(x)
        return x


class MLPHead(nn.Module):
    out: int
    hid: int = 64

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.out, kernel_init=orthogonal(1.0))(
            MLPTrunk(hid=self.hid)(x)
        )


class GRUCore(nn.Module):
    hid: int = 64

    @nn.compact
    def __call__(self, x, mask):
        x = nn.relu(
            nn.LayerNorm()(
                nn.Dense(self.hid, kernel_init=orthogonal(np.sqrt(2)))(x)
            )
        )
        shape = (self.hid, self.hid)
        init = orthogonal(np.sqrt(2))
        wz = self.param("wz", init, shape)
        uz = self.param("uz", init, shape)
        bz = self.param("bz", nn.initializers.zeros, (self.hid,))
        wr = self.param("wr", init, shape)
        ur = self.param("ur", init, shape)
        br = self.param("br", nn.initializers.zeros, (self.hid,))
        wh = self.param("wh", init, shape)
        uh = self.param("uh", init, shape)
        bh = self.param("bh", nn.initializers.zeros, (self.hid,))

        n = x.shape[0]
        def step(hidden, inputs):
            value, valid = inputs
            update = jax.nn.sigmoid(value @ wz + hidden @ uz + bz)
            reset = jax.nn.sigmoid(value @ wr + hidden @ ur + br)
            proposal = jnp.tanh(value @ wh + (reset * hidden) @ uh + bh)
            changed = (1.0 - update) * proposal + update * hidden
            output = jnp.where(valid[:, None], changed, hidden)
            return output, output

        hidden = jnp.zeros((n, self.hid), dtype=x.dtype)
        _, states = jax.lax.scan(
            step,
            hidden,
            (jnp.swapaxes(x, 0, 1), jnp.swapaxes(mask.astype(bool), 0, 1)),
        )
        return nn.LayerNorm()(jnp.swapaxes(states, 0, 1))


class SeqGaussian(nn.Module):
    lat: int = 8
    hid: int = 64

    @nn.compact
    def __call__(self, x, mask):
        hidden = GRUCore(hid=self.hid)(x, mask)
        return nn.Dense(self.lat)(hidden), nn.Dense(self.lat)(hidden)


def _last_valid(values, mask):
    index = jnp.maximum(jnp.sum(mask.astype(jnp.int32), axis=-1) - 1, 0)
    return values[jnp.arange(values.shape[0]), index]


def _masked_mse(predicted, target, mask):
    error = jnp.sum((predicted - target) ** 2, axis=-1)
    weight = mask.astype(error.dtype)
    return jnp.sum(error * weight) / jnp.maximum(jnp.sum(weight), 1.0)


def _free_bits_kl(mu, logvar, free_bits: float):
    per_dim = jnp.mean(
        -0.5 * (1.0 + logvar - mu**2 - jnp.exp(logvar)), axis=0
    )
    return jnp.sum(jnp.maximum(per_dim, free_bits)), jnp.sum(per_dim)


def _beta(step: int, steps: int, beta_max: float) -> float:
    return beta_max * min(1.0, step / max(1, steps // 2))


def _validate_episode_inputs(state, action, lengths, config: ActionDecoderConfig):
    state = np.asarray(state, dtype=np.float32)
    if config.action_type not in {"discrete", "continuous"}:
        raise ValueError("action_type must be 'discrete' or 'continuous'")
    if config.action_dim < 1:
        raise ValueError("action_dim must be positive")
    continuous = config.action_type == "continuous"
    action = np.asarray(action, dtype=np.float32 if continuous else np.int32)
    if state.ndim != 3:
        raise ValueError("state must have shape (episodes, time, features)")
    expected_shape = state.shape[:2] + ((config.action_dim,) if continuous else ())
    if action.shape != expected_shape:
        suffix = ", action_dim" if continuous else ""
        raise ValueError(f"action must have shape (episodes, time{suffix})")
    mask = episode_mask(lengths, state.shape[1])
    if mask.shape[0] != state.shape[0]:
        raise ValueError("lengths must have one value per episode")
    if continuous:
        valid_action = action[mask]
        if not np.isfinite(valid_action).all() or np.any(np.abs(valid_action) > 1.0):
            raise ValueError("valid continuous actions must be finite and lie in [-1, 1]")
        if not np.isfinite(state[mask]).all():
            raise ValueError("valid states must be finite")
        # Padding is absent data, including when an exporter uses NaN sentinels.
        action = np.where(mask[..., None], action, 0.0)
        state = np.where(mask[..., None], state, 0.0)
    return state, action, np.asarray(lengths, dtype=np.int32), mask


def _standardize_state(state, mask, mean, std):
    normalized = (state - mean) / std
    return normalized.astype(np.float32) * mask[..., None].astype(np.float32)


def _one_hot_actions(action, mask, n_actions: int):
    valid_actions = action[mask]
    if np.any(valid_actions < 0) or np.any(valid_actions >= n_actions):
        raise ValueError(f"valid actions must lie in [0, {n_actions - 1}]")
    safe = np.where(mask, action, 0)
    one_hot = np.eye(n_actions, dtype=np.float32)[safe]
    return one_hot * mask[..., None].astype(np.float32)


def _window_inputs(
    state,
    action,
    lengths,
    *,
    state_mean,
    state_std,
    config: ActionDecoderConfig,
):
    state, action, lengths, mask = _validate_episode_inputs(state, action, lengths, config)
    if state.shape[-1] != len(state_mean):
        raise ValueError("state feature dimension does not match the frozen encoder")
    standardized = _standardize_state(state, mask, state_mean, state_std)
    action_features = (
        action if config.action_type == "continuous"
        else _one_hot_actions(action, mask, config.n_actions)
    )
    windows = make_windows(lengths, config.window)
    state_windows = gather_windows(standardized, windows)
    action_windows = gather_windows(action_features, windows)
    return windows, state_windows, action_windows


def decode_action_decoder(
    decoder_params: Any,
    state: jax.Array,
    latent: jax.Array,
    config: ActionDecoderConfig,
) -> jax.Array:
    """Decode standardized state and context with matching leading dimensions.

    This pure JAX application can be called inside a compiled planner.  The
    continuous extension returns bounded actions; the original discrete model
    returns its unchanged, unsquashed action scores.  Callers normalize state
    using the frozen encoder's training statistics before invoking this function.
    """

    out = config.action_dim if config.action_type == "continuous" else config.n_actions
    value = MLPHead(out=out, hid=config.hid).apply(
        decoder_params, jnp.concatenate([latent, state], axis=-1)
    )
    return jnp.tanh(value) if config.action_type == "continuous" else value


def encode_action_decoder_vae(
    encoder: FrozenActionEncoder,
    state: np.ndarray,
    action: np.ndarray,
    lengths: np.ndarray,
) -> ActionDecoderEncoding:
    """Apply a frozen encoder and return every prefix plus pooled episode means."""

    windows, state_windows, action_windows = _window_inputs(
        state,
        action,
        lengths,
        state_mean=encoder.state_mean,
        state_std=encoder.state_std,
        config=encoder.config,
    )
    sequence = np.concatenate([state_windows, action_windows], axis=-1)
    model = SeqGaussian(lat=encoder.config.lat, hid=encoder.config.hid)
    prefix_mu, _ = model.apply(
        encoder.params,
        jnp.asarray(sequence),
        jnp.asarray(windows.mask),
    )
    prefix_latents = np.asarray(prefix_mu, dtype=np.float32)
    window_latents = prefix_latents[
        np.arange(len(windows)), windows.lengths - 1
    ]
    episode_latents = pool_window_latents(
        window_latents, windows.episode, len(lengths)
    )
    return ActionDecoderEncoding(
        windows=windows,
        prefix_latents=prefix_latents,
        window_latents=window_latents,
        episode_latents=episode_latents,
    )


def fit_action_decoder_vae(
    state: np.ndarray,
    action: np.ndarray,
    lengths: np.ndarray,
    train_episode_idx: np.ndarray,
    rng: jax.Array,
    *,
    config: ActionDecoderConfig | None = None,
) -> ActionDecoderFit:
    """Fit ``0s`` without labels and encode every supplied episode.

    State normalization and optimizer updates use windows originating only from
    ``train_episode_idx``. Each update samples ``config.batch`` training windows
    uniformly with replacement, matching the original implementation.
    """

    cfg = ActionDecoderConfig() if config is None else config
    if cfg.lat < 1 or cfg.hid < 1 or cfg.steps < 0 or cfg.batch < 1:
        raise ValueError("lat, hid, and batch must be positive; steps cannot be negative")

    state, action, lengths, mask = _validate_episode_inputs(state, action, lengths, cfg)
    train_idx = np.asarray(train_episode_idx, dtype=np.int32)
    if train_idx.ndim != 1 or len(train_idx) == 0:
        raise ValueError("train_episode_idx must be a non-empty vector")
    if np.any(train_idx < 0) or np.any(train_idx >= len(state)):
        raise ValueError("training episode index is out of bounds")

    train_rows = state[train_idx][mask[train_idx]]
    state_mean = train_rows.mean(axis=0).astype(np.float32)
    state_std = (train_rows.std(axis=0) + 1e-6).astype(np.float32)
    windows, state_windows, action_windows = _window_inputs(
        state,
        action,
        lengths,
        state_mean=state_mean,
        state_std=state_std,
        config=cfg,
    )
    train_windows = np.where(np.isin(windows.episode, train_idx))[0].astype(
        np.int32
    )
    if len(train_windows) == 0:
        raise ValueError("training episodes produced no windows")

    sequence = np.concatenate([state_windows, action_windows], axis=-1)
    encoder_model = SeqGaussian(lat=cfg.lat, hid=cfg.hid)
    decoder_model = MLPHead(
        out=cfg.action_dim if cfg.action_type == "continuous" else cfg.n_actions,
        hid=cfg.hid,
    )
    rng, encoder_key, decoder_key = jax.random.split(rng, 3)
    params = {
        "e": encoder_model.init(
            encoder_key,
            jnp.asarray(sequence[:1]),
            jnp.asarray(windows.mask[:1]),
        ),
        "d": decoder_model.init(
            decoder_key,
            jnp.zeros((1, cfg.lat + state.shape[-1]), dtype=jnp.float32),
        ),
    }
    optimizer = optax.adam(cfg.learning_rate)
    optimizer_state = optimizer.init(params)

    sequence_jax = jnp.asarray(sequence)
    state_jax = jnp.asarray(state_windows)
    target_jax = jnp.asarray(action_windows)
    mask_jax = jnp.asarray(windows.mask)
    train_jax = jnp.asarray(train_windows)

    def decode(model_params, latent, step_state):
        batch_size, time_steps, _ = step_state.shape
        repeated = jnp.broadcast_to(
            latent[:, None, :], (batch_size, time_steps, latent.shape[-1])
        )
        return decode_action_decoder(
            model_params["d"],
            step_state.reshape(batch_size * time_steps, -1),
            repeated.reshape(batch_size * time_steps, -1),
            cfg,
        ).reshape(batch_size, time_steps, -1)

    def loss_fn(model_params, sa_batch, state_batch, target_batch, batch_mask, key, beta):
        mu_prefix, logvar_prefix = encoder_model.apply(
            model_params["e"], sa_batch, batch_mask
        )
        mu = _last_valid(mu_prefix, batch_mask)
        logvar = _last_valid(logvar_prefix, batch_mask)
        latent = mu + jnp.exp(0.5 * logvar) * jax.random.normal(key, mu.shape)
        reconstruction = _masked_mse(
            decode(model_params, latent, state_batch), target_batch, batch_mask
        )
        penalty, raw_kl = _free_bits_kl(mu, logvar, cfg.free_bits)
        return reconstruction + beta * penalty, (reconstruction, raw_kl)

    @jax.jit
    def update(model_params, opt_state, index, key, beta):
        (loss, (reconstruction, raw_kl)), gradient = jax.value_and_grad(
            loss_fn, has_aux=True
        )(
            model_params,
            sequence_jax[index],
            state_jax[index],
            target_jax[index],
            mask_jax[index],
            key,
            beta,
        )
        updates, opt_state = optimizer.update(gradient, opt_state)
        model_params = optax.apply_updates(model_params, updates)
        return model_params, opt_state, loss, reconstruction, raw_kl

    history: list[dict[str, float | int]] = []
    log_every = max(1, cfg.steps // 60)
    for step in range(cfg.steps):
        rng, batch_key, sample_key = jax.random.split(rng, 3)
        sampled = train_jax[
            jax.random.randint(batch_key, (cfg.batch,), 0, len(train_windows))
        ]
        beta = _beta(step, cfg.steps, cfg.beta_max)
        params, optimizer_state, loss, reconstruction, raw_kl = update(
            params, optimizer_state, sampled, sample_key, beta
        )
        if step % log_every == 0 or step == cfg.steps - 1:
            history.append(
                {
                    "step": step,
                    "loss": float(loss),
                    "action_mse": float(reconstruction),
                    "kl": float(raw_kl),
                    "beta": float(beta),
                }
            )

    frozen = FrozenActionEncoder(
        params=params["e"],
        state_mean=state_mean,
        state_std=state_std,
        config=cfg,
    )
    encoding = encode_action_decoder_vae(frozen, state, action, lengths)
    predicted = np.asarray(
        decode(
            params,
            jnp.asarray(encoding.window_latents),
            state_jax,
        )
    )
    action_accuracy = None
    action_mse = None
    if cfg.action_type == "continuous":
        action_mse = float(np.sum((predicted - action_windows) ** 2, axis=-1)[windows.mask].mean())
    else:
        correct = (predicted.argmax(axis=-1) == np.asarray(gather_windows(action, windows)))[
            windows.mask
        ]
        action_accuracy = float(correct.mean())
    return ActionDecoderFit(
        encoder=frozen,
        decoder_params=params["d"],
        encoding=encoding,
        history=tuple(history),
        decoder_action_accuracy=action_accuracy,
        decoder_action_mse=action_mse,
    )
