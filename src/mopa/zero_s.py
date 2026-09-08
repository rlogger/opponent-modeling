"""Continuous ``0s`` opponent adapter for Equation 3 only.

The original short-window action-decoder VAE lives in ``action_decoder``. This
adapter uses its *own frozen decoder*, not a second opponent network:
``v = decoder(state, z); x_next = dynamics(x, u, v)``. An identity world-state
encoder makes imagined positions/velocities available to that decoder and
makes diagnostic trajectories directly interpretable. This is not a learned
SimNorm-state TD-MPC2 result or evidence of online strategy learning.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
from flax.training.train_state import TrainState

from mopa.action_decoder import (
    ActionDecoderConfig,
    ActionDecoderFit,
    FrozenActionEncoder,
    SeqGaussian,
    decode_action_decoder,
    encode_action_decoder_vae,
    pool_episode_prefix_latents,
)


def zero_s_features(state: Any) -> jax.Array:
    """Causal 8D features: blue/red positions, then blue/red current velocities.

    Input follows ``continuous_data.MARKOV_STATE_FIELDS`` for 1v1 SimpleTag:
    66 coordinates, red before blue. Unlike upstream's next-step displacement,
    these features are all available before the action being predicted.
    """
    state = jnp.asarray(state, jnp.float32)
    if state.shape[-1] != 66:
        raise ValueError("0s bridge requires the 66D 1v1 Markov-state schema")
    return state[..., jnp.array([2, 3, 0, 1, 6, 7, 4, 5])]


def strategy_prototypes(latents: np.ndarray, labels: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    """Three train-only, episode-equal means; never average latent coordinates."""
    z, y, idx = np.asarray(latents), np.asarray(labels), np.asarray(train_idx)
    if z.ndim != 2 or y.shape != (len(z),) or not np.isfinite(z).all():
        raise ValueError("finite episode latents and aligned objective labels required")
    if idx.ndim != 1 or not len(idx) or idx.dtype.kind not in "iu":
        raise ValueError("train_idx must be a nonempty integer vector")
    if np.any(idx < 0) or np.any(idx >= len(z)) or len(np.unique(idx)) != len(idx):
        raise ValueError("train_idx must contain unique, in-bounds episode indices")
    if set(np.unique(y[idx])) != {0, 1, 2}:
        raise ValueError("training episodes must include capture, risk, and curious")
    return np.stack([z[idx[y[idx] == k]].mean(0) for k in range(3)]).astype(np.float32)


class ZeroSContext(NamedTuple):
    """Bounded streaming history: one window and completed-window latent sum."""

    window: jax.Array
    observed: jax.Array
    completed_sum: jax.Array
    context: jax.Array


@dataclass(frozen=True)
class ZeroSOpponent:
    encoder: FrozenActionEncoder
    decoder_params: Any
    prototypes: np.ndarray

    def __post_init__(self) -> None:
        cfg = self.encoder.config
        if cfg.action_type != "continuous" or cfg.action_dim != 2:
            raise ValueError("the world-model adapter requires a continuous 2D 0s model")
        if np.asarray(self.encoder.state_mean).shape != (8,) or np.asarray(self.encoder.state_std).shape != (8,):
            raise ValueError("0s adapter requires eight state features")
        if not np.isfinite(self.encoder.state_mean).all() or not np.isfinite(self.encoder.state_std).all() or np.any(self.encoder.state_std <= 0):
            raise ValueError("state normalization must be finite with positive scales")
        if np.asarray(self.prototypes).shape != (3, cfg.lat) or not np.isfinite(self.prototypes).all():
            raise ValueError("one finite latent prototype is required per objective")

    @classmethod
    def from_fit(cls, fit: ActionDecoderFit, labels: np.ndarray, train_idx: np.ndarray) -> "ZeroSOpponent":
        return cls(fit.encoder, fit.decoder_params, strategy_prototypes(fit.episode_latents, labels, train_idx))

    def actions(self, state: Any, context: Any) -> jax.Array:
        """Apply the original decoder to raw Markov state and an 8D context."""
        features = (zero_s_features(state) - self.encoder.state_mean) / self.encoder.state_std
        return decode_action_decoder(self.decoder_params, features, jnp.asarray(context), self.encoder.config)

    def context(self, state: np.ndarray, red_action: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        """At decision t encode only completed pairs [state_s, action_s], s < t.

        Batched offline computation of recurrent prefixes is equivalent to
        recomputing from the observed history. At t=0 context is zero; after
        termination it stays at the final available prefix. No objective label
        or future action is an input. Full-episode prototypes are a separate,
        explicitly known-type diagnostic, not this causal inference path.
        """
        n, t, _ = np.asarray(red_action).shape
        if np.asarray(state).shape != (n, t + 1, 66):
            raise ValueError("state must have shape (episodes, actions_time + 1, 66)")
        encoding = encode_action_decoder_vae(
            self.encoder, np.asarray(zero_s_features(state[:, :-1])), red_action, lengths,
        )
        return np.stack([
            pool_episode_prefix_latents(encoding.prefix_latents, encoding.windows, np.minimum(lengths, step), n)
            for step in range(t + 1)
        ], axis=1)

    def initial_context(self, batch_size: int) -> ZeroSContext:
        """Start fresh episodes with zero context before any action is observed."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        cfg = self.encoder.config
        return ZeroSContext(
            jnp.zeros((batch_size, cfg.window, 8 + cfg.action_dim), jnp.float32),
            jnp.zeros(batch_size, jnp.int32),
            jnp.zeros((batch_size, cfg.lat), jnp.float32),
            jnp.zeros((batch_size, cfg.lat), jnp.float32),
        )

    def update_context(self, carry: ZeroSContext, state: Any, red_action: Any, active: Any) -> ZeroSContext:
        """Consume a completed (pre-step state, actual red action) pair.

        Call after stepping the real environment, including its terminating
        transition. ``active`` means the episode was alive *before* that step;
        inactive rows are left unchanged. The exact frozen encoder is reapplied
        only to the current short window, so memory and work never grow with
        episode length. Fully completed windows and the current partial window
        receive equal weight, matching :meth:`context` at every decision time.
        This method supports JIT and scan; no labels or future states are used.
        """
        state, red_action = jnp.asarray(state, jnp.float32), jnp.asarray(red_action, jnp.float32)
        active = jnp.asarray(active, bool)
        batch = carry.observed.shape[0]
        if state.shape != (batch, 66) or red_action.shape != (batch, 2) or active.shape != (batch,):
            raise ValueError("streaming 0s expects state (B,66), red_action (B,2), active (B,)")
        cfg = self.encoder.config
        offset = carry.observed % cfg.window
        rows = jnp.arange(batch)
        # Erase the preceding window only when the next observed pair arrives.
        window = jnp.where(offset[:, None, None] == 0, 0.0, carry.window)
        features = (zero_s_features(state) - self.encoder.state_mean) / self.encoder.state_std
        window = window.at[rows, offset].set(jnp.concatenate([features, red_action], axis=-1))
        mask = jnp.arange(cfg.window)[None, :] <= offset[:, None]
        posterior, _ = SeqGaussian(lat=cfg.lat, hid=cfg.hid).apply(self.encoder.params, window, mask)
        current = posterior[rows, offset]
        candidate = ZeroSContext(
            window,
            carry.observed + 1,
            carry.completed_sum + jnp.where((offset + 1 == cfg.window)[:, None], current, 0.0),
            (carry.completed_sum + current) / (carry.observed // cfg.window + 1)[:, None],
        )
        return jax.tree.map(
            lambda new, old: jnp.where(active.reshape((batch,) + (1,) * (new.ndim - 1)), new, old),
            candidate, carry,
        )

    def attach(self, agent: Any, obs_mean: np.ndarray, obs_std: np.ndarray) -> Any:
        """Install the frozen 0s decoder in TD-MPC's existing Equation 3 path.

        ``set_to_zero`` also prevents optimizer weight decay from moving the
        decoder. The training loop still measures red reconstruction error,
        but neither that auxiliary loss nor TD gradients retrain this model.
        Build this adapter before restoring a serialized agent: apply_fn and
        optimizer transformations are static and not stored in Flax msgpack.
        """
        model = agent.model
        if model.opponent_mode != "factored" or model.encoder_type != "identity":
            raise ValueError("0s requires factored mode and the identity state encoder")
        if model.context_dim != self.encoder.config.lat or model.latent_dim != 66 or model.action_dim != 2:
            raise ValueError("0s and world-model dimensions disagree")
        mean, std = np.asarray(obs_mean, np.float32), np.asarray(obs_std, np.float32)
        if mean.shape != (66,) or std.shape != (66,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
            raise ValueError("world-state normalization must be finite, positive, and 66D")
        # Check that these are the identity encoder's actual statistics. Passing
        # another normalization silently corrupts every imagined opponent action.
        encoded_mean = model.encode(jnp.asarray(mean), model.encoder.params, jax.random.PRNGKey(0))
        encoded_scale = model.encode(jnp.asarray(mean + std), model.encoder.params, jax.random.PRNGKey(0))
        if not np.allclose(encoded_mean, 0, atol=1e-5) or not np.allclose(encoded_scale, 1, atol=1e-4):
            raise ValueError("normalization does not match the attached identity encoder")
        cfg = self.encoder.config
        feature_mean, feature_std = jnp.asarray(self.encoder.state_mean), jnp.asarray(self.encoder.state_std)

        def apply(variables, inputs):
            raw_state = inputs[..., :66] * std + mean
            features = (zero_s_features(raw_state) - feature_mean) / feature_std
            return decode_action_decoder(variables, features, inputs[..., 66:], cfg)

        red = TrainState.create(apply_fn=apply, params=self.decoder_params["params"], tx=optax.set_to_zero())
        return agent.replace(model=model.replace(red_model=red), red_loss_scale=0.0)

    def save(self, path: str | Path) -> None:
        """Save both halves of 0s, train statistics, configuration and prototypes."""
        payload = {
            "schema": 1,
            "feature_schema": "blue_red_position_velocity_v1",
            "config": asdict(self.encoder.config),
            "encoder_params": serialization.to_state_dict(self.encoder.params),
            "decoder_params": serialization.to_state_dict(self.decoder_params),
            "state_mean": np.asarray(self.encoder.state_mean),
            "state_std": np.asarray(self.encoder.state_std),
            "prototypes": np.asarray(self.prototypes),
        }
        Path(path).write_bytes(serialization.msgpack_serialize(payload))

    @classmethod
    def load(cls, path: str | Path) -> "ZeroSOpponent":
        payload = serialization.msgpack_restore(Path(path).read_bytes())
        if payload["schema"] != 1 or payload["feature_schema"] != "blue_red_position_velocity_v1":
            raise ValueError("unsupported 0s artifact schema")
        encoder = FrozenActionEncoder(payload["encoder_params"], payload["state_mean"], payload["state_std"], ActionDecoderConfig(**payload["config"]))
        return cls(encoder, payload["decoder_params"], payload["prototypes"])


@jax.jit
def rollout_zero_s(agent, initial_state, blue_actions, context, obs_mean, obs_std):
    """Fixed-context, fixed-blue-action *learned* autoregressive diagnostic.

    Inputs: (B,66), (H,B,2), (B,C). Outputs: (H+1,B,66), (H,B,2).
    No simulator states, recorded red actions, teacher forcing, or prototype
    updates are used inside the rollout. This intentionally returns a fixed
    diagnostic horizon; points past predicted capture are extrapolation, not
    valid episodic return estimates. The normal MPPI path uses continuation.
    """
    model = agent.model
    x0 = model.encode(initial_state, model.encoder.params, jax.random.PRNGKey(0))

    def step(x, u):
        v = model.red_action(x, context, model.red_model.params)
        new_x = model.next(x, model.transition_inputs(u, context, v), model.dynamics_model.params)
        return new_x, (new_x, v)

    _, (xs, actions) = jax.lax.scan(step, x0, blue_actions)
    states = jnp.concatenate([x0[None], xs], axis=0) * obs_std + obs_mean
    return states, actions
