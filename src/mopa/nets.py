"""Shared network modules for loading MAPPO actors and greedy rollouts.

``ActorLogits`` is param-compatible with ``scripts.train_mappo.Actor`` but
returns raw logits (no Distrax) so offline eval / BC can ``argmax`` without
the trainer stack.

``ContinuousActor`` is the single definition of the two-dimensional
tanh-squashed diagonal-Gaussian specialist actor used by the continuous MAPPO
trainer and by every downstream rollout (the deterministic policy is
``tanh(mean)``).
"""
from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal

from mopa.continuous import clamp_log_std


class ActorLogits(nn.Module):
    """Two-layer MLP policy head returning raw action logits."""

    action_dim: int
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, obs):  # noqa: ANN001
        x = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(obs)
        x = nn.relu(x)
        x = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        return nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(x)


class ContinuousActor(nn.Module):
    """Two-layer MLP returning ``(mean, log_std)`` of a pre-tanh Gaussian.

    ``log_std`` is a state-independent learned vector (initialized to zero,
    i.e. unit standard deviation) clamped to ``[LOG_STD_MIN, LOG_STD_MAX]`` and
    broadcast to ``mean``'s shape. The executed action is ``tanh(u)`` with
    ``u ~ N(mean, exp(log_std))``; the deterministic action is ``tanh(mean)``.
    """

    action_dim: int = 2
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, obs):  # noqa: ANN001
        x = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(obs)
        x = nn.relu(x)
        x = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        mean = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(x)
        log_std = self.param(
            "log_std", nn.initializers.zeros, (self.action_dim,)
        )
        log_std = clamp_log_std(jnp.broadcast_to(log_std, mean.shape))
        return mean, log_std

    def deterministic_action(self, obs):  # noqa: ANN001
        mean, _ = self(obs)
        return jnp.tanh(mean)
