"""Tanh-squashed diagonal Gaussian utilities for two-dimensional actions.

Shared by the continuous MAPPO specialists, continuous behaviour cloning, and
evaluation code. Conventions:

- ``mean`` and ``log_std`` parameterize the pre-squash Gaussian over ``u``;
- the executed action is ``a = tanh(u) in (-1, 1)^2``;
- ``log_prob`` is the density of ``a`` obtained from ``u`` by the change of
  variables ``log p(a) = log N(u; mean, std) - sum_i log(1 - tanh(u_i)^2)``,
  with the numerically stable identity
  ``log(1 - tanh(u)^2) = 2 (log 2 - u - softplus(-2u))``.

Every function is pure ``jax.numpy`` and sums over the trailing action axis, so
joint log-probabilities of multi-dimensional actions are correct by
construction (PPO must never use per-dimension probabilities).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

LOG_STD_MIN = -5.0
LOG_STD_MAX = 1.0
_LOG_2PI = jnp.log(2.0 * jnp.pi)

__all__ = [
    "LOG_STD_MAX",
    "LOG_STD_MIN",
    "atanh_clipped",
    "clamp_log_std",
    "gaussian_entropy",
    "tanh_gaussian_log_prob",
    "tanh_gaussian_sample",
    "tanh_log_det_jacobian",
]


def clamp_log_std(log_std: jax.Array) -> jax.Array:
    """Clip raw log-std values to ``[LOG_STD_MIN, LOG_STD_MAX]``."""
    return jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)


def tanh_log_det_jacobian(u: jax.Array) -> jax.Array:
    """``sum_i log(1 - tanh(u_i)^2)`` computed stably; shape ``u.shape[:-1]``."""
    return jnp.sum(2.0 * (jnp.log(2.0) - u - jax.nn.softplus(-2.0 * u)), axis=-1)


def tanh_gaussian_log_prob(
    mean: jax.Array, log_std: jax.Array, u: jax.Array
) -> jax.Array:
    """Log-density of ``a = tanh(u)`` for a diagonal Gaussian over ``u``.

    Sums over the trailing action dimension and applies the tanh
    change-of-variables correction. ``u`` is the *pre-squash* sample; callers
    should store it during rollouts instead of inverting ``tanh`` on ``a``.
    """
    std = jnp.exp(log_std)
    normalized = (u - mean) / std
    gaussian = -0.5 * jnp.sum(normalized**2 + 2.0 * log_std + _LOG_2PI, axis=-1)
    return gaussian - tanh_log_det_jacobian(u)


def tanh_gaussian_sample(
    key: jax.Array, mean: jax.Array, log_std: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Reparameterized sample; returns ``(u, tanh(u))``."""
    eps = jax.random.normal(key, mean.shape, dtype=mean.dtype)
    u = mean + jnp.exp(log_std) * eps
    return u, jnp.tanh(u)


def gaussian_entropy(log_std: jax.Array) -> jax.Array:
    """Entropy of the *pre-squash* diagonal Gaussian, summed over dimensions.

    The squashed distribution has no closed-form entropy; PPO uses this base
    entropy as its exploration bonus, which is the standard proxy.
    """
    return jnp.sum(log_std + 0.5 * (1.0 + _LOG_2PI), axis=-1)


def atanh_clipped(a: jax.Array, eps: float = 1e-6) -> jax.Array:
    """Inverse tanh on actions clipped away from the boundary (analysis only)."""
    return jnp.arctanh(jnp.clip(a, -1.0 + eps, 1.0 - eps))
