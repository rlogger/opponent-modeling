"""Opponent-mode input contract (Ellen's Eqs 1–3).

Read this module first when extending Equation 3. Physics, reward, and
continuation never receive context ``c`` in ``factored`` mode; only the red
policy head and blue Q / policy prior do.

| mode           | dynamics / reward / continue | Q            | policy prior |
|----------------|------------------------------|--------------|--------------|
| ``implicit``   | ``(x, u)``                   | ``(x, u)``   | ``pi(x)``    |
| ``conditioned``| ``(x, u, c)``                | ``(x, u, c)``| ``pi(x, c)`` |
| ``factored``   | ``(x, u, v)``, ``v=red(x,c)``| ``(x, u, c)``| ``pi(x, c)`` |
"""
from __future__ import annotations

from typing import Any, Optional

import jax
import jax.numpy as jnp

OPPONENT_MODES = ("implicit", "conditioned", "factored")

__all__ = [
    "OPPONENT_MODES",
    "input_widths",
    "policy_inputs",
    "transition_extra_dim",
    "transition_inputs",
    "value_extra_dim",
    "value_inputs",
]


def transition_extra_dim(opponent_mode: str, context_dim: int, action_dim: int) -> int:
    """Width appended to ``u`` for dynamics / reward / continue inputs."""
    if opponent_mode == "conditioned":
        return context_dim
    if opponent_mode == "factored":
        return action_dim  # red action v
    return 0


def value_extra_dim(opponent_mode: str, context_dim: int) -> int:
    return 0 if opponent_mode == "implicit" else context_dim


def input_widths(
    opponent_mode: str, latent_dim: int, action_dim: int, context_dim: int
) -> tuple[int, int, int, int, int]:
    """Return ``(transition_extra, value_extra, transition_in, value_in, policy_in)``."""
    t_extra = transition_extra_dim(opponent_mode, context_dim, action_dim)
    v_extra = value_extra_dim(opponent_mode, context_dim)
    transition_in = latent_dim + action_dim + t_extra
    value_in = latent_dim + action_dim + v_extra
    policy_in = latent_dim + (0 if opponent_mode == "implicit" else context_dim)
    return t_extra, v_extra, transition_in, value_in, policy_in


def transition_inputs(
    opponent_mode: str,
    u: jax.Array,
    c: Optional[jax.Array] = None,
    v: Optional[jax.Array] = None,
) -> jax.Array:
    """Assemble dynamics / reward / continue action channel: ``u`` | ``[u,c]`` | ``[u,v]``."""
    if opponent_mode == "conditioned":
        return jnp.concatenate([u, c], axis=-1)
    if opponent_mode == "factored":
        return jnp.concatenate([u, v], axis=-1)
    return u


def value_inputs(
    opponent_mode: str, u: jax.Array, c: Optional[jax.Array] = None
) -> jax.Array:
    if opponent_mode == "implicit":
        return u
    return jnp.concatenate([u, c], axis=-1)


def policy_inputs(
    opponent_mode: str, x: jax.Array, c: Optional[jax.Array] = None
) -> jax.Array:
    if opponent_mode == "implicit":
        return x
    return jnp.concatenate([x, c], axis=-1)


def factored_transition_ignores_context(
    opponent_mode: str, u: Any, c_a: Any, c_b: Any, v: Any
) -> bool:
    """True iff swapping ``c`` does not change ``transition_inputs`` (Eq 3 contract)."""
    if opponent_mode != "factored":
        return False
    a = transition_inputs(opponent_mode, u, c_a, v)
    b = transition_inputs(opponent_mode, u, c_b, v)
    return bool(jnp.array_equal(a, b))
