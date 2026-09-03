"""Continuous-action boundary adapter for the MPE simple-tag family.

JaxMARL's continuous MPE action is a redundant vector ``a in [0, 1]^5`` that
the physics decodes as the force ``(a[2] - a[1], a[4] - a[3])`` (before the
per-agent acceleration gain). Learning and planning code in this repository
uses the identifiable two-dimensional action ``(a_x, a_y) in [-1, 1]^2``.
The conversion happens only at the environment boundary:

    [a_x, a_y] -> [0, max(-a_x, 0), max(a_x, 0), max(-a_y, 0), max(a_y, 0)]

so that the decoded force equals the original two-dimensional action exactly.
All functions are pure ``jax.numpy`` and safe under ``jit`` and ``vmap``.
"""
from __future__ import annotations

from typing import Any, Mapping

import jax.numpy as jnp

CONTINUOUS_ACTION_DIM = 2
MPE_ACTION_DIM = 5
ACTION_LOW = -1.0
ACTION_HIGH = 1.0

__all__ = [
    "ACTION_HIGH",
    "ACTION_LOW",
    "CONTINUOUS_ACTION_DIM",
    "MPE_ACTION_DIM",
    "clip_action",
    "from_mpe_action",
    "joint_action_dict",
    "to_mpe_action",
]


def clip_action(action: Any) -> jnp.ndarray:
    """Clip a ``(..., 2)`` action to the declared ``[-1, 1]`` bounds."""
    return jnp.clip(jnp.asarray(action, dtype=jnp.float32), ACTION_LOW, ACTION_HIGH)


def to_mpe_action(action: Any) -> jnp.ndarray:
    """Map ``(..., 2)`` actions in ``[-1, 1]`` to JaxMARL ``(..., 5)`` vectors."""
    a = jnp.asarray(action, dtype=jnp.float32)
    if a.shape[-1] != CONTINUOUS_ACTION_DIM:
        raise ValueError(
            f"continuous action must end with {CONTINUOUS_ACTION_DIM} dims, "
            f"got shape {a.shape}"
        )
    ax, ay = a[..., 0], a[..., 1]
    zero = jnp.zeros_like(ax)
    return jnp.stack(
        [
            zero,
            jnp.maximum(-ax, 0.0),
            jnp.maximum(ax, 0.0),
            jnp.maximum(-ay, 0.0),
            jnp.maximum(ay, 0.0),
        ],
        axis=-1,
    )


def from_mpe_action(mpe_action: Any) -> jnp.ndarray:
    """Decode a JaxMARL ``(..., 5)`` vector to the force ``(a[2]-a[1], a[4]-a[3])``.

    This mirrors ``SimpleMPE._decode_continuous_action`` before the
    acceleration gain, so ``from_mpe_action(to_mpe_action(a)) == a``.
    """
    m = jnp.asarray(mpe_action, dtype=jnp.float32)
    if m.shape[-1] != MPE_ACTION_DIM:
        raise ValueError(
            f"MPE action must end with {MPE_ACTION_DIM} dims, got shape {m.shape}"
        )
    return jnp.stack([m[..., 2] - m[..., 1], m[..., 4] - m[..., 3]], axis=-1)


def joint_action_dict(
    env: Any,
    blue_action: Any,
    red_actions: Mapping[str, Any] | Any,
) -> dict[str, jnp.ndarray]:
    """Build the env action dict from two-dimensional blue and red actions.

    ``blue_action`` has shape ``(..., 2)``. ``red_actions`` is either a mapping
    from adversary name to ``(..., 2)`` or an array ``(..., P, 2)`` ordered like
    ``env.adversaries``.
    """
    actions: dict[str, jnp.ndarray] = {}
    if isinstance(red_actions, Mapping):
        for name in env.adversaries:
            actions[name] = to_mpe_action(red_actions[name])
    else:
        red = jnp.asarray(red_actions, dtype=jnp.float32)
        if red.shape[-2] != len(env.adversaries):
            raise ValueError("red_actions must have one row per adversary")
        for i, name in enumerate(env.adversaries):
            actions[name] = to_mpe_action(red[..., i, :])
    for name in env.good_agents:
        actions[name] = to_mpe_action(blue_action)
    return actions
