"""Small, integer resource allocations and the game's two payoff conventions."""

from numbers import Integral

import jax.numpy as jnp


def _validate_budget(budget):
    if isinstance(budget, bool) or not isinstance(budget, Integral) or budget < 1:
        raise ValueError("budget must be a positive integer")


def validate_allocation(allocation, budget):
    """Return a static length-three tuple using exactly ``budget`` units."""
    _validate_budget(budget)
    try:
        allocation = tuple(allocation)
    except TypeError as exc:
        raise ValueError("allocation must contain three nonnegative integers") from exc
    if len(allocation) != 3 or any(
        isinstance(value, bool) or not isinstance(value, Integral) or value < 0
        for value in allocation
    ):
        raise ValueError("allocation must contain three nonnegative integers")
    if sum(allocation) != budget:
        raise ValueError("allocation must use exactly the team's budget")
    return tuple(int(value) for value in allocation)


def enumerate_allocations(budget):
    """All nonnegative three-zone allocations, shape ``((n+1)*(n+2)/2, 3)``.

    Construction is a host-side operation for a static integer budget. The
    returned array can be closed over by a JIT-compiled controller.
    """
    _validate_budget(budget)
    return jnp.asarray(
        [(a, b, budget - a - b)
         for a in range(budget + 1) for b in range(budget - a + 1)],
        dtype=jnp.int32,
    )


def allocation_scores(allocations, opponent_counts, reward_mode="ownership"):
    """Score hypothetical final allocations against fixed opponent counts.

    Inputs broadcast over leading dimensions, with three zones on the last
    axis. Opponents currently outside every zone spend no unit in these counts.
    This static payoff does not predict transit rewards or opponent movement.
    """
    if reward_mode not in ("ownership", "zero_sum"):
        raise ValueError("reward_mode must be 'ownership' or 'zero_sum'")
    allocations, opponent_counts = jnp.asarray(allocations), jnp.asarray(opponent_counts)
    for name, values in (("allocations", allocations), ("opponent_counts", opponent_counts)):
        if values.ndim < 1 or values.shape[-1] != 3:
            raise ValueError(f"{name} must have exactly three zones on its final axis")
    wins = jnp.sum(allocations > opponent_counts, axis=-1)
    if reward_mode == "zero_sum":
        return wins - jnp.sum(allocations < opponent_counts, axis=-1)
    return wins
