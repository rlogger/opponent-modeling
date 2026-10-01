"""Deterministic mathematical baselines for the spatial three-zone game.

These controllers solve tiny allocation/assignment problems, with no learned
model. In particular, ``reactive`` is a static-response heuristic: it treats
the currently observed opponent counts as fixed. It is neither an equilibrium
solver nor a best response to an opponent's future policy or cumulative return.
"""

from dataclasses import dataclass
from itertools import product
from numbers import Integral

import jax
import jax.numpy as jnp

from .allocations import allocation_scores, validate_allocation
from .environment import BlottoState, SpatialBlotto

MAX_EXACT_TEAM_SIZE = 6
POLICIES = ("fixed", "balanced", "cyclic", "reactive")


def _mirror(allocation):
    return allocation[0], allocation[2], allocation[1]


@dataclass(frozen=True, eq=False)
class TeamController:
    """A pure JAX callable: ``controller(state) -> actions[team_size, 2]``.

    Construct with :func:`make_controller`. ``target_zones(state)`` supplies
    integer agent-to-zone assignments for diagnostics. Both methods support
    ``jax.jit`` and ``jax.vmap`` and can run inside ``jax.lax.scan``.

    Only current positions, the public step counter (cyclic only), and the
    terminal flag enter decisions. Access to ``BlottoState`` is a convenience:
    it adds no opponent-goal, future-action, or private policy information to
    the environment's full-position observations. Velocities and accumulated
    team scores are unused.
    """

    env: SpatialBlotto
    team_index: int
    policy: str
    allocation: tuple[int, int, int]
    period: int
    _assignments: jax.Array
    _counts: jax.Array

    def target_zones(self, state: BlottoState):
        """Maximize the static objective, then minimize total center distance.

        Assignment ties use a deterministic zone order: top/left/right for
        red and top/right/left for blue. This mirrored convention makes exact
        horizontal mirror ties fair between teams, while deliberately picking
        one of several equally good assignments rather than randomizing.
        """
        n = self.env.team_size
        own = state.p_pos[self.team_index * n:(self.team_index + 1) * n]
        targets = self.env.zone_centers[self._assignments]
        delta = targets - own[None, :, :]
        # hypot rescales before squaring, preserving tiny representable distances.
        travel = jnp.hypot(delta[..., 0], delta[..., 1]).sum(axis=-1)
        if self.policy == "reactive":
            opponent_counts = self.env.zone_counts(state)[1 - self.team_index]
            scores = allocation_scores(
                self._counts, opponent_counts, self.env.reward_mode
            )
            eligible = scores == jnp.max(scores)
        else:
            desired = jnp.asarray(self.allocation, dtype=jnp.int32)
            if self.policy == "cyclic":
                # Opposite rotation directions preserve horizontal mirroring.
                shift = (state.step // self.period) % 3
                shift = shift if self.team_index == 0 else -shift
                desired = jnp.roll(desired, shift)
            eligible = jnp.all(self._counts == desired, axis=-1)
        index = jnp.argmin(jnp.where(eligible, travel, jnp.inf))
        return self._assignments[index]

    def __call__(self, state: BlottoState):
        """Bounded straight-line motion toward centers, stopping without overshoot.

        With no obstacles or collisions, straight lines are shortest paths to
        the selected center points. Entry into a scoring circle occurs before
        reaching its center; this controller intentionally continues to center.
        """
        n = self.env.team_size
        own = state.p_pos[self.team_index * n:(self.team_index + 1) * n]
        delta = self.env.zone_centers[self.target_zones(state)] - own
        distance = jnp.hypot(delta[:, 0], delta[:, 1])[:, None]
        actions = delta / jnp.maximum(distance, self.env.dt * self.env.max_speed)
        return jnp.where(state.done, jnp.zeros_like(actions), actions)


def make_controller(env, team, policy="balanced", *, allocation=None, period=40):
    """Build a deterministic baseline, validating all static configuration.

    ``team`` is ``'red'`` or ``'blue'``. Available policies:

    * ``fixed``: require three nonnegative integer counts summing to team size.
    * ``balanced``: counts differ by at most one. Remainder units go to top,
      then the team's starting side: red 2v2 is (1,1,0), blue is (1,0,1).
    * ``cyclic``: rotate counts every ``period`` steps. The default red counts
      are (n-1,1,0), or (1,0,0) for n=1; blue defaults mirror those counts and
      rotates in the opposite direction. Explicit counts use global zone order.
      The period must fit the environment's positive int32 step counter.
    * ``reactive``: exactly maximize eventual allocation payoff against current
      opponent zone counts, according to ``env.reward_mode``. This optimizes a
      static snapshot, with travel distance as the secondary objective.

    Each policy enumerates all ``3**n`` agent-to-zone assignments to minimize
    travel exactly. The explicit limit is 1 <= n <= 6 (at most 729 candidates),
    independent of the environment's larger-team support. Configuration stays
    static after construction; use a new controller for a different environment.
    """
    if team not in ("red", "blue"):
        raise ValueError("team must be 'red' or 'blue'")
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}")
    if env.team_size > MAX_EXACT_TEAM_SIZE:
        raise ValueError(
            f"exact controllers support team_size <= {MAX_EXACT_TEAM_SIZE}; "
            "assignment enumeration grows as 3**team_size"
        )
    if (isinstance(period, bool) or not isinstance(period, Integral)
            or not 1 <= period <= 2**31 - 1):
        raise ValueError("period must be an integer between 1 and 2147483647 (int32)")
    n, team_index = env.team_size, int(team == "blue")
    if policy == "fixed":
        allocation = validate_allocation(allocation, n)
    elif policy == "cyclic":
        if allocation is None:
            allocation = (n - 1, 1, 0) if n >= 2 else (1, 0, 0)
            if team_index:
                allocation = _mirror(allocation)
        allocation = validate_allocation(allocation, n)
    else:
        if allocation is not None:
            raise ValueError("allocation is only used by fixed and cyclic policies")
        counts = [n // 3] * 3
        for zone in range(n % 3):
            counts[zone] += 1
        allocation = _mirror(counts) if team_index else tuple(counts)
    assignments = tuple(product(range(3), repeat=n))
    if team_index:
        assignments = tuple(tuple((0, 2, 1)[zone] for zone in row) for row in assignments)
    counts = tuple(tuple(row.count(zone) for zone in range(3)) for row in assignments)
    return TeamController(
        env=env,
        team_index=team_index,
        policy=policy,
        allocation=allocation,
        period=int(period),
        _assignments=jnp.asarray(assignments, dtype=jnp.int32),
        _counts=jnp.asarray(counts, dtype=jnp.int32),
    )
