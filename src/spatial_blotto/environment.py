"""Three-zone spatial Blotto; see docs/spatial-blotto.md for the game rules.

This module owns its dynamics and rewards. JaxMARL supplies only the environment
interface, automatic episode reset, and spaces; MPE physics is not inherited.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from functools import partial
from numbers import Integral, Real

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from jaxmarl.environments.multi_agent_env import MultiAgentEnv, State
from jaxmarl.environments.spaces import Box


@struct.dataclass
class BlottoState(State):
    """Immutable JAX pytree. Agent rows are red first, then blue, by ID.

    Positions and realized velocities are float32; ``step`` and cumulative raw
    ownership scores are int32. ``done`` is a scalar boolean. Only reset and
    environment transitions are guaranteed to produce valid states.
    """

    p_pos: jax.Array
    p_vel: jax.Array
    team_scores: jax.Array  # cumulative raw ownership points, [red, blue]


class SpatialBlotto(MultiAgentEnv):
    """Two equal teams, three zones, simultaneous continuous velocity control.

    The default reward is the team's number of strictly owned zones, replicated
    for every teammate. ``reward_mode='zero_sum'`` instead returns own minus
    opponent ownership count. That is a different game objective when ties occur.

    Configuration is static after construction. ``step_env`` returns the actual
    next state, while inherited ``step`` auto-resets at the finite task horizon.
    Agent order is red IDs followed by blue IDs. No collisions, acceleration,
    momentum, hidden goals, or stochastic transitions are introduced.

    Float32 numerical contract: dt, max_speed, and their product lie between
    float32.tiny and 2**63, inclusive. This leaves headroom for reciprocals and
    squared velocities. Radius is at least float32.tiny; its float32 value must
    lie below the tangent radius by more than eight float32 epsilons. The
    horizon is at most 2**24 so normalized elapsed time distinguishes all steps.
    """

    def __init__(
        self,
        team_size: int = 3,
        max_steps: int = 200,
        dt: float = 0.1,
        max_speed: float = 1.0,
        zone_radius: float = 0.22,
        reward_mode: str = "ownership",
    ):
        for name, value in (("team_size", team_size), ("max_steps", max_steps)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        # Every step must remain distinguishable in the float32 time channel.
        # This also keeps the maximum raw score (3 * max_steps) within int32.
        if max_steps > 2**24:
            raise ValueError("max_steps must be <= 16777216 for float32 elapsed time")
        smallest = float(np.finfo(np.float32).tiny)
        largest = float(2**63)
        for name, value in (
            ("dt", dt),
            ("max_speed", max_speed),
            ("zone_radius", zone_radius),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or not smallest <= value <= largest
            ):
                raise ValueError(
                    f"{name} must be finite and between float32.tiny and 2**63"
                )
        displacement = float(dt) * float(max_speed)
        if not smallest <= displacement <= largest:
            raise ValueError("dt * max_speed must be between float32.tiny and 2**63")
        if reward_mode not in ("ownership", "zero_sum"):
            raise ValueError("reward_mode must be 'ownership' or 'zero_sum'")
        # Equilateral triangle of circumradius .75. Strictly disjoint circles
        # ensure a unit can contribute to at most one zone, including boundaries.
        # The mathematical tangent threshold alone is insufficient: nearby
        # host float64 radii and float32 centers can round to touching circles.
        radius_limit = math.sqrt(3) * 0.375 - 8 * np.finfo(np.float32).eps
        if float(np.float32(zone_radius)) >= radius_limit:
            raise ValueError("zone circles must be disjoint with a float32 margin")

        super().__init__(num_agents=2 * int(team_size))
        self.team_size = int(team_size)
        self.max_steps = int(max_steps)
        self.num_zones = 3
        self.dt = float(dt)
        self.max_speed = float(max_speed)
        self.zone_radius = float(zone_radius)
        self.reward_mode = reward_mode
        self.arena = 1.5
        self.red_agents = tuple(f"red_{i}" for i in range(team_size))
        self.blue_agents = tuple(f"blue_{i}" for i in range(team_size))
        self.agents = self.red_agents + self.blue_agents
        self.zone_centers = jnp.array(
            [
                [0.0, 0.75],
                [-math.sqrt(3) * 0.375, -0.375],
                [math.sqrt(3) * 0.375, -0.375],
            ],
            dtype=jnp.float32,
        )
        # Per-agent order: other teammates by ID, then opponents by ID.
        self._others = tuple(
            tuple(
                j
                for j in range(self.num_agents)
                if j != i and j // team_size == i // team_size
            )
            + tuple(
                j for j in range(self.num_agents) if j // team_size != i // team_size
            )
            for i in range(self.num_agents)
        )
        self.obs_size = 4 * self.num_agents + 10 + team_size
        self.state_size = 4 * self.num_agents + 1
        self.observation_spaces = {
            a: Box(-1.0, 1.0, (self.obs_size,)) for a in self.agents
        }
        self.action_spaces = {a: Box(-1.0, 1.0, (2,)) for a in self.agents}
        self._state_space = Box(-1.0, 1.0, (self.state_size,))

    def state_space(self):
        """Bounds and shape of the normalized vector returned by ``get_state``."""
        return self._state_space

    def validate_actions(self, actions):
        """Host-side check for policy/controller output; returns ``None``.

        Require exactly the agent keys and finite real vectors of shape (2,)
        representable in float32. Values outside [-1, 1] are valid and clipped
        by dynamics. Call at a Python boundary, outside jit/vmap/scan; compiled
        transitions require finite inputs and do not perform a host callback.
        """
        self._check_action_keys(actions)
        for agent in self.agents:
            value = np.asarray(actions[agent])
            if value.shape != (2,):
                raise ValueError(f"action for {agent} must have shape (2,)")
            if value.dtype.kind not in "fiu":
                raise ValueError(f"action for {agent} must be a real numeric vector")
            with np.errstate(over="ignore", invalid="ignore"):
                value = value.astype(np.float32)
            if not np.isfinite(value).all():
                raise ValueError(f"action for {agent} must be finite in float32")

    def _check_action_keys(self, actions):
        if not isinstance(actions, Mapping) or set(actions) != set(self.agents):
            raise ValueError(
                "actions must contain exactly the environment's agent keys"
            )

    def _check_state_shapes(self, state):
        expected = {
            "p_pos": (self.num_agents, 2),
            "p_vel": (self.num_agents, 2),
            "team_scores": (2,),
            "step": (),
            "done": (),
        }
        for field, shape in expected.items():
            if np.shape(getattr(state, field)) != shape:
                raise ValueError(f"state.{field} must have shape {shape}")

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key):
        # Match teammates across a mirror-symmetric arena to avoid a built-in
        # starting advantage. Seeds vary the starting positions, not hidden goals.
        jitter = jax.random.uniform(
            key,
            (self.team_size, 2),
            minval=-0.08,
            maxval=0.08,
            dtype=jnp.float32,
        )
        red = jnp.array([-1.05, 0.0], dtype=jnp.float32) + jitter
        positions = jnp.concatenate(
            (red, red * jnp.array([-1.0, 1.0], dtype=jnp.float32))
        )
        state = BlottoState(
            done=jnp.array(False),
            step=jnp.array(0, dtype=jnp.int32),
            p_pos=positions,
            p_vel=jnp.zeros_like(positions),
            team_scores=jnp.zeros(2, dtype=jnp.int32),
        )
        return self.get_obs(state), state

    def zone_counts(self, state):
        """Integer counts [team, zone]; a unit on a zone boundary counts."""
        delta = state.p_pos[:, None, :] - self.zone_centers[None, :, :]
        # hypot scales before squaring, preserving tiny distances. Squaring
        # directly can underflow both sides to zero and count outside units.
        inside = jnp.hypot(delta[..., 0], delta[..., 1]) <= self.zone_radius
        return inside.reshape(2, self.team_size, self.num_zones).sum(
            axis=1, dtype=jnp.int32
        )

    def zone_owners(self, state):
        """+1 red, -1 blue, 0 neutral, including empty and contested ties."""
        counts = self.zone_counts(state)
        return jnp.sign(counts[0] - counts[1])

    def get_obs(self, state):
        """Bounded full-state observations without goals or future actions.

        Layout: self position(2), self velocity(2), other relative positions
        (2*(N-1)), other velocities(2*(N-1)), relative zone centers(6), zone
        ownership from own perspective(3), elapsed time(1), teammate ID(n).
        """
        self._check_state_shapes(state)
        owners = self.zone_owners(state).astype(jnp.float32)
        # Subtracting nearby float32 positions can produce an apparent velocity
        # just above the speed cap. Preserve measured velocity in state, but keep
        # the declared normalized observation bounds exact.
        velocity = jnp.clip(state.p_vel / self.max_speed, -1.0, 1.0)
        obs = {}
        for i, agent in enumerate(self.agents):
            other = jnp.array(self._others[i], dtype=jnp.int32)
            perspective = 1.0 if i < self.team_size else -1.0
            obs[agent] = jnp.concatenate(
                (
                    state.p_pos[i] / self.arena,
                    velocity[i],
                    ((state.p_pos[other] - state.p_pos[i]) / (2 * self.arena)).ravel(),
                    velocity[other].ravel(),
                    ((self.zone_centers - state.p_pos[i]) / (2 * self.arena)).ravel(),
                    owners * perspective,
                    jnp.asarray([state.step / self.max_steps], dtype=jnp.float32),
                    jax.nn.one_hot(
                        i % self.team_size, self.team_size, dtype=jnp.float32
                    ),
                )
            )
        return obs

    def get_state(self, state):
        """Physical Markov state for a centralized critic/model, shape (4*N+1,).

        Cumulative score is a logging accumulator, not an input to future rewards.
        Fixed zone geometry and configuration belong to the environment.
        """
        self._check_state_shapes(state)
        return jnp.concatenate(
            (
                (state.p_pos / self.arena).ravel(),
                jnp.clip(state.p_vel / self.max_speed, -1.0, 1.0).ravel(),
                jnp.asarray([state.step / self.max_steps], dtype=jnp.float32),
            )
        )

    @partial(jax.jit, static_argnums=(0,))
    def step_env(self, key, state, actions):
        """Simultaneous move, then score. Finished states are absorbing.

        ``key`` is accepted for JaxMARL compatibility; movement is deterministic.
        Actions must be finite real float32-representable vectors of shape (2,).
        Keys, shapes, and numeric dtypes are checked at tracing time; use
        ``validate_actions`` outside JAX transforms to check finite values.
        Components are clipped to [-1,1], then the vector is projected onto the
        unit disk to cap speed. Previous velocity does not affect movement.
        """
        del key
        self._check_state_shapes(state)
        self._check_action_keys(actions)
        values = []
        for agent in self.agents:
            value = jnp.asarray(actions[agent])
            if value.shape != (2,):
                raise ValueError(f"action for {agent} must have shape (2,)")
            if value.dtype.kind not in "fiu":
                raise ValueError(f"action for {agent} must be a real numeric vector")
            values.append(value.astype(jnp.float32))
        action = jnp.stack(values)
        action = jnp.clip(action, -1.0, 1.0)
        direction = action / jnp.maximum(
            jnp.linalg.norm(action, axis=-1, keepdims=True), 1.0
        )
        positions = jnp.clip(
            state.p_pos + self.dt * self.max_speed * direction,
            -self.arena,
            self.arena,
        )
        candidate = state.replace(
            p_pos=positions,
            p_vel=(positions - state.p_pos) / self.dt,
            step=state.step + 1,
            done=state.step + 1 >= self.max_steps,
        )
        active = ~state.done
        owners = self.zone_owners(candidate)
        scores = jnp.array(
            [jnp.sum(owners == 1), jnp.sum(owners == -1)], dtype=jnp.int32
        )
        scores = jnp.where(active, scores, 0)
        candidate = candidate.replace(team_scores=state.team_scores + scores)
        next_state = jax.tree.map(
            lambda new, old: jnp.where(active, new, old), candidate, state
        )
        if self.reward_mode == "zero_sum":
            margin = scores[0] - scores[1]
            team_reward = jnp.array([margin, -margin], dtype=jnp.float32)
        else:
            team_reward = scores.astype(jnp.float32)
        obs = self.get_obs(next_state)
        rewards = {
            a: team_reward[i // self.team_size] for i, a in enumerate(self.agents)
        }
        dones = {a: next_state.done for a in self.agents}
        dones["__all__"] = next_state.done
        info = {
            "zone_counts": self.zone_counts(next_state),
            "zone_owners": self.zone_owners(next_state),
            "ownership_scores": scores,
            "team_rewards": team_reward,
            "team_scores": next_state.team_scores,
            # This finite-horizon game includes time in the observation. The
            # horizon ends the task, not an artificial rollout truncation.
            "terminated": next_state.done,
            "truncated": jnp.array(False),
            # These always describe the actual next state, even when inherited
            # step() returns a reset observation/state at the terminal transition.
            "terminal_observation": obs,
            "terminal_state": self.get_state(next_state),
        }
        return obs, next_state, rewards, dones, info

    @property
    def agent_classes(self):
        return {"red": self.red_agents, "blue": self.blue_agents}

    def get_avail_actions(self, state):
        raise NotImplementedError(
            "continuous Box actions do not use a discrete action mask"
        )
