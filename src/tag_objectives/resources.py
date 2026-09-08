"""SimpleTagMPE variant with collectible resources.

Extends JaxMARL simple-tag with:

1. **Fixed obstacles.** Landmark positions are fixed at construction time and
   reapplied on every reset.
2. **Collectible resources.** At reset, ``num_resources`` positions spawn
   according to the placement pattern. The prey sees resource positions and
   collection status; predators do not. The prey earns ``collect_reward`` for
   each resource within ``collect_radius``.
3. **Placement modes** that induce distinguishable prey strategies:
   - ``"circle"``: evenly spaced on a circle of radius ``circle_radius``.
   - ``"corners"``: corners of a square with half-width ``corner_offset``.
   - ``"random"``: each episode picks circle or corners.
"""
from __future__ import annotations

from functools import partial
from typing import List, Optional

import chex
import jax
import jax.numpy as jnp
from flax import struct
from jaxmarl.environments.mpe.simple_tag import SimpleTagMPE
from jaxmarl.environments.spaces import Box


@struct.dataclass
class ResourceState:
    p_pos: chex.Array
    p_vel: chex.Array
    c: chex.Array
    done: chex.Array
    step: int
    goal: int = None
    resource_pos: chex.Array = None
    collected: chex.Array = None


class SimpleTagResourcesMPE(SimpleTagMPE):
    """SimpleTagMPE with fixed obstacles and collectible resources."""

    def __init__(
        self,
        num_resources: int = 4,
        placement: str = "random",
        collect_radius: float = 0.15,
        collect_reward: float = 5.0,
        circle_radius: float = 0.6,
        corner_offset: float = 0.8,
        obstacle_positions: Optional[List[List[float]]] = None,
        pred_max_speed: Optional[float] = None,
        pred_accel: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if pred_max_speed is not None:
            self.max_speed = self.max_speed.at[: self.num_adversaries].set(
                float(pred_max_speed)
            )
        if pred_accel is not None:
            self.accel = self.accel.at[: self.num_adversaries].set(float(pred_accel))

        assert placement in ("circle", "corners", "random"), placement
        self.num_resources = num_resources
        self._placement = placement
        self.collect_radius = float(collect_radius)
        self.collect_reward = float(collect_reward)

        angles = 2 * jnp.pi * jnp.arange(num_resources) / num_resources
        self._circle_positions = jnp.stack(
            [circle_radius * jnp.cos(angles), circle_radius * jnp.sin(angles)],
            axis=-1,
        )

        if num_resources == 4:
            self._corner_positions = jnp.array(
                [
                    [-corner_offset, -corner_offset],
                    [-corner_offset, corner_offset],
                    [corner_offset, -corner_offset],
                    [corner_offset, corner_offset],
                ]
            )
        else:
            angles_c = (
                jnp.pi / 4 + 2 * jnp.pi * jnp.arange(num_resources) / num_resources
            )
            self._corner_positions = jnp.stack(
                [
                    corner_offset * jnp.cos(angles_c),
                    corner_offset * jnp.sin(angles_c),
                ],
                axis=-1,
            )

        if obstacle_positions is not None:
            positions = jnp.asarray(obstacle_positions, dtype=jnp.float32)
            assert positions.shape == (self.num_landmarks, 2), (
                f"obstacle_positions must be ({self.num_landmarks}, 2); "
                f"got {positions.shape}"
            )
            self.fixed_obstacle_positions = positions
        else:
            self.fixed_obstacle_positions = None

        prey_obs_dim = 14 + num_resources * 3
        for a in self.good_agents:
            self.observation_spaces[a] = Box(-jnp.inf, jnp.inf, (prey_obs_dim,))

    def _place_resources(self, key: chex.PRNGKey) -> chex.Array:
        if self._placement == "circle":
            return self._circle_positions
        if self._placement == "corners":
            return self._corner_positions
        use_circle = jax.random.bernoulli(key)
        return jnp.where(use_circle, self._circle_positions, self._corner_positions)

    @partial(jax.jit, static_argnums=[0])
    def reset(self, key: chex.PRNGKey):
        key_a, key_l, key_place = jax.random.split(key, 3)

        agent_pos = jax.random.uniform(
            key_a, (self.num_agents, 2), minval=-1.0, maxval=+1.0
        )
        if self.fixed_obstacle_positions is not None:
            landmark_pos = self.fixed_obstacle_positions
        else:
            landmark_pos = jax.random.uniform(
                key_l, (self.num_landmarks, 2), minval=-1.0, maxval=+1.0
            )
        p_pos = jnp.concatenate([agent_pos, landmark_pos])

        resource_pos = self._place_resources(key_place)

        state = ResourceState(
            p_pos=p_pos,
            p_vel=jnp.zeros((self.num_entities, self.dim_p)),
            c=jnp.zeros((self.num_agents, self.dim_c)),
            done=jnp.full((self.num_agents,), False),
            step=0,
            resource_pos=resource_pos,
            collected=jnp.zeros(self.num_resources, dtype=bool),
        )
        return self.get_obs(state), state

    @partial(jax.jit, static_argnums=[0])
    def step_env(self, key: chex.PRNGKey, state: ResourceState, actions: dict):
        obs, new_state, reward, dones, info = super().step_env(key, state, actions)

        prey_idx = self.num_adversaries
        prey_pos = new_state.p_pos[prey_idx]
        dists = jnp.linalg.norm(new_state.resource_pos - prey_pos[None, :], axis=-1)
        newly_collected = (dists < self.collect_radius) & ~new_state.collected
        new_collected = new_state.collected | newly_collected
        n_new = jnp.sum(newly_collected.astype(jnp.float32))

        new_state = new_state.replace(collected=new_collected)
        obs = self.get_obs(new_state)

        prey_name = self.good_agents[0]
        reward = {**reward, prey_name: reward[prey_name] + self.collect_reward * n_new}
        info["resources_collected"] = jnp.sum(new_collected.astype(jnp.float32))

        return obs, new_state, reward, dones, info

    def get_obs(self, state) -> dict:
        base_obs = super().get_obs(state)

        for i, a in enumerate(self.good_agents):
            prey_idx = i + self.num_adversaries
            prey_pos = state.p_pos[prey_idx]
            rel_pos = state.resource_pos - prey_pos[None, :]
            collected_f = state.collected.astype(jnp.float32)
            resource_obs = jnp.concatenate([rel_pos.flatten(), collected_f])
            base_obs[a] = jnp.concatenate([base_obs[a], resource_obs])

        return base_obs
