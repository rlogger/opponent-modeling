"""Simple-tag task with objective-typed predators.

One-to-many predators vs one prey, scattered resources, lava discs, early
termination on first capture, and predator rewards that differ by hidden
objective type. Objectives are pluggable: a predator type is an entry in
``OBJECTIVES``, either a declarative linear spec or an arbitrary callable.

    from tag_objectives import ObjectiveSpec, register_objective

    register_objective("ambusher", lambda env: ObjectiveSpec(
        capture=env.capture_bonus, dist=0.0, lava=-env.base_lava_penalty,
        novelty=0.0, still=0.05))

    register_objective("my_type", lambda env: my_reward_fn)

ctx channels (computed once per step): ``capture_f`` scalar, per-predator
``dist`` / ``in_lava`` / ``new_cell`` / ``pred_pos``, ``prey_pos``,
``prey_lava``, plus ``state`` / ``new_state``.

Current defaults: lava penalty is -100 for the risk-averse predator and 0 for
everyone else (``lava_penalty=100``, ``base_lava_penalty=0``,
``prey_lava_penalty=0``).

Mixed teams: ``pred_type`` may be a tuple of registered names, one per
predator (spec-based objectives only). Capture credit is shared; novelty is
team-shared.
"""
from __future__ import annotations

from functools import partial
from typing import Any, Callable, Optional, Union

import chex
import jax
import jax.numpy as jnp
from flax import struct
from jaxmarl.environments.mpe.simple_tag import SimpleTagMPE
from jaxmarl.environments.spaces import Box

from tag_objectives.resources import ResourceState, SimpleTagResourcesMPE

ObjectiveBuilder = Callable[[Any], Union["ObjectiveSpec", Callable[..., Any]]]


@struct.dataclass
class ObjectiveState(ResourceState):
    lava_pos: chex.Array = None
    lava_rad: chex.Array = None
    visited: chex.Array = None
    capture_t: chex.Array = None


@struct.dataclass
class ObjectiveSpec:
    """Linear reward over the standard channels, applied per predator i:

        r_i = capture * capture_f
            + dist * dist_i
            + lava * in_lava_i
            + novelty * new_cell_i
            + still * -|v_i|
    """

    capture: float = 0.0
    dist: float = 0.0
    lava: float = 0.0
    novelty: float = 0.0
    still: float = 0.0


OBJECTIVES: dict[str, ObjectiveBuilder] = {}


def register_objective(name: str, builder: ObjectiveBuilder) -> None:
    """Register a predator objective.

    ``builder(env)`` must return an :class:`ObjectiveSpec` or a
    ``callable(env, ctx) ->`` per-predator reward array.
    """
    OBJECTIVES[name] = builder


register_objective(
    "capture", lambda env: ObjectiveSpec(dist=-1.0, lava=-env.base_lava_penalty)
)
register_objective(
    "risk",
    lambda env: ObjectiveSpec(
        capture=env.capture_bonus,
        dist=-env.dense_chase_coef,
        lava=-env.lava_penalty,
    ),
)
register_objective(
    "curious",
    lambda env: ObjectiveSpec(
        capture=env.capture_bonus,
        dist=-env.dense_chase_coef,
        lava=-env.base_lava_penalty,
        novelty=env.novelty_bonus,
    ),
)


class SimpleTagObjectivesMPE(SimpleTagResourcesMPE):
    """Resource/lava simple-tag with pluggable predator objectives."""

    PRED_TYPES = ("capture", "risk", "curious")

    def __init__(
        self,
        pred_type: str = "capture",
        arena: float = 2.0,
        num_resources: int = 16,
        m_nearest_resources: int = 10,
        num_lava: int = 3,
        n_nearest_lava: int = 3,
        lava_radius_min: float = 0.35,
        lava_radius_max: float = 0.60,
        frac_resources_near_lava: float = 0.5,
        lava_penalty: float = 100.0,
        base_lava_penalty: float = 0.0,
        prey_lava_penalty: float = 0.0,
        novelty_bonus: float = 0.5,
        capture_bonus: float = 10.0,
        dense_chase_coef: float = 0.0,
        grid_size: int = 16,
        min_start_dist: float = 2.2,
        collect_radius: float = 0.15,
        collect_reward: float = 5.0,
        pred_max_speed: float = 1.4,
        prey_max_speed: float = 1.3,
        pred_accel: float = 3.0,
        prey_accel: float = 4.0,
        obstacle_positions: Optional[list[list[float]]] = None,
        max_steps: int = 100,
        **kwargs,
    ):
        pred_types = (
            tuple(pred_type) if isinstance(pred_type, (tuple, list)) else (pred_type,)
        )
        for t in pred_types:
            if t not in OBJECTIVES:
                raise ValueError(
                    f"pred_type must be a registered objective {sorted(OBJECTIVES)}, "
                    f"got {t!r}"
                )
        if not 1 <= m_nearest_resources <= num_resources:
            raise ValueError("m_nearest_resources must be in [1, num_resources]")
        if not 1 <= n_nearest_lava <= num_lava:
            raise ValueError("n_nearest_lava must be in [1, num_lava]")
        if not 0.0 <= frac_resources_near_lava <= 1.0:
            raise ValueError("frac_resources_near_lava must be in [0, 1]")

        kwargs.setdefault("num_adversaries", 1)
        kwargs.setdefault("num_good_agents", 1)
        kwargs.setdefault("num_obs", 0)
        kwargs.setdefault("max_steps", max_steps)

        super().__init__(
            num_resources=num_resources,
            placement="random",
            collect_radius=collect_radius,
            collect_reward=collect_reward,
            obstacle_positions=obstacle_positions,
            pred_max_speed=pred_max_speed,
            pred_accel=pred_accel,
            **kwargs,
        )

        if self.num_good_agents != 1:
            raise ValueError("the task defines exactly one prey")

        prey_idx = self.num_adversaries
        self.max_speed = self.max_speed.at[prey_idx].set(float(prey_max_speed))
        self.accel = self.accel.at[prey_idx].set(float(prey_accel))

        self.pred_type = pred_type
        self.arena = float(arena)
        self.m_nearest_resources = int(m_nearest_resources)
        self.num_lava = int(num_lava)
        self.n_nearest_lava = int(n_nearest_lava)
        self.lava_radius_min = float(lava_radius_min)
        self.lava_radius_max = float(lava_radius_max)
        self.frac_resources_near_lava = float(frac_resources_near_lava)
        self.n_near_lava_resources = int(
            round(self.frac_resources_near_lava * num_resources)
        )
        self.lava_penalty = float(lava_penalty)
        self.base_lava_penalty = float(base_lava_penalty)
        self.prey_lava_penalty = float(prey_lava_penalty)
        self.novelty_bonus = float(novelty_bonus)
        self.capture_bonus = float(capture_bonus)
        self.dense_chase_coef = float(dense_chase_coef)
        self.grid_size = int(grid_size)
        self.min_start_dist = float(min_start_dist)

        if len(pred_types) > 1 and len(pred_types) != self.num_adversaries:
            raise ValueError("tuple pred_type must have one entry per predator")
        self.pred_types = (
            pred_types if len(pred_types) > 1 else pred_types * self.num_adversaries
        )
        self.pred_type = (
            pred_type if isinstance(pred_type, str) else "+".join(pred_types)
        )

        resolved = [OBJECTIVES[t](self) for t in self.pred_types]
        if len(set(self.pred_types)) == 1 and not isinstance(resolved[0], ObjectiveSpec):
            self._objective: Union[ObjectiveSpec, Callable] = resolved[0]
        else:
            if not all(isinstance(r, ObjectiveSpec) for r in resolved):
                raise ValueError("mixed predator teams require spec-based objectives")
            self._objective = ObjectiveSpec(
                capture=jnp.array([r.capture for r in resolved]),
                dist=jnp.array([r.dist for r in resolved]),
                lava=jnp.array([r.lava for r in resolved]),
                novelty=jnp.array([r.novelty for r in resolved]),
                still=jnp.array([r.still for r in resolved]),
            )

        pred_base = (
            4
            + 2 * self.num_landmarks
            + 2 * (self.num_agents - 1)
            + 2 * self.num_good_agents
        )
        prey_base = 4 + 2 * self.num_landmarks + 2 * (self.num_agents - 1)
        lava_dim = self.n_nearest_lava * 3
        resource_dim = self.m_nearest_resources * 2
        for a in self.adversaries:
            self.observation_spaces[a] = Box(-jnp.inf, jnp.inf, (pred_base + lava_dim,))
        for a in self.good_agents:
            self.observation_spaces[a] = Box(
                -jnp.inf, jnp.inf, (prey_base + resource_dim + lava_dim,)
            )

    def _grid_index(self, pos: chex.Array) -> chex.Array:
        scaled = jnp.floor((pos + self.arena) / (2.0 * self.arena) * self.grid_size)
        cell = jnp.clip(scaled.astype(jnp.int32), 0, self.grid_size - 1)
        return cell[..., 0] * self.grid_size + cell[..., 1]

    def _arena_bounds_reward(self, pos: chex.Array) -> chex.Array:
        return jnp.sum(self.map_bounds_reward(jnp.abs(pos) / self.arena))

    def _in_lava_pos(self, state: ObjectiveState, pos: chex.Array) -> chex.Array:
        d = jnp.linalg.norm(pos[..., None, :] - state.lava_pos, axis=-1)
        return jnp.any(d < state.lava_rad, axis=-1)

    def _in_lava(self, state: ObjectiveState, agent_idx: int) -> chex.Array:
        return self._in_lava_pos(state, state.p_pos[agent_idx])

    def _nearest_lava_obs(self, state: ObjectiveState, pos: chex.Array) -> chex.Array:
        rel = state.lava_pos - pos[None, :]
        d = jnp.linalg.norm(rel, axis=-1)
        _, idx = jax.lax.top_k(-d, self.n_nearest_lava)
        lava = jnp.concatenate([rel[idx], state.lava_rad[idx, None]], axis=-1)
        return lava.reshape(-1)

    def _nearest_resource_obs(
        self, state: ObjectiveState, pos: chex.Array
    ) -> chex.Array:
        rel = state.resource_pos - pos[None, :]
        d = jnp.linalg.norm(rel, axis=-1) + state.collected.astype(jnp.float32) * 1e6
        _, idx = jax.lax.top_k(-d, self.m_nearest_resources)
        return rel[idx].reshape(-1)

    def _sample_lava(self, key_pos, key_rad, agent_pos):
        lava_rad = jax.random.uniform(
            key_rad,
            (self.num_lava,),
            minval=self.lava_radius_min,
            maxval=self.lava_radius_max,
        )
        cands = jax.random.uniform(
            key_pos,
            (self.num_lava, 4, 2),
            minval=-0.9 * self.arena,
            maxval=0.9 * self.arena,
        )
        d = jnp.linalg.norm(
            cands[:, :, None, :] - agent_pos[None, None, :, :], axis=-1
        )
        valid = jnp.all(d > lava_rad[:, None, None] + 0.35, axis=-1)
        first_valid = jnp.argmax(valid.astype(jnp.int32), axis=1)
        lava_pos = cands[jnp.arange(self.num_lava), first_valid]
        return lava_pos, lava_rad

    def _sample_resources(self, key, lava_pos, lava_rad):
        key_disc, key_r, key_ang, key_uni = jax.random.split(key, 4)
        k = self.n_near_lava_resources
        uniform = jax.random.uniform(
            key_uni, (self.num_resources, 2), minval=-self.arena, maxval=self.arena
        )
        if k == 0:
            return uniform
        disc = jax.random.randint(key_disc, (k,), 0, self.num_lava)
        r = lava_rad[disc] * jax.random.uniform(key_r, (k,), minval=0.3, maxval=1.4)
        ang = jax.random.uniform(key_ang, (k,), minval=0.0, maxval=2.0 * jnp.pi)
        offset = jnp.stack([r * jnp.cos(ang), r * jnp.sin(ang)], axis=-1)
        near = jnp.clip(lava_pos[disc] + offset, -self.arena, self.arena)
        return jnp.concatenate([near, uniform[k:]], axis=0)

    @partial(jax.jit, static_argnums=[0])
    def reset(self, key: chex.PRNGKey):
        key_spawn, key_l, key_res, key_lava, key_rad = jax.random.split(key, 5)
        key_ang, key_dist, key_center, key_jitter = jax.random.split(key_spawn, 4)

        P = self.num_adversaries
        angle = jax.random.uniform(key_ang, (), minval=0.0, maxval=2.0 * jnp.pi)
        direction = jnp.array([jnp.cos(angle), jnp.sin(angle)])
        dist = jax.random.uniform(
            key_dist, (), minval=self.min_start_dist, maxval=1.35 * self.arena
        )
        center = jax.random.uniform(key_center, (2,), minval=-0.15, maxval=0.15)
        jitter = jax.random.uniform(key_jitter, (P + 1, 2), minval=-0.08, maxval=0.08)
        pred_pos = center - 0.5 * dist * direction + jitter[:P]
        prey_pos = center + 0.5 * dist * direction + jitter[P]
        agent_pos = jnp.concatenate([pred_pos, prey_pos[None]], axis=0)
        agent_pos = jnp.clip(agent_pos, -0.85 * self.arena, 0.85 * self.arena)

        if self.fixed_obstacle_positions is not None:
            landmark_pos = self.fixed_obstacle_positions
        else:
            landmark_pos = jax.random.uniform(
                key_l, (self.num_landmarks, 2), minval=-self.arena, maxval=self.arena
            )
        p_pos = jnp.concatenate([agent_pos, landmark_pos], axis=0)

        lava_pos, lava_rad = self._sample_lava(key_lava, key_rad, agent_pos)
        resource_pos = self._sample_resources(key_res, lava_pos, lava_rad)

        start_cells = self._grid_index(agent_pos[:P])
        visited = (
            jnp.zeros((self.grid_size * self.grid_size,), dtype=bool)
            .at[start_cells]
            .set(True)
        )

        state = ObjectiveState(
            p_pos=p_pos,
            p_vel=jnp.zeros((self.num_entities, self.dim_p)),
            c=jnp.zeros((self.num_agents, self.dim_c)),
            done=jnp.full((self.num_agents,), False),
            step=0,
            resource_pos=resource_pos,
            collected=jnp.zeros(self.num_resources, dtype=bool),
            lava_pos=lava_pos,
            lava_rad=lava_rad,
            visited=visited,
            capture_t=jnp.array(-1, dtype=jnp.int32),
        )
        return self.get_obs(state), state

    @partial(jax.jit, static_argnums=[0])
    def step_env(self, key: chex.PRNGKey, state: ObjectiveState, actions: dict):
        _, new_state, _, _, info = super().step_env(key, state, actions)

        P = self.num_adversaries
        prey_idx = P
        pred_idxs = jnp.arange(P)
        prey_name = self.good_agents[0]

        captures = jnp.stack(
            [self.is_collision(prey_idx, i, new_state) for i in range(P)]
        )
        capture = jnp.any(captures)
        capture_f = capture.astype(jnp.float32)
        pred_pos = new_state.p_pos[:P]
        prey_pos = new_state.p_pos[prey_idx]
        dist = jnp.linalg.norm(pred_pos - prey_pos[None, :], axis=-1)
        pred_lava = self._in_lava_pos(new_state, pred_pos).astype(jnp.float32)
        prey_lava = self._in_lava(new_state, prey_idx).astype(jnp.float32)

        cells = self._grid_index(pred_pos)
        unvisited = ~new_state.visited[cells]
        same = cells[:, None] == cells[None, :]
        first_owner = jnp.argmax(same, axis=1)
        new_cell = (unvisited & (first_owner == pred_idxs)).astype(jnp.float32)
        visited = new_state.visited.at[cells].set(True)

        ctx = dict(
            capture_f=capture_f,
            dist=dist,
            in_lava=pred_lava,
            new_cell=new_cell,
            pred_pos=pred_pos,
            prey_pos=prey_pos,
            prey_lava=prey_lava,
            state=state,
            new_state=new_state,
        )
        obj = self._objective
        if isinstance(obj, ObjectiveSpec):
            speed = jnp.linalg.norm(new_state.p_vel[:P], axis=-1)
            pred_rewards = (
                obj.capture * capture_f
                + obj.dist * dist
                + obj.lava * pred_lava
                + obj.novelty * new_cell
                - obj.still * speed
            )
        else:
            pred_rewards = obj(self, ctx)

        n_new = jnp.sum(
            new_state.collected.astype(jnp.float32) - state.collected.astype(jnp.float32)
        )
        prey_reward = (
            -10.0 * capture_f
            - self._arena_bounds_reward(prey_pos)
            + self.collect_reward * n_new
            - self.prey_lava_penalty * prey_lava
        )
        rewards = {a: pred_rewards[i] for i, a in enumerate(self.adversaries)}
        rewards[prey_name] = prey_reward

        timeout = new_state.step >= self.max_steps
        done_all = capture | timeout
        done = jnp.full((self.num_agents,), done_all)
        capture_t = jnp.where(
            capture & (new_state.capture_t < 0),
            new_state.step.astype(jnp.int32),
            new_state.capture_t,
        )
        new_state = new_state.replace(done=done, visited=visited, capture_t=capture_t)
        obs = self.get_obs(new_state)

        dones = {a: done[i] for i, a in enumerate(self.agents)}
        dones.update({"__all__": done_all})

        resources_collected = jnp.sum(new_state.collected.astype(jnp.float32))
        res_lava_d = jnp.linalg.norm(
            new_state.resource_pos[:, None, :] - new_state.lava_pos[None, :, :],
            axis=-1,
        )
        near_lava = jnp.any(res_lava_d < 1.25 * new_state.lava_rad[None, :], axis=-1)
        near_lava_collected = jnp.sum(
            (new_state.collected & near_lava).astype(jnp.float32)
        )
        coverage = jnp.sum(visited.astype(jnp.float32))
        zeros = jnp.zeros((self.num_agents,), dtype=jnp.float32)
        info["resources_collected"] = zeros.at[prey_idx].set(resources_collected)
        info["near_lava_collected"] = zeros.at[prey_idx].set(near_lava_collected)
        info["new_resources_collected"] = zeros.at[prey_idx].set(n_new)
        info["pred_lava"] = zeros.at[pred_idxs].set(pred_lava)
        info["prey_lava"] = zeros.at[prey_idx].set(prey_lava)
        info["captured"] = jnp.full((self.num_agents,), capture_f)
        info["capture_t"] = jnp.full((self.num_agents,), capture_t.astype(jnp.float32))
        info["pred_new_cell"] = zeros.at[pred_idxs].set(new_cell)
        info["pred_coverage"] = zeros.at[pred_idxs].set(coverage)

        return obs, new_state, rewards, dones, info

    def get_obs(self, state: ObjectiveState) -> dict:
        base_obs = SimpleTagMPE.get_obs(self, state)

        for i, a in enumerate(self.adversaries):
            pos = state.p_pos[i]
            base_obs[a] = jnp.concatenate(
                [base_obs[a], self._nearest_lava_obs(state, pos)]
            )

        for i, a in enumerate(self.good_agents):
            idx = i + self.num_adversaries
            pos = state.p_pos[idx]
            base_obs[a] = jnp.concatenate(
                [
                    base_obs[a],
                    self._nearest_resource_obs(state, pos),
                    self._nearest_lava_obs(state, pos),
                ]
            )

        return base_obs
