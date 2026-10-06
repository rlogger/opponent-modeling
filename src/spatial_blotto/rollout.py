"""Simultaneous team control, complete episodes, and explicit transition exports.

Controllers see the same current physical state. Targets are diagnostics, never
observation channels. Collection uses step_env to retain the terminal transition.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from spatial_blotto.environment import BlottoState, SpatialBlotto

TeamPolicy = Callable[[BlottoState], jax.Array]


@struct.dataclass
class Episode:
    """T transitions and T+1 actual states; no reset state is spliced in."""

    states: BlottoState
    observations: jax.Array  # [T+1, agent, observation channel]
    world_states: jax.Array  # [T+1, centralized state channel]
    actions: jax.Array  # [T, agent, xy]
    team_rewards: jax.Array  # [T, red/blue], counted once per team
    ownership_scores: jax.Array  # [T, red/blue], raw regardless of reward mode
    terminated: jax.Array  # [T]; finite game horizon is terminal
    truncated: jax.Array  # [T]; no administrative cuts in this collector
    zone_counts: jax.Array  # [T+1, team, zone]
    zone_owners: jax.Array  # [T+1, zone]
    target_zones: jax.Array  # [T+1, agent]; diagnostic, -1 if policy omits it


def joint_actions(env: SpatialBlotto, red, blue) -> dict[str, jax.Array]:
    """Adapt two centralized team actions to the JaxMARL agent-keyed API."""
    red, blue = jnp.asarray(red), jnp.asarray(blue)
    if red.shape != (env.team_size, 2) or blue.shape != (env.team_size, 2):
        raise ValueError("each team action must have shape (team_size, 2)")
    array = jnp.concatenate((red, blue), axis=0)
    return {name: array[i] for i, name in enumerate(env.agents)}


def _targets(env, controller, state):
    method = getattr(controller, "target_zones", None)
    if method is None:
        return jnp.full((env.team_size,), -1, jnp.int32)
    targets = jnp.asarray(method(state))
    if targets.shape != (env.team_size,) or not jnp.issubdtype(
        targets.dtype, jnp.integer
    ):
        raise ValueError("target_zones must return one integer per team member")
    return targets


def rollout_episode(
    env: SpatialBlotto, red: TeamPolicy, blue: TeamPolicy, key: jax.Array
) -> Episode:
    """Pure JAX, jit/vmap-compatible complete finite-horizon match.

    The callable contract is policy(current_state) -> (team_size, 2) finite bounded
    actions in within-team ID order. A future learned-policy adapter can construct
    its permitted observations from that state; no RL trainer is implemented here.
    """
    reset_key, rollout_key = jax.random.split(key)
    _, initial = env.reset(reset_key)

    def targets(state):
        return jnp.concatenate((_targets(env, red, state), _targets(env, blue, state)))

    def advance(carry, _):
        state, rng = carry
        rng, step_key = jax.random.split(rng)
        actions = joint_actions(env, red(state), blue(state))
        _, following, _, _, info = env.step_env(step_key, state, actions)
        transition = (
            following,
            jnp.stack([actions[a] for a in env.agents]),
            info["team_rewards"],
            info["ownership_scores"],
            info["terminated"],
            info["truncated"],
        )
        return (following, rng), transition

    _, history = jax.lax.scan(
        advance, (initial, rollout_key), None, length=env.max_steps
    )
    states = jax.tree.map(
        lambda first, tail: jnp.concatenate((first[None], tail)), initial, history[0]
    )

    def observe(state):
        obs = env.get_obs(state)
        return jnp.stack([obs[a] for a in env.agents])

    observations = jax.vmap(observe)(states)
    return Episode(
        states=states,
        observations=observations,
        world_states=jax.vmap(env.get_state)(states),
        actions=history[1],
        team_rewards=history[2],
        ownership_scores=history[3],
        terminated=history[4],
        truncated=history[5],
        zone_counts=jax.vmap(env.zone_counts)(states),
        zone_owners=jax.vmap(env.zone_owners)(states),
        target_zones=jax.vmap(targets)(states),
    )


def validate_episode(env: SpatialBlotto, episode: Episode) -> None:
    """Validate shapes, finite values, lifecycle and score/state alignment on host.

    This boundary deliberately runs outside JIT, before artifact writes. The
    simulator itself remains composable with jit/vmap/scan.
    """
    e = jax.device_get(episode)
    t, n = env.max_steps, env.num_agents
    shapes = {
        "observations": (t + 1, n, env.obs_size),
        "world_states": (t + 1, env.state_size),
        "actions": (t, n, 2),
        "team_rewards": (t, 2),
        "ownership_scores": (t, 2),
        "terminated": (t,),
        "truncated": (t,),
        "zone_counts": (t + 1, 2, 3),
        "zone_owners": (t + 1, 3),
        "target_zones": (t + 1, n),
    }
    state_shapes = {
        "p_pos": (t + 1, n, 2),
        "p_vel": (t + 1, n, 2),
        "step": (t + 1,),
        "done": (t + 1,),
        "team_scores": (t + 1, 2),
    }
    for obj, expected in ((e, shapes), (e.states, state_shapes)):
        for name, shape in expected.items():
            if np.shape(getattr(obj, name)) != shape:
                raise ValueError(f"{name} shape must be {shape}")
    for leaf in jax.tree.leaves(e):
        if not np.isrealobj(leaf) or not np.isfinite(leaf).all():
            raise ValueError(
                "episode contains nonfinite values; inspect controller outputs"
            )
    if (np.abs(e.actions) > 1 + 1e-6).any():
        raise ValueError("controller emitted actions outside the declared Box")
    if (np.abs(e.states.p_pos) > env.arena + 1e-6).any():
        raise ValueError("episode position exceeds arena bounds")
    if (np.abs(e.observations) > 1 + 1e-6).any() or (
        np.abs(e.world_states) > 1 + 1e-6
    ).any():
        raise ValueError("episode observations exceed declared bounds")
    expected_done = np.arange(t + 1) == t
    if (
        not np.array_equal(e.states.done, expected_done)
        or not np.array_equal(e.terminated, expected_done[1:])
        or e.truncated.any()
    ):
        raise ValueError("expected exactly one terminal transition at the task horizon")
    if not np.array_equal(e.states.step, np.arange(t + 1)):
        raise ValueError("episode contains a reset or missing state")
    expected_counts = np.asarray(jax.vmap(env.zone_counts)(episode.states))
    if not np.array_equal(e.zone_counts, expected_counts):
        raise ValueError("zone counts disagree with recorded positions")
    owners = np.sign(expected_counts[:, 0] - expected_counts[:, 1])
    raw = np.stack(
        ((owners[1:] == 1).sum(axis=-1), (owners[1:] == -1).sum(axis=-1)), axis=-1
    )
    scores = np.concatenate((np.zeros((1, 2), dtype=np.int64), raw.cumsum(axis=0)))
    if (
        not np.array_equal(e.zone_owners, owners)
        or not np.array_equal(e.ownership_scores, raw)
        or not np.array_equal(e.states.team_scores, scores)
    ):
        raise ValueError(
            "ownership or cumulative team score disagrees with zone counts"
        )
    rewards = (
        raw
        if env.reward_mode == "ownership"
        else np.stack((raw[:, 0] - raw[:, 1], raw[:, 1] - raw[:, 0]), axis=-1)
    )
    if not np.array_equal(e.team_rewards, rewards):
        raise ValueError(
            "team rewards disagree with the configured ownership objective"
        )
    if (
        not np.issubdtype(e.target_zones.dtype, np.integer)
        or ((e.target_zones < -1) | (e.target_zones >= env.num_zones)).any()
    ):
        raise ValueError("diagnostic target zone is invalid")

    def replay_step(state, actions):
        mapping = {agent: actions[i] for i, agent in enumerate(env.agents)}
        return env.step_env(jax.random.PRNGKey(0), state, mapping)[1]

    expected_next = jax.vmap(replay_step)(
        jax.tree.map(lambda a: a[:-1], episode.states), episode.actions
    )
    for actual, expected in zip(
        jax.tree.leaves(e.states), jax.tree.leaves(expected_next), strict=True
    ):
        if not np.allclose(actual[1:], expected, atol=1e-6, rtol=1e-6):
            raise ValueError(
                "recorded actions do not reproduce the next physical state"
            )
    # Reconcile redundant arrays; observations never acquire target diagnostics.
    expected_world = np.asarray(jax.vmap(env.get_state)(episode.states))
    expected_obs = jax.vmap(env.get_obs)(episode.states)
    expected_obs = np.stack([expected_obs[a] for a in env.agents], axis=1)
    if not np.allclose(e.world_states, expected_world, atol=1e-6) or not np.allclose(
        e.observations, expected_obs, atol=1e-6
    ):
        raise ValueError("observations disagree with recorded physical states")


def episode_payload(
    env: SpatialBlotto, episode: Episode, *, seed: int, controllers: dict, scenario: str
) -> dict:
    """Serializable replay and provenance. No learning or equilibrium claim."""
    validate_episode(env, episode)
    e = jax.device_get(episode)
    target_ids = np.asarray(e.target_zones)
    targets = np.asarray(env.zone_centers)[np.maximum(target_ids, 0)]
    targets = np.where((target_ids >= 0)[..., None], targets, e.states.p_pos)
    package = Path(__file__).parent
    return {
        "schemaVersion": 1,
        "seed": seed,
        "steps": env.max_steps,
        "teamSize": env.team_size,
        "rewardMode": env.reward_mode,
        "arena": env.arena,
        "dt": env.dt,
        "maxSpeed": env.max_speed,
        "radius": env.zone_radius,
        "centers": np.asarray(env.zone_centers).tolist(),
        "agents": list(env.agents),
        "controllers": controllers,
        "scenario": scenario,
        "provenance": {
            "policyKind": "mathematical-baseline-no-training",
            "sourceSha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(package.glob("*.py"))
            },
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("jax", "jaxlib", "jaxmarl", "flax", "numpy")
            },
            "terminalSemantics": "finite-task-horizon; no autoreset state in transitions",
            "targetSemantics": "current controller diagnostics, excluded from observations",
        },
        "summary": {
            "teamReturns": e.team_rewards.sum(axis=0).tolist(),
            "rawOwnershipScores": e.states.team_scores[-1].tolist(),
            "finalZoneCounts": e.zone_counts[-1].tolist(),
            "finalZoneOwners": e.zone_owners[-1].tolist(),
        },
        "frames": {
            "positions": e.states.p_pos.tolist(),
            "velocities": e.states.p_vel.tolist(),
            "counts": e.zone_counts.tolist(),
            "owners": e.zone_owners.tolist(),
            "scores": e.states.team_scores.tolist(),
            "rewards": np.concatenate((np.zeros((1, 2)), e.team_rewards)).tolist(),
            "done": e.states.done.tolist(),
            "targetZones": target_ids.tolist(),
            "targets": targets.tolist(),
        },
    }


def save_trajectory(
    path: str | Path, episode: Episode, metadata: dict, *, env: SpatialBlotto
) -> Path:
    """Non-pickled NPZ: action/reward at t aligns with observations t and t+1.

    Diagnostic target zones are intentionally omitted from the learning arrays.
    The renderer metadata remains explicitly labeled, separate from observations.
    """
    validate_episode(env, episode)
    metadata_text = json.dumps(metadata, allow_nan=False, sort_keys=True)
    path = Path(path).expanduser().resolve()
    if path.suffix != ".npz":
        raise ValueError("trajectory path must end in .npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    e = jax.device_get(episode)
    np.savez_compressed(
        path,
        observations=e.observations,
        world_states=e.world_states,
        positions=e.states.p_pos,
        velocities=e.states.p_vel,
        actions=e.actions,
        team_rewards=e.team_rewards,
        ownership_scores=e.ownership_scores,
        team_scores=e.states.team_scores,
        terminated=e.terminated,
        truncated=e.truncated,
        zone_counts=e.zone_counts,
        zone_owners=e.zone_owners,
        metadata_json=np.asarray(metadata_text),
    )
    return path
