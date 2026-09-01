"""Behaviour-cloning sample construction for predator policies.

Defaults target the 1v1 objective-typed env (one predator). Multi-predator
teams are supported via ``n_predators``.
"""
from __future__ import annotations

import numpy as np

from mopa.features import EP_LEN


def valid_bc_steps(t0: int, t1: int, ep_len: int = EP_LEN) -> tuple[int, ...]:
    """Steps with a valid one-step velocity feature.

    Auto-reset lands at indices ``0, ep_len+1, …``; velocity at those indices
    spans two episodes and must be excluded.
    """
    if t0 < 0 or t1 <= t0:
        raise ValueError(f"invalid step range [{t0}, {t1})")
    period = ep_len + 1
    return tuple(t for t in range(max(1, t0), t1) if t % period != 0)


def episode_end_steps(
    capture_t: np.ndarray | None,
    num_steps: int,
) -> np.ndarray | None:
    """Per-episode exclusive end indices for BC (drop post-capture frames)."""
    if capture_t is None:
        return None
    from mopa.features import episode_lengths

    return episode_lengths(capture_t, num_steps)


def predator_state_features(
    preds: np.ndarray,
    prey: np.ndarray,
    prev_preds: np.ndarray,
    prev_prey: np.ndarray,
    lava_pos: np.ndarray | None = None,
    lava_rad: np.ndarray | None = None,
) -> np.ndarray:
    """Build per-predator BC features for one timestep.

    ``preds`` / ``prev_preds``: ``(N, P, 2)``. Features contain all agent
    positions, a one-step velocity proxy, optional lava geometry, and a
    predator-id one-hot. Lava is part of the predator observation and is
    therefore required by the real objective dataset's risk-aware policy.
    """
    if preds.ndim != 3 or preds.shape[-1] != 2:
        raise ValueError("preds must have shape (N, P, 2)")
    if prev_preds.shape != preds.shape:
        raise ValueError("prev_preds must match preds")
    if prey.shape != (len(preds), 2) or prev_prey.shape != prey.shape:
        raise ValueError("prey and prev_prey must have shape (N, 2)")

    n, p, _ = preds.shape
    pos = np.concatenate([preds.reshape(n, p * 2), prey], -1)
    prev_pos = np.concatenate([prev_preds.reshape(n, p * 2), prev_prey], -1)
    vel = pos - prev_pos
    if (lava_pos is None) != (lava_rad is None):
        raise ValueError("lava_pos and lava_rad must be provided together")
    common = [pos, vel]
    if lava_pos is not None and lava_rad is not None:
        lava_positions = np.asarray(lava_pos, dtype=np.float32)
        lava_radii = np.asarray(lava_rad, dtype=np.float32)
        if lava_positions.ndim != 3 or lava_positions.shape[:1] != (n,):
            raise ValueError("lava_pos must have shape (N, L, 2)")
        if lava_positions.shape[-1] != 2:
            raise ValueError("lava_pos must have shape (N, L, 2)")
        if lava_radii.shape != lava_positions.shape[:2]:
            raise ValueError("lava_rad must have shape (N, L)")
        common.append(lava_positions.reshape(n, -1))
        common.append(lava_radii)
    shared = np.concatenate(common, axis=-1)
    feat_dim = shared.shape[-1] + p
    feats = np.zeros((n, p, feat_dim), np.float32)
    for i in range(p):
        pid = np.zeros((n, p), np.float32)
        pid[:, i] = 1.0
        feats[:, i] = np.concatenate([shared, pid], -1)
    return feats


def build_predator_samples(
    ds: dict[str, np.ndarray],
    t0: int,
    t1: int,
    ep_len: int = EP_LEN,
    capture_t: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build ``(state, action, episode_id)`` samples over a step interval.

    When ``capture_t`` is provided (or present on ``ds``), steps at or after
    each episode's capture time are dropped so frozen post-capture frames and
    synthetic zero actions never enter BC.
    """
    states, actions, episodes, _, _ = build_predator_samples_with_time(
        ds,
        t0,
        t1,
        ep_len=ep_len,
        capture_t=capture_t,
    )
    return states, actions, episodes


def build_predator_observation_samples(
    ds: dict[str, np.ndarray],
    t0: int = 0,
    t1: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build exact opponent-observation BC samples.

    Unlike :func:`build_predator_samples`, this path does not construct a
    velocity proxy and therefore includes the initial action at ``t = 0``.
    Observations must be the pre-action inputs recorded during rollout.
    """
    observations, actions, episodes, _, _ = (
        build_predator_observation_samples_with_time(ds, t0=t0, t1=t1)
    )
    return observations, actions, episodes


def build_predator_observation_samples_with_time(
    ds: dict[str, np.ndarray],
    t0: int = 0,
    t1: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return exact observations, actions, and causal sample provenance.

    ``ds['pred_obs']`` has shape ``(episode, state_time, predator, feature)``.
    It may contain either the ``T`` pre-action observations or all ``T + 1``
    states; only ``pred_obs[:, :T]`` is paired with ``ds['pred_act']``. Samples satisfy
    ``t0 <= timestep < min(t1, valid_length)``. The preferred validity source
    is ``ds['valid_length']``; capture metadata is accepted for compatibility.
    """
    observations = np.asarray(ds["pred_obs"])
    actions = np.asarray(ds["pred_act"])
    if observations.ndim != 4:
        raise ValueError(
            "ds['pred_obs'] must have shape (N, T, P, observation_dim)"
        )
    if observations.shape[-1] < 1:
        raise ValueError("pred_obs observation_dim must be positive")
    if actions.ndim != 3:
        raise ValueError("ds['pred_act'] must have shape (N, T, P)")
    if (
        actions.shape[0] != observations.shape[0]
        or actions.shape[2] != observations.shape[2]
        or observations.shape[1] not in {actions.shape[1], actions.shape[1] + 1}
    ):
        raise ValueError(
            "pred_obs must align with pred_act and contain T or T + 1 states"
        )
    if not np.all(np.isfinite(observations)):
        raise ValueError("pred_obs must contain only finite values")

    n_episodes, _, n_predators, observation_dim = observations.shape
    horizon = int(actions.shape[1])
    stop = horizon if t1 is None else int(t1)
    if t0 < 0 or stop <= t0 or stop > horizon:
        raise ValueError(f"invalid observation step range [{t0}, {stop})")

    if "valid_length" in ds:
        raw_lengths = np.asarray(ds["valid_length"])
        if raw_lengths.shape != (n_episodes,):
            raise ValueError("valid_length must have shape (N,)")
        if not np.issubdtype(raw_lengths.dtype, np.integer):
            if not np.all(np.equal(raw_lengths, np.floor(raw_lengths))):
                raise ValueError("valid_length must contain integers")
        lengths = raw_lengths.astype(np.int32)
    elif "capture_t" in ds:
        capture_t = np.asarray(ds["capture_t"])
        if capture_t.shape != (n_episodes,):
            raise ValueError("capture_t must have shape (N,)")
        lengths = np.where(capture_t >= 0, capture_t, horizon).astype(np.int32)
    else:
        lengths = np.full(n_episodes, horizon, dtype=np.int32)
    if np.any(lengths < 0) or np.any(lengths > horizon):
        raise ValueError("valid lengths must lie within the action horizon")

    sample_observations: list[np.ndarray] = []
    sample_actions: list[np.ndarray] = []
    sample_episodes: list[np.ndarray] = []
    sample_timesteps: list[np.ndarray] = []
    sample_predator_ids: list[np.ndarray] = []
    episode_ids = np.arange(n_episodes, dtype=np.int32)
    for timestep in range(t0, stop):
        keep = timestep < lengths
        if not np.any(keep):
            continue
        kept_episodes = episode_ids[keep]
        for predator_id in range(n_predators):
            sample_observations.append(observations[keep, timestep, predator_id])
            sample_actions.append(actions[keep, timestep, predator_id])
            sample_episodes.append(kept_episodes)
            sample_timesteps.append(
                np.full(len(kept_episodes), timestep, dtype=np.int32)
            )
            sample_predator_ids.append(
                np.full(len(kept_episodes), predator_id, dtype=np.int32)
            )

    if not sample_observations:
        return (
            np.zeros((0, observation_dim), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
        )
    return (
        np.concatenate(sample_observations).astype(np.float32),
        np.concatenate(sample_actions).astype(np.int32),
        np.concatenate(sample_episodes).astype(np.int32),
        np.concatenate(sample_timesteps).astype(np.int32),
        np.concatenate(sample_predator_ids).astype(np.int32),
    )


def build_predator_samples_with_time(
    ds: dict[str, np.ndarray],
    t0: int,
    t1: int,
    ep_len: int = EP_LEN,
    capture_t: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build BC samples and retain causal provenance for each sample.

    Returns ``(state, action, episode_id, timestep, predator_id)``.  Keeping
    ``timestep`` explicit lets callers condition a policy on the trajectory
    prefix available *before* that action instead of accidentally attaching a
    full-episode latent that contains future information.
    """
    prey = ds["prey_pos"]
    preds = ds["pred_pos"]
    acts = ds["pred_act"]
    if prey.ndim != 3 or prey.shape[-1] != 2:
        raise ValueError("ds['prey_pos'] must have shape (N, T + 1, 2)")
    if preds.ndim != 4 or preds.shape[-1] != 2:
        raise ValueError("ds['pred_pos'] must have shape (N, T + 1, P, 2)")
    n_pred = preds.shape[2]
    if acts.shape[:2] != (len(prey), preds.shape[1] - 1):
        raise ValueError("ds['pred_act'] must have shape (N, T, P)")
    if acts.shape[2] != n_pred:
        raise ValueError("pred_act predator dim must match pred_pos")
    if t1 > acts.shape[1]:
        raise ValueError(f"t1={t1} exceeds action horizon {acts.shape[1]}")

    if capture_t is None and "capture_t" in ds:
        capture_t = ds["capture_t"]
    ends = episode_end_steps(capture_t, acts.shape[1])
    lava_pos = ds.get("lava_pos")
    lava_rad = ds.get("lava_rad")
    if (lava_pos is None) != (lava_rad is None):
        raise ValueError("dataset lava_pos and lava_rad must be provided together")

    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    episodes: list[np.ndarray] = []
    timesteps: list[np.ndarray] = []
    predator_ids: list[np.ndarray] = []
    ep_ids = np.arange(len(prey), dtype=np.int32)
    for t in valid_bc_steps(t0, t1, ep_len):
        if ends is None:
            keep = np.ones(len(prey), dtype=bool)
        else:
            keep = t < ends
            if not np.any(keep):
                continue
        feats = predator_state_features(
            preds[keep, t],
            prey[keep, t],
            preds[keep, t - 1],
            prey[keep, t - 1],
            None if lava_pos is None else lava_pos[keep],
            None if lava_rad is None else lava_rad[keep],
        )
        kept_eps = ep_ids[keep]
        for p in range(n_pred):
            states.append(feats[:, p])
            actions.append(acts[keep, t, p])
            episodes.append(kept_eps)
            timesteps.append(np.full(len(kept_eps), t, dtype=np.int32))
            predator_ids.append(np.full(len(kept_eps), p, dtype=np.int32))

    if not states:
        lava_dim = 0 if lava_pos is None else 3 * int(lava_pos.shape[1])
        feat_dim = 5 * n_pred + 4 + lava_dim
        return (
            np.zeros((0, feat_dim), np.float32),
            np.zeros((0,), np.int32),
            np.zeros((0,), np.int32),
            np.zeros((0,), np.int32),
            np.zeros((0,), np.int32),
        )

    return (
        np.concatenate(states).astype(np.float32),
        np.concatenate(actions).astype(np.int32),
        np.concatenate(episodes).astype(np.int32),
        np.concatenate(timesteps).astype(np.int32),
        np.concatenate(predator_ids).astype(np.int32),
    )
