#!/usr/bin/env python3
"""Run the matched four-arm opponent behaviour-cloning experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax.traverse_util import flatten_dict

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mopa.bc import (  # noqa: E402
    BCPolicy,
    build_observation_samples_with_time,
    evaluate_bc,
    fit_bc,
)
from mopa.data import OBJECTIVE_TYPES, objective_dataset  # noqa: E402
from mopa.encoders import (  # noqa: E402
    encode_jepa_gru,
    train_jepa_gru_with_params,
)
from mopa.features import predator_sequence_features  # noqa: E402
from mopa.manifest import (  # noqa: E402
    file_sha256,
    git_dirty,
    git_sha,
    package_versions,
)
from mopa.nets import ActorLogits  # noqa: E402
from mopa.trajectory_export import validate_objective_dataset  # noqa: E402
from tag_objectives import SimpleTagObjectivesMPE  # noqa: E402
from tag_objectives.teams import freeze_tree  # noqa: E402

ARMS = ("no_z", "real_z", "shuffled_z", "oracle")
CONDITION_DIM = 3
ACTION_CLASSES = tuple(range(5))
OFFLINE_METRICS = {
    "episode_strategy_macro_nll": "lower",
    "episode_strategy_macro_accuracy": "higher",
    "action_balanced_accuracy": "higher",
}
CLOSED_LOOP_METRICS = {
    "expert_action_agreement": "higher",
    "predator_ade": "lower",
    "capture_rate_gap": "lower",
    "survival_time_gap": "lower",
    "pred_lava_steps_gap": "lower",
    "pred_coverage_gap": "lower",
}


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _dataset_manifest(
    args: argparse.Namespace,
    *,
    run_git_sha: str | None,
    run_git_dirty: bool | None,
    checkpoints: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "git_sha": run_git_sha,
        "git_dirty": run_git_dirty,
        "rollout": {
            "n_eps": args.n_eps,
            "num_steps": args.num_steps,
            "rollout_seed": args.rollout_seed,
            "checkpoint_seeds": list(args.ckpt_seeds),
            "prey_type": "capture",
            "num_adversaries": 1,
        },
        "source_checkpoints": [
            {
                key: row[key]
                for key in ("objective", "team", "seed", "sha256")
            }
            for row in checkpoints
        ],
        "dependencies": package_versions(),
    }


def _load_or_create_dataset(
    args: argparse.Namespace,
    *,
    run_git_sha: str | None,
    run_git_dirty: bool | None,
    checkpoints: list[dict[str, Any]],
) -> tuple[dict[str, np.ndarray], str, Path]:
    path = args.dataset_cache
    manifest_path = path.with_name(f"{path.name}.manifest.json")
    expected_manifest = _dataset_manifest(
        args,
        run_git_sha=run_git_sha,
        run_git_dirty=run_git_dirty,
        checkpoints=checkpoints,
    )
    if path.is_file():
        if run_git_dirty is not False:
            raise ValueError(
                "dataset cache reuse requires a clean worktree; use a new cache path"
            )
        if not manifest_path.is_file():
            raise ValueError("dataset cache is missing its provenance manifest")
        cached_manifest = json.loads(manifest_path.read_text())
        if cached_manifest != expected_manifest:
            raise ValueError(
                "dataset cache provenance does not match this code/config/checkpoints"
            )
        with np.load(path, allow_pickle=False) as raw:
            data = {name: np.asarray(raw[name]) for name in raw.files}
        source = "cache"
    else:
        dataset = objective_dataset(
            n_eps=args.n_eps,
            ckpt_seeds=args.ckpt_seeds,
            rng0=args.rollout_seed,
            num_steps=args.num_steps,
            logdir=args.logdir,
            num_adversaries=1,
            prey_type="capture",
        )
        data = dataset.as_dict()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **data)
        manifest_path.write_text(
            json.dumps(expected_manifest, indent=2, sort_keys=True) + "\n"
        )
        source = "fresh_rollout"
    if "pred_obs" not in data:
        raise ValueError("dataset lacks exact pred_obs; regenerate the BC cache")
    return data, source, manifest_path


def _validate_dataset(
    data: dict[str, np.ndarray],
    *,
    checkpoint_seeds: tuple[int, ...],
    n_eps: int,
    num_steps: int,
    rollout_seed: int,
) -> None:
    summary = validate_objective_dataset(data)
    actions = np.asarray(data["pred_act"])
    labels = np.asarray(data["label"])
    checkpoints = np.asarray(data["ckpt_seed"])
    n, horizon, predators = actions.shape
    expected_n = len(OBJECTIVE_TYPES) * len(checkpoint_seeds) * n_eps
    if n != expected_n or horizon != num_steps:
        raise ValueError(
            "dataset cache does not match --n-eps/--num-steps; regenerate it"
        )
    if set(np.unique(checkpoints).tolist()) != set(checkpoint_seeds):
        raise ValueError("dataset checkpoint seeds do not match --ckpt-seeds")
    if predators != 1 or summary.get("predator_observation_dim") is None:
        raise ValueError("the matched BC experiment currently requires one predator")
    for checkpoint in checkpoint_seeds:
        key = jax.random.PRNGKey(rollout_seed + checkpoint)
        _, reset_key = jax.random.split(key)
        expected_keys = np.asarray(jax.random.split(reset_key, n_eps))
        for label in range(len(OBJECTIVE_TYPES)):
            rows = np.flatnonzero((checkpoints == checkpoint) & (labels == label))
            if len(rows) != n_eps or not np.array_equal(
                np.asarray(data["env_seed"])[rows], expected_keys
            ):
                raise ValueError(
                    "dataset reset keys do not match --rollout-seed/--n-eps"
                )


def _scale_sequence_train_only(
    sequence: np.ndarray,
    lengths: np.ndarray,
    train_episodes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = np.arange(sequence.shape[1])[None, :] < lengths[:, None]
    train_values = sequence[train_episodes][valid[train_episodes]]
    mean = train_values.mean(axis=0).astype(np.float32)
    std = (train_values.std(axis=0) + 1e-6).astype(np.float32)
    scaled = ((sequence - mean) / std).astype(np.float32)
    scaled *= valid[..., None].astype(np.float32)
    return scaled, mean, std


def _derangement(indices: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Map every supplied episode to another episode without fixed points."""
    ids = np.asarray(indices, dtype=np.int32)
    if len(ids) < 2:
        raise ValueError("a shuffled-z split needs at least two episodes")
    order = rng.permutation(ids)
    mapping = np.empty(int(ids.max()) + 1, dtype=np.int32)
    mapping[order] = np.roll(order, 1)
    if np.any(mapping[ids] == ids):
        raise AssertionError("failed to construct an episode derangement")
    return mapping


def split_local_derangement(
    validation_episodes: np.ndarray,
    seed: int,
    checkpoint_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Derange episodes without crossing a split or checkpoint when supplied."""
    validation = np.asarray(validation_episodes, dtype=bool)
    checkpoints = (
        np.zeros(len(validation), dtype=np.int32)
        if checkpoint_ids is None
        else np.asarray(checkpoint_ids, dtype=np.int32)
    )
    if checkpoints.shape != validation.shape:
        raise ValueError("checkpoint IDs must align with the episode split")
    rng = np.random.default_rng(seed)
    output = np.empty(len(validation), dtype=np.int32)
    for split_value in (False, True):
        for checkpoint in np.unique(checkpoints[validation == split_value]):
            ids = np.flatnonzero(
                (validation == split_value) & (checkpoints == checkpoint)
            )
            local = _derangement(ids, rng)
            output[ids] = local[ids]
    return output


def _shuffle_contingency(
    labels: np.ndarray,
    episode_permutation: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    """Count action-episode labels against shuffled latent-source labels."""
    labels = np.asarray(labels, dtype=np.int32)
    permutation = np.asarray(episode_permutation, dtype=np.int32)
    selected = np.asarray(mask, dtype=bool)
    counts = np.zeros((len(OBJECTIVE_TYPES), len(OBJECTIVE_TYPES)), dtype=np.int32)
    np.add.at(counts, (labels[selected], labels[permutation[selected]]), 1)
    return {
        "label_order": list(OBJECTIVE_TYPES),
        "rows": "action_episode_label",
        "columns": "latent_source_label",
        "counts": counts.tolist(),
        "same_label_fraction": float(np.trace(counts) / max(counts.sum(), 1)),
    }


def causal_sample_latents(
    prefix_latents: np.ndarray,
    episode_ids: np.ndarray,
    timesteps: np.ndarray,
) -> np.ndarray:
    """Attach only the latent from transitions completed before the action."""
    latent = np.asarray(prefix_latents, dtype=np.float32)
    episodes = np.asarray(episode_ids, dtype=np.int32)
    time = np.asarray(timesteps, dtype=np.int32)
    if latent.ndim != 3 or episodes.shape != time.shape:
        raise ValueError("latents and sample provenance do not align")
    output = np.zeros((len(time), latent.shape[-1]), dtype=np.float32)
    observed = time > 0
    output[observed] = latent[episodes[observed], time[observed] - 1]
    return output


def _pad_condition(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] > CONDITION_DIM:
        raise ValueError(f"conditioning must have at most {CONDITION_DIM} columns")
    output = np.zeros((len(values), CONDITION_DIM), dtype=np.float32)
    output[:, : values.shape[1]] = values
    return output


def conditioning_arms(
    prefix_latents: np.ndarray,
    labels: np.ndarray,
    episodes: np.ndarray,
    timesteps: np.ndarray,
    episode_permutation: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build the four matched three-column conditioning matrices."""
    real = causal_sample_latents(prefix_latents, episodes, timesteps)
    shuffled = causal_sample_latents(
        prefix_latents, episode_permutation[episodes], timesteps
    )
    oracle = np.eye(CONDITION_DIM, dtype=np.float32)[labels[episodes]]
    return {
        "no_z": np.zeros((len(episodes), CONDITION_DIM), dtype=np.float32),
        "real_z": _pad_condition(real),
        "shuffled_z": _pad_condition(shuffled),
        "oracle": oracle,
    }


def _save_tree(path: Path, tree: Any, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = {
        f"params/{name}": np.asarray(value)
        for name, value in flatten_dict(tree, sep="/").items()
    }
    values.update({name: np.asarray(value) for name, value in arrays.items()})
    np.savez_compressed(path, **values)


def _checkpoint_path(logdir: Path, objective: str, team: str, seed: int) -> Path:
    return logdir / (
        f"mappo_objectives_{objective}_MPE_simple_tag_v3_"
        f"{team}_actor_seed0_vmap{seed}.safetensors"
    )


def _checkpoint_manifest(
    logdir: Path, checkpoint_seeds: tuple[int, ...]
) -> list[dict[str, Any]]:
    sources = [
        (objective, "pred", seed)
        for seed in checkpoint_seeds
        for objective in OBJECTIVE_TYPES
    ] + [("capture", "prey", seed) for seed in checkpoint_seeds]
    manifest = []
    for objective, team, seed in sources:
        path = _checkpoint_path(logdir, objective, team, seed)
        if not path.is_file():
            raise FileNotFoundError(f"missing source checkpoint: {path}")
        manifest.append(
            {
                "objective": objective,
                "team": team,
                "seed": int(seed),
                "path": str(path),
                "sha256": file_sha256(path),
            }
        )
    return manifest


def _pad_obs(obs: jax.Array, width: int) -> jax.Array:
    if obs.shape[-1] == width:
        return obs
    return jnp.concatenate(
        [obs, jnp.zeros(obs.shape[:-1] + (width - obs.shape[-1],), obs.dtype)],
        axis=-1,
    )


def _online_latent(
    prey_history: list[np.ndarray],
    pred_history: list[np.ndarray],
    done: np.ndarray,
    capture_t: np.ndarray,
    encoder_params: Any,
    sequence_mean: np.ndarray,
    sequence_std: np.ndarray,
    *,
    horizon: int,
    latent_dim: int,
    hidden_dim: int,
) -> np.ndarray:
    completed = len(prey_history) - 1
    batch = len(done)
    prey = np.zeros((batch, horizon + 1, 2), dtype=np.float32)
    pred = np.zeros((batch, horizon + 1, 1, 2), dtype=np.float32)
    prey[:, : completed + 1] = np.stack(prey_history, axis=1)
    pred[:, : completed + 1] = np.stack(pred_history, axis=1)
    lengths = np.where(done, capture_t, completed).astype(np.int32)
    lengths = np.clip(lengths, 1, horizon)
    sequence = predator_sequence_features(
        prey, pred, lengths, velocity_mode="legacy_forward"
    )
    valid = np.arange(horizon)[None, :] < lengths[:, None]
    sequence = ((sequence - sequence_mean) / sequence_std).astype(np.float32)
    sequence *= valid[..., None].astype(np.float32)
    latents = encode_jepa_gru(
        encoder_params,
        sequence,
        lengths,
        lat=latent_dim,
        hid=hidden_dim,
    )
    return latents[np.arange(batch), lengths - 1]


def _mean_gap(actual: np.ndarray, reference: np.ndarray) -> float:
    return float(
        abs(
            np.asarray(actual, dtype=np.float64).mean()
            - np.asarray(reference, dtype=np.float64).mean()
        )
    )


def closed_loop_continuation(
    policy: BCPolicy,
    data: dict[str, np.ndarray],
    *,
    arm: str,
    heldout_checkpoint: int,
    encoder_params: Any,
    sequence_mean: np.ndarray,
    sequence_std: np.ndarray,
    latent_dim: int,
    hidden_dim: int,
    ctx: int,
    rollout_seed: int,
    requested_episodes: int,
    shuffle_seed: int,
    logdir: Path,
) -> dict[str, Any]:
    """Replay an expert prefix, then evaluate greedy BC under its own states."""
    from jaxmarl.wrappers.baselines import load_params

    labels = np.asarray(data["label"], dtype=np.int32)
    checkpoints = np.asarray(data["ckpt_seed"], dtype=np.int32)
    selected: list[np.ndarray] = []
    group_size: int | None = None
    for label in range(len(OBJECTIVE_TYPES)):
        group = np.flatnonzero((checkpoints == heldout_checkpoint) & (labels == label))
        if group_size is None:
            group_size = len(group)
        if len(group) != group_size:
            raise ValueError("closed-loop groups must have equal sizes")
        selected.append(group)
    assert group_size is not None
    eligible = np.all(
        np.stack([np.asarray(data["valid_length"])[group] > ctx for group in selected]),
        axis=0,
    )
    eligible_offsets = np.flatnonzero(eligible)
    num_episodes = (
        len(eligible_offsets) if requested_episodes == 0 else requested_episodes
    )
    if num_episodes < 1 or len(eligible_offsets) < num_episodes:
        raise ValueError(
            "not enough matched heldout episodes survive the requested prefix"
        )
    offsets = eligible_offsets[:num_episodes]
    episode_indices = np.concatenate([group[offsets] for group in selected])
    strategy = np.repeat(np.arange(3, dtype=np.int32), num_episodes)
    batch = len(episode_indices)
    horizon = int(data["pred_act"].shape[1])
    if not 0 < ctx < horizon or np.any(data["valid_length"][episode_indices] <= ctx):
        raise ValueError("every closed-loop episode must survive the expert prefix")

    env = SimpleTagObjectivesMPE(pred_type="capture", num_adversaries=1)
    pred_name = env.adversaries[0]
    prey_name = env.good_agents[0]
    pred_index = env.agents.index(pred_name)
    prey_index = env.agents.index(prey_name)
    action_dim = int(env.action_space(pred_name).n)
    obs_width = max(env.observation_space(agent).shape[0] for agent in env.agents)
    actor = ActorLogits(action_dim=action_dim, hidden_dim=128)
    pred_params = {
        label: load_params(
            str(_checkpoint_path(logdir, objective, "pred", heldout_checkpoint))
        )
        for label, objective in enumerate(OBJECTIVE_TYPES)
    }
    prey_params = load_params(
        str(_checkpoint_path(logdir, "capture", "prey", heldout_checkpoint))
    )

    rng = jax.random.PRNGKey(rollout_seed + heldout_checkpoint)
    rng, reset_rng = jax.random.split(rng)
    all_reset_keys = jax.random.split(reset_rng, group_size)
    reset_keys = jnp.concatenate([all_reset_keys[offsets]] * 3, axis=0)
    expected_reset = np.asarray(data["env_seed"])[episode_indices]
    if not np.array_equal(np.asarray(reset_keys), expected_reset):
        raise ValueError("closed-loop reset keys do not match the dataset")
    obs, state = jax.vmap(env.reset)(reset_keys)
    done = jnp.zeros((batch,), dtype=bool)
    pred_lava_steps = jnp.zeros((batch,), dtype=jnp.float32)
    prey_history = [np.asarray(state.p_pos[:, prey_index])]
    pred_history = [np.asarray(state.p_pos[:, pred_index : pred_index + 1])]
    agreement_sum = np.zeros(batch, dtype=np.int32)
    agreement_count = np.zeros(batch, dtype=np.int32)
    closed_shuffle = _derangement(
        np.arange(batch, dtype=np.int32), np.random.default_rng(shuffle_seed)
    )

    for timestep in range(horizon):
        active = ~done
        if timestep <= ctx:
            np.testing.assert_allclose(
                np.asarray(obs[pred_name]),
                np.asarray(data["pred_obs"])[episode_indices, timestep, 0],
                atol=1e-6,
            )
        expert_pred_parts = []
        for label in range(3):
            rows = slice(label * num_episodes, (label + 1) * num_episodes)
            logits = actor.apply(
                pred_params[label], _pad_obs(obs[pred_name][rows], obs_width)
            )
            expert_pred_parts.append(jnp.argmax(logits, axis=-1).astype(jnp.int32))
        expert_pred_action = jnp.concatenate(expert_pred_parts)
        prey_logits = actor.apply(prey_params, _pad_obs(obs[prey_name], obs_width))
        prey_action = jnp.argmax(prey_logits, axis=-1).astype(jnp.int32)

        if timestep < ctx:
            pred_action = expert_pred_action
            expected_pred = np.asarray(data["pred_act"])[episode_indices, timestep, 0]
            expected_prey = np.asarray(data["prey_act"])[episode_indices, timestep]
            if not np.array_equal(np.asarray(pred_action), expected_pred):
                raise ValueError("expert predator prefix does not replay exactly")
            if not np.array_equal(np.asarray(prey_action), expected_prey):
                raise ValueError("expert prey prefix does not replay exactly")
        else:
            if arm == "oracle":
                condition = np.eye(CONDITION_DIM, dtype=np.float32)[strategy]
            elif arm == "no_z":
                condition = np.zeros((batch, CONDITION_DIM), dtype=np.float32)
            else:
                latent = _online_latent(
                    prey_history,
                    pred_history,
                    np.asarray(done),
                    np.asarray(state.capture_t),
                    encoder_params,
                    sequence_mean,
                    sequence_std,
                    horizon=horizon,
                    latent_dim=latent_dim,
                    hidden_dim=hidden_dim,
                )
                if arm == "shuffled_z":
                    latent = latent[closed_shuffle]
                condition = _pad_condition(latent)
            features = np.concatenate(
                [np.asarray(obs[pred_name], dtype=np.float32), condition], axis=-1
            )
            pred_action = jnp.asarray(policy.greedy_action(features), dtype=jnp.int32)
            active_np = np.asarray(active)
            agreement_sum[active_np] += (
                np.asarray(pred_action)[active_np]
                == np.asarray(expert_pred_action)[active_np]
            ).astype(np.int32)
            agreement_count[active_np] += 1

        rng, step_rng = jax.random.split(rng)
        all_step_keys = jax.random.split(step_rng, group_size)
        step_keys = jnp.concatenate([all_step_keys[offsets]] * 3, axis=0)
        actions = {pred_name: pred_action, prey_name: prey_action}
        new_obs, new_state, _, dones, info = jax.vmap(env.step_env)(
            step_keys, state, actions
        )
        active_float = active.astype(jnp.float32)
        pred_lava_steps += info["pred_lava"][:, pred_index] * active_float
        state = freeze_tree(active, new_state, state)
        obs = freeze_tree(active, new_obs, obs)
        done = done | dones["__all__"]
        prey_history.append(np.asarray(state.p_pos[:, prey_index]))
        pred_history.append(np.asarray(state.p_pos[:, pred_index : pred_index + 1]))
        if timestep < ctx:
            np.testing.assert_allclose(
                pred_history[-1],
                np.asarray(data["pred_pos"])[episode_indices, timestep + 1],
                atol=1e-6,
            )

    capture_t = np.asarray(state.capture_t)
    captured = capture_t >= 0
    clone_length = np.where(captured, capture_t, horizon).astype(np.int32)
    clone_pred = np.stack(pred_history, axis=1)
    expert_pred = np.asarray(data["pred_pos"])[episode_indices]
    expert_length = np.asarray(data["valid_length"])[episode_indices]
    common_length = np.minimum(clone_length, expert_length)
    time = np.arange(horizon + 1)[None, :]
    continuation = (time > ctx) & (time <= common_length[:, None])
    distance = np.linalg.norm(clone_pred[..., 0, :] - expert_pred[..., 0, :], axis=-1)
    episode_ade = np.sum(distance * continuation, axis=1) / np.maximum(
        continuation.sum(axis=1), 1
    )
    coverage = np.asarray(state.visited).sum(axis=-1).astype(np.float32)
    survival = np.where(captured, capture_t, horizon).astype(np.float32)
    episode_agreement = agreement_sum / np.maximum(agreement_count, 1)

    per_strategy: dict[str, Any] = {}
    for label, name in enumerate(OBJECTIVE_TYPES):
        rows = strategy == label
        source_rows = episode_indices[rows]
        per_strategy[name] = {
            "n_episodes": int(rows.sum()),
            "expert_action_agreement": float(episode_agreement[rows].mean()),
            "predator_ade": float(episode_ade[rows].mean()),
            "capture_rate_gap": _mean_gap(
                captured[rows], data["captured"][source_rows]
            ),
            "survival_time_gap": _mean_gap(
                survival[rows], data["survival_time"][source_rows]
            ),
            "pred_lava_steps_gap": _mean_gap(
                np.asarray(pred_lava_steps)[rows], data["pred_lava_steps"][source_rows]
            ),
            "pred_coverage_gap": _mean_gap(
                coverage[rows], data["pred_coverage"][source_rows]
            ),
        }
    metric_names = next(iter(per_strategy.values())).keys() - {"n_episodes"}
    macro = {
        name: float(np.mean([row[name] for row in per_strategy.values()]))
        for name in metric_names
    }
    macro["n_continuation_actions"] = int(agreement_count.sum())
    result = {
        "n_eligible_matched_episodes": int(len(eligible_offsets)),
        "macro": macro,
        "per_strategy": per_strategy,
    }
    if arm == "shuffled_z":
        result["shuffle_label_contingency"] = _shuffle_contingency(
            strategy, closed_shuffle, np.ones(batch, dtype=bool)
        )
    return result


def _fold_metric_means(
    folds: list[dict[str, Any]], section: str, arm: str, metric: str
) -> list[float]:
    means = []
    for fold in folds:
        values = []
        for seed in fold["seeds"]:
            row = seed[section].get(arm)
            if row is None:
                continue
            if section == "closed_loop":
                row = row["macro"]
            if metric in row:
                values.append(float(row[metric]))
        if values:
            means.append(float(np.mean(values)))
    return means


def _aggregate_section(
    folds: list[dict[str, Any]], section: str, metrics: dict[str, str]
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for arm in ARMS:
        output[arm] = {}
        for metric in metrics:
            fold_means = _fold_metric_means(folds, section, arm, metric)
            if fold_means:
                values = np.asarray(fold_means, dtype=np.float64)
                output[arm][metric] = {
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "fold_means": values.tolist(),
                }
    return output


def _paired_section(
    folds: list[dict[str, Any]], section: str, metrics: dict[str, str]
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for arm in ARMS[1:]:
        output[arm] = {}
        for metric, better_when in metrics.items():
            baseline = _fold_metric_means(folds, section, "no_z", metric)
            candidate = _fold_metric_means(folds, section, arm, metric)
            if not baseline or len(baseline) != len(candidate):
                continue
            delta = np.asarray(candidate) - np.asarray(baseline)
            improvement = delta if better_when == "higher" else -delta
            output[arm][metric] = {
                "better_when": better_when,
                "delta_mean": float(delta.mean()),
                "delta_std": float(delta.std()),
                "fold_deltas": delta.tolist(),
                "all_folds_better": bool(np.all(improvement > 0.0)),
            }
    return output


def _summaries(folds: list[dict[str, Any]]) -> dict[str, Any]:
    paired_offline = _paired_section(folds, "offline", OFFLINE_METRICS)
    paired_closed = _paired_section(folds, "closed_loop", CLOSED_LOOP_METRICS)

    def gate(section: dict[str, Any], metric: str) -> bool | None:
        row = section.get("real_z", {}).get(metric)
        return None if row is None else bool(row["all_folds_better"])

    def arm_gate(
        section: str,
        candidate: str,
        baseline: str,
        metric: str,
        *,
        better_when: str,
    ) -> bool | None:
        candidate_values = _fold_metric_means(folds, section, candidate, metric)
        baseline_values = _fold_metric_means(folds, section, baseline, metric)
        if not candidate_values or len(candidate_values) != len(baseline_values):
            return None
        delta = np.asarray(candidate_values) - np.asarray(baseline_values)
        improvement = delta if better_when == "higher" else -delta
        return bool(np.all(improvement > 0.0))

    offline_gate = gate(paired_offline, "episode_strategy_macro_nll")
    shuffled_gate = arm_gate(
        "offline",
        "real_z",
        "shuffled_z",
        "episode_strategy_macro_nll",
        better_when="lower",
    )
    oracle_gate = arm_gate(
        "offline",
        "oracle",
        "no_z",
        "episode_strategy_macro_nll",
        better_when="lower",
    )
    agreement_gate = gate(paired_closed, "expert_action_agreement")
    ade_gate = gate(paired_closed, "predator_ade")
    offline_control_gate = (
        None
        if None in {offline_gate, shuffled_gate, oracle_gate}
        else bool(offline_gate and shuffled_gate and oracle_gate)
    )
    joint_gate = (
        None
        if agreement_gate is None or ade_gate is None
        else bool(offline_control_gate and agreement_gate and ade_gate)
    )
    return {
        "offline": _aggregate_section(folds, "offline", OFFLINE_METRICS),
        "closed_loop": _aggregate_section(folds, "closed_loop", CLOSED_LOOP_METRICS),
        "paired_vs_no_z": {
            "offline": paired_offline,
            "closed_loop": paired_closed,
        },
        "claim_gates": {
            "real_z_offline_nll_better_on_every_fold": offline_gate,
            "real_z_offline_nll_better_than_shuffled_on_every_fold": shuffled_gate,
            "oracle_offline_nll_better_than_no_z_on_every_fold": oracle_gate,
            "offline_controls_pass": offline_control_gate,
            "real_z_closed_loop_agreement_better_on_every_fold": agreement_gate,
            "real_z_closed_loop_ade_better_on_every_fold": ade_gate,
            "real_z_primary_joint_gate": joint_gate,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/bc"))
    parser.add_argument(
        "--dataset-cache", type=Path, default=Path("artifacts/bc/dataset.npz")
    )
    parser.add_argument("--logdir", type=Path, default=Path("logs/MPE_simple_tag_v3"))
    parser.add_argument("--n-eps", type=int, default=200)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--rollout-seed", type=int, default=0)
    parser.add_argument("--ckpt-seeds", type=_parse_ints, default=(0, 1, 2))
    parser.add_argument("--seeds", type=_parse_ints, default=(0, 1, 2))
    parser.add_argument("--encoder-steps", type=int, default=5000)
    parser.add_argument("--bc-steps", type=int, default=4000)
    parser.add_argument("--lat", type=int, default=2)
    parser.add_argument("--hid", type=int, default=32)
    parser.add_argument("--ctx", type=int, default=10)
    parser.add_argument("--closed-loop-eps", type=int, default=0)
    parser.add_argument("--skip-closed-loop", action="store_true")
    parser.add_argument("--require-clean", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_git_sha = git_sha(_ROOT)
    run_git_dirty = git_dirty(_ROOT)
    if args.require_clean and run_git_dirty is not False:
        raise RuntimeError("--require-clean needs a clean Git working tree")
    if len(args.ckpt_seeds) != 3 or len(set(args.ckpt_seeds)) != 3:
        raise ValueError("--ckpt-seeds must contain three distinct seeds")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must be non-empty and distinct")
    if args.lat < 1 or args.lat > CONDITION_DIM:
        raise ValueError(f"--lat must be in [1, {CONDITION_DIM}]")
    if min(args.n_eps, args.num_steps, args.encoder_steps, args.bc_steps, args.ctx) < 1:
        raise ValueError(
            "episode, horizon, context, and training budgets must be positive"
        )
    if args.ctx >= args.num_steps:
        raise ValueError("--ctx must be smaller than --num-steps")
    if args.closed_loop_eps < 0:
        raise ValueError("--closed-loop-eps must be nonnegative")

    checkpoints = _checkpoint_manifest(args.logdir, args.ckpt_seeds)
    data, dataset_source, dataset_manifest_path = _load_or_create_dataset(
        args,
        run_git_sha=run_git_sha,
        run_git_dirty=run_git_dirty,
        checkpoints=checkpoints,
    )
    _validate_dataset(
        data,
        checkpoint_seeds=args.ckpt_seeds,
        n_eps=args.n_eps,
        num_steps=args.num_steps,
        rollout_seed=args.rollout_seed,
    )
    print(
        f"BC dataset: {len(data['label'])} episodes, "
        f"{data['pred_obs'].shape[-1]}-D exact observations ({dataset_source})"
    )
    dataset_hash = file_sha256(args.dataset_cache)
    dataset_manifest_hash = file_sha256(dataset_manifest_path)
    observations, actions, episodes, timesteps, _ = build_observation_samples_with_time(
        data
    )
    labels = np.asarray(data["label"], dtype=np.int32)
    checkpoint_ids = np.asarray(data["ckpt_seed"], dtype=np.int32)
    lengths = np.asarray(data["valid_length"], dtype=np.int32)
    sequence_raw = predator_sequence_features(
        np.asarray(data["prey_pos"]),
        np.asarray(data["pred_pos"]),
        lengths,
        velocity_mode="legacy_forward",
    )

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    folds: list[dict[str, Any]] = []
    for heldout in args.ckpt_seeds:
        print(f"BC fold: held-out checkpoint {heldout}")
        validation_episodes = checkpoint_ids == heldout
        train_episodes = np.flatnonzero(~validation_episodes)
        scaled_sequence, sequence_mean, sequence_std = _scale_sequence_train_only(
            sequence_raw, lengths, train_episodes
        )
        sample_validation = validation_episodes[episodes]
        fold_result: dict[str, Any] = {
            "heldout_checkpoint": int(heldout),
            "n_train_episodes": int((~validation_episodes).sum()),
            "n_validation_episodes": int(validation_episodes.sum()),
            "seeds": [],
        }
        for seed in args.seeds:
            print(f"  paired encoder/BC seed {seed}")
            _, encoder_params, _ = train_jepa_gru_with_params(
                scaled_sequence[train_episodes],
                lengths[train_episodes],
                jax.random.PRNGKey(seed),
                lat=args.lat,
                hid=args.hid,
                steps=args.encoder_steps,
            )
            prefix_latents = encode_jepa_gru(
                encoder_params,
                scaled_sequence,
                lengths,
                lat=args.lat,
                hid=args.hid,
            )
            episode_permutation = split_local_derangement(
                validation_episodes,
                10_000 + 100 * int(heldout) + seed,
                checkpoint_ids,
            )
            conditions = conditioning_arms(
                prefix_latents,
                labels,
                episodes,
                timesteps,
                episode_permutation,
            )
            seed_dir = args.artifact_dir / f"fold_{heldout}" / f"seed_{seed}"
            encoder_path = seed_dir / "encoder.npz"
            _save_tree(
                encoder_path,
                encoder_params,
                sequence_mean=sequence_mean,
                sequence_std=sequence_std,
            )
            encoder_hash = file_sha256(encoder_path)
            seed_result: dict[str, Any] = {
                "seed": int(seed),
                "offline": {},
                "closed_loop": {},
                "shuffle_label_contingency": {
                    "train": _shuffle_contingency(
                        labels, episode_permutation, ~validation_episodes
                    ),
                    "validation": _shuffle_contingency(
                        labels, episode_permutation, validation_episodes
                    ),
                },
                "encoder_artifact": {
                    "path": str(encoder_path),
                    "sha256": encoder_hash,
                },
                "policy_artifacts": {},
            }
            for arm in ARMS:
                features = np.concatenate([observations, conditions[arm]], axis=-1)
                metadata = {
                    "arm": arm,
                    "heldout_checkpoint": int(heldout),
                    "seed": int(seed),
                    "dataset_sha256": dataset_hash,
                    "dataset_manifest_sha256": dataset_manifest_hash,
                    "observation_dim": int(observations.shape[1]),
                    "conditioning_dim": CONDITION_DIM,
                    "action_classes": list(ACTION_CLASSES),
                    "encoder": "gru_jepa" if arm in {"real_z", "shuffled_z"} else None,
                    "encoder_sha256": (
                        encoder_hash if arm in {"real_z", "shuffled_z"} else None
                    ),
                    "encoder_steps": (
                        args.encoder_steps if arm in {"real_z", "shuffled_z"} else None
                    ),
                    "latent_dim": (
                        args.lat if arm in {"real_z", "shuffled_z"} else None
                    ),
                    "encoder_hidden_dim": (
                        args.hid if arm in {"real_z", "shuffled_z"} else None
                    ),
                    "causal_timing": "z_from_completed_transition_t_minus_1",
                }
                policy = fit_bc(
                    features[~sample_validation],
                    actions[~sample_validation],
                    seed,
                    steps=args.bc_steps,
                    n_actions=len(ACTION_CLASSES),
                    metadata=metadata,
                )
                policy_path = seed_dir / f"{arm}.npz"
                policy.save(policy_path)
                seed_result["policy_artifacts"][arm] = {
                    "path": str(policy_path),
                    "sha256": file_sha256(policy_path),
                }
                reloaded = BCPolicy.load(policy_path)
                metrics = evaluate_bc(
                    reloaded,
                    features[sample_validation],
                    actions[sample_validation],
                    episode_ids=episodes[sample_validation],
                    strategy_labels=labels[episodes[sample_validation]],
                )
                seed_result["offline"][arm] = _jsonable(metrics)
                if not args.skip_closed_loop:
                    seed_result["closed_loop"][arm] = closed_loop_continuation(
                        reloaded,
                        data,
                        arm=arm,
                        heldout_checkpoint=int(heldout),
                        encoder_params=encoder_params,
                        sequence_mean=sequence_mean,
                        sequence_std=sequence_std,
                        latent_dim=args.lat,
                        hidden_dim=args.hid,
                        ctx=args.ctx,
                        rollout_seed=args.rollout_seed,
                        requested_episodes=args.closed_loop_eps,
                        shuffle_seed=20_000 + 100 * int(heldout) + seed,
                        logdir=args.logdir,
                    )
            fold_result["seeds"].append(seed_result)
        folds.append(fold_result)

    results = {
        "schema_version": 1,
        "git_sha": run_git_sha,
        "git_dirty": run_git_dirty,
        "dependencies": package_versions(),
        "checkpoints": checkpoints,
        "dataset": {
            "path": str(args.dataset_cache),
            "source": dataset_source,
            "sha256": dataset_hash,
            "manifest_path": str(dataset_manifest_path),
            "manifest_sha256": dataset_manifest_hash,
            "n_episodes": int(len(labels)),
            "observation_dim": int(observations.shape[1]),
            "n_action_samples": int(len(actions)),
        },
        "config": {
            "checkpoint_seeds": list(args.ckpt_seeds),
            "seeds": list(args.seeds),
            "n_eps_per_strategy_checkpoint": args.n_eps,
            "num_steps": args.num_steps,
            "rollout_seed": args.rollout_seed,
            "encoder_steps": args.encoder_steps,
            "bc_steps": args.bc_steps,
            "latent_dim": args.lat,
            "hidden_dim": args.hid,
            "conditioning_dim": CONDITION_DIM,
            "ctx": args.ctx,
            "closed_loop_requested_episodes_per_strategy": None
            if args.skip_closed_loop
            else args.closed_loop_eps,
            "closed_loop_zero_means": "all_matched_episodes_surviving_ctx",
            "require_clean": args.require_clean,
            "arms": list(ARMS),
        },
        "protocol": {
            "split": "leave_one_checkpoint_out",
            "prey_policy": "fixed_capture_family_same_checkpoint_seed",
            "primary_metric": "episode_then_strategy_macro_action_nll",
            "action_selection": "greedy_argmax",
            "shuffled_z": "split_and_checkpoint_local_episode_derangement",
            "closed_loop": "exact_expert_prefix_then_bc_continuation",
            "closed_loop_selection": "matched_reset_groups_surviving_ctx",
        },
        "summary": _summaries(folds),
        "folds": folds,
    }
    output_path = args.artifact_dir / "results.json"
    output_path.write_text(
        json.dumps(_jsonable(results), indent=2, sort_keys=True) + "\n"
    )
    print(f"Wrote BC results to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
