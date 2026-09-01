#!/usr/bin/env python3
"""Run Part 1 and write a versioned JSON manifest.

The default is a smoke run. ``--synthetic`` creates a deterministic
three-strategy dataset and never reads MAPPO/JaxMARL checkpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

BC_CONDITION_DIM = 3


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logdir", type=Path, default=Path("logs") / "MPE_simple_tag_v3"
    )
    parser.add_argument("--n-eps", type=int, default=16)
    parser.add_argument("--rollout-seed", type=int, default=0)
    parser.add_argument(
        "--prey-objective",
        choices=("capture", "risk", "curious", "matched"),
        default="capture",
        help="Use one fixed prey checkpoint family; 'matched' is a confounded control.",
    )
    parser.add_argument("--ckpt-seeds", type=str, default="0,1,2")
    parser.add_argument("--encoder-seeds", type=str, default="0,1,2")
    parser.add_argument("--bc-seeds", type=str, default="0,1,2")
    parser.add_argument("--lat", type=int, default=2)
    parser.add_argument("--ctx", type=int, default=10)
    parser.add_argument("--prefixes", type=str, default="5,10,25,50,100")
    parser.add_argument("--val-frac", type=float, default=0.34)
    parser.add_argument(
        "--split", choices=("checkpoint", "episode"), default="checkpoint"
    )
    parser.add_argument("--encoder-steps", type=int, default=200)
    parser.add_argument(
        "--action-decoder-steps",
        type=int,
        default=None,
        help="0s updates; defaults to --encoder-steps (legacy report used 1500).",
    )
    parser.add_argument("--action-decoder-window", type=int, default=8)
    parser.add_argument("--action-decoder-lat", type=int, default=8)
    parser.add_argument("--action-decoder-hid", type=int, default=64)
    parser.add_argument(
        "--action-decoder-state-mode",
        choices=("causal_past", "legacy_forward"),
        default="causal_past",
        help=(
            "Use past displacement by default. legacy_forward reproduces "
            "Shashank's action-consequence feature and is replication-only."
        ),
    )
    parser.add_argument("--bc-steps", type=int, default=200)
    parser.add_argument("--hid", type=int, default=32)
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument(
        "--run-kind", choices=("smoke", "full"), default="smoke"
    )
    parser.add_argument("--out", type=Path, default=Path("artifacts/part1_run.json"))
    parser.add_argument("--dataset-cache", type=Path, default=None)
    parser.add_argument("--skip-bc", action="store_true")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run a deterministic infrastructure smoke without checkpoints.",
    )
    parser.add_argument(
        "--dry-run-checks",
        action="store_true",
        help="Validate checkpoint files, write a dry manifest, and exit.",
    )
    return parser


def _params_path(logdir: Path, objective: str, team: str, seed: int) -> Path:
    alg = f"mappo_objectives_{objective}"
    return logdir / f"{alg}_MPE_simple_tag_v3_{team}_actor_seed0_vmap{seed}.safetensors"


def _checkpoint_refs(
    logdir: Path,
    seeds: tuple[int, ...],
    prey_objective: str,
) -> list[dict[str, Any]]:
    from mopa.manifest import file_sha256

    refs: list[dict[str, Any]] = []
    requested = [
        (objective, "pred", seed)
        for objective in ("capture", "risk", "curious")
        for seed in seeds
    ]
    prey_objectives = (
        ("capture", "risk", "curious")
        if prey_objective == "matched"
        else (prey_objective,)
    )
    requested.extend(
        (objective, "prey", seed)
        for objective in prey_objectives
        for seed in seeds
    )
    for objective, team, seed in requested:
        path = _params_path(logdir, objective, team, seed)
        ref: dict[str, Any] = {
            "alg": f"mappo_objectives_{objective}",
            "team": team,
            "seed": seed,
            "path": str(path),
            "exists": path.is_file(),
        }
        if path.is_file():
            ref["sha256"] = file_sha256(path)
        refs.append(ref)
    return refs


def _synthetic_objective_dataset(
    n_eps: int,
    checkpoint_seeds: tuple[int, ...],
    num_steps: int,
):
    """Build small deterministic trajectories with three visible strategies."""
    from mopa.types import ObjectiveObservationDataset

    rows: dict[str, list[np.ndarray]] = {
        name: [] for name in ObjectiveObservationDataset.__dataclass_fields__
    }
    directions = np.asarray([[1.0, 0.2], [-0.2, 1.0], [-0.8, -0.6]])
    offsets = np.asarray([[0.45, -0.20], [-0.20, 0.45], [-0.40, -0.35]])
    time = np.arange(num_steps + 1, dtype=np.float32)

    for label in range(3):
        for checkpoint_seed in checkpoint_seeds:
            for episode in range(n_eps):
                phase = 0.17 * episode + 0.11 * checkpoint_seed
                base = np.asarray(
                    [-0.75 + 0.70 * label, -0.45 + 0.35 * label],
                    dtype=np.float32,
                )
                wave = 0.035 * np.stack(
                    [np.sin(0.35 * time + phase), np.cos(0.27 * time + phase)],
                    axis=-1,
                )
                shift = np.asarray(
                    [0.006 * episode, -0.004 * episode + 0.01 * checkpoint_seed],
                    dtype=np.float32,
                )
                prey = (
                    base + 0.025 * time[:, None] * directions[label] + wave + shift
                ).astype(np.float32)
                pred = (
                    prey
                    + offsets[label]
                    + 0.02
                    * np.stack(
                        [np.cos(0.19 * time + phase), np.sin(0.23 * time + phase)],
                        axis=-1,
                    )
                ).astype(np.float32)[:, None, :]

                rows["prey_pos"].append(prey[None])
                rows["pred_pos"].append(pred[None])
                prey_velocity = np.zeros_like(prey)
                pred_velocity = np.zeros_like(pred[:, 0])
                prey_velocity[1:] = prey[1:] - prey[:-1]
                pred_velocity[1:] = pred[1:, 0] - pred[:-1, 0]
                pred_obs = np.concatenate(
                    [pred[:, 0], prey, pred_velocity, prey_velocity], axis=-1
                )[:, None, :]
                rows["pred_obs"].append(pred_obs[None].astype(np.float32))
                rows["lava_pos"].append(
                    np.asarray([[[1.5, -1.5]]], dtype=np.float32)
                )
                rows["lava_rad"].append(np.asarray([[0.2]], dtype=np.float32))
                rows["prey_act"].append(
                    np.full((1, num_steps), (label + 1) % 3, dtype=np.int32)
                )
                rows["pred_act"].append(
                    np.full((1, num_steps, 1), label, dtype=np.int32)
                )
                rows["capture_t"].append(np.asarray([-1], dtype=np.int32))
                rows["captured"].append(np.asarray([0], dtype=np.int32))
                rows["survival_time"].append(
                    np.asarray([num_steps], dtype=np.float32)
                )
                rows["pred_lava_steps"].append(np.asarray([label], dtype=np.float32))
                rows["prey_lava_steps"].append(np.asarray([0.0], dtype=np.float32))
                rows["resources_collected"].append(
                    np.asarray([label == 2], dtype=np.float32)
                )
                rows["pred_coverage"].append(
                    np.asarray([0.2 + 0.2 * label], dtype=np.float32)
                )
                rows["label"].append(np.asarray([label], dtype=np.int32))
                rows["ckpt_seed"].append(
                    np.asarray([checkpoint_seed], dtype=np.int32)
                )
                rows["env_seed"].append(
                    np.asarray(
                        [[checkpoint_seed, episode]],
                        dtype=np.uint32,
                    )
                )
                rows["valid_length"].append(
                    np.asarray([num_steps], dtype=np.int32)
                )

    return ObjectiveObservationDataset(
        **{name: np.concatenate(values, axis=0) for name, values in rows.items()}
    )


def _load_or_create_dataset(
    args: argparse.Namespace,
    checkpoint_seeds: tuple[int, ...],
    prefixes: tuple[int, ...],
):
    from mopa.types import ObjectiveDataset, ObjectiveObservationDataset

    if args.dataset_cache is not None and args.dataset_cache.is_file():
        with np.load(args.dataset_cache, allow_pickle=False) as raw:
            values = {
                name: raw[name] for name in ObjectiveDataset.__dataclass_fields__
            }
            if "pred_obs" in raw.files:
                values["pred_obs"] = raw["pred_obs"]
        dataset_type = (
            ObjectiveObservationDataset if "pred_obs" in values else ObjectiveDataset
        )
        return dataset_type(**values), "cache"

    if args.synthetic:
        horizon = max(8, 2 * args.ctx, max(prefixes))
        dataset = _synthetic_objective_dataset(
            args.n_eps, checkpoint_seeds, horizon
        )
        source = "synthetic"
    else:
        from mopa.data import objective_dataset

        dataset = objective_dataset(
            n_eps=args.n_eps,
            ckpt_seeds=checkpoint_seeds,
            rng0=args.rollout_seed,
            logdir=args.logdir,
            prey_type=(
                None if args.prey_objective == "matched" else args.prey_objective
            ),
        )
        source = "mappo_checkpoints"

    if args.dataset_cache is not None:
        args.dataset_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.dataset_cache, **dataset.as_dict())
    return dataset, source


def _validate_args(
    args: argparse.Namespace,
    checkpoint_seeds: tuple[int, ...],
    encoder_seeds: tuple[int, ...],
    bc_seeds: tuple[int, ...],
    prefixes: tuple[int, ...],
) -> None:
    if args.n_eps < 1 or args.lat < 1 or args.hid < 1 or args.ctx < 1:
        raise ValueError("n-eps, lat, hid, and ctx must be positive")
    if not args.skip_bc and args.lat > BC_CONDITION_DIM:
        raise ValueError(
            f"BC latent size cannot exceed its {BC_CONDITION_DIM}-D condition"
        )
    if (
        args.action_decoder_window < 2
        or args.action_decoder_lat < 1
        or args.action_decoder_hid < 1
    ):
        raise ValueError("0s window must be >= 2 and its latent/hidden sizes positive")
    if args.encoder_steps < 0 or args.bc_steps < 0:
        raise ValueError("training steps must be non-negative")
    if args.action_decoder_steps is not None and args.action_decoder_steps < 0:
        raise ValueError("action-decoder-steps must be non-negative")
    if not 0.0 < args.val_frac < 1.0:
        raise ValueError("val-frac must be between zero and one")
    if args.calibration_bins < 1:
        raise ValueError("calibration-bins must be positive")
    if args.dataset_cache is not None and args.dataset_cache.suffix != ".npz":
        raise ValueError("dataset-cache must use a .npz suffix")
    if not checkpoint_seeds or not encoder_seeds or not prefixes:
        raise ValueError("checkpoint, encoder, and prefix lists must be non-empty")
    if not args.skip_bc and not bc_seeds:
        raise ValueError("bc-seeds must be non-empty unless --skip-bc is used")
    if not args.skip_bc and len(encoder_seeds) != len(bc_seeds):
        raise ValueError(
            "encoder-seeds and bc-seeds must have equal length for paired BC runs"
        )
    if any(prefix < 1 for prefix in prefixes):
        raise ValueError("prefixes must be positive")
    for name, seeds in (
        ("checkpoint", checkpoint_seeds),
        ("encoder", encoder_seeds),
        ("BC", bc_seeds),
    ):
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"{name} seeds must be unique")
    if args.run_kind == "full" and not args.synthetic:
        if args.split != "checkpoint":
            raise ValueError("a full run requires --split checkpoint")
        if args.prey_objective == "matched":
            raise ValueError(
                "a full run requires one fixed --prey-objective to avoid policy-ID leakage"
            )
        if args.action_decoder_state_mode != "causal_past":
            raise ValueError(
                "a full run requires causal_past 0s state features; "
                "legacy_forward is replication-only"
            )
        if args.dataset_cache is not None and args.dataset_cache.exists():
            raise ValueError(
                "a full run must generate fresh, checkpoint-bound rollouts; "
                "dataset-cache may name a new output file only"
            )
        if len(checkpoint_seeds) < 3 or len(encoder_seeds) < 3:
            raise ValueError(
                "a full run requires at least three checkpoint and encoder seeds"
            )
        if args.skip_bc or len(bc_seeds) < 3:
            raise ValueError(
                "a full run requires BC and at least three distinct BC seeds"
            )
        if args.encoder_steps < 1 or args.bc_steps < 1:
            raise ValueError("a full run requires positive encoder and BC steps")
        if args.action_decoder_steps is not None and args.action_decoder_steps < 1:
            raise ValueError("a full run requires positive 0s training steps")


def _summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "runs": [float(value) for value in array],
    }


def _environment_summary(
    dataset: dict[str, np.ndarray],
    episode_mask: np.ndarray,
) -> dict[str, Any]:
    """Summarize required behavior metrics by objective label."""
    names = ("capture", "risk", "curious")
    fields = {
        "capture_rate": "captured",
        "survival_time": "survival_time",
        "resources_collected": "resources_collected",
        "pred_lava_steps": "pred_lava_steps",
        "prey_lava_steps": "prey_lava_steps",
        "pred_coverage": "pred_coverage",
    }
    mask = np.asarray(episode_mask, dtype=bool)
    if mask.shape != np.asarray(dataset["label"]).shape:
        raise ValueError("environment summary mask must align with episodes")
    output: dict[str, Any] = {}
    for label, name in enumerate(names):
        selected = mask & (np.asarray(dataset["label"]) == label)
        if not np.any(selected):
            raise ValueError(f"environment summary has no {name} episodes")
        report: dict[str, Any] = {"n_episodes": int(selected.sum())}
        for metric_name, field_name in fields.items():
            values = np.asarray(dataset[field_name], dtype=np.float64)[selected]
            report[metric_name] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
            }
        output[name] = report
    return output


def _standardize_sequence_train_only(
    sequence: np.ndarray,
    lengths: np.ndarray,
    train_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = np.arange(sequence.shape[1])[None, :] < lengths[:, None]
    train_values = sequence[train_idx][valid[train_idx]]
    mu = train_values.mean(axis=0)
    sd = train_values.std(axis=0) + 1e-6
    scaled = ((sequence - mu) / sd).astype(np.float32)
    scaled *= valid[..., None].astype(np.float32)
    return scaled, mu.astype(np.float32), sd.astype(np.float32)


def _score_latents(
    z: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    *,
    seed: int,
    calibration_bins: int,
) -> dict[str, Any]:
    from mopa.belief import fit_latent_belief
    from mopa.encoders import collapse_diagnostics
    from mopa.metrics import (
        calibration_metrics,
        train_only_metrics,
        train_only_standardize,
    )

    scores = train_only_metrics(
        z[train_idx],
        labels[train_idx],
        z[val_idx],
        labels[val_idx],
        n_classes=3,
        seed=seed,
    )
    z_train, z_query, _, _ = train_only_standardize(
        z[train_idx], z[val_idx]
    )
    belief = fit_latent_belief(z_train, labels[train_idx], z_query)
    class_to_index = {int(label): i for i, label in enumerate(belief.classes)}
    calibration_labels = np.asarray(
        [class_to_index[int(label)] for label in labels[val_idx]], dtype=np.int32
    )
    return {
        "seed": int(seed),
        "probe": float(scores["probe"]),
        "gmm_ari": float(scores["ari"]),
        "collapse_train": collapse_diagnostics(z[train_idx]),
        "collapse_heldout": collapse_diagnostics(z[val_idx]),
        "calibration": calibration_metrics(
            belief.probs, calibration_labels, n_bins=calibration_bins
        ),
    }


def _aggregate_latent_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    calibration_names = ("nll", "brier", "ece", "classwise_ece")
    return {
        "heldout_probe": _summary([run["probe"] for run in runs]),
        "heldout_gmm_ari": _summary([run["gmm_ari"] for run in runs]),
        "collapse": [
            {
                "seed": run["seed"],
                "train": run["collapse_train"],
                "heldout": run["collapse_heldout"],
            }
            for run in runs
        ],
        "calibration": {
            **{
                name: _summary([run["calibration"][name] for run in runs])
                for name in calibration_names
            },
            "reliability": [
                {
                    "seed": run["seed"],
                    "bins": run["calibration"]["reliability"],
                }
                for run in runs
            ],
        },
        "per_seed": [
            {
                "seed": run["seed"],
                "probe": run["probe"],
                "gmm_ari": run["gmm_ari"],
            }
            for run in runs
        ],
    }


def _causal_sample_latents(
    prefix_latents: np.ndarray,
    episode_ids: np.ndarray,
    timesteps: np.ndarray,
) -> np.ndarray:
    """Return the prefix latent available immediately before each action."""
    z = np.asarray(prefix_latents)
    episode_ids = np.asarray(episode_ids, dtype=np.int32)
    timesteps = np.asarray(timesteps, dtype=np.int32)
    if z.ndim != 3 or episode_ids.shape != timesteps.shape:
        raise ValueError("prefix latents and sample provenance do not align")
    if np.any(timesteps < 0) or np.any(timesteps > z.shape[1]):
        raise ValueError("causal BC timestep is outside the latent horizon")
    # Feature t - 1 ends at the state available for action t. Feature t would
    # include the transition caused by the action being predicted.
    result = np.zeros((len(timesteps), z.shape[-1]), dtype=np.float32)
    observed = timesteps > 0
    result[observed] = z[
        episode_ids[observed], timesteps[observed] - 1
    ]
    return result


def _pad_bc_condition(values: np.ndarray) -> np.ndarray:
    condition = np.asarray(values, dtype=np.float32)
    if condition.ndim != 2 or condition.shape[1] > BC_CONDITION_DIM:
        raise ValueError("BC condition has an invalid shape")
    padded = np.zeros((len(condition), BC_CONDITION_DIM), dtype=np.float32)
    padded[:, : condition.shape[1]] = condition
    return padded


def _split_local_episode_derangement(
    validation_episodes: np.ndarray,
    seed: int,
    checkpoint_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Shuffle episodes without crossing a split or checkpoint."""
    validation = np.asarray(validation_episodes, dtype=bool)
    checkpoints = (
        np.zeros(len(validation), dtype=np.int32)
        if checkpoint_ids is None
        else np.asarray(checkpoint_ids, dtype=np.int32)
    )
    if checkpoints.shape != validation.shape:
        raise ValueError("checkpoint IDs must align with the episode split")
    rng = np.random.default_rng(seed)
    permutation = np.arange(len(validation), dtype=np.int32)
    for fold in (False, True):
        for checkpoint in np.unique(checkpoints[validation == fold]):
            episode_ids = np.flatnonzero(
                (validation == fold) & (checkpoints == checkpoint)
            ).astype(np.int32)
            if len(episode_ids) < 2:
                raise ValueError("shuffled-z needs two episodes per split/checkpoint")
            order = rng.permutation(episode_ids)
            permutation[order] = np.roll(order, 1)
    if np.any(permutation == np.arange(len(permutation))):
        raise AssertionError("episode derangement contains a fixed point")
    return permutation


def _bc_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = (
        "episode_strategy_macro_nll",
        "episode_strategy_macro_accuracy",
        "action_balanced_accuracy",
        "episode_macro_nll",
        "episode_macro_accuracy",
        "nll",
        "accuracy",
    )
    return {
        **{
            name: _summary([float(run[name]) for run in runs])
            for name in metric_names
        },
        "per_seed": runs,
    }


def _sequential_mixture_evaluation(
    dataset: dict[str, np.ndarray],
    validation_episodes: np.ndarray,
    num_steps: int,
    calibration_bins: int,
) -> dict[str, Any]:
    from mopa.bc import build_samples_with_time
    from mopa.belief import categorical_entropy
    from mopa.metrics import calibration_metrics
    from mopa.strategy import SequentialOpponentModel, mixture_policy_metrics

    states, actions, episodes, timesteps, predator_ids = build_samples_with_time(
        dataset, ctx=1, t_max=num_steps
    )
    validation = validation_episodes[episodes]
    if not np.any(validation) or not np.any(~validation):
        raise ValueError(
            "sequential evaluation needs valid actions in both train and validation"
        )
    model = SequentialOpponentModel.fit(
        states[~validation],
        actions[~validation],
        dataset["label"][episodes[~validation]],
        action_classes=np.arange(5, dtype=np.int32),
    )

    mixture_probabilities: list[np.ndarray] = []
    oracle_probabilities: list[np.ndarray] = []
    predictive_beliefs: list[np.ndarray] = []
    posterior_beliefs: list[np.ndarray] = []
    strategy_targets: list[np.ndarray] = []
    observed_actions: list[np.ndarray] = []
    class_to_index = {
        int(label): index
        for index, label in enumerate(model.policy.strategy_classes)
    }
    pairs = np.stack([episodes, predator_ids], axis=-1)
    for episode, predator_id in np.unique(pairs[validation], axis=0):
        rows = np.flatnonzero(
            validation & (episodes == episode) & (predator_ids == predator_id)
        )
        rows = rows[np.argsort(timesteps[rows], kind="stable")]
        trace = model.infer(states[rows], actions[rows])
        mixture_probabilities.append(
            model.mixture_probs(states[rows], trace.predictive)
        )
        label_index = class_to_index[int(dataset["label"][episode])]
        one_hot = np.zeros_like(trace.predictive)
        one_hot[:, label_index] = 1.0
        oracle_probabilities.append(model.mixture_probs(states[rows], one_hot))
        predictive_beliefs.append(trace.predictive)
        posterior_beliefs.append(trace.probs)
        strategy_targets.append(
            np.full(len(rows), label_index, dtype=np.int32)
        )
        observed_actions.append(actions[rows])

    probs = np.concatenate(mixture_probabilities, axis=0)
    oracle_probs = np.concatenate(oracle_probabilities, axis=0)
    predictive = np.concatenate(predictive_beliefs, axis=0)
    posterior = np.concatenate(posterior_beliefs, axis=0)
    strategy_target = np.concatenate(strategy_targets, axis=0)
    acts = np.concatenate(observed_actions, axis=0)
    result = mixture_policy_metrics(
        probs, acts, action_classes=model.policy.action_classes
    )
    oracle_result = mixture_policy_metrics(
        oracle_probs, acts, action_classes=model.policy.action_classes
    )

    def belief_report(probabilities: np.ndarray) -> dict[str, Any]:
        report = calibration_metrics(
            probabilities,
            strategy_target,
            n_bins=calibration_bins,
        )
        return {
            "accuracy": float(
                np.mean(np.argmax(probabilities, axis=-1) == strategy_target)
            ),
            "mean_entropy": float(categorical_entropy(probabilities).mean()),
            "calibration": report,
        }

    return {
        "top1": float(result["top1"]),
        "nll": float(result["nll"]),
        "oracle_strategy_top1": float(oracle_result["top1"]),
        "oracle_strategy_nll": float(oracle_result["nll"]),
        "predictive_strategy_belief": belief_report(predictive),
        "posterior_strategy_belief": belief_report(posterior),
        "strategy_classes": [
            int(label) for label in model.policy.strategy_classes
        ],
        "n_actions": int(len(acts)),
        "belief_timing": "predictive_before_observed_action",
        "strategy_labels_at_inference": False,
        "policy_fit_scope": "training_split_only",
    }


def _dry_run_manifest(
    args: argparse.Namespace,
    config: dict[str, Any],
    checkpoints: list[dict[str, Any]],
) -> tuple[dict[str, Any], int]:
    from mopa.manifest import build_manifest

    missing = [checkpoint for checkpoint in checkpoints if not checkpoint["exists"]]
    checks_ok = args.synthetic or not missing
    manifest = build_manifest(
        config=config,
        checkpoints=checkpoints,
        metrics={
            "dry_run": True,
            "checkpoints_ok": checks_ok,
            "missing_count": len(missing),
        },
        split={"mode": args.split, "materialized": False},
        stages={
            "checkpoint_validation": {
                "status": "smoke_passed" if checks_ok else "blocked",
                "checks_passed": checks_ok,
            },
            "data": {"status": "not_run"},
            "environment_evaluation": {"status": "not_run"},
            "representations": {"status": "not_run"},
            "evaluation": {"status": "not_run"},
            "behaviour_cloning": {"status": "not_run"},
            "cpl_preferences": {"status": "not_run"},
        },
        run_kind="dry_run",
        cwd=_ROOT,
    )
    return manifest, 0 if checks_ok else 2


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    checkpoint_seeds = _parse_ints(args.ckpt_seeds)
    encoder_seeds = _parse_ints(args.encoder_seeds)
    bc_seeds = _parse_ints(args.bc_seeds)
    prefixes = tuple(sorted(set(_parse_ints(args.prefixes))))
    try:
        _validate_args(args, checkpoint_seeds, encoder_seeds, bc_seeds, prefixes)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2

    action_decoder_steps = (
        args.encoder_steps
        if args.action_decoder_steps is None
        else args.action_decoder_steps
    )

    from mopa.manifest import build_manifest, file_sha256, git_dirty, write_manifest

    # Synthetic data can validate plumbing, but can never make a full claim.
    effective_run_kind = "smoke" if args.synthetic else args.run_kind
    if effective_run_kind == "full" and git_dirty(_ROOT) is not False:
        print(
            "a full run requires a clean Git worktree for exact provenance",
            file=sys.stderr,
        )
        return 2
    config: dict[str, Any] = {
        "n_eps": args.n_eps,
        "rollout_seed": args.rollout_seed,
        "prey_objective": args.prey_objective,
        "ckpt_seeds": list(checkpoint_seeds),
        "encoder_seeds": list(encoder_seeds),
        "bc_seeds": list(bc_seeds),
        "lat": args.lat,
        "ctx": args.ctx,
        "prefixes": list(prefixes),
        "val_frac": args.val_frac,
        "split": args.split,
        "logdir": str(args.logdir),
        "encoder_steps": args.encoder_steps,
        "action_decoder_0s": {
            "strategy_key": "sa_short_seq_action_decoder_vae",
            "source_commit": "7bae281091a96fc83cadad3671bef070bd371675",
            "steps": action_decoder_steps,
            "window": args.action_decoder_window,
            "lat": args.action_decoder_lat,
            "hid": args.action_decoder_hid,
            "batch": 128,
            "beta_max": 1.0,
            "free_bits": 0.2,
            "state_mode": args.action_decoder_state_mode,
            "legacy_reference": {
                "steps": 1500,
                "window": 8,
                "lat": 8,
                "hid": 64,
            },
        },
        "bc_steps": args.bc_steps,
        "hid": args.hid,
        "calibration_bins": args.calibration_bins,
        "skip_bc": args.skip_bc,
        "synthetic": args.synthetic,
        "requested_run_kind": args.run_kind,
        "effective_run_kind": effective_run_kind,
    }
    checkpoints = (
        []
        if args.synthetic
        else _checkpoint_refs(
            args.logdir, checkpoint_seeds, args.prey_objective
        )
    )

    if args.dry_run_checks:
        manifest, return_code = _dry_run_manifest(args, config, checkpoints)
        write_manifest(args.out, manifest)
        print(f"Wrote dry-run manifest to {args.out}")
        return return_code

    missing = [checkpoint for checkpoint in checkpoints if not checkpoint["exists"]]
    if missing:
        print(
            f"Missing {len(missing)} checkpoint files under {args.logdir}",
            file=sys.stderr,
        )
        for checkpoint in missing[:6]:
            print(f"  - {checkpoint['path']}", file=sys.stderr)
        return 2

    import jax

    from mopa.action_decoder import (
        ActionDecoderConfig,
        fit_action_decoder_vae,
        pool_episode_prefix_latents,
    )
    from mopa.encoders import (
        encode_jepa,
        encode_jepa_gru,
        encode_vae,
        gather_prefix_latents,
        train_jepa_gru_with_params,
        train_jepa_with_params,
        train_vae_with_params,
    )
    from mopa.features import (
        episode_lengths,
        length_matched_prefix,
        predator_sequence_features,
        sequence_slice,
        trailing_sequence_slice,
    )
    from mopa.metrics import (
        train_only_metrics,
        train_only_oracle_acc,
        train_only_survival_time_probe_acc,
    )
    from mopa.splits import checkpoint_validation_mask, episode_validation_mask

    dataset, dataset_source = _load_or_create_dataset(
        args, checkpoint_seeds, prefixes
    )
    if not args.skip_bc and not hasattr(dataset, "pred_obs"):
        print(
            "behaviour cloning requires exact pred_obs; regenerate the dataset cache",
            file=sys.stderr,
        )
        return 2
    num_steps = int(dataset.pred_act.shape[1])
    if args.ctx >= num_steps:
        print("ctx must be smaller than the dataset horizon", file=sys.stderr)
        return 2
    labels = np.asarray(dataset.label, dtype=np.int32)
    if set(np.unique(labels).tolist()) != {0, 1, 2}:
        print("Part 1 requires strategy labels 0, 1, and 2", file=sys.stderr)
        return 2

    lengths = episode_lengths(dataset.capture_t, num_steps)
    if not np.array_equal(lengths, np.asarray(dataset.valid_length)):
        print("dataset valid_length does not match capture metadata", file=sys.stderr)
        return 2
    if np.asarray(dataset.env_seed).shape != (len(labels), 2):
        print("dataset env_seed must store one JAX key per episode", file=sys.stderr)
        return 2
    sequence_raw = predator_sequence_features(
        dataset.prey_pos, dataset.pred_pos, lengths
    )
    action_decoder_state_raw = predator_sequence_features(
        dataset.prey_pos,
        dataset.pred_pos,
        lengths,
        velocity_mode=args.action_decoder_state_mode,
    )
    if dataset.pred_act.ndim != 3 or dataset.pred_act.shape[-1] != 1:
        print("0s currently requires exactly one predator", file=sys.stderr)
        return 2
    if args.split == "checkpoint":
        validation_episodes = checkpoint_validation_mask(
            dataset.ckpt_seed, rng_seed=0, val_frac=args.val_frac
        )
    else:
        validation_episodes = episode_validation_mask(
            np.arange(len(labels)), rng_seed=0, val_frac=args.val_frac
        )
    train_idx = np.flatnonzero(~validation_episodes)
    val_idx = np.flatnonzero(validation_episodes)
    if not len(train_idx) or not len(val_idx):
        print("Empty train or validation split", file=sys.stderr)
        return 2
    if set(np.unique(labels[train_idx]).tolist()) != {0, 1, 2}:
        print("Training split must contain all three strategies", file=sys.stderr)
        return 2
    if set(np.unique(labels[val_idx]).tolist()) != {0, 1, 2}:
        print("Validation split must contain all three strategies", file=sys.stderr)
        return 2

    sequence, sequence_mu, sequence_sd = _standardize_sequence_train_only(
        sequence_raw, lengths, train_idx
    )
    window_width = min(args.ctx, max(1, num_steps // 2))
    comparison_lengths = length_matched_prefix(lengths, window_width)
    context_raw = trailing_sequence_slice(
        sequence, comparison_lengths, window_width
    )
    target_raw = sequence_slice(
        sequence, window_width, window_width, lengths=lengths
    )
    jepa_train_idx = train_idx[
        lengths[train_idx] >= 2 * window_width
    ]
    if not len(jepa_train_idx):
        print(
            "fixed-window JEPA needs training episodes with a full future target",
            file=sys.stderr,
        )
        return 2
    if set(np.unique(labels[jepa_train_idx]).tolist()) != {0, 1, 2}:
        print(
            "fixed-window JEPA needs full future targets for all three strategies",
            file=sys.stderr,
        )
        return 2
    common_train = np.concatenate(
        [context_raw[train_idx], target_raw[jepa_train_idx]], axis=0
    )
    window_mu = common_train.mean(axis=0)
    window_sd = common_train.std(axis=0) + 1e-6
    context = ((context_raw - window_mu) / window_sd).astype(np.float32)
    target = ((target_raw - window_mu) / window_sd).astype(np.float32)

    action_decoder_name = "sa_short_seq_action_decoder_vae"
    model_names = (
        "gru_jepa",
        "fixed_window_jepa",
        "beta_vae",
        "random_projection",
        action_decoder_name,
    )
    representation_runs: dict[str, list[dict[str, Any]]] = {
        name: [] for name in model_names
    }
    anytime_runs: dict[str, dict[int, list[dict[str, Any]]]] = {
        name: {prefix: [] for prefix in prefixes} for name in model_names
    }
    oracle_runs: list[float] = []
    gru_prefix_latents: dict[int, np.ndarray] = {}
    action_decoder_accuracy_runs: list[float] = []
    action_decoder_unit_runs: list[dict[str, float | int]] = []
    action_decoder_fit_metadata: dict[str, Any] | None = None
    action_decoder_config = ActionDecoderConfig(
        lat=args.action_decoder_lat,
        hid=args.action_decoder_hid,
        window=args.action_decoder_window,
        steps=action_decoder_steps,
    )

    for seed in encoder_seeds:
        _, gru_params, _ = train_jepa_gru_with_params(
            sequence[train_idx],
            lengths[train_idx],
            jax.random.PRNGKey(seed),
            lat=args.lat,
            hid=args.hid,
            steps=args.encoder_steps,
        )
        gru_all = encode_jepa_gru(
            gru_params,
            sequence,
            lengths,
            lat=args.lat,
            hid=args.hid,
        )
        gru_prefix_latents[seed] = gru_all

        _, jepa_params = train_jepa_with_params(
            context[jepa_train_idx],
            target[jepa_train_idx],
            jax.random.PRNGKey(10_000 + seed),
            lat=args.lat,
            steps=args.encoder_steps,
        )
        _, vae_params = train_vae_with_params(
            context[train_idx],
            jax.random.PRNGKey(20_000 + seed),
            lat=args.lat,
            steps=args.encoder_steps,
        )
        random = np.random.default_rng(30_000 + seed)
        projection = random.normal(
            size=(context.shape[1], args.lat)
        ).astype(np.float32) / np.sqrt(context.shape[1])
        action_decoder_fit = fit_action_decoder_vae(
            action_decoder_state_raw,
            np.asarray(dataset.pred_act[:, :, 0], dtype=np.int32),
            lengths,
            train_idx,
            jax.random.PRNGKey(1_000 + seed),
            config=action_decoder_config,
        )
        action_decoder_accuracy_runs.append(
            action_decoder_fit.decoder_action_accuracy
        )
        action_window_episode = action_decoder_fit.encoding.windows.episode
        train_window = np.isin(action_window_episode, train_idx)
        val_window = np.isin(action_window_episode, val_idx)
        unit_scores = train_only_metrics(
            action_decoder_fit.window_latents[train_window],
            labels[action_window_episode[train_window]],
            action_decoder_fit.window_latents[val_window],
            labels[action_window_episode[val_window]],
            n_classes=3,
            seed=seed,
        )
        action_decoder_unit_runs.append(
            {
                "seed": int(seed),
                "probe": float(unit_scores["probe"]),
                "gmm_ari": float(unit_scores["ari"]),
            }
        )
        if action_decoder_fit_metadata is None:
            action_decoder_fit_metadata = {
                "n_windows": int(len(action_window_episode)),
                "n_train_windows": int(train_window.sum()),
                "n_validation_windows": int(val_window.sum()),
                "state_train_mean": [
                    float(value)
                    for value in action_decoder_fit.encoder.state_mean
                ],
                "state_train_std": [
                    float(value)
                    for value in action_decoder_fit.encoder.state_std
                ],
            }

        full_latents = {
            "gru_jepa": gather_prefix_latents(gru_all, comparison_lengths),
            "fixed_window_jepa": encode_jepa(jepa_params, context, lat=args.lat),
            "beta_vae": encode_vae(vae_params, context, lat=args.lat),
            "random_projection": context @ projection,
            action_decoder_name: action_decoder_fit.episode_latents,
        }
        for name, latent in full_latents.items():
            representation_runs[name].append(
                _score_latents(
                    latent,
                    labels,
                    train_idx,
                    val_idx,
                    seed=seed,
                    calibration_bins=args.calibration_bins,
                )
            )

        oracle_runs.append(
            train_only_oracle_acc(
                context[train_idx],
                labels[train_idx],
                context[val_idx],
                labels[val_idx],
                hidden=max(8, args.hid),
                seed=seed,
            )
        )

        for prefix in prefixes:
            prefix_lengths = length_matched_prefix(lengths, min(prefix, num_steps))
            prefix_context_raw = trailing_sequence_slice(
                sequence,
                prefix_lengths,
                window_width,
            )
            prefix_context = (
                (prefix_context_raw - window_mu) / window_sd
            ).astype(np.float32)
            prefix_latents = {
                "gru_jepa": gather_prefix_latents(gru_all, prefix_lengths),
                "fixed_window_jepa": encode_jepa(
                    jepa_params, prefix_context, lat=args.lat
                ),
                "beta_vae": encode_vae(
                    vae_params, prefix_context, lat=args.lat
                ),
                "random_projection": prefix_context @ projection,
                action_decoder_name: pool_episode_prefix_latents(
                    action_decoder_fit.prefix_latents,
                    action_decoder_fit.encoding.windows,
                    prefix_lengths,
                    n_episodes=len(labels),
                ),
            }
            for name, latent in prefix_latents.items():
                anytime_runs[name][prefix].append(
                    _score_latents(
                        latent,
                        labels,
                        train_idx,
                        val_idx,
                        seed=seed,
                        calibration_bins=args.calibration_bins,
                    )
                )

    representations = {
        name: {
            **_aggregate_latent_runs(runs),
            "training_scope": "train_fold_only",
            "heldout_encoding": "frozen",
            "feature_schema": "predator_sequence_features",
            "comparison_prefix": int(window_width),
        }
        for name, runs in representation_runs.items()
    }
    if action_decoder_fit_metadata is None:
        raise RuntimeError("0s training produced no fit metadata")
    representations[action_decoder_name].update(
        {
            "source_strategy": "0s",
            "source_commit": "7bae281091a96fc83cadad3671bef070bd371675",
            "feature_schema": "standardized_state_plus_current_action_onehot",
            "state_feature_mode": args.action_decoder_state_mode,
            "contains_current_actions": True,
            "contains_future_state": (
                args.action_decoder_state_mode == "legacy_forward"
            ),
            "temporal_scope": "full_valid_episode_post_hoc",
            "comparison_prefix": "full_valid_episode",
            "window_pooling": "equal_weight_mean",
            "window_unit_probe": _summary(
                [run["probe"] for run in action_decoder_unit_runs]
            ),
            "window_unit_gmm_ari": _summary(
                [run["gmm_ari"] for run in action_decoder_unit_runs]
            ),
            "decoder_action_accuracy": _summary(
                action_decoder_accuracy_runs
            ),
            "window_unit_per_seed": action_decoder_unit_runs,
            **action_decoder_fit_metadata,
        }
    )
    representations["supervised_oracle"] = {
        "heldout_accuracy": _summary(oracle_runs),
        "training_scope": "train_fold_only",
        "uses_strategy_labels": True,
        "feature_schema": "predator_sequence_features/fixed_window",
    }
    anytime: dict[str, Any] = {
        "requested_prefixes": list(prefixes),
        "prefix_semantics": "length_matched",
    }
    for name in model_names:
        anytime[name] = {
            str(prefix): {
                **_aggregate_latent_runs(anytime_runs[name][prefix]),
                "effective_steps": int(
                    min(prefix, num_steps)
                    if name in {"gru_jepa", action_decoder_name}
                    else min(prefix, num_steps, window_width)
                ),
            }
            for prefix in prefixes
        }

    survival_probe = train_only_survival_time_probe_acc(
        dataset.survival_time[train_idx],
        labels[train_idx],
        dataset.survival_time[val_idx],
        labels[val_idx],
    )
    metrics: dict[str, Any] = {
        "representations": representations,
        "anytime": anytime,
        "controls": {"survival_time_heldout_probe": float(survival_probe)},
    }

    dataset_dict = dataset.as_dict()
    metrics["environment"] = {
        "all": _environment_summary(
            dataset_dict, np.ones(len(labels), dtype=bool)
        ),
        "heldout": _environment_summary(dataset_dict, validation_episodes),
    }
    metrics["sequential_belief_mixture"] = _sequential_mixture_evaluation(
        dataset_dict,
        validation_episodes,
        num_steps,
        args.calibration_bins,
    )

    if args.skip_bc:
        metrics["bc"] = {"skipped": True}
    else:
        from mopa.bc import (
            build_observation_samples_with_time,
            evaluate_bc,
            fit_bc,
        )

        observations, actions, episodes, timesteps, _ = (
            build_observation_samples_with_time(dataset_dict, t_max=num_steps)
        )
        sample_validation = validation_episodes[episodes]
        sample_labels = labels[episodes]
        arm_runs: dict[str, list[dict[str, Any]]] = {
            name: [] for name in ("no_z", "real_z", "shuffled_z", "oracle")
        }
        for replicate, (encoder_seed, bc_seed) in enumerate(
            zip(encoder_seeds, bc_seeds, strict=True)
        ):
            real_z = _causal_sample_latents(
                gru_prefix_latents[encoder_seed], episodes, timesteps
            )
            shuffle_seed = 10_000 + replicate
            episode_permutation = _split_local_episode_derangement(
                validation_episodes,
                shuffle_seed,
                np.asarray(dataset.ckpt_seed),
            )
            shuffled_z = _causal_sample_latents(
                gru_prefix_latents[encoder_seed],
                episode_permutation[episodes],
                timesteps,
            )
            conditions = {
                "no_z": np.zeros(
                    (len(episodes), BC_CONDITION_DIM), dtype=np.float32
                ),
                "real_z": _pad_bc_condition(real_z),
                "shuffled_z": _pad_bc_condition(shuffled_z),
                "oracle": np.eye(BC_CONDITION_DIM, dtype=np.float32)[
                    sample_labels
                ],
            }
            for arm, condition in conditions.items():
                features = np.concatenate([observations, condition], axis=-1)
                policy = fit_bc(
                    features[~sample_validation],
                    actions[~sample_validation],
                    bc_seed,
                    steps=args.bc_steps,
                    metadata={
                        "arm": arm,
                        "replicate": replicate,
                        "encoder_seed": int(encoder_seed),
                        "bc_seed": int(bc_seed),
                        "condition_dim": BC_CONDITION_DIM,
                    },
                )
                run = evaluate_bc(
                    policy,
                    features[sample_validation],
                    actions[sample_validation],
                    episode_ids=episodes[sample_validation],
                    strategy_labels=sample_labels[sample_validation],
                )
                arm_runs[arm].append(
                    {
                        "replicate": replicate,
                        "encoder_seed": int(encoder_seed),
                        "bc_seed": int(bc_seed),
                        "shuffle_seed": (
                            shuffle_seed if arm == "shuffled_z" else None
                        ),
                        "n_train": int((~sample_validation).sum()),
                        "n_val": int(sample_validation.sum()),
                        **run,
                    }
                )

        split_digest = hashlib.sha256(
            np.asarray(sample_validation, dtype=np.uint8).tobytes()
        ).hexdigest()
        metrics["bc"] = {
            "no_z": {
                **_bc_summary(arm_runs["no_z"]),
                "conditioning": "three_zeros",
            },
            "real_z": {
                **_bc_summary(arm_runs["real_z"]),
                "conditioning": "gru_prefix_latent_at_t_minus_1",
            },
            "shuffled_z": {
                **_bc_summary(arm_runs["shuffled_z"]),
                "conditioning": "split_local_deranged_gru_prefix_latent",
            },
            "oracle": {
                **_bc_summary(arm_runs["oracle"]),
                "conditioning": "strategy_one_hot",
                "uses_strategy_labels": True,
            },
            "condition_dim": BC_CONDITION_DIM,
            "feature_schema": "exact_predator_observation_plus_condition",
            "contains_future_episode_information": False,
            "initial_latent": "zero_at_t0",
            "replicate_pairing": "encoder_seed_and_bc_seed_by_position",
            "shared_validation_mask_sha256": split_digest,
            "split_source": "manifest_episode_split_broadcast_to_samples",
        }

    split_info = {
        "mode": args.split,
        "seed": 0,
        "val_frac": args.val_frac,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "train_ckpt_seeds": sorted(
            int(value) for value in np.unique(dataset.ckpt_seed[train_idx])
        ),
        "val_ckpt_seeds": sorted(
            int(value) for value in np.unique(dataset.ckpt_seed[val_idx])
        ),
        "shared_across_encoder_and_bc_seeds": True,
        "episodes": [
            {
                "index": int(index),
                "fold": "validation" if validation_episodes[index] else "train",
                "strategy_label": int(dataset.label[index]),
                "checkpoint_seed": int(dataset.ckpt_seed[index]),
                "environment_key": [
                    int(value) for value in dataset.env_seed[index]
                ],
                "valid_length": int(dataset.valid_length[index]),
            }
            for index in range(len(labels))
        ],
    }
    status = (
        "full_run_finished"
        if effective_run_kind == "full"
        else "smoke_passed"
    )
    stages = {
        "data": {"status": status, "source": dataset_source},
        "environment_evaluation": {"status": status},
        "representations": {"status": status},
        "anytime_and_calibration": {"status": status},
        "sequential_belief_mixture": {"status": status},
        "behaviour_cloning": {
            "status": "not_run" if args.skip_bc else status
        },
        "cpl_preferences": {
            "status": "not_run",
            "reason": "requires a real planner-generated preference dataset",
        },
    }
    artifacts: list[dict[str, Any]] = []
    if args.dataset_cache is not None and args.dataset_cache.is_file():
        artifacts.append(
            {
                "kind": "dataset_cache",
                "path": str(args.dataset_cache),
                "sha256": file_sha256(args.dataset_cache),
            }
        )
    notes = [
        "Strategy labels are evaluation targets and supervised-oracle/policy-fit labels; they are not SSL encoder inputs.",
        "All learned encoders fit on the training fold and encode held-out episodes with frozen parameters.",
        "The four BC arms use exact observations, one shared split, and paired encoder/BC seeds.",
        "Causal BC uses zero at t=0 and the prefix latent at t-1 thereafter.",
        "The 0s headline representation mean-pools every valid action window and is post-hoc; it is not used by causal BC.",
        "The sequential mixture scores each action with the predictive belief before that action is observed.",
        "A dry or smoke run is infrastructure evidence, not a full experiment result.",
        "full_run_finished records execution only; scientific adequacy still depends on the declared protocol and success criteria.",
    ]
    if args.synthetic:
        notes.append(
            "Synthetic data are a deterministic infrastructure smoke test and cannot support empirical claims."
        )
    if args.synthetic and args.run_kind == "full":
        notes.append("Requested full status was downgraded to smoke for synthetic data.")
    if args.action_decoder_state_mode == "legacy_forward":
        notes.append(
            "0s legacy_forward mode exposes the transition caused by the current action and is replication-only."
        )

    manifest = build_manifest(
        config={
            **config,
            "num_steps": num_steps,
            "window_width": window_width,
            "comparison_prefix": window_width,
            "window_jepa_train_episodes": int(len(jepa_train_idx)),
            "window_jepa_target_requirement": "valid_length_at_least_2x_window",
            "sequence_feature_dim": int(sequence.shape[-1]),
            "action_decoder_input_dim": int(
                action_decoder_state_raw.shape[-1] + 5
            ),
            "bc_condition_dim": BC_CONDITION_DIM,
            **(
                {
                    "bc_observation_dim": int(dataset.pred_obs.shape[-1]),
                    "bc_input_dim": int(
                        dataset.pred_obs.shape[-1] + BC_CONDITION_DIM
                    ),
                }
                if hasattr(dataset, "pred_obs")
                else {}
            ),
            "sequence_train_mean": [float(value) for value in sequence_mu],
            "sequence_train_std": [float(value) for value in sequence_sd],
        },
        checkpoints=checkpoints,
        metrics=metrics,
        split=split_info,
        stages=stages,
        artifacts=artifacts,
        run_kind=effective_run_kind,
        notes=notes,
        cwd=_ROOT,
    )
    write_manifest(args.out, manifest)
    primary = metrics["representations"]["gru_jepa"]["heldout_probe"]
    print(
        f"GRU-JEPA held-out probe {primary['mean']:.3f} +/- {primary['std']:.3f}"
    )
    print(f"Wrote {effective_run_kind} manifest to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
