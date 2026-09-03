#!/usr/bin/env python3
"""Gate 2: matched four-arm continuous opponent behaviour cloning.

Arms share one ``tanh(MLP)`` trunk trained with mean squared error on exact red
observations plus a three-column condition slot:

- ``no_c``       : zeros;
- ``real_c``     : causal GRU-JEPA context ``c_t = g(H_<t)`` (zero at ``t = 0``);
- ``shuffled_c`` : split- and checkpoint-local episode derangement of ``real_c``;
- ``oracle``     : true objective one-hot.

Leave-one-checkpoint-out folds, paired encoder/BC seeds, held-out offline
error metrics, and a closed-loop continuation that replays the exact expert
prefix and then lets the cloned predator act against the frozen prey.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mopa.bc_continuous import (  # noqa: E402
    ContinuousBCPolicy,
    continuous_bc_metrics,
    fit_continuous_bc,
)
from mopa.context import (  # noqa: E402
    CONTEXT_DIM,
    CausalContextEncoder,
    causal_sample_context,
    derangement,
    split_local_derangement,
    train_context_encoder,
)
from mopa.continuous_data import (  # noqa: E402
    DEFAULT_CONTINUOUS_LOGDIR,
    OBJECTIVE_TYPES,
    continuous_checkpoint_path,
    deterministic_specialist_action,
    load_continuous_actor_params,
    load_continuous_dataset,
)
from mopa.manifest import (  # noqa: E402
    file_sha256,
    git_dirty,
    git_sha,
    package_versions,
)
from tag_objectives import joint_action_dict, make_env  # noqa: E402
from tag_objectives.teams import freeze_tree  # noqa: E402

ARMS = ("no_c", "real_c", "shuffled_c", "oracle")
OFFLINE_METRICS = {
    "episode_strategy_macro_mse": "lower",
    "episode_strategy_macro_direction_cosine": "higher",
    "episode_strategy_macro_agreement": "higher",
    "mae": "lower",
}
CLOSED_LOOP_METRICS = {
    "expert_action_agreement": "higher",
    "predator_ade": "lower",
    "predator_fde": "lower",
    "capture_rate_gap": "lower",
    "survival_time_gap": "lower",
    "pred_lava_steps_gap": "lower",
    "pred_coverage_gap": "lower",
    "resources_collected_gap": "lower",
}


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


# --------------------------------------------------------------------------- #
# Samples and conditioning
# --------------------------------------------------------------------------- #
def build_red_samples(
    data: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exact pre-action red observations, red actions, episode ids, timesteps."""
    obs = np.asarray(data["red_observation"])[:, :-1]
    act = np.asarray(data["red_action"])
    valid = np.asarray(data["valid_mask"], dtype=bool)
    episodes, timesteps = np.nonzero(valid)
    return (
        obs[episodes, timesteps].astype(np.float32),
        act[episodes, timesteps].astype(np.float32),
        episodes.astype(np.int32),
        timesteps.astype(np.int32),
    )


def conditioning_arms(
    prefix_latents: np.ndarray,
    labels: np.ndarray,
    episodes: np.ndarray,
    timesteps: np.ndarray,
    episode_permutation: np.ndarray,
) -> dict[str, np.ndarray]:
    real = causal_sample_context(prefix_latents, episodes, timesteps, CONTEXT_DIM)
    shuffled = causal_sample_context(
        prefix_latents, episode_permutation[episodes], timesteps, CONTEXT_DIM
    )
    oracle = np.eye(CONTEXT_DIM, dtype=np.float32)[labels[episodes]]
    return {
        "no_c": np.zeros((len(episodes), CONTEXT_DIM), dtype=np.float32),
        "real_c": real,
        "shuffled_c": shuffled,
        "oracle": oracle,
    }


def _shuffle_contingency(
    labels: np.ndarray, permutation: np.ndarray, mask: np.ndarray
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int32)
    perm = np.asarray(permutation, dtype=np.int32)
    sel = np.asarray(mask, dtype=bool)
    counts = np.zeros((3, 3), dtype=np.int32)
    np.add.at(counts, (labels[sel], labels[perm[sel]]), 1)
    return {
        "label_order": list(OBJECTIVE_TYPES),
        "counts": counts.tolist(),
        "same_label_fraction": float(np.trace(counts) / max(counts.sum(), 1)),
    }


# --------------------------------------------------------------------------- #
# Closed loop
# --------------------------------------------------------------------------- #
def _mean_gap(actual: np.ndarray, reference: np.ndarray) -> float:
    return float(abs(np.mean(actual, dtype=np.float64) - np.mean(reference, dtype=np.float64)))


def closed_loop_continuation(
    policy: ContinuousBCPolicy,
    data: dict[str, np.ndarray],
    *,
    arm: str,
    heldout_checkpoint: int,
    encoder: CausalContextEncoder,
    ctx: int,
    requested_episodes: int,
    shuffle_seed: int,
    logdir: Path,
    agreement_tolerance: float = 0.25,
) -> dict[str, Any]:
    """Replay the exact expert prefix, then evaluate the cloned red predator."""
    labels = np.asarray(data["objective_label"], dtype=np.int32)
    ckpts = np.asarray(data["checkpoint_seed"], dtype=np.int32)
    valid_length = np.asarray(data["valid_length"], dtype=np.int32)
    groups = [np.flatnonzero((ckpts == heldout_checkpoint) & (labels == lab)) for lab in range(3)]
    group_size = len(groups[0])
    if any(len(g) != group_size for g in groups):
        raise ValueError("closed-loop groups must have equal sizes")
    eligible = np.all(np.stack([valid_length[g] > ctx for g in groups]), axis=0)
    offsets = np.flatnonzero(eligible)
    if requested_episodes:
        offsets = offsets[:requested_episodes]
    if len(offsets) == 0:
        raise ValueError("no matched held-out episodes survive the expert prefix")
    episode_indices = np.concatenate([g[offsets] for g in groups])
    n_per = len(offsets)
    strategy = np.repeat(np.arange(3, dtype=np.int32), n_per)
    batch = len(episode_indices)
    horizon = int(data["red_action"].shape[1])

    env = make_env("capture", continuous=True)
    pred_name, prey_name = env.adversaries[0], env.good_agents[0]
    pred_index, prey_index = env.agents.index(pred_name), env.agents.index(prey_name)
    obs_width = max(env.observation_space(a).shape[0] for a in env.agents)
    pred_params = {
        lab: load_continuous_actor_params(
            continuous_checkpoint_path(logdir, objective, "pred", heldout_checkpoint)
        )
        for lab, objective in enumerate(OBJECTIVE_TYPES)
    }
    prey_params = load_continuous_actor_params(
        continuous_checkpoint_path(logdir, "capture", "prey", heldout_checkpoint)
    )
    act_fn = jax.jit(lambda p, o: deterministic_specialist_action(p, o, obs_width))
    bc_act = jax.jit(policy.act)
    step_fn = jax.jit(jax.vmap(env.step_env))

    reset_keys = jnp.asarray(data["environment_seed"][episode_indices], dtype=jnp.uint32)
    step_seed = jnp.asarray(data["step_seed"][episode_indices], dtype=jnp.uint32)
    obs, state = jax.vmap(env.reset)(reset_keys)
    done = np.zeros(batch, dtype=bool)
    pred_lava = np.zeros(batch, dtype=np.float32)
    prey_hist = [np.asarray(state.p_pos[:, prey_index])]
    pred_hist = [np.asarray(state.p_pos[:, pred_index : pred_index + 1])]
    agree_sum = np.zeros(batch, dtype=np.float64)
    agree_cnt = np.zeros(batch, dtype=np.int32)
    closed_shuffle = derangement(np.arange(batch, dtype=np.int32), np.random.default_rng(shuffle_seed))

    for t in range(horizon):
        active = ~done
        red_obs = np.asarray(obs[pred_name], dtype=np.float32)
        if t <= ctx:
            np.testing.assert_allclose(
                red_obs, np.asarray(data["red_observation"])[episode_indices, t], atol=1e-6
            )
        expert_parts = [
            np.asarray(act_fn(pred_params[lab], obs[pred_name][lab * n_per : (lab + 1) * n_per]))
            for lab in range(3)
        ]
        expert_red = np.concatenate(expert_parts)
        blue = np.asarray(act_fn(prey_params, obs[prey_name]))
        if t < ctx:
            red = np.asarray(data["red_action"])[episode_indices, t]
            blue = np.asarray(data["blue_action"])[episode_indices, t]
            # The stored prefix is replayed verbatim; the recomputed expert action
            # must agree up to float32 batch-shape effects (different XLA tiling).
            gap = float(np.max(np.abs(red[active] - expert_red[active]), initial=0.0))
            if gap > 1e-4:
                raise ValueError(f"expert red prefix does not replay (max gap {gap:.2e})")
        else:
            if arm == "oracle":
                cond = np.eye(CONTEXT_DIM, dtype=np.float32)[strategy]
            elif arm == "no_c":
                cond = np.zeros((batch, CONTEXT_DIM), dtype=np.float32)
            else:
                cond = encoder.online_context(
                    prey_hist, pred_hist, done, np.asarray(state.capture_t), horizon=horizon
                )
                if arm == "shuffled_c":
                    cond = cond[closed_shuffle]
            red = np.asarray(bc_act(jnp.asarray(np.concatenate([red_obs, cond], -1))))
            err = np.linalg.norm(red - expert_red, axis=-1)
            agree_sum[active] += (err[active] <= agreement_tolerance)
            agree_cnt[active] += 1
        keys = jax.vmap(lambda k: jax.random.fold_in(k, t))(step_seed)
        actions = joint_action_dict(env, jnp.asarray(blue), jnp.asarray(red)[:, None, :])
        new_obs, new_state, _, dones, info = step_fn(keys, state, actions)
        pred_lava += np.asarray(info["pred_lava"][:, pred_index]) * active
        active_j = jnp.asarray(active)
        state = freeze_tree(active_j, new_state, state)
        obs = freeze_tree(active_j, new_obs, obs)
        done = done | np.asarray(dones["__all__"])
        prey_hist.append(np.asarray(state.p_pos[:, prey_index]))
        pred_hist.append(np.asarray(state.p_pos[:, pred_index : pred_index + 1]))
        if t < ctx:
            np.testing.assert_allclose(
                pred_hist[-1], np.asarray(data["pred_pos"])[episode_indices, t + 1], atol=1e-6
            )

    capture_t = np.asarray(state.capture_t)
    captured = capture_t >= 0
    clone_len = np.where(captured, capture_t, horizon).astype(np.int32)
    clone_pred = np.stack(pred_hist, axis=1)[..., 0, :]
    expert_pred = np.asarray(data["pred_pos"])[episode_indices][..., 0, :]
    common = np.minimum(clone_len, valid_length[episode_indices])
    time = np.arange(horizon + 1)[None, :]
    cont = (time > ctx) & (time <= common[:, None])
    dist = np.linalg.norm(clone_pred - expert_pred, axis=-1)
    ade = np.sum(dist * cont, axis=1) / np.maximum(cont.sum(1), 1)
    fde = dist[np.arange(batch), np.maximum(common, ctx + 1).clip(max=horizon)]
    coverage = np.asarray(state.visited).sum(-1).astype(np.float32)
    resources = np.asarray(state.collected).sum(-1).astype(np.float32)
    survival = np.where(captured, capture_t, horizon).astype(np.float32)
    agreement = agree_sum / np.maximum(agree_cnt, 1)

    per_strategy: dict[str, Any] = {}
    for lab, name in enumerate(OBJECTIVE_TYPES):
        rows = strategy == lab
        src = episode_indices[rows]
        per_strategy[name] = {
            "n_episodes": int(rows.sum()),
            "expert_action_agreement": float(agreement[rows].mean()),
            "predator_ade": float(ade[rows].mean()),
            "predator_fde": float(fde[rows].mean()),
            "capture_rate_gap": _mean_gap(captured[rows], data["captured"][src]),
            "survival_time_gap": _mean_gap(survival[rows], data["survival_time"][src]),
            "pred_lava_steps_gap": _mean_gap(pred_lava[rows], data["pred_lava_steps"][src]),
            "pred_coverage_gap": _mean_gap(coverage[rows], data["pred_coverage"][src]),
            "resources_collected_gap": _mean_gap(
                resources[rows], data["resources_collected"][src]
            ),
            "clone_capture_rate": float(captured[rows].mean()),
            "expert_capture_rate": float(np.mean(data["captured"][src])),
        }
    names = next(iter(per_strategy.values())).keys() - {"n_episodes"}
    macro = {m: float(np.mean([row[m] for row in per_strategy.values()])) for m in names}
    macro["n_continuation_actions"] = int(agree_cnt.sum())
    out = {
        "n_eligible_matched_episodes": int(eligible.sum()),
        "n_episodes_per_strategy": int(n_per),
        "macro": macro,
        "per_strategy": per_strategy,
    }
    if arm == "shuffled_c":
        out["shuffle_label_contingency"] = _shuffle_contingency(
            strategy, closed_shuffle, np.ones(batch, dtype=bool)
        )
    return out


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _fold_metric_means(folds, section, arm, metric) -> list[float]:  # noqa: ANN001
    means = []
    for fold in folds:
        values = []
        for seed in fold["seeds"]:
            row = seed[section].get(arm)
            if row is None:
                continue
            if section == "closed_loop":
                row = row["macro"]
            if metric in row and row[metric] is not None:
                values.append(float(row[metric]))
        if values:
            means.append(float(np.mean(values)))
    return means


def _aggregate_section(folds, section, metrics) -> dict[str, Any]:  # noqa: ANN001
    out: dict[str, Any] = {}
    for arm in ARMS:
        out[arm] = {}
        for metric in metrics:
            fm = _fold_metric_means(folds, section, arm, metric)
            if fm:
                v = np.asarray(fm)
                out[arm][metric] = {"mean": float(v.mean()), "std": float(v.std()), "fold_means": v.tolist()}
    return out


def _paired(folds, section, metrics, candidate, baseline) -> dict[str, Any]:  # noqa: ANN001
    out: dict[str, Any] = {}
    for metric, better in metrics.items():
        b = _fold_metric_means(folds, section, baseline, metric)
        c = _fold_metric_means(folds, section, candidate, metric)
        if not b or len(b) != len(c):
            continue
        delta = np.asarray(c) - np.asarray(b)
        improvement = delta if better == "higher" else -delta
        out[metric] = {
            "better_when": better,
            "delta_mean": float(delta.mean()),
            "delta_std": float(delta.std()),
            "fold_deltas": delta.tolist(),
            "all_folds_better": bool(np.all(improvement > 0.0)),
        }
    return out


def summarize(folds: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = {
        "real_c_vs_no_c": ("real_c", "no_c"),
        "real_c_vs_shuffled_c": ("real_c", "shuffled_c"),
        "oracle_vs_no_c": ("oracle", "no_c"),
        "shuffled_c_vs_no_c": ("shuffled_c", "no_c"),
    }
    paired = {
        name: {
            "offline": _paired(folds, "offline", OFFLINE_METRICS, c, b),
            "closed_loop": _paired(folds, "closed_loop", CLOSED_LOOP_METRICS, c, b),
        }
        for name, (c, b) in pairs.items()
    }

    def gate(pair: str, section: str, metric: str) -> bool | None:
        row = paired[pair][section].get(metric)
        return None if row is None else bool(row["all_folds_better"])

    g = {
        "real_c_offline_mse_better_than_no_c_on_every_fold": gate(
            "real_c_vs_no_c", "offline", "episode_strategy_macro_mse"
        ),
        "real_c_offline_mse_better_than_shuffled_c_on_every_fold": gate(
            "real_c_vs_shuffled_c", "offline", "episode_strategy_macro_mse"
        ),
        "oracle_offline_mse_better_than_no_c_on_every_fold": gate(
            "oracle_vs_no_c", "offline", "episode_strategy_macro_mse"
        ),
        "real_c_closed_loop_ade_better_than_no_c_on_every_fold": gate(
            "real_c_vs_no_c", "closed_loop", "predator_ade"
        ),
        "real_c_closed_loop_ade_better_than_shuffled_c_on_every_fold": gate(
            "real_c_vs_shuffled_c", "closed_loop", "predator_ade"
        ),
        "real_c_closed_loop_agreement_better_than_no_c_on_every_fold": gate(
            "real_c_vs_no_c", "closed_loop", "expert_action_agreement"
        ),
        "oracle_closed_loop_ade_better_than_no_c_on_every_fold": gate(
            "oracle_vs_no_c", "closed_loop", "predator_ade"
        ),
    }
    offline_gate = None if None in {
        g["real_c_offline_mse_better_than_no_c_on_every_fold"],
        g["real_c_offline_mse_better_than_shuffled_c_on_every_fold"],
    } else bool(
        g["real_c_offline_mse_better_than_no_c_on_every_fold"]
        and g["real_c_offline_mse_better_than_shuffled_c_on_every_fold"]
    )
    closed_gate = None if None in {
        g["real_c_closed_loop_ade_better_than_no_c_on_every_fold"],
        g["real_c_closed_loop_ade_better_than_shuffled_c_on_every_fold"],
    } else bool(
        g["real_c_closed_loop_ade_better_than_no_c_on_every_fold"]
        and g["real_c_closed_loop_ade_better_than_shuffled_c_on_every_fold"]
    )
    oracle_gate = g["oracle_offline_mse_better_than_no_c_on_every_fold"]
    g["gate2_real_c_offline_controls_pass"] = offline_gate
    g["gate2_real_c_closed_loop_controls_pass"] = closed_gate
    g["gate2_oracle_type_information_is_action_relevant"] = oracle_gate
    g["gate2_pass"] = (
        None
        if None in {offline_gate, closed_gate, oracle_gate}
        else bool(offline_gate and closed_gate and oracle_gate)
    )
    return {
        "offline": _aggregate_section(folds, "offline", OFFLINE_METRICS),
        "closed_loop": _aggregate_section(folds, "closed_loop", CLOSED_LOOP_METRICS),
        "paired": paired,
        "claim_gates": g,
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-dir", type=Path, default=Path("artifacts/bc_continuous"))
    p.add_argument("--dataset", type=Path, default=Path("artifacts/continuous/dataset.npz"))
    p.add_argument("--logdir", type=Path, default=DEFAULT_CONTINUOUS_LOGDIR)
    p.add_argument("--seeds", type=_parse_ints, default=(0, 1, 2))
    p.add_argument("--encoder-steps", type=int, default=5000)
    p.add_argument("--bc-steps", type=int, default=4000)
    p.add_argument("--lat", type=int, default=2)
    p.add_argument("--hid", type=int, default=32)
    p.add_argument("--ctx", type=int, default=10)
    p.add_argument("--closed-loop-eps", type=int, default=0, help="0 = all eligible")
    p.add_argument("--skip-closed-loop", action="store_true")
    p.add_argument("--require-clean", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_sha, run_dirty = git_sha(_ROOT), git_dirty(_ROOT)
    if args.require_clean and run_dirty is not False:
        raise RuntimeError("--require-clean needs a clean Git working tree")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must be non-empty and distinct")
    if not 1 <= args.lat <= CONTEXT_DIM:
        raise ValueError(f"--lat must be in [1, {CONTEXT_DIM}]")
    if min(args.encoder_steps, args.bc_steps, args.ctx) < 1:
        raise ValueError("training budgets and ctx must be positive")

    ds = load_continuous_dataset(args.dataset)
    data = ds.as_dict()
    manifest_path = args.dataset.with_name("dataset.manifest.json")
    dataset_manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
    if dataset_manifest is None:
        raise FileNotFoundError("dataset manifest is required for provenance")
    checkpoint_hashes = {
        (row["objective"], row["team"], row["seed"]): row["sha256"]
        for row in dataset_manifest["source_checkpoints"]
    }
    for (objective, team, seed), sha in checkpoint_hashes.items():
        path = continuous_checkpoint_path(args.logdir, objective, team, seed)
        if not path.is_file() or file_sha256(path) != sha:
            raise ValueError(f"checkpoint {path} does not match the dataset manifest")
    ckpt_seeds = tuple(sorted(int(s) for s in np.unique(ds.checkpoint_seed)))
    if args.ctx >= int(ds.red_action.shape[1]):
        raise ValueError("--ctx must be smaller than the episode horizon")

    observations, actions, episodes, timesteps = build_red_samples(data)
    labels = np.asarray(ds.objective_label, dtype=np.int32)
    ckpt_ids = np.asarray(ds.checkpoint_seed, dtype=np.int32)
    lengths = np.asarray(ds.valid_length, dtype=np.int32)
    print(
        f"continuous BC dataset: {len(labels)} episodes, {len(actions)} red actions, "
        f"{observations.shape[1]}-D observations, checkpoints {ckpt_seeds}"
    )

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    folds: list[dict[str, Any]] = []
    for heldout in ckpt_seeds:
        print(f"fold: held-out checkpoint {heldout}")
        val_eps = ckpt_ids == heldout
        train_eps = np.flatnonzero(~val_eps)
        sample_val = val_eps[episodes]
        fold: dict[str, Any] = {
            "heldout_checkpoint": int(heldout),
            "n_train_episodes": int((~val_eps).sum()),
            "n_validation_episodes": int(val_eps.sum()),
            "seeds": [],
        }
        for seed in args.seeds:
            print(f"  paired encoder/BC seed {seed}")
            encoder = train_context_encoder(
                ds.prey_pos,
                ds.pred_pos,
                lengths,
                train_eps,
                seed,
                latent_dim=args.lat,
                hidden_dim=args.hid,
                steps=args.encoder_steps,
                metadata={"heldout_checkpoint": int(heldout), "dataset": str(args.dataset)},
            )
            prefix = encoder.prefix_latents(ds.prey_pos, ds.pred_pos, lengths)
            permutation = split_local_derangement(val_eps, 10_000 + 100 * heldout + seed, ckpt_ids)
            conds = conditioning_arms(prefix, labels, episodes, timesteps, permutation)
            seed_dir = args.artifact_dir / f"fold_{heldout}" / f"seed_{seed}"
            enc_path = seed_dir / "context_encoder.npz"
            encoder.save(enc_path)
            seed_result: dict[str, Any] = {
                "seed": int(seed),
                "offline": {},
                "closed_loop": {},
                "shuffle_label_contingency": {
                    "train": _shuffle_contingency(labels, permutation, ~val_eps),
                    "validation": _shuffle_contingency(labels, permutation, val_eps),
                },
                "encoder_artifact": {"path": str(enc_path), "sha256": file_sha256(enc_path)},
                "policy_artifacts": {},
            }
            for arm in ARMS:
                feats = np.concatenate([observations, conds[arm]], axis=-1)
                policy = fit_continuous_bc(
                    feats[~sample_val],
                    actions[~sample_val],
                    seed,
                    steps=args.bc_steps,
                    metadata={
                        "arm": arm,
                        "heldout_checkpoint": int(heldout),
                        "observation_dim": int(observations.shape[1]),
                        "context_dim": CONTEXT_DIM,
                        "encoder_sha256": (
                            seed_result["encoder_artifact"]["sha256"]
                            if arm in {"real_c", "shuffled_c"}
                            else None
                        ),
                        "causal_timing": "c_t_from_transitions_before_t",
                    },
                )
                policy_path = seed_dir / f"{arm}.npz"
                policy.save(policy_path)
                reloaded = ContinuousBCPolicy.load(policy_path)
                seed_result["policy_artifacts"][arm] = {
                    "path": str(policy_path),
                    "sha256": file_sha256(policy_path),
                }
                pred = reloaded.predict(feats[sample_val])
                seed_result["offline"][arm] = continuous_bc_metrics(
                    pred,
                    actions[sample_val],
                    episode_ids=episodes[sample_val],
                    strategy_labels=labels[episodes[sample_val]],
                )
                print(
                    f"    {arm:11s} heldout MSE {seed_result['offline'][arm]['episode_strategy_macro_mse']:.4f}"
                    f"  cos {seed_result['offline'][arm]['episode_strategy_macro_direction_cosine']:.3f}"
                )
                if not args.skip_closed_loop:
                    seed_result["closed_loop"][arm] = closed_loop_continuation(
                        reloaded,
                        data,
                        arm=arm,
                        heldout_checkpoint=int(heldout),
                        encoder=encoder,
                        ctx=args.ctx,
                        requested_episodes=args.closed_loop_eps,
                        shuffle_seed=20_000 + 100 * heldout + seed,
                        logdir=args.logdir,
                    )
                    m = seed_result["closed_loop"][arm]["macro"]
                    print(
                        f"    {arm:11s} closed-loop ADE {m['predator_ade']:.3f} "
                        f"agreement {m['expert_action_agreement']:.3f}"
                    )
            fold["seeds"].append(seed_result)
        folds.append(fold)

    results = {
        "schema_version": 1,
        "git_sha": run_sha,
        "git_dirty": run_dirty,
        "dependencies": package_versions(),
        "dataset": {
            "path": str(args.dataset),
            "sha256": file_sha256(args.dataset),
            "manifest_path": str(manifest_path),
            "manifest_sha256": file_sha256(manifest_path),
            "n_episodes": int(len(labels)),
            "n_red_actions": int(len(actions)),
            "observation_dim": int(observations.shape[1]),
        },
        "config": {
            "checkpoint_seeds": list(ckpt_seeds),
            "seeds": list(args.seeds),
            "encoder_steps": args.encoder_steps,
            "bc_steps": args.bc_steps,
            "latent_dim": args.lat,
            "hidden_dim": args.hid,
            "context_dim": CONTEXT_DIM,
            "ctx": args.ctx,
            "closed_loop_requested_episodes_per_strategy": (
                None if args.skip_closed_loop else args.closed_loop_eps
            ),
            "arms": list(ARMS),
            "require_clean": args.require_clean,
        },
        "protocol": {
            "model": "tanh(MLP(red_observation, condition)); MSE",
            "split": "leave_one_checkpoint_out",
            "prey_policy": "fixed_capture_family_same_checkpoint_seed_deterministic_mean",
            "primary_metric": "episode_then_strategy_macro_mse",
            "shuffled_c": "split_and_checkpoint_local_episode_derangement",
            "closed_loop": "exact_expert_prefix_then_bc_red_vs_frozen_prey",
            "agreement_tolerance_l2": 0.25,
        },
        "summary": summarize(folds),
        "folds": folds,
    }
    out = args.artifact_dir / "results.json"
    out.write_text(json.dumps(_jsonable(results), indent=2, sort_keys=True) + "\n")
    print(json.dumps(_jsonable(results["summary"]["claim_gates"]), indent=1))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
