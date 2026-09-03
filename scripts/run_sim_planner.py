#!/usr/bin/env python3
"""Gate 3: simulator-backed MPPI planner ladder with opponent-context controls.

For every held-out checkpoint family and opponent type, on matched reset keys:

1. fixed-policy control: the frozen MAPPO prey (deterministic mean);
2. random control;
3. planner + true red specialist (rung 1 of the component ladder);
4. planner + oracle-conditioned BC red with correct / zero / wrong one-hot context;
5. planner + causal-context BC red with online / zero / shuffled context;
6. planner + unconditioned BC red.

The imagined dynamics are the exact simulator; the terminal value is zero.
Blue return (sum of prey rewards), capture, survival, resources, and lava are
reported per opponent type with paired deltas against the controls.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mopa.bc_continuous import ContinuousBCPolicy  # noqa: E402
from mopa.context import CausalContextEncoder  # noqa: E402
from mopa.continuous_data import (  # noqa: E402
    DEFAULT_CONTINUOUS_LOGDIR,
    OBJECTIVE_TYPES,
    continuous_checkpoint_path,
    load_continuous_actor_params,
    load_continuous_dataset,
)
from mopa.evaluation import (  # noqa: E402
    mappo_prey_controller,
    random_controller,
    run_matched_episodes,
)
from mopa.manifest import (  # noqa: E402
    file_sha256,
    git_dirty,
    git_sha,
    package_versions,
)
from mopa.sim_planner import (  # noqa: E402
    MPPIConfig,
    jit_planner,
    make_bc_red_policy,
    make_true_red_policy,
)
from tag_objectives import make_env  # noqa: E402

METRICS = (
    "blue_return",
    "captured",
    "survival_time",
    "resources_collected",
    "prey_lava_steps",
    "pred_lava_steps",
)


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(v) for v in value.split(",") if v.strip())


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


def planner_controller(planner, cfg: MPPIConfig):  # noqa: ANN001
    """Wrap a compiled planner as a ``BlueController`` with warm-started plans."""

    def blue(state, obs, context, carry, key, t):  # noqa: ANN001
        del obs, t
        batch = context.shape[0]
        prev_mean = (
            jnp.zeros((batch, cfg.horizon, 2)) if carry is None else carry
        )
        action, (mean, _) = planner(state, context, prev_mean, key)
        return action, mean

    return blue


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-dir", type=Path, default=Path("artifacts/sim_planner"))
    p.add_argument("--dataset", type=Path, default=Path("artifacts/continuous/dataset.npz"))
    p.add_argument("--bc-artifacts", type=Path, default=Path("artifacts/bc_continuous"))
    p.add_argument("--logdir", type=Path, default=DEFAULT_CONTINUOUS_LOGDIR)
    p.add_argument("--heldout", type=_parse_ints, default=(0, 1, 2))
    p.add_argument("--bc-seed", type=int, default=0)
    p.add_argument("--n-eps", type=int, default=32)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--population", type=int, default=128)
    p.add_argument("--elites", type=int, default=16)
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--max-plan-std", type=float, default=1.0)
    p.add_argument("--discount", type=float, default=0.99)
    p.add_argument("--controllers", default="all")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = MPPIConfig(
        horizon=args.horizon,
        population_size=args.population,
        num_elites=args.elites,
        mppi_iterations=args.iterations,
        max_plan_std=args.max_plan_std,
        discount=args.discount,
    )
    ds = load_continuous_dataset(args.dataset)
    horizon_env = int(ds.blue_action.shape[1])
    env = make_env("capture", continuous=True)
    prey_name = env.good_agents[0]
    obs_width = max(env.observation_space(a).shape[0] for a in env.agents)

    all_controllers = [
        "mappo_prey",
        "random",
        "planner_true_red",
        "planner_bc_oracle_correct",
        "planner_bc_oracle_zero",
        "planner_bc_oracle_wrong",
        "planner_bc_real_online",
        "planner_bc_real_zero",
        "planner_bc_real_shuffled",
        "planner_bc_no_c",
    ]
    controllers = all_controllers if args.controllers == "all" else args.controllers.split(",")

    results: dict[str, Any] = {
        "schema_version": 1,
        "git_sha": git_sha(_ROOT),
        "git_dirty": git_dirty(_ROOT),
        "dependencies": package_versions(),
        "dataset_sha256": file_sha256(args.dataset),
        "planner": cfg.__dict__,
        "config": {
            "heldout_checkpoints": list(args.heldout),
            "bc_seed": args.bc_seed,
            "n_eps_per_type": args.n_eps,
            "controllers": controllers,
            "terminal_value": "zero",
            "imagined_dynamics": "exact_simulator",
            "context_inside_horizon": "held_fixed",
        },
        "runs": [],
        "artifacts": {},
    }
    per_episode: dict[tuple[int, str, str], dict[str, np.ndarray]] = {}
    for heldout in args.heldout:
        seed_dir = args.bc_artifacts / f"fold_{heldout}" / f"seed_{args.bc_seed}"
        encoder = CausalContextEncoder.load(seed_dir / "context_encoder.npz")
        bc = {
            arm: ContinuousBCPolicy.load(seed_dir / f"{arm}.npz")
            for arm in ("oracle", "real_c", "no_c")
        }
        results["artifacts"][str(heldout)] = {
            "context_encoder": file_sha256(seed_dir / "context_encoder.npz"),
            **{arm: file_sha256(seed_dir / f"{arm}.npz") for arm in bc},
        }
        prey_params = load_continuous_actor_params(
            continuous_checkpoint_path(args.logdir, "capture", "prey", heldout)
        )
        for label, pred_type in enumerate(OBJECTIVE_TYPES):
            rows = np.flatnonzero((ds.checkpoint_seed == heldout) & (ds.objective_label == label))
            rows = rows[: args.n_eps]
            reset_keys = ds.environment_seed[rows]
            step_seed = ds.step_seed[rows]
            red_params = load_continuous_actor_params(
                continuous_checkpoint_path(args.logdir, pred_type, "pred", heldout)
            )
            specs = {
                "mappo_prey": (mappo_prey_controller(prey_params, obs_width, prey_name), "zero", None),
                "random": (random_controller(prey_name), "zero", None),
                "planner_true_red": (make_true_red_policy(red_params, obs_width), "zero", None),
                "planner_bc_oracle_correct": (make_bc_red_policy(bc["oracle"]), "oracle", None),
                "planner_bc_oracle_zero": (make_bc_red_policy(bc["oracle"]), "zero", None),
                "planner_bc_oracle_wrong": (make_bc_red_policy(bc["oracle"]), "wrong_oracle", None),
                "planner_bc_real_online": (make_bc_red_policy(bc["real_c"]), "online", encoder),
                "planner_bc_real_zero": (make_bc_red_policy(bc["real_c"]), "zero", None),
                "planner_bc_real_shuffled": (make_bc_red_policy(bc["real_c"]), "shuffled", encoder),
                "planner_bc_no_c": (make_bc_red_policy(bc["no_c"]), "zero", None),
            }
            for name in controllers:
                blue_or_red, context_mode, enc = specs[name]
                if name.startswith("planner"):
                    blue = planner_controller(jit_planner(env, blue_or_red, cfg), cfg)
                else:
                    blue = blue_or_red
                t0 = time.time()
                out = run_matched_episodes(
                    env,
                    red_params,
                    blue,
                    reset_keys,
                    step_seed,
                    horizon=horizon_env,
                    context_mode=context_mode,
                    label=label,
                    encoder=enc,
                    shuffle_seed=1000 * heldout + label,
                )
                dt = time.time() - t0
                per_episode[(heldout, pred_type, name)] = out
                row = {
                    "heldout_checkpoint": heldout,
                    "opponent": pred_type,
                    "controller": name,
                    "context_mode": context_mode,
                    "n_episodes": int(len(rows)),
                    "seconds": dt,
                    "blue_action_abs_max": out["blue_action_abs_max"],
                    **{m: {"mean": float(np.mean(out[m])), "std": float(np.std(out[m]))} for m in METRICS},
                }
                results["runs"].append(row)
                print(
                    f"h{heldout} {pred_type:8s} {name:28s} return {row['blue_return']['mean']:8.2f}"
                    f"  captured {row['captured']['mean']:.2f}  res {row['resources_collected']['mean']:.2f}"
                    f"  ({dt:.0f}s)"
                )

    # Paired comparisons on matched resets.
    def paired(cand: str, base: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for metric in METRICS:
            deltas = []
            for (h, typ, name), val in per_episode.items():
                if name != cand:
                    continue
                other = per_episode.get((h, typ, base))
                if other is None:
                    continue
                d = np.asarray(val[metric], np.float64) - np.asarray(other[metric], np.float64)
                deltas.append({"heldout": h, "opponent": typ, "delta_mean": float(d.mean()), "n": int(len(d))})
            if deltas:
                means = np.asarray([r["delta_mean"] for r in deltas])
                out[metric] = {
                    "delta_mean": float(means.mean()),
                    "all_groups_better": bool(np.all(means > 0)) if metric in {"blue_return", "survival_time", "resources_collected"} else bool(np.all(means < 0)),
                    "groups": deltas,
                }
        return out

    comparisons = {
        "planner_true_red_vs_mappo_prey": paired("planner_true_red", "mappo_prey"),
        "planner_true_red_vs_random": paired("planner_true_red", "random"),
        "planner_bc_oracle_correct_vs_zero": paired("planner_bc_oracle_correct", "planner_bc_oracle_zero"),
        "planner_bc_oracle_correct_vs_wrong": paired("planner_bc_oracle_correct", "planner_bc_oracle_wrong"),
        "planner_bc_real_online_vs_zero": paired("planner_bc_real_online", "planner_bc_real_zero"),
        "planner_bc_real_online_vs_shuffled": paired("planner_bc_real_online", "planner_bc_real_shuffled"),
        "planner_bc_real_online_vs_no_c": paired("planner_bc_real_online", "planner_bc_no_c"),
        "planner_bc_oracle_correct_vs_true_red": paired("planner_bc_oracle_correct", "planner_true_red"),
        "planner_bc_real_online_vs_true_red": paired("planner_bc_real_online", "planner_true_red"),
    }
    results["comparisons"] = {k: v for k, v in comparisons.items() if v}
    abs_max = max((r["blue_action_abs_max"] for r in results["runs"]), default=0.0)
    tr = results["comparisons"].get("planner_true_red_vs_mappo_prey", {}).get("blue_return")
    rr = results["comparisons"].get("planner_true_red_vs_random", {}).get("blue_return")
    results["claim_gates"] = {
        "actions_within_bounds": bool(abs_max <= 1.0 + 1e-6),
        "planner_true_red_return_beats_fixed_policy_all_groups": None if tr is None else tr["all_groups_better"],
        "planner_true_red_return_beats_random_all_groups": None if rr is None else rr["all_groups_better"],
        "planner_true_red_mean_return_delta_vs_fixed_policy": None if tr is None else tr["delta_mean"],
        "planner_true_red_mean_return_delta_vs_random": None if rr is None else rr["delta_mean"],
    }
    results["claim_gates"]["gate3_pass"] = (
        None
        if None in {results["claim_gates"]["planner_true_red_return_beats_fixed_policy_all_groups"],
                    results["claim_gates"]["planner_true_red_return_beats_random_all_groups"]}
        else bool(
            results["claim_gates"]["actions_within_bounds"]
            and results["claim_gates"]["planner_true_red_return_beats_fixed_policy_all_groups"]
            and results["claim_gates"]["planner_true_red_return_beats_random_all_groups"]
        )
    )
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.artifact_dir / "results.json"
    out_path.write_text(json.dumps(_jsonable(results), indent=2, sort_keys=True) + "\n")
    np.savez_compressed(
        args.artifact_dir / "per_episode.npz",
        **{
            f"{h}__{typ}__{name}__{m}": np.asarray(v[m])
            for (h, typ, name), v in per_episode.items()
            for m in METRICS
        },
    )
    print(json.dumps(_jsonable(results["claim_gates"]), indent=1))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
