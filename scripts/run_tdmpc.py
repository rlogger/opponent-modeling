#!/usr/bin/env python3
"""Gate 4/5 driver: train, evaluate, and compare TD-MPC opponent modes.

Subcommands
-----------
train      Fit one world model (``--mode`` implicit | conditioned | factored,
           ``--encoder`` identity | mlp) on the continuous dataset with the
           evaluation checkpoint family held out, then report model error vs
           persistence and reward / termination calibration on both splits.
evaluate   Run the trained controller in the real environment against the
           held-out specialists on matched resets with opponent-context
           controls, plus the factored invariance checks.
compare    Aggregate matched runs (modes x seeds) into one results file with
           the Gate 5 claim gates.

Every run writes a manifest binding the dataset, context encoder, config,
seed, and git state to the produced checkpoint.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mopa.context import CONTEXT_DIM, CausalContextEncoder  # noqa: E402
from mopa.continuous_data import (  # noqa: E402
    DEFAULT_CONTINUOUS_LOGDIR,
    OBJECTIVE_TYPES,
    continuous_checkpoint_path,
    load_continuous_actor_params,
    load_continuous_dataset,
    markov_state,
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
from mopa.tdmpc import create_agent, load_config  # noqa: E402
from mopa.tdmpc_data import (  # noqa: E402
    SequenceReplay,
    attach_context,
    multistep_model_error,
    reward_calibration,
    state_statistics,
    termination_calibration,
)
from tag_objectives import make_env  # noqa: E402

METRICS = ("blue_return", "captured", "survival_time", "resources_collected", "prey_lava_steps")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, jax.Array):
        return _jsonable(np.asarray(value))
    return value


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(v) for v in value.split(",") if v.strip())


# --------------------------------------------------------------------------- #
# Shared setup
# --------------------------------------------------------------------------- #
def load_context(ds, bc_artifacts: Path, heldout: int, bc_seed: int, source: str):  # noqa: ANN001
    data = ds.as_dict()
    enc_path = bc_artifacts / f"fold_{heldout}" / f"seed_{bc_seed}" / "context_encoder.npz"
    encoder = CausalContextEncoder.load(enc_path)
    causal = encoder.causal_context(ds.prey_pos, ds.pred_pos, ds.valid_length)
    return encoder, enc_path, attach_context(data, causal, source=source)


def run_dir(root: Path, mode: str, encoder: str, heldout: int, seed: int, context_source: str) -> Path:
    return root / f"{mode}__{encoder}__ctx-{context_source}__h{heldout}__s{seed}"


def save_agent(agent, path: Path) -> None:  # noqa: ANN001
    path.write_bytes(flax.serialization.to_bytes(agent))


def load_agent(template, path: Path):  # noqa: ANN001
    return flax.serialization.from_bytes(template, path.read_bytes())


def build_template(run: Path):  # noqa: ANN001
    manifest = json.loads((run / "manifest.json").read_text())
    cfg = manifest["config"]
    stats = np.load(run / "state_stats.npz")
    agent = create_agent(
        cfg,
        int(manifest["state_dim"]),
        key=jax.random.PRNGKey(int(manifest["seed"])),
        obs_mean=stats["mean"],
        obs_std=stats["std"],
    )
    return load_agent(agent, run / "agent.msgpack"), manifest, stats


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #
def cmd_train(args: argparse.Namespace) -> int:
    cfg = load_config(profile=args.profile)
    cfg["opponent_mode"] = args.mode
    cfg["encoder"]["type"] = args.encoder
    # The frozen context encoder always produces CONTEXT_DIM columns.
    cfg["context_dim"] = 0 if (args.mode == "implicit" and args.context_source == "none") else CONTEXT_DIM
    ds = load_continuous_dataset(args.dataset)
    data = ds.as_dict()
    source = "zero" if args.context_source == "none" else args.context_source
    encoder, enc_path, context = load_context(ds, args.bc_artifacts, args.heldout, args.bc_seed, source)
    if args.mode == "implicit":
        context = np.zeros_like(context)[..., : cfg["context_dim"]]
    train_eps = np.flatnonzero(ds.checkpoint_seed != args.heldout)
    eval_eps = np.flatnonzero(ds.checkpoint_seed == args.heldout)
    mean, std = state_statistics(ds.state[train_eps], ds.valid_mask[train_eps])
    state_dim = int(ds.state.shape[-1])
    agent = create_agent(cfg, state_dim, key=jax.random.PRNGKey(args.seed), obs_mean=mean, obs_std=std)
    horizon = agent.horizon
    replay = SequenceReplay.from_dataset(data, train_eps, horizon, context)
    eval_replay = SequenceReplay.from_dataset(data, eval_eps, horizon, context)
    rng = np.random.default_rng(args.seed)
    key = jax.random.PRNGKey(10_000 + args.seed)

    out = run_dir(args.out, args.mode, args.encoder, args.heldout, args.seed, args.context_source)
    out.mkdir(parents=True, exist_ok=True)
    log: list[dict[str, float]] = []
    t0 = time.time()
    for step in range(1, args.updates + 1):
        batch = replay.sample(rng, agent.batch_size)
        key, k = jax.random.split(key)
        agent, info = agent.update(**batch, key=k)
        if step % args.log_every == 0 or step == args.updates:
            row = {
                "step": step,
                "seconds": time.time() - t0,
                **{
                    k_: float(np.asarray(info[k_]))
                    for k_ in ("total_loss", "consistency_loss", "reward_loss", "value_loss", "continue_loss", "red_loss", "policy_loss")
                },
            }
            log.append(row)
            print(
                f"[{args.mode}/{args.encoder} h{args.heldout} s{args.seed}] step {step:6d} "
                f"total {row['total_loss']:.4f} cons {row['consistency_loss']:.4f} rew {row['reward_loss']:.4f} "
                f"val {row['value_loss']:.4f} cont {row['continue_loss']:.4f} red {row['red_loss']:.4f} "
                f"pi {row['policy_loss']:.4f} ({row['seconds']:.0f}s)",
                flush=True,
            )
    if not all(np.isfinite(list(log[-1].values()))):
        raise RuntimeError("non-finite loss at end of training")

    save_agent(agent, out / "agent.msgpack")
    np.savez(out / "state_stats.npz", mean=mean, std=std)
    evaluation = {}
    for split, rep in (("train", replay), ("heldout", eval_replay)):
        evaluation[split] = {
            "n_transitions": rep.n_transitions,
            "model_error": multistep_model_error(agent, rep, horizons=(1, 3, 10), obs_std=std, seed=args.seed),
            "reward_calibration": reward_calibration(agent, rep, seed=args.seed),
            "termination_calibration": termination_calibration(agent, rep, seed=args.seed),
        }
    manifest = {
        "schema_version": 1,
        "git_sha": git_sha(_ROOT),
        "git_dirty": git_dirty(_ROOT),
        "dependencies": package_versions(),
        "config": cfg,
        "profile": args.profile,
        "mode": args.mode,
        "encoder": args.encoder,
        "context_source": args.context_source,
        "heldout_checkpoint": args.heldout,
        "seed": args.seed,
        "updates": args.updates,
        "state_dim": state_dim,
        "dataset": {"path": str(args.dataset), "sha256": file_sha256(args.dataset)},
        "context_encoder": {"path": str(enc_path), "sha256": file_sha256(enc_path), "frozen": True},
        "world_encoder_trained": args.encoder == "mlp",
        "train_episodes": int(len(train_eps)),
        "heldout_episodes": int(len(eval_eps)),
        "train_transitions": replay.n_transitions,
        "training_log": log,
        "evaluation": evaluation,
        "agent_sha256": file_sha256(out / "agent.msgpack"),
    }
    (out / "manifest.json").write_text(json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n")
    print(json.dumps(_jsonable({k: v["model_error"]["per_horizon"] for k, v in evaluation.items()}), indent=1))
    print(f"Wrote {out}")
    return 0


# --------------------------------------------------------------------------- #
# evaluate
# --------------------------------------------------------------------------- #
def tdmpc_controller(agent, env):  # noqa: ANN001
    act = jax.jit(
        lambda a, s, prev, ctx, key: a.act(
            markov_state(env, s), prev_plan=prev, mpc=True, deterministic=True, context=ctx, key=key
        )
    )

    def blue(state, obs, context, carry, key, t):  # noqa: ANN001
        del obs, t
        action, plan = act(agent, state, carry, context, key)
        return action, plan

    return blue


def factored_invariance_checks(agent, ds, eval_eps, context, seed: int) -> dict[str, Any] | None:  # noqa: ANN001
    """Numerical Gate 5 requirements for Equation 3 on real held-out latents."""
    m = agent.model
    if m.opponent_mode != "factored":
        return None
    rng = np.random.default_rng(seed)
    e = rng.choice(eval_eps, size=64)
    t = np.floor(rng.random(64) * (ds.valid_length[e] - 1)).astype(int)
    key = jax.random.PRNGKey(seed)
    x = m.encode(jnp.asarray(ds.state[e, t]), m.encoder.params, key)
    u = jnp.asarray(ds.blue_action[e, t])
    v = jnp.asarray(ds.red_action[e, t])
    c_real = jnp.asarray(context[e, t])
    one_hot = jnp.eye(CONTEXT_DIM)
    c_alt = [one_hot[i][None].repeat(64, 0) for i in range(CONTEXT_DIM)] + [jnp.zeros((64, CONTEXT_DIM))]
    a_real = m.transition_inputs(u, c_real, v)
    inv = {"dynamics": 0.0, "reward": 0.0, "continue": 0.0}
    red_change = []
    for c in c_alt:
        a = m.transition_inputs(u, c, v)
        inv["dynamics"] = max(inv["dynamics"], float(jnp.max(jnp.abs(m.next(x, a, m.dynamics_model.params) - m.next(x, a_real, m.dynamics_model.params)))))
        inv["reward"] = max(inv["reward"], float(jnp.max(jnp.abs(m.reward(x, a, m.reward_model.params)[0] - m.reward(x, a_real, m.reward_model.params)[0]))))
        if m.predict_continues:
            inv["continue"] = max(inv["continue"], float(jnp.max(jnp.abs(m.continue_logits(x, a, m.continue_model.params) - m.continue_logits(x, a_real, m.continue_model.params)))))
        red_change.append(float(jnp.mean(jnp.linalg.norm(m.red_action(x, c, m.red_model.params) - m.red_action(x, c_real, m.red_model.params), axis=-1))))
    # Varied red actions change the next-state prediction.
    x_v = m.next(x, m.transition_inputs(u, c_real, v), m.dynamics_model.params)
    x_nv = m.next(x, m.transition_inputs(u, c_real, -v), m.dynamics_model.params)
    return {
        "max_abs_change_under_context_swap": inv,
        "physics_reward_termination_context_invariant": all(val == 0.0 for val in inv.values()),
        "mean_red_action_change_under_context_swap": red_change,
        "red_action_depends_on_context": bool(max(red_change) > 1e-4),
        "mean_next_state_change_when_red_action_flipped": float(jnp.mean(jnp.linalg.norm(x_v - x_nv, axis=-1))),
        "next_state_depends_on_red_action": bool(jnp.mean(jnp.linalg.norm(x_v - x_nv, axis=-1)) > 1e-4),
    }


def cmd_evaluate(args: argparse.Namespace) -> int:
    agent, manifest, stats = build_template(args.run)
    mode = manifest["mode"]
    heldout = int(manifest["heldout_checkpoint"])
    ds = load_continuous_dataset(args.dataset)
    encoder, enc_path, context = load_context(ds, args.bc_artifacts, heldout, args.bc_seed, "causal")
    if file_sha256(enc_path) != manifest["context_encoder"]["sha256"]:
        raise ValueError("context encoder does not match the training manifest")
    eval_eps = np.flatnonzero(ds.checkpoint_seed == heldout)
    env = make_env("capture", continuous=True)
    prey_name = env.good_agents[0]
    obs_width = max(env.observation_space(a).shape[0] for a in env.agents)
    horizon = int(ds.blue_action.shape[1])
    context_modes = ["online", "zero", "shuffled", "wrong_oracle", "oracle"] if mode != "implicit" else ["zero"]
    if args.context_modes:
        context_modes = args.context_modes.split(",")
    blue = tdmpc_controller(agent, env)
    prey_params = load_continuous_actor_params(continuous_checkpoint_path(args.logdir, "capture", "prey", heldout))
    runs = []
    per_episode: dict[str, np.ndarray] = {}
    for label, pred_type in enumerate(OBJECTIVE_TYPES):
        rows = np.flatnonzero((ds.checkpoint_seed == heldout) & (ds.objective_label == label))[: args.n_eps]
        reset_keys, step_seed = ds.environment_seed[rows], ds.step_seed[rows]
        red_params = load_continuous_actor_params(continuous_checkpoint_path(args.logdir, pred_type, "pred", heldout))
        specs = [("tdmpc", blue, cm, encoder if cm in {"online", "shuffled"} else None) for cm in context_modes]
        if args.controls:
            specs += [
                ("mappo_prey", mappo_prey_controller(prey_params, obs_width, prey_name), "zero", None),
                ("random", random_controller(prey_name), "zero", None),
            ]
        for name, ctrl, cm, enc in specs:
            t0 = time.time()
            out = run_matched_episodes(
                env, red_params, ctrl, reset_keys, step_seed, horizon=horizon,
                context_mode=cm, label=label, encoder=enc, shuffle_seed=1000 * heldout + label,
            )
            dt = time.time() - t0
            key = f"{pred_type}__{name}__{cm}"
            for mtr in METRICS:
                per_episode[f"{key}__{mtr}"] = np.asarray(out[mtr])
            row = {
                "opponent": pred_type,
                "controller": name,
                "context_mode": cm,
                "n_episodes": int(len(rows)),
                "seconds_total": dt,
                "seconds_per_env_step": dt / (horizon * len(rows)),
                "blue_action_abs_max": out["blue_action_abs_max"],
                **{mtr: {"mean": float(np.mean(out[mtr])), "std": float(np.std(out[mtr]))} for mtr in METRICS},
            }
            runs.append(row)
            print(
                f"{pred_type:8s} {name:10s} ctx={cm:12s} return {row['blue_return']['mean']:8.2f} "
                f"captured {row['captured']['mean']:.2f} res {row['resources_collected']['mean']:.2f} ({dt:.0f}s)",
                flush=True,
            )
    result = {
        "schema_version": 1,
        "git_sha": git_sha(_ROOT),
        "git_dirty": git_dirty(_ROOT),
        "run": str(args.run),
        "agent_sha256": file_sha256(args.run / "agent.msgpack"),
        "manifest_sha256": file_sha256(args.run / "manifest.json"),
        "mode": mode,
        "encoder": manifest["encoder"],
        "seed": manifest["seed"],
        "heldout_checkpoint": heldout,
        "n_eps_per_opponent": args.n_eps,
        "planner": {k: manifest["config"]["tdmpc2"][k] for k in ("horizon", "population_size", "policy_prior_samples", "num_elites", "mppi_iterations")},
        "runs": runs,
        "factored_invariance": factored_invariance_checks(agent, ds, eval_eps, context, int(manifest["seed"])),
    }
    (args.run / "evaluation.json").write_text(json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n")
    np.savez_compressed(args.run / "evaluation_per_episode.npz", **per_episode)
    print(f"Wrote {args.run / 'evaluation.json'}")
    return 0


# --------------------------------------------------------------------------- #
# compare
# --------------------------------------------------------------------------- #
def cmd_compare(args: argparse.Namespace) -> int:
    runs = sorted(p for p in args.root.iterdir() if (p / "evaluation.json").is_file())
    if not runs:
        raise FileNotFoundError(f"no evaluated runs under {args.root}")
    evals = [json.loads((p / "evaluation.json").read_text()) for p in runs]
    manifests = [json.loads((p / "manifest.json").read_text()) for p in runs]
    per_ep = [dict(np.load(p / "evaluation_per_episode.npz")) for p in runs]
    table: dict[str, Any] = {}
    modes = sorted({e["mode"] for e in evals})
    for e, m, pe in zip(evals, manifests, per_ep):
        key = f"{e['mode']}__{e['encoder']}"
        table.setdefault(key, {"seeds": [], "model_error_heldout": [], "rows": {}})
        table[key]["seeds"].append(e["seed"])
        table[key]["model_error_heldout"].append(m["evaluation"]["heldout"]["model_error"]["per_horizon"])
        for row in e["runs"]:
            r_key = f"{row['opponent']}__{row['controller']}__{row['context_mode']}"
            table[key]["rows"].setdefault(r_key, {mtr: [] for mtr in METRICS})
            for mtr in METRICS:
                table[key]["rows"][r_key][mtr].append(row[mtr]["mean"])
    summary: dict[str, Any] = {}
    for key, entry in table.items():
        summary[key] = {
            "n_seeds": len(entry["seeds"]),
            "seeds": entry["seeds"],
            "rows": {
                r: {mtr: {"mean": float(np.mean(v)), "std": float(np.std(v)), "per_seed": v} for mtr, v in vals.items()}
                for r, vals in entry["rows"].items()
            },
        }

    def paired(cand_mode: str, cand_ctx: str, base_mode: str, base_ctx: str, metric: str, encoder: str) -> dict | None:
        deltas = []
        for e, pe in zip(evals, per_ep):
            if e["mode"] != cand_mode or e["encoder"] != encoder:
                continue
            base = next(
                (pb for eb, pb in zip(evals, per_ep) if eb["mode"] == base_mode and eb["encoder"] == encoder and eb["seed"] == e["seed"]),
                None,
            )
            if base is None:
                continue
            for typ in OBJECTIVE_TYPES:
                a = pe.get(f"{typ}__tdmpc__{cand_ctx}__{metric}")
                b = base.get(f"{typ}__tdmpc__{base_ctx}__{metric}")
                if a is None or b is None:
                    continue
                deltas.append({"seed": e["seed"], "opponent": typ, "delta_mean": float(np.mean(a) - np.mean(b))})
        if not deltas:
            return None
        d = np.asarray([r["delta_mean"] for r in deltas])
        return {"delta_mean": float(d.mean()), "delta_std": float(d.std()), "fraction_positive": float(np.mean(d > 0)), "n_pairs": int(len(d)), "pairs": deltas}

    encoders = sorted({e["encoder"] for e in evals})
    comparisons: dict[str, Any] = {}
    for enc in encoders:
        comparisons[enc] = {
            "factored_online_vs_implicit": {m: paired("factored", "online", "implicit", "zero", m, enc) for m in METRICS},
            "conditioned_online_vs_implicit": {m: paired("conditioned", "online", "implicit", "zero", m, enc) for m in METRICS},
            "factored_online_vs_zero_context": {m: paired("factored", "online", "factored", "zero", m, enc) for m in METRICS},
            "factored_online_vs_shuffled_context": {m: paired("factored", "online", "factored", "shuffled", m, enc) for m in METRICS},
            "factored_online_vs_wrong_oracle": {m: paired("factored", "online", "factored", "wrong_oracle", m, enc) for m in METRICS},
            "factored_oracle_vs_implicit": {m: paired("factored", "oracle", "implicit", "zero", m, enc) for m in METRICS},
            "conditioned_oracle_vs_implicit": {m: paired("conditioned", "oracle", "implicit", "zero", m, enc) for m in METRICS},
        }
    invariance = [
        {"run": str(p), "seed": e["seed"], **(e["factored_invariance"] or {})}
        for p, e in zip(runs, evals)
        if e["factored_invariance"] is not None
    ]
    gates: dict[str, Any] = {}
    for enc in encoders:
        fi = comparisons[enc]["factored_online_vs_implicit"]["blue_return"]
        gates[enc] = {
            "factored_improves_blue_return_over_implicit_mean": None if fi is None else fi["delta_mean"] > 0,
            "factored_improves_blue_return_over_implicit_all_pairs": None if fi is None else fi["fraction_positive"] == 1.0,
            "factored_online_beats_zero_context_mean": (
                None if comparisons[enc]["factored_online_vs_zero_context"]["blue_return"] is None
                else comparisons[enc]["factored_online_vs_zero_context"]["blue_return"]["delta_mean"] > 0
            ),
            "factored_online_beats_shuffled_context_mean": (
                None if comparisons[enc]["factored_online_vs_shuffled_context"]["blue_return"] is None
                else comparisons[enc]["factored_online_vs_shuffled_context"]["blue_return"]["delta_mean"] > 0
            ),
            "factored_online_beats_wrong_oracle_mean": (
                None if comparisons[enc]["factored_online_vs_wrong_oracle"]["blue_return"] is None
                else comparisons[enc]["factored_online_vs_wrong_oracle"]["blue_return"]["delta_mean"] > 0
            ),
            "all_factored_runs_pass_action_clamp_invariance": bool(
                invariance and all(r.get("physics_reward_termination_context_invariant") for r in invariance)
            ),
            "n_factored_seeds": sum(1 for e in evals if e["mode"] == "factored" and e["encoder"] == enc),
        }
        g = gates[enc]
        g["gate5_success_claim_supported"] = (
            None
            if g["factored_improves_blue_return_over_implicit_mean"] is None
            else bool(
                g["factored_improves_blue_return_over_implicit_mean"]
                and g["all_factored_runs_pass_action_clamp_invariance"]
                and g["n_factored_seeds"] >= 3
            )
        )
    results = {
        "schema_version": 1,
        "git_sha": git_sha(_ROOT),
        "git_dirty": git_dirty(_ROOT),
        "dependencies": package_versions(),
        "runs": [str(p) for p in runs],
        "modes": modes,
        "encoders": encoders,
        "summary": summary,
        "comparisons": comparisons,
        "factored_invariance": invariance,
        "claim_gates": gates,
    }
    out = args.root / "comparison.json"
    out.write_text(json.dumps(_jsonable(results), indent=2, sort_keys=True) + "\n")
    print(json.dumps(_jsonable(gates), indent=1))
    print(f"Wrote {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--dataset", type=Path, default=Path("artifacts/continuous/dataset.npz"))
    tr.add_argument("--bc-artifacts", type=Path, default=Path("artifacts/bc_continuous"))
    tr.add_argument("--bc-seed", type=int, default=0)
    tr.add_argument("--out", type=Path, default=Path("artifacts/tdmpc"))
    tr.add_argument("--profile", default="gate4")
    tr.add_argument("--mode", choices=("implicit", "conditioned", "factored"), required=True)
    tr.add_argument("--encoder", choices=("identity", "mlp"), default="identity")
    tr.add_argument("--context-source", choices=("causal", "oracle", "zero", "none"), default="causal")
    tr.add_argument("--heldout", type=int, default=2)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--updates", type=int, default=10000)
    tr.add_argument("--log-every", type=int, default=500)
    tr.set_defaults(func=cmd_train)
    ev = sub.add_parser("evaluate")
    ev.add_argument("run", type=Path)
    ev.add_argument("--dataset", type=Path, default=Path("artifacts/continuous/dataset.npz"))
    ev.add_argument("--bc-artifacts", type=Path, default=Path("artifacts/bc_continuous"))
    ev.add_argument("--bc-seed", type=int, default=0)
    ev.add_argument("--logdir", type=Path, default=DEFAULT_CONTINUOUS_LOGDIR)
    ev.add_argument("--n-eps", type=int, default=48)
    ev.add_argument("--context-modes", default="")
    ev.add_argument("--controls", action="store_true")
    ev.set_defaults(func=cmd_evaluate)
    cp = sub.add_parser("compare")
    cp.add_argument("--root", type=Path, default=Path("artifacts/tdmpc"))
    cp.set_defaults(func=cmd_compare)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
