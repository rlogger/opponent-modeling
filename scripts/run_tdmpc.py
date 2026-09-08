#!/usr/bin/env python3
"""Train and evaluate Equation 1 first: ``x_next = dynamics(x, blue_action)``.

The default implicit baseline requires continuous data and specialist policies,
but no opponent BC or context-encoder artifacts. Other modes remain explicit.

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
    FEATURE_MAPS,
    SequenceReplay,
    attach_context,
    multistep_model_error,
    reward_calibration,
    state_statistics,
    termination_calibration,
    world_model_observation,
)
from mopa.zero_s import ZeroSOpponent  # noqa: E402
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
    if source == "zero":
        return None, None, attach_context(data, None, source="zero")
    enc_path = bc_artifacts / f"fold_{heldout}" / f"seed_{bc_seed}" / "context_encoder.npz"
    encoder = CausalContextEncoder.load(enc_path)
    causal = encoder.causal_context(ds.prey_pos, ds.pred_pos, ds.valid_length)
    return encoder, enc_path, attach_context(data, causal, source=source)


def run_dir(root: Path, mode: str, encoder: str, heldout: int, seed: int, context_source: str, features: str) -> Path:
    suffix = "" if features == "markov" else f"__{features}"
    return root / f"{mode}__{encoder}__ctx-{context_source}__h{heldout}__s{seed}{suffix}"


def save_agent(agent, path: Path) -> None:  # noqa: ANN001
    path.write_bytes(flax.serialization.to_bytes(agent))


def load_agent(template, path: Path):  # noqa: ANN001
    return flax.serialization.from_bytes(template, path.read_bytes())


def build_template(run: Path):  # noqa: ANN001
    manifest = json.loads((run / "manifest.json").read_text())
    opponent = None
    if "source_0s_commit" in manifest:
        # The 0s runner stores a separate config and a frozen decoder whose
        # static apply_fn/optimizer must exist BEFORE Flax restores weights.
        for name in ("config.json", "state_stats.npz", "opponent.msgpack", "agent.msgpack"):
            if file_sha256(run / name) != manifest["artifacts"][name]:
                raise ValueError(f"0s artifact does not match training manifest: {name}")
        cfg = json.loads((run / "config.json").read_text())["world_model"]
        opponent = ZeroSOpponent.load(run / "opponent.msgpack")
        manifest = {**manifest, "config": cfg, "state_dim": 66,
                    "mode": cfg["opponent_mode"], "encoder": cfg["encoder"]["type"],
                    "features": "markov", "context_source": "zero_s"}
    cfg = manifest["config"]
    stats = np.load(run / "state_stats.npz")
    agent = create_agent(
        cfg,
        int(manifest["state_dim"]),
        key=jax.random.PRNGKey(int(manifest["seed"])),
        obs_mean=stats["mean"],
        obs_std=stats["std"],
    )
    if opponent is not None:
        agent = opponent.attach(agent, stats["mean"], stats["std"])
    return load_agent(agent, run / "agent.msgpack"), manifest, stats


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #
def cmd_train(args: argparse.Namespace) -> int:
    cfg = load_config(profile=args.profile)
    cfg["opponent_mode"] = args.mode
    cfg["encoder"]["type"] = args.encoder
    # Equation 1 has no context input or context-checkpoint dependency.
    args.context_source = "none" if args.mode == "implicit" else (args.context_source or "causal")
    cfg["context_dim"] = 0 if args.mode == "implicit" else CONTEXT_DIM
    ds = load_continuous_dataset(args.dataset)
    data = ds.as_dict()
    source = "zero" if args.context_source == "none" else args.context_source
    # Preserve existing context-checkpoint provenance for explicit nonimplicit
    # ablations; only Equation 1 is independent of that artifact.
    load_source = "causal" if args.mode != "implicit" and source == "zero" else source
    encoder, enc_path, context = load_context(ds, args.bc_artifacts, args.heldout, args.bc_seed, load_source)
    if source == "zero":
        context = np.zeros_like(context)
    if args.mode == "implicit":
        context = np.zeros_like(context)[..., : cfg["context_dim"]]
    train_eps = np.flatnonzero(ds.checkpoint_seed != args.heldout)
    eval_eps = np.flatnonzero(ds.checkpoint_seed == args.heldout)
    mean, std = state_statistics(ds.state[train_eps], ds.valid_mask[train_eps], feature_map=args.features)
    state_dim = int(mean.shape[0])
    agent = create_agent(cfg, state_dim, key=jax.random.PRNGKey(args.seed), obs_mean=mean, obs_std=std)
    horizon = agent.horizon
    replay = SequenceReplay.from_dataset(data, train_eps, horizon, context, feature_map=args.features)
    eval_replay = SequenceReplay.from_dataset(data, eval_eps, horizon, context, feature_map=args.features)
    rng = np.random.default_rng(args.seed)
    key = jax.random.PRNGKey(10_000 + args.seed)

    out = run_dir(args.out, args.mode, args.encoder, args.heldout, args.seed, args.context_source, args.features)
    out.mkdir(parents=True, exist_ok=True)
    log: list[dict[str, float]] = []
    t0 = time.time()
    step = 0

    def train_steps(n: int) -> None:
        nonlocal agent, key, step
        for _ in range(n):
            step += 1
            batch = replay.sample(rng, agent.batch_size)
            key, k = jax.random.split(key)
            agent, info = agent.update(**batch, key=k)
            if step % args.log_every == 0 or step == total_updates:
                row = {
                    "step": step,
                    "seconds": time.time() - t0,
                    "replay_transitions": replay.n_transitions,
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

    total_updates = args.updates + args.online_rounds * args.updates_per_round
    train_steps(args.updates)
    online_log: list[dict[str, Any]] = []
    collect_rng = np.random.default_rng(777 + args.seed)
    for round_index in range(args.online_rounds):
        summary = collect_online_round(
            agent, ds, replay, encoder,
            heldout=args.heldout, features=args.features, context_source=args.context_source,
            mode=args.mode, episodes_per_group=args.online_episodes, logdir=args.logdir,
            rng=collect_rng, horizon=int(ds.blue_action.shape[1]),
        )
        summary["round"] = round_index
        summary["seconds"] = time.time() - t0
        online_log.append(summary)
        mean_ret = float(np.mean([g["blue_return_mean"] for g in summary["groups"]]))
        print(
            f"[{args.mode}/{args.encoder} h{args.heldout} s{args.seed}] online round {round_index}: "
            f"{summary['n_episodes']} episodes, {summary['n_transitions']} transitions, "
            f"mean collected return {mean_ret:.2f}; replay {replay.n_transitions} ({summary['seconds']:.0f}s)",
            flush=True,
        )
        train_steps(args.updates_per_round)
    if not all(np.isfinite([v for v in log[-1].values() if isinstance(v, float)])):
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
        "features": args.features,
        "context_source": args.context_source,
        "heldout_checkpoint": args.heldout,
        "seed": args.seed,
        "updates": args.updates,
        "online": {
            "rounds": args.online_rounds,
            "episodes_per_group_per_round": args.online_episodes,
            "updates_per_round": args.updates_per_round,
            "opponents": "training checkpoint families only (held-out family never collected)",
            "exploration": "upstream train=True elite sample + MPPI noise",
            "log": online_log,
        },
        "total_updates": total_updates,
        "final_replay_transitions": replay.n_transitions,
        "final_replay_episodes": replay.n_episodes,
        "state_dim": state_dim,
        "dataset": {"path": str(args.dataset), "sha256": file_sha256(args.dataset)},
        "context_encoder": (
            {"path": str(enc_path), "sha256": file_sha256(enc_path), "frozen": True}
            if enc_path is not None else None
        ),
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
class TDMPCController:
    """``BlueController`` wrapper around a TD-MPC agent (agent swappable per round).

    ``explore=True`` uses the upstream training-time action (sample from the
    elite distribution plus MPPI noise); ``False`` evaluates the best elite.
    """

    def __init__(self, agent, env, features: str = "markov", *, explore: bool = False):  # noqa: ANN001
        self.agent = agent
        self._act = jax.jit(
            lambda a, s, prev, ctx, key: a.act(
                world_model_observation(markov_state(env, s), feature_map=features),
                prev_plan=prev,
                mpc=True,
                deterministic=not explore,
                train=explore,
                context=ctx,
                key=key,
            )
        )

    def __call__(self, state, obs, context, carry, key, t):  # noqa: ANN001
        del obs, t
        action, plan = self._act(self.agent, state, carry, context, key)
        return action, plan


def tdmpc_controller(agent, env, features: str = "markov"):  # noqa: ANN001
    return TDMPCController(agent, env, features, explore=False)


def collect_online_round(
    agent,  # noqa: ANN001
    ds,  # noqa: ANN001
    replay: SequenceReplay,
    encoder: CausalContextEncoder | None,
    *,
    heldout: int,
    features: str,
    context_source: str,
    mode: str,
    episodes_per_group: int,
    logdir: Path,
    rng: np.random.Generator,
    horizon: int,
) -> dict[str, Any]:
    """Collect episodes with the current planner against the *training* specialists.

    Fresh reset/step keys, exploration noise on, causal context computed online
    from the real history (the same frozen encoder used for training data).
    """
    env = make_env("capture", continuous=True)
    controller = TDMPCController(agent, env, features, explore=True)
    train_ckpts = sorted(int(c) for c in set(ds.checkpoint_seed.tolist()) - {heldout})
    summary: dict[str, Any] = {"groups": [], "n_episodes": 0, "n_transitions": 0}
    for ckpt in train_ckpts:
        for label, pred_type in enumerate(OBJECTIVE_TYPES):
            red_params = load_continuous_actor_params(
                continuous_checkpoint_path(logdir, pred_type, "pred", ckpt)
            )
            base = int(rng.integers(0, 2**31 - 1))
            reset_keys = np.asarray(jax.random.split(jax.random.PRNGKey(base), episodes_per_group), np.uint32)
            step_seed = np.asarray(jax.random.split(jax.random.PRNGKey(base + 1), episodes_per_group), np.uint32)
            out = run_matched_episodes(
                env, red_params, controller, reset_keys, step_seed, horizon=horizon,
                context_mode="zero" if mode == "implicit" else "online",
                label=label, encoder=encoder,
                shuffle_seed=base, record_transitions=True,
            )
            tr = out["transitions"]
            if context_source == "oracle":
                ctx = np.repeat(np.eye(CONTEXT_DIM, dtype=np.float32)[label][None, None], horizon + 1, axis=1)
                ctx = np.repeat(ctx, episodes_per_group, axis=0)
            elif context_source in {"zero", "none"}:
                ctx = np.zeros((episodes_per_group, horizon + 1, CONTEXT_DIM), np.float32)
            else:
                ctx = encoder.causal_context(tr["prey_pos"], tr["pred_pos"], tr["valid_length"])
            if mode == "implicit":
                ctx = ctx[..., : replay.context.shape[-1]] * 0.0
            replay.append(tr, ctx, feature_map=features)
            summary["groups"].append(
                {
                    "checkpoint": ckpt,
                    "opponent": pred_type,
                    "n_episodes": int(episodes_per_group),
                    "blue_return_mean": float(np.mean(out["blue_return"])),
                    "captured_mean": float(np.mean(out["captured"])),
                    "resources_mean": float(np.mean(out["resources_collected"])),
                }
            )
            summary["n_episodes"] += int(episodes_per_group)
            summary["n_transitions"] += int(tr["valid_length"].sum())
    return summary


def factored_invariance_checks(agent, ds, eval_eps, context, seed: int, features: str = "markov") -> dict[str, Any] | None:  # noqa: ANN001
    """Numerical Gate 5 requirements for Equation 3 on real held-out latents."""
    m = agent.model
    if m.opponent_mode != "factored":
        return None
    rng = np.random.default_rng(seed)
    e = rng.choice(eval_eps, size=64)
    t = np.floor(rng.random(64) * (ds.valid_length[e] - 1)).astype(int)
    key = jax.random.PRNGKey(seed)
    obs = world_model_observation(ds.state[e, t], feature_map=features)
    x = m.encode(jnp.asarray(obs), m.encoder.params, key)
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
    if args.n_eps < 1:
        raise ValueError("n-eps must be positive")
    agent, manifest, stats = build_template(args.run)
    mode = manifest["mode"]
    heldout = int(manifest["heldout_checkpoint"])
    zero_s = ZeroSOpponent.load(args.run / "opponent.msgpack") if manifest.get("context_source") == "zero_s" else None
    ds = load_continuous_dataset(args.dataset)
    if zero_s is not None:
        if file_sha256(args.dataset) != manifest["dataset"]["sha256"]:
            raise ValueError("0s evaluation dataset does not match the training manifest")
        encoder, context = None, None
    else:
        source = "zero" if mode == "implicit" else "causal"
        encoder, enc_path, context = load_context(ds, args.bc_artifacts, heldout, args.bc_seed, source)
        if mode != "implicit" and file_sha256(enc_path) != manifest["context_encoder"]["sha256"]:
            raise ValueError("context encoder does not match the training manifest")
    eval_eps = np.flatnonzero(ds.checkpoint_seed == heldout)
    env = make_env("capture", continuous=True)
    prey_name = env.good_agents[0]
    obs_width = max(env.observation_space(a).shape[0] for a in env.agents)
    horizon = int(ds.blue_action.shape[1])
    context_modes = ["online", "zero", "shuffled", "wrong_oracle", "oracle"] if mode != "implicit" else ["zero"]
    if zero_s is not None:
        context_modes = ["online"]
    if args.context_modes:
        context_modes = args.context_modes.split(",")
    if mode == "implicit" and context_modes != ["zero"]:
        raise ValueError("Equation 1 has no context input; use --context-modes zero")
    features = manifest.get("features", "markov")
    blue = tdmpc_controller(agent, env, features)
    prey_path = continuous_checkpoint_path(args.logdir, "capture", "prey", heldout)
    prey_params = load_continuous_actor_params(prey_path) if args.controls else None
    output_dir = args.out or (args.run / "closed_loop" if zero_s is not None else args.run)
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = [np.flatnonzero((ds.checkpoint_seed == heldout) & (ds.objective_label == label))[:args.n_eps]
              for label in range(len(OBJECTIVE_TYPES))]
    if any(len(rows) != args.n_eps for rows in groups):
        raise ValueError("n-eps exceeds available held-out episodes for an opponent")
    if "shuffled" in context_modes and args.n_eps < 2:
        raise ValueError("shuffled context requires n-eps >= 2")
    if zero_s is not None:
        for rows in groups[1:]:
            if not np.array_equal(ds.environment_seed[rows], ds.environment_seed[groups[0]]):
                raise ValueError("0s evaluation requires matched reset keys across opponents")
    runs = []
    checkpoints = []
    per_episode: dict[str, np.ndarray] = {}
    for label, pred_type in enumerate(OBJECTIVE_TYPES):
        rows = groups[label]
        reset_keys, step_seed = ds.environment_seed[rows], ds.step_seed[rows]
        red_path = continuous_checkpoint_path(args.logdir, pred_type, "pred", heldout)
        if zero_s is not None:
            digest = file_sha256(red_path)
            expected = next(item["sha256"] for item in manifest["specialist_checkpoints"] if item["type"] == pred_type)
            if digest != expected:
                raise ValueError(f"specialist checkpoint does not match 0s manifest: {pred_type}")
            checkpoints.append({"type": pred_type, "path": str(red_path), "sha256": digest})
        red_params = load_continuous_actor_params(red_path)
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
                zero_s=zero_s if name == "tdmpc" else None, record_transitions=args.record,
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
            if "controller_seconds_per_batch" in out:
                seconds = out["controller_seconds_per_batch"]
                row["controller_timing"] = {
                    "first_two_calls_seconds": seconds[:2].tolist(),
                    "steady_median_seconds_per_batch": float(np.median(seconds[2:])) if len(seconds) > 2 else None,
                    "batch_size": len(rows),
                    "scope": "synchronized action calls; first two may compile; excludes context inference and environment stepping",
                }
            if args.record:
                from tag_objectives.rendering import save_episode_replay

                trace_path = output_dir / f"{key}.npz"
                np.savez_compressed(trace_path, **out["transitions"], environment_seed=reset_keys,
                                    step_seed=step_seed, dataset_episode=rows,
                                    objective_label=np.full(len(rows), label), checkpoint_seed=np.full(len(rows), heldout))
                # First matched episode, never selected for a favorable outcome.
                replay = save_episode_replay(out["transitions"], env, output_dir / key,
                                            title=f"Real environment · {pred_type} opponent · {name} / {cm}",
                                            camera=args.replay_camera)
                row["recordings"] = {"transitions": trace_path.name, **{k: v.name for k, v in replay.items()},
                                     "rendered_dataset_episode": int(rows[0]), "camera": args.replay_camera}
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
        "features": features,
        "seed": manifest["seed"],
        "heldout_checkpoint": heldout,
        "n_eps_per_opponent": args.n_eps,
        "planner": {k: manifest["config"]["tdmpc2"][k] for k in ("horizon", "population_size", "policy_prior_samples", "num_elites", "mppi_iterations")},
        "runs": runs,
        "factored_invariance": (None if zero_s is not None else
                                factored_invariance_checks(agent, ds, eval_eps, context, int(manifest["seed"]), features)),
    }
    if zero_s is not None:
        result.update(
            opponent_model="frozen_zero_s", context_dim=zero_s.encoder.config.lat,
            context_protocol="At decision t, only observed (state_s, red_action_s), s < t; frozen within each imagined horizon. Oracle modes use train-only class prototypes, not specialist actions.",
            scope="Saved identity-state Equation 3 controller, no training or parameter updates; small runs verify execution, not performance gains.",
            dataset={"path": str(args.dataset), "sha256": file_sha256(args.dataset)},
            specialist_checkpoints=checkpoints,
            checkpoint_artifacts={name: file_sha256(args.run / name) for name in
                                  ("agent.msgpack", "opponent.msgpack", "config.json", "state_stats.npz")},
            code={str(p.relative_to(_ROOT)): file_sha256(p) for p in
                  [Path(__file__).resolve(), _ROOT / "src/mopa/evaluation.py", _ROOT / "src/mopa/zero_s.py",
                   _ROOT / "src/mopa/tdmpc.py", _ROOT / "src/mopa/mppi.py",
                   *sorted((_ROOT / "src/tag_objectives").glob("*.py"))]},
        )
        if args.controls:
            result["prey_control_checkpoint"] = {"path": str(prey_path), "sha256": file_sha256(prey_path)}
    (output_dir / "evaluation.json").write_text(json.dumps(_jsonable(result), indent=2, sort_keys=True, allow_nan=False) + "\n")
    np.savez_compressed(output_dir / "evaluation_per_episode.npz", **per_episode)
    print(f"Wrote {output_dir / 'evaluation.json'}")
    return 0


# --------------------------------------------------------------------------- #
# compare
# --------------------------------------------------------------------------- #
_MODEL_ERROR_KEYS = ("model_mse", "persistence_mse", "ratio_model_over_persistence", "position_rmse_model", "position_rmse_persistence")
_REWARD_KEYS = ("explained_variance", "mae", "mse")
_TERMINATION_KEYS = ("brier", "ece_10_bins", "capture_auroc", "capture_recall_at_0.5", "hard_threshold_accuracy")


def _model_quality(manifest: dict[str, Any]) -> dict[str, Any]:
    """Gate 4 model-quality scalars of one training run (from its manifest)."""
    out: dict[str, Any] = {
        "seed": manifest["seed"],
        "features": manifest.get("features", "markov"),
        "total_updates": manifest.get("total_updates", manifest["updates"]),
        "final_replay_transitions": manifest.get("final_replay_transitions"),
    }
    for split in ("train", "heldout"):
        ev = manifest["evaluation"][split]
        entry: dict[str, Any] = {
            "model_error": {
                k: {kk: vv for kk, vv in row.items() if kk in _MODEL_ERROR_KEYS}
                for k, row in ev["model_error"]["per_horizon"].items()
                if row is not None
            },
            "reward": {k: ev["reward_calibration"][k] for k in _REWARD_KEYS},
        }
        if ev.get("termination_calibration"):
            entry["termination"] = {k: ev["termination_calibration"][k] for k in _TERMINATION_KEYS}
        out[split] = entry
    online = manifest.get("online", {}).get("log", [])
    out["online_rounds"] = [
        {
            "round": r["round"],
            "n_transitions": r["n_transitions"],
            "mean_collected_return": float(np.mean([g["blue_return_mean"] for g in r["groups"]])),
            "per_opponent": {
                typ: {
                    "blue_return_mean": float(np.mean([g["blue_return_mean"] for g in r["groups"] if g["opponent"] == typ])),
                    "captured_mean": float(np.mean([g["captured_mean"] for g in r["groups"] if g["opponent"] == typ])),
                }
                for typ in OBJECTIVE_TYPES
                if any(g["opponent"] == typ for g in r["groups"])
            },
        }
        for r in online
    ]
    return out


def _aggregate_model_quality(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean / std over seeds of every Gate 4 scalar in ``_model_quality`` output."""

    def agg(values: list[Any]) -> dict[str, float] | None:
        vals = [float(v) for v in values if v is not None]
        if not vals:
            return None
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}

    out: dict[str, Any] = {"n_seeds": len(per_seed)}
    for split in ("train", "heldout"):
        entry: dict[str, Any] = {"model_error": {}, "reward": {}}
        horizons = sorted({k for q in per_seed for k in q[split]["model_error"]}, key=int)
        for k in horizons:
            entry["model_error"][k] = {
                m: agg([q[split]["model_error"].get(k, {}).get(m) for q in per_seed]) for m in _MODEL_ERROR_KEYS
            }
        entry["reward"] = {m: agg([q[split]["reward"][m] for q in per_seed]) for m in _REWARD_KEYS}
        if all("termination" in q[split] for q in per_seed):
            entry["termination"] = {m: agg([q[split]["termination"][m] for q in per_seed]) for m in _TERMINATION_KEYS}
        out[split] = entry
    n_rounds = min((len(q["online_rounds"]) for q in per_seed), default=0)
    out["online_rounds"] = [
        {
            "round": i,
            "mean_collected_return": agg([q["online_rounds"][i]["mean_collected_return"] for q in per_seed]),
            "per_opponent": {
                typ: {
                    m: agg([q["online_rounds"][i]["per_opponent"].get(typ, {}).get(m) for q in per_seed])
                    for m in ("blue_return_mean", "captured_mean")
                }
                for typ in OBJECTIVE_TYPES
            },
        }
        for i in range(n_rounds)
    ]
    return out


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
        table.setdefault(key, {"seeds": [], "model_quality": [], "rows": {}})
        table[key]["seeds"].append(e["seed"])
        table[key]["model_quality"].append(_model_quality(m))
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
            "model_quality_per_seed": entry["model_quality"],
            "model_quality": _aggregate_model_quality(entry["model_quality"]),
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
    tr.add_argument("--profile", default="equation1")
    tr.add_argument("--mode", choices=("implicit", "conditioned", "factored"), default="implicit")
    tr.add_argument("--encoder", choices=("identity", "mlp"), default="identity")
    tr.add_argument("--context-source", choices=("causal", "oracle", "zero", "none"), default=None,
                    help="defaults to none for implicit, causal for other modes")
    tr.add_argument("--features", choices=FEATURE_MAPS, default="relative")
    tr.add_argument("--heldout", type=int, default=2)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--updates", type=int, default=10000)
    tr.add_argument("--log-every", type=int, default=500)
    tr.add_argument("--online-rounds", type=int, default=0)
    tr.add_argument("--online-episodes", type=int, default=8, help="per opponent x training checkpoint")
    tr.add_argument("--updates-per-round", type=int, default=2000)
    tr.add_argument("--logdir", type=Path, default=DEFAULT_CONTINUOUS_LOGDIR)
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
    ev.add_argument("--out", type=Path, help="evaluation output directory; defaults to RUN/closed_loop for 0s")
    ev.add_argument("--record", action="store_true", help="save real transitions and first matched episode GIF/PNG per evaluation arm")
    ev.add_argument("--replay-camera", choices=("arena", "full"), default="arena",
                    help="fixed arena-scale replay (default), or full-trajectory diagnostic view")
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
