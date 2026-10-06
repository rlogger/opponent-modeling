#!/usr/bin/env python3
"""Matched continuous control: implicit, BC, 0s, and equal-input PPO responses.

No environment changes. Frozen training opponents 0/1; checkpoint 2 is used
only after all predeclared training rounds. See the experiment PROTOCOL.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from functools import lru_cache, partial
from pathlib import Path

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mopa.action_decoder import (  # noqa: E402
    ActionDecoderConfig,
    fit_action_decoder_vae,
)
from mopa.bc_continuous import FrozenBCOpponent, fit_continuous_bc  # noqa: E402
from mopa.benchmark_report import summarize_benchmark  # noqa: E402
from mopa.continuous_data import (  # noqa: E402
    DEFAULT_CONTINUOUS_LOGDIR,
    OBJECTIVE_TYPES,
    continuous_checkpoint_path,
    load_continuous_actor_params,
    load_continuous_dataset,
    markov_state,
)
from mopa.evaluation import run_matched_episodes  # noqa: E402
from mopa.manifest import file_sha256, git_sha, package_versions  # noqa: E402
from mopa.response_ppo import (  # noqa: E402
    ResponsePPO,
    create_response,
    update_response,
    warm_start_response,
)
from mopa.tdmpc import create_agent, load_config  # noqa: E402
from mopa.tdmpc_data import SequenceReplay, state_statistics  # noqa: E402
from mopa.zero_s import ZeroSOpponent, zero_s_features  # noqa: E402
from tag_objectives import make_env  # noqa: E402

ARMS = ("implicit", "bc", "0s", "ppo", "ppo_z")
HISTORY_ARMS = ("0s", "ppo_z")


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.ndarray, jax.Array)):
        return np.asarray(value).tolist()
    return value.item() if isinstance(value, np.generic) else value


def write_json(path, value):
    pending = path.with_suffix(".pending.json")
    pending.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n")
    pending.replace(path)


def code_hashes():
    files = [Path(__file__), ROOT / "uv.lock", ROOT / "configs/tdmpc2.yaml",
             *sorted((ROOT / "src").rglob("*.py"))]
    return {str(p.relative_to(ROOT)): file_sha256(p) for p in files}


def specialist_binding(args):
    source = json.loads(args.dataset.with_suffix(".manifest.json").read_text())
    expected = {(r["seed"], r["objective"]): r["sha256"]
                for r in source["source_checkpoints"] if r["team"] == "pred"}
    result = {}
    for c in (0, 1, 2):
        for objective in OBJECTIVE_TYPES:
            path = continuous_checkpoint_path(args.logdir, objective, "pred", c)
            sha = file_sha256(path)
            if sha != expected[c, objective]:
                raise ValueError(f"specialist does not match the dataset provenance: {path}")
            result[f"{c}:{objective}"] = {"path": str(path.resolve()), "sha256": sha}
    return result


def keys(seed, round_index, group, batch, n, *, evaluation=False):
    # Evaluation resets are identical across objectives and arms within a seed.
    key = jax.random.PRNGKey(900_000 if evaluation else 100_000)
    for component in (seed, *( () if evaluation else (round_index, group, batch))):
        key = jax.random.fold_in(key, component)
    reset_key, step_key = jax.random.split(key)
    base = int(np.asarray(key)[0]) & 0x7FFFFFF0
    return (np.asarray(jax.random.split(reset_key, n)),
            np.asarray(jax.random.split(step_key, n)), base)


@lru_cache(maxsize=1)
def benchmark_env():
    return make_env("capture", continuous=True)


@partial(jax.jit, static_argnames=("training",))
def td_action(agent, state, context, carry, key, *, training):
    return agent.act(state, prev_plan=carry, mpc=True, deterministic=not training,
                     train=training, context=context, key=key)


@partial(jax.jit, static_argnames=("training",))
def ppo_action(agent, state, context, key, *, training):
    return agent.sample(state, context, key) if training else {"actions": agent.act(state, context, key)}


def replay_context(transitions):
    return np.concatenate([transitions["context"], transitions["final_context"][:, None]], axis=1)


def advantage_targets(reward, value, terminated, valid, discount=0.99, gae_lambda=0.95):
    """Bootstrap time/budget truncations, never captures or invalid padding."""
    reward, value = np.asarray(reward), np.asarray(value)
    valid, terminated = np.asarray(valid, bool), np.asarray(terminated, bool)
    if value.shape != (reward.shape[0], reward.shape[1] + 1) or valid.shape != reward.shape or terminated.shape != reward.shape:
        raise ValueError("unaligned episode arrays for GAE")
    advantage = np.zeros_like(reward, dtype=np.float32)
    carry = np.zeros(len(reward), np.float32)
    for t in reversed(range(reward.shape[1])):
        delta = reward[:, t] + discount * (~terminated[:, t]) * value[:, t + 1] - value[:, t]
        carry = np.where(valid[:, t], delta + discount * gae_lambda * (~terminated[:, t]) * carry, 0.0)
        advantage[:, t] = carry
    return advantage, np.where(valid, advantage + value[:, :-1], 0).astype(np.float32)


def offline_returns(data):
    reward = data["blue_reward"]
    result, carry = np.zeros_like(reward), np.zeros(len(reward), np.float32)
    for t in reversed(range(reward.shape[1])):
        carry = np.where(data["valid_mask"][:, t], reward[:, t] + 0.99 * carry, 0.0)
        result[:, t] = carry
    return result


class Controller:
    """One evaluator adapter; preserves exact PPO pre-tanh training samples."""

    def __init__(self, agent, env, *, training):
        self.agent, self.env, self.training = agent, env, training
        self.samples = {}
        self.is_ppo = isinstance(agent, ResponsePPO)

    def __call__(self, state, obs, context, carry, key, t):
        del obs
        raw = markov_state(self.env, state)
        if self.is_ppo:
            sample = ppo_action(self.agent, raw, context, key, training=self.training)
            if self.training:
                self.samples[t] = jax.tree.map(np.asarray, sample)
            return sample["actions"], carry
        return td_action(self.agent, raw, context, carry, key, training=self.training)


def prepare(args, seed, data, mean, std):
    out = args.out / f"seed_{seed}" / "shared"
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    binding = {"dataset": file_sha256(args.dataset), "code": code_hashes(), "seed": seed,
               "dataset_manifest": file_sha256(args.dataset.with_suffix(".manifest.json")),
               "specialists": args.specialists,
               "encoder_steps": args.encoder_steps, "smoke": args.smoke}
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        if any(saved[k] != v for k, v in binding.items()):
            raise ValueError(f"shared artifact configuration/source changed: {out}")
        for name, sha in saved["artifacts"].items():
            if file_sha256(out / name) != sha:
                raise ValueError(f"shared artifact changed: {out / name}")
    else:
        print(f"[seed {seed}] fitting train-only 0s and vanilla BC", flush=True)
        cfg = ActionDecoderConfig(action_type="continuous", steps=args.encoder_steps)
        fit = fit_action_decoder_vae(np.asarray(zero_s_features(data["state"][:, :-1])),
                                    data["red_action"], data["valid_length"], np.arange(len(data["state"])),
                                    jax.random.PRNGKey(seed), config=cfg)
        opponent = ZeroSOpponent.from_fit(fit, data["objective_label"], np.arange(len(data["state"])))
        opponent.save(out / "0s.msgpack")
        context = opponent.context(data["state"], data["red_action"], data["valid_length"])
        np.savez_compressed(out / "context.npz", context=context)
        valid = data["valid_mask"]
        bc = FrozenBCOpponent(fit_continuous_bc(
            np.asarray(zero_s_features(data["state"][:, :-1]))[valid], data["red_action"][valid], seed,
            steps=args.encoder_steps, batch_size=128, hidden_size=64,
            metadata={"training_checkpoints": [0, 1], "feature_schema": "zero_s_current_state_8d"}))
        bc.save(out / "bc.npz")
        np.savez(out / "stats.npz", mean=mean, std=std)
        write_json(manifest_path, {**binding, "training_checkpoints": [0, 1],
                   "artifacts": {n: file_sha256(out / n) for n in ("0s.msgpack", "bc.npz", "context.npz", "stats.npz")}})
    return (ZeroSOpponent.load(out / "0s.msgpack"), FrozenBCOpponent.load(out / "bc.npz"),
            np.load(out / "context.npz")["context"])


def ppo_batch(agent, tr, samples):
    context = replay_context(tr)
    values = np.asarray(jax.jit(agent.value)(jnp.asarray(tr["state"]), jnp.asarray(context)))
    advantage, returns = advantage_targets(tr["blue_reward"], values, tr["terminated_capture"], tr["valid_mask"])
    b, t = tr["blue_action"].shape[:2]
    batch = {"observations": tr["state"][:, :-1], "context": tr["context"],
             "actions": tr["blue_action"], "returns": returns, "advantages": advantage,
             "valid_mask": tr["valid_mask"]}
    for name, shape in (("pre_tanh", (b, t, 2)), ("log_probs", (b, t)), ("old_values", (b, t))):
        batch[name] = np.zeros(shape, np.float32)
        for step, sample in samples.items():
            batch[name][:, step] = sample[name]
    return batch


def collect(args, agent, opponent, seed, round_index, out, params):
    """Same exact valid-transition quota per objective/checkpoint/round/arm."""
    env = benchmark_env()
    controller = Controller(agent, env, training=True)
    trajectories, ppo_batches, logs = [], [], []
    for group, (checkpoint, label) in enumerate((c, k) for c in (0, 1) for k in range(3)):
        remaining, batch_index = args.transitions_per_group, 0
        while remaining:
            reset, step_keys, base = keys(seed, round_index, group, batch_index, args.batch_episodes)
            controller.samples = {}
            result = run_matched_episodes(
                env, params[checkpoint, label], controller, reset, step_keys, horizon=100,
                context_mode="online" if opponent is not None else "zero", label=label,
                zero_s=opponent, context_width=None if opponent is not None else 0,
                max_transitions=remaining, shuffle_seed=base, record_transitions=True)
            tr = result["transitions"]
            n = int(tr["valid_length"].sum())
            if n < 1 or n > remaining:
                raise RuntimeError("transition quota failed to make valid progress")
            if controller.is_ppo:
                ppo_batches.append(ppo_batch(agent, tr, controller.samples))
            ppo_recording = ({f"ppo_{name}": ppo_batches[-1][name] for name in
                              ("pre_tanh", "log_probs", "old_values", "returns", "advantages")}
                             if controller.is_ppo else {})
            path = out / f"round_{round_index:02d}_group_{group}_batch_{batch_index:03d}.npz"
            np.savez_compressed(path, **tr, **ppo_recording, environment_seed=reset, step_seed=step_keys,
                                checkpoint_seed=np.full(len(reset), checkpoint),
                                objective_label=np.full(len(reset), label))
            trajectories.append(tr)
            logs.append({"file": path.name, "sha256": file_sha256(path), "checkpoint": checkpoint,
                         "objective": OBJECTIVE_TYPES[label], "valid_transitions": n})
            remaining -= n
            batch_index += 1
    return trajectories, ppo_batches, logs


def checkpoint(agent, out, manifest):
    pending = out / "agent.pending.msgpack"
    if isinstance(agent, ResponsePPO):
        agent.save(pending)
    else:
        pending.write_bytes(flax.serialization.to_bytes(agent))
    pending.replace(out / "agent.msgpack")
    write_json(out / "manifest.json", {**manifest, "agent_sha256": file_sha256(out / "agent.msgpack")})


def train_and_evaluate(args, seed, arm, data, mean, std, shared, params):
    out = args.out / f"seed_{seed}" / arm
    out.mkdir(parents=True, exist_ok=True)
    zero_s, bc, all_context = shared
    context = all_context if arm in HISTORY_ARMS else all_context[..., :0]
    opponent = zero_s if arm in HISTORY_ARMS else None
    config = load_config(profile="smoke" if args.smoke else "gate4")
    config.update(opponent_mode="implicit" if arm == "implicit" else "factored", context_dim=context.shape[-1])
    config["encoder"]["type"] = "identity"
    config["world_model"].update(hidden_dim=16 if args.smoke else 128, predict_continues=True)
    config["tdmpc2"]["continue_loss_scale"] = 1.0
    config["factored"]["red_loss_scale"] = 0.0
    is_ppo = arm.startswith("ppo")
    agent = (create_response(mean, std, context_dim=context.shape[-1], seed=seed, hidden=16 if args.smoke else 128)
             if is_ppo else create_agent(config, 66, key=jax.random.PRNGKey(seed), obs_mean=mean, obs_std=std))
    if arm == "0s":
        agent = zero_s.attach(agent, mean, std)
    elif arm == "bc":
        agent = bc.attach(agent, mean, std)
    expected_red = (None if is_ppo or arm == "implicit"
                    else flax.serialization.to_bytes(agent.model.red_model.params))
    replay = None if is_ppo else SequenceReplay.from_dataset(data, np.arange(len(data["state"])), agent.horizon, context)
    binding = {"arm": arm, "seed": seed, "code": code_hashes(), "dataset": file_sha256(args.dataset),
               "dataset_path": str(args.dataset.resolve()),
               "dataset_manifest_path": str(args.dataset.with_suffix(".manifest.json").resolve()),
               "dataset_manifest": file_sha256(args.dataset.with_suffix(".manifest.json")),
               "specialists": args.specialists,
               "shared_manifest": file_sha256(args.out / f"seed_{seed}" / "shared" / "manifest.json"),
               "budget": {k: getattr(args, k) for k in ("offline_updates", "rounds", "updates_per_round", "transitions_per_group", "batch_episodes", "eval_episodes", "smoke")}}
    ppo_config = ({name: getattr(agent, name) for name in
                   ("context_dim", "hidden", "learning_rate", "clip_epsilon", "entropy_coefficient", "value_coefficient", "max_grad_norm")}
                  if is_ppo else {})
    manifest = {**binding, "config": config if not is_ppo else {
                    **ppo_config, "method": "PPO response", "epochs": 4, "minibatch": 128,
                    "discount": 0.99, "gae_lambda": 0.95, "warm_start": "offline BC plus finite-episode-return value regression"},
                "git_sha": git_sha(ROOT), "dependencies": package_versions(), "rounds": [],
                "offline_transitions": int(data["valid_mask"].sum()), "online_transitions": 0,
                "train_checkpoints": [0, 1], "heldout_checkpoint": 2, "status": "training"}
    previous = out / "manifest.json"
    rng, update_key = np.random.default_rng(seed), jax.random.PRNGKey(10_000 + seed)
    started = time.monotonic()

    def update_world(n):
        nonlocal agent, update_key
        for i in range(n):
            update_key, k = jax.random.split(update_key)
            agent, info = agent.update(**replay.sample(rng, agent.batch_size), key=k)
            if i == n - 1:
                info = {k: float(np.asarray(info[k])) for k in
                        ("total_loss", "consistency_loss", "reward_loss", "value_loss", "continue_loss", "red_loss", "policy_loss")}
                if not np.isfinite(list(info.values())).all():
                    raise FloatingPointError("nonfinite world-model loss")
                return info

    if previous.exists():
        manifest = json.loads(previous.read_text())
        if any(manifest[k] != v for k, v in binding.items()):
            raise ValueError(f"refusing source/config-mismatched resume: {out}")
        if file_sha256(out / "agent.msgpack") != manifest["agent_sha256"]:
            raise ValueError("controller checkpoint hash mismatch")
        agent = ResponsePPO.load(out / "agent.msgpack") if is_ppo else flax.serialization.from_bytes(agent, (out / "agent.msgpack").read_bytes())
        rng.bit_generator.state = manifest["replay_rng"]
        update_key = jnp.asarray(manifest["update_key"], jnp.uint32)
        for row in manifest["rounds"]:
            for record in row["data"]:
                path = out / record["file"]
                if file_sha256(path) != record["sha256"]:
                    raise ValueError(f"replay hash mismatch: {path}")
                if replay is not None:
                    with np.load(path) as saved:
                        if not np.isin(saved["checkpoint_seed"], [0, 1]).all():
                            raise ValueError("held-out opponent in training replay")
                        tr = dict(saved)
                    replay.append(tr, replay_context(tr))
    else:
        print(f"[{arm} seed {seed}] offline training {args.offline_updates} updates", flush=True)
        if is_ppo:
            agent, info = warm_start_response(agent, {
                "observations": data["state"][:, :-1], "context": context[:, :-1],
                "actions": data["blue_action"], "returns": offline_returns(data), "valid_mask": data["valid_mask"],
            }, update_key, steps=args.offline_updates, minibatch_size=128)
        else:
            info = update_world(args.offline_updates)
        manifest.update(offline_metrics=info, replay_rng=rng.bit_generator.state, update_key=update_key)
        checkpoint(agent, out, manifest)
    if expected_red is not None and expected_red != flax.serialization.to_bytes(agent.model.red_model.params):
        raise RuntimeError("saved/trained decoder does not match the frozen shared opponent")
    for round_index in range(len(manifest["rounds"]), args.rounds):
        print(f"[{arm} seed {seed}] round {round_index + 1}/{args.rounds}, quota {6 * args.transitions_per_group} real transitions", flush=True)
        frozen_hash = file_sha256(args.out / f"seed_{seed}" / "shared" / "0s.msgpack")
        frozen_red = None if is_ppo or arm == "implicit" else flax.serialization.to_bytes(agent.model.red_model.params)
        tr, ppo_data, logs = collect(args, agent, opponent, seed, round_index, out, params)
        if is_ppo:
            batch = {k: np.concatenate([b[k] for b in ppo_data]) for k in ppo_data[0]}
            update_key, k = jax.random.split(update_key)
            agent, info = update_response(agent, batch, k, epochs=4, minibatch_size=128)
        else:
            for episode_batch in tr:
                replay.append(episode_batch, replay_context(episode_batch))
            info = update_world(args.updates_per_round)
            if frozen_red is not None and frozen_red != flax.serialization.to_bytes(agent.model.red_model.params):
                raise RuntimeError("frozen opponent decoder changed")
        if frozen_hash != file_sha256(args.out / f"seed_{seed}" / "shared" / "0s.msgpack"):
            raise RuntimeError("frozen encoder changed")
        n = sum(row["valid_transitions"] for row in logs)
        if n != 6 * args.transitions_per_group:
            raise RuntimeError("unequal training interaction quota")
        manifest["online_transitions"] += n
        manifest["rounds"].append({"round": round_index, "data": logs, "metrics": info,
                                   "online_transitions": manifest["online_transitions"]})
        manifest.update(replay_rng=rng.bit_generator.state, update_key=update_key)
        checkpoint(agent, out, manifest)
        print(f"[{arm} seed {seed}] saved {manifest['online_transitions']} real transitions ({time.monotonic() - started:.1f}s this process)", flush=True)
    result_path = out / "evaluation.json"
    if result_path.exists():
        if manifest.get("status") != "complete" or file_sha256(result_path) != manifest.get("evaluation_sha256"):
            raise ValueError("partial or changed evaluation; preserve and use a new output")
        return
    print(f"[{arm} seed {seed}] final held-out evaluation", flush=True)
    env = benchmark_env()
    controller = Controller(agent, env, training=False)
    records, trace_hashes = [], {}
    reset, step_keys, base = keys(seed, 0, 0, 0, args.eval_episodes, evaluation=True)
    for label, objective in enumerate(OBJECTIVE_TYPES):
        result = run_matched_episodes(
            env, params[2, label], controller, reset, step_keys, horizon=100,
            context_mode="online" if opponent is not None else "zero", label=label,
            zero_s=opponent, context_width=None if opponent is not None else 0,
            shuffle_seed=base, record_transitions=True)
        trace = out / f"heldout_{objective}.npz"
        np.savez_compressed(trace, **result["transitions"], environment_seed=reset, step_seed=step_keys)
        trace_hashes[trace.name] = file_sha256(trace)
        records.append({"arm": arm, "seed": seed, "objective": objective, "episode_keys": reset,
                        "metrics": {k: result[k] for k in ("blue_return", "captured", "resources_collected")}})
        print(f"[{arm} seed {seed}] {objective}: {np.mean(result['blue_return']):+.3f}", flush=True)
    write_json(result_path, records)
    if args.specialists != specialist_binding(args):
        raise RuntimeError("specialist checkpoints changed during the run")
    manifest.update(status="complete", evaluation_sha256=file_sha256(result_path),
                    evaluation_traces=trace_hashes,
                    seconds_this_process=time.monotonic() - started)
    checkpoint(agent, out, manifest)


def summarize(args):
    from mopa.benchmark_report import audit_benchmark
    audit = audit_benchmark(args.out, ARMS, args.seeds, smoke=args.smoke)
    write_json(args.out / "verification.json", audit)
    records = []
    for seed in args.seeds:
        for arm in ARMS:
            out = args.out / f"seed_{seed}" / arm
            manifest = json.loads((out / "manifest.json").read_text())
            if manifest["status"] != "complete" or file_sha256(out / "evaluation.json") != manifest["evaluation_sha256"]:
                raise ValueError("cannot summarize incomplete or changed results")
            records.extend(json.loads((out / "evaluation.json").read_text()))
    result = summarize_benchmark(records, ARMS, args.seeds)
    write_json(args.out / "comparison.json", result)
    lines = ["# Matched continuous-control benchmark", "", "SMOKE ONLY; not performance evidence." if args.smoke else
             "Means ± sample SD across independently trained seed means, not episode SD.", "",
             "| Controller | Capture | Risk | Curious | Overall |", "|---|---:|---:|---:|---:|"]
    for arm in ARMS:
        r = result["results"][arm]["blue_return"]
        cells = [r["by_objective"][o] for o in OBJECTIVE_TYPES] + [r["overall"]]
        lines.append("| " + arm + " | " + " | ".join(f"{c['mean']:+.2f} ± {c['seed_std']:.2f}" for c in cells) + " |")
    lines += ["", "State-only matched inputs: implicit, BC, PPO. State plus identical causal 0s context: 0s, PPO-z.",
              "0s versus BC/implicit changes history use as well as the model; it does not isolate latent representation quality.",
              "All arms get the same training-only offline dataset and exact fresh valid-transition quota. PPO uses an explicit offline BC/value warm-start; TD-MPC uses replay.",
              "Training checkpoints 0/1 only; final evaluation checkpoint 2. No test-based checkpoint selection. Three seeds remain a limited uncertainty estimate.",
              "", "[Protocol](PROTOCOL.md) · [Full metrics and paired intervals](comparison.json)", ""]
    (args.out / "REPORT.md").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true",
                        help="Explicitly opt in to experiment fitting, rollouts, evaluation or aggregation")
    parser.add_argument("--phase", choices=("all", "prepare", "train", "summarize"), default="all")
    parser.add_argument("--dataset", type=Path, default=ROOT / "experiments/main_env_20260908/continuous_data/dataset.npz")
    parser.add_argument("--logdir", type=Path, default=ROOT / DEFAULT_CONTINUOUS_LOGDIR)
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/matched_control_20260908")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--encoder-steps", type=int, default=1500)
    parser.add_argument("--offline-updates", type=int, default=2000)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--updates-per-round", type=int, default=1000)
    parser.add_argument("--transitions-per-group", type=int, default=600)
    parser.add_argument("--batch-episodes", type=int, default=8)
    parser.add_argument("--eval-episodes", type=int, default=24)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute:
        print("Implementation-only mode. No fitting, rollouts, evaluation or metrics were run. "
              "See docs/control-pipeline.md. Experiments require explicit --execute.")
        return 0
    args.seeds = [int(s) for s in args.seeds.split(",")]
    args.arms = args.arms.split(",")
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds) or not args.seeds or not set(args.arms) <= set(ARMS):
        parser.error("distinct nonnegative seeds and known arms required")
    if args.smoke:
        if args.out == ROOT / "experiments/matched_control_20260908":
            parser.error("smoke requires a separate explicit output directory")
        args.encoder_steps = args.offline_updates = args.updates_per_round = 2
        args.rounds, args.transitions_per_group, args.batch_episodes, args.eval_episodes = 1, 4, 2, 2
    if any(getattr(args, k) < 1 for k in ("encoder_steps", "offline_updates", "rounds", "updates_per_round", "transitions_per_group", "batch_episodes", "eval_episodes")):
        parser.error("all budgets must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.phase == "summarize":
        summarize(args)
        return 0
    ds = load_continuous_dataset(args.dataset)
    args.specialists = specialist_binding(args)
    if set(np.unique(ds.checkpoint_seed)) != {0, 1, 2}:
        raise ValueError("benchmark requires training checkpoints 0/1 and held-out checkpoint 2")
    train = np.flatnonzero(np.isin(ds.checkpoint_seed, [0, 1]))
    data = {k: v[train] for k, v in ds.as_dict().items()}
    mean, std = state_statistics(data["state"], data["valid_mask"], feature_map="markov")
    params = {(c, k): load_continuous_actor_params(continuous_checkpoint_path(args.logdir, objective, "pred", c))
              for c in (0, 1, 2) for k, objective in enumerate(OBJECTIVE_TYPES)}
    for seed in args.seeds:
        shared = prepare(args, seed, data, mean, std)
        if args.phase != "prepare":
            for arm in args.arms:
                train_and_evaluate(args, seed, arm, data, mean, std, shared, params)
    if args.phase == "all" and set(args.arms) == set(ARMS) and len(args.seeds) >= 2:
        summarize(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
