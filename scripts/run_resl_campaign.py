#!/usr/bin/env python3
"""Fresh source-bound execution of the documented five-arm control comparison.

Uses the established controllers, 0s fitting and PPO implementation. The
protocol supplies every budget; pilot results remain feasibility evidence.
Historical output is never changed. EVAL-02's representation comparison uses
a separate protocol and fresh controller fits, following causal prediction
validation; it never substitutes a sixth arm into the original benchmark.
"""
from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import run_control_benchmark as historical  # noqa: E402

from mopa.action_decoder import (  # noqa: E402
    ActionDecoderConfig,
    fit_action_decoder_vae,
)
from mopa.bc_continuous import FrozenBCOpponent, fit_continuous_bc  # noqa: E402
from mopa.benchmark_report import summarize_benchmark  # noqa: E402
from mopa.causal_opponent import (  # noqa: E402
    CausalOpponent,
    CausalOpponentConfig,
    fit_causal_opponent,
    match_encoder_capacity,
)
from mopa.continuous_data import (  # noqa: E402
    OBJECTIVE_TYPES,
    continuous_checkpoint_path,
    load_continuous_actor_params,
    load_continuous_dataset,
    markov_state,
)
from mopa.evaluation import specialist_action_function  # noqa: E402
from mopa.manifest import file_sha256, package_versions  # noqa: E402
from mopa.response_ppo import (  # noqa: E402
    ResponsePPO,
    create_response,
    update_response,
    warm_start_response,
)
from mopa.tdmpc import check_update, create_agent, load_config  # noqa: E402
from mopa.tdmpc_data import SequenceReplay, state_statistics  # noqa: E402
from mopa.zero_s import ZeroSOpponent, zero_s_features  # noqa: E402

ARMS = historical.ARMS
REPRESENTATION_ARMS = ("implicit", "bc", "history", "causal_vae", "ppo", "ppo_history")
WORLD_REPRESENTATION_ARMS = ("implicit", "implicit_mlp")
CONTEXT_MODEL = {"0s": "0s", "ppo_z": "0s", "history": "history",
                 "causal_vae": "causal_vae", "ppo_history": "causal_vae"}
write_json = historical.write_json
SCHEMA = "resl_control_campaign_v1"


def source_hashes():
    files = {Path(__file__), ROOT / "scripts/run_control_benchmark.py", ROOT / "uv.lock",
             ROOT / "configs/tdmpc2.yaml", ROOT / "experiments/matched_control_20260908/verify.py",
             *list((ROOT / "src").rglob("*.py"))}
    return {str(p.relative_to(ROOT)): file_sha256(p) for p in sorted(files)}


def representation_configurations(cfg):
    """Derive then verify every predeclared predictor before any fitting."""
    selected = CausalOpponentConfig(**{**cfg["predictor_config"], "steps": cfg["encoder_steps"]})
    reference = replace(selected, method="recurrent_vae", encoder_hid=None)
    history, gap = match_encoder_capacity(replace(selected, method="deterministic_history",
        encoder_hid=None, beta=0., sample_training=False), reference)
    bc = replace(selected, method="bc", history=0, max_history=None, encoder_hid=None,
                 beta=0., sample_training=False)
    values = {"bc": asdict(bc), "history": asdict(history), "causal_vae": asdict(selected)}
    if values != cfg.get("predictor_configurations"):
        raise ValueError("freeze the derived per-arm predictor configurations in the private protocol")
    if gap != cfg.get("history_encoder_parameter_gap"):
        raise ValueError("freeze the unavoidable nearest-width parameter-count gap")
    return values


def protocol_configuration(protocol, stage):
    """No budget defaults, silent profile reduction, or six-arm substitution."""
    cfg = dict(protocol[f"control_{stage}"])
    study = cfg.get("study", "architecture")
    arms = {"architecture": ARMS, "representation": REPRESENTATION_ARMS,
            "world_representation": WORLD_REPRESENTATION_ARMS}.get(study, ())
    if not arms or tuple(cfg["arms"]) != arms:
        raise ValueError("explicit separate architecture/representation arm set required")
    if study == "representation":
        required = {"feature_schema", "history", "latent_dim", "hid", "batch", "learning_rate",
                    "beta", "free_bits", "sample_training", "objective", "method"}
        if not required <= cfg.get("predictor_config", {}).keys():
            raise ValueError("representation study requires the frozen causal predictor configuration")
        if cfg["predictor_config"]["method"] not in {"mlp_vae", "recurrent_vae"}:
            raise ValueError("declare the VAE selected before controller evaluation")
        if cfg["predictor_config"]["objective"] != "past_next_action" or cfg["predictor_config"]["latent_dim"] != 8:
            raise ValueError("causal controller comparison requires the declared past-only 8D context")
    seeds = protocol[f"{stage}_seeds"]
    if not seeds or any(type(s) is not int or s < 0 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("distinct nonnegative fitting seeds required")
    for key in ("encoder_steps", "offline_updates", "rounds", "updates_per_round",
                "transitions_per_training_group", "batch_episodes", "eval_episodes_per_objective",
                "ppo_epochs", "ppo_minibatch"):
        if type(cfg.get(key)) is not int or cfg[key] < 1:
            raise ValueError(f"positive explicit protocol budget required: {key}")
    if cfg["terminal_contract"] not in {"finite_100_step_v1", "capture_only_bootstrap_timeouts"}:
        raise ValueError("explicit documented timeout target required")
    fixed_planner = {"planner_profile": "gate4", "horizon": 3, "population_size": 512,
                     "policy_prior_samples": 24, "num_elites": 64, "mppi_iterations": 6}
    if any(cfg.get(k) != v for k, v in fixed_planner.items()):
        raise ValueError("retain the documented gate4 planner for the architecture comparison")
    if cfg["batch_episodes"] != 1:
        raise ValueError("random specialist selection per episode requires training batch_episodes=1")
    if cfg["evaluation_batch_episodes"] != cfg["eval_episodes_per_objective"]:
        raise ValueError("evaluation currently uses one paired batch per objective")
    if cfg["opponent_sampling"] != "random next eligible training specialist at each episode under exact per-type transition quotas":
        raise ValueError("unsupported specialist sampling contract")
    train = cfg["training_checkpoints"]
    if not train or len(set(train)) != len(train) or not set(train) <= {0, 1}:
        raise ValueError("explicit training checkpoint list required")
    if cfg["evaluation_checkpoint"] in train:
        raise ValueError("training and evaluation checkpoint families overlap")
    if stage == "pilot" and cfg["evaluation_checkpoint"] != 1:
        raise ValueError("pilot must evaluate validation checkpoint 1 only")
    if stage == "main":
        fixed = {"encoder_steps": 1500, "offline_updates": 2000, "rounds": 6,
                 "updates_per_round": 1000, "transitions_per_training_group": 600,
                 "eval_episodes_per_objective": 24, "ppo_epochs": 4, "ppo_minibatch": 128}
        if seeds != [0, 1, 2] or train != [0, 1] or cfg["evaluation_checkpoint"] != 2:
            raise ValueError("main must preserve predeclared seeds and checkpoint split")
        if any(cfg[k] != v for k, v in fixed.items()):
            raise ValueError("main budget differs from the documented five-arm protocol")
    if study == "representation":
        representation_configurations(cfg)
    if study == "world_representation" and cfg.get("world_encoders") != {
        "implicit": "identity", "implicit_mlp": "mlp"}:
        raise ValueError("world representation comparison requires explicit identity/mlp global-state inputs")
    return cfg, seeds


def value_terminals(trace, contract):
    """Keep raw capture/truncation flags; derive only the learning target mask."""
    valid = np.asarray(trace["valid_mask"], bool)
    captured = np.asarray(trace["terminated_capture"], bool)
    state = np.asarray(trace["state"])
    if state.shape != (*valid.shape[:1], valid.shape[1] + 1, 66) or captured.shape != valid.shape:
        raise ValueError("unaligned 66D trace and terminal flags")
    if not np.isfinite(state).all() or np.any(captured & ~valid):
        raise ValueError("invalid state or capture evidence")
    times = state[..., 65]
    if np.any(times < 0) or np.any(times > 1 + 1e-6):
        raise ValueError("physical time must stay within the 100-step episode")
    if contract == "finite_100_step_v1":
        return valid & (captured | (times[:, 1:] >= 1 - 1e-6))
    if contract == "capture_only_bootstrap_timeouts":
        return captured & valid
    raise ValueError("unknown timeout target contract")


def training_trace(trace, contract):
    """SequenceReplay's historical field name is adapted without editing raw data."""
    terminal = value_terminals(trace, contract)
    return {**trace, "terminated_capture": terminal,
            "truncated_timeout": np.asarray(trace["truncated_timeout"], bool) & ~terminal}


def ppo_batch(agent, trace, samples, contract):
    context = historical.replay_context(trace)
    values = np.asarray(jax.jit(agent.value)(jnp.asarray(trace["state"]), jnp.asarray(context)))
    advantage, returns = historical.advantage_targets(
        trace["blue_reward"], values, value_terminals(trace, contract), trace["valid_mask"])
    batch = {"observations": trace["state"][:, :-1], "context": trace["context"],
             "actions": trace["blue_action"], "returns": returns, "advantages": advantage,
             "valid_mask": trace["valid_mask"]}
    b, t = trace["valid_mask"].shape
    for name, shape in (("pre_tanh", (b, t, 2)), ("log_probs", (b, t)), ("old_values", (b, t))):
        batch[name] = np.zeros(shape, np.float32)
        for step, sample in samples.items():
            batch[name][:, step] = sample[name]
    return batch


def verify_files(directory, hashes):
    for name, expected in hashes.items():
        if file_sha256(directory / name) != expected:
            raise ValueError(f"changed artifact: {directory / name}")


def protocol_comparisons(statistics, arms, study):
    """Use frozen contrasts; historical reporter defaults cannot redefine them."""
    names = statistics[f"primary_{study}_comparisons"]
    pairs = [tuple(name.split(" minus ")) for name in names]
    if not pairs or len(set(pairs)) != len(pairs) or any(
        len(pair) != 2 or pair[0] not in arms or pair[1] not in arms or pair[0] == pair[1]
        for pair in pairs
    ):
        raise ValueError("protocol comparisons must be distinct declared arm pairs")
    threshold = statistics["practically_meaningful_return_difference"]
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool) or not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("predeclared practical return difference must be positive and finite")
    return pairs


def report_comparisons(records, binding):
    cfg, statistics = binding["configuration"], binding["statistics"]
    pairs = protocol_comparisons(statistics, cfg["arms"], cfg.get("study", "architecture"))
    result = summarize_benchmark(records, cfg["arms"], binding["seeds"], comparisons=pairs)
    threshold = statistics["practically_meaningful_return_difference"]
    result["predeclared_statistics"] = statistics
    result["practical_return_comparisons"] = {
        name: {"threshold": threshold, "mean_difference": row["metrics"]["blue_return"]["overall"]["mean"],
               "status": "meets mean-difference criterion" if row["metrics"]["blue_return"]["overall"]["mean"] >= threshold else "failed mean-difference criterion",
               "scope": "descriptive threshold comparison; uncertainty remains in paired comparison intervals"}
        for name, row in result["comparisons"].items()
    }
    return result


def bind_protocol(args):
    private = args.spec_repo.resolve()
    path = args.protocol.resolve()
    relative = path.relative_to(private)
    committed = subprocess.check_output(["git", "show", f"{args.spec_commit}:{relative}"], cwd=private)
    if len(args.spec_commit) != 40 or committed != path.read_bytes():
        raise ValueError("protocol must match the exact committed private specification")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if args.code_commit != head:
        raise ValueError("executable HEAD does not match the requested code commit")
    paths = ["src", "scripts", "configs", "uv.lock", "experiments/matched_control_20260908/verify.py"]
    tracked = subprocess.check_output(["git", "diff", "--name-only", "HEAD", "--", *paths], cwd=ROOT, text=True)
    untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "--", *paths], cwd=ROOT, text=True)
    if tracked.strip() or untracked.strip():
        raise ValueError("commit reviewed executable source before execution")
    protocol = json.loads(committed)
    configuration, seeds = protocol_configuration(protocol, args.stage)
    protocol_comparisons(protocol["statistics"], configuration["arms"], configuration.get("study", "architecture"))
    if file_sha256(args.dataset) != protocol["dataset_sha256"]:
        raise ValueError("dataset differs from protocol")
    return protocol, configuration, seeds, {
        "schema": SCHEMA, "stage": args.stage,
        "specification_commit": args.spec_commit, "protocol_path": str(relative),
        "protocol_sha256": file_sha256(path), "protocol_id": protocol["protocol_id"],
        "protocol_version": protocol["version"], "executable_commit": head,
        "source_hashes": source_hashes(), "configuration": configuration, "seeds": seeds,
        "statistics": protocol["statistics"],
        "dataset_path": str(args.dataset.resolve()), "dataset_sha256": file_sha256(args.dataset),
        "dataset_manifest_sha256": file_sha256(args.dataset.with_suffix(".manifest.json")),
        "specialists": historical.specialist_binding(args), "runtime": package_versions(),
    }


def prepare_shared(out, seed, data, mean, std, binding):
    shared = out / f"seed_{seed}" / "shared"
    path = shared / "manifest.json"
    study = binding["configuration"].get("study", "architecture")
    if binding["source_hashes"] != source_hashes():
        raise ValueError("source changed before fitting or shared-artifact restore")
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["binding"] != binding or saved["seed"] != seed or saved["status"] != "complete":
            raise ValueError("shared source binding changed or incomplete fit; preserve and use fresh output")
        verify_files(shared, saved["artifacts"])
    elif study == "world_representation":
        shared.mkdir(parents=True, exist_ok=False)
        np.savez(shared / "stats.npz", mean=mean, std=std)
        write_json(path, {"binding": binding, "seed": seed, "status": "complete",
                         "fit_seconds": 0., "opponent_fitting": "none",
                         "artifacts": {"stats.npz": file_sha256(shared / "stats.npz")}})
    elif study == "representation":
        shared.mkdir(parents=True, exist_ok=False)
        write_json(path, {"binding": binding, "seed": seed, "status": "fitting"})
        cfg = binding["configuration"]
        started = time.monotonic()
        configurations = {}
        for name, values in representation_configurations(cfg).items():
            predictor_cfg = CausalOpponentConfig(**values)
            model, history = fit_causal_opponent(
                data["state"], data["red_action"], data["valid_length"], np.arange(len(data["state"])),
                jax.random.PRNGKey(seed), config=predictor_cfg, labels=data["objective_label"],
                training_state_path=shared / f"{name}_training_state.msgpack")
            model.save(shared / f"{name}.msgpack")
            write_json(shared / f"{name}_training_history.json", history)
            np.savez_compressed(shared / f"{name}_context.npz",
                                context=model.context(data["state"], data["red_action"], data["valid_length"]))
            configurations[name] = asdict(predictor_cfg)
        np.savez(shared / "stats.npz", mean=mean, std=std)
        if binding["source_hashes"] != source_hashes():
            raise RuntimeError("source changed while fitting shared predictors")
        write_json(path, {"binding": binding, "seed": seed, "status": "complete",
                         "predictor_configurations": configurations, "fit_seconds": time.monotonic() - started,
                         "artifacts": {p.name: file_sha256(p) for p in shared.iterdir() if p != path}})
    else:
        shared.mkdir(parents=True, exist_ok=False)
        write_json(path, {"binding": binding, "seed": seed, "status": "fitting"})
        steps = binding["configuration"]["encoder_steps"]
        cfg = ActionDecoderConfig(action_type="continuous", steps=steps)
        started = time.monotonic()
        fit = fit_action_decoder_vae(
            np.asarray(zero_s_features(data["state"][:, :-1])), data["red_action"],
            data["valid_length"], np.arange(len(data["state"])), jax.random.PRNGKey(seed), config=cfg,
            training_state_path=shared / "0s_training_state.msgpack")
        opponent = ZeroSOpponent.from_fit(fit, data["objective_label"], np.arange(len(data["state"])))
        opponent.save(shared / "0s.msgpack")
        write_json(shared / "0s_training_history.json", fit.history)
        context = opponent.context(data["state"], data["red_action"], data["valid_length"])
        np.savez_compressed(shared / "context.npz", context=context)
        valid = data["valid_mask"]
        bc = FrozenBCOpponent(fit_continuous_bc(
            np.asarray(zero_s_features(data["state"][:, :-1]))[valid], data["red_action"][valid], seed,
            steps=steps, batch_size=128, hidden_size=64,
            training_state_path=shared / "bc_training_state.msgpack",
            metadata={"training_checkpoints": binding["configuration"]["training_checkpoints"],
                      "feature_schema": "zero_s_current_state_8d"}))
        bc.save(shared / "bc.npz")
        np.savez(shared / "stats.npz", mean=mean, std=std)
        if binding["source_hashes"] != source_hashes():
            raise RuntimeError("source changed while fitting shared predictors")
        write_json(path, {"binding": binding, "seed": seed, "status": "complete",
                         "encoder_config": asdict(cfg), "fit_seconds": time.monotonic() - started,
                         "artifacts": {p.name: file_sha256(p) for p in shared.iterdir() if p != path}})
    if study == "architecture":
        models = {"0s": ZeroSOpponent.load(shared / "0s.msgpack"), "bc": FrozenBCOpponent.load(shared / "bc.npz")}
        contexts = {"0s": np.load(shared / "context.npz")["context"]}
    elif study == "representation":
        models = {name: CausalOpponent.load(shared / f"{name}.msgpack") for name in ("bc", "history", "causal_vae")}
        contexts = {name: np.load(shared / f"{name}_context.npz")["context"] for name in models}
    else:
        models, contexts = {}, {}
    return {"models": models, "contexts": contexts, "manifest_sha256": file_sha256(path)}


def collect(cfg, agent, opponent, params, seed, round_index, directory):
    env = historical.benchmark_env()
    controller = historical.Controller(agent, env, training=True)
    groups = [(c, label) for c in cfg["training_checkpoints"] for label in range(3)]
    remaining = np.full(len(groups), cfg["transitions_per_training_group"], dtype=int)
    batches = np.zeros(len(groups), dtype=int)
    scheduler = np.random.default_rng(np.random.SeedSequence([seed, round_index, 82317]))
    traces, ppo_data, records = [], [], []
    while remaining.any():
        group = int(scheduler.choice(np.flatnonzero(remaining)))
        checkpoint, label = groups[group]
        reset, step_keys, base = historical.keys(seed, round_index, group, int(batches[group]), 1)
        controller.samples = {}
        result = historical.run_matched_episodes(
            env, params[checkpoint, label], controller, reset, step_keys, horizon=100,
            context_mode="online" if opponent else "zero", label=label, zero_s=opponent,
            context_width=None if opponent else 0, max_transitions=int(remaining[group]),
            shuffle_seed=base, record_transitions=True)
        trace = result["transitions"]
        count = int(trace["valid_mask"].sum())
        if not 0 < count <= remaining[group]:
            raise RuntimeError("invalid transition quota progress")
        extras = {}
        if controller.is_ppo:
            batch = ppo_batch(agent, trace, controller.samples, cfg["terminal_contract"])
            ppo_data.append(batch)
            extras = {f"ppo_{k}": batch[k] for k in ("pre_tanh", "log_probs", "old_values", "returns", "advantages")}
        name = f"episode_{len(records):05d}.npz"
        np.savez_compressed(directory / name, **trace, **extras,
                            terminal_for_value=value_terminals(trace, cfg["terminal_contract"]),
                            environment_seed=reset, step_seed=step_keys,
                            checkpoint_seed=np.full(1, checkpoint), objective_label=np.full(1, label))
        records.append({"file": str((directory / name).relative_to(directory.parents[1])),
                        "sha256": file_sha256(directory / name), "checkpoint": checkpoint,
                        "objective": OBJECTIVE_TYPES[label], "valid_transitions": count,
                        "controller_seconds": result["controller_seconds_per_batch"].tolist()})
        traces.append(trace)
        remaining[group] -= count
        batches[group] += 1
    return traces, ppo_data, records


def save_checkpoint(agent, out, manifest, rng, update_key):
    number = len(manifest["rounds"])
    name = f"checkpoint_{number:03d}.msgpack"
    if (out / name).exists():
        raise FileExistsError("a completed checkpoint is immutable")
    if isinstance(agent, ResponsePPO):
        agent.save(out / name)
    else:
        (out / name).write_bytes(flax.serialization.to_bytes(agent))
    manifest.update(checkpoint=name, checkpoint_sha256=file_sha256(out / name),
                    replay_rng=rng.bit_generator.state, update_key=np.asarray(update_key).tolist())
    write_json(out / "manifest.json", manifest)


def controller_configuration(arm, context_dim):
    """Existing normalized global-state versus learned SimNorm world encoders."""
    cfg = load_config(profile="gate4")
    cfg.update(opponent_mode="implicit" if arm in WORLD_REPRESENTATION_ARMS else "factored",
               context_dim=context_dim)
    cfg["encoder"]["type"] = "mlp" if arm == "implicit_mlp" else "identity"
    if arm == "implicit_mlp":
        cfg["encoder"]["normalize_inputs"] = True
    cfg["world_model"].update(hidden_dim=128, predict_continues=True)
    cfg["tdmpc2"]["continue_loss_scale"] = 1.0
    cfg["factored"]["red_loss_scale"] = 0.0
    return cfg


def run_arm(out, seed, arm, data, mean, std, shared, params, binding):
    cfg = binding["configuration"]
    directory = out / f"seed_{seed}" / arm
    path = directory / "manifest.json"
    if binding["source_hashes"] != source_hashes():
        raise ValueError("source changed before controller initialization or restore")
    context_model = CONTEXT_MODEL.get(arm)
    context = shared["contexts"][context_model] if context_model else np.zeros((*data["state"].shape[:2], 0), np.float32)
    opponent = shared["models"][context_model] if context_model else None
    model_cfg = controller_configuration(arm, context.shape[-1])
    is_ppo = arm.startswith("ppo")
    agent = (create_response(mean, std, context_dim=context.shape[-1], seed=seed, hidden=128) if is_ppo
             else create_agent(model_cfg, 66, key=jax.random.PRNGKey(seed), obs_mean=mean, obs_std=std))
    if arm in {"0s", "bc", "history", "causal_vae"}:
        agent = shared["models"][arm].attach(agent, mean, std)
    frozen_red = None if is_ppo or arm in WORLD_REPRESENTATION_ARMS else flax.serialization.to_bytes(agent.model.red_model.params)
    replay = None if is_ppo else SequenceReplay.from_dataset(
        training_trace(data, cfg["terminal_contract"]), np.arange(len(data["state"])), agent.horizon, context)
    rng, update_key = np.random.default_rng(seed), jax.random.PRNGKey(10_000 + seed)
    identity = {"binding": binding, "seed": seed, "arm": arm, "shared_manifest_sha256": shared["manifest_sha256"]}
    manifest = {**identity, "status": "training", "rounds": [], "offline_transitions": int(data["valid_mask"].sum()),
                "online_transitions": 0, "model_configuration": model_cfg if not is_ppo else {
                    k: getattr(agent, k) for k in ("context_dim", "hidden", "learning_rate", "clip_epsilon",
                                                 "entropy_coefficient", "value_coefficient", "max_grad_norm")}}
    if path.exists():
        manifest = json.loads(path.read_text())
        if any(manifest[k] != v for k, v in identity.items()):
            raise ValueError("resume source/configuration/artifact binding changed")
        verify_files(directory, {manifest["checkpoint"]: manifest["checkpoint_sha256"]})
        checkpoint = directory / manifest["checkpoint"]
        agent = ResponsePPO.load(checkpoint) if is_ppo else flax.serialization.from_bytes(agent, checkpoint.read_bytes())
        if frozen_red is not None and frozen_red != flax.serialization.to_bytes(agent.model.red_model.params):
            raise ValueError("restored opponent weights differ from the frozen shared predictor")
        rng.bit_generator.state = manifest["replay_rng"]
        update_key = jnp.asarray(manifest["update_key"], jnp.uint32)
        for row in manifest["rounds"]:
            for record in row["data"]:
                verify_files(directory, {record["file"]: record["sha256"]})
                if replay is not None:
                    with np.load(directory / record["file"], allow_pickle=False) as raw:
                        trace = dict(raw)
                    replay.append(training_trace(trace, cfg["terminal_contract"]), historical.replay_context(trace))
    else:
        directory.mkdir(parents=True, exist_ok=False)

    def update_world(count):
        nonlocal agent, update_key
        history = []
        for step in range(count):
            update_key, key = jax.random.split(update_key)
            started = time.monotonic()
            agent, info = agent.update(**replay.sample(rng, agent.batch_size), key=key)
            check_update(info, step)
            row = {k: float(np.asarray(v)) for k, v in info.items() if np.ndim(v) == 0}
            row.update(step=step, seconds=time.monotonic() - started)
            history.append(row)
        return history

    if not path.exists():
        if is_ppo:
            update_key, offline_key = jax.random.split(update_key)
            started = time.monotonic()
            agent, metrics = warm_start_response(agent, {
                "observations": data["state"][:, :-1], "context": context[:, :-1],
                "actions": data["blue_action"], "returns": historical.offline_returns(data),
                "valid_mask": data["valid_mask"]}, offline_key, steps=cfg["offline_updates"], minibatch_size=128)
            history = [{**metrics, "seconds": time.monotonic() - started}]
        else:
            history = update_world(cfg["offline_updates"])
        write_json(directory / "offline_history.json", history)
        manifest["offline_history_sha256"] = file_sha256(directory / "offline_history.json")
        save_checkpoint(agent, directory, manifest, rng, update_key)
    for round_index in range(len(manifest["rounds"]), cfg["rounds"]):
        parent = directory / f"round_{round_index:03d}"
        parent.mkdir(exist_ok=True)
        attempt = parent / f"attempt_{len(list(parent.iterdir())):03d}"
        attempt.mkdir(exist_ok=False)
        traces, batches, records = collect(cfg, agent, opponent, params, seed, round_index, attempt)
        if is_ppo:
            batch = {k: np.concatenate([b[k] for b in batches]) for k in batches[0]}
            update_key, key = jax.random.split(update_key)
            started = time.monotonic()
            agent, metrics = update_response(agent, batch, key, epochs=cfg["ppo_epochs"], minibatch_size=cfg["ppo_minibatch"])
            history = [{**metrics, "seconds": time.monotonic() - started}]
        else:
            for trace in traces:
                replay.append(training_trace(trace, cfg["terminal_contract"]), historical.replay_context(trace))
            history = update_world(cfg["updates_per_round"])
        if frozen_red is not None and frozen_red != flax.serialization.to_bytes(agent.model.red_model.params):
            raise RuntimeError("frozen opponent weights changed")
        write_json(attempt / "training_history.json", history)
        count = sum(r["valid_transitions"] for r in records)
        if count != 3 * len(cfg["training_checkpoints"]) * cfg["transitions_per_training_group"]:
            raise RuntimeError("incorrect round interaction quota")
        manifest["online_transitions"] += count
        manifest["rounds"].append({"round": round_index, "data": records,
            "history": str((attempt / "training_history.json").relative_to(directory)),
            "history_sha256": file_sha256(attempt / "training_history.json")})
        save_checkpoint(agent, directory, manifest, rng, update_key)
        print(f"{arm} seed {seed}: {manifest['online_transitions']} valid online transitions", flush=True)
    if manifest["status"] == "complete":
        verify_files(directory, manifest["evaluation_artifacts"])
        return
    eval_dir = directory / f"evaluation_attempt_{len(list(directory.glob('evaluation_attempt_*'))):03d}"
    eval_dir.mkdir(exist_ok=False)
    controller = historical.Controller(agent, historical.benchmark_env(), training=False)
    reset, step_keys, base = historical.keys(seed, 0, 0, 0, cfg["eval_episodes_per_objective"], evaluation=True)
    records = []
    for label, objective in enumerate(OBJECTIVE_TYPES):
        result = historical.run_matched_episodes(
            historical.benchmark_env(), params[cfg["evaluation_checkpoint"], label], controller,
            reset, step_keys, horizon=100, context_mode="online" if opponent else "zero", label=label,
            zero_s=opponent, context_width=None if opponent else 0, shuffle_seed=base, record_transitions=True)
        trace = result["transitions"]
        np.savez_compressed(eval_dir / f"{objective}.npz", **trace, environment_seed=reset, step_seed=step_keys,
                            terminal_for_value=value_terminals(trace, cfg["terminal_contract"]))
        records.append({"arm": arm, "seed": seed, "objective": objective, "episode_keys": reset,
                        "metrics": {k: result[k] for k in ("blue_return", "captured", "resources_collected")}})
    write_json(eval_dir / "metrics.json", records)
    manifest.update(status="complete", evaluation_directory=eval_dir.name,
                    evaluation_artifacts={str(p.relative_to(directory)): file_sha256(p) for p in eval_dir.iterdir()})
    if binding["source_hashes"] != source_hashes():
        raise RuntimeError("source changed during execution")
    write_json(path, manifest)


def summarize(out, binding, params):
    """Recompute results and independently replay all completed raw recordings."""
    verifier_path = ROOT / "experiments/matched_control_20260908/verify.py"
    spec = importlib.util.spec_from_file_location("resl_numerical_verifier", verifier_path)
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    records, evidence, paired_step_keys = [], [], {}
    env = historical.benchmark_env()
    step = jax.jit(jax.vmap(env.step_env))
    state = jax.jit(lambda s: markov_state(env, s))
    cfg = binding["configuration"]
    train_resets, eval_resets = set(), set()
    if binding["source_hashes"] != source_hashes():
        raise ValueError("verification executable differs from campaign source")
    dataset = Path(binding["dataset_path"])
    if file_sha256(dataset) != binding["dataset_sha256"]:
        raise ValueError("dataset changed before verification")
    with np.load(dataset, allow_pickle=False) as raw:
        selected = np.isin(raw["checkpoint_seed"], cfg["training_checkpoints"])
        offline = {k: v[selected] for k, v in raw.items()}
    expected_offline = int(offline["valid_mask"].sum())
    train_resets.update(tuple(k) for k in offline["environment_seed"].tolist())
    mean, std = state_statistics(offline["state"], offline["valid_mask"], feature_map="markov")
    red_policies = {
        (checkpoint, objective): specialist_action_function(parameters, 35)
        for (checkpoint, label), parameters in params.items()
        for objective in (OBJECTIVE_TYPES[label],)
    }
    for seed in binding["seeds"]:
        shared = out / f"seed_{seed}" / "shared"
        shared_manifest = json.loads((shared / "manifest.json").read_text())
        if shared_manifest["binding"] != binding or shared_manifest["seed"] != seed or shared_manifest["status"] != "complete":
            raise ValueError("shared fitting identity mismatch")
        verify_files(shared, shared_manifest["artifacts"])
        with np.load(shared / "stats.npz", allow_pickle=False) as stats:
            np.testing.assert_array_equal(stats["mean"], mean)
            np.testing.assert_array_equal(stats["std"], std)
        if cfg.get("study", "architecture") == "architecture":
            opponents = {"0s": ZeroSOpponent.load(shared / "0s.msgpack")}
        elif cfg["study"] == "representation":
            opponents = {name: CausalOpponent.load(shared / f"{name}.msgpack") for name in ("history", "causal_vae")}
        else:
            opponents = {}
        for arm in cfg["arms"]:
            opponent = opponents.get(CONTEXT_MODEL.get(arm))
            directory = out / f"seed_{seed}" / arm
            manifest = json.loads((directory / "manifest.json").read_text())
            if manifest["binding"] != binding or manifest["status"] != "complete" or manifest["seed"] != seed or manifest["arm"] != arm:
                raise ValueError("incomplete or incompatible campaign")
            if manifest["shared_manifest_sha256"] != file_sha256(shared / "manifest.json"):
                raise ValueError("unpaired shared predictor/normalization")
            if manifest["offline_transitions"] != expected_offline:
                raise ValueError("offline transition budget mismatch")
            verify_files(directory, {manifest["checkpoint"]: manifest["checkpoint_sha256"], **manifest["evaluation_artifacts"]})
            verify_files(directory, {"offline_history.json": manifest["offline_history_sha256"]})
            if len(manifest["rounds"]) != cfg["rounds"]:
                raise ValueError("incomplete training rounds")
            all_traces, seen_paths, training_keys = [], set(), set()
            online_count = 0
            for index, row in enumerate(manifest["rounds"]):
                counts = {(c, o): 0 for c in cfg["training_checkpoints"] for o in OBJECTIVE_TYPES}
                if row["round"] != index:
                    raise ValueError("unordered training rounds")
                verify_files(directory, {row["history"]: row["history_sha256"]})
                for trace in row["data"]:
                    if trace["file"] in seen_paths:
                        raise ValueError("duplicate training recording")
                    seen_paths.add(trace["file"])
                    verify_files(directory, {trace["file"]: trace["sha256"]})
                    counts[trace["checkpoint"], trace["objective"]] += trace["valid_transitions"]
                    all_traces.append((directory / trace["file"], trace["checkpoint"], trace["objective"], train_resets, trace["valid_transitions"]))
                if set(counts.values()) != {cfg["transitions_per_training_group"]}:
                    raise ValueError("unequal valid-transition quota")
                online_count += sum(counts.values())
            if online_count != manifest["online_transitions"]:
                raise ValueError("online transition total mismatch")
            eval_dir = directory / manifest["evaluation_directory"]
            actual_records = json.loads((eval_dir / "metrics.json").read_text())
            if len(actual_records) != 3 or {r["objective"] for r in actual_records} != set(OBJECTIVE_TYPES):
                raise ValueError("incomplete objective evaluation")
            for label, objective in enumerate(OBJECTIVE_TYPES):
                file = eval_dir / f"{objective}.npz"
                with np.load(file, allow_pickle=False) as raw:
                    tr = dict(raw)
                lengths = tr["valid_mask"].sum(1)
                if len(lengths) != cfg["eval_episodes_per_objective"] or np.any(lengths <= 0):
                    raise ValueError("incorrect evaluation episode count")
                pair = seed, objective
                if pair in paired_step_keys and not np.array_equal(paired_step_keys[pair], tr["step_seed"]):
                    raise ValueError("paired evaluation step keys disagree")
                paired_step_keys[pair] = tr["step_seed"]
                actual = {"blue_return": np.where(tr["valid_mask"], tr["blue_reward"], 0).sum(1, dtype=np.float64),
                          "captured": tr["terminated_capture"].any(1).astype(float),
                          "resources_collected": tr["state"][np.arange(len(lengths)), lengths, 40:56].sum(1)}
                record = next(r for r in actual_records if r["objective"] == objective)
                if record["arm"] != arm or record["seed"] != seed or not np.array_equal(record["episode_keys"], tr["environment_seed"]):
                    raise ValueError("evaluation identity mismatch")
                for metric, values in actual.items():
                    np.testing.assert_allclose(record["metrics"][metric], values, atol=2e-5, rtol=1e-6)
                records.append({**record, "metrics": actual, "step_keys": tr["step_seed"]})
                all_traces.append((file, cfg["evaluation_checkpoint"], objective, eval_resets, None))
            for file, checkpoint, objective, reset_set, count in all_traces:
                result = verifier.verify_trace(file, env, step, state, red_policies[checkpoint, objective],
                                               opponent)
                if count is not None and result["valid_transitions"] != count:
                    raise ValueError("raw trace transition count differs from quota record")
                with np.load(file, allow_pickle=False) as raw:
                    np.testing.assert_array_equal(raw["terminal_for_value"], value_terminals(dict(raw), cfg["terminal_contract"]))
                    keys = [tuple(k) for k in raw["environment_seed"].tolist()]
                    if len(set(keys)) != len(keys):
                        raise ValueError("duplicate resets within a trace")
                    if count is not None:
                        if training_keys.intersection(keys):
                            raise ValueError("duplicate training episode resets")
                        training_keys.update(keys)
                        if not np.all(raw["checkpoint_seed"] == checkpoint) or not np.all(raw["objective_label"] == OBJECTIVE_TYPES.index(objective)):
                            raise ValueError("recorded specialist labels disagree")
                    reset_set.update(keys)
                evidence.append({"path": str(file.relative_to(out)), **result})
    if train_resets & eval_resets:
        raise ValueError("training and evaluation resets overlap")
    result = report_comparisons(records, binding)
    result["scope"] = "feasibility only" if binding["stage"] == "pilot" else f"documented {cfg.get('study', 'architecture')} comparison; reused opponent population"
    write_json(out / "numerical_verification.json", evidence)
    write_json(out / "comparison.json", result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--stage", choices=("pilot", "main"), required=True)
    parser.add_argument("--phase", choices=("all", "prepare", "train", "summarize"), default="all")
    for name in ("spec-repo", "protocol", "dataset", "logdir", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--spec-commit", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--seeds", help="subset for sequential execution; summary always requires complete protocol")
    parser.add_argument("--arms", help="subset for sequential execution; summary always requires every protocol arm")
    args = parser.parse_args(argv)
    if not args.execute:
        print("No experiment executed. Supply --execute only after correctness and protocol review.")
        return 0
    _, cfg, seeds, binding = bind_protocol(args)
    selected_seeds = seeds if not args.seeds else [int(s) for s in args.seeds.split(",")]
    arms = cfg["arms"]
    selected_arms = list(arms) if not args.arms else args.arms.split(",")
    if len(set(selected_seeds)) != len(selected_seeds) or not set(selected_seeds) <= set(seeds) or not selected_seeds:
        raise ValueError("invalid execution seed subset")
    if len(set(selected_arms)) != len(selected_arms) or not set(selected_arms) <= set(arms) or not selected_arms:
        raise ValueError("invalid execution arm subset")
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = args.out / "campaign.json"
        if path.exists():
            if json.loads(path.read_text()) != binding:
                raise ValueError("campaign source binding changed; use a new output")
        elif set(p.name for p in args.out.iterdir()) != {"execution.lock"}:
            raise FileExistsError("fresh campaign directory required")
        else:
            write_json(path, binding)
        ds = load_continuous_dataset(args.dataset)
        train = np.flatnonzero(np.isin(ds.checkpoint_seed, cfg["training_checkpoints"]))
        data = {k: v[train] for k, v in ds.as_dict().items()}
        if args.stage == "main" and (len(train) != 1200 or int(data["valid_mask"].sum()) != 86_899):
            raise ValueError("main offline data budget differs from original protocol")
        mean, std = state_statistics(data["state"], data["valid_mask"], feature_map="markov")
        checkpoints = set(cfg["training_checkpoints"]) | {cfg["evaluation_checkpoint"]}
        params = {(c, label): load_continuous_actor_params(continuous_checkpoint_path(args.logdir, objective, "pred", c))
                  for c in checkpoints for label, objective in enumerate(OBJECTIVE_TYPES)}
        if args.phase != "summarize":
            try:
                for seed in selected_seeds:
                    shared = prepare_shared(args.out, seed, data, mean, std, binding)
                    if args.phase != "prepare":
                        for arm in selected_arms:
                            run_arm(args.out, seed, arm, data, mean, std, shared, params, binding)
            except Exception as error:
                failure = args.out / f"failure_{time.time_ns()}.json"
                write_json(failure, {"type": type(error).__name__, "error": str(error),
                                     "seed": seed, "arm": locals().get("arm"), "phase": args.phase})
                raise
        if args.phase == "summarize" or (args.phase == "all" and set(selected_seeds) == set(seeds) and set(selected_arms) == set(arms)):
            summarize(args.out, binding, params)
        if historical.specialist_binding(SimpleNamespace(dataset=args.dataset, logdir=args.logdir)) != binding["specialists"]:
            raise RuntimeError("frozen specialist artifacts changed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
