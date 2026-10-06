#!/usr/bin/env python3
"""A06: real completed trajectories, replay batches, and original 0s weight updates."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mopa.continuous_data import (  # noqa: E402
    OBJECTIVE_TYPES,
    continuous_checkpoint_path,
    deterministic_specialist_action,
    load_continuous_actor_params,
    load_continuous_dataset,
    markov_state,
    replay_episodes,
)
from mopa.evaluation import (  # noqa: E402
    run_matched_episodes,
    specialist_action_function,
)
from mopa.manifest import file_sha256, package_versions  # noqa: E402
from mopa.opponent_updates import (  # noqa: E402
    OpponentReplay,
    update_zero_s_from_replay,
)
from mopa.tdmpc import create_agent  # noqa: E402
from mopa.tdmpc_data import state_statistics  # noqa: E402
from mopa.zero_s import ZeroSOpponent  # noqa: E402
from tag_objectives import make_env  # noqa: E402


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def check_files(binding):
    for path, digest in binding.items():
        if file_sha256(Path(path)) != digest:
            raise ValueError(f"bound source/artifact changed: {path}")


def source_binding(commit):
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() != commit:
        raise ValueError("exact executable HEAD required")
    for command in (["diff", "--name-only", "HEAD", "--", "src", "scripts", "configs"],
                    ["ls-files", "--others", "--exclude-standard", "--", "src", "scripts", "configs"]):
        if subprocess.check_output(["git", *command], cwd=ROOT, text=True).strip():
            raise ValueError("committed clean executable source required")
    paths = [*ROOT.glob("src/**/*.py"), *ROOT.glob("scripts/*.py"),
             ROOT / "configs/tdmpc2.yaml", ROOT / "uv.lock",
             ROOT / "experiments/matched_control_20260908/verify.py"]
    result = {str(p): file_sha256(p) for p in paths}
    for path, digest in result.items():
        data = subprocess.check_output(["git", "show", f"{commit}:{Path(path).relative_to(ROOT)}"], cwd=ROOT)
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("source differs from committed bytes")
    return result


def check_complete(trace, max_steps):
    """Reject administrative cuts even when their historical flag says timeout."""
    valid, lengths = np.asarray(trace["valid_mask"]), np.asarray(trace["valid_length"])
    if not np.array_equal(valid, np.arange(valid.shape[1])[None] < lengths[:, None]) or np.any(lengths <= 0):
        raise ValueError("nonempty contiguous completed trajectories required")
    term, trunc = trace["terminated_capture"], trace["truncated_timeout"]
    end = term | trunc
    if np.any(term & trunc) or not np.array_equal(end.sum(1), np.ones(len(lengths), int)):
        raise ValueError("one physical endpoint required per episode")
    if not np.array_equal(np.argmax(end, axis=1), lengths - 1):
        raise ValueError("episode endpoint must end the valid prefix")
    # The public Markov contract's final field is step/max_steps.
    endpoint_time = trace["state"][np.arange(len(lengths)), lengths, -1]
    if np.any(trunc.any(1) & (endpoint_time < 1.0)) or np.any(lengths > max_steps):
        raise ValueError("administrative quota cuts are not completed episodes")


def raw_errors(prediction, target, lengths, labels):
    mask = np.arange(target.shape[1])[None] < lengths[:, None]
    error = np.where(mask, np.sum((prediction - target) ** 2, axis=-1), 0.0)
    if not np.isfinite(error).all() or np.any(lengths <= 0):
        raise ValueError("finite nonempty evaluation required")
    values = error.sum(1) / lengths
    return {"episode_mean_vector_mse": float(values.mean()),
            "transition_mean_vector_mse": float(error.sum() / lengths.sum()),
            "episode_values": values.tolist(),
            "by_objective": {name: float(values[labels == k].mean()) for k, name in enumerate(OBJECTIVE_TYPES) if np.any(labels == k)}}


def add_identities(keys, fingerprints, previous_keys, previous_fingerprints):
    key_set = {tuple(k) for k in np.asarray(keys).tolist()}
    hashes = set(fingerprints)
    if len(key_set) != len(keys) or len(hashes) != len(keys) or key_set & previous_keys or hashes & previous_fingerprints:
        raise ValueError("reset key or complete initial simulator state overlaps")
    previous_keys.update(key_set)
    previous_fingerprints.update(hashes)


def fingerprints(env, keys):
    _, states = jax.vmap(env.reset)(jnp.asarray(keys, jnp.uint32))
    leaves = [np.asarray(v) for v in jax.tree.leaves(states)]
    return [hashlib.sha256(b"".join(np.ascontiguousarray(v[i]).tobytes() for v in leaves)).hexdigest() for i in range(len(keys))]


def append_trace(buffer, trace, source, origin):
    for i, length in enumerate(trace["valid_length"]):
        length = int(length)
        buffer.append(trace["state"][i, :length + 1], trace["red_action"][i, :length],
                      episode_id=f"{source}#{i}", origin=origin, source_ref=f"{source}#{i}", completed=True, real=True)


def load_verifier():
    path = ROOT / "experiments/matched_control_20260908/verify.py"
    spec = importlib.util.spec_from_file_location("opponent_update_replay_verifier", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.verify_trace


def protocol_binding(path, commit):
    spec_root = next((p for p in path.resolve().parents if (p / ".git").exists()), None)
    if spec_root is None or len(commit) != 40:
        raise ValueError("committed private protocol required")
    original = subprocess.check_output(["git", "show", f"{commit}:{path.resolve().relative_to(spec_root)}"], cwd=spec_root)
    if original != path.read_bytes():
        raise ValueError("protocol differs from specification commit")
    p = json.loads(original)
    if p["instruction_id"] != "A06" or p["status"] != "frozen":
        raise ValueError("frozen A06 protocol required")
    expected = {"fit_seeds": [11, 22], "initial_training_checkpoint_family": 0,
                "online_training_checkpoint_family": 0, "evaluation_checkpoint_family": 1,
                "capacity_episodes": 606, "online_rounds": 2, "completed_episodes_per_objective_per_round": 1,
                "additional_updates_per_round": 32, "initial_optimizer_step": 64,
                "evaluation_episodes_per_objective": 3, "physical_episode_limit": 100}
    if any(p[k] != v for k, v in expected.items()):
        raise ValueError("A06 numerical settings differ from the reviewed driver")
    if [item["fit_seed"] for item in p["initial_artifacts"]] != p["fit_seeds"]:
        raise ValueError("each frozen fitting seed needs exactly one initial artifact")
    return p


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running", "instruction_id": "A06", "specification_commit": args.spec_commit,
                "executable_commit": args.code_commit, "runs": []}
    write_json(args.output / "manifest.json", manifest)
    try:
        protocol = protocol_binding(args.protocol, args.spec_commit)
        bound = source_binding(args.code_commit)
        bound[str(args.protocol.resolve())] = file_sha256(args.protocol)
        dataset = protocol["dataset"]
        bound.update({dataset["path"]: dataset["sha256"], dataset["sidecar_path"]: dataset["sidecar_sha256"]})
        for initial in protocol["initial_artifacts"]:
            for item in initial.values():
                if isinstance(item, dict) and "path" in item:
                    bound[item["path"]] = item["sha256"]
        check_files(bound)
        data = load_continuous_dataset(dataset["path"]).as_dict()
        initial_rows = np.flatnonzero(data["checkpoint_seed"] == 0)
        if len(initial_rows) != 600:
            raise ValueError("expected600 initial family0 episodes")
        check_complete({k: v[initial_rows] for k, v in data.items()}, 100)
        initial_replay = replay_episodes(data, initial_rows)
        if any(v > 1e-5 for k, v in initial_replay.items() if k.startswith("max_abs_error_")) or initial_replay["termination_flags_match_fraction"] != 1:
            raise ValueError("initial real-data numerical replay failed")
        env = make_env("capture", continuous=True)
        verify = load_verifier()
        step_fn, state_fn = jax.jit(jax.vmap(env.step_env)), jax.jit(lambda s: markov_state(env, s))
        reserved_keys = {tuple(k) for k in data["environment_seed"].tolist()}
        # All historical dataset resets are reserved, including inspected holdouts.
        unique_keys = np.unique(data["environment_seed"], axis=0)
        reserved_states = set(fingerprints(env, unique_keys))
        manifest.update(protocol_id=protocol["protocol_id"], protocol_sha256=file_sha256(args.protocol),
                        source_and_input_hashes=bound, dependencies=package_versions(), initial_replay=initial_replay)
        for initial in protocol["initial_artifacts"]:
            seed = initial["fit_seed"]
            directory = args.output / f"seed_{seed}"
            directory.mkdir()
            shared_path = Path(initial["shared_manifest"]["path"])
            shared = json.loads(shared_path.read_text())
            if shared["seed"] != seed or shared["status"] != "complete":
                raise ValueError("initial shared fitting identity or completion differs")
            for name, digest in shared["artifacts"].items():
                bound[str(shared_path.parent / name)] = digest
            if shared["binding"]["configuration"]["training_checkpoints"] != [0] or shared["binding"]["dataset_sha256"] != dataset["sha256"]:
                raise ValueError("initial fitting provenance is not family0")
            specialist_records = shared["binding"]["specialists"]
            red = {}
            for family in (0, 1):
                for label, objective in enumerate(OBJECTIVE_TYPES):
                    item = specialist_records[f"{family}:{objective}"]
                    expected = continuous_checkpoint_path(args.logdir, objective, "pred", family)
                    if expected.resolve() != Path(item["path"]).resolve():
                        raise ValueError("specialist directory differs from bound source")
                    bound[item["path"]] = item["sha256"]
                    red[family, label] = load_continuous_actor_params(item["path"])
            if {specialist_records[f"0:{o}"]["sha256"] for o in OBJECTIVE_TYPES} & {specialist_records[f"1:{o}"]["sha256"] for o in OBJECTIVE_TYPES}:
                raise ValueError("held-out specialist hashes overlap initial training")
            check_files(bound)
            # Verify actual initial specialist outputs, not just sidecar labels.
            for label in range(3):
                rows = initial_rows[data["objective_label"][initial_rows] == label]
                mask = data["valid_mask"][rows]
                observations = jnp.asarray(data["red_observation"][rows, :-1][mask])
                predicted = deterministic_specialist_action(red[0, label], observations, 35)
                np.testing.assert_allclose(predicted, data["red_action"][rows][mask], rtol=0, atol=1e-6)
            checkpoint = Path(initial["controller"]["path"])
            controller_manifest = json.loads(Path(initial["controller_manifest"]["path"]).read_text())
            if (controller_manifest["seed"] != seed or controller_manifest["arm"] != "0s"
                    or controller_manifest["status"] != "complete" or controller_manifest["binding"] != shared["binding"]):
                raise ValueError("initial controller identity or binding differs")
            if controller_manifest["checkpoint_sha256"] != initial["controller"]["sha256"] or controller_manifest["shared_manifest_sha256"] != initial["shared_manifest"]["sha256"]:
                raise ValueError("controller and shared manifest disagree")
            opponent = ZeroSOpponent.load(shared_path.parent / "0s.msgpack")
            with np.load(shared_path.parent / "stats.npz") as stats:
                mean, std = state_statistics(data["state"][initial_rows], data["valid_mask"][initial_rows])
                np.testing.assert_array_equal(stats["mean"], mean)
                np.testing.assert_array_equal(stats["std"], std)
                agent = create_agent(controller_manifest["model_configuration"], 66, key=jax.random.PRNGKey(seed), obs_mean=stats["mean"], obs_std=stats["std"])
                agent = opponent.attach(agent, stats["mean"], stats["std"])
            agent = serialization.from_bytes(agent, checkpoint.read_bytes())
            frozen_agent = serialization.to_bytes(agent)
            act = jax.jit(lambda state, context, key: agent.act(state, mpc=False, deterministic=True, context=context, key=key)[0])

            def blue(state, obs, context, carry, key, t):
                del obs, t
                return act(markov_state(env, state), context, key), carry

            replay, evaluation = OpponentReplay(606), OpponentReplay(9)
            for index in initial_rows:
                length = int(data["valid_length"][index])
                replay.append(data["state"][index, :length + 1], data["red_action"][index, :length],
                              episode_id=f"dataset:{dataset['sha256']}#{index}", source_ref=f"{dataset['path']}#{index}", origin="initial", completed=True, real=True)
            certificates, eval_labels = [], []

            def collect(family, label, number, round_index, target, origin):
                reset_ids = [(860000 + seed * 1000 + label * 10 + i) if origin == "evaluation"
                             else (760000 + seed * 1000 + round_index * 100 + label) for i in range(number)]
                split = [jax.random.split(jax.random.PRNGKey(i)) for i in reset_ids]
                reset, steps = np.stack([s[0] for s in split]), np.stack([s[1] for s in split])
                identities = fingerprints(env, reset)
                add_identities(reset, identities, reserved_keys, reserved_states)
                result = run_matched_episodes(env, red[family, label], blue, reset, steps, horizon=100,
                                             context_mode="online", label=label, zero_s=opponent,
                                             shuffle_seed=reset_ids[0], record_transitions=True)
                tr = result["transitions"]
                check_complete(tr, 100)
                path = directory / f"{origin}_round{round_index}_{OBJECTIVE_TYPES[label]}.npz"
                np.savez_compressed(path, **tr, environment_seed=reset, step_seed=steps,
                                    checkpoint_seed=np.full(number, family), objective_label=np.full(number, label))
                red_fn = specialist_action_function(red[family, label], 35)
                certification = verify(path, env, step_fn, state_fn, red_fn, opponent)
                certification["complete_initial_state_sha256"] = identities
                certification["physically_complete"] = True
                certificates.append(certification)
                append_trace(target, tr, str(path), "online" if origin == "online" else "initial")

            for label in range(3):
                collect(1, label, 3, 0, evaluation, "evaluation")
                eval_labels.extend([label] * 3)
            state_path = Path(initial["opponent_training_state"]["path"])
            original_state = serialization.msgpack_restore(state_path.read_bytes())
            for name in ("state_mean", "state_std"):
                np.testing.assert_array_equal(original_state[name], getattr(opponent.encoder, name))
            for saved, live in ((original_state["params"]["e"], opponent.encoder.params),
                                (original_state["params"]["d"], opponent.decoder_params)):
                for a, b in zip(jax.tree.leaves(saved), jax.tree.leaves(live), strict=True):
                    np.testing.assert_array_equal(a, b)
            if original_state["step"] != 64 or original_state["anneal_steps"] != 64:
                raise ValueError("initial opponent optimizer schedule differs")
            result = {"seed": seed, "initial_controller_sha256": initial["controller"]["sha256"], "rounds": [], "certificates": certificates}
            manifest["runs"].append(result)
            for round_index in range(2):
                print(f"A06 seed{seed}: collect complete episodes and update round{round_index + 1}", flush=True)
                check_files(bound)
                for label in range(3):
                    collect(0, label, 1, round_index, replay, "online")
                next_path = directory / f"opponent_training_state_{round_index + 1}.msgpack"
                fit, report, traces = update_zero_s_from_replay(replay, evaluation, state_path, updates=32, training_state_path=next_path)
                continued = serialization.msgpack_restore(next_path.read_bytes())
                if continued["step"] != 96 + 32 * round_index or int(continued["optimizer"]["0"]["count"]) != continued["step"]:
                    raise ValueError("optimizer did not continue")
                for name in ("state_mean", "state_std"):
                    np.testing.assert_array_equal(continued[name], original_state[name])
                if continued["anneal_steps"] != 64 or report["encoder_parameter_change_l2"] <= 0 or report["decoder_parameter_change_l2"] <= 0:
                    raise ValueError("original schedule and actual opponent weight changes required")
                if serialization.to_bytes(agent) != frozen_agent:
                    raise ValueError("controller/world/value weights changed")
                for stage in ("before", "after"):
                    actual = raw_errors(traces[stage + "_prediction"], traces["target"], traces["valid_length"], np.asarray(eval_labels))
                    np.testing.assert_allclose(actual["episode_values"], report[stage]["episode_values"], rtol=0, atol=1e-7)
                    report[stage] = actual
                np.savez_compressed(directory / f"prediction_round{round_index}.npz", **traces, objective_label=eval_labels)
                write_json(directory / f"training_history_round{round_index}.json", fit.history)
                write_json(directory / f"update_round{round_index}.json", report)
                opponent = ZeroSOpponent(fit.encoder, fit.decoder_params, opponent.prototypes)
                state_path = next_path
                bound[str(next_path)] = report["checkpoint_sha256"]
                result["rounds"].append(report)
                write_json(args.output / "manifest.json", manifest)
            check_files(bound)
        manifest.update(status="completed", claim="Original0s replay-to-opponent-weight mechanism only; no control improvement or co-training claim",
                        artifacts={str(p.relative_to(args.output)): file_sha256(p) for p in sorted(args.output.rglob("*")) if p.is_file() and p.name != "manifest.json"})
        check_files(bound)
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(args.output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--spec-commit", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--logdir", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
