#!/usr/bin/env python3
"""Source-bound A01/P07/P10 frozen-prefix diagnostics; never fit model weights."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mopa import mppi  # noqa: E402
from mopa.bc_continuous import FrozenBCOpponent  # noqa: E402
from mopa.continuous_data import (  # noqa: E402
    OBJECTIVE_TYPES,
    load_continuous_actor_params,
    markov_state,
)
from mopa.manifest import file_sha256, package_versions  # noqa: E402
from mopa.planning_diagnostics import (  # noqa: E402
    closed_loop_suffix,
    compare_candidates,
    discounted_rewards,
    frozen_prefix,
    imagined_candidates,
    imagined_policy_commands,
    simulator_candidates,
    simulator_policy_tail,
    specialist_on_imagined_observation,
)
from mopa.tdmpc import create_agent  # noqa: E402
from mopa.zero_s import ZeroSOpponent  # noqa: E402
from tag_objectives import make_env  # noqa: E402


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def save_json(path, value):
    Path(path).write_text(json.dumps(jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def code_hashes():
    paths = [Path(__file__), ROOT / "uv.lock", *(ROOT / "src").rglob("*.py")]
    return {str(p.relative_to(ROOT)): file_sha256(p) for p in paths}


def verified_manifest(path):
    manifest = json.loads(path.read_text())
    for name, digest in manifest.get("artifacts", {}).items():
        if file_sha256(path.parent / name) != digest:
            raise ValueError(f"artifact mismatch: {path.parent / name}")
    return manifest


def load_controller(campaign, seed, expected_binding):
    directory = campaign / f"seed_{seed}"
    shared = verified_manifest(directory / "shared/manifest.json")
    manifest = verified_manifest(directory / "0s/manifest.json")
    if shared["status"] != "complete" or manifest["status"] != "complete" or shared["seed"] != seed or manifest["seed"] != seed:
        raise ValueError("a complete independently fitted0s controller is required")
    if manifest["binding"] != expected_binding or shared["binding"] != expected_binding or manifest["shared_manifest_sha256"] != file_sha256(directory / "shared/manifest.json"):
        raise ValueError("controller/shared provenance differs")
    checkpoint = directory / "0s" / manifest["checkpoint"]
    if file_sha256(checkpoint) != manifest["checkpoint_sha256"]:
        raise ValueError("controller checkpoint changed")
    with np.load(directory / "shared/stats.npz", allow_pickle=False) as source:
        mean, std = source["mean"], source["std"]
    opponent = ZeroSOpponent.load(directory / "shared/0s.msgpack")
    bc = FrozenBCOpponent.load(directory / "shared/bc.npz")
    agent = create_agent(manifest["model_configuration"], 66, key=jax.random.PRNGKey(seed), obs_mean=mean, obs_std=std)
    agent = opponent.attach(agent, mean, std)
    red_before = flax.serialization.to_bytes(agent.model.red_model.params)
    agent = flax.serialization.from_bytes(agent, checkpoint.read_bytes())
    if red_before != flax.serialization.to_bytes(agent.model.red_model.params):
        raise ValueError("controller red head differs from shared frozen predictor")
    dependencies = {str(path): file_sha256(path) for path in [checkpoint, directory / "0s/manifest.json",
        directory / "shared/manifest.json", *(directory / "shared" / name for name in shared["artifacts"])]}
    return agent, opponent, bc, mean, std, manifest, dependencies


def scene_keys(seed, checkpoint, reset_index):
    root = jax.random.PRNGKey(1_310_005)
    for value in (seed, checkpoint, reset_index):
        root = jax.random.fold_in(root, value)
    return jax.random.split(root, 4)  # reset, simulator noise, blue policy, candidates


def subset(result, indices):
    return {k: v[indices] for k, v in result.items() if isinstance(v, np.ndarray) and v.ndim > 0}


def plot_diagnostic(directory, metadata):
    """Preselected first-reset panels; underlying numerical arrays remain saved."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["learned", "state_only_BC", "actual_specialist_on_imagined_observation", "actual_red_action_injection"]
    for row in metadata["steps"]:
        folder = directory / f"horizon_{row['horizon']}"
        with np.load(folder / "actual.npz", allow_pickle=False) as source:
            actual = dict(source)
        index = row["selected_candidate"]
        figures, axes = plt.subplots(2, 2, figsize=(11, 8))
        real_length = int(actual["valid"][index].sum())
        for axis, offset, title in ((axes[0, 0], 2, "Blue position"), (axes[0, 1], 0, "Red position")):
            trajectory = actual["state"][index, :real_length + 1]
            axis.plot(trajectory[:, offset], trajectory[:, offset + 1], "k-o", markersize=3, label="real simulator")
            axis.set_title(title)
            axis.set_aspect("equal", adjustable="datalim")
        axes[1, 0].plot(np.arange(real_length), actual["reward"][index, :real_length], "k-o", label="real reward")
        axes[1, 1].step(np.arange(real_length), actual["continues"][index, :real_length], "k-", where="mid", label="real continuation")
        for name in names:
            with np.load(folder / f"{name}.npz", allow_pickle=False) as source:
                prediction = dict(source)
            length = int(prediction["valid"][index].sum())
            label = name.replace("_", " ")
            for axis, offset in ((axes[0, 0], 2), (axes[0, 1], 0)):
                trajectory = prediction["state"][index]
                line = axis.plot(trajectory[:length + 1, offset], trajectory[:length + 1, offset + 1], label=label)[0]
                if length < row["horizon"]:
                    axis.plot(trajectory[length:, offset], trajectory[length:, offset + 1], ":", color=line.get_color(), alpha=.5)
            axes[1, 0].plot(np.arange(length), prediction["reward"][index, :length], label=label)
            axes[1, 1].plot(np.arange(length), prediction["continues_probability"][index, :length], label=label)
        axes[1, 0].set_title("Immediate reward, before each method's stop")
        axes[1, 1].set_title("Continuation probability; hard gate remains0.5")
        for axis in axes.flat:
            axis.legend(fontsize=6)
            axis.grid(alpha=.2)
        figures.suptitle(f"Frozen prefix{metadata['prefix_length']}; actual selected MPPI candidate; horizon{row['horizon']}\nDotted paths after predicted stop are extrapolation; red-policy oracle is not a return upper bound", fontsize=10)
        figures.tight_layout()
        figures.savefig(folder / "selected_candidate.png", dpi=150)
        plt.close(figures)



def full_remainder_scene(directory, env, agent, opponent, bc, mean, std, red_params,
                         complete, context, step_seed, candidate_key):
    """A01: one supplied blue sequence shared by all full-remainder branches."""
    directory.mkdir()
    remaining = env.max_steps - int(np.asarray(complete.step))
    initial = np.asarray(markov_state(env, complete))
    key = jax.random.fold_in(candidate_key, 100)
    generation = imagined_policy_commands(agent, initial, context, mean, std, remaining, key=key)
    np.savez_compressed(directory / "commands.npz", **{k: v for k, v in generation.items() if isinstance(v, np.ndarray)},
                        generation_key=np.asarray(key), step_key=np.asarray(step_seed), fixed_context=context)
    result = {"status": generation["status"], "requested_steps": remaining,
        "generated_steps": generation["generated_steps"],
        "generation": "Frozen deterministic blue prior at its own learned imagined states; installed learned red head each step; original prefix context fixed; no simulator inputs after prefix",
        "comparison": "Exactly the same supplied blue commands in learned, BC, actual-specialist imagined worlds and complete-state real simulator",
        "stopping": "Predicted/real valid masks separate. Any generated or predicted transitions after predicted stop are extrapolation; real simulator state freezes after physical stop",
        "planner_scope": "One full-remainder supplied sequence; no MPPI horizon or candidate-search change and no policy performance claim",
        "methods": {}}
    if generation["status"] != "complete":
        save_json(directory / "metrics.json", result)
        return result
    commands = generation["blue_action"][None]
    actual = simulator_candidates(env, complete, commands, red_params, step_seed)
    actual.pop("complete_final_state")
    np.savez_compressed(directory / "actual.npz", **actual)
    sources = {
        "learned": lambda raw, c, t: opponent.actions(raw, c),
        "state_only_BC": lambda raw, c, t: bc.actions(raw),
        "actual_specialist_on_imagined_observation": lambda raw, c, t: specialist_on_imagined_observation(env, complete, raw, red_params),
    }
    for name, source in sources.items():
        try:
            prediction = imagined_candidates(agent, initial, commands, context, mean, std, source, key=key)
        except FloatingPointError as error:
            result["methods"][name] = {"status": "failed_nonfinite_imagination", "error": str(error)}
            continue
        np.savez_compressed(directory / f"{name}.npz", **prediction)
        if name == "learned":
            np.testing.assert_allclose(prediction["state"][0], generation["state"], atol=2e-4, rtol=2e-5)
            np.testing.assert_array_equal(prediction["valid"][0], generation["valid"])
        metrics = compare_candidates(prediction, actual, std, agent.discount,
                                     capture_distance=float(env.rad[0] + env.rad[1]), arena=env.arena)
        # The finite physical-budget result is the relevant endpoint. A learned
        # Q beyond physical timeout is not a true continuation reference.
        for name_to_remove in ("tail_mean", "ranking_with_Q_tail_against_finite_outcome", "Q_comparison_limit"):
            metrics.pop(name_to_remove)
        metrics.update(status="complete", predicted_valid_steps=int(prediction["valid"].sum()),
                       post_stop_extrapolation_steps=int((~prediction["valid"]).sum()))
        result["methods"][name] = metrics
    if any(row["status"] != "complete" for row in result["methods"].values()):
        result["status"] = "failed_nonfinite_imagination"
    save_json(directory / "metrics.json", result)
    return result


def plot_full_remainder(directory):
    """First predeclared reset only; do not choose visually favorable scenes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = directory / "full_remainder"
    if not (folder / "actual.npz").exists():
        return
    with np.load(folder / "actual.npz", allow_pickle=False) as source:
        actual = dict(source)
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    real_length = int(actual["valid"][0].sum())
    for axis, offset, title in ((axes[0, 0], 2, "Blue trajectory"), (axes[0, 1], 0, "Red trajectory")):
        axis.plot(actual["state"][0, :real_length + 1, offset], actual["state"][0, :real_length + 1, offset + 1], "k-", label="real supplied-command replay")
        axis.set_title(title)
        axis.set_aspect("equal", adjustable="datalim")
    axes[1, 0].plot(actual["reward"][0, :real_length], "k-", label="real reward")
    axes[1, 1].step(np.arange(real_length), actual["continues"][0, :real_length], "k-", where="mid", label="real continuation")
    for name in ("learned", "state_only_BC", "actual_specialist_on_imagined_observation"):
        path = folder / f"{name}.npz"
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as source:
            prediction = dict(source)
        length = int(prediction["valid"][0].sum())
        for axis, offset in ((axes[0, 0], 2), (axes[0, 1], 0)):
            trajectory = prediction["state"][0]
            line = axis.plot(trajectory[:length + 1, offset], trajectory[:length + 1, offset + 1], label=name)[0]
            if length < prediction["valid"].shape[1]:
                axis.plot(trajectory[length:, offset], trajectory[length:, offset + 1], ":", color=line.get_color(), alpha=.5)
        axes[1, 0].plot(prediction["reward"][0, :length], label=name)
        axes[1, 1].plot(prediction["continues_probability"][0, :length], label=name)
    axes[1, 0].set_title("Immediate reward before each stop")
    axes[1, 1].set_title("Continuation before each stop")
    for axis in axes.flat:
        axis.legend(fontsize=6)
        axis.grid(alpha=.2)
    figure.suptitle("A01 full remainder: fixed blue commands generated by learned-world deterministic prior\nSame supplied commands in all branches; dotted paths after predicted stop are extrapolation", fontsize=10)
    figure.tight_layout()
    figure.savefig(folder / "full_remainder.png", dpi=150)
    plt.close(figure)

def diagnostic_scene(directory, env, agent, opponent, bc, mean, std, red_params,
                     prefix, step_seed, candidate_key, cfg, *, label, shuffled_context):
    complete = prefix["complete_state"]
    initial = np.asarray(markov_state(env, complete))
    context = np.asarray(prefix["context_carry"].context[0])
    metadata = {"prefix_length": prefix["valid_length"], "requested_prefix_length": prefix["requested_length"],
                "initial_state": initial, "context": context, "steps": []}
    (directory / "complete_prefix_state.msgpack").write_bytes(flax.serialization.to_bytes(complete))
    (directory / "complete_prefix_context.msgpack").write_bytes(flax.serialization.to_bytes(prefix["context_carry"]))
    np.savez_compressed(directory / "prefix.npz", **{k: v for k, v in prefix.items() if isinstance(v, np.ndarray)})
    if prefix["valid_length"] != prefix["requested_length"] or np.asarray(complete.done).any():
        metadata["status"] = "stopped before requested imagination; no replacement reset selected"
        save_json(directory / "metrics.json", metadata)
        return metadata
    for horizon in cfg["horizons"]:
        h_dir = directory / f"horizon_{horizon}"
        h_dir.mkdir()
        key = jax.random.fold_in(candidate_key, horizon)
        planner_key, random_key, random_value_key = jax.random.split(key, 3)
        x = agent.model.encode(jnp.asarray(initial)[None], agent.model.encoder.params, jax.random.PRNGKey(0))
        chosen_action, plan, inspection = mppi.plan(agent, x, horizon, jnp.asarray(context)[None],
            deterministic=True, train=False, key=planner_key, return_diagnostics=True)
        # Verify the observer did not change the actual action or warm start.
        plain_action, plain_plan = mppi.plan(agent, x, horizon, jnp.asarray(context)[None],
            deterministic=True, train=False, key=planner_key)
        np.testing.assert_array_equal(chosen_action, plain_action)
        for observed, original in zip(plan, plain_plan, strict=True):
            np.testing.assert_array_equal(observed, original)
        planner_actions = np.asarray(inspection["candidate_actions"])[0]
        random_actions = np.asarray(jax.random.uniform(random_key, (cfg["random_candidates"], horizon, 2), minval=-1., maxval=1.))
        actions = np.concatenate([planner_actions, random_actions])
        selected = int(np.asarray(inspection["selected_population_index"])[0])
        elite = np.asarray(inspection["elite_indices"])[0]
        random_rows = np.arange(len(planner_actions), len(actions))
        value_key = inspection["estimate_value_key"]
        pools = [(np.arange(len(planner_actions)), value_key), (random_rows, random_value_key)]
        np.savez_compressed(h_dir / "candidate_inputs.npz", actions=actions, selected=selected, elite=elite,
                            random_rows=random_rows, planner_key=np.asarray(planner_key), value_key=np.asarray(value_key),
                            original_planner_values=np.asarray(inspection["candidate_values"])[0])
        actual = simulator_candidates(env, complete, actions, red_params, step_seed)
        final_state = actual.pop("complete_final_state")
        # Re-run identical raw branches for deterministic replay; this is not an independently implemented simulator.
        replayed = simulator_candidates(env, complete, actions, red_params, step_seed)
        replayed.pop("complete_final_state")
        for name in actual:
            np.testing.assert_array_equal(actual[name], replayed[name])
        tails = [simulator_policy_tail(agent, env, jax.tree.map(lambda value: value[indices], final_state),
            red_params, context, step_seed, int(np.asarray(complete.step)) + horizon,
            max(0, env.max_steps - int(np.asarray(complete.step)) - horizon), key=pool_key)
            for indices, pool_key in pools]
        tail = {name: np.concatenate([part[name] for part in tails]) for name in tails[0]}
        np.savez_compressed(h_dir / "actual.npz", **actual, **{f"tail_{k}": v for k, v in tail.items()})
        sources = {
            "learned": (context, lambda raw, c, t: opponent.actions(raw, c)),
            "state_only_BC": (context, lambda raw, c, t: bc.actions(raw)),
            "actual_specialist_on_imagined_observation": (context, lambda raw, c, t: specialist_on_imagined_observation(env, complete, raw, red_params)),
            "actual_red_action_injection": (context, lambda raw, c, t: actual["red_action"][:, t]),
            "learned_zero_context": (np.zeros_like(context), lambda raw, c, t: opponent.actions(raw, c)),
            "learned_prototype_context": (opponent.prototypes[label], lambda raw, c, t: opponent.actions(raw, c)),
        }
        if shuffled_context is not None:
            sources["learned_shuffled_prefix_context"] = (shuffled_context, lambda raw, c, t: opponent.actions(raw, c))
        rows = {}
        true_current = actual["state"][:, :-1].reshape(-1, 66)
        for name, (conditioning, source) in sources.items():
            parts = []
            for indices, pool_key in pools:
                red_source = (lambda raw, c, t, selected=indices: actual["red_action"][selected, t]) if name == "actual_red_action_injection" else source
                parts.append(imagined_candidates(agent, initial, actions[indices], conditioning, mean, std, red_source, key=pool_key))
            predicted = {field: np.concatenate([part[field] for part in parts]) for field in parts[0]}
            unchanged_context = name in {"learned", "state_only_BC", "actual_specialist_on_imagined_observation", "actual_red_action_injection"}
            if name == "learned":
                np.testing.assert_allclose(predicted["planner_return"][:len(planner_actions)],
                    np.asarray(inspection["candidate_values"])[0], atol=2e-4, rtol=2e-5)
            np.savez_compressed(h_dir / f"{name}.npz", **predicted)
            metrics = compare_candidates(predicted, actual, std, agent.discount,
                capture_distance=float(env.rad[0] + env.rad[1]), arena=env.arena,
                actual_tail=tail["return"] if unchanged_context else None)
            if name == "actual_red_action_injection":
                metrics["injected_action_boundary"] = "Actual red actions only on real valid transitions; zero after real stop is padding and any later model trajectory is extrapolation"
                metrics["injected_padding_transitions"] = int((~actual["valid"] & predicted["valid"]).sum())
            if name != "actual_red_action_injection":
                matched = np.asarray(source(true_current, np.broadcast_to(conditioning, (len(true_current), len(conditioning))), 0)).reshape(actual["red_action"].shape)
                metrics["opponent_action_error_on_matched_true_states"] = float(np.square(matched - actual["red_action"])[actual["valid"]].sum(-1).mean())
            if unchanged_context:
                realized = discounted_rewards(actual["reward"], actual["valid"], agent.discount) + agent.discount ** horizon * tail["return"]
                error = predicted["planner_return"] - realized
                metrics["deployed_selected_candidate_error"] = float(error[selected])
                metrics["deployed_final_elite_mean_error"] = float(error[elite].mean())
                metrics["independent_random_mean_error"] = float(error[random_rows].mean())
                if name == "learned":
                    original_error = np.asarray(inspection["candidate_values"])[0] - realized[:len(planner_actions)]
                    metrics["original_planner_selected_value_error"] = float(original_error[selected])
                    metrics["original_planner_elite_value_error"] = float(original_error[elite].mean())
            else:
                metrics["tail_calibration_status"] = "not compared: changed context also changes conditional blue policy"
            rows[name] = metrics
        save_json(h_dir / "metrics.json", rows)
        metadata["steps"].append({"horizon": horizon, "methods": rows, "selected_candidate": selected})
    metadata["full_remainder"] = full_remainder_scene(directory / "full_remainder", env, agent, opponent, bc, mean, std,
        red_params, complete, context, step_seed, candidate_key)
    metadata["status"] = "completed diagnostic"
    save_json(directory / "metrics.json", metadata)
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--stage", choices=("pilot", "main"), required=True)
    for name in ("campaign", "out", "spec-repo", "protocol"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--spec-commit", required=True)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args(argv)
    if not args.execute:
        print("No diagnostics executed; freeze protocol and reviewed source first.")
        return 0
    protocol_relative = args.protocol.resolve().relative_to(args.spec_repo.resolve())
    committed = subprocess.check_output(["git", "show", f"{args.spec_commit}:{protocol_relative}"], cwd=args.spec_repo)
    if len(args.spec_commit) != 40 or committed != args.protocol.read_bytes():
        raise ValueError("private protocol bytes must match the supplied exact commit")
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() != args.code_commit:
        raise ValueError("diagnostic source HEAD mismatch")
    for command in (["git", "diff", "--name-only", "HEAD", "--", "src", "scripts", "uv.lock"],
                    ["git", "ls-files", "--others", "--exclude-standard", "--", "src", "scripts"]):
        if subprocess.check_output(command, cwd=ROOT, text=True).strip():
            raise ValueError("commit reviewed diagnostic source before executing")
    protocol = json.loads(committed)
    cfg = protocol["frozen_diagnostics"][args.stage]
    if cfg["prefix_lengths"] != [1, 8] or cfg["horizons"] != [1, 3, 10] or cfg["random_candidates"] != 32:
        raise ValueError("unexpected diagnostic budgets; amend and review driver before changing them")
    if len(cfg["fitting_seeds"]) != len(set(cfg["fitting_seeds"])) or cfg["resets_per_specialist"] < 1:
        raise ValueError("unique fitting seeds and a positive reset budget required")
    campaign_binding = json.loads((args.campaign / "campaign.json").read_text())
    if not set(cfg["fitting_seeds"]) <= set(campaign_binding["seeds"]) or not set(cfg["specialist_checkpoints"]) <= set(campaign_binding["configuration"]["training_checkpoints"]):
        raise ValueError("A01 requires frozen fitted models and specialists seen during training")
    args.out.mkdir(parents=True, exist_ok=False)
    sources = code_hashes()
    dependencies = {str(args.protocol): file_sha256(args.protocol),
                    str(args.campaign / "campaign.json"): file_sha256(args.campaign / "campaign.json")}
    binding = {"private_commit": args.spec_commit, "protocol_sha256": file_sha256(args.protocol),
               "executable_commit": args.code_commit, "sources": sources, "configuration": cfg,
               "source_campaign": campaign_binding, "source_campaign_sha256": file_sha256(args.campaign / "campaign.json"),
               "runtime": package_versions(), "status": "running"}
    save_json(args.out / "manifest.json", binding)
    env = make_env("capture", continuous=True)
    if env.max_steps != 100:
        raise ValueError("this protocol and clock invariant require the unchanged100-step environment")
    records = []
    for seed in cfg["fitting_seeds"]:
        agent, opponent, bc, mean, std, source_manifest, artifacts = load_controller(args.campaign, seed, campaign_binding)
        dependencies.update(artifacts)
        changes = {name: {"training": digest, "diagnostic": sources.get(name)}
                   for name, digest in source_manifest["binding"]["source_hashes"].items()
                   if sources.get(name) != digest and name.startswith("src/")}
        if not set(changes) <= set(cfg.get("allowed_training_source_differences", [])):
            raise ValueError(f"unreviewed executable changes since frozen controller fitting: {sorted(changes)}")
        binding.setdefault("training_source_differences", {})[str(seed)] = changes
        before = flax.serialization.to_bytes(agent)
        for checkpoint in cfg["specialist_checkpoints"]:
            specialists = {}
            for label, objective in enumerate(OBJECTIVE_TYPES):
                item = source_manifest["binding"]["specialists"][f"{checkpoint}:{objective}"]
                if file_sha256(item["path"]) != item["sha256"]:
                    raise ValueError("frozen source specialist changed")
                dependencies[item["path"]] = item["sha256"]
                specialists[label] = load_continuous_actor_params(item["path"])
            for reset_index in range(cfg["resets_per_specialist"]):
                reset, step_seed, policy_key, candidate_key = scene_keys(seed, checkpoint, reset_index)
                for length in cfg["prefix_lengths"]:
                    prefixes = {label: frozen_prefix(env, agent, opponent, specialists[label], reset, step_seed, length, policy_key)
                                for label in range(3)}
                    for label, objective in enumerate(OBJECTIVE_TYPES):
                        name = f"seed_{seed}_checkpoint_{checkpoint}_{objective}_reset_{reset_index}_prefix_{length}"
                        directory = args.out / name
                        directory.mkdir()
                        prefix, donor = prefixes[label], prefixes[(label + 1) % 3]
                        shuffled = np.asarray(donor["context_carry"].context[0]) if donor["valid_length"] == length else None
                        row = diagnostic_scene(directory, env, agent, opponent, bc, mean, std, specialists[label],
                            prefix, step_seed, candidate_key, cfg, label=label, shuffled_context=shuffled)
                        row.update(seed=seed, checkpoint=checkpoint, objective=objective, reset_index=reset_index,
                                   reset_key=reset, step_key=step_seed)
                        if row["status"] == "completed diagnostic":
                            row["closed_loop"] = {}
                            for mode in ("policy_only", "MPPI"):
                                real = closed_loop_suffix(env, agent, opponent, specialists[label], prefix["complete_state"],
                                    prefix["context_carry"], step_seed, policy_key, mpc=mode == "MPPI")
                                np.savez_compressed(directory / f"closed_loop_{mode}.npz", **real)
                                row["closed_loop"][mode] = {"suffix_return": float(real["reward"].sum()),
                                    "whole_episode_return": float(real["reward"].sum() + prefix["reward"].sum()),
                                    "captured": real["captured"], "resources_collected": real["resources_collected"],
                                    "controller_seconds": float(real["controller_seconds"].sum())}
                            if reset_index == 0:
                                plot_diagnostic(directory, row)
                                plot_full_remainder(directory)
                        records.append(row)
                        save_json(args.out / "results.json", records)
                        print(name, row["status"], flush=True)
        if before != flax.serialization.to_bytes(agent):
            raise RuntimeError("frozen controller weights changed during diagnostic")
    if sources != code_hashes():
        raise RuntimeError("diagnostic executable changed")
    for path, digest in dependencies.items():
        if file_sha256(path) != digest:
            raise RuntimeError(f"diagnostic dependency changed: {path}")
    binding.update(status="complete", records=len(records), dependencies=dependencies,
        failed_full_remainder_diagnostics=sum(row.get("full_remainder", {}).get("status", "complete") != "complete" for row in records),
        artifacts={str(p.relative_to(args.out)): file_sha256(p)
        for p in args.out.rglob("*") if p.is_file() and p.name != "manifest.json"})
    save_json(args.out / "manifest.json", binding)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
