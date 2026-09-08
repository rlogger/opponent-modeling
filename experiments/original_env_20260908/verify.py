"""Audit complete recorded experiments; never run against an adapting checkpoint.

Example: --directory baseline --directory final --run adapted
Paths are relative to this experiment directory unless absolute.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from mopa.continuous_data import (
    _step_keys,
    deterministic_specialist_action,
    load_continuous_actor_params,
    markov_state,
)
from mopa.manifest import file_sha256, package_versions
from mopa.zero_s import ZeroSOpponent
from tag_objectives import joint_action_dict, make_env
from tag_objectives.teams import freeze_tree

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TYPES = ("capture", "risk", "curious")


def read_json(path):
    return json.loads(Path(path).read_text())


def repository_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def close(actual, expected, *, atol=0.):
    actual, expected = np.asarray(actual), np.asarray(expected)
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=0.)
    return float(np.max(np.abs(actual.astype(float) - expected.astype(float)), initial=0.))


def audit_trace(path, *, opponent=None, specialist=None, checkpoints=(2,)):
    with np.load(path, allow_pickle=False) as archive:
        data = dict(archive)
    n, horizon = data["blue_action"].shape[:2]
    labels = np.unique(data["objective_label"])
    assert len(labels) == 1 and labels[0] in range(3)
    assert np.isin(data["checkpoint_seed"], checkpoints).all()
    valid = data["valid_mask"]
    np.testing.assert_array_equal(valid, np.arange(horizon)[None] < data["valid_length"][:, None])
    assert np.all((data["valid_length"] >= 1) & (data["valid_length"] <= horizon))
    for key in ("state", "blue_action", "red_action", "blue_reward", "context"):
        assert np.isfinite(data[key]).all(), key
    for key in ("blue_action", "red_action"):
        assert np.abs(data[key]).max() <= 1. + 1e-6
        close(data[key][~valid], 0.)
    env = make_env(TYPES[int(labels[0])], continuous=True)
    obs, state = jax.vmap(env.reset)(jnp.asarray(data["environment_seed"], jnp.uint32))
    step_fn = jax.jit(jax.vmap(env.step_env))
    state_fn = jax.jit(lambda s: markov_state(env, s))
    red_fn = None
    if specialist is not None:
        params = load_continuous_actor_params(specialist)
        red_fn = jax.jit(lambda o: deterministic_specialist_action(params, o, 35))
    state_error = close(state_fn(state), data["state"][:, 0])
    reward_error, specialist_error = 0., 0.
    done = np.zeros(n, bool)
    returns, pred_lava, prey_lava = np.zeros(n), np.zeros(n), np.zeros(n)
    for t in range(horizon):
        active = ~done
        np.testing.assert_array_equal(active, valid[:, t])
        if red_fn is not None:
            expected = np.where(active[:, None], red_fn(obs["adversary_0"]), 0.)
            specialist_error = max(specialist_error, close(data["red_action"][:, t], expected, atol=1e-6))
        actions = joint_action_dict(env, data["blue_action"][:, t], data["red_action"][:, t, None])
        new_obs, new_state, reward, dones, info = step_fn(
            _step_keys(jnp.asarray(data["step_seed"], jnp.uint32), t), state, actions
        )
        actual_reward = np.where(active, reward["agent_0"], 0.)
        reward_error = max(reward_error, close(actual_reward, data["blue_reward"][:, t]))
        returns += actual_reward
        pred_lava += np.asarray(info["pred_lava"][:, 0]) * active
        prey_lava += np.asarray(info["prey_lava"][:, 1]) * active
        capture = np.asarray(info["captured"][:, 1]) > .5
        ended = np.asarray(dones["__all__"]) | (t == horizon - 1)
        np.testing.assert_array_equal(data["terminated_capture"][:, t], active & capture)
        np.testing.assert_array_equal(data["truncated_timeout"][:, t], active & ended & ~capture)
        state = freeze_tree(jnp.asarray(active), new_state, state)
        obs = freeze_tree(jnp.asarray(active), new_obs, obs)
        state_error = max(state_error, close(state_fn(state), data["state"][:, t + 1]))
        done |= ended
    np.testing.assert_array_equal(state.capture_t, data["capture_t"])
    close(data["prey_pos"], data["state"][..., 2:4])
    close(data["pred_pos"][..., 0, :], data["state"][..., :2])
    context_error = replay_error = future_error = None
    if opponent is not None:
        expected = opponent.context(data["state"], data["red_action"], data["valid_length"])
        context_error = close(data["context"], expected[:, :-1], atol=2e-5)
        close(data["context"][:, 0], 0.)
        if "replay_context" in data:
            replay_error = close(data["replay_context"], expected, atol=2e-5)
        # Alter pairs at and after t; the decision at t must not change.
        t = horizon // 2
        changed_state, changed_action = data["state"].copy(), -data["red_action"].copy()
        changed_state[:, t:] += .25
        changed_action[:, :t] = data["red_action"][:, :t]
        altered = opponent.context(changed_state, changed_action, data["valid_length"])
        future_error = close(altered[:, :t + 1], expected[:, :t + 1], atol=2e-5)
    else:
        close(data["context"], 0.)
    captured = np.asarray(state.capture_t) >= 0
    metrics = dict(blue_return=returns.astype(np.float32), captured=captured.astype(np.float32),
                   survival_time=np.where(captured, state.capture_t, horizon).astype(np.float32),
                   resources_collected=np.asarray(state.collected).sum(-1).astype(np.float32),
                   pred_lava_steps=pred_lava.astype(np.float32), prey_lava_steps=prey_lava.astype(np.float32))
    result = dict(path=str(path), sha256=file_sha256(path), episodes=n, valid_transitions=int(valid.sum()),
                  state_max_error=state_error, reward_max_error=reward_error,
                  specialist_action_max_error=specialist_error if red_fn is not None else None,
                  capture_timeout_and_padding_exact=True, causal_context_max_error=context_error,
                  replay_context_max_error=replay_error, future_pair_perturbation_max_error=future_error)
    print(f"Verified {path.name}: {n} episodes, {int(valid.sum())} transitions", flush=True)
    return result, metrics, data


def audit_evaluation(directory, matched):
    evaluation = read_json(directory / "evaluation.json")
    run = repository_path(evaluation["run"])
    assert file_sha256(run / "manifest.json") == evaluation["manifest_sha256"]
    for name, digest in evaluation["checkpoint_artifacts"].items():
        assert file_sha256(run / name) == digest, name
    for name in ("objectives", "resources", "actions"):
        relative = f"src/tag_objectives/{name}.py"
        assert file_sha256(ROOT / relative) == evaluation["code"][relative]
    dataset_path = repository_path(evaluation["dataset"]["path"])
    assert file_sha256(dataset_path) == evaluation["dataset"]["sha256"]
    with np.load(dataset_path, allow_pickle=False) as dataset:
        source_rows = {name: dataset[name] for name in
                       ("objective_label", "checkpoint_seed", "environment_seed", "step_seed")}
    opponent = ZeroSOpponent.load(run / "opponent.msgpack")
    records = []
    with np.load(directory / "evaluation_per_episode.npz", allow_pickle=False) as per_episode:
        for row in evaluation["runs"]:
            key = f"{row['opponent']}__{row['controller']}__{row['context_mode']}"
            assert row["context_mode"] == ("online" if row["controller"] == "tdmpc" else "zero")
            specialist = next(s for s in evaluation["specialist_checkpoints"] if s["type"] == row["opponent"])
            specialist_path = repository_path(specialist["path"])
            assert file_sha256(specialist_path) == specialist["sha256"]
            audit, metrics, data = audit_trace(
                directory / row["recordings"]["transitions"], specialist=specialist_path,
                opponent=opponent if row["controller"] == "tdmpc" else None,
            )
            for metric, values in metrics.items():
                if metric not in row:  # Predator lava exposure is not a published blue metric.
                    continue
                close(values, per_episode[f"{key}__{metric}"], atol=1e-4)
                close(values.mean(), row[metric]["mean"], atol=1e-4)
                close(values.std(), row[metric]["std"], atol=1e-4)
            expected_rows = np.flatnonzero((source_rows["checkpoint_seed"] == 2)
                                          & (source_rows["objective_label"] == TYPES.index(row["opponent"])))[:row["n_episodes"]]
            np.testing.assert_array_equal(data["dataset_episode"], expected_rows)
            for field in source_rows:
                np.testing.assert_array_equal(data[field], source_rows[field][expected_rows])
            for field in ("environment_seed", "step_seed", "dataset_episode"):
                match_key = (row["opponent"], field)
                if match_key in matched:
                    np.testing.assert_array_equal(matched[match_key], data[field])
                else:
                    matched[match_key] = data[field]
                if field != "dataset_episode":
                    cross_objective_key = ("all_objectives", field)
                    if cross_objective_key in matched:
                        np.testing.assert_array_equal(matched[cross_objective_key], data[field])
                    else:
                        matched[cross_objective_key] = data[field]
            records.append(audit)
    return dict(directory=str(directory), evaluation_sha256=file_sha256(directory / "evaluation.json"),
                checkpoint_artifacts=evaluation["checkpoint_artifacts"], metrics_match=True, traces=records)


def audit_adaptation(run):
    manifest = read_json(run / "manifest.json")
    adaptation = manifest["online_adaptation"]
    parent = repository_path(manifest["parent_run"]["path"])
    assert file_sha256(parent / "agent.msgpack") == manifest["parent_run"]["agent_sha256"]
    assert file_sha256(parent / "manifest.json") == manifest["parent_run"]["manifest_sha256"]
    for name, digest in manifest["artifacts"].items():
        assert file_sha256(run / name) == digest, name
    frozen = {}
    for name in ("opponent.msgpack", "config.json", "state_stats.npz"):
        frozen[name] = file_sha256(run / name)
        assert frozen[name] == file_sha256(parent / name), name
    before = serialization.msgpack_restore((parent / "agent.msgpack").read_bytes())["model"]
    after = serialization.msgpack_restore((run / "agent.msgpack").read_bytes())["model"]
    changes = {}
    for component in ("encoder", "red_model", "dynamics_model", "reward_model", "value_model", "policy_model", "continue_model"):
        left, right = before[component]["params"], after[component]["params"]
        assert jax.tree.structure(left) == jax.tree.structure(right)
        delta = [np.asarray(b, np.float64) - np.asarray(a, np.float64)
                 for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True)]
        changes[component] = dict(max_abs_change=max([float(np.abs(x).max(initial=0.)) for x in delta] or [0.]),
                                  changed_parameters=sum(int(np.count_nonzero(x)) for x in delta),
                                  l2_change=float(np.sqrt(sum(float(np.square(x).sum()) for x in delta))))
        assert changes[component]["max_abs_change"] == 0. if component in {"encoder", "red_model"} else changes[component]["max_abs_change"] > 0.
    dataset_path = repository_path(manifest["dataset"]["path"])
    assert file_sha256(dataset_path) == manifest["dataset"]["sha256"]
    with np.load(dataset_path, allow_pickle=False) as dataset:
        training = np.flatnonzero(dataset["checkpoint_seed"] != 2)
        np.testing.assert_array_equal(training, manifest["train_episodes"])
        assert set(dataset["checkpoint_seed"][training]) == {0, 1}
        assert int(dataset["valid_length"][training].sum()) == adaptation["offline_transitions"]
    assert adaptation["training_checkpoints"] == [0, 1] and adaptation["heldout_checkpoint"] == 2
    assert adaptation["updates_completed"] == sum(r["updates"] for r in adaptation["rounds"])
    opponent = ZeroSOpponent.load(run / "opponent.msgpack")
    records = []
    for round_info in adaptation["rounds"]:
        for group in round_info["groups"]:
            checkpoint = int(group["checkpoint"])
            assert checkpoint in (0, 1)
            path, specialist = Path(group["transitions"]["path"]), Path(group["specialist"]["path"])
            assert file_sha256(path) == group["transitions"]["sha256"]
            assert file_sha256(specialist) == group["specialist"]["sha256"]
            assert specialist.name.endswith(f"_vmap{checkpoint}.safetensors")
            audit, metrics, _ = audit_trace(path, opponent=opponent, specialist=specialist, checkpoints=(checkpoint,))
            close(metrics["blue_return"].mean(), group["blue_return_mean"], atol=1e-4)
            records.append(audit)
    expected = {str(Path(item["path"])): item["sha256"] for item in adaptation["data"]}
    assert {item["path"]: item["sha256"] for item in records} == expected
    assert sum(item["valid_transitions"] for item in records) == adaptation["online_transitions"]
    return dict(run=str(run), frozen_artifacts=frozen, component_changes=changes,
                training_checkpoints=[0, 1], heldout_checkpoint=2, updates=adaptation["updates_completed"],
                offline_transitions=adaptation["offline_transitions"], online_traces=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", action="append", required=True, help="complete evaluation directory")
    parser.add_argument("--run", help="optional completed adapted run; never pass a live training run")
    args = parser.parse_args()
    source = [Path(__file__), ROOT / "src/mopa/evaluation.py", ROOT / "src/mopa/zero_s.py",
              ROOT / "src/mopa/continuous_data.py", ROOT / "src/mopa/tdmpc.py", ROOT / "src/mopa/mppi.py",
              *(ROOT / "src/tag_objectives" / name for name in ("objectives.py", "resources.py", "actions.py")),
              ROOT / "uv.lock"]
    result = dict(passed=False, dependencies=package_versions(),
                  code={str(path.relative_to(ROOT)): file_sha256(path) for path in source})
    try:
        matched = {}
        result["evaluations"] = [audit_evaluation((HERE / path).resolve(), matched) for path in args.directory]
        result["matched_reset_step_keys_and_dataset_rows"] = True
        if args.run:
            result["adaptation"] = audit_adaptation((HERE / args.run).resolve())
        result["passed"] = True
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (HERE / "verification.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Verification passed: {HERE / 'verification.json'}", flush=True)


if __name__ == "__main__":
    main()
