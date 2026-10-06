"""Replay immutable completed-round traces; --available is a snapshot, not completion."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from mopa.continuous_data import deterministic_specialist_action, load_continuous_actor_params, markov_state
from mopa.manifest import file_sha256, package_versions
from mopa.zero_s import ZeroSOpponent
from tag_objectives import joint_action_dict, make_env
from tag_objectives.teams import freeze_tree

HERE, ARMS = Path(__file__).resolve().parent, ("implicit", "bc", "0s", "ppo", "ppo_z")
ROOT = HERE.parents[1]


def close(actual, expected, *, atol=0.):
    actual, expected = np.asarray(actual), np.asarray(expected)
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=0.)
    return float(np.abs(actual.astype(float) - expected.astype(float)).max(initial=0.))


def verify_trace(path, env, step_fn, state_fn, red_fn, opponent=None):
    with np.load(path, allow_pickle=False) as archive:
        data = dict(archive)
    valid, lengths = data["valid_mask"], data["valid_length"]
    n, horizon = valid.shape
    np.testing.assert_array_equal(valid, np.arange(horizon)[None] < lengths[:, None])
    assert np.all((lengths >= 0) & (lengths <= horizon))
    for name in ("state", "blue_action", "red_action", "blue_reward", "context", "final_context"):
        assert np.isfinite(data[name]).all(), name
    for name in ("blue_action", "red_action"):
        assert np.abs(data[name]).max(initial=0.) <= 1. + 1e-6
        close(data[name][~valid], 0.)
    obs, state = jax.vmap(env.reset)(jnp.asarray(data["environment_seed"], jnp.uint32))
    state_error = close(state_fn(state), data["state"][:, 0])
    reward_error, red_error = 0., 0.
    for t in range(horizon):
        active = valid[:, t]  # Quota cuts can stop still-live episodes, including at reset.
        assert not np.any(active & np.asarray(state.done).any(axis=1))
        red_error = max(red_error, close(data["red_action"][:, t], np.where(active[:, None], red_fn(obs["adversary_0"]), 0.), atol=1e-6))
        actions = joint_action_dict(env, data["blue_action"][:, t], data["red_action"][:, t, None])
        keys = jax.vmap(lambda k: jax.random.fold_in(k, t))(jnp.asarray(data["step_seed"], jnp.uint32))
        new_obs, new_state, reward, _, info = step_fn(keys, state, actions)
        reward_error = max(reward_error, close(np.where(active, reward["agent_0"], 0.), data["blue_reward"][:, t]))
        captured = np.asarray(info["captured"][:, 1]) > .5
        np.testing.assert_array_equal(data["terminated_capture"][:, t], active & captured)
        np.testing.assert_array_equal(data["truncated_timeout"][:, t], active & (lengths == t + 1) & ~captured)
        state, obs = freeze_tree(jnp.asarray(active), new_state, state), freeze_tree(jnp.asarray(active), new_obs, obs)
        state_error = max(state_error, close(state_fn(state), data["state"][:, t + 1]))
    np.testing.assert_array_equal(state.capture_t, data["capture_t"])
    close(data["prey_pos"], data["state"][..., 2:4])
    close(data["pred_pos"][..., 0, :], data["state"][..., :2])
    context_error = future_error = None
    if opponent is None:
        assert data["context"].shape == (n, horizon, 0) and data["final_context"].shape == (n, 0)
    else:
        assert data["context"].shape == (n, horizon, 8) and data["final_context"].shape == (n, 8)
        close(data["context"][lengths == 0], 0.)
        close(data["final_context"][lengths == 0], 0.)
        rows = np.flatnonzero(lengths > 0)
        context_error, future_error = 0., 0.
        if len(rows):
            expected = opponent.context(data["state"][rows], data["red_action"][rows], lengths[rows])
            context_error = max(close(data["context"][rows], expected[:, :-1], atol=2e-5),
                                close(data["final_context"][rows], expected[:, -1], atol=2e-5))
            t = min(3, int(lengths[rows].max()) - 1)
            changed_state, changed_action = data["state"][rows].copy(), data["red_action"][rows].copy()
            changed_state[:, t:] += .25
            changed_action[:, t:] *= -1
            altered = opponent.context(changed_state, changed_action, lengths[rows])
            future_error = close(altered[:, :t + 1], expected[:, :t + 1])
    return dict(path=str(Path(path).resolve()), sha256=file_sha256(path), episodes=n,
                zero_length_rows=int((lengths == 0).sum()), valid_transitions=int(valid.sum()),
                state_max_error=state_error, reward_max_error=reward_error,
                specialist_action_max_error=red_error, causal_context_max_error=context_error,
                future_pair_perturbation_max_error=future_error, capture_quota_padding_exact=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Explicitly enable numerical replay")
    parser.add_argument("--root", type=Path, default=HERE)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--available", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute:
        print("Numerical verification is paused. Explicit --execute is required.")
        return 0
    seeds = [int(s) for s in args.seeds.split(",")]
    output = args.root / "numerical_verification.json"
    sources = [Path(__file__), ROOT / "uv.lock", ROOT / "src/mopa/zero_s.py", ROOT / "src/mopa/continuous_data.py",
               *(ROOT / "src/tag_objectives" / name for name in ("objectives.py", "resources.py", "actions.py", "teams.py"))]
    binding = {"code": {str(p.relative_to(ROOT)): file_sha256(p) for p in sources}, "dependencies": package_versions()}
    result = dict(passed=False, complete=False, scope="completed-artifact snapshot" if args.available else "all requested completed controllers",
                  bindings=binding, snapshots=[], traces=[], reused_traces=0)
    previous = json.loads(output.read_text()) if output.exists() else {}
    cache = {r["path"]: r for r in previous.get("traces", [])} if previous.get("passed") and previous.get("bindings") == binding else {}
    env = make_env("capture", continuous=True)
    step_fn, state_fn = jax.jit(jax.vmap(env.step_env)), jax.jit(lambda s: markov_state(env, s))
    red_functions, finished = {}, 0
    try:
        for seed in seeds:
            shared_path = args.root / f"seed_{seed}" / "shared" / "manifest.json"
            if not shared_path.exists() and args.available:
                continue
            shared = json.loads(shared_path.read_text())
            opponent = None
            for arm in ARMS:
                folder = args.root / f"seed_{seed}" / arm
                manifest_path = folder / "manifest.json"
                if not manifest_path.exists() and args.available:
                    continue
                raw = manifest_path.read_bytes()  # One atomic snapshot; never read the live agent.
                manifest = json.loads(raw)
                assert file_sha256(shared_path) == manifest["shared_manifest"]
                for name, sha in binding["code"].items():
                    if name in manifest["code"]:
                        assert sha == manifest["code"][name], name
                completed = manifest["status"] == "complete"
                assert args.available or completed, f"controller not complete: {folder}"
                finished += int(completed)
                result["snapshots"].append(dict(seed=seed, arm=arm, manifest_sha256=hashlib.sha256(raw).hexdigest(),
                                                rounds=len(manifest["rounds"]), controller_complete=completed))
                context_sha = shared["artifacts"]["0s.msgpack"] if arm in ("0s", "ppo_z") else None
                if context_sha:
                    assert file_sha256(shared_path.parent / "0s.msgpack") == context_sha
                    opponent = opponent or ZeroSOpponent.load(shared_path.parent / "0s.msgpack")
                recordings = [(r["file"], r["sha256"], r["checkpoint"], r["objective"]) for round_ in manifest["rounds"] for r in round_["data"]]
                if completed:
                    recordings += [(name, sha, 2, name.removeprefix("heldout_").removesuffix(".npz")) for name, sha in manifest["evaluation_traces"].items()]
                for name, sha, checkpoint, objective in recordings:
                    path = folder / name
                    assert file_sha256(path) == sha, path
                    specialist = manifest["specialists"][f"{checkpoint}:{objective}"]
                    assert file_sha256(specialist["path"]) == specialist["sha256"]
                    identity = dict(context_sha256=context_sha, specialist_sha256=specialist["sha256"])
                    cached = cache.get(str(path.resolve()))
                    if cached and cached["sha256"] == sha and all(cached[k] == v for k, v in identity.items()):
                        audited = cached
                        result["reused_traces"] += 1
                    else:
                        if specialist["sha256"] not in red_functions:
                            params = load_continuous_actor_params(specialist["path"])
                            red_functions[specialist["sha256"]] = jax.jit(lambda o, p=params: deterministic_specialist_action(p, o, 35))
                        audited = {**verify_trace(path, env, step_fn, state_fn, red_functions[specialist["sha256"]], opponent if context_sha else None), **identity}
                    result["traces"].append(audited)
                print(f"Verified snapshot: seed {seed} {arm}, {len(recordings)} immutable traces", flush=True)
        result.update(passed=True, complete=finished == len(seeds) * len(ARMS),
                      valid_transitions=sum(r["valid_transitions"] for r in result["traces"]))
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        pending = output.with_suffix(".pending.json")
        pending.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        pending.replace(output)


if __name__ == "__main__":
    main()
