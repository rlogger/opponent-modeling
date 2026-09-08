"""Forward-only checkpoint audit on fixed, previously saved smoke traces."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("tdmpc_runner", ROOT / "scripts/run_tdmpc.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def main():
    original = ROOT / "experiments/shashank_comparison_20260908/continuous/seed_0"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=original)
    parser.add_argument("--out", type=Path, default=Path(__file__).with_suffix(".json"))
    args = parser.parse_args()
    run = args.run.resolve()
    agent, _, stats = runner.build_template(run)
    model = agent.model
    sources = [ROOT / name for name in ("src/mopa/tdmpc.py", "src/mopa/mppi.py",
                                       "src/mopa/zero_s.py", "scripts/run_tdmpc.py")]
    sources.append(Path(__file__).resolve())
    report = {
        "scope": "Forward-only diagnostic on the OLD saved six-episode smoke traces. "
                 "Not a restored-environment rerun, new held-out performance test, or training result.",
        "regions": "Inside/outside means max(abs(blue position)) <= 2 or > 2 at transition start. "
                   "These are geometric counts, not a statistical training-support test; no wall is implied.",
        "reward_protocol": "One-step reward head at the recorded pre-step state and actual blue action. "
                           "Compare actual recorded red action with frozen 0s prediction using recorded causal context.",
        "q_protocol": "Mean of all five online Q heads at recorded state, blue action and causal context; "
                      "dropout key 1. Positive Q is a model estimate, not a measured future return.",
        "reproduce": "uv run --locked --extra train --extra plot --extra dev python "
                     f"experiments/original_env_20260908/reward_diagnostic.py --run {run} --out {args.out}",
        "run": str(run.relative_to(ROOT)),
        "evaluation_only": True,
        "artifacts_sha256": {name: runner.file_sha256(run / name) for name in
                             ("agent.msgpack", "opponent.msgpack", "state_stats.npz", "config.json", "manifest.json")},
        "source_sha256": {str(path.relative_to(ROOT)): runner.file_sha256(path) for path in sources},
        "horizon": agent.horizon,
        "value_scale": np.asarray(agent.value_scale).tolist(),
        "position_mean": stats["mean"][:4].tolist(),
        "position_std": stats["std"][:4].tolist(),
        "opponents": {},
    }
    for name in ("capture", "risk", "curious"):
        path = original / "closed_loop" / f"{name}__tdmpc__online.npz"
        with np.load(path, allow_pickle=False) as raw:
            data = dict(raw)
        active = data["valid_mask"]
        state = data["state"][:, :-1][active]
        blue, red, context = (jnp.asarray(data[key][active]) for key in
                              ("blue_action", "red_action", "context"))
        x = model.encode(jnp.asarray(state), model.encoder.params, jax.random.PRNGKey(0))
        generated_red = model.red_action(x, context, model.red_model.params)
        predictions = []
        for red_action in (red, generated_red):
            reward, _ = model.reward(x, model.transition_inputs(blue, context, red_action),
                                     model.reward_model.params)
            predictions.append(np.asarray(reward))
        values, _ = model.Q(x, model.value_inputs(blue, context), model.value_model.params,
                            jax.random.PRNGKey(1))
        values = np.asarray(values).mean(axis=0)
        actual = data["blue_reward"][active]
        outside = np.max(np.abs(state[:, 2:4]), axis=-1) > 2

        def summarize(mask):
            if not mask.any():
                return {"n_transitions": 0}
            return {
                "n_transitions": int(mask.sum()),
                "actual_reward_mean": float(actual[mask].mean()),
                "predicted_reward_actual_red_mean": float(predictions[0][mask].mean()),
                "predicted_reward_generated_red_mean": float(predictions[1][mask].mean()),
                "reward_actual_red_mae": float(np.abs(predictions[0][mask] - actual[mask]).mean()),
                "q_mean": float(values[mask].mean()),
                "q_min": float(values[mask].min()),
                "q_max": float(values[mask].max()),
            }

        report["opponents"][name] = {
            "replay_path": str(path.relative_to(ROOT)),
            "replay_sha256": runner.file_sha256(path),
            "n_episodes": int(len(active)),
            "all": summarize(np.ones(len(state), bool)),
            "inside_abs2": summarize(~outside),
            "outside_abs2": summarize(outside),
        }
    stats.close()
    output = args.out
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(output)


if __name__ == "__main__":
    main()
