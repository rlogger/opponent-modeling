"""Native specialist summaries and exact replay of the first matched episode."""
from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from mopa.continuous_data import markov_state
from tag_objectives import SimpleTagObjectivesMPE
from tag_objectives.rendering import save_episode_replay

OUT = Path(__file__).resolve().parent
OBJECTIVES = ("capture", "risk", "curious")


def main():
    data = dict(np.load(OUT / "dataset.npz"))
    target = OUT / "behavior.png"
    if target.exists() or (OUT / "replays").exists():
        raise SystemExit("Refusing to overwrite existing native-main visual outputs")
    metrics = (("captured", "Capture rate"), ("pred_lava_steps", "Predator lava steps"),
               ("pred_coverage", "Visited grid cells"), ("survival_time", "Episode steps"))
    fig, axes = plt.subplots(1, 4, figsize=(12, 3), constrained_layout=True)
    for ax, (key, label) in zip(axes, metrics):
        values = [float(data[key][data["label"] == i].mean()) for i in range(3)]
        ax.bar(OBJECTIVES, values, color=("#c5534d", "#327aa8", "#dba638"), alpha=0.8)
        for i, value in enumerate(values):
            per_seed = [float(data[key][(data["label"] == i) & (data["ckpt_seed"] == seed)].mean())
                        for seed in range(3)]
            ax.scatter(np.full(3, i), per_seed, s=15, color="#333333", zorder=3)
            ax.text(i, max(value, max(per_seed)) + max(values) * 0.035,
                    f"{value:.3f}" if key == "captured" else f"{value:.2f}", ha="center", fontsize=9)
        ax.set(title=label, ylim=(0, max(1e-3, max(values)) * 1.5))
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Exact-main environment · discrete MAPPO specialists\n600 episodes/type; dots are checkpoint means; fixed capture prey", fontsize=11)
    fig.savefig(target, dpi=150)
    plt.close(fig)

    replay_report = {}
    for label, objective in enumerate(OBJECTIVES):
        episode = int(np.flatnonzero((data["label"] == label) & (data["ckpt_seed"] == 0))[0])
        length = int(data["valid_length"][episode])
        env = SimpleTagObjectivesMPE(pred_type=objective)
        rng, reset = jax.random.split(jax.random.PRNGKey(0))
        np.testing.assert_array_equal(jax.random.split(reset, 200)[0], data["env_seed"][episode])
        _, state = env.reset(jnp.asarray(data["env_seed"][episode]))
        states = [np.asarray(markov_state(env, state))]
        rewards, captured, timeout, keys = [], [], [], []
        max_position_error = 0.0
        for t in range(length):
            rng, key = jax.random.split(rng)
            key = jax.random.split(key, 200)[0]
            actions = {"adversary_0": jnp.asarray(data["pred_act"][episode, t, 0]),
                       "agent_0": jnp.asarray(data["prey_act"][episode, t])}
            _, state, reward, done, _ = env.step_env(key, state, actions)
            expected = np.concatenate((data["pred_pos"][episode, t + 1],
                                       data["prey_pos"][episode, t + 1][None]), axis=0)
            np.testing.assert_allclose(state.p_pos, expected, rtol=0, atol=1e-6)
            max_position_error = max(max_position_error, float(np.max(np.abs(np.asarray(state.p_pos) - expected))))
            states.append(np.asarray(markov_state(env, state)))
            rewards.append(float(reward["agent_0"]))
            captured.append(int(state.capture_t) >= 0)
            timeout.append(bool(done["__all__"]) and not captured[-1])
            keys.append(np.asarray(key))
        replay = {"state": np.asarray(states)[None], "blue_reward": np.asarray(rewards)[None],
                  "terminated_capture": np.asarray(captured)[None], "truncated_timeout": np.asarray(timeout)[None],
                  "valid_length": np.asarray([length]), "reset_key": data["env_seed"][episode][None],
                  "step_key": np.asarray(keys)[None], "blue_action": data["prey_act"][episode:episode + 1, :length],
                  "red_action": data["pred_act"][episode:episode + 1, :length]}
        path = OUT / "replays" / objective
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path.with_suffix(".npz"), **replay)
        save_episode_replay(replay, env, path, title=f"Main · {objective} specialist · first matched episode")
        replay_report[objective] = {"dataset_episode": episode, "checkpoint_seed": 0, "valid_length": length,
                                    "max_position_error_vs_dataset": max_position_error,
                                    "blue_return": float(sum(rewards)), "captured": bool(captured[-1]),
                                    "selection": "first episode of checkpoint 0; no outcome selection"}
    (OUT / "replay_manifest.json").write_text(json.dumps(replay_report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(replay_report, indent=2), flush=True)


if __name__ == "__main__":
    main()
