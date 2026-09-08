#!/usr/bin/env python3
"""Render frozen Equation 3 rollouts; verify the original experiment first."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import flax.serialization
import jax
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mopa.manifest import file_sha256  # noqa: E402
from mopa.tdmpc import create_agent  # noqa: E402
from mopa.zero_s import ZeroSOpponent, rollout_zero_s  # noqa: E402

TYPES = ("capture", "risk", "curious")
RED, BLUE = "#c84a45", "#2876b4"


def verify(run, dataset):
    manifest = json.loads((run / "manifest.json").read_text())
    expected = {run / k: manifest["artifacts"][k] for k in
                ("agent.msgpack", "opponent.msgpack", "config.json", "state_stats.npz", "latent_swap_rollouts.npz")}
    for name in ("src/mopa/action_decoder.py", "src/mopa/zero_s.py", "src/mopa/tdmpc.py", "src/mopa/mppi.py"):
        expected[ROOT / name] = manifest["code"][name]
    expected[dataset] = manifest["dataset"]["sha256"]
    for path, digest in expected.items():
        if file_sha256(path) != digest:
            raise ValueError(f"Original experiment hash mismatch: {path}")
    return manifest, {str(path.resolve()): value for path, value in expected.items()}


def scene(ax, initial, bound, title):
    resources = initial[8:40].reshape(16, 2)
    remaining = initial[40:56] < 0.5
    ax.scatter(*resources[remaining].T, color="#d9a729", marker="*", s=22, alpha=0.8)
    for center, radius in zip(initial[56:62].reshape(3, 2), initial[62:65]):
        ax.add_patch(Circle(center, radius, color="#d58d80", alpha=0.18))
    ax.set(title=title, xlabel="x", ylabel="y", aspect="equal", xlim=(-bound, bound), ylim=(-bound, bound))
    ax.grid(alpha=0.1)
    ax.spines[["top", "right"]].set_visible(False)


def animate_existing(run, out):
    with np.load(run / "latent_swap_rollouts.npz", allow_pickle=False) as saved:
        states = saved["state"]
    if not np.isfinite(states).all():
        raise ValueError("Existing rollout contains nonfinite states")
    bound = max(1.15, float(np.abs(states[..., :4]).max()) * 1.08)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.7), layout="constrained")
    artists = []
    for k, ax in enumerate(axs):
        scene(ax, states[0, k], bound, f"{TYPES[k].title()} prototype")
        red_line, = ax.plot([], [], color=RED, lw=2, label="opponent (red)")
        blue_line, = ax.plot([], [], color=BLUE, lw=2, label="ego (blue)")
        red_dot, = ax.plot([], [], "o", color=RED, markersize=6)
        blue_dot, = ax.plot([], [], "o", color=BLUE, markersize=6)
        artists.append((red_line, blue_line, red_dot, blue_dot))
    axs[0].legend(loc="upper left", fontsize=7)
    heading = fig.suptitle("", fontsize=11)

    def frame(t):
        heading.set_text(f"Frozen Equation 3 · learned rollout · step {t}/{len(states) - 1}\n"
                         "Same initial state and blue actions; fixed horizon, no simulator/reset; initial map shown")
        for k, (red_line, blue_line, red_dot, blue_dot) in enumerate(artists):
            red_line.set_data(states[:t + 1, k, 0], states[:t + 1, k, 1])
            blue_line.set_data(states[:t + 1, k, 2], states[:t + 1, k, 3])
            red_dot.set_data([states[t, k, 0]], [states[t, k, 1]])
            blue_dot.set_data([states[t, k, 2]], [states[t, k, 3]])
        return [heading, *(artist for group in artists for artist in group)]

    animation = FuncAnimation(fig, frame, frames=len(states), interval=150, blit=False)
    animation.save(out / "latent_swap_animation.gif", writer=PillowWriter(fps=7), dpi=110)
    frame(len(states) - 1)
    fig.savefig(out / "latent_swap_animation_final.png", dpi=160)
    plt.close(fig)


def additional_scenes(run, dataset, out, manifest):
    config = json.loads((run / "config.json").read_text())["world_model"]
    with np.load(run / "state_stats.npz", allow_pickle=False) as stats:
        mean, std = stats["mean"], stats["std"]
    opponent = ZeroSOpponent.load(run / "opponent.msgpack")
    template = create_agent(config, 66, key=jax.random.PRNGKey(manifest["seed"]), obs_mean=mean, obs_std=std)
    template = opponent.attach(template, mean, std)
    agent = flax.serialization.from_bytes(template, (run / "agent.msgpack").read_bytes())
    with np.load(dataset, allow_pickle=False) as data:
        risk = np.flatnonzero((data["checkpoint_seed"] == manifest["heldout_checkpoint"]) & (data["objective_label"] == 1))
        if len(risk) < 3:
            raise ValueError("Need at least three held-out risk episodes")
        episodes = risk[[0, len(risk) // 2, -1]]
        horizon = min(30, int(data["valid_length"][episodes].min()))
        initial = np.repeat(data["state"][episodes, 0, None], 3, axis=1)
        blue = np.repeat(data["blue_action"][episodes, :horizon, None], 3, axis=2)
    contexts = np.tile(opponent.prototypes, (3, 1))
    batched_blue = blue.transpose(1, 0, 2, 3).reshape(horizon, 9, 2)
    state, action = rollout_zero_s(agent, initial.reshape(9, 66), batched_blue, contexts, mean, std)
    state = np.asarray(state).reshape(horizon + 1, 3, 3, 66).transpose(1, 0, 2, 3)
    action = np.asarray(action).reshape(horizon, 3, 3, 2).transpose(1, 0, 2, 3)
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError("Additional learned rollout contains nonfinite outputs")
    np.savez(out / "additional_rollouts.npz", episode=episodes, horizon=horizon, state=state,
             red_action=action, blue_action=blue, context=opponent.prototypes)
    fig, axs = plt.subplots(3, 3, figsize=(12, 12), layout="constrained")
    for row, episode in enumerate(episodes):
        bound = max(1.15, float(np.abs(state[row, ..., :4]).max()) * 1.08)
        for k, ax in enumerate(axs[row]):
            scene(ax, state[row, 0, k], bound, f"Episode {episode} · {TYPES[k]} prototype")
            for offset, color, label in ((0, RED, "opponent (red)"), (2, BLUE, "ego (blue)")):
                path = state[row, :, k, offset:offset + 2]
                ax.plot(*path.T, color=color, lw=1.8, label=label)
                ax.scatter(*path[0], color=color, marker="x", s=30)
                ax.scatter(*path[-1], color=color, marker="o", s=25)
            if row == k == 0:
                ax.legend(fontsize=7)
    fig.suptitle(f"Three held-out scenes · frozen Equation 3 · {horizon} learned steps\n"
                 "Same initial state and blue inputs within each row; × start, ● end. Initial map shown.\n"
                 "Fixed horizon: states beyond capture are extrapolations, without simulator ground truth.", fontsize=11)
    fig.savefig(out / "additional_scenes.png", dpi=160)
    plt.close(fig)
    return episodes.tolist(), horizon


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "experiments/continuous_0s")
    parser.add_argument("--dataset", type=Path, default=ROOT / "artifacts/continuous/dataset.npz")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    manifest, inputs = verify(args.run, args.dataset)
    out = args.out or args.run / "visuals"
    out.mkdir(parents=True, exist_ok=True)
    animate_existing(args.run, out)
    episodes, horizon = additional_scenes(args.run, args.dataset, out, manifest)
    outputs = ("latent_swap_animation.gif", "latent_swap_animation_final.png", "additional_scenes.png", "additional_rollouts.npz")
    record = {"original_manifest_sha256": file_sha256(args.run / "manifest.json"), "verified_inputs": inputs,
              "visualization_source": {str(Path(__file__).resolve()): file_sha256(Path(__file__))},
              "episodes": episodes, "horizon": horizon, "training_updates": 0,
              "selection": "first, middle, last held-out risk episodes",
              "outputs": {name: file_sha256(out / name) for name in outputs},
              "limitations": "Frozen learned fixed-horizon rollouts; no simulator ground truth; states past capture are extrapolations."}
    (out / "rollout_manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"out": str(out), "episodes": episodes, "horizon": horizon, "outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()
