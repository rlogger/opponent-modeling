#!/usr/bin/env python3
"""Render matched recorded episodes with the shared arena renderer; no training."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mopa.manifest import file_sha256  # noqa: E402
from tag_objectives import make_env  # noqa: E402
from tag_objectives.rendering import ArenaRenderer, frame_from_markov  # noqa: E402

TYPES = ("capture", "risk", "curious")


def recorded_frame(data, episode, time_index, env):
    """A terminal transition at L-1 belongs to frame L; never render padding."""
    t = min(int(time_index), int(data["valid_length"][episode]))
    if t < 0:
        raise ValueError("time_index cannot be negative")
    terminal = None
    if t:
        if data["terminated_capture"][episode, t - 1]:
            terminal = "capture"
        elif data["truncated_timeout"][episode, t - 1]:
            terminal = "timeout"
    frame = frame_from_markov(env, data["state"][episode, t], terminal=terminal,
                             score=float(data["blue_reward"][episode, :t].sum()))
    history = data["state"][episode, :t + 1, :2 * env.num_agents].reshape(t + 1, env.num_agents, 2)
    return frame, history


def viewport(data, episodes, env, steps):
    """Shared offline viewport includes all shown agents, resources and hazards."""
    limit = env.arena
    for episode in episodes:
        end = min(steps, int(data["valid_length"][episode])) + 1
        limit = max(limit, float(np.abs(data["state"][episode, :end, :4]).max()) + float(np.max(env.rad)))
        initial = frame_from_markov(env, data["state"][episode, 0])
        limit = max(limit, float(np.abs(initial.resource_pos).max()),
                    float((np.abs(initial.lava_pos) + initial.lava_rad[:, None]).max()))
    return limit + 0.15


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "artifacts/continuous/dataset.npz")
    parser.add_argument("--checkpoint", type=int, default=2)
    parser.add_argument("--episode", type=int, default=0, help="Index within each objective/checkpoint group; not success-selected")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/environment")
    args = parser.parse_args()
    if args.episode < 0 or args.steps < 1 or args.fps < 1:
        parser.error("episode must be nonnegative; steps and fps must be positive")
    with np.load(args.dataset, allow_pickle=False) as raw:
        data = {k: raw[k] for k in ("state", "valid_length", "checkpoint_seed", "objective_label", "environment_seed",
                                    "terminated_capture", "truncated_timeout", "blue_reward")}
    episodes = []
    for k in range(3):
        group = np.flatnonzero((data["objective_label"] == k) & (data["checkpoint_seed"] == args.checkpoint))
        if args.episode >= len(group):
            raise ValueError(f"episode index outside {TYPES[k]} group")
        episodes.append(int(group[args.episode]))
    for episode in episodes[1:]:
        np.testing.assert_array_equal(data["state"][episode, 0], data["state"][episodes[0], 0])
        np.testing.assert_array_equal(data["environment_seed"][episode], data["environment_seed"][episodes[0]])
    envs = [make_env(name, continuous=True) for name in TYPES]
    steps = min(args.steps, int(data["valid_length"][episodes].max()))
    limit = viewport(data, episodes, envs[0], steps)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    dataset_hash = file_sha256(args.dataset)
    env_sources = [ROOT / "src/tag_objectives" / name for name in ("objectives.py", "resources.py", "actions.py")]
    before = {str(path.relative_to(ROOT)): file_sha256(path) for path in env_sources}
    fig, axs = plt.subplots(1, 3, figsize=(14, 6.4), facecolor="#f4f1e9")
    fig.subplots_adjust(left=0.035, right=0.985, bottom=0.17, top=0.78, wspace=0.13)
    renderers = [ArenaRenderer(ax, env, source="recorded", title=name.title(), limit=limit)
                 for ax, env, name in zip(axs, envs, TYPES)]
    heading = fig.suptitle("", y=0.97, fontsize=13, color="#243b46", fontweight="bold")
    fig.supxlabel("Red = opponent predator · Blue = controlled prey · Gold = resource · Hatched discs = lava\n"
                  "Dashed square is a soft shaping reference, not a wall. Lava penalizes only risk at current defaults.",
                  y=0.025, fontsize=8, color="#58666b")
    elapsed = []

    def draw(t):
        start = time.perf_counter()
        heading.set_text(f"SAME START, THREE OBJECTIVES  /  step {t:03d}\n"
                         "Recorded continuous control · fixed prey policy checkpoint")
        artists = []
        for renderer, env, episode in zip(renderers, envs, episodes):
            frame, history = recorded_frame(data, episode, t, env)
            artists.extend(renderer.draw(frame, history=history))
        elapsed.append(time.perf_counter() - start)
        return artists

    draw(min(12, steps))
    fig.savefig(out / "arena.png", dpi=175, facecolor=fig.get_facecolor())
    clip = FuncAnimation(fig, draw, frames=steps + 1, interval=1000 / args.fps, blit=False)
    clip.save(out / "replay.gif", writer=PillowWriter(fps=args.fps), dpi=110)
    draw(min(steps, int(data["valid_length"][episodes[0]])))
    fig.savefig(out / "terminal.png", dpi=175, facecolor=fig.get_facecolor())
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 7.6), layout="constrained", facecolor="#f4f1e9")
    debug = ArenaRenderer(ax, envs[0], source="recorded", title="Observation geometry", limit=limit, observations=True)
    frame, history = recorded_frame(data, episodes[0], min(12, steps), envs[0])
    debug.draw(frame, history=history)
    fig.suptitle("DEBUG VIEW / observer overlay, not extra policy inputs", fontsize=12, color="#243b46")
    fig.supxlabel("Links show nearest available resources for blue and nearest lava for each agent.\n"
                  "Predators do not observe resources; the shared renderer is an external observer.", fontsize=9)
    fig.savefig(out / "observations.png", dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    assert before == {str(path.relative_to(ROOT)): file_sha256(path) for path in env_sources}
    assert dataset_hash == file_sha256(args.dataset)
    summary = {"source": "recorded continuous dataset; no training or new rollout generation",
               "source_repository": "rlogger/marl-opp-aware", "reference_commit": "aecbab5daf6da402029e953be114a52e62a46c26",
               "episodes": dict(zip(TYPES, episodes)), "selection": "same group index, matched reset keys, no outcome filter",
               "matched_initial_states": True, "clip_frames": steps + 1,
               "observed_frames_per_episode": {name: min(steps, int(data['valid_length'][ep])) + 1 for name, ep in zip(TYPES, episodes)},
               "captured_panels_hold_terminal_frame": True, "shared_view_halfwidth": limit,
               "agent_radii": np.asarray(envs[0].rad).tolist(), "arena_halfwidth": envs[0].arena,
               "renderer_artist_update_median_ms": 1000 * float(np.median(elapsed[1:])),
               "timing_scope": "artist updates only; excludes image drawing and encoding",
               "dataset_sha256": dataset_hash, "environment_sources_unchanged": before,
               "renderer_sha256": file_sha256(ROOT / "src/tag_objectives/rendering.py"),
               "script_sha256": file_sha256(Path(__file__)),
               "outputs": {name: file_sha256(out / name) for name in ("arena.png", "replay.gif", "terminal.png", "observations.png")}}
    (out / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "README.md").write_text(
        "# Environment presentation\n\nOne renderer, existing data, unchanged simulation. "
        "Predator objectives are observer labels, never new policy inputs.\n\n"
        "![Recorded game](replay.gif)\n\n![Arena detail](arena.png)\n\n"
        "![Capture-aware playback](terminal.png)\n\n![Observation debug overlay](observations.png)\n\n"
        f"Checkpoint {args.checkpoint}, group episode {args.episode}; absolute episode IDs {episodes}. "
        "Starts and reset keys match; blue actions may differ as the fixed policy reacts to each opponent. "
        "Captured panels hold their observed terminal frame while other panels continue. "
        "The fixed offline viewport includes the whole shown clip. These examples are not aggregate performance results.\n\n"
        "```bash\nuv run --locked --extra train --extra plot python scripts/render_environment.py\n```\n\n"
        "Rendering and data checks: [manifest](manifest.json). Visualization options: [catalog](../../docs/VISUALIZATION.md).\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
