"""One read-only arena renderer for simulator, recorded, and model frames."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RenderFrame:
    positions: np.ndarray
    velocities: np.ndarray
    resource_pos: np.ndarray
    collected: np.ndarray
    lava_pos: np.ndarray
    lava_rad: np.ndarray
    step: int
    terminal: str | None = None
    score: float | None = None

    def __post_init__(self):
        positions, resources, lava = map(np.asarray, (self.positions, self.resource_pos, self.lava_pos))
        for name, array in (("positions", positions), ("resource_pos", resources), ("lava_pos", lava)):
            if array.ndim != 2 or array.shape[1] != 2 or not np.isfinite(array).all():
                raise ValueError(f"{name} must be finite with shape (count, 2)")
        if len(positions) < 2 or np.shape(self.velocities) != positions.shape or not np.isfinite(self.velocities).all():
            raise ValueError("velocities must be finite and match at least two agent positions")
        if np.shape(self.collected) != (len(resources),) or not np.isfinite(self.collected).all():
            raise ValueError("collected must be finite and match resource positions")
        if np.shape(self.lava_rad) != (len(lava),) or not np.isfinite(self.lava_rad).all() or np.any(np.asarray(self.lava_rad) <= 0):
            raise ValueError("lava radii must be finite, positive, and match lava positions")
        if not np.isfinite(self.step) or self.step != int(self.step):
            raise ValueError("step must be a finite integer")
        if self.terminal not in (None, "capture", "timeout"):
            raise ValueError("terminal must be capture, timeout, or None")
        if self.score is not None and not np.isfinite(self.score):
            raise ValueError("score must be finite when supplied")


def _check_env(env):
    if env.num_landmarks:
        raise ValueError("the shared render contract does not include landmarks")
    if env.num_good_agents != 1:
        raise ValueError("renderer requires exactly one prey")


def frame_from_state(env, state):
    """Read an existing ObjectiveState without stepping, resetting, or using RNG."""
    _check_env(env)
    step = int(np.asarray(state.step))
    terminal = "capture" if int(np.asarray(state.capture_t)) >= 0 else ("timeout" if step >= env.max_steps else None)
    return RenderFrame(np.asarray(state.p_pos)[:env.num_agents], np.asarray(state.p_vel)[:env.num_agents],
                       np.asarray(state.resource_pos), np.asarray(state.collected), np.asarray(state.lava_pos),
                       np.asarray(state.lava_rad), step, terminal)


def frame_from_markov(env, vector, *, terminal=None, score=None):
    """Decode the recorded Markov schema; flags and score must come from the caller.

    Model collection values remain predictions. The renderer thresholds them
    for display only and marks the count as predicted. A rollout caller can
    replace ``step`` with its known frame index instead of predicted model time.
    """
    _check_env(env)
    a, r, lava_count = env.num_agents, env.num_resources, env.num_lava
    v = np.asarray(vector)
    if v.shape != (4 * a + 3 * r + 3 * lava_count + 1,) or not np.isfinite(v).all():
        raise ValueError("Markov vector must be finite and match environment dimensions")
    offset = 4 * a
    resources, collected = v[offset:offset + 2 * r].reshape(r, 2), v[offset + 2 * r:offset + 3 * r]
    offset += 3 * r
    return RenderFrame(v[:2 * a].reshape(a, 2), v[2 * a:4 * a].reshape(a, 2), resources, collected,
                       v[offset:offset + 2 * lava_count].reshape(lava_count, 2), v[offset + 2 * lava_count:offset + 3 * lava_count],
                       int(np.rint(float(v[-1]) * env.max_steps)), terminal, score)


class ArenaRenderer:
    """Update a fixed set of Matplotlib artists; inputs are always read-only.

    ``limit`` is a symmetric camera half-width, not a physical barrier. The
    default is a fixed arena-scale view; edge annotations identify off-screen
    agents at their real coordinates without changing the simulation geometry.
    ``history`` contains only already-visible positions through this frame.
    """

    def __init__(self, ax, env, *, source, title="", limit=None, observations=False, trail_length=18):
        from matplotlib.collections import LineCollection
        from matplotlib.patches import Circle, Rectangle

        _check_env(env)
        if source not in ("simulator", "recorded", "model"):
            raise ValueError("source must be simulator, recorded, or model")
        limit = float(env.arena + 0.5 if limit is None else limit)
        if not np.isfinite(limit) or limit <= 0 or trail_length < 0:
            raise ValueError("limit must be finite and positive; trail_length cannot be negative")
        self.ax, self.env, self.source = ax, env, source
        self.limit = limit
        self.observations, self.trail_length = bool(observations), int(trail_length)
        self.radii = np.asarray(env.rad)[:env.num_agents]
        if self.radii.shape != (env.num_agents,) or not np.isfinite(self.radii).all() or np.any(self.radii <= 0):
            raise ValueError("agent radii must be positive, finite, and match agents")
        self.colors = ["#c5534d"] * env.num_adversaries + ["#327aa8"]
        ax.set(facecolor="#f8f5ee", aspect="equal", xlim=(-limit, limit), ylim=(-limit, limit))
        ax.set_xticks(np.arange(np.ceil(-limit), np.floor(limit) + 1))
        ax.set_yticks(np.arange(np.ceil(-limit), np.floor(limit) + 1))
        ax.grid(color="#e8e4db", linewidth=0.65, zorder=0)
        ax.tick_params(colors="#a09a8f", labelsize=7, length=0)
        ax.spines[:].set_visible(False)
        # The two blank lines reserve header height in tight/constrained layouts.
        ax.set_title(title + "\n\n", loc="left", fontsize=12, color="#3e423e", pad=8)
        ax.add_patch(Rectangle((-env.arena, -env.arena), 2 * env.arena, 2 * env.arena,
                               fill=False, edgecolor="#868b80", linewidth=1.2, linestyle=(0, (5, 3)), zorder=3))
        self.badge = ax.annotate("MODEL / fixed-horizon" if source == "model" else source.upper(),
                                 (0, 1), xycoords="axes fraction", xytext=(0, 24), textcoords="offset points",
                                 ha="left", va="bottom", fontsize=7.5, color="#6d6860", annotation_clip=False)
        self.hud = ax.annotate("", (0, 1), xycoords="axes fraction", xytext=(0, 9), textcoords="offset points",
                               va="bottom", fontsize=8, color="#4c514b", annotation_clip=False)
        note = f"Dashed ±{env.arena:g} reference · not a wall"
        if observations:
            note += "\nDebug observer overlay: nearest resources / lava (not policy input)"
        self.note = ax.set_xlabel(note, loc="left", labelpad=12, fontsize=7, color="#8b857a")
        self.lava = [Circle((0, 0), 1, facecolor="#e8b8a2", edgecolor="#be8067", linewidth=1.0,
                            hatch="...", alpha=0.5, zorder=2) for _ in range(env.num_lava)]
        for patch in self.lava:
            ax.add_patch(patch)
        self.resources = ax.scatter([], [], s=30, marker="D", facecolor="#dba638", edgecolor="#bc8626", linewidth=0.6, zorder=4)
        self.ghosts = ax.scatter([], [], s=13, marker="D", facecolor="none", edgecolor="#cbbd9e", linewidth=0.65, alpha=0.65, zorder=3)
        self.links = LineCollection([], colors="#94a9a6", linewidths=0.65, linestyles="dotted", alpha=0.6, zorder=3)
        ax.add_collection(self.links)
        self.trails, self.agents, self.ticks, self.labels, self.offscreen = [], [], [], [], []
        for i, (radius, color) in enumerate(zip(self.radii, self.colors)):
            trail = LineCollection([], linewidths=1.7, zorder=5)
            ax.add_collection(trail)
            self.trails.append(trail)
            body = Circle((0, 0), radius, facecolor=color, edgecolor="#ffffff", linewidth=1.1, zorder=8)
            ax.add_patch(body)
            self.agents.append(body)
            tick, = ax.plot([], [], color=color, linewidth=1.4, alpha=0.8, zorder=7)
            self.ticks.append(tick)
            name = f"P{i + 1}" if i < env.num_adversaries else "B"
            self.labels.append(ax.text(0, 0, name, ha="center", va="bottom", fontsize=7, color=color, fontweight="bold", zorder=9))
            self.offscreen.append(ax.annotate(
                "", xy=(0, 0), xytext=(0, 0), textcoords="offset points",
                color=color, fontsize=7, va="center", visible=False, zorder=10,
                arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.0},
            ))

    def draw(self, frame, history=None):
        from matplotlib.colors import to_rgba

        frame.__post_init__()
        positions, resources, lava = map(np.asarray, (frame.positions, frame.resource_pos, frame.lava_pos))
        if positions.shape != (self.env.num_agents, 2) or resources.shape != (self.env.num_resources, 2) or lava.shape != (self.env.num_lava, 2):
            raise ValueError("frame geometry does not match renderer environment")
        history = positions[None] if history is None else np.asarray(history)
        if history.ndim != 3 or history.shape[1:] != positions.shape or not len(history) or not np.isfinite(history).all():
            raise ValueError("history must be finite with shape (time, agents, 2)")
        if not np.allclose(history[-1], positions, atol=1e-6):
            raise ValueError("history must end at the current frame")
        collected = np.asarray(frame.collected) >= 0.5
        self.resources.set_offsets(resources[~collected])
        self.ghosts.set_offsets(resources[collected])
        for patch, center, radius in zip(self.lava, lava, frame.lava_rad):
            patch.center, patch.radius = tuple(center), float(radius)
        for i, (body, tick, label, trail, color) in enumerate(zip(self.agents, self.ticks, self.labels, self.trails, self.colors)):
            point, radius = positions[i], self.radii[i]
            body.center = tuple(point)
            label.set_position(point + np.array([0, radius + 0.035]))
            outside = bool(np.any(np.abs(point) > self.limit))
            indicator = self.offscreen[i]
            indicator.set_visible(outside)
            body.set_visible(not outside)
            label.set_visible(not outside)
            tick.set_visible(not outside)
            if outside:
                # This is an edge pointer, not a relocated/clamped agent body.
                indicator.xy = point * (0.93 * self.limit / np.abs(point).max())
                indicator.set_position((-10 if point[0] >= 0 else 10,
                                        (-1 if point[1] >= 0 else 1) * (8 + 16 * i)))
                indicator.set_ha("right" if point[0] >= 0 else "left")
                name = f"P{i + 1}" if i < self.env.num_adversaries else "B"
                indicator.set_text(f"{name} off-screen ({point[0]:.1f}, {point[1]:.1f})")
            velocity = np.asarray(frame.velocities)[i]
            speed = np.linalg.norm(velocity)
            end = point + velocity / max(float(speed), 1e-12) * radius * 2.8
            tick.set_data([point[0], end[0]], [point[1], end[1]])
            path = history[-(self.trail_length + 1):, i] if self.trail_length else history[-1:, i]
            segments = np.stack([path[:-1], path[1:]], axis=1)
            trail.set_segments(segments)
            trail.set_color([to_rgba(color, alpha) for alpha in np.linspace(0.06, 0.5, len(segments))])
        links = []
        if self.observations:
            available = resources[~collected]
            nearest = np.argsort(np.linalg.norm(available - positions[-1], axis=1))[:self.env.m_nearest_resources]
            links.extend([[positions[-1], available[i]] for i in nearest])
            for point in positions:
                nearest = np.argsort(np.linalg.norm(lava - point, axis=1))[:self.env.n_nearest_lava]
                links.extend([[point, lava[i]] for i in nearest])
        self.links.set_segments(links)
        count_label = "pred. collected" if self.source == "model" else "collected"
        text = f"step {frame.step:03d}   ·   {count_label} {int(collected.sum())}/{len(collected)}"
        if self.source != "model":
            if frame.score is not None:
                text += f"   ·   blue score {frame.score:.1f}"
            if frame.terminal:
                text += "   ·   " + ("CAPTURED" if frame.terminal == "capture" else "TIMEOUT")
        self.hud.set_text(text)
        return (self.hud, self.resources, self.ghosts, self.links, *self.lava,
                *self.trails, *self.agents, *self.ticks, *self.labels, *self.offscreen)


def save_episode_replay(data, env, path: Path, *, episode=0,
                        title="Real environment replay", fps=10, camera="arena"):
    """Export one recorded episode as a GIF and final-frame PNG, never padding.

    ``state`` has shape (episodes, transitions + 1, Markov dimension); rewards
    and termination flags have shape (episodes, transitions). Terminal flags
    belong to the resulting state, and the final terminal frame holds for 1 s.
    ``camera="arena"`` keeps the game readable at a fixed arena-scale zoom;
    ``"full"`` fits the entire valid trajectory for diagnostics. A later escape
    must not shrink the arena from frame zero in the default view.
    No simulator calls or policy changes are made by this observer-only helper.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    if camera not in ("arena", "full"):
        raise ValueError("camera must be arena or full")
    states = np.asarray(data["state"])
    rewards = np.asarray(data["blue_reward"])
    captured = np.asarray(data["terminated_capture"])
    timed_out = np.asarray(data["truncated_timeout"])
    lengths = np.asarray(data["valid_length"])
    if (states.ndim != 3 or states.shape[1] < 1 or
            rewards.shape != (states.shape[0], states.shape[1] - 1) or
            captured.shape != rewards.shape or timed_out.shape != rewards.shape or
            lengths.shape != (states.shape[0],)):
        raise ValueError("replay arrays must have matching episode and transition dimensions")
    if not isinstance(episode, (int, np.integer)) or not 0 <= episode < len(states):
        raise ValueError("episode index is outside replay data")
    length = lengths[episode]
    if not np.isfinite(length) or length != int(length) or not 0 <= length <= rewards.shape[1]:
        raise ValueError("valid_length must be an integer within the recorded transitions")
    if not np.isfinite(fps) or fps < 1 or fps != int(fps):
        raise ValueError("fps must be a positive integer")
    length, fps = int(length), int(fps)
    if not np.isfinite(rewards[episode, :length]).all():
        raise ValueError("valid rewards must be finite")
    if np.any(captured[episode, :max(length - 1, 0)] | timed_out[episode, :max(length - 1, 0)]):
        raise ValueError("valid_length must stop at the first terminal transition")
    scores = np.concatenate(([0.0], np.cumsum(rewards[episode, :length])))
    frames = []
    for t in range(length + 1):
        terminal = None
        if t:
            terminal = "capture" if captured[episode, t - 1] else ("timeout" if timed_out[episode, t - 1] else None)
        frames.append(frame_from_markov(env, states[episode, t], terminal=terminal, score=float(scores[t])))
    history = np.stack([frame.positions for frame in frames])
    initial = frames[0]
    limit = max(env.arena, float(np.abs(history).max()) + float(np.max(env.rad)))
    if len(initial.resource_pos):
        limit = max(limit, float(np.abs(initial.resource_pos).max()))
    if len(initial.lava_pos):
        limit = max(limit, float((np.abs(initial.lava_pos) + initial.lava_rad[:, None]).max()))
    path = Path(path)
    paths = {suffix: path.with_suffix("." + suffix) for suffix in ("gif", "png")}
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 7.2), facecolor="#f4f1e9")
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.12, top=0.83)
    try:
        renderer = ArenaRenderer(ax, env, source="recorded", title=title,
                                 limit=limit + 0.15 if camera == "full" else None)

        def draw(t):
            return renderer.draw(frames[t], history=history[:t + 1])

        indices = list(range(length + 1)) + ([length] * fps if frames[-1].terminal else [])
        clip = FuncAnimation(fig, draw, frames=indices, interval=1000 / fps, blit=False)
        clip.save(paths["gif"], writer=PillowWriter(fps=fps), dpi=100)
        draw(length)
        fig.savefig(paths["png"], dpi=140, facecolor=fig.get_facecolor())
    finally:
        plt.close(fig)
    return paths
