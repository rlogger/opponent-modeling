"""Cheap recorded MPC replay export checks; no training or simulator changes."""
from types import SimpleNamespace

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
pytest.importorskip("PIL")

from PIL import Image  # noqa: E402

from tag_objectives.rendering import ArenaRenderer, save_episode_replay  # noqa: E402


def _trace():
    env = SimpleNamespace(num_landmarks=0, num_good_agents=1, num_agents=2,
                          num_adversaries=1, num_resources=1, num_lava=1,
                          arena=2.5, rad=np.array([0.1, 0.075]), max_steps=5)
    # Two actual transitions followed by deliberately invalid padding.
    state = np.full((1, 4, 15), np.nan)
    for t in range(3):
        state[0, t] = [-1 + 0.1 * t, 0, 1 - 0.2 * t, 0, 0.1, 0, -0.2, 0,
                       0, 1, float(t == 2), -1, 1, 0.3, t / env.max_steps]
    data = {"state": state, "blue_reward": np.array([[1., 2., np.nan]]),
            "valid_length": np.array([2]),
            "terminated_capture": np.array([[False, True, False]]),
            "truncated_timeout": np.array([[False, False, False]])}
    return env, data


def test_export_uses_only_valid_observed_frames(tmp_path, monkeypatch):
    env, data = _trace()
    before = {key: value.copy() for key, value in data.items()}
    observed = []
    original = ArenaRenderer.draw

    def record(self, frame, history=None):
        observed.append((frame.step, frame.score, frame.terminal, self.source, len(history)))
        return original(self, frame, history)

    monkeypatch.setattr(ArenaRenderer, "draw", record)
    paths = save_episode_replay(data, env, tmp_path / "replays" / "capture", fps=2,
                                title="Real environment / TD-MPC online 0s")
    assert set(paths) == {"gif", "png"}
    with Image.open(paths["gif"]) as gif:
        assert gif.format == "GIF" and gif.n_frames >= 3
        duration = 0
        for t in range(gif.n_frames):
            gif.seek(t)
            duration += gif.info["duration"]
        assert duration >= 2500  # Three observed frames plus a 1 s terminal hold.
    with Image.open(paths["png"]) as png:
        assert png.format == "PNG" and min(png.size) > 100
        png.load()
    assert set(observed) == {(0, 0.0, None, "recorded", 1),
                             (1, 1.0, None, "recorded", 2),
                             (2, 3.0, "capture", "recorded", 3)}
    for key in data:
        np.testing.assert_array_equal(data[key], before[key])


@pytest.mark.parametrize("length", [-1, 4, 0.5])
def test_replay_rejects_invalid_length(tmp_path, length):
    env, data = _trace()
    data["valid_length"] = np.array([length])
    with pytest.raises(ValueError, match="valid_length"):
        save_episode_replay(data, env, tmp_path / "bad")


def test_replay_rejects_post_terminal_transitions(tmp_path):
    env, data = _trace()
    data["terminated_capture"][0, 0] = True
    with pytest.raises(ValueError, match="first terminal"):
        save_episode_replay(data, env, tmp_path / "bad")


@pytest.mark.parametrize("agent, name", [(0, "P1"), (1, "B")])
def test_default_camera_stays_on_arena_and_labels_escape(tmp_path, monkeypatch, agent, name):
    env, data = _trace()
    data["state"][0, 2, 2 * agent:2 * agent + 2] = [80., -70.]
    before = {key: value.copy() for key, value in data.items()}
    observed = []
    original = ArenaRenderer.draw

    def record(self, frame, history=None):
        artists = original(self, frame, history)
        observed.append({"step": frame.step, "xlim": self.ax.get_xlim(),
                         "ylim": self.ax.get_ylim(),
                         "visible": [label.get_visible() for label in self.offscreen],
                         "text": self.offscreen[agent].get_text(),
                         "anchor": np.array(self.offscreen[agent].xy),
                         "positions": np.array([body.center for body in self.agents])})
        return artists

    monkeypatch.setattr(ArenaRenderer, "draw", record)
    save_episode_replay(data, env, tmp_path / "arena", fps=1)
    limit = env.arena + 0.5
    assert {item["step"] for item in observed} == {0, 1, 2}
    for item in observed:
        # A future escape must not shrink frame zero or move the camera later.
        assert item["xlim"] == (-limit, limit)
        assert item["ylim"] == (-limit, limit)
        np.testing.assert_array_equal(item["positions"],
                                      data["state"][0, item["step"], :4].reshape(2, 2))
        assert item["visible"] == [item["step"] == 2 and i == agent for i in range(2)]
        if item["step"] == 2:
            assert item["text"].startswith(f"{name} off-screen")
            assert "80" in item["text"] and "-70" in item["text"]
            assert 0.9 * limit <= np.abs(item["anchor"]).max() <= limit
    for key in data:
        np.testing.assert_array_equal(data[key], before[key])


def test_full_camera_contains_complete_valid_path(tmp_path, monkeypatch):
    env, data = _trace()
    data["state"][0, 2, 2:4] = [80., -70.]
    observed = []
    original = ArenaRenderer.draw

    def record(self, frame, history=None):
        artists = original(self, frame, history)
        observed.append((self.ax.get_xlim(), self.ax.get_ylim(),
                         [label.get_visible() for label in self.offscreen]))
        return artists

    monkeypatch.setattr(ArenaRenderer, "draw", record)
    save_episode_replay(data, env, tmp_path / "full", fps=1, camera="full")
    valid_positions = data["state"][0, :3, :4].reshape(-1, 2, 2)
    assert observed
    assert len({(xlim, ylim) for xlim, ylim, _ in observed}) == 1
    for xlim, ylim, visible in observed:
        assert np.isfinite(xlim + ylim).all()  # Ignore deliberately NaN padding.
        for dim, (lower, upper) in enumerate((xlim, ylim)):
            assert lower <= np.min(valid_positions[..., dim] - env.rad)
            assert upper >= np.max(valid_positions[..., dim] + env.rad)
        assert visible == [False, False]


@pytest.mark.parametrize("camera", ["auto", "", None])
def test_replay_rejects_unknown_camera(tmp_path, camera):
    env, data = _trace()
    with pytest.raises(ValueError, match="camera"):
        save_episode_replay(data, env, tmp_path / "bad", camera=camera)
