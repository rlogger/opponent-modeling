"""Rendering is a read-only view, with observed events and no padded future."""
from dataclasses import replace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("jaxmarl")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mopa.continuous_data import markov_state  # noqa: E402
from tag_objectives import make_env, to_mpe_action  # noqa: E402
from tag_objectives.rendering import (  # noqa: E402
    ArenaRenderer,
    frame_from_markov,
    frame_from_state,
)


@pytest.fixture
def scene():
    env = make_env("risk", continuous=True)
    return env, env.reset(jax.random.PRNGKey(7))[1]


@pytest.mark.parametrize("predators", [1, 3])
def test_live_and_recorded_adapters_match(predators):
    env = make_env("risk", continuous=True, num_adversaries=predators)
    _, state = env.reset(jax.random.PRNGKey(1))
    live = frame_from_state(env, state)
    recorded = frame_from_markov(env, np.asarray(markov_state(env, state)))
    for name in ("positions", "velocities", "resource_pos", "collected", "lava_pos", "lava_rad"):
        np.testing.assert_array_equal(getattr(live, name), getattr(recorded, name))
    assert live.step == recorded.step == 0
    assert live.terminal is None


def test_rendering_is_read_only_and_does_not_change_transition(scene, monkeypatch, tmp_path):
    env, state = scene
    action = {a: to_mpe_action(jnp.array([0.3, -0.2])) for a in env.agents}
    key = jax.random.PRNGKey(91)
    expected = env.step_env(key, state, action)
    frame = frame_from_state(env, state)
    before = [np.array(x) for x in jax.tree_util.tree_leaves(state)]
    frame_before = {k: np.array(v) for k, v in vars(frame).items() if isinstance(v, np.ndarray)}
    for value in frame_before:
        getattr(frame, value).setflags(write=False)
    fig, ax = plt.subplots()

    def forbidden(*args, **kwargs):
        raise AssertionError("rendering must not access simulation or RNG")

    with monkeypatch.context() as guard:
        for method in ("reset", "step", "step_env"):
            guard.setattr(env, method, forbidden)
        guard.setattr(jax.random, "split", forbidden)
        guard.setattr(np.random, "random", forbidden)
        renderer = ArenaRenderer(ax, env, source="simulator", observations=True)
        artists = renderer.draw(frame)
        renderer.draw(frame)
        assert len(artists) == len(renderer.draw(frame))
        np.testing.assert_allclose([body.radius for body in renderer.agents], np.asarray(env.rad))
        fig.savefig(tmp_path / "frame.png")
    plt.close(fig)
    assert (tmp_path / "frame.png").stat().st_size > 1000
    for prior, after in zip(before, jax.tree_util.tree_leaves(state)):
        np.testing.assert_array_equal(prior, after)
    for name, prior in frame_before.items():
        np.testing.assert_array_equal(prior, getattr(frame, name))
    actual = env.step_env(key, state, action)
    for a, b in zip(jax.tree_util.tree_leaves(expected), jax.tree_util.tree_leaves(actual)):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("event", ["capture", "timeout"])
def test_recorded_terminal_frame_ignores_padding_and_future_rewards(scene, event):
    spec = spec_from_file_location("render_environment", Path(__file__).resolve().parents[1] / "scripts/render_environment.py")
    script = module_from_spec(spec)
    spec.loader.exec_module(script)
    env, state = scene
    states = np.repeat(np.asarray(markov_state(env, state))[None, None], 6, axis=1)
    states[0, :, -1] = np.arange(6) / env.max_steps
    states[0, 4:] = np.nan  # two padded frames must never reach the renderer
    data = dict(state=states, valid_length=np.array([3]), blue_reward=np.array([[1., 2., 3., 999., 999.]]),
                terminated_capture=np.zeros((1, 5), bool), truncated_timeout=np.zeros((1, 5), bool))
    data["terminated_capture" if event == "capture" else "truncated_timeout"][0, 2] = True
    frame, history = script.recorded_frame(data, 0, 2, env)
    assert frame.terminal is None and frame.score == 3.0 and len(history) == 3
    frame, history = script.recorded_frame(data, 0, 100, env)
    assert frame.terminal == event and frame.step == 3 and frame.score == 6.0
    assert history.shape == (4, 2, 2) and np.isfinite(history).all()


def test_model_output_cannot_claim_observed_outcomes(scene):
    env, state = scene
    frame = replace(frame_from_state(env, state), terminal="capture", score=987.0)
    fig, ax = plt.subplots()
    renderer = ArenaRenderer(ax, env, source="model")
    renderer.draw(frame)
    assert "MODEL / fixed-horizon" in renderer.badge.get_text()
    assert "pred. collected" in renderer.hud.get_text()
    assert "CAPTURED" not in renderer.hud.get_text() and "score" not in renderer.hud.get_text()
    plt.close(fig)


def test_invalid_geometry_and_future_trails_are_rejected(scene):
    env, state = scene
    frame = frame_from_state(env, state)
    with pytest.raises(ValueError, match="dimensions"):
        frame_from_markov(env, np.zeros(65))
    with pytest.raises(ValueError, match="finite"):
        replace(frame, positions=np.full((2, 2), np.nan))
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="source"):
        ArenaRenderer(ax, env, source="unlabelled")
    renderer = ArenaRenderer(ax, env, source="recorded")
    with pytest.raises(ValueError, match="current frame"):
        renderer.draw(frame, history=np.stack([frame.positions, frame.positions + 1]))
    env.num_landmarks = 1
    with pytest.raises(ValueError, match="landmarks"):
        frame_from_state(env, state)
    plt.close(fig)
