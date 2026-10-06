"""Controller/environment/export integration, separate from future RL training."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jaxmarl")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from spatial_blotto import SpatialBlotto  # noqa: E402
from spatial_blotto.controllers import make_controller  # noqa: E402
from spatial_blotto.rollout import (  # noqa: E402
    episode_payload,
    joint_actions,
    rollout_episode,
    save_trajectory,
    validate_episode,
)

ROOT = Path(__file__).resolve().parents[1]


def driver():
    spec = importlib.util.spec_from_file_location(
        "blotto_demo", ROOT / "scripts/demo_spatial_blotto.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_team_adapter_preserves_team_and_unit_order():
    env = SpatialBlotto(team_size=2)
    red = jnp.array([[0.1, 0.2], [0.3, 0.4]])
    blue = -red
    actions = jax.jit(lambda a, b: joint_actions(env, a, b))(red, blue)
    for i, name in enumerate(env.agents):
        np.testing.assert_array_equal(actions[name], np.concatenate((red, blue))[i])
    with pytest.raises(ValueError, match="shape"):
        joint_actions(env, jnp.zeros((3, 2)), blue)


@pytest.mark.parametrize("size", [2, 3])
@pytest.mark.parametrize("mode", ["ownership", "zero_sum"])
def test_complete_episode_preserves_terminal_and_transition_alignment(size, mode):
    env = SpatialBlotto(team_size=size, max_steps=24, reward_mode=mode)
    red, blue = (make_controller(env, t, "balanced") for t in ("red", "blue"))
    episode = jax.jit(lambda k: rollout_episode(env, red, blue, k))(
        jax.random.PRNGKey(11)
    )
    validate_episode(env, episode)
    assert episode.observations.shape == (25, size * 2, env.obs_size)
    assert episode.world_states.shape == (25, env.state_size)
    assert episode.actions.shape == (24, size * 2, 2)
    assert int(episode.states.step[-1]) == 24
    assert bool(episode.states.done[-1])
    assert not np.asarray(episode.states.done[:-1]).any()
    np.testing.assert_array_equal(episode.zone_counts[-1].sum(axis=1), [size, size])
    for t in (0, 9, 23):
        before = jax.tree.map(lambda a: a[t], episode.states)
        actions = {a: episode.actions[t, i] for i, a in enumerate(env.agents)}
        obs, after, _, _, info = env.step_env(jax.random.PRNGKey(0), before, actions)
        np.testing.assert_allclose(after.p_pos, episode.states.p_pos[t + 1], atol=1e-6)
        np.testing.assert_array_equal(info["team_rewards"], episode.team_rewards[t])
        np.testing.assert_array_equal(
            np.stack([obs[a] for a in env.agents]), episode.observations[t + 1]
        )
    if mode == "zero_sum":
        np.testing.assert_array_equal(episode.team_rewards.sum(axis=1), 0)


def test_vectorized_matches_equal_independent_seeded_runs():
    env = SpatialBlotto(team_size=2, max_steps=4)
    red = make_controller(env, "red", "reactive")
    blue = make_controller(env, "blue", "cyclic", period=2)

    def run(key):
        return rollout_episode(env, red, blue, key)

    keys = jax.random.split(jax.random.PRNGKey(17), 3)
    batched = jax.jit(jax.vmap(run))(keys)
    for i, key in enumerate(keys):
        single = run(key)
        for actual, expected in zip(
            jax.tree.leaves(batched), jax.tree.leaves(single), strict=True
        ):
            np.testing.assert_allclose(actual[i], expected, atol=1e-6)


@pytest.mark.parametrize("size", [1, 6])
def test_complete_rollout_at_supported_controller_team_size_limits(size):
    env = SpatialBlotto(team_size=size, max_steps=2)
    red = make_controller(env, "red", "reactive")
    blue = make_controller(env, "blue", "cyclic", period=1)
    episode = jax.jit(lambda key: rollout_episode(env, red, blue, key))(
        jax.random.PRNGKey(13)
    )
    validate_episode(env, episode)
    assert episode.actions.shape == (2, 2 * size, 2)
    assert bool(episode.states.done[-1])


def test_simultaneous_policies_receive_same_current_state_and_no_target_api_is_required():
    env = SpatialBlotto(team_size=1, max_steps=3)

    def red(s):
        return jnp.tanh(s.p_pos[1:2] - s.p_pos[:1])

    def blue(s):
        return jnp.tanh(s.p_pos[:1] - s.p_pos[1:2])

    e = rollout_episode(env, red, blue, jax.random.PRNGKey(0))
    np.testing.assert_array_equal(e.actions[:, 0], -e.actions[:, 1])
    np.testing.assert_array_equal(e.target_zones, -1)
    validate_episode(env, e)


def test_export_is_finite_nonpickled_and_does_not_put_targets_in_learning_arrays(
    tmp_path,
):
    env = SpatialBlotto(max_steps=3)
    red, blue = (make_controller(env, t, "balanced") for t in ("red", "blue"))
    e = rollout_episode(env, red, blue, jax.random.PRNGKey(5))
    payload = episode_payload(
        env,
        e,
        seed=5,
        controllers={"red": {"name": "balanced"}, "blue": {"name": "balanced"}},
        scenario="balanced",
    )
    assert len(payload["frames"]["positions"]) == 4
    assert payload["frames"]["rewards"][0] == [0, 0]
    assert payload["frames"]["done"] == [False, False, False, True]
    assert payload["provenance"]["sourceSha256"]["environment.py"]
    path = save_trajectory(
        tmp_path / "match.npz",
        e,
        {k: v for k, v in payload.items() if k != "frames"},
        env=env,
    )
    with np.load(path, allow_pickle=False) as data:
        assert "target_zones" not in data.files
        assert "targets" not in data.files
        assert data["observations"].shape[0] == data["actions"].shape[0] + 1
        np.testing.assert_array_equal(data["actions"], e.actions)
        assert json.loads(str(data["metadata_json"]))["seed"] == 5
    with pytest.raises(ValueError, match="npz"):
        save_trajectory(tmp_path / "wrong.json", e, {}, env=env)
    with pytest.raises(ValueError, match="nonfinite"):
        validate_episode(env, e.replace(actions=e.actions.at[0, 0, 0].set(jnp.nan)))


@pytest.mark.parametrize("scenario", ["cyclic", "balanced", "reactive", "rotating"])
@pytest.mark.parametrize("size", [2, 3])
def test_demo_scenarios_complete_and_record_current_targets(scenario, size):
    payload, e = driver().simulate(
        3, 30, scenario, "ownership", team_size=size, period=10
    )
    assert len(payload["frames"]["targetZones"]) == 31
    assert len(payload["agents"]) == size * 2
    assert np.isfinite(e.actions).all()
    assert sum(payload["frames"]["done"]) == 1
    if scenario == "cyclic":
        np.testing.assert_array_equal(
            payload["frames"]["counts"][-1], [[size - 1, 1, 0], [0, size - 1, 1]]
        )
    if scenario == "balanced" and size == 3:
        assert payload["frames"]["owners"][-1] == [0, 0, 0]
    if scenario == "rotating":
        assert (
            payload["frames"]["targetZones"][0] != payload["frames"]["targetZones"][10]
        )


def test_cli_installed_package_produces_playable_html_json_and_npz(tmp_path):
    html, js, trace = (
        tmp_path / name for name in ("replay.html", "replay.json", "trace.npz")
    )
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/demo_spatial_blotto.py"),
            "--team-size",
            "2",
            "--steps",
            "4",
            "--scenario",
            "reactive",
            "--reward-mode",
            "zero_sum",
            "--output",
            str(html),
            "--json",
            str(js),
            "--trajectory",
            str(trace),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(js.read_text())
    assert payload["teamSize"] == 2
    assert payload["summary"]["teamReturns"][0] == -payload["summary"]["teamReturns"][1]
    assert 'id="replay-data"' in html.read_text()
    with np.load(trace, allow_pickle=False) as data:
        assert data["actions"].shape == (4, 4, 2)


@pytest.mark.parametrize(
    "arguments",
    [
        ["--team-size", "0"],
        ["--team-size", "1000000000"],
        ["--steps", "0"],
        ["--seed", "-1"],
        ["--red-allocation", "1,1,0"],
        ["--period", "0"],
        ["--period", "2147483648"],
        ["--trajectory", "bad.json"],
        ["--red-allocation", "-1,2,2"],
        ["--blue-policy", "reactive", "--blue-allocation", "1,1,1"],
    ],
)
def test_cli_rejects_invalid_configuration_before_writing(tmp_path, arguments):
    output = tmp_path / "invalid.html"
    with pytest.raises(SystemExit) as error:
        driver().main(["--output", str(output), *arguments])
    assert error.value.code == 2
    assert not output.exists()


@pytest.mark.parametrize("seed", [True, 1.5, -1, 2**32])
def test_simulate_rejects_invalid_seed_type_and_range(seed):
    with pytest.raises(ValueError, match="seed must be an integer"):
        driver().simulate(seed=seed, steps=1)


def test_simulate_rejects_large_team_before_constructing_environment(monkeypatch):
    from spatial_blotto import cli

    def should_not_construct(**kwargs):
        raise AssertionError("environment must not be constructed")

    monkeypatch.setattr(cli, "SpatialBlotto", should_not_construct)
    with pytest.raises(ValueError, match="demo team_size"):
        cli.simulate(team_size=10**9, steps=1)


@pytest.mark.parametrize(
    "kind",
    [
        "missing_observation",
        "truncated_shape",
        "done",
        "reward",
        "counts",
        "action_state_mismatch",
        "nonfinite",
    ],
)
def test_export_rejects_malformed_episodes_without_overwriting(tmp_path, kind):
    env = SpatialBlotto(max_steps=2)
    red, blue = (make_controller(env, team, "balanced") for team in ("red", "blue"))
    e = rollout_episode(env, red, blue, jax.random.PRNGKey(0))
    if kind == "missing_observation":
        e = e.replace(observations=e.observations[:-1])
    elif kind == "truncated_shape":
        e = e.replace(truncated=e.truncated[:0])
    elif kind == "done":
        e = e.replace(states=e.states.replace(done=jnp.zeros_like(e.states.done)))
    elif kind == "reward":
        e = e.replace(team_rewards=jnp.full_like(e.team_rewards, 9))
    elif kind == "counts":
        e = e.replace(zone_counts=e.zone_counts.at[-1, 0, 0].set(99))
    elif kind == "action_state_mismatch":
        e = e.replace(actions=e.actions.at[0, 0].set(jnp.zeros(2)))
    else:
        e = e.replace(actions=e.actions.at[0, 0, 0].set(jnp.nan))
    path = tmp_path / "protected.npz"
    path.write_bytes(b"previous result")
    with pytest.raises(ValueError):
        save_trajectory(path, e, {}, env=env)
    assert path.read_bytes() == b"previous result"
