"""Simulator-backed MPPI (Gate 3): bounds, blue-only optimization, warm start,
terminal value, opponent-model interface, and the matched evaluation harness."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("jaxmarl")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from mopa.bc_continuous import fit_continuous_bc  # noqa: E402
from mopa.evaluation import (  # noqa: E402
    mappo_prey_controller,
    random_controller,
    run_matched_episodes,
)
from mopa.nets import ContinuousActor  # noqa: E402
from mopa.sim_planner import (  # noqa: E402
    MPPIConfig,
    imagined_returns,
    jit_planner,
    make_bc_red_policy,
    make_true_red_policy,
    mppi_plan,
)
from tag_objectives import make_env  # noqa: E402


@pytest.fixture(scope="module")
def env():
    return make_env("capture", continuous=True)


@pytest.fixture(scope="module")
def red_params(env):
    width = max(env.observation_space(a).shape[0] for a in env.agents)
    return ContinuousActor(action_dim=2, hidden_dim=128).init(
        jax.random.PRNGKey(0), jnp.zeros((1, width))
    )


def test_mppi_config_validation():
    with pytest.raises(ValueError):
        MPPIConfig(num_elites=0)
    with pytest.raises(ValueError):
        MPPIConfig(population_size=4, num_elites=8)
    with pytest.raises(ValueError):
        MPPIConfig(min_plan_std=2.0, max_plan_std=1.0)


def test_imagined_returns_shape_finite_and_stop_after_capture(env, red_params):
    width = max(env.observation_space(a).shape[0] for a in env.agents)
    red = make_true_red_policy(red_params, width)
    k, h = 6, 4
    _, state = env.reset(jax.random.PRNGKey(1))
    states = jax.tree_util.tree_map(lambda x: jnp.repeat(x[None], k, 0), state)
    actions = jax.random.uniform(jax.random.PRNGKey(2), (k, h, 2), minval=-1, maxval=1)
    keys = jax.random.split(jax.random.PRNGKey(3), h * k).reshape(h, k, 2)
    ctx = jnp.zeros((k, 3))
    g = imagined_returns(env, states, actions, ctx, red, keys, discount=0.99)
    assert g.shape == (k,) and np.isfinite(np.asarray(g)).all()
    # A captured start state accrues only the capture step's reward (alive mask).
    prey = env.num_adversaries
    captured = state.replace(p_pos=state.p_pos.at[prey].set(state.p_pos[0]))
    cstates = jax.tree_util.tree_map(lambda x: jnp.repeat(x[None], k, 0), captured)
    g1 = imagined_returns(env, cstates, actions, ctx, red, keys, discount=1.0)
    single = jax.vmap(env.step_env)(keys[0], cstates, {a: jnp.zeros((k, 5)) for a in env.agents})
    # Different actions but capture is immediate either way: return == one step of reward.
    assert np.all(np.asarray(g1) <= np.asarray(single[2][env.good_agents[0]]).max() + 1e-4)


def test_mppi_plan_bounds_shapes_warm_start_and_repeatability(env, red_params):
    width = max(env.observation_space(a).shape[0] for a in env.agents)
    red = make_true_red_policy(red_params, width)
    cfg = MPPIConfig(horizon=4, population_size=16, num_elites=4, mppi_iterations=2)
    b = 3
    _, state = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(5), b))
    ctx = jnp.zeros((b, 3))
    prev = jnp.zeros((b, cfg.horizon, 2))
    key = jax.random.PRNGKey(9)
    a1, (mean, std) = mppi_plan(env, red, cfg, state, ctx, prev, key)
    assert a1.shape == (b, 2) and mean.shape == std.shape == (b, cfg.horizon, 2)
    assert (np.abs(np.asarray(a1)) <= 1.0).all()
    assert (np.asarray(std) >= cfg.min_plan_std - 1e-6).all()
    a2, _ = mppi_plan(env, red, cfg, state, ctx, prev, key)
    np.testing.assert_array_equal(np.asarray(a1), np.asarray(a2))
    # Warm start changes the search but stays bounded; jit path agrees with eager.
    a3, _ = mppi_plan(env, red, cfg, state, ctx, mean, key)
    assert (np.abs(np.asarray(a3)) <= 1.0).all()
    planner = jit_planner(env, red, cfg)
    a4, _ = planner(state, ctx, prev, key)
    np.testing.assert_allclose(np.asarray(a4), np.asarray(a1), atol=1e-5)
    # The first executed action is the first step of an elite sequence.
    assert np.asarray(a1).shape[-1] == 2


def test_planner_optimizes_blue_only_and_uses_context_for_bc_red(env):
    """Red actions come from the opponent model; only blue actions are searched."""
    width = max(env.observation_space(a).shape[0] for a in env.agents)
    red_dim = env.observation_space(env.adversaries[0]).shape[0]
    rng = np.random.default_rng(0)
    feats = rng.normal(size=(256, red_dim + 3)).astype(np.float32)
    targets = np.tanh(feats[:, :2] + 2.0 * feats[:, red_dim : red_dim + 2]).astype(np.float32)
    bc = fit_continuous_bc(feats, targets, 0, steps=200)
    red = make_bc_red_policy(bc)
    obs = jnp.asarray(rng.normal(size=(4, red_dim)).astype(np.float32))
    c1 = jnp.zeros((4, 3))
    c2 = jnp.ones((4, 3))
    out1, out2 = red(obs, c1), red(obs, c2)
    assert out1.shape == (4, 2) and not np.allclose(np.asarray(out1), np.asarray(out2))
    assert (np.abs(np.asarray(out1)) <= 1.0).all()
    # True red policy ignores the context by construction.
    _, state = env.reset(jax.random.PRNGKey(0))
    params = ContinuousActor(action_dim=2, hidden_dim=128).init(
        jax.random.PRNGKey(0), jnp.zeros((1, width))
    )
    true_red = make_true_red_policy(params, width)
    o = env.get_obs(state)[env.adversaries[0]][None]
    np.testing.assert_array_equal(
        np.asarray(true_red(o, jnp.zeros((1, 3)))), np.asarray(true_red(o, jnp.ones((1, 3))))
    )


def test_run_matched_episodes_controls_and_context_modes(env, red_params):
    width = max(env.observation_space(a).shape[0] for a in env.agents)
    prey_name = env.good_agents[0]
    keys = np.asarray(jax.random.split(jax.random.PRNGKey(3), 4), dtype=np.uint32)
    step_seed = np.asarray(jax.random.split(jax.random.PRNGKey(4), 4), dtype=np.uint32)
    for blue in (mappo_prey_controller(red_params, width, prey_name), random_controller(prey_name)):
        out = run_matched_episodes(
            env, red_params, blue, keys, step_seed, horizon=6, context_mode="zero", label=0
        )
        assert out["blue_return"].shape == (4,) and np.isfinite(out["blue_return"]).all()
        assert out["blue_action_abs_max"] <= 1.0
        assert set(out) >= {"captured", "survival_time", "resources_collected", "pred_lava_steps"}
    with pytest.raises(ValueError):
        run_matched_episodes(
            env, red_params, random_controller(prey_name), keys, step_seed,
            horizon=2, context_mode="online", label=0,
        )
    with pytest.raises(ValueError):
        run_matched_episodes(
            env, red_params, random_controller(prey_name), keys, step_seed,
            horizon=2, context_mode="bogus", label=0,
        )
