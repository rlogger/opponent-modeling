"""Gate 4 TD-MPC core: three opponent modes through one implementation.

Covers implicit / conditioned / factored construction and updates, the target
network update, causal context inputs, the factored action-clamp /
controllability invariance, termination-vs-truncation masking, the same-
transition continuation head, the identity-encoder baseline, the episode-
bounded replay sampler, and finite losses.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")
pytest.importorskip("optax")
pytest.importorskip("distrax")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from mopa.tdmpc import OPPONENT_MODES, create_agent, load_config  # noqa: E402
from mopa.tdmpc_data import (  # noqa: E402
    SequenceReplay,
    attach_context,
    multistep_model_error,
    reward_calibration,
    state_statistics,
    termination_calibration,
)

OBS_DIM = 6
CTX = 3


def _config(mode: str, *, encoder: str = "mlp", continues: bool = True):
    cfg = load_config(profile="smoke")
    cfg["opponent_mode"] = mode
    cfg["context_dim"] = CTX
    cfg["encoder"]["type"] = encoder
    cfg["world_model"]["predict_continues"] = continues
    cfg["tdmpc2"]["continue_loss_scale"] = 1.0
    return cfg


def _agent(mode: str, seed: int = 0, **kw):
    cfg = _config(mode, **kw)
    extra = {}
    if cfg["encoder"]["type"] == "identity":
        extra = dict(obs_mean=np.zeros(OBS_DIM, np.float32), obs_std=np.ones(OBS_DIM, np.float32))
    return create_agent(cfg, OBS_DIM, key=jax.random.PRNGKey(seed), **extra)


def _batch(agent, key, *, capture_at=None):
    h, b = agent.horizon, agent.batch_size
    k = jax.random.split(key, 6)
    terminated = np.zeros((h, b), bool)
    truncated = np.zeros((h, b), bool)
    if capture_at is not None:
        terminated[capture_at, 0] = True
    truncated[-1, 1] = True
    return dict(
        observations=jax.random.normal(k[0], (h, b, OBS_DIM)),
        next_observations=jax.random.normal(k[1], (h, b, OBS_DIM)),
        actions=jax.random.uniform(k[2], (h, b, 2), minval=-1, maxval=1),
        red_actions=jax.random.uniform(k[3], (h, b, 2), minval=-1, maxval=1),
        rewards=jax.random.normal(k[4], (h, b)),
        terminated=jnp.asarray(terminated),
        truncated=jnp.asarray(truncated),
        context=jax.random.normal(k[5], (h, b, CTX)),
        next_context=jax.random.normal(jax.random.fold_in(k[5], 1), (h, b, CTX)),
    )


def _leaves(agent):
    return jax.tree_util.tree_leaves(
        {
            "e": agent.model.encoder.params,
            "d": agent.model.dynamics_model.params,
            "r": agent.model.reward_model.params,
            "p": agent.model.policy_model.params,
            "v": agent.model.value_model.params,
            "c": agent.model.continue_model.params if agent.model.predict_continues else {},
            "red": agent.model.red_model.params if agent.model.red_model is not None else {},
        }
    )


@pytest.mark.parametrize("mode", OPPONENT_MODES)
def test_each_mode_builds_updates_and_stays_finite(mode):
    agent = _agent(mode)
    m = agent.model
    assert m.opponent_mode == mode and m.context_dim == CTX
    extra = {"implicit": 0, "conditioned": CTX, "factored": 2}[mode]
    assert m.dynamics_model.params["layers_0"]["Dense_0"]["kernel"].shape[0] == m.latent_dim + 2 + extra
    q_extra = 0 if mode == "implicit" else CTX
    q_kernel = m.value_model.params["VmapSequential_0"]["layers_0"]["Dense_0"]["kernel"]
    assert q_kernel.shape == (m.num_value_nets, m.latent_dim + 2 + q_extra, m.latent_dim)
    assert (m.red_model is not None) == (mode == "factored")
    assert m.continue_model is not None

    batch = _batch(agent, jax.random.PRNGKey(1), capture_at=1)
    before = _leaves(agent)
    new, info = agent.update(**batch, key=jax.random.PRNGKey(2))
    for k in ("consistency_loss", "reward_loss", "value_loss", "continue_loss", "red_loss", "total_loss", "policy_loss"):
        assert np.isfinite(np.asarray(info[k])).all(), k
    assert (float(info["red_loss"]) > 0.0) == (mode == "factored")
    assert float(info["continue_loss"]) > 0.0
    after = _leaves(new)
    assert all(bool(jnp.all(jnp.isfinite(a))) for a in after)
    assert any(not bool(jnp.array_equal(a, b)) for a, b in zip(before, after))
    # Target network moved by EMA toward (but not onto) the online ensemble.
    t0 = jax.tree_util.tree_leaves(agent.model.target_value_model.params)
    t1 = jax.tree_util.tree_leaves(new.model.target_value_model.params)
    v1 = jax.tree_util.tree_leaves(new.model.value_model.params)
    assert any(not bool(jnp.array_equal(a, b)) for a, b in zip(t0, t1))
    for a, b, v in zip(t0, t1, v1):
        np.testing.assert_allclose(np.asarray(b), np.asarray(a) * (1 - agent.tau) + np.asarray(v) * agent.tau, atol=1e-6)
    # Planning and acting with context stay bounded.
    obs = jax.random.normal(jax.random.PRNGKey(3), (OBS_DIM,))
    ctx = jnp.ones((CTX,))
    a, plan = new.act(obs, context=ctx, key=jax.random.PRNGKey(4))
    assert a.shape == (2,) and (np.abs(np.asarray(a)) <= 1).all() and plan is not None
    a_pi, _ = new.act(obs, context=ctx, mpc=False, key=jax.random.PRNGKey(4))
    assert (np.abs(np.asarray(a_pi)) <= 1).all()


def test_factored_physics_reward_termination_are_context_invariant():
    """Central controllability invariant: fixed (x, u, v), varied c -> unchanged."""
    agent = _agent("factored")
    m = agent.model
    key = jax.random.PRNGKey(0)
    x = m.encode(jax.random.normal(key, (5, OBS_DIM)), m.encoder.params, key)
    u = jax.random.uniform(jax.random.PRNGKey(1), (5, 2), minval=-1, maxval=1)
    v = jax.random.uniform(jax.random.PRNGKey(2), (5, 2), minval=-1, maxval=1)
    c1, c2 = jnp.zeros((5, CTX)), jax.random.normal(jax.random.PRNGKey(3), (5, CTX))
    a1, a2 = m.transition_inputs(u, c1, v), m.transition_inputs(u, c2, v)
    np.testing.assert_array_equal(np.asarray(a1), np.asarray(a2))
    np.testing.assert_array_equal(
        np.asarray(m.next(x, a1, m.dynamics_model.params)), np.asarray(m.next(x, a2, m.dynamics_model.params))
    )
    np.testing.assert_array_equal(
        np.asarray(m.reward(x, a1, m.reward_model.params)[0]),
        np.asarray(m.reward(x, a2, m.reward_model.params)[0]),
    )
    np.testing.assert_array_equal(
        np.asarray(m.continue_logits(x, a1, m.continue_model.params)),
        np.asarray(m.continue_logits(x, a2, m.continue_model.params)),
    )
    # Fixed x, varied c changes the red action; varied v changes next state.
    v1 = m.red_action(x, c1, m.red_model.params)
    v2 = m.red_action(x, c2, m.red_model.params)
    assert v1.shape == (5, 2) and (np.abs(np.asarray(v1)) <= 1).all()
    assert not np.allclose(np.asarray(v1), np.asarray(v2))
    x_a = m.next(x, m.transition_inputs(u, c1, v), m.dynamics_model.params)
    x_b = m.next(x, m.transition_inputs(u, c1, -v), m.dynamics_model.params)
    assert not np.allclose(np.asarray(x_a), np.asarray(x_b))
    # Q and the blue prior receive context, not a red action.
    assert m.value_inputs(u, c1).shape[-1] == 2 + CTX
    assert m.policy_inputs(x, c1).shape[-1] == m.latent_dim + CTX


def test_conditioned_mode_routes_context_into_physics_and_implicit_ignores_it():
    cond = _agent("conditioned").model
    x = jnp.zeros((2, cond.latent_dim))
    u = jnp.zeros((2, 2))
    c1, c2 = jnp.zeros((2, CTX)), jnp.ones((2, CTX))
    assert not np.allclose(
        np.asarray(cond.next(x, cond.transition_inputs(u, c1), cond.dynamics_model.params)),
        np.asarray(cond.next(x, cond.transition_inputs(u, c2), cond.dynamics_model.params)),
    )
    imp = _agent("implicit").model
    np.testing.assert_array_equal(np.asarray(imp.transition_inputs(u, c1, u)), np.asarray(u))
    np.testing.assert_array_equal(np.asarray(imp.value_inputs(u, c2)), np.asarray(u))
    # Implicit mode update ignores context/red actions entirely.
    agent = _agent("implicit")
    b = _batch(agent, jax.random.PRNGKey(7))
    _, i1 = agent.update(**b, key=jax.random.PRNGKey(8))
    b2 = dict(b, context=b["context"] + 5.0, red_actions=-b["red_actions"])
    _, i2 = agent.update(**b2, key=jax.random.PRNGKey(8))
    assert float(i1["total_loss"]) == float(i2["total_loss"])


def test_planner_uses_context_causally_and_never_chooses_red_actions():
    agent = _agent("factored")
    # Reward / Q heads are zero-initialized upstream, so take one gradient step
    # to make imagined returns depend on the transition inputs at all.
    agent, _ = agent.update(**_batch(agent, jax.random.PRNGKey(9)), key=jax.random.PRNGKey(10))
    m = agent.model
    obs = jax.random.normal(jax.random.PRNGKey(0), (OBS_DIM,))
    key = jax.random.PRNGKey(1)
    a1, _ = agent.act(obs, context=jnp.zeros((CTX,)), key=key)
    a2, _ = agent.act(obs, context=jnp.ones((CTX,)) * 3.0, key=key)
    assert (np.abs(np.asarray(a1)) <= 1).all() and (np.abs(np.asarray(a2)) <= 1).all()
    assert a1.shape == (2,)  # blue action only; no red component is returned
    # estimate_value depends on context only through red(x, c) / Q / pi, and the
    # red action inside the rollout equals red_action at the current latent.
    x = m.encode(obs, m.encoder.params, key)
    x_pop = x[None].repeat(4, 0)
    acts = jax.random.uniform(jax.random.PRNGKey(2), (4, agent.horizon, 2), minval=-1, maxval=1)
    g1 = agent.estimate_value(x_pop, acts, agent.horizon, key, context=jnp.zeros((4, CTX)))
    g2 = agent.estimate_value(x_pop, acts, agent.horizon, key, context=jnp.ones((4, CTX)))
    assert g1.shape == (4,) and np.isfinite(np.asarray(g1)).all()
    assert not np.allclose(np.asarray(g1), np.asarray(g2))


def test_termination_masks_losses_and_truncation_keeps_bootstrap():
    agent = _agent("implicit")
    # Warm up so the (zero-initialized) target Q head is non-trivial and the
    # bootstrap term can actually differ between termination and truncation.
    for i in range(3):
        agent, _ = agent.update(**_batch(agent, jax.random.PRNGKey(100 + i)), key=jax.random.PRNGKey(200 + i))
    # Make the target ensemble clearly non-zero (EMA with tau=0.01 barely moved).
    scaled = jax.tree_util.tree_map(lambda p: p * 50.0, agent.model.value_model.params)
    agent = agent.replace(
        model=agent.model.replace(
            target_value_model=agent.model.target_value_model.replace(params=scaled)
        )
    )
    h, b = agent.horizon, agent.batch_size
    base = _batch(agent, jax.random.PRNGKey(11))
    base["terminated"] = jnp.zeros((h, b), bool)
    base["truncated"] = jnp.zeros((h, b), bool)
    # Steps after a capture at t=0 for episode 0 are masked: perturbing them
    # must not change any loss.
    cap = dict(base, terminated=base["terminated"].at[0, 0].set(True))
    _, i_cap = agent.update(**cap, key=jax.random.PRNGKey(12))
    pert = dict(cap)
    pert["rewards"] = cap["rewards"].at[1:, 0].add(100.0)
    pert["actions"] = cap["actions"].at[1:, 0].set(0.0)
    pert["next_observations"] = cap["next_observations"].at[1:, 0].add(50.0)
    _, i_pert = agent.update(**pert, key=jax.random.PRNGKey(12))
    for k in ("consistency_loss", "reward_loss", "value_loss", "continue_loss"):
        assert float(i_cap[k]) == pytest.approx(float(i_pert[k]), rel=1e-6), k
    # Same episode ending by truncation instead: the TD target still bootstraps
    # (differs from the terminated case) while the masking is identical.
    trunc = dict(base, truncated=base["truncated"].at[0, 0].set(True))
    _, i_trunc = agent.update(**trunc, key=jax.random.PRNGKey(12))
    assert float(i_trunc["consistency_loss"]) == pytest.approx(float(i_cap["consistency_loss"]), rel=1e-6)
    # Deterministic computation: any bootstrap contribution shows up exactly.
    assert float(i_trunc["value_loss"]) != float(i_cap["value_loss"])


def test_identity_encoder_is_the_normalized_state_and_trains():
    agent = _agent("conditioned", encoder="identity")
    m = agent.model
    assert m.encoder_type == "identity" and m.latent_dim == OBS_DIM
    obs = jax.random.normal(jax.random.PRNGKey(0), (3, OBS_DIM))
    np.testing.assert_allclose(np.asarray(m.encode(obs, m.encoder.params, jax.random.PRNGKey(1))), np.asarray(obs), atol=1e-6)
    x_next = m.next(obs, m.transition_inputs(jnp.zeros((3, 2)), jnp.zeros((3, CTX))), m.dynamics_model.params)
    assert x_next.shape == (3, OBS_DIM)  # no SimNorm simplex constraint
    b = _batch(agent, jax.random.PRNGKey(5))
    new, info = agent.update(**b, key=jax.random.PRNGKey(6))
    assert np.isfinite(float(info["total_loss"]))
    assert jax.tree_util.tree_leaves(new.model.encoder.params) == []
    # Identity mode requires normalization statistics.
    cfg = _config("implicit", encoder="identity")
    with pytest.raises(ValueError, match="obs_mean"):
        create_agent(cfg, OBS_DIM, key=jax.random.PRNGKey(0))


def _fake_dataset(n=6, horizon=12, state_dim=OBS_DIM, seed=0):
    rng = np.random.default_rng(seed)
    vl = rng.integers(3, horizon + 1, size=n)
    term = np.zeros((n, horizon), bool)
    trunc = np.zeros((n, horizon), bool)
    for i, L in enumerate(vl):
        if i % 2 == 0:
            term[i, L - 1] = True
        else:
            trunc[i, L - 1] = True
    data = dict(
        state=rng.normal(size=(n, horizon + 1, state_dim)).astype(np.float32),
        blue_action=rng.uniform(-1, 1, size=(n, horizon, 2)).astype(np.float32),
        red_action=rng.uniform(-1, 1, size=(n, horizon, 2)).astype(np.float32),
        blue_reward=rng.normal(size=(n, horizon)).astype(np.float32),
        terminated_capture=term,
        truncated_timeout=trunc,
        valid_mask=np.arange(horizon)[None, :] < vl[:, None],
        valid_length=vl.astype(np.int32),
        objective_label=rng.integers(0, 3, size=n).astype(np.int32),
    )
    return data


def test_sequence_replay_never_crosses_episodes_and_flags_padding():
    data = _fake_dataset()
    ctx = attach_context(data, None, source="oracle")
    replay = SequenceReplay.from_dataset(data, np.arange(6), horizon=4, context=ctx)
    rng = np.random.default_rng(0)
    batch = replay.sample(rng, 64)
    assert batch["observations"].shape == (4, 64, OBS_DIM)
    assert batch["context"].shape == (4, 64, 3)
    term = np.asarray(batch["terminated"])
    trunc = np.asarray(batch["truncated"])
    done = term | trunc
    # After a done flag inside a window every later step is padding (repeated
    # last valid step) and carries no further flag.
    for j in range(64):
        idx = np.flatnonzero(done[:, j])
        if len(idx):
            first = idx[0]
            assert not done[first + 1 :, j].any()
            for k in range(first + 1, 4):
                np.testing.assert_array_equal(
                    np.asarray(batch["observations"])[k, j], np.asarray(batch["observations"])[first, j]
                )
    assert replay.n_transitions == int(data["valid_length"].sum())
    with pytest.raises(ValueError):
        attach_context(data, None, source="causal")
    zero = attach_context(data, None, source="zero")
    assert zero.shape == (6, 13, 3) and not zero.any()


def test_model_error_and_calibration_utilities_run():
    data = _fake_dataset()
    ctx = attach_context(data, None, source="zero")
    replay = SequenceReplay.from_dataset(data, np.arange(6), horizon=3, context=ctx)
    mean, std = state_statistics(data["state"], data["valid_mask"])
    cfg = _config("factored", encoder="identity")
    agent = create_agent(cfg, OBS_DIM, key=jax.random.PRNGKey(0), obs_mean=mean, obs_std=std)
    err = multistep_model_error(agent, replay, horizons=(1, 2), max_starts=50, obs_std=std)
    for k in ("1", "2"):
        row = err["per_horizon"][k]
        assert row["n_starts"] > 0 and np.isfinite(row["model_mse"]) and np.isfinite(row["persistence_mse"])
        assert "position_rmse_model" in row
    rc = reward_calibration(agent, replay, max_samples=40)
    assert rc["n_samples"] == 40 and np.isfinite(rc["mse"]) and rc["calibration"]["bins"]
    tc = termination_calibration(agent, replay, max_samples=40)
    assert tc is not None and 0.0 <= tc["brier"] <= 1.0 and tc["n_capture_transitions"] >= 1
