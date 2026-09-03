"""Gate 0 parity tests for the pinned TD-MPC2-JAX source port.

Covers ``src/mopa/tdmpc.py`` and ``src/mopa/mppi.py`` (ported from
ShaneFlandermeyer/tdmpc2-jax @ 5b05ff452424896d709848e1f249bd67e269b8a1) with
``action_dim=2``, ``opponent_mode=implicit``, ``context_dim=0``.

These are fixed-seed shape/finiteness/bounds/repeatability checks plus direct
numerical checks of the two compatibility substitutions (SimNorm reshape and the
Distrax diagonal Gaussian). The continuous-control smoke at the end is a
component-integration test only; it is not evidence of control performance.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")
pytest.importorskip("optax")
pytest.importorskip("distrax")

import distrax  # noqa: E402
import flax.serialization  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from mopa import mppi  # noqa: E402
from mopa.tdmpc import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    create_agent,
    load_config,
    simnorm,
    symexp,
    symlog,
    two_hot,
    two_hot_inv,
    validate_gate0_config,
)

OBS_DIM = 4
ACTION_DIM = 2
REPO_ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_IMPORTS = (
    "hydra",
    "tensorflow",
    "tensorflow_probability",
    "orbax",
    "dm_control",
    "einops",
    "jaxtyping",
    "tdmpc2_jax",
)


# --------------------------------------------------------------------------- #
# Fixtures and helpers
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def smoke_config():
    return load_config(profile="smoke")


@pytest.fixture(scope="module")
def agent(smoke_config):
    return create_agent(smoke_config, OBS_DIM, key=jax.random.PRNGKey(0))


def _trainable_params(agent):
    m = agent.model
    trees = {
        "encoder": m.encoder.params,
        "dynamics": m.dynamics_model.params,
        "reward": m.reward_model.params,
        "policy": m.policy_model.params,
        "value": m.value_model.params,
    }
    if m.predict_continues:
        trees["continue"] = m.continue_model.params
    return trees


def _leaves(tree):
    return jax.tree_util.tree_leaves(tree)


def _all_finite(tree) -> bool:
    return all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in _leaves(tree))


def _synthetic_batch(agent, key):
    """Random (horizon, batch) sequences shaped like the upstream replay sample."""
    h, b = agent.horizon, agent.batch_size
    k_obs, k_next, k_act, k_rew = jax.random.split(key, 4)
    observations = jax.random.normal(k_obs, (h, b, OBS_DIM), dtype=jnp.float32)
    next_observations = jax.random.normal(k_next, (h, b, OBS_DIM), dtype=jnp.float32)
    actions = jax.random.uniform(k_act, (h, b, ACTION_DIM), minval=-1.0, maxval=1.0)
    rewards = jax.random.normal(k_rew, (h, b), dtype=jnp.float32)
    terminated = jnp.zeros((h, b), dtype=bool)
    truncated = jnp.zeros((h, b), dtype=bool).at[-1, 0].set(True)
    return dict(
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        terminated=terminated,
        truncated=truncated,
    )


# --------------------------------------------------------------------------- #
# Provenance, layout, and configuration
# --------------------------------------------------------------------------- #
def test_provenance_files_exist_and_pin_commit():
    upstream = REPO_ROOT / "third_party" / "tdmpc2-jax" / "UPSTREAM.md"
    license_file = REPO_ROOT / "third_party" / "tdmpc2-jax" / "LICENSE"
    assert upstream.is_file()
    assert license_file.is_file()
    text = upstream.read_text()
    assert "5b05ff452424896d709848e1f249bd67e269b8a1" in text
    assert "ShaneFlandermeyer/tdmpc2-jax" in text
    assert license_file.read_text().startswith("MIT License")
    # No runtime code under third_party/.
    assert not list((REPO_ROOT / "third_party" / "tdmpc2-jax").glob("*.py"))


def test_ported_files_have_no_forbidden_imports():
    pattern = re.compile(r"^\s*(?:from|import)\s+([\w\.]+)", re.MULTILINE)
    for rel in ("src/mopa/tdmpc.py", "src/mopa/mppi.py"):
        modules = {m.split(".")[0] for m in pattern.findall((REPO_ROOT / rel).read_text())}
        assert not (modules & set(FORBIDDEN_IMPORTS)), (rel, modules)


def test_reference_config_retains_upstream_defaults():
    cfg = load_config()
    assert cfg["action_dim"] == ACTION_DIM
    assert cfg["opponent_mode"] == "implicit"
    assert cfg["context_dim"] == 0
    td = cfg["tdmpc2"]
    assert td["horizon"] == 3
    assert td["population_size"] == 512
    assert td["policy_prior_samples"] == 24
    assert td["num_elites"] == 64
    assert td["mppi_iterations"] == 6
    assert td["discount"] == 0.99
    assert cfg["world_model"]["num_value_nets"] == 5
    assert "profiles" not in cfg
    assert DEFAULT_CONFIG_PATH.name == "tdmpc2.yaml"


def test_smoke_profile_only_overrides_named_keys(smoke_config):
    ref = load_config()
    # Reference values that the smoke profile must not weaken.
    for k in ("horizon", "discount", "rho", "tau", "entropy_coef"):
        assert smoke_config["tdmpc2"][k] == ref["tdmpc2"][k]
    assert smoke_config["world_model"]["num_value_nets"] == 5
    assert smoke_config["action_dim"] == ACTION_DIM
    assert smoke_config["opponent_mode"] == "implicit"
    assert smoke_config["context_dim"] == 0
    with pytest.raises(KeyError):
        load_config(profile="does-not-exist")


def test_gate0_validator_pins_the_single_agent_baseline(smoke_config):
    """The Gate 0 baseline is implicit mode with no context; other settings are
    later-gate configurations and must not silently pass as the baseline."""
    validate_gate0_config(smoke_config)
    with pytest.raises(NotImplementedError, match="Gate 0"):
        validate_gate0_config(dict(smoke_config, opponent_mode="conditioned", context_dim=3))
    with pytest.raises(NotImplementedError, match="Gate 0"):
        validate_gate0_config(dict(smoke_config, context_dim=8))
    with pytest.raises(NotImplementedError, match="context_dim >= 1"):
        create_agent(dict(smoke_config, opponent_mode="factored"), OBS_DIM, key=jax.random.PRNGKey(0))
    with pytest.raises(NotImplementedError, match="opponent_mode"):
        create_agent(dict(smoke_config, opponent_mode="bogus"), OBS_DIM, key=jax.random.PRNGKey(0))


# --------------------------------------------------------------------------- #
# Compatibility substitutions: direct numerical checks
# --------------------------------------------------------------------------- #
def test_simnorm_matches_reshape_softmax_reshape():
    # einops '...(L V) -> ... L V' with V given is a row-major split, i.e. the
    # flat index is l*V + v. Reproduce that explicitly in NumPy.
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 3, 16)).astype(np.float32)
    V = 8
    grouped = x.reshape(2, 3, 16 // V, V)
    e = np.exp(grouped - grouped.max(axis=-1, keepdims=True))
    expected = (e / e.sum(axis=-1, keepdims=True)).reshape(2, 3, 16)

    out = np.asarray(simnorm(jnp.asarray(x), simplex_dim=V))
    assert out.shape == x.shape
    np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)
    # Each group of V consecutive entries is a simplex.
    np.testing.assert_allclose(out.reshape(2, 3, 2, V).sum(-1), 1.0, atol=1e-6)
    # Grouping check: perturbing one entry only changes its own group.
    x2 = x.copy()
    x2[0, 0, 3] += 5.0
    out2 = np.asarray(simnorm(jnp.asarray(x2), simplex_dim=V))
    assert not np.allclose(out[0, 0, :V], out2[0, 0, :V])
    np.testing.assert_array_equal(out[0, 0, V:], out2[0, 0, V:])
    np.testing.assert_array_equal(out[1:], out2[1:])


def test_two_hot_symlog_roundtrip():
    x = jnp.asarray([-50.0, -3.2, -0.5, 0.0, 0.7, 4.0, 120.0])
    low, high, num_bins = -10.0, 10.0, 101
    enc = two_hot(x, low, high, num_bins)
    assert enc.shape == (7, num_bins)
    np.testing.assert_allclose(np.asarray(enc.sum(-1)), 1.0, atol=1e-6)
    dec = two_hot_inv(enc, low, high, num_bins, apply_softmax=False)
    np.testing.assert_allclose(np.asarray(dec), np.asarray(x), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np.asarray(symexp(symlog(x))), np.asarray(x), rtol=1e-5)


def test_policy_log_prob_matches_explicit_diagonal_normal_with_tanh_correction(agent):
    """Distrax substitution: log N(u; mu, diag(sigma)) - sum 2(log2 - u - softplus(-2u)).

    The mean/log_std transform and the tanh Jacobian term are unchanged
    upstream expressions; the check pins the density that Distrax supplies.
    Tolerance 1e-5 (float32).
    """
    model = agent.model
    key = jax.random.PRNGKey(3)
    x = model.encode(
        jax.random.normal(key, (6, OBS_DIM)), model.encoder.params, jax.random.PRNGKey(4)
    )
    raw = model.policy_model.apply_fn({"params": model.policy_model.params}, x)
    mean_raw, log_std_raw = jnp.split(raw, 2, axis=-1)
    log_std = -10 + 0.5 * (2 - (-10)) * (jnp.tanh(log_std_raw) + 1)

    def explicit_log_prob(u):
        gauss = -0.5 * jnp.sum(
            ((u - mean_raw) / jnp.exp(log_std)) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi),
            axis=-1,
        )
        jac = jnp.sum(2 * (jnp.log(2) - u - jax.nn.softplus(-2 * u)), axis=-1)
        return gauss - jac

    # Deterministic path: u = mean.
    act_det, mean_out, log_std_out, lp_det = model.sample_actions(
        x, model.policy_model.params, deterministic=True, key=key
    )
    np.testing.assert_allclose(np.asarray(act_det), np.tanh(np.asarray(mean_raw)), atol=1e-6)
    np.testing.assert_allclose(np.asarray(mean_out), np.tanh(np.asarray(mean_raw)), atol=1e-6)
    np.testing.assert_allclose(np.asarray(log_std_out), np.asarray(log_std), atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(lp_det), np.asarray(explicit_log_prob(mean_raw)), atol=1e-5
    )

    # Stochastic path: reproduce the pre-squash sample with the same key.
    sample_key = jax.random.PRNGKey(11)
    u = distrax.MultivariateNormalDiag(loc=mean_raw, scale_diag=jnp.exp(log_std)).sample(
        seed=sample_key
    )
    act_s, _, _, lp_s = model.sample_actions(
        x, model.policy_model.params, deterministic=False, key=sample_key
    )
    np.testing.assert_allclose(np.asarray(act_s), np.tanh(np.asarray(u)), atol=1e-6)
    np.testing.assert_allclose(np.asarray(lp_s), np.asarray(explicit_log_prob(u)), atol=1e-5)
    # Reparameterized: sample == loc + scale * eps with eps drawn from the key.
    eps = (u - mean_raw) / jnp.exp(log_std)
    assert np.isfinite(np.asarray(eps)).all()
    assert not np.allclose(np.asarray(eps), 0.0)


# --------------------------------------------------------------------------- #
# World model components
# --------------------------------------------------------------------------- #
def test_encode_next_reward_q_shapes_and_finiteness(agent):
    model = agent.model
    B = 5
    latent_dim, simnorm_dim = model.latent_dim, model.simnorm_dim
    obs = jax.random.normal(jax.random.PRNGKey(1), (B, OBS_DIM))
    a = jax.random.uniform(jax.random.PRNGKey(2), (B, ACTION_DIM), minval=-1, maxval=1)

    x = model.encode(obs, model.encoder.params, jax.random.PRNGKey(0))
    assert x.shape == (B, latent_dim)
    assert np.isfinite(np.asarray(x)).all()
    np.testing.assert_allclose(
        np.asarray(x).reshape(B, latent_dim // simnorm_dim, simnorm_dim).sum(-1),
        1.0,
        atol=1e-5,
    )

    # Equation 1: the dynamics head consumes exactly [x, u].
    x_next = model.next(x, a, model.dynamics_model.params)
    assert x_next.shape == (B, latent_dim)
    assert np.isfinite(np.asarray(x_next)).all()
    dyn_in = model.dynamics_model.params["layers_0"]["Dense_0"]["kernel"].shape[0]
    assert dyn_in == latent_dim + ACTION_DIM

    r, r_logits = model.reward(x, a, model.reward_model.params)
    assert r.shape == (B,)
    assert r_logits.shape == (B, model.num_bins)
    assert np.isfinite(np.asarray(r)).all()

    q, q_logits = model.Q(x, a, model.value_model.params, jax.random.PRNGKey(5))
    assert q.shape == (model.num_value_nets, B)
    assert q_logits.shape == (model.num_value_nets, B, model.num_bins)
    assert model.num_value_nets == 5
    assert np.isfinite(np.asarray(q)).all()
    # Target ensemble starts as an exact copy of the online ensemble.
    for online, target in zip(
        _leaves(model.value_model.params), _leaves(model.target_value_model.params)
    ):
        np.testing.assert_array_equal(np.asarray(online), np.asarray(target))


def test_components_are_repeatable_under_fixed_seeds(agent):
    model = agent.model
    obs = jax.random.normal(jax.random.PRNGKey(7), (3, OBS_DIM))
    x1 = model.encode(obs, model.encoder.params, jax.random.PRNGKey(0))
    x2 = model.encode(obs, model.encoder.params, jax.random.PRNGKey(0))
    np.testing.assert_array_equal(np.asarray(x1), np.asarray(x2))
    a1 = model.sample_actions(x1, model.policy_model.params, key=jax.random.PRNGKey(9))
    a2 = model.sample_actions(x1, model.policy_model.params, key=jax.random.PRNGKey(9))
    for u, v in zip(a1, a2):
        np.testing.assert_array_equal(np.asarray(u), np.asarray(v))
    a3 = model.sample_actions(x1, model.policy_model.params, key=jax.random.PRNGKey(10))
    assert not np.allclose(np.asarray(a1[0]), np.asarray(a3[0]))


def test_policy_mean_and_samples_are_bounded(agent):
    model = agent.model
    x = model.encode(
        jax.random.normal(jax.random.PRNGKey(12), (64, OBS_DIM)) * 5.0,
        model.encoder.params,
        jax.random.PRNGKey(0),
    )
    action, mean, log_std, log_prob = model.sample_actions(
        x, model.policy_model.params, key=jax.random.PRNGKey(13)
    )
    assert action.shape == mean.shape == (64, ACTION_DIM)
    assert log_std.shape == (64, ACTION_DIM)
    assert log_prob.shape == (64,)
    for arr in (action, mean):
        arr = np.asarray(arr)
        assert np.isfinite(arr).all()
        assert (arr >= -1.0).all() and (arr <= 1.0).all()
    assert np.isfinite(np.asarray(log_prob)).all()
    ls = np.asarray(log_std)
    assert (ls >= -10.0).all() and (ls <= 2.0).all()


# --------------------------------------------------------------------------- #
# Planner
# --------------------------------------------------------------------------- #
def test_plan_shapes_bounds_and_repeatability(agent):
    model = agent.model
    x = model.encode(
        jax.random.normal(jax.random.PRNGKey(20), (OBS_DIM,)),
        model.encoder.params,
        jax.random.PRNGKey(0),
    )
    key = jax.random.PRNGKey(21)
    action, (mean, std) = agent.plan(x, horizon=agent.horizon, key=key)
    assert action.shape == (ACTION_DIM,)
    assert mean.shape == std.shape == (agent.horizon, ACTION_DIM)
    for arr in (action, mean, std):
        assert np.isfinite(np.asarray(arr)).all()
    assert (np.abs(np.asarray(action)) <= 1.0).all()
    assert (np.asarray(std) >= agent.min_plan_std - 1e-6).all()
    assert (np.asarray(std) <= agent.max_plan_std + 1e-6).all()

    action_again, _ = agent.plan(x, horizon=agent.horizon, key=key)
    np.testing.assert_array_equal(np.asarray(action), np.asarray(action_again))
    # Module function and method are the same computation.
    action_fn, _ = mppi.plan(
        agent, x=x, horizon=agent.horizon, context=jnp.zeros((0,)), key=key
    )
    np.testing.assert_array_equal(np.asarray(action), np.asarray(action_fn))

    # Warm start, deterministic, and train-noise variants all stay bounded.
    for kwargs in (
        dict(prev_plan=(mean, std)),
        dict(deterministic=True),
        dict(train=True),
        dict(prev_plan=(mean, std), deterministic=True, train=True),
    ):
        a, (m, s) = agent.plan(x, horizon=agent.horizon, key=key, **kwargs)
        assert a.shape == (ACTION_DIM,)
        assert np.isfinite(np.asarray(a)).all()
        assert (np.abs(np.asarray(a)) <= 1.0).all()

    # Batched planning.
    xb = model.encode(
        jax.random.normal(jax.random.PRNGKey(22), (3, OBS_DIM)),
        model.encoder.params,
        jax.random.PRNGKey(0),
    )
    ab, (mb, sb) = agent.plan(xb, horizon=agent.horizon, key=key)
    assert ab.shape == (3, ACTION_DIM)
    assert mb.shape == sb.shape == (3, agent.horizon, ACTION_DIM)
    assert (np.abs(np.asarray(ab)) <= 1.0).all()


def test_estimate_value_shape_and_finiteness(agent):
    model = agent.model
    x = model.encode(
        jax.random.normal(jax.random.PRNGKey(30), (OBS_DIM,)),
        model.encoder.params,
        jax.random.PRNGKey(0),
    )
    n = agent.population_size
    x_t = x[None, :].repeat(n, axis=0)
    actions = jax.random.uniform(
        jax.random.PRNGKey(31), (n, agent.horizon, ACTION_DIM), minval=-1, maxval=1
    )
    values = agent.estimate_value(x_t, actions, agent.horizon, jax.random.PRNGKey(32))
    assert values.shape == (n,)
    assert np.isfinite(np.asarray(values)).all()


def test_act_with_planner_and_with_policy_prior(agent):
    obs = jax.random.normal(jax.random.PRNGKey(40), (OBS_DIM,))
    key = jax.random.PRNGKey(41)
    a_mpc, plan = agent.act(obs, prev_plan=None, mpc=True, key=key)
    a_pi, no_plan = agent.act(obs, mpc=False, key=key)
    assert a_mpc.shape == a_pi.shape == (ACTION_DIM,)
    assert plan is not None and no_plan is None
    for a in (a_mpc, a_pi):
        assert np.isfinite(np.asarray(a)).all()
        assert (np.abs(np.asarray(a)) <= 1.0).all()


# --------------------------------------------------------------------------- #
# Update and checkpointing
# --------------------------------------------------------------------------- #
def test_update_changes_parameters_and_keeps_losses_finite(agent):
    batch = _synthetic_batch(agent, jax.random.PRNGKey(50))
    before = _trainable_params(agent)
    new_agent, info = agent.update(**batch, key=jax.random.PRNGKey(51))

    for k in (
        "consistency_loss",
        "reward_loss",
        "value_loss",
        "continue_loss",
        "total_loss",
        "policy_loss",
    ):
        assert np.isfinite(np.asarray(info[k])).all(), k
    assert np.isfinite(np.asarray(info["policy_log_std"])).all()
    assert np.isfinite(np.asarray(info["value_scale"])).all()
    assert float(info["continue_loss"]) == 0.0  # predict_continues=False upstream

    after = _trainable_params(new_agent)
    changed = 0
    for name in before:
        for b, a in zip(_leaves(before[name]), _leaves(after[name])):
            assert b.shape == a.shape
            assert bool(jnp.all(jnp.isfinite(a)))
            changed += int(not bool(jnp.array_equal(b, a)))
    assert changed >= 1
    assert _all_finite(after)

    # Target EMA moved toward the new online ensemble but is not equal to it.
    t_before = _leaves(agent.model.target_value_model.params)
    t_after = _leaves(new_agent.model.target_value_model.params)
    v_after = _leaves(new_agent.model.value_model.params)
    assert any(not bool(jnp.array_equal(x, y)) for x, y in zip(t_before, t_after))
    assert any(not bool(jnp.array_equal(x, y)) for x, y in zip(t_after, v_after))
    assert _all_finite(new_agent.model.target_value_model.params)
    assert int(new_agent.model.dynamics_model.step) == int(agent.model.dynamics_model.step) + 1

    # Fixed-seed repeatability of a full update.
    again, info_again = agent.update(**batch, key=jax.random.PRNGKey(51))
    assert float(info["total_loss"]) == float(info_again["total_loss"])
    for a, b in zip(_leaves(after), _leaves(_trainable_params(again))):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_checkpoint_roundtrip_reproduces_deterministic_actions(agent, smoke_config):
    batch = _synthetic_batch(agent, jax.random.PRNGKey(60))
    trained, _ = agent.update(**batch, key=jax.random.PRNGKey(61))

    blob = flax.serialization.to_bytes(trained)
    # Template built from a different seed so restoration is what matters.
    template = create_agent(smoke_config, OBS_DIM, key=jax.random.PRNGKey(999))
    restored = flax.serialization.from_bytes(template, blob)

    for a, b in zip(_leaves(_trainable_params(trained)), _leaves(_trainable_params(restored))):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    np.testing.assert_array_equal(
        np.asarray(trained.value_scale), np.asarray(restored.value_scale)
    )

    obs = jax.random.normal(jax.random.PRNGKey(62), (OBS_DIM,))
    key = jax.random.PRNGKey(63)
    for mpc in (True, False):
        a_ref, _ = trained.act(obs, mpc=mpc, deterministic=True, key=key)
        a_res, _ = restored.act(obs, mpc=mpc, deterministic=True, key=key)
        a_tmp, _ = template.act(obs, mpc=mpc, deterministic=True, key=key)
        np.testing.assert_array_equal(np.asarray(a_ref), np.asarray(a_res))
        assert not np.allclose(np.asarray(a_ref), np.asarray(a_tmp))


def test_golden_fixed_seed_values_from_verified_port(agent):
    """Fixed-seed values recorded from the port that was compared bitwise against
    upstream 5b05ff4 (see third_party/tdmpc2-jax/UPSTREAM.md). Any refactor of
    the implicit single-agent path must reproduce them."""
    batch = _synthetic_batch(agent, jax.random.PRNGKey(50))
    new_agent, info = agent.update(**batch, key=jax.random.PRNGKey(51))
    assert float(info["total_loss"]) == pytest.approx(1.4424139261245728, abs=2e-5)
    assert float(info["consistency_loss"]) == pytest.approx(0.02596951089799404, abs=2e-6)
    assert float(info["reward_loss"]) == pytest.approx(4.615118980407715, abs=2e-5)
    assert float(info["value_loss"]) == pytest.approx(4.61511754989624, abs=2e-5)
    assert float(info["policy_loss"]) == pytest.approx(0.00045023064012639225, abs=2e-6)
    x = agent.model.encode(
        jax.random.normal(jax.random.PRNGKey(20), (OBS_DIM,)),
        agent.model.encoder.params,
        jax.random.PRNGKey(0),
    )
    a, _ = agent.plan(x, horizon=3, key=jax.random.PRNGKey(21))
    np.testing.assert_allclose(
        np.asarray(a), [0.03614749386906624, -0.05706670507788658], atol=2e-5
    )
    a2, _ = new_agent.act(
        jax.random.normal(jax.random.PRNGKey(62), (OBS_DIM,)),
        mpc=True,
        deterministic=True,
        key=jax.random.PRNGKey(63),
    )
    np.testing.assert_allclose(
        np.asarray(a2), [0.06102251634001732, -0.019713396206498146], atol=2e-5
    )


# --------------------------------------------------------------------------- #
# Reference configuration instantiates in the locked environment
# --------------------------------------------------------------------------- #
def test_reference_config_instantiates_and_runs_core_calls():
    ref = create_agent(load_config(), OBS_DIM, key=jax.random.PRNGKey(0))
    assert ref.horizon == 3
    assert ref.population_size == 512
    assert ref.policy_prior_samples == 24
    assert ref.num_elites == 64
    assert ref.mppi_iterations == 6
    assert ref.model.num_value_nets == 5
    assert ref.discount == 0.99
    assert ref.model.action_dim == ACTION_DIM
    assert ref.model.latent_dim == 512

    obs = jax.random.normal(jax.random.PRNGKey(1), (2, OBS_DIM))
    x = ref.model.encode(obs, ref.model.encoder.params, jax.random.PRNGKey(2))
    assert x.shape == (2, 512)
    a = jax.random.uniform(jax.random.PRNGKey(3), (2, ACTION_DIM), minval=-1, maxval=1)
    assert ref.model.next(x, a, ref.model.dynamics_model.params).shape == (2, 512)
    assert ref.model.reward(x, a, ref.model.reward_model.params)[0].shape == (2,)
    q, _ = ref.model.Q(x, a, ref.model.value_model.params, jax.random.PRNGKey(4))
    assert q.shape == (5, 2)
    act, mean, _, lp = ref.model.sample_actions(
        x, ref.model.policy_model.params, key=jax.random.PRNGKey(5)
    )
    assert act.shape == (2, ACTION_DIM)
    assert (np.abs(np.asarray(act)) <= 1.0).all()
    assert np.isfinite(np.asarray(lp)).all()


# --------------------------------------------------------------------------- #
# Gate 0 continuous-control smoke (component integration only)
# --------------------------------------------------------------------------- #
def _linear_system():
    theta = 0.3
    rot = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=np.float32,
    )
    A = np.zeros((4, 4), dtype=np.float32)
    A[:2, :2] = 0.9 * rot
    A[2:, 2:] = 0.8 * rot
    B = np.array([[0.5, 0.0], [0.0, 0.5], [0.2, -0.1], [0.1, 0.2]], dtype=np.float32)
    return A, B


def _step(state, action, A, B):
    state_next = np.clip(A @ state + B @ action, -1.0, 1.0).astype(np.float32)
    reward = -float(np.sum(state_next**2)) - 0.01 * float(np.sum(action**2))
    return state_next, np.float32(reward)


def test_gate0_continuous_control_smoke(agent):
    """Encode -> bounded action (planner) -> step dummy system -> record -> update.

    A synthetic single-agent linear system with clipped states; this asserts the
    pieces fit together and stay finite. It says nothing about control quality.
    """
    A, B = _linear_system()
    horizon, batch_size = agent.horizon, agent.batch_size
    n_steps = batch_size + horizon - 1  # enough sliding windows for one batch

    rng = np.random.default_rng(0)
    state = rng.uniform(-0.5, 0.5, size=4).astype(np.float32)
    key = jax.random.PRNGKey(70)

    states = [state]
    actions, rewards = [], []
    plan = None
    for _ in range(n_steps):
        key, enc_key, act_key = jax.random.split(key, 3)
        x = agent.model.encode(jnp.asarray(state), agent.model.encoder.params, enc_key)
        assert x.shape == (agent.model.latent_dim,)
        action, plan = agent.plan(x, horizon=horizon, prev_plan=plan, train=True, key=act_key)
        action = np.asarray(action, dtype=np.float32)
        assert action.shape == (ACTION_DIM,)
        assert np.isfinite(action).all()
        assert (action >= -1.0).all() and (action <= 1.0).all()

        state, reward = _step(state, action, A, B)
        assert np.isfinite(state).all() and np.isfinite(reward)
        states.append(state)
        actions.append(action)
        rewards.append(reward)

    states = np.stack(states)  # (n_steps + 1, 4)
    actions = np.stack(actions)  # (n_steps, 2)
    rewards = np.asarray(rewards, dtype=np.float32)  # (n_steps,)
    assert (np.abs(actions) <= 1.0).all()
    assert (np.abs(states) <= 1.0).all()

    idx = np.arange(batch_size)[:, None] + np.arange(horizon)[None, :]  # (B, H)
    batch = dict(
        observations=jnp.asarray(states[idx].transpose(1, 0, 2)),
        next_observations=jnp.asarray(states[idx + 1].transpose(1, 0, 2)),
        actions=jnp.asarray(actions[idx].transpose(1, 0, 2)),
        rewards=jnp.asarray(rewards[idx].T),
        terminated=jnp.zeros((horizon, batch_size), dtype=bool),
        truncated=jnp.zeros((horizon, batch_size), dtype=bool),
    )
    assert batch["observations"].shape == (horizon, batch_size, 4)
    assert batch["actions"].shape == (horizon, batch_size, ACTION_DIM)

    before = _trainable_params(agent)
    new_agent, info = agent.update(**batch, key=jax.random.PRNGKey(71))
    for k in ("consistency_loss", "reward_loss", "value_loss", "policy_loss", "total_loss"):
        assert np.isfinite(np.asarray(info[k])).all(), k
    after = _trainable_params(new_agent)
    assert _all_finite(after)
    assert any(
        not bool(jnp.array_equal(b, a))
        for b, a in zip(_leaves(before), _leaves(after))
    )
