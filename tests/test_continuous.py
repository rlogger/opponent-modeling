"""Tanh-squashed Gaussian utilities and the continuous specialist actor (Gate 1)."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")
pytest.importorskip("distrax")

import distrax  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from mopa.continuous import (  # noqa: E402
    LOG_STD_MAX,
    LOG_STD_MIN,
    atanh_clipped,
    gaussian_entropy,
    tanh_gaussian_log_prob,
    tanh_gaussian_sample,
    tanh_log_det_jacobian,
)
from mopa.nets import ContinuousActor  # noqa: E402


def test_tanh_gaussian_log_prob_matches_distrax_transformed_density():
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    mean = jax.random.normal(k1, (64, 2)) * 0.5
    log_std = jax.random.uniform(k2, (64, 2), minval=-2.0, maxval=0.5)
    u, a = tanh_gaussian_sample(k3, mean, log_std)
    assert a.shape == (64, 2)
    assert (np.abs(np.asarray(a)) < 1.0).all()

    ours = tanh_gaussian_log_prob(mean, log_std, u)
    base = distrax.MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
    ref = distrax.Transformed(base, distrax.Block(distrax.Tanh(), 1)).log_prob(a)
    np.testing.assert_allclose(np.asarray(ours), np.asarray(ref), rtol=1e-4, atol=1e-4)
    # The correction is exactly the summed log(1 - tanh(u)^2).
    np.testing.assert_allclose(
        np.asarray(tanh_log_det_jacobian(u)),
        np.asarray(jnp.sum(jnp.log(1.0 - jnp.tanh(u) ** 2 + 1e-12), axis=-1)),
        atol=1e-4,
    )
    # Joint density: per-dimension terms sum (never a per-dimension probability).
    one_dim = sum(
        np.asarray(tanh_gaussian_log_prob(mean[:, i : i + 1], log_std[:, i : i + 1], u[:, i : i + 1]))
        for i in range(2)
    )
    np.testing.assert_allclose(np.asarray(ours), one_dim, atol=1e-5)


def test_gaussian_entropy_and_inverse_helpers():
    log_std = jnp.asarray([[0.0, 0.0], [-1.0, 0.5]])
    ref = distrax.MultivariateNormalDiag(
        loc=jnp.zeros((2, 2)), scale_diag=jnp.exp(log_std)
    ).entropy()
    np.testing.assert_allclose(np.asarray(gaussian_entropy(log_std)), np.asarray(ref), atol=1e-5)
    a = jnp.asarray([[0.3, -0.9], [1.0, -1.0]])
    u = atanh_clipped(a)
    assert np.isfinite(np.asarray(u)).all()
    np.testing.assert_allclose(np.asarray(jnp.tanh(u))[0], np.asarray(a)[0], atol=1e-6)


def test_continuous_actor_shapes_bounds_and_log_std_clamp():
    actor = ContinuousActor(action_dim=2, hidden_dim=16)
    obs = jax.random.normal(jax.random.PRNGKey(1), (5, 7))
    params = actor.init(jax.random.PRNGKey(0), obs)
    mean, log_std = actor.apply(params, obs)
    assert mean.shape == log_std.shape == (5, 2)
    np.testing.assert_allclose(np.asarray(log_std), 0.0)  # zero-initialized
    det = actor.apply(params, obs, method=ContinuousActor.deterministic_action)
    np.testing.assert_allclose(np.asarray(det), np.tanh(np.asarray(mean)), atol=1e-6)
    assert (np.abs(np.asarray(det)) <= 1.0).all()

    extreme = jax.tree_util.tree_map(lambda x: x, params)
    extreme["params"]["log_std"] = jnp.asarray([10.0, -10.0])
    _, clamped = actor.apply(extreme, obs)
    np.testing.assert_allclose(np.asarray(clamped)[0], [LOG_STD_MAX, LOG_STD_MIN])
    # jit / vmap friendly.
    jit_mean, _ = jax.jit(actor.apply)(params, obs)
    np.testing.assert_allclose(np.asarray(jit_mean), np.asarray(mean), atol=1e-6)
    vm = jax.vmap(lambda o: actor.apply(params, o)[0])(obs)
    np.testing.assert_allclose(np.asarray(vm), np.asarray(mean), atol=1e-6)
