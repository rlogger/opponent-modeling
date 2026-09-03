"""Continuous opponent BC and the frozen causal context encoder (Gate 2)."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from mopa.bc_continuous import (  # noqa: E402
    ContinuousBCPolicy,
    continuous_bc_metrics,
    fit_continuous_bc,
)
from mopa.context import (  # noqa: E402
    CONTEXT_DIM,
    CausalContextEncoder,
    causal_sample_context,
    derangement,
    split_local_derangement,
    train_context_encoder,
)


def _linear_dataset(seed=0, n=512, d=6):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d)).astype(np.float32)
    w = rng.normal(size=(d, 2)).astype(np.float32)
    y = np.tanh(x @ w * 0.7).astype(np.float32)
    return x, y


def test_fit_continuous_bc_learns_bounded_regression_and_roundtrips(tmp_path):
    x, y = _linear_dataset()
    untrained = fit_continuous_bc(x, y, 0, steps=0)
    trained = fit_continuous_bc(x, y, 0, steps=400, metadata={"arm": "test"})
    before = continuous_bc_metrics(untrained.predict(x), y)["mse"]
    after = continuous_bc_metrics(trained.predict(x), y)["mse"]
    assert after < before * 0.25
    pred = trained.predict(x)
    assert pred.shape == (len(x), 2) and (np.abs(pred) <= 1.0).all()
    assert trained.metadata["arm"] == "test" and trained.metadata["loss"] == "mean_squared_error"

    # PyTree policy: jit / vmap on `act` and identical outputs after save/load.
    path = tmp_path / "policy.npz"
    trained.save(path)
    loaded = ContinuousBCPolicy.load(path)
    np.testing.assert_allclose(loaded.predict(x), pred, atol=1e-6)
    jitted = jax.jit(lambda pol, f: pol.act(f))(loaded, jnp.asarray(x[:4]))
    np.testing.assert_allclose(np.asarray(jitted), pred[:4], atol=1e-6)
    leaves = jax.tree_util.tree_leaves(loaded)
    assert len(leaves) >= 8  # params + mean + std are pytree leaves
    with pytest.raises(ValueError):
        trained.predict(x[:, :3])
    with pytest.raises(ValueError):
        fit_continuous_bc(x, np.full_like(y, 1.5), 0, steps=1)


def test_continuous_bc_metrics_hand_values():
    pred = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [-1.0, 0.0]])
    tgt = np.array([[1.0, 0.0], [0.0, -1.0], [0.5, 0.5], [0.0, 0.0]])
    m = continuous_bc_metrics(
        pred, tgt, episode_ids=np.array([0, 0, 1, 1]), strategy_labels=np.array([0, 0, 1, 1])
    )
    # squared errors: 0, 4, 0, 1 -> mse 1.25
    assert m["mse"] == pytest.approx(1.25)
    assert m["agreement"] == pytest.approx(0.5)
    # cosines over moving targets (rows 0,1,2): 1, -1, 1 -> mean 1/3
    assert m["direction_cosine"] == pytest.approx(1 / 3)
    assert m["episode_macro_mse"] == pytest.approx((2.0 + 0.5) / 2)
    assert m["n_strategies"] == 2 and m["per_strategy"]["0"]["n_episodes"] == 1
    assert m["target_saturation"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        continuous_bc_metrics(pred[:, :1], tgt[:, :1])


def test_causal_context_timing_and_derangements():
    latents = np.zeros((2, 5, 2), dtype=np.float32)
    latents[..., 0] = np.arange(5)[None, :] + 1
    ctx = causal_sample_context(latents, np.array([0, 1, 0]), np.array([0, 1, 4]))
    assert ctx.shape == (3, CONTEXT_DIM)
    np.testing.assert_array_equal(ctx[0], 0.0)  # c_0 is zero
    assert ctx[1, 0] == 1.0 and ctx[2, 0] == 4.0  # z[t-1]
    validation = np.array([False] * 6 + [True] * 6)
    checkpoints = np.repeat(np.arange(3), 4)
    perm = split_local_derangement(validation, seed=3, checkpoint_ids=checkpoints)
    assert np.all(perm != np.arange(12))
    np.testing.assert_array_equal(validation[perm], validation)
    np.testing.assert_array_equal(checkpoints[perm], checkpoints)
    with pytest.raises(ValueError):
        derangement(np.array([0]), np.random.default_rng(0))


def test_context_encoder_trains_encodes_causally_and_roundtrips(tmp_path):
    rng = np.random.default_rng(0)
    n, horizon = 12, 9
    prey = np.cumsum(rng.normal(scale=0.1, size=(n, horizon + 1, 2)), axis=1).astype(np.float32)
    pred = np.cumsum(rng.normal(scale=0.1, size=(n, horizon + 1, 1, 2)), axis=1).astype(np.float32)
    lengths = rng.integers(4, horizon + 1, size=n).astype(np.int32)
    enc = train_context_encoder(
        prey, pred, lengths, np.arange(8), seed=0, latent_dim=2, hidden_dim=8, steps=3
    )
    ctx = enc.causal_context(prey, pred, lengths)
    assert ctx.shape == (n, horizon + 1, CONTEXT_DIM)
    np.testing.assert_array_equal(ctx[:, 0], 0.0)
    np.testing.assert_array_equal(ctx[..., 2], 0.0)  # padded column
    # Causality: changing the future leaves c_t unchanged.
    prey2 = prey.copy()
    prey2[:, 5:] += 3.0
    ctx2 = enc.causal_context(prey2, pred, lengths)
    np.testing.assert_allclose(ctx[:, :5], ctx2[:, :5], atol=1e-5)
    # Online context equals the offline causal context for the same prefix.
    completed = 4
    hist_prey = [prey[:, t] for t in range(completed + 1)]
    hist_pred = [pred[:, t] for t in range(completed + 1)]
    online = enc.online_context(
        hist_prey, hist_pred, np.zeros(n, bool), np.full(n, -1), horizon=horizon
    )
    np.testing.assert_allclose(online, ctx[:, completed], atol=1e-5)
    np.testing.assert_array_equal(
        enc.online_context(hist_prey[:1], hist_pred[:1], np.zeros(n, bool), np.full(n, -1), horizon=horizon),
        0.0,
    )
    path = tmp_path / "enc.npz"
    enc.save(path)
    loaded = CausalContextEncoder.load(path)
    np.testing.assert_allclose(loaded.causal_context(prey, pred, lengths), ctx, atol=1e-6)
    assert loaded.metadata["training_steps"] == 3
