"""Tests for the 7/31-review JEPA training procedure (GRU, L2, LayerNorm)."""
import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")
pytest.importorskip("optax")

import jax
import jax.numpy as jnp

from mopa.encoders import (
    Enc,
    GRUEnc,
    Pred,
    collapse_diagnostics,
    effective_rank,
    encode_jepa,
    encode_jepa_gru,
    encode_vae,
    gather_prefix_latents,
    squared_l2_prediction_loss,
    train_identity_jepa,
    train_jepa,
    train_jepa_gru,
    train_jepa_gru_with_params,
    train_jepa_with_params,
    train_vae_with_params,
    unit_normalize,
)


def _param_names(tree):
    names = set()

    def visit(node, prefix=""):
        for k, v in node.items():
            if isinstance(v, dict):
                visit(v, prefix + k + "/")
            else:
                names.add(prefix + k)

    visit(jax.tree_util.tree_map(lambda x: None, tree)["params"])
    return names


@pytest.mark.parametrize(
    "module,x",
    [
        (Enc(lat=2), jnp.zeros((1, 8))),
        (Pred(lat=2), jnp.zeros((1, 2))),
        (GRUEnc(lat=2, hid=16), jnp.zeros((1, 4, 3))),
    ],
)
def test_encoders_use_layernorm_not_batchnorm(module, x):
    params = module.init(jax.random.PRNGKey(0), x)
    names = _param_names(params)
    assert any("LayerNorm" in n for n in names)
    assert not any("BatchNorm" in n for n in names)


def test_gru_prefix_latent_ignores_future_steps():
    enc = GRUEnc(lat=3, hid=16)
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (5, 10, 4))
    params = enc.init(rng, x)
    z = enc.apply(params, x)
    assert z.shape == (5, 10, 3)

    x_mut = x.at[:, 6:].set(99.0)
    z_mut = enc.apply(params, x_mut)
    np.testing.assert_allclose(z[:, :6], z_mut[:, :6], atol=1e-5)
    assert not np.allclose(z[:, 9], z_mut[:, 9])


def test_gru_pad_mask_ignores_suffix_garbage():
    enc = GRUEnc(lat=2, hid=16)
    rng = jax.random.PRNGKey(1)
    x = jax.random.normal(rng, (4, 8, 3))
    lengths = np.array([3, 5, 4, 6], dtype=np.int32)
    params = enc.init(rng, x, lengths)
    z = enc.apply(params, x, lengths)

    x_bad = x.at[:, :, :].set(x)
    x_bad = x_bad.at[0, 3:].set(123.0)
    x_bad = x_bad.at[1, 5:].set(-77.0)
    z_bad = enc.apply(params, x_bad, lengths)

    for i, L in enumerate(lengths):
        np.testing.assert_allclose(z[i, :L], z_bad[i, :L], atol=1e-5)
        np.testing.assert_allclose(z[i, L - 1], z_bad[i, L - 1], atol=1e-5)


def test_train_jepa_gru_handles_variable_lengths():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(32, 12, 4)).astype(np.float32)
    lengths = rng.integers(5, 13, size=32)
    z = train_jepa_gru(
        x, lengths, jax.random.PRNGKey(0), lat=2, hid=16, steps=5
    )
    assert z.shape == (32, 12, 2)
    assert np.isfinite(z).all()
    # Unit-sphere outputs.
    norms = np.linalg.norm(z[:, 0], axis=-1)
    np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-4)


@pytest.mark.parametrize("target_mode", ["later_prefix", "full"])
def test_gru_jepa_uses_fresh_keys_for_context_and_target(target_mode):
    # Typed-key checking catches accidental key reuse in compiled samplers.
    # Both target modes remain repeatable from a fresh equal-valued root key.
    x = np.random.default_rng(52).normal(size=(3, 5, 2)).astype(np.float32)
    lengths = np.array([2, 4, 5], np.int32)
    with jax.debug_key_reuse(True):
        first, _, _ = train_jepa_gru_with_params(
            x, lengths, jax.random.key(52), hid=4, steps=2,
            target_mode=target_mode,
        )
        repeated, _, _ = train_jepa_gru_with_params(
            x, lengths, jax.random.key(52), hid=4, steps=2,
            target_mode=target_mode,
        )
    np.testing.assert_array_equal(first, repeated)
    assert np.isfinite(first).all()


def test_gru_jepa_never_promotes_short_episode_padding_to_valid_data():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(8, 6, 3)).astype(np.float32)
    lengths = np.array([1, 2, 2, 3, 4, 5, 6, 1], dtype=np.int32)
    x[lengths[:, None] <= np.arange(6)[None, :]] = 99.0

    z, _, _ = train_jepa_gru_with_params(
        x,
        lengths,
        jax.random.PRNGKey(0),
        lat=2,
        hid=8,
        steps=3,
        tmin=3,
    )

    assert z.shape == (8, 6, 2)
    assert np.isfinite(z).all()

    with pytest.raises(ValueError, match="length >= 2"):
        train_jepa_gru_with_params(
            x[:2],
            np.ones(2, dtype=np.int32),
            jax.random.PRNGKey(1),
            lat=2,
            hid=8,
            steps=1,
        )


def test_train_jepa_gru_with_params_roundtrip():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(16, 10, 3)).astype(np.float32)
    lengths = np.full(16, 8, dtype=np.int32)
    z, params, target = train_jepa_gru_with_params(
        x, lengths, jax.random.PRNGKey(0), lat=2, hid=16, steps=5
    )
    z2 = encode_jepa_gru(params, x, lengths, lat=2, hid=16)
    np.testing.assert_allclose(z, z2, atol=1e-5)
    assert target is not None
    full = gather_prefix_latents(z, lengths)
    assert full.shape == (16, 2)


def test_train_jepa_smoke():
    rng = np.random.default_rng(0)
    xc = rng.normal(size=(32, 6)).astype(np.float32)
    xt = rng.normal(size=(32, 6)).astype(np.float32)
    z = train_jepa(xc, xt, jax.random.PRNGKey(0), lat=2, steps=5)
    assert z.shape == (32, 2)
    assert np.isfinite(z).all()


def test_window_jepa_and_vae_frozen_encode_roundtrip():
    rng = np.random.default_rng(4)
    xc = rng.normal(size=(16, 6)).astype(np.float32)
    xt = rng.normal(size=(16, 6)).astype(np.float32)
    zj, jepa_params = train_jepa_with_params(
        xc, xt, jax.random.PRNGKey(0), lat=2, steps=3
    )
    zv, vae_params = train_vae_with_params(
        xc, jax.random.PRNGKey(1), lat=2, steps=3
    )
    np.testing.assert_allclose(zj, encode_jepa(jepa_params, xc, lat=2), atol=1e-5)
    np.testing.assert_allclose(zv, encode_vae(vae_params, xc, lat=2), atol=1e-5)


def test_squared_l2_prediction_loss_matches_hand_calculation():
    predicted = jnp.asarray([[1.0, 2.0], [0.0, 1.0]])
    target = jnp.asarray([[4.0, 6.0], [0.0, -1.0]])
    # Per-row squared L2 values are 25 and 4; batch mean is 14.5.
    assert float(squared_l2_prediction_loss(predicted, target)) == pytest.approx(14.5)


def test_collapse_diagnostics_distinguish_constant_vs_diverse():
    const = np.ones((20, 2), dtype=np.float32)
    d_const = collapse_diagnostics(const)
    assert d_const["across_std_mean"] < 1e-6
    assert d_const["effective_rank"] < 1.1

    rng = np.random.default_rng(0)
    diverse = rng.normal(size=(20, 2)).astype(np.float32)
    d_div = collapse_diagnostics(diverse)
    assert d_div["across_std_mean"] > 0.5
    assert effective_rank(diverse) > effective_rank(const)


def test_identity_jepa_requires_explicit_label_flag():
    rng = np.random.default_rng(0)
    xc = rng.normal(size=(8, 4)).astype(np.float32)
    xt = rng.normal(size=(8, 4)).astype(np.float32)
    y = np.array([0, 0, 1, 1, 2, 2, 0, 1], dtype=np.int32)
    with pytest.raises(ValueError, match="uses_eval_labels"):
        train_identity_jepa(xc, xt, y, jax.random.PRNGKey(0), steps=2)


def test_unit_normalize_maps_to_sphere():
    z = jnp.asarray([[3.0, 4.0], [0.0, 2.0]])
    u = np.asarray(unit_normalize(z))
    np.testing.assert_allclose(np.linalg.norm(u, axis=-1), [1.0, 1.0], atol=1e-5)
