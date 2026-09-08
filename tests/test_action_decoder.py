import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mopa.action_decoder import (
    ActionDecoderConfig,
    MLPHead,
    decode_action_decoder,
    encode_action_decoder_vae,
    fit_action_decoder_vae,
    gather_windows,
    make_windows,
    pool_episode_prefix_latents,
    pool_window_latents,
)


def test_nonoverlapping_windows_padding_and_pooling():
    windows = make_windows(np.array([3, 10], dtype=np.int32), width=4)

    np.testing.assert_array_equal(windows.episode, [0, 1, 1, 1])
    np.testing.assert_array_equal(windows.start, [0, 0, 4, 8])
    np.testing.assert_array_equal(windows.lengths, [3, 4, 4, 2])
    np.testing.assert_array_equal(
        windows.mask,
        [
            [True, True, True, False],
            [True, True, True, True],
            [True, True, True, True],
            [True, True, False, False],
        ],
    )

    values = np.arange(20, dtype=np.float32).reshape(2, 10)
    gathered = gather_windows(values, windows)
    np.testing.assert_array_equal(gathered[0], [0, 1, 2, 0])
    np.testing.assert_array_equal(gathered[-1], [18, 19, 0, 0])

    latent = np.array([[1.0], [2.0], [4.0], [9.0]], dtype=np.float32)
    pooled = pool_window_latents(latent, windows.episode, n_episodes=2)
    np.testing.assert_allclose(pooled[:, 0], [1.0, 5.0])


def test_fit_frozen_masking_prefix_pooling_and_label_free_api():
    rng = np.random.default_rng(4)
    state = rng.normal(size=(4, 6, 3)).astype(np.float32)
    action = rng.integers(0, 5, size=(4, 6), dtype=np.int32)
    lengths = np.array([3, 6, 5, 2], dtype=np.int32)
    train_idx = np.array([0, 1, 2], dtype=np.int32)
    config = ActionDecoderConfig(lat=2, hid=4, window=3, steps=0, batch=5)

    fit = fit_action_decoder_vae(
        state,
        action,
        lengths,
        train_idx,
        jax.random.PRNGKey(0),
        config=config,
    )

    assert "label" not in inspect.signature(fit_action_decoder_vae).parameters
    assert fit.encoding.episode_latents.shape == (4, 2)
    assert fit.encoding.window_latents.shape == (6, 2)
    assert fit.encoding.prefix_latents.shape == (6, 3, 2)
    assert fit.history == ()
    assert 0.0 <= fit.decoder_action_accuracy <= 1.0

    valid_train = np.arange(state.shape[1])[None, :] < lengths[train_idx, None]
    expected_rows = state[train_idx][valid_train]
    np.testing.assert_allclose(fit.encoder.state_mean, expected_rows.mean(0))

    padded_state = state.copy()
    padded_action = action.copy()
    for episode, length in enumerate(lengths):
        padded_state[episode, length:] = 10_000.0
        padded_action[episode, length:] = 4

    frozen = encode_action_decoder_vae(
        fit.encoder, padded_state, padded_action, lengths
    )
    np.testing.assert_allclose(
        frozen.prefix_latents, fit.encoding.prefix_latents, atol=1e-6
    )
    np.testing.assert_allclose(
        frozen.episode_latents, fit.encoding.episode_latents, atol=1e-6
    )

    full_prefix = pool_episode_prefix_latents(
        fit.encoding.prefix_latents,
        fit.encoding.windows,
        lengths,
        n_episodes=len(lengths),
    )
    np.testing.assert_allclose(full_prefix, fit.encoding.episode_latents, atol=1e-6)

    zero_prefix = pool_episode_prefix_latents(
        fit.encoding.prefix_latents,
        fit.encoding.windows,
        np.zeros_like(lengths),
        n_episodes=len(lengths),
    )
    np.testing.assert_array_equal(
        zero_prefix, np.zeros_like(fit.encoding.episode_latents)
    )


def test_discrete_decoder_keeps_unsquashed_scores():
    config = ActionDecoderConfig(lat=2, hid=4)
    state = jnp.array([[0.2, -0.3, 1.2]], dtype=jnp.float32)
    latent = jnp.array([[0.1, -0.7]], dtype=jnp.float32)
    model = MLPHead(out=config.n_actions, hid=config.hid)
    inputs = jnp.concatenate([latent, state], axis=-1)
    params = model.init(jax.random.PRNGKey(5), inputs)
    expected = model.apply(params, inputs)
    actual = decode_action_decoder(params, state, latent, config)
    assert actual.shape == (1, 5)
    np.testing.assert_array_equal(actual, expected)


def test_continuous_fit_bounded_decoder_and_causal_prefix_pooling():
    rng = np.random.default_rng(7)
    state = rng.normal(size=(4, 6, 3)).astype(np.float32)
    action = np.tanh(state[..., :2] * 0.5).astype(np.float32)
    lengths = np.array([3, 6, 5, 2], dtype=np.int32)
    train_idx = np.array([0, 1, 2], dtype=np.int32)
    config = ActionDecoderConfig(
        lat=2, hid=4, window=3, steps=2, batch=5, action_type="continuous"
    )
    fit = fit_action_decoder_vae(
        state, action, lengths, train_idx, jax.random.PRNGKey(0), config=config
    )

    assert fit.episode_latents.shape == (4, 2)
    assert fit.window_latents.shape == (6, 2)
    assert fit.prefix_latents.shape == (6, 3, 2)
    assert fit.decoder_action_accuracy is None
    assert np.isfinite(fit.decoder_action_mse)
    assert len(fit.history) == 2
    assert all(np.isfinite(list(row.values())).all() for row in fit.history)
    assert all(np.isfinite(p).all() for p in jax.tree_util.tree_leaves(fit.decoder_params))

    normalized = (state[:, 0] - fit.encoder.state_mean) / fit.encoder.state_std
    decode = jax.jit(lambda params, s, z: decode_action_decoder(params, s, z, config))
    prediction = decode(fit.decoder_params, normalized, fit.episode_latents)
    assert prediction.shape == (4, 2)
    assert np.isfinite(prediction).all()
    assert np.all(np.abs(prediction) <= 1)
    full_prefix = pool_episode_prefix_latents(
        fit.prefix_latents, fit.encoding.windows, lengths, len(lengths)
    )
    np.testing.assert_allclose(full_prefix, fit.episode_latents, atol=1e-6)

    # Neither padding nor unseen future actions/states can affect a prefix.
    changed_state = state.copy()
    changed_action = action.copy()
    observed = np.minimum(lengths, 2)
    for episode, length in enumerate(lengths):
        changed_state[episode, 2:length] += 20.0
        changed_action[episode, 2:length] *= -1.0
        changed_state[episode, length:] = np.nan
        changed_action[episode, length:] = np.nan
    changed = encode_action_decoder_vae(
        fit.encoder, changed_state, changed_action, lengths
    )
    expected_prefix = pool_episode_prefix_latents(
        fit.prefix_latents, fit.encoding.windows, observed, len(lengths)
    )
    actual_prefix = pool_episode_prefix_latents(
        changed.prefix_latents, changed.windows, observed, len(lengths)
    )
    np.testing.assert_allclose(actual_prefix, expected_prefix, atol=1e-6)


@pytest.mark.parametrize("bad_action", [np.nan, np.inf, 1.01, -1.01])
def test_continuous_actions_reject_nonfinite_or_unbounded_values(bad_action):
    action = np.zeros((1, 3, 2), dtype=np.float32)
    action[0, 0, 0] = bad_action
    with pytest.raises(ValueError, match="finite and lie in"):
        fit_action_decoder_vae(
            np.zeros((1, 3, 2)), action, np.array([3]), np.array([0]),
            jax.random.PRNGKey(0),
            config=ActionDecoderConfig(steps=0, action_type="continuous"),
        )


def test_continuous_actions_require_vector_shape():
    with pytest.raises(ValueError, match="action_dim"):
        fit_action_decoder_vae(
            np.zeros((1, 3, 2)), np.zeros((1, 3)), np.array([3]), np.array([0]),
            jax.random.PRNGKey(0),
            config=ActionDecoderConfig(steps=0, action_type="continuous"),
        )
