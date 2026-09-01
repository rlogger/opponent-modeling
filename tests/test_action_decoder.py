import inspect

import jax
import numpy as np

from mopa.action_decoder import (
    ActionDecoderConfig,
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
