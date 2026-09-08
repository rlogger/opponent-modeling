"""Streaming 0s contexts match offline causal prefixes without future inputs."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mopa.action_decoder import (
    ActionDecoderConfig,
    FrozenActionEncoder,
    MLPHead,
    SeqGaussian,
)
from mopa.zero_s import ZeroSOpponent


@pytest.fixture(scope="module")
def opponent():
    cfg = ActionDecoderConfig(action_type="continuous", hid=8)
    params = SeqGaussian(lat=cfg.lat, hid=cfg.hid).init(
        jax.random.PRNGKey(11), jnp.zeros((1, cfg.window, 10)), jnp.ones((1, cfg.window), bool),
    )
    decoder = MLPHead(out=2, hid=cfg.hid).init(jax.random.PRNGKey(12), jnp.zeros((1, 16)))
    encoder = FrozenActionEncoder(params, np.linspace(-1, 1, 8, dtype=np.float32),
                                  np.linspace(0.5, 2, 8, dtype=np.float32), cfg)
    return ZeroSOpponent(encoder, decoder, np.zeros((3, 8), np.float32))


def test_streaming_matches_every_offline_prefix_across_windows_and_terminals(opponent):
    rng = np.random.default_rng(20)
    lengths = np.array([1, 7, 8, 9, 15, 16, 17, 24, 25])
    state = rng.normal(size=(len(lengths), 26, 66)).astype(np.float32)
    action = rng.uniform(-1, 1, size=(len(lengths), 25, 2)).astype(np.float32)
    expected = opponent.context(state, action, lengths)
    initial = opponent.initial_context(len(lengths))
    np.testing.assert_array_equal(initial.context, 0)

    def step(carry, inputs):
        raw, red, active = inputs
        new = opponent.update_context(carry, raw, red, active)
        return new, new.context

    @jax.jit
    def stream(carry, states, actions, active):
        return jax.lax.scan(step, carry, (states, actions, active))

    final, contexts = stream(
        initial, jnp.swapaxes(state[:, :-1], 0, 1), jnp.swapaxes(action, 0, 1),
        jnp.arange(25)[:, None] < lengths[None, :],
    )
    actual = np.concatenate([np.asarray(initial.context)[:, None], np.asarray(contexts).swapaxes(0, 1)], axis=1)
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)
    np.testing.assert_array_equal(final.observed, lengths)
    assert final.window.shape == (len(lengths), 8, 10)
    assert final.completed_sum.shape == (len(lengths), 8)


def test_inactive_rows_freeze_every_field_even_with_unusable_inputs(opponent):
    carry = opponent.initial_context(2)
    update = jax.jit(lambda c, state, action, active: opponent.update_context(c, state, action, active))
    carry = update(carry, jnp.ones((2, 66)), jnp.full((2, 2), 0.5), jnp.ones(2, bool))
    frozen = update(carry, jnp.full((2, 66), jnp.nan), jnp.full((2, 2), jnp.nan), jnp.zeros(2, bool))
    for before, after in zip(jax.tree.leaves(carry), jax.tree.leaves(frozen)):
        np.testing.assert_array_equal(before, after)
    mixed = update(carry, jnp.ones((2, 66)), jnp.zeros((2, 2)), jnp.array([True, False]))
    for before, after in zip(jax.tree.leaves(carry), jax.tree.leaves(mixed)):
        np.testing.assert_array_equal(before[1], after[1])
    np.testing.assert_array_equal(mixed.observed, [2, 1])


def test_streaming_context_uses_observed_red_actions_and_saved_encoder(opponent, tmp_path):
    initial = opponent.initial_context(1)
    state = jnp.ones((1, 66))
    positive = opponent.update_context(initial, state, jnp.ones((1, 2)), jnp.ones(1, bool))
    negative = opponent.update_context(initial, state, -jnp.ones((1, 2)), jnp.ones(1, bool))
    assert not np.allclose(positive.context, negative.context)
    opponent.save(tmp_path / "opponent.msgpack")
    restored = ZeroSOpponent.load(tmp_path / "opponent.msgpack")
    restored_context = restored.update_context(restored.initial_context(1), state, jnp.ones((1, 2)), jnp.ones(1, bool))
    np.testing.assert_array_equal(positive.context, restored_context.context)
    # A new episode has no access to any context from the preceding episode.
    np.testing.assert_array_equal(restored.initial_context(1).context, 0)


def test_streaming_rejects_incompatible_shapes(opponent):
    with pytest.raises(ValueError, match="positive"):
        opponent.initial_context(0)
    with pytest.raises(ValueError, match="streaming 0s"):
        opponent.update_context(opponent.initial_context(2), jnp.zeros((2, 17)), jnp.zeros((2, 2)), jnp.ones(2, bool))
