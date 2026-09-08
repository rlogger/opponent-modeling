"""0s-to-Equation-3 contracts, independent of expensive experiment artifacts."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import serialization

from mopa.action_decoder import ActionDecoderConfig, fit_action_decoder_vae
from mopa.tdmpc import create_agent, load_config
from mopa.tdmpc_data import SequenceReplay
from mopa.zero_s import (
    ZeroSOpponent,
    rollout_zero_s,
    strategy_prototypes,
    zero_s_features,
)


@pytest.fixture(scope="module")
def tiny():
    rng = np.random.default_rng(20)
    state = rng.normal(size=(6, 9, 66)).astype(np.float32)
    actions = np.tanh(state[:, :-1, :2]).astype(np.float32)
    lengths = np.array([4, 8, 7, 5, 8, 6])
    train = np.arange(3)
    fit = fit_action_decoder_vae(
        np.asarray(zero_s_features(state[:, :-1])), actions, lengths, train,
        jax.random.PRNGKey(0),
        config=ActionDecoderConfig(action_type="continuous", hid=8, steps=2, batch=8),
    )
    opponent = ZeroSOpponent.from_fit(fit, np.tile(np.arange(3), 2), train)
    cfg = load_config(profile="smoke")
    cfg.update(opponent_mode="factored", context_dim=8)
    cfg["encoder"]["type"] = "identity"
    cfg["world_model"]["hidden_dim"] = 16
    mean, std = rng.normal(size=66).astype(np.float32), rng.uniform(0.5, 2, 66).astype(np.float32)
    return opponent, state, actions, lengths, cfg, mean, std


def test_features_are_current_state_not_future():
    state = np.arange(66, dtype=np.float32)
    np.testing.assert_array_equal(zero_s_features(state), [2, 3, 0, 1, 6, 7, 4, 5])
    with pytest.raises(ValueError, match="66D"):
        zero_s_features(np.zeros(17))


def test_prototypes_use_only_equal_weight_training_episodes():
    z = np.array([[1, 2], [3, 4], [10, 20], [30, 40], [100, 200], [300, 400], [999, 999]], np.float32)
    labels = np.array([0, 0, 1, 1, 2, 2, 0])
    expected = np.array([[2, 3], [20, 30], [200, 300]])
    np.testing.assert_array_equal(strategy_prototypes(z, labels, np.arange(6)), expected)
    z[-1] = -999
    np.testing.assert_array_equal(strategy_prototypes(z, labels, np.arange(6)), expected)
    with pytest.raises(ValueError, match="include"):
        strategy_prototypes(z, labels, np.arange(4))
    with pytest.raises(ValueError, match="unique"):
        strategy_prototypes(z, labels, np.array([0, 0, 2, 4]))


def test_context_is_strictly_causal_and_freezes_after_terminal(tiny):
    opponent, state, actions, lengths, *_ = tiny
    ctx = opponent.context(state, actions, lengths)
    assert ctx.shape == (6, 9, 8)
    np.testing.assert_array_equal(ctx[:, 0], 0)
    changed_state, changed_actions = state.copy(), actions.copy()
    changed_state[:, 3:] += 100  # even current decision-state 3 is not in z_3
    changed_actions[:, 3:] *= -1
    altered = opponent.context(changed_state, changed_actions, lengths)
    np.testing.assert_allclose(ctx[:, :4], altered[:, :4], atol=1e-6)
    for ep, length in enumerate(lengths):
        np.testing.assert_allclose(ctx[ep, length:], np.broadcast_to(ctx[ep, length], ctx[ep, length:].shape))


def test_adapter_uses_original_decoder_and_rejects_wrong_normalization(tiny):
    opponent, state, _, _, cfg, mean, std = tiny
    agent = create_agent(cfg, 66, key=jax.random.PRNGKey(1), obs_mean=mean, obs_std=std)
    with pytest.raises(ValueError, match="does not match"):
        opponent.attach(agent, mean + 1, std)
    agent = opponent.attach(agent, mean, std)
    raw = jnp.asarray(state[:3, 0])
    x = agent.model.encode(raw, agent.model.encoder.params, jax.random.PRNGKey(2))
    expected = opponent.actions(raw, opponent.prototypes)
    actual = agent.model.red_action(x, jnp.asarray(opponent.prototypes), agent.model.red_model.params)
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    assert np.isfinite(actual).all() and np.max(np.abs(actual)) <= 1
    # Clamp both physical actions: changing z cannot enter the physics input.
    u = jnp.ones((3, 2)) * 0.2
    np.testing.assert_array_equal(
        agent.model.transition_inputs(u, opponent.prototypes, actual),
        agent.model.transition_inputs(u, opponent.prototypes + 10, actual),
    )


def test_frozen_decoder_update_plan_and_fresh_checkpoint_roundtrip(tiny, tmp_path):
    opponent, state, actions, lengths, cfg, mean, std = tiny
    def template(opp):
        base = create_agent(cfg, 66, key=jax.random.PRNGKey(1), obs_mean=mean, obs_std=std)
        return opp.attach(base, mean, std)
    agent = template(opponent)
    data = dict(state=state, blue_action=actions * 0.4, red_action=actions,
                blue_reward=np.zeros((6, 8), np.float32),
                terminated_capture=np.zeros((6, 8), bool),
                truncated_timeout=np.arange(8)[None, :] == lengths[:, None] - 1,
                valid_length=lengths)
    context = opponent.context(state, actions, lengths)
    replay = SequenceReplay.from_dataset(data, np.arange(3), agent.horizon, context)
    batch = replay.sample(np.random.default_rng(3), agent.batch_size)
    updated, info = agent.update(**batch, key=jax.random.PRNGKey(4))
    assert all(np.isfinite(np.asarray(info[k])).all() for k in ("total_loss", "red_loss", "policy_loss"))
    for before, after in zip(jax.tree.leaves(agent.model.red_model.params), jax.tree.leaves(updated.model.red_model.params)):
        np.testing.assert_array_equal(before, after)
    assert any(not np.array_equal(before, after) for before, after in zip(
        jax.tree.leaves(agent.model.dynamics_model.params), jax.tree.leaves(updated.model.dynamics_model.params)))
    action, _ = updated.act(jnp.asarray(state[0, 0]), context=jnp.asarray(opponent.prototypes[0]), key=jax.random.PRNGKey(5))
    assert action.shape == (2,) and np.max(np.abs(action)) <= 1

    opponent.save(tmp_path / "opponent.msgpack")
    restored_opponent = ZeroSOpponent.load(tmp_path / "opponent.msgpack")
    np.testing.assert_array_equal(restored_opponent.context(state, actions, lengths), context)
    (tmp_path / "agent.msgpack").write_bytes(serialization.to_bytes(updated))
    restored = serialization.from_bytes(template(restored_opponent), (tmp_path / "agent.msgpack").read_bytes())
    restored_action, _ = restored.act(jnp.asarray(state[0, 0]), context=jnp.asarray(opponent.prototypes[0]), key=jax.random.PRNGKey(5))
    np.testing.assert_array_equal(restored_action, action)

    # Same starting state and blue actions; only three prototype vectors differ.
    initial = jnp.repeat(state[0, :1], 3, axis=0)
    blue = jnp.zeros((3, 3, 2))
    states, red = rollout_zero_s(restored, initial, blue, jnp.asarray(opponent.prototypes), mean, std)
    assert states.shape == (4, 3, 66) and red.shape == (3, 3, 2)
    np.testing.assert_allclose(states[0], initial, atol=1e-6)
    assert np.isfinite(states).all() and np.max(np.abs(red)) <= 1
    np.testing.assert_allclose(red[1], restored_opponent.actions(states[1], opponent.prototypes), atol=1e-6)
    assert not np.allclose(red[0, 0], red[0, 1])
