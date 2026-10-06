"""Frozen vanilla BC in the no-context factored world-model path."""
import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import serialization

from mopa.bc_continuous import FrozenBCOpponent, fit_continuous_bc
from mopa.tdmpc import create_agent, load_config, validate_config
from mopa.tdmpc_data import SequenceReplay
from mopa.zero_s import zero_s_features


@pytest.fixture(scope="module")
def tiny():
    rng = np.random.default_rng(31)
    state = rng.normal(size=(6, 9, 66)).astype(np.float32)
    features = np.asarray(zero_s_features(state[:, :-1]))
    actions = np.tanh(features[..., :2]).astype(np.float32)
    policy = fit_continuous_bc(features.reshape(-1, 8), actions.reshape(-1, 2), 2,
                               steps=10, batch_size=8, hidden_size=8)
    opponent = FrozenBCOpponent(policy)
    cfg = load_config(profile="smoke")
    cfg.update(opponent_mode="factored", context_dim=0)
    cfg["encoder"]["type"] = "identity"
    cfg["world_model"]["hidden_dim"] = 16
    mean = rng.normal(size=66).astype(np.float32)
    std = rng.uniform(0.5, 2, 66).astype(np.float32)
    return opponent, state, actions, cfg, mean, std


def test_factored_accepts_empty_context_but_conditioned_rejects_it(tiny):
    _, _, _, cfg, mean, std = tiny
    agent = create_agent(cfg, 66, key=jax.random.PRNGKey(0), obs_mean=mean, obs_std=std)
    assert agent.model.context_dim == 0
    assert agent.model.transition_extra_dim == 2
    assert agent.model.value_extra_dim == 0
    bad = copy.deepcopy(cfg)
    bad["opponent_mode"] = "conditioned"
    with pytest.raises(NotImplementedError, match="context_dim"):
        validate_config(bad)
    bad["opponent_mode"], bad["context_dim"] = "factored", -1
    with pytest.raises(ValueError, match="negative"):
        validate_config(bad)


def test_adapter_uses_only_current_eight_features_and_checks_contract(tiny):
    opponent, state, _, cfg, mean, std = tiny
    raw = jnp.asarray(state[:3, 0])
    context = jnp.zeros((3, 0))
    expected = opponent.policy.act(zero_s_features(raw))
    np.testing.assert_array_equal(opponent.actions(raw, context), expected)
    changed = raw.at[:, 8:].add(100)
    np.testing.assert_array_equal(opponent.actions(changed, context), expected)
    with pytest.raises(ValueError, match="empty context"):
        opponent.actions(raw, jnp.zeros((3, 8)))
    with pytest.raises(ValueError, match="66D"):
        opponent.actions(raw[:, :8])
    base = create_agent(cfg, 66, key=jax.random.PRNGKey(0), obs_mean=mean, obs_std=std)
    with pytest.raises(ValueError, match="does not match"):
        opponent.attach(base, mean + 1, std)
    with pytest.raises(ValueError, match="positive"):
        opponent.attach(base, mean, -std)
    with pytest.raises(ValueError, match="no latent context"):
        opponent.attach(base.replace(model=base.model.replace(context_dim=8)), mean, std)
    agent = opponent.attach(base, mean, std)
    x = agent.model.encode(raw, agent.model.encoder.params, jax.random.PRNGKey(1))
    actual = agent.model.red_action(x, context, agent.model.red_model.params)
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    assert np.isfinite(actual).all() and np.max(np.abs(actual)) <= 1
    assert agent.red_loss_scale == 0


def test_frozen_bc_update_plan_and_checkpoint_roundtrip(tiny, tmp_path):
    opponent, state, actions, cfg, mean, std = tiny

    def template(opp):
        base = create_agent(cfg, 66, key=jax.random.PRNGKey(0), obs_mean=mean, obs_std=std)
        return opp.attach(base, mean, std)

    agent = template(opponent)
    lengths = np.array([4, 8, 7, 5, 8, 6])
    data = dict(state=state, blue_action=actions * 0.4, red_action=actions,
                blue_reward=np.zeros((6, 8), np.float32),
                terminated_capture=np.zeros((6, 8), bool),
                truncated_timeout=np.arange(8)[None, :] == lengths[:, None] - 1,
                valid_length=lengths)
    context = np.zeros((6, 9, 0), np.float32)
    replay = SequenceReplay.from_dataset(data, np.arange(3), agent.horizon, context)
    batch = replay.sample(np.random.default_rng(3), agent.batch_size)
    updated, info = agent.update(**batch, key=jax.random.PRNGKey(4))
    assert all(np.isfinite(np.asarray(info[k])).all() for k in ("total_loss", "red_loss", "policy_loss"))
    for before, after in zip(jax.tree.leaves(agent.model.red_model.params),
                             jax.tree.leaves(updated.model.red_model.params), strict=True):
        np.testing.assert_array_equal(before, after)
    assert any(not np.array_equal(before, after) for before, after in zip(
        jax.tree.leaves(agent.model.dynamics_model.params),
        jax.tree.leaves(updated.model.dynamics_model.params), strict=True))
    action, plan = updated.act(jnp.asarray(state[0, 0]), context=jnp.zeros((0,)),
                               key=jax.random.PRNGKey(5))
    assert action.shape == (2,) and np.isfinite(action).all() and np.max(np.abs(action)) <= 1
    assert plan[0].shape == (agent.horizon, 2)
    path = tmp_path / "opponent.npz"
    opponent.save(path)
    loaded = FrozenBCOpponent.load(path)
    np.testing.assert_array_equal(loaded.actions(state), opponent.actions(state))
    restored = serialization.from_bytes(template(loaded), serialization.to_bytes(updated))
    restored_action, _ = restored.act(jnp.asarray(state[0, 0]), context=jnp.zeros((0,)),
                                      key=jax.random.PRNGKey(5))
    np.testing.assert_array_equal(restored_action, action)
