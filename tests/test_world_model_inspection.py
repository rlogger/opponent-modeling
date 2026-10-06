"""Small synthetic inference tests only: no optimizer updates or environment."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import serialization

from mopa.action_decoder import ActionDecoderConfig, FrozenActionEncoder, MLPHead
from mopa.tdmpc import create_agent, load_config
from mopa.world_model_inspection import rollout_prototypes
from mopa.zero_s import ZeroSOpponent, rollout_zero_s


@pytest.fixture(scope="module")
def inputs():
    rng = np.random.default_rng(3)
    return dict(initial_state=rng.normal(size=(2, 66)).astype(np.float32),
                blue_actions=rng.uniform(-0.7, 0.7, (3, 2, 2)).astype(np.float32),
                prototypes=rng.normal(size=(3, 8)).astype(np.float32),
                obs_mean=rng.normal(size=66).astype(np.float32),
                obs_std=rng.uniform(0.5, 2, 66).astype(np.float32))


@pytest.fixture(scope="module")
def agents(inputs):
    result = {}
    for mode in ("implicit", "conditioned", "factored"):
        cfg = load_config(profile="smoke")
        cfg.update(opponent_mode=mode, context_dim=0 if mode == "implicit" else 8)
        cfg["encoder"]["type"] = "identity"
        cfg["world_model"].update(hidden_dim=8, predict_continues=True)
        cfg["factored"]["red_hidden_dim"] = 8
        result[mode] = create_agent(cfg, 66, key=jax.random.PRNGKey(0),
                                    obs_mean=inputs["obs_mean"], obs_std=inputs["obs_std"])
    return result


def observable(agent):
    """Analytic test heads expose routing without training the persistence init."""
    def dynamics(_variables, value):
        controls = value[..., 66:]
        signal = controls.sum(-1)
        return jnp.zeros(value.shape[:-1] + (66,)).at[..., 0].set(signal * 0.1)

    def continuation(_variables, value):
        return (value[..., :1] + value[..., 66:].sum(-1, keepdims=True)) * 0.2

    def red(_variables, value):
        return jnp.tanh(value[..., :2] + value[..., 66:68])

    model = agent.model.replace(
        dynamics_model=agent.model.dynamics_model.replace(apply_fn=dynamics),
        continue_model=agent.model.continue_model.replace(apply_fn=continuation))
    if model.red_model is not None:
        model = model.replace(red_model=model.red_model.replace(apply_fn=red))
    return agent.replace(model=model)


@pytest.mark.parametrize("mode", ["implicit", "conditioned", "factored"])
def test_actual_smoke_models_shapes_normalization_and_no_mutation(agents, inputs, mode):
    agent = agents[mode]
    before = serialization.to_bytes(agent)
    out = rollout_prototypes(agent, **inputs)
    assert serialization.to_bytes(agent) == before
    assert out["state"].shape == (4, 2, 3, 66)
    assert out["continuation"].shape == (3, 2, 3)
    np.testing.assert_allclose(out["state"][0], np.broadcast_to(inputs["initial_state"][:, None], (2, 3, 66)), atol=1e-6)
    assert all(isinstance(value, np.ndarray) and np.isfinite(value).all() for value in out.values())
    assert np.all((out["continuation"] >= 0) & (out["continuation"] <= 1))
    assert ("red_action" in out) == (mode == "factored")
    if mode == "factored":
        assert out["red_action"].shape == (3, 2, 3, 2)
        assert np.max(np.abs(out["red_action"])) <= 1


def test_implicit_ignores_prototypes_and_legacy_unused_context_dimension(agents, inputs):
    agent = observable(agents["implicit"])
    out = rollout_prototypes(agent, **inputs)
    altered = rollout_prototypes(agent.replace(model=agent.model.replace(context_dim=3)),
                                  **{**inputs, "prototypes": inputs["prototypes"] + 20})
    for key in out:
        np.testing.assert_array_equal(out[key], altered[key])
        np.testing.assert_array_equal(out[key][:, :, 0], out[key][:, :, 1])


def test_conditioned_routes_fixed_prototype_into_each_transition(agents, inputs):
    agent = observable(agents["conditioned"])
    out = rollout_prototypes(agent, **inputs)
    assert not np.allclose(out["state"][1, :, 0], out["state"][1, :, 1])
    assert not np.allclose(out["continuation"][:, :, 0], out["continuation"][:, :, 1])
    x = agent.model.encode(jnp.asarray(inputs["initial_state"]), agent.model.encoder.params, jax.random.PRNGKey(0))
    for t, blue in enumerate(inputs["blue_actions"]):
        context = jnp.broadcast_to(inputs["prototypes"][1], (2, 8))
        a = agent.model.transition_inputs(jnp.asarray(blue), context)
        expected = jax.nn.sigmoid(agent.model.continue_logits(x, a, agent.model.continue_model.params))
        np.testing.assert_allclose(out["continuation"][t, :, 1], expected, atol=1e-6)
        x = agent.model.next(x, a, agent.model.dynamics_model.params)
        np.testing.assert_allclose(out["state"][t + 1, :, 1], x * inputs["obs_std"] + inputs["obs_mean"], atol=1e-6)


def test_factored_recomputes_red_from_current_state_without_same_step_blue(agents, inputs):
    agent = observable(agents["factored"])
    out = rollout_prototypes(agent, **inputs)
    altered = rollout_prototypes(agent, **{**inputs, "blue_actions": -inputs["blue_actions"]})
    np.testing.assert_array_equal(out["red_action"][0], altered["red_action"][0])
    assert not np.allclose(out["red_action"][1], altered["red_action"][1])
    for t in range(3):
        x = (out["state"][t] - inputs["obs_mean"]) / inputs["obs_std"]
        c = np.broadcast_to(inputs["prototypes"], (2, 3, 8))
        expected = agent.model.red_action(jnp.asarray(x), jnp.asarray(c), agent.model.red_model.params)
        np.testing.assert_allclose(out["red_action"][t], expected, atol=1e-6)


def test_factored_clamped_joint_actions_make_physics_and_continuation_invariant(agents, inputs):
    agent = observable(agents["factored"])
    fixed = np.full_like(inputs["blue_actions"], 0.2)
    out = rollout_prototypes(agent, **inputs, clamped_red_actions=fixed)
    changed = rollout_prototypes(agent, **{**inputs, "prototypes": inputs["prototypes"] * -4}, clamped_red_actions=fixed)
    for key in out:
        np.testing.assert_array_equal(out[key], changed[key])
        np.testing.assert_array_equal(out[key][:, :, 0], out[key][:, :, 2])
    np.testing.assert_array_equal(out["red_action"], np.broadcast_to(fixed[:, :, None], (3, 2, 3, 2)))


def test_attached_0s_matches_existing_rollout_without_training(agents, inputs):
    cfg = ActionDecoderConfig(action_type="continuous", hid=8, steps=0)
    decoder = MLPHead(out=2, hid=8).init(jax.random.PRNGKey(5), jnp.zeros((1, 16)))
    # This test calls the decoder only; no history encoder is fitted or invoked.
    encoder = FrozenActionEncoder({}, np.zeros(8, np.float32), np.ones(8, np.float32), cfg)
    opponent = ZeroSOpponent(encoder, decoder, inputs["prototypes"])
    agent = opponent.attach(observable(agents["factored"]), inputs["obs_mean"], inputs["obs_std"])
    out = rollout_prototypes(agent, **inputs)
    initial = np.broadcast_to(inputs["initial_state"][:, None], (2, 3, 66)).reshape(6, 66)
    blue = np.broadcast_to(inputs["blue_actions"][:, :, None], (3, 2, 3, 2)).reshape(3, 6, 2)
    context = np.broadcast_to(inputs["prototypes"], (2, 3, 8)).reshape(6, 8)
    states, red = rollout_zero_s(agent, initial, blue, context, inputs["obs_mean"], inputs["obs_std"])
    np.testing.assert_allclose(out["state"], np.asarray(states).reshape(4, 2, 3, 66), atol=1e-6)
    np.testing.assert_allclose(out["red_action"], np.asarray(red).reshape(3, 2, 3, 2), atol=1e-6)


def test_missing_continuation_returns_ones_without_freezing(agents, inputs):
    agent = observable(agents["implicit"])
    agent = agent.replace(model=agent.model.replace(predict_continues=False, continue_model=None))
    out = rollout_prototypes(agent, **inputs)
    np.testing.assert_array_equal(out["continuation"], np.ones((3, 2, 3)))
    assert not np.allclose(out["state"][0], out["state"][-1])


@pytest.mark.parametrize("field,value", [
    ("initial_state", np.zeros((2, 65))), ("initial_state", np.zeros((0, 66))),
    ("blue_actions", np.zeros((3, 1, 2))), ("blue_actions", np.full((3, 2, 2), 1.01)),
    ("prototypes", np.zeros((3, 3))), ("prototypes", np.full((3, 8), np.nan)),
    ("obs_mean", np.zeros(65)), ("obs_std", np.zeros(66)),
])
def test_rejects_invalid_arrays(agents, inputs, field, value):
    with pytest.raises(ValueError):
        rollout_prototypes(agents["factored"], **{**inputs, field: value})


def test_rejects_incompatible_models_normalization_and_clamps(agents, inputs):
    for replacement in ({"context_dim": 3}, {"encoder_type": "mlp"}, {"latent_dim": 111}, {"action_dim": 5}):
        agent = agents["conditioned"].replace(model=agents["conditioned"].model.replace(**replacement))
        with pytest.raises(ValueError):
            rollout_prototypes(agent, **inputs)
    with pytest.raises(ValueError, match="Normalization does not match"):
        rollout_prototypes(agents["factored"], **{**inputs, "obs_mean": inputs["obs_mean"] + 1})
    with pytest.raises(ValueError, match="require factored"):
        rollout_prototypes(agents["implicit"], **inputs, clamped_red_actions=inputs["blue_actions"])
    for bad in (np.zeros((3, 2, 3, 2)), np.full((3, 2, 2), np.nan), np.full((3, 2, 2), 1.01)):
        with pytest.raises(ValueError, match="clamped_red_actions"):
            rollout_prototypes(agents["factored"], **inputs, clamped_red_actions=bad)
