"""Independent numerical replay accepts quota cuts but rejects invented outcomes."""
import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mopa.continuous_data import deterministic_specialist_action, markov_state
from mopa.evaluation import run_matched_episodes
from test_control_benchmark import env_and_red, opponent  # noqa: F401


@pytest.fixture(scope="module")
def verifier():
    path = Path(__file__).resolve().parents[1] / "experiments/matched_control_20260908/verify.py"
    spec = importlib.util.spec_from_file_location("control_replay_verifier", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def record(env, red, opponent, quota):
    reset = np.asarray(jax.random.split(jax.random.PRNGKey(90), 3))
    step_keys = np.asarray(jax.random.split(jax.random.PRNGKey(91), 3))

    def blue(state, obs, context, carry, key, t):
        return jnp.tile(jnp.array([[.3, -.2]]), (3, 1)), carry

    result = run_matched_episodes(env, red, blue, reset, step_keys, horizon=5,
        context_mode="online" if opponent else "zero", label=0, zero_s=opponent,
        context_width=None if opponent else 0, max_transitions=quota, record_transitions=True)
    return {**result["transitions"], "environment_seed": reset, "step_seed": step_keys}


def audit(verifier, path, env, red, opponent=None):
    return verifier.verify_trace(path, env, jax.jit(jax.vmap(env.step_env)),
        jax.jit(lambda s: markov_state(env, s)),
        jax.jit(lambda obs: deterministic_specialist_action(red, obs, 35)), opponent)


@pytest.mark.parametrize("quota,use_context", [(1, False), (7, False), (None, False), (1, True), (7, True)])
def test_real_replay_quota_padding_and_causal_context(verifier, env_and_red, opponent, tmp_path, quota, use_context):  # noqa: F811
    env, red = env_and_red
    context = opponent if use_context else None
    path = tmp_path / "trace.npz"
    np.savez(path, **record(env, red, context, quota))
    result = audit(verifier, path, env, red, context)
    assert result["valid_transitions"] == (15 if quota is None else quota)
    assert result["state_max_error"] == result["reward_max_error"] == result["specialist_action_max_error"] == 0.
    assert result["zero_length_rows"] == (2 if quota == 1 else 0)
    if use_context:
        assert result["causal_context_max_error"] < 2e-5
        assert result["future_pair_perturbation_max_error"] == 0.


@pytest.mark.parametrize("corruption", ["reward", "red_action", "state", "capture", "padding", "final_context"])
def test_numerical_replay_rejects_corruption(verifier, env_and_red, opponent, tmp_path, corruption):  # noqa: F811
    env, red = env_and_red
    data = record(env, red, opponent, 7)
    if corruption == "reward":
        data["blue_reward"][0, 0] += 1.
    elif corruption == "red_action":
        data["red_action"][0, 0, 0] += .1
    elif corruption == "state":
        data["state"][0, 0, 0] += .1
    elif corruption == "capture":
        data["terminated_capture"][0, 0] = True
    elif corruption == "padding":
        data["blue_action"][0, -1, 0] = .1
    else:
        data["final_context"][0, 0] += 1.
    path = tmp_path / "corrupt.npz"
    np.savez(path, **data)
    with pytest.raises(AssertionError):
        audit(verifier, path, env, red, opponent)
