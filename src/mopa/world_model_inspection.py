"""Fixed-input, learned trajectory inspection; no planning, training, or simulator.

Prototype swaps are oracle interventions, not online history inference. The
diagnostic horizon is fixed: continuation probabilities are returned, but states
after a predicted stop are extrapolations, not valid episode continuations.
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


@partial(jax.jit, static_argnames=("clamped",))
def _rollout(model, initial, blue, context, mean, std, red, *, clamped):
    x0 = model.encode(initial, model.encoder.params, jax.random.PRNGKey(0))

    def step(x, actions):
        u, fixed_v = (jnp.broadcast_to(a[:, None], x.shape[:-1] + (2,)) for a in actions)
        v = None
        if model.opponent_mode == "factored":
            v = fixed_v if clamped else model.red_action(x, context, model.red_model.params)
        a = model.transition_inputs(u, context, v)
        continuation = (jax.nn.sigmoid(model.continue_logits(x, a, model.continue_model.params))
                        if model.predict_continues else jnp.ones(x.shape[:-1], jnp.float32))
        new_x = model.next(x, a, model.dynamics_model.params)
        row = {"state": new_x, "continuation": continuation}
        if v is not None:
            row["red_action"] = v
        return new_x, row

    _, result = jax.lax.scan(step, x0, (blue, red))
    result["state"] = jnp.concatenate([x0[None], result["state"]], axis=0) * std + mean
    return result


def rollout_prototypes(agent, initial_state, blue_actions, prototypes, obs_mean, obs_std,
                       *, clamped_red_actions=None):
    """Roll out every initial state under every fixed 8D opponent prototype.

    Inputs: initial_state[B,66], blue_actions[H,B,2], prototypes[K,8], and
    checkpoint normalization[66]. Optional clamped_red_actions[H,B,2] is valid
    only for factored models and holds the joint actions identical across K.

    Returns NumPy state[H+1,B,K,66], continuation[H,B,K], and, only for factored
    models, red_action[H,B,K,2]. Continuation[t] belongs to transition t -> t+1.
    Implicit models ignore prototypes; conditioned/factored models must already
    have matching context dimensions. No checkpoint weights or modes are changed.
    The caller must bind prototypes to their original encoder/checkpoint; equal
    dimensions alone cannot verify that semantic provenance.
    """
    model = agent.model
    if model.encoder_type != "identity" or model.latent_dim != 66 or model.action_dim != 2:
        raise ValueError("Inspection requires an identity 66D state model with 2D actions")
    if model.opponent_mode not in {"implicit", "conditioned", "factored"}:
        raise ValueError("Unsupported opponent mode")
    if model.opponent_mode != "implicit" and model.context_dim != 8:
        raise ValueError("Conditioned/factored checkpoint must use the same 8D prototype context")
    if model.predict_continues and model.continue_model is None:
        raise ValueError("Missing enabled continuation head")
    if model.opponent_mode == "factored" and model.red_model is None:
        raise ValueError("Missing factored opponent head")
    initial, blue, z, mean, std = (np.asarray(value, dtype=np.float32) for value in
                                 (initial_state, blue_actions, prototypes, obs_mean, obs_std))
    if initial.ndim != 2 or initial.shape[1] != 66 or not len(initial):
        raise ValueError("initial_state must have shape (B,66), B > 0")
    if blue.ndim != 3 or blue.shape[1:] != (len(initial), 2) or not len(blue):
        raise ValueError("blue_actions must have shape (H,B,2), H > 0")
    if z.ndim != 2 or z.shape[1] != 8 or not len(z):
        raise ValueError("prototypes must have shape (K,8), K > 0")
    if mean.shape != (66,) or std.shape != (66,) or np.any(std <= 0):
        raise ValueError("Normalization must be 66D with positive scales")
    if not all(np.isfinite(value).all() for value in (initial, blue, z, mean, std)):
        raise ValueError("All rollout inputs must be finite")
    if np.any(np.abs(blue) > 1):
        raise ValueError("blue_actions must be bounded in [-1,1]")
    clamped = clamped_red_actions is not None
    red = np.zeros_like(blue)
    if clamped:
        if model.opponent_mode != "factored":
            raise ValueError("Clamped red actions require factored mode")
        red = np.asarray(clamped_red_actions, dtype=np.float32)
        if red.shape != blue.shape or not np.isfinite(red).all() or np.any(np.abs(red) > 1):
            raise ValueError("clamped_red_actions must be finite, bounded, and shaped (H,B,2)")

    key = jax.random.PRNGKey(0)
    encoded = np.asarray(model.encode(jnp.asarray(np.stack([mean, mean + std])), model.encoder.params, key))
    if (not np.allclose(encoded[0], 0, atol=1e-5)
            or not np.allclose(encoded[1], 1, atol=1e-4)):
        raise ValueError("Normalization does not match the checkpoint identity encoder")
    shape = (len(initial), len(z))
    context = (np.zeros(shape + (model.context_dim,), np.float32) if model.opponent_mode == "implicit"
               else np.broadcast_to(z, shape + (8,)))
    result = _rollout(model, jnp.broadcast_to(initial[:, None], shape + (66,)),
                      jnp.asarray(blue), jnp.asarray(context), jnp.asarray(mean),
                      jnp.asarray(std), jnp.asarray(red), clamped=clamped)
    result = {name: np.asarray(value) for name, value in result.items()}
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError("Nonfinite imagined output; shorten the diagnostic horizon or inspect the checkpoint")
    if "red_action" in result and np.any(np.abs(result["red_action"]) > 1):
        raise ValueError("Opponent produced an unbounded imagined action")
    return result
