"""Blue-only PPO response to frozen opponents, with matched state/history input.

Actor and critic both see the same normalized 66D state and optional causal
8D context. This is not the legacy jointly trained, local-observation MAPPO.
The caller owns opponent selection, causal context, real rollouts and returns;
this module cannot update a red specialist or the frozen context encoder.
"""
from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization, struct
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState

from mopa.continuous import (
    atanh_clipped,
    gaussian_entropy,
    tanh_gaussian_log_prob,
    tanh_gaussian_sample,
)
from mopa.nets import ContinuousActor


class ResponseValue(nn.Module):
    hidden: int = 128

    @nn.compact
    def __call__(self, features):
        for _ in range(2):
            features = nn.relu(nn.Dense(
                self.hidden, kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0),
            )(features))
        return nn.Dense(1, kernel_init=orthogonal(1), bias_init=constant(0))(features)[..., 0]


@struct.dataclass
class ResponsePPO:
    actor: TrainState
    critic: TrainState
    state_mean: jax.Array
    state_std: jax.Array
    context_dim: int = struct.field(pytree_node=False, default=0)
    hidden: int = struct.field(pytree_node=False, default=128)
    learning_rate: float = struct.field(pytree_node=False, default=3e-4)
    clip_epsilon: float = struct.field(pytree_node=False, default=0.2)
    entropy_coefficient: float = struct.field(pytree_node=False, default=0.01)
    value_coefficient: float = struct.field(pytree_node=False, default=0.5)
    max_grad_norm: float = struct.field(pytree_node=False, default=0.5)

    def features(self, observations, context):
        observations, context = jnp.asarray(observations), jnp.asarray(context)
        if observations.shape[-1:] != (66,) or context.shape != observations.shape[:-1] + (self.context_dim,):
            raise ValueError("response inputs must have matching (...,66) state and (...,context_dim) context")
        return jnp.concatenate(((observations - self.state_mean) / self.state_std, context), axis=-1)

    def value(self, observations, context):
        return self.critic.apply_fn(self.critic.params, self.features(observations, context))

    def sample(self, observations, context, key):
        """Collect exact pre-tanh samples for the subsequent PPO ratio."""
        features = self.features(observations, context)
        mean, log_std = self.actor.apply_fn(self.actor.params, features)
        pre_tanh, action = tanh_gaussian_sample(key, mean, log_std)
        return {
            "actions": action,
            "pre_tanh": pre_tanh,
            "log_probs": tanh_gaussian_log_prob(mean, log_std, pre_tanh),
            "old_values": self.critic.apply_fn(self.critic.params, features),
        }

    def act(self, observations, context, key, deterministic=True):
        mean, log_std = self.actor.apply_fn(self.actor.params, self.features(observations, context))
        return jnp.tanh(mean) if deterministic else tanh_gaussian_sample(key, mean, log_std)[1]

    def save(self, path: str | Path) -> None:
        configuration = {
            name: getattr(self, name) for name in (
                "context_dim", "hidden", "learning_rate", "clip_epsilon",
                "entropy_coefficient", "value_coefficient", "max_grad_norm",
            )
        }
        Path(path).write_bytes(serialization.msgpack_serialize({
            "schema": 1, "method": "blue_only_ppo_frozen_opponents",
            "configuration": configuration,
            "state": serialization.to_state_dict(self),
        }))

    @classmethod
    def load(cls, path: str | Path) -> "ResponsePPO":
        payload = serialization.msgpack_restore(Path(path).read_bytes())
        if payload.get("schema") != 1 or payload.get("method") != "blue_only_ppo_frozen_opponents":
            raise ValueError("unsupported PPO response artifact")
        state = payload["state"]
        template = create_response(state["state_mean"], state["state_std"], **payload["configuration"])
        return serialization.from_state_dict(template, state)


def create_response(
    state_mean, state_std, *, context_dim=0, seed=0, hidden=128,
    learning_rate=3e-4, clip_epsilon=0.2, entropy_coefficient=0.01,
    value_coefficient=0.5, max_grad_norm=0.5,
) -> ResponsePPO:
    mean, std = np.asarray(state_mean, np.float32), np.asarray(state_std, np.float32)
    if mean.shape != (66,) or std.shape != (66,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("finite 66D state mean and positive standard deviation required")
    if context_dim not in (0, 8) or hidden < 1:
        raise ValueError("context_dim must be 0 or 8 and hidden must be positive")
    if not all(np.isfinite(v) for v in (learning_rate, clip_epsilon, entropy_coefficient, value_coefficient, max_grad_norm)) or learning_rate <= 0 or not 0 < clip_epsilon < 1 or entropy_coefficient < 0 or value_coefficient <= 0 or max_grad_norm <= 0:
        raise ValueError("invalid PPO optimizer or loss configuration")
    actor, critic = ContinuousActor(hidden_dim=hidden), ResponseValue(hidden)
    ka, kc = jax.random.split(jax.random.PRNGKey(seed))
    inputs = jnp.zeros((1, 66 + context_dim), jnp.float32)
    optimizer = optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(learning_rate, eps=1e-5))
    return ResponsePPO(
        TrainState.create(apply_fn=actor.apply, params=actor.init(ka, inputs), tx=optimizer),
        TrainState.create(apply_fn=critic.apply, params=critic.init(kc, inputs), tx=optimizer),
        jnp.asarray(mean), jnp.asarray(std), context_dim, hidden, learning_rate,
        clip_epsilon, entropy_coefficient, value_coefficient, max_grad_norm,
    )


def _prepare_batch(agent: ResponsePPO, batch: dict[str, Any], *, warm_start: bool):
    observations = np.asarray(batch["observations"], np.float32)
    if observations.ndim < 2 or observations.shape[-1] != 66:
        raise ValueError("observations must end in 66 state features")
    lead = observations.shape[:-1]
    mask = np.asarray(batch.get("valid_mask", np.ones(lead, bool)), bool)
    if mask.shape != lead or not mask.any():
        raise ValueError("valid_mask must align with observations and select training samples")
    dimensions = {"observations": 66, "context": agent.context_dim, "returns": None}
    if warm_start:
        dimensions["actions"] = 2
    else:
        dimensions.update(pre_tanh=2, log_probs=None, old_values=None, advantages=None)
    arrays = {}
    for name, width in dimensions.items():
        if name not in batch:
            raise ValueError(f"missing training field {name}; PPO requires exact collected pre_tanh samples")
        value = np.asarray(batch[name], np.float32)
        expected = lead if width is None else lead + (width,)
        if value.shape != expected:
            raise ValueError(f"{name} shape {value.shape} does not match {expected}")
        selected = value[mask]
        if not np.isfinite(selected).all():
            raise ValueError(f"non-finite valid samples in {name}")
        arrays[name] = jnp.asarray(selected)
    if warm_start:
        if np.any(np.abs(np.asarray(arrays["actions"])) > 1 + 1e-6):
            raise ValueError("BC demonstration actions must lie in [-1,1]")
        # Only demonstrations need inversion. PPO always uses recorded u.
        arrays["pre_tanh"] = atanh_clipped(arrays.pop("actions"))
    elif "actions" in batch:
        actions = np.asarray(batch["actions"], np.float32)
        if actions.shape != lead + (2,) or not np.allclose(actions[mask], np.tanh(np.asarray(arrays["pre_tanh"])), atol=1e-6):
            raise ValueError("collected actions must equal tanh of recorded pre_tanh")
    if not warm_start:
        advantages = arrays["advantages"]
        arrays["advantages"] = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return arrays


def _minibatch(agent, batch, weight, *, warm_start):
    features = agent.features(batch["observations"], batch["context"])

    def average(values):
        return jnp.sum(values * weight) / jnp.maximum(weight.sum(), 1)

    def actor_loss(params):
        mean, log_std = agent.actor.apply_fn(params, features)
        log_probs = tanh_gaussian_log_prob(mean, log_std, batch["pre_tanh"])
        entropy = average(gaussian_entropy(log_std))
        if warm_start:
            return -average(log_probs), (entropy, jnp.array(0.), jnp.array(0.))
        log_ratio = log_probs - batch["log_probs"]
        ratio = jnp.exp(log_ratio)
        advantage = batch["advantages"]
        objective = jnp.minimum(ratio * advantage, jnp.clip(ratio, 1 - agent.clip_epsilon, 1 + agent.clip_epsilon) * advantage)
        loss = -average(objective) - agent.entropy_coefficient * entropy
        return loss, (entropy, average((ratio - 1) - log_ratio), average(jnp.abs(ratio - 1) > agent.clip_epsilon))

    def value_loss(params):
        value = agent.critic.apply_fn(params, features)
        error = jnp.square(value - batch["returns"])
        if not warm_start:
            clipped = batch["old_values"] + jnp.clip(value - batch["old_values"], -agent.clip_epsilon, agent.clip_epsilon)
            error = jnp.maximum(error, jnp.square(clipped - batch["returns"]))
        return 0.5 * agent.value_coefficient * average(error)

    (actor_loss_value, (entropy, kl, clipped)), actor_grads = jax.value_and_grad(actor_loss, has_aux=True)(agent.actor.params)
    critic_loss_value, critic_grads = jax.value_and_grad(value_loss)(agent.critic.params)
    agent = agent.replace(
        actor=agent.actor.apply_gradients(grads=actor_grads),
        critic=agent.critic.apply_gradients(grads=critic_grads),
    )
    return agent, {
        "actor_loss": actor_loss_value, "value_loss": critic_loss_value,
        "entropy": entropy, "approx_kl": kl, "clip_fraction": clipped,
    }


@partial(jax.jit, static_argnames=("warm_start",))
def _run_updates(agent, data, indices, weights, *, warm_start):
    def update(current, selected):
        rows, weight = selected
        batch = jax.tree.map(lambda value: value[rows], data)
        return _minibatch(current, batch, weight, warm_start=warm_start)
    return jax.lax.scan(update, agent, (indices, weights))


def _metrics(history, samples, updates):
    metrics = {name: float(np.asarray(values).mean()) for name, values in history.items()}
    for name in ("actor_loss", "value_loss"):
        metrics[f"{name}_first"] = float(np.asarray(history[name])[0])
        metrics[f"{name}_last"] = float(np.asarray(history[name])[-1])
    if not all(np.isfinite(value) for value in metrics.values()):
        raise FloatingPointError("non-finite PPO response optimization metrics")
    return {**metrics, "valid_samples": int(samples), "gradient_updates": int(updates)}


def update_response(agent, batch, key, *, epochs=4, minibatch_size=128):
    """Optimize one fresh on-policy batch; padded/terminated rows are excluded."""
    if epochs < 1 or minibatch_size < 1:
        raise ValueError("positive epochs and minibatch_size required")
    data = _prepare_batch(agent, batch, warm_start=False)
    n = data["observations"].shape[0]
    padded = ((n + minibatch_size - 1) // minibatch_size) * minibatch_size
    permutations = jax.vmap(lambda k: jax.random.permutation(k, n))(jax.random.split(key, epochs))
    indices = jnp.pad(permutations, ((0, 0), (0, padded - n))).reshape(-1, minibatch_size)
    weights = jnp.broadcast_to(jnp.arange(padded) < n, (epochs, padded)).reshape(-1, minibatch_size).astype(jnp.float32)
    updated, history = _run_updates(agent, data, indices, weights, warm_start=False)
    return updated, _metrics(history, n, len(indices))


def warm_start_response(agent, batch, key, *, steps=2000, minibatch_size=128):
    """BC actor / actual-return value initialization on train-only demonstrations.

    Returns must come from real recorded rewards, not imagined rollouts. This
    offline optimization is reported separately from on-policy PPO updates.
    The caller must supply only training specialist episodes and causal context.
    """
    if steps < 1 or minibatch_size < 1:
        raise ValueError("positive steps and minibatch_size required")
    data = _prepare_batch(agent, batch, warm_start=True)
    n = data["observations"].shape[0]
    indices = jax.random.randint(key, (steps, minibatch_size), 0, n)
    updated, history = _run_updates(agent, data, indices, jnp.ones_like(indices, jnp.float32), warm_start=True)
    return updated, _metrics(history, n, steps)
