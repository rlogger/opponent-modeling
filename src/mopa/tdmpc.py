"""Single-agent TD-MPC2 core (Gate 0 source port).

Ported from ShaneFlandermeyer/tdmpc2-jax (MIT), pinned commit
``5b05ff452424896d709848e1f249bd67e269b8a1``:

- ``tdmpc2_jax/common/{activations,util,loss,scale}.py``  -> helper section
- ``tdmpc2_jax/networks/{mlp,ensemble}.py``                -> network section
- ``tdmpc2_jax/world_model.py``                            -> ``WorldModel``
- ``tdmpc2_jax/tdmpc2.py`` (create / act / update)         -> ``TDMPC2``
- ``tdmpc2_jax/train.py`` (encoder + agent construction)   -> ``build_encoder``,
  ``create_agent``

``TDMPC2.plan`` / ``estimate_value`` live in :mod:`mopa.mppi`.

Compatibility substitutions (documented in ``third_party/tdmpc2-jax/UPSTREAM.md``):
``einops.rearrange`` -> ``jnp.reshape``; ``tfd.MultivariateNormalDiag`` ->
``distrax.MultivariateNormalDiag``; ``jaxtyping`` -> plain ``jax.Array``/``Any``.
The world latent is named ``x`` (upstream ``z``) per the handoff notation. No
algorithmic changes; the transition is Equation 1, ``x_next = d(x, u)``.
"""
from __future__ import annotations

import copy
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
from flax import struct
from flax.training.train_state import TrainState

from mopa import mppi

PRNGKey = jax.Array
Params = Any
PyTree = Any

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "tdmpc2.yaml"

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "Ensemble",
    "NormedLinear",
    "TDMPC2",
    "WorldModel",
    "build_encoder",
    "create_agent",
    "load_config",
    "mish",
    "percentile_normalization",
    "sg",
    "simnorm",
    "soft_crossentropy",
    "symexp",
    "symlog",
    "two_hot",
    "two_hot_inv",
    "validate_gate0_config",
]


# --------------------------------------------------------------------------- #
# Helpers: upstream tdmpc2_jax/common/{activations,util,loss,scale}.py
# --------------------------------------------------------------------------- #
def mish(x: jax.Array) -> jax.Array:
    return x * jnp.tanh(jnp.log(1 + jnp.exp(x)))


def simnorm(x: jax.Array, simplex_dim: int = 8) -> jax.Array:
    # Substitution: einops ``'...(L V) -> ... L V'`` / ``'... L V -> ... (L V)'``
    # replaced by row-major ``jnp.reshape``. Grouping, softmax axis, and output
    # shape are identical.
    shape = x.shape
    x = jnp.reshape(x, (*shape[:-1], shape[-1] // simplex_dim, simplex_dim))
    x = jax.nn.softmax(x, axis=-1)
    return jnp.reshape(x, shape)


def symlog(x: jax.Array) -> jax.Array:
    return jnp.sign(x) * jnp.log(1 + jnp.abs(x))


def symexp(x: jax.Array) -> jax.Array:
    return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1)


def two_hot(x: jax.Array, low: float, high: float, num_bins: int) -> jax.Array:
    """Two-hot encode symlog-transformed values into ``num_bins`` bins."""
    bin_size = (high - low) / (num_bins - 1)

    x = jnp.clip(symlog(x), low, high)
    bin_index = jnp.floor((x - low) / bin_size).astype(int)
    bin_offset = (x - low) / bin_size - bin_index.astype(float)

    two_hot = jax.nn.one_hot(bin_index, num_bins) * (1 - bin_offset[..., None]) + (
        jax.nn.one_hot(bin_index + 1, num_bins) * bin_offset[..., None]
    )
    return two_hot


def two_hot_inv(
    x: jax.Array,
    low: float,
    high: float,
    num_bins: int,
    apply_softmax: bool = True,
) -> jax.Array:
    bins = jnp.linspace(low, high, num_bins)

    if apply_softmax:
        x = jax.nn.softmax(x, axis=-1)

    x = jnp.sum(x * bins, axis=-1)
    return symexp(x)


def sg(x: PyTree) -> PyTree:
    return jax.tree.map(jax.lax.stop_gradient, x)


def soft_crossentropy(
    pred_logits: jax.Array,
    target: jax.Array,
    low: float,
    high: float,
    num_bins: int,
) -> jax.Array:
    pred = jax.nn.log_softmax(pred_logits, axis=-1)
    target = two_hot(target, low, high, num_bins)
    return -(pred * target).sum(axis=-1)


def percentile_normalization(
    x: jax.Array,
    prev_scale: jax.Array,
    percentile_range: jax.Array = jnp.array([5, 95]),
    tau: float = 0.01,
) -> jax.Array:
    """Running scale of the range between two percentiles of ``x``."""
    percentiles = jnp.percentile(x, percentile_range)
    scale = percentiles[1] - percentiles[0]

    return tau * scale + (1 - tau) * prev_scale


# --------------------------------------------------------------------------- #
# Networks: upstream tdmpc2_jax/networks/{mlp,ensemble}.py
# --------------------------------------------------------------------------- #
class NormedLinear(nn.Module):
    features: int
    activation: Optional[Callable[[jax.Array], jax.Array]] = None
    dropout_rate: Optional[float] = None

    kernel_init: Callable = nn.initializers.truncated_normal(stddev=0.02)
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array, train: bool = True) -> jax.Array:
        x = nn.Dense(
            features=self.features,
            kernel_init=self.kernel_init,
            bias_init=nn.initializers.zeros_init(),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )(x)

        x = nn.LayerNorm(dtype=self.dtype)(x)
        if self.activation is not None:
            x = self.activation(x)

        if self.dropout_rate is not None and self.dropout_rate > 0:
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

        return x


class Ensemble(nn.Module):
    base_module: Any
    num: int = 2

    @nn.compact
    def __call__(self, *args, **kwargs):
        ensemble = nn.vmap(
            self.base_module,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.num,
        )
        return ensemble()(*args, **kwargs)


# --------------------------------------------------------------------------- #
# World model: upstream tdmpc2_jax/world_model.py
# --------------------------------------------------------------------------- #
class WorldModel(struct.PyTreeNode):
    # Models
    encoder: TrainState
    dynamics_model: TrainState
    reward_model: TrainState
    policy_model: TrainState
    value_model: TrainState
    target_value_model: TrainState
    continue_model: Optional[TrainState]
    # Spaces
    action_dim: int = struct.field(pytree_node=False)
    # Architecture
    latent_dim: int = struct.field(pytree_node=False)
    simnorm_dim: int = struct.field(pytree_node=False)
    num_value_nets: int = struct.field(pytree_node=False)
    num_bins: int = struct.field(pytree_node=False)
    symlog_min: float
    symlog_max: float
    predict_continues: bool = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        # Spaces
        action_dim: int,
        # Encoder module
        encoder: TrainState,
        # World model
        latent_dim: int,
        value_dropout: float,
        num_value_nets: int,
        num_bins: int,
        symlog_min: float,
        symlog_max: float,
        simnorm_dim: int,
        predict_continues: bool,
        # Optimization
        learning_rate: float,
        max_grad_norm: float = 20,
        # Misc
        dtype: jnp.dtype = jnp.float32,
        *,
        key: PRNGKey,
    ) -> "WorldModel":
        dynamics_key, reward_key, value_key, policy_key, continue_key = (
            jax.random.split(key, 5)
        )

        # Latent forward dynamics model
        dynamics_module = nn.Sequential(
            [
                NormedLinear(latent_dim, activation=mish, dtype=dtype),
                NormedLinear(latent_dim, activation=mish, dtype=dtype),
                NormedLinear(latent_dim, activation=None, dtype=dtype),
            ]
        )
        dynamics_model = TrainState.create(
            apply_fn=dynamics_module.apply,
            params=dynamics_module.init(
                dynamics_key, jnp.zeros(latent_dim + action_dim)
            )["params"],
            tx=optax.chain(
                optax.zero_nans(),
                optax.clip_by_global_norm(max_grad_norm),
                optax.adamw(learning_rate),
            ),
        )

        # Transition reward model
        reward_module = nn.Sequential(
            [
                NormedLinear(latent_dim, activation=mish, dtype=dtype),
                NormedLinear(latent_dim, activation=mish, dtype=dtype),
                nn.Dense(num_bins, kernel_init=nn.initializers.zeros),
            ]
        )
        reward_model = TrainState.create(
            apply_fn=reward_module.apply,
            params=reward_module.init(
                reward_key, jnp.zeros(latent_dim + action_dim)
            )["params"],
            tx=optax.chain(
                optax.zero_nans(),
                optax.clip_by_global_norm(max_grad_norm),
                optax.adamw(learning_rate),
            ),
        )

        # Policy model
        policy_module = nn.Sequential(
            [
                NormedLinear(latent_dim, activation=mish, dtype=dtype),
                NormedLinear(latent_dim, activation=mish, dtype=dtype),
                nn.Dense(
                    2 * action_dim,
                    kernel_init=nn.initializers.truncated_normal(0.02),
                ),
            ]
        )
        policy_model = TrainState.create(
            apply_fn=policy_module.apply,
            params=policy_module.init(policy_key, jnp.zeros(latent_dim))["params"],
            tx=optax.chain(
                optax.zero_nans(),
                optax.clip_by_global_norm(max_grad_norm),
                optax.adamw(learning_rate),
            ),
        )

        # Return/value model (ensemble)
        value_param_key, value_dropout_key = jax.random.split(value_key)
        value_base = partial(
            nn.Sequential,
            [
                NormedLinear(
                    latent_dim,
                    activation=mish,
                    dropout_rate=value_dropout,
                    dtype=dtype,
                ),
                NormedLinear(latent_dim, activation=mish, dtype=dtype),
                nn.Dense(num_bins, kernel_init=nn.initializers.zeros),
            ],
        )
        value_ensemble = Ensemble(value_base, num=num_value_nets)
        value_model = TrainState.create(
            apply_fn=value_ensemble.apply,
            params=value_ensemble.init(
                {"params": value_param_key, "dropout": value_dropout_key},
                jnp.zeros(latent_dim + action_dim),
            )["params"],
            tx=optax.chain(
                optax.zero_nans(),
                optax.clip_by_global_norm(max_grad_norm),
                optax.adamw(learning_rate),
            ),
        )
        target_value_model = TrainState.create(
            apply_fn=value_ensemble.apply,
            params=copy.deepcopy(value_model.params),
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        if predict_continues:
            continue_module = nn.Sequential(
                [
                    NormedLinear(latent_dim, activation=mish, dtype=dtype),
                    NormedLinear(latent_dim, activation=mish, dtype=dtype),
                    nn.Dense(1, kernel_init=nn.initializers.zeros),
                ]
            )
            continue_model = TrainState.create(
                apply_fn=continue_module.apply,
                params=continue_module.init(continue_key, jnp.zeros(latent_dim))[
                    "params"
                ],
                tx=optax.chain(
                    optax.zero_nans(),
                    optax.clip_by_global_norm(max_grad_norm),
                    optax.adamw(learning_rate),
                ),
            )
        else:
            continue_model = None

        return cls(
            # Spaces
            action_dim=action_dim,
            # Models
            encoder=encoder,
            dynamics_model=dynamics_model,
            reward_model=reward_model,
            policy_model=policy_model,
            value_model=value_model,
            target_value_model=target_value_model,
            continue_model=continue_model,
            # Architecture
            latent_dim=latent_dim,
            simnorm_dim=simnorm_dim,
            num_value_nets=num_value_nets,
            num_bins=num_bins,
            symlog_min=float(symlog_min),
            symlog_max=float(symlog_max),
            predict_continues=predict_continues,
        )

    @jax.jit
    def encode(self, obs: PyTree, params: Params, key: PRNGKey) -> jax.Array:
        x = self.encoder.apply_fn(
            {"params": params}, obs, rngs={"dropout": key}
        ).astype(jnp.float32)
        return simnorm(x, simplex_dim=self.simnorm_dim)

    @jax.jit
    def next(self, x: jax.Array, a: jax.Array, params: Params) -> jax.Array:
        # Equation 1: x_next = d(x, u). The action is the agent's own action only.
        x = self.dynamics_model.apply_fn(
            {"params": params}, jnp.concatenate([x, a], axis=-1)
        ).astype(jnp.float32)
        return simnorm(x, simplex_dim=self.simnorm_dim)

    @jax.jit
    def reward(
        self, x: jax.Array, a: jax.Array, params: Params
    ) -> Tuple[jax.Array, jax.Array]:
        x = jnp.concatenate([x, a], axis=-1)
        logits = self.reward_model.apply_fn({"params": params}, x).astype(
            jnp.float32
        )
        reward = two_hot_inv(logits, self.symlog_min, self.symlog_max, self.num_bins)
        return reward, logits

    @partial(jax.jit, static_argnames=("deterministic",))
    def sample_actions(
        self,
        x: jax.Array,
        params: Params,
        deterministic: bool = False,
        min_log_std: float = -10,
        max_log_std: float = 2,
        *,
        key: PRNGKey,
    ) -> Tuple[jax.Array, ...]:
        # Chunk the policy model output to get mean and logstd
        mean, log_std = jnp.split(
            self.policy_model.apply_fn({"params": params}, x).astype(jnp.float32),
            2,
            axis=-1,
        )
        log_std = min_log_std + 0.5 * (max_log_std - min_log_std) * (
            jnp.tanh(log_std) + 1
        )

        # Substitution: tfd.MultivariateNormalDiag -> distrax.MultivariateNormalDiag
        # (diagonal Gaussian, reparameterized sampling, same loc/scale_diag
        # parameterization).
        action_dist = distrax.MultivariateNormalDiag(
            loc=mean, scale_diag=jnp.exp(log_std)
        )
        if deterministic:
            action = mean
        else:
            action = action_dist.sample(seed=key)
        log_probs = action_dist.log_prob(action)

        # Squash tanh
        log_probs -= jnp.sum(
            (2 * (jnp.log(2) - action - jax.nn.softplus(-2 * action))), axis=-1
        )
        mean = jnp.tanh(mean)
        action = jnp.tanh(action)
        return action, mean, log_std, log_probs

    @jax.jit
    def Q(
        self, x: jax.Array, a: jax.Array, params: Params, key: PRNGKey
    ) -> Tuple[jax.Array, jax.Array]:
        x = jnp.concatenate([x, a], axis=-1)
        logits = self.value_model.apply_fn(
            {"params": params}, x, rngs={"dropout": key}
        ).astype(jnp.float32)

        Q = two_hot_inv(logits, self.symlog_min, self.symlog_max, self.num_bins)
        return Q, logits


# --------------------------------------------------------------------------- #
# Agent: upstream tdmpc2_jax/tdmpc2.py (create / act / update)
# --------------------------------------------------------------------------- #
class TDMPC2(struct.PyTreeNode):
    model: WorldModel
    value_scale: jax.Array

    # Planning
    horizon: int = struct.field(pytree_node=False)
    mppi_iterations: int = struct.field(pytree_node=False)
    population_size: int = struct.field(pytree_node=False)
    policy_prior_samples: int = struct.field(pytree_node=False)
    num_elites: int = struct.field(pytree_node=False)
    min_plan_std: float
    max_plan_std: float
    temperature: float
    # Optimization
    batch_size: int = struct.field(pytree_node=False)
    discount: float
    rho: float
    consistency_loss_scale: float
    reward_loss_scale: float
    value_loss_scale: float
    continue_loss_scale: float
    entropy_coef: float
    tau: float

    @classmethod
    def create(
        cls,
        world_model: WorldModel,
        # Planning
        horizon: int,
        mppi_iterations: int,
        population_size: int,
        policy_prior_samples: int,
        num_elites: int,
        min_plan_std: float,
        max_plan_std: float,
        temperature: float,
        # Optimization
        discount: float,
        batch_size: int,
        rho: float,
        consistency_loss_scale: float,
        reward_loss_scale: float,
        value_loss_scale: float,
        continue_loss_scale: float,
        entropy_coef: float,
        tau: float,
    ) -> "TDMPC2":
        return cls(
            model=world_model,
            horizon=horizon,
            mppi_iterations=mppi_iterations,
            population_size=population_size,
            policy_prior_samples=policy_prior_samples,
            num_elites=num_elites,
            min_plan_std=min_plan_std,
            max_plan_std=max_plan_std,
            temperature=temperature,
            discount=discount,
            batch_size=batch_size,
            rho=rho,
            consistency_loss_scale=consistency_loss_scale,
            reward_loss_scale=reward_loss_scale,
            value_loss_scale=value_loss_scale,
            continue_loss_scale=continue_loss_scale,
            entropy_coef=entropy_coef,
            tau=tau,
            value_scale=jnp.array([1.0]),
        )

    @partial(jax.jit, static_argnames=("mpc", "deterministic", "train"))
    def act(
        self,
        obs: PyTree,
        prev_plan: Optional[Tuple[jax.Array, jax.Array]] = None,
        mpc: bool = True,
        deterministic: bool = False,
        train: bool = False,
        *,
        key: PRNGKey,
    ) -> Tuple[jax.Array, Optional[Tuple[jax.Array, jax.Array]]]:
        encoder_key, action_key = jax.random.split(key, 2)
        x = self.model.encode(
            obs=obs, params=self.model.encoder.params, key=encoder_key
        )

        if mpc:
            action, plan = self.plan(
                x=x,
                horizon=self.horizon,
                prev_plan=prev_plan,
                deterministic=deterministic,
                train=train,
                key=action_key,
            )
        else:
            action, _, _, _ = self.model.sample_actions(
                x=x,
                deterministic=deterministic,
                params=self.model.policy_model.params,
                key=action_key,
            )
            plan = None

        return action, plan

    def plan(
        self,
        x: jax.Array,
        horizon: int,
        prev_plan: Optional[Tuple[jax.Array, jax.Array]] = None,
        deterministic: bool = False,
        train: bool = False,
        *,
        key: PRNGKey,
    ) -> Tuple[jax.Array, Tuple[jax.Array, jax.Array]]:
        """MPPI planning; see :func:`mopa.mppi.plan`."""
        return mppi.plan(
            self,
            x=x,
            horizon=horizon,
            prev_plan=prev_plan,
            deterministic=deterministic,
            train=train,
            key=key,
        )

    def estimate_value(
        self, x: jax.Array, actions: jax.Array, horizon: int, key: PRNGKey
    ) -> jax.Array:
        """Imagined return of an action sequence; see :func:`mopa.mppi.estimate_value`."""
        return mppi.estimate_value(self, x=x, actions=actions, horizon=horizon, key=key)

    @jax.jit
    def update(
        self,
        observations: PyTree,
        actions: jax.Array,
        rewards: jax.Array,
        next_observations: PyTree,
        terminated: jax.Array,
        truncated: jax.Array,
        *,
        key: PRNGKey,
    ) -> Tuple["TDMPC2", Dict[str, Any]]:
        world_model_key, policy_key = jax.random.split(key, 2)

        def world_model_loss_fn(
            encoder_params: flax.core.FrozenDict,
            dynamics_params: flax.core.FrozenDict,
            value_params: flax.core.FrozenDict,
            reward_params: flax.core.FrozenDict,
            continue_params: flax.core.FrozenDict,
        ) -> Tuple[jax.Array, Dict[str, Any]]:
            encoder_key, value_key = jax.random.split(world_model_key, 2)
            lam = self.rho ** jnp.arange(self.horizon)
            lam /= jnp.sum(lam)

            ###########################################################
            # Encoder forward pass
            ###########################################################
            all_obs = jax.tree.map(
                lambda x, y: jnp.stack([x, y], axis=0),
                observations,
                next_observations,
            )
            all_xs = self.model.encode(
                obs=all_obs, params=encoder_params, key=encoder_key
            )
            encoder_xs = jax.tree.map(lambda x: x[0], all_xs)
            next_xs = jax.tree.map(lambda x: x[1], all_xs)

            ###########################################################
            # Latent rollout (dynamics + consistency loss)
            ###########################################################
            done = jnp.logical_or(terminated, truncated)
            finished = jnp.zeros((self.horizon + 1, self.batch_size), dtype=bool)
            latent_xs = jnp.zeros(
                (self.horizon + 1, self.batch_size, self.model.latent_dim)
            )
            latent_xs = latent_xs.at[0].set(encoder_xs[0])
            consistency_loss = 0
            for t in range(self.horizon):
                x = self.model.next(
                    x=latent_xs[t], a=actions[t], params=dynamics_params
                )
                consistency_loss += lam[t] * jnp.mean(
                    (x - sg(next_xs[t])) ** 2, where=~finished[t][:, None]
                )
                latent_xs = latent_xs.at[t + 1].set(x)
                finished = finished.at[t + 1].set(
                    jnp.logical_or(finished[t], done[t])
                )

            ###########################################################
            # Reward loss
            ###########################################################
            _, reward_logits = self.model.reward(
                x=latent_xs[:-1], a=actions, params=reward_params
            )
            reward_loss = jnp.sum(
                lam[:, None]
                * soft_crossentropy(
                    pred_logits=reward_logits,
                    target=rewards,
                    low=self.model.symlog_min,
                    high=self.model.symlog_max,
                    num_bins=self.model.num_bins,
                ),
                axis=0,
                where=~finished[:-1],
            ).mean()

            ###########################################################
            # Value loss
            ###########################################################
            next_action_key, value_target_key, ensemble_key, value_key = (
                jax.random.split(value_key, 4)
            )

            # TD targets
            next_action = self.model.sample_actions(
                x=next_xs,
                deterministic=False,
                params=self.model.policy_model.params,
                key=next_action_key,
            )[0]
            Qs, _ = self.model.Q(
                x=next_xs,
                a=next_action,
                params=self.model.target_value_model.params,
                key=value_target_key,
            )
            # Subsample value networks
            inds = jax.random.choice(
                ensemble_key,
                jnp.arange(0, self.model.num_value_nets),
                shape=(2,),
                replace=False,
            )
            Q = Qs[inds].min(axis=0)
            td_targets = rewards + (1 - terminated) * self.discount * Q

            _, Q_logits = self.model.Q(
                x=latent_xs[:-1], a=actions, params=value_params, key=value_key
            )
            # Upstream sums this term over axis=1 (batch) rather than axis=0
            # (time) as in the reward loss. Preserved verbatim for parity.
            value_loss = jnp.sum(
                lam[:, None]
                * soft_crossentropy(
                    pred_logits=Q_logits,
                    target=sg(td_targets),
                    low=self.model.symlog_min,
                    high=self.model.symlog_max,
                    num_bins=self.model.num_bins,
                ),
                axis=1,
                where=~finished[:-1],
            ).mean()

            ###########################################################
            # Continue loss
            ###########################################################
            if self.model.predict_continues:
                continue_logits = self.model.continue_model.apply_fn(
                    {"params": continue_params}, latent_xs[:-1]
                ).squeeze(-1)
                continue_loss = optax.sigmoid_binary_cross_entropy(
                    continue_logits, 1 - terminated
                ).mean()
            else:
                continue_loss = 0.0

            total_loss = (
                self.consistency_loss_scale * consistency_loss
                + self.reward_loss_scale * reward_loss
                + self.value_loss_scale * value_loss
                + self.continue_loss_scale * continue_loss
            )

            return total_loss, {
                "consistency_loss": consistency_loss,
                "reward_loss": reward_loss,
                "value_loss": value_loss,
                "continue_loss": continue_loss,
                "total_loss": total_loss,
                "latent_xs": latent_xs,
                "finished": finished,
            }

        # Update world model
        (
            (encoder_grads, dynamics_grads, value_grads, reward_grads, continue_grads),
            model_info,
        ) = jax.grad(world_model_loss_fn, argnums=(0, 1, 2, 3, 4), has_aux=True)(
            self.model.encoder.params,
            self.model.dynamics_model.params,
            self.model.value_model.params,
            self.model.reward_model.params,
            self.model.continue_model.params
            if self.model.predict_continues
            else None,
        )
        new_encoder = self.model.encoder.apply_gradients(grads=encoder_grads)
        new_dynamics_model = self.model.dynamics_model.apply_gradients(
            grads=dynamics_grads
        )
        new_reward_model = self.model.reward_model.apply_gradients(
            grads=reward_grads
        )
        new_value_model = self.model.value_model.apply_gradients(grads=value_grads)
        new_target_value_model = self.model.target_value_model.replace(
            params=optax.incremental_update(
                new_value_model.params,
                self.model.target_value_model.params,
                self.tau,
            )
        )
        if self.model.predict_continues:
            new_continue_model = self.model.continue_model.apply_gradients(
                grads=continue_grads
            )
        else:
            new_continue_model = self.model.continue_model

        # Update policy
        latent_xs = model_info.pop("latent_xs")
        finished = model_info.pop("finished")

        def policy_loss_fn(actor_params: flax.core.FrozenDict):
            action_key, Q_key = jax.random.split(policy_key, 2)
            actions, _, log_std, log_probs = self.model.sample_actions(
                x=latent_xs,
                deterministic=False,
                params=actor_params,
                key=action_key,
            )

            # Compute policy objective (equation 4)
            lam = self.rho ** jnp.arange(self.horizon + 1)
            lam /= jnp.sum(lam)
            Qs, _ = self.model.Q(
                x=latent_xs, a=actions, params=new_value_model.params, key=Q_key
            )
            Q = Qs.mean(axis=0)
            Q_scale = percentile_normalization(Q[0], self.value_scale).clip(1, None)
            policy_loss = jnp.sum(
                lam[:, None] * (self.entropy_coef * log_probs - Q / sg(Q_scale)),
                axis=0,
                where=~finished,
            ).mean()
            return policy_loss, {
                "policy_loss": policy_loss,
                "policy_log_std": log_std,
                "value_scale": Q_scale,
            }

        policy_grads, policy_info = jax.grad(policy_loss_fn, has_aux=True)(
            self.model.policy_model.params
        )
        new_policy = self.model.policy_model.apply_gradients(grads=policy_grads)

        # Update model
        new_agent = self.replace(
            model=self.model.replace(
                encoder=new_encoder,
                dynamics_model=new_dynamics_model,
                reward_model=new_reward_model,
                value_model=new_value_model,
                policy_model=new_policy,
                target_value_model=new_target_value_model,
                continue_model=new_continue_model,
            ),
            value_scale=policy_info["value_scale"],
        )
        info = {**model_info, **policy_info}

        return new_agent, info


# --------------------------------------------------------------------------- #
# Construction: upstream train.py encoder/agent setup + local config loading
# --------------------------------------------------------------------------- #
def build_encoder(
    obs_dim: int,
    encoder_dim: int,
    num_encoder_layers: int,
    latent_dim: int,
    learning_rate: float,
    max_grad_norm: float,
    dtype: jnp.dtype = jnp.float32,
    *,
    key: PRNGKey,
) -> TrainState:
    """State encoder ``h(s) -> pre-SimNorm latent`` as built in upstream ``train.py``."""
    encoder_module = nn.Sequential(
        [
            NormedLinear(encoder_dim, activation=mish, dtype=dtype)
            for _ in range(num_encoder_layers - 1)
        ]
        + [NormedLinear(latent_dim, activation=None, dtype=dtype)]
    )
    return TrainState.create(
        apply_fn=encoder_module.apply,
        params=encoder_module.init(key, jnp.zeros(obs_dim))["params"],
        tx=optax.chain(
            optax.zero_nans(),
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(learning_rate),
        ),
    )


def _deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_update(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_config(
    path: str | Path = DEFAULT_CONFIG_PATH, profile: Optional[str] = None
) -> Dict[str, Any]:
    """Load ``configs/tdmpc2.yaml``.

    ``profile=None`` returns the reference (upstream-default) configuration.
    ``profile="smoke"`` applies the explicitly named reduced-size overrides used
    for compilation-speed tests; it never replaces the reference values on disk.
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)
    profiles = cfg.pop("profiles", {}) or {}
    if profile is not None:
        if profile not in profiles:
            raise KeyError(f"unknown config profile {profile!r}")
        cfg = _deep_update(cfg, profiles[profile])
    return cfg


def validate_gate0_config(config: Dict[str, Any]) -> None:
    """Gate 0 accepts only the unchanged upstream single-agent computation."""
    if config.get("opponent_mode") != "implicit":
        raise NotImplementedError(
            "Gate 0 supports opponent_mode='implicit' only; got "
            f"{config.get('opponent_mode')!r}"
        )
    if int(config.get("context_dim", 0)) != 0:
        raise NotImplementedError(
            f"Gate 0 supports context_dim=0 only; got {config.get('context_dim')!r}"
        )
    if config["world_model"]["latent_dim"] % config["world_model"]["simnorm_dim"]:
        raise ValueError("latent_dim must be a multiple of simnorm_dim")


def create_agent(config: Dict[str, Any], obs_dim: int, *, key: PRNGKey) -> TDMPC2:
    """Build encoder, world model, and agent as upstream ``train.py`` does."""
    validate_gate0_config(config)
    encoder_cfg = config["encoder"]
    model_cfg = dict(config["world_model"])
    tdmpc_cfg = dict(config["tdmpc2"])

    dtype = jnp.dtype(model_cfg.pop("dtype", "float32"))
    _, model_key, encoder_key = jax.random.split(key, 3)

    encoder = build_encoder(
        obs_dim=obs_dim,
        encoder_dim=int(encoder_cfg["encoder_dim"]),
        num_encoder_layers=int(encoder_cfg["num_encoder_layers"]),
        latent_dim=int(model_cfg["latent_dim"]),
        learning_rate=float(encoder_cfg["learning_rate"]),
        max_grad_norm=float(model_cfg["max_grad_norm"]),
        dtype=dtype,
        key=encoder_key,
    )
    model = WorldModel.create(
        action_dim=int(np.prod(config["action_dim"])),
        encoder=encoder,
        latent_dim=int(model_cfg["latent_dim"]),
        value_dropout=float(model_cfg["value_dropout"]),
        num_value_nets=int(model_cfg["num_value_nets"]),
        num_bins=int(model_cfg["num_bins"]),
        symlog_min=float(model_cfg["symlog_min"]),
        symlog_max=float(model_cfg["symlog_max"]),
        simnorm_dim=int(model_cfg["simnorm_dim"]),
        predict_continues=bool(model_cfg["predict_continues"]),
        learning_rate=float(model_cfg["learning_rate"]),
        max_grad_norm=float(model_cfg["max_grad_norm"]),
        dtype=dtype,
        key=model_key,
    )
    if model.action_dim >= 20:
        tdmpc_cfg["mppi_iterations"] += 2

    return TDMPC2.create(
        world_model=model,
        horizon=int(tdmpc_cfg["horizon"]),
        mppi_iterations=int(tdmpc_cfg["mppi_iterations"]),
        population_size=int(tdmpc_cfg["population_size"]),
        policy_prior_samples=int(tdmpc_cfg["policy_prior_samples"]),
        num_elites=int(tdmpc_cfg["num_elites"]),
        min_plan_std=float(tdmpc_cfg["min_plan_std"]),
        max_plan_std=float(tdmpc_cfg["max_plan_std"]),
        temperature=float(tdmpc_cfg["temperature"]),
        discount=float(tdmpc_cfg["discount"]),
        batch_size=int(tdmpc_cfg["batch_size"]),
        rho=float(tdmpc_cfg["rho"]),
        consistency_loss_scale=float(tdmpc_cfg["consistency_loss_scale"]),
        reward_loss_scale=float(tdmpc_cfg["reward_loss_scale"]),
        value_loss_scale=float(tdmpc_cfg["value_loss_scale"]),
        continue_loss_scale=float(tdmpc_cfg["continue_loss_scale"]),
        entropy_coef=float(tdmpc_cfg["entropy_coef"]),
        tau=float(tdmpc_cfg["tau"]),
    )
