"""Two-team MAPPO (CTDE) for the objective-typed predator env.

Each team (pred / prey) has a shared actor-critic pair. Objectives-only:
builds SimpleTagObjectivesMPE from config (constructor defaults are SoT).

Run from repo root:
    python scripts/train_mappo.py alg=mappo_objectives_capture NUM_SEEDS=3

``ACTION_TYPE: Continuous`` switches both teams to a two-dimensional
tanh-squashed diagonal Gaussian (``mopa.nets.ContinuousActor``). PPO then uses
the joint log-probability summed over action dimensions with the tanh
change-of-variables correction (``mopa.continuous``), stores the pre-squash
sample ``u`` for exact ratio computation, and converts ``tanh(u)`` to the
JaxMARL ``[0, 1]^5`` vector only at the environment boundary
(``tag_objectives.to_mpe_action``).
"""
from __future__ import annotations

import copy
import os
import sys
import time
import zlib
from pathlib import Path
from typing import Dict, List, NamedTuple

import distrax
import flax.linen as nn
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from jaxmarl.wrappers.baselines import MPELogWrapper
from omegaconf import OmegaConf

# Hydra changes CWD; keep repo-root configs reachable.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mopa.continuous import (  # noqa: E402
    gaussian_entropy,
    tanh_gaussian_log_prob,
    tanh_gaussian_sample,
)
from mopa.nets import ContinuousActor  # noqa: E402
from tag_objectives import (  # noqa: E402
    CONTINUOUS_ACTION_DIM,
    SimpleTagObjectivesMPE,
    to_mpe_action,
)

_CONFIG_DIR = str(_REPO_ROOT / "configs")
CONTINUOUS = "Continuous"
DISCRETE = "Discrete"


class Actor(nn.Module):
    action_dim: int
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, obs):
        x = nn.Dense(
            self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        x = nn.relu(x)
        x = nn.Dense(
            self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        x = nn.relu(x)
        logits = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(x)
        return distrax.Categorical(logits=logits)


class Critic(nn.Module):
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, world_state):
        x = nn.Dense(
            self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(world_state)
        x = nn.relu(x)
        x = nn.Dense(
            self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        x = nn.relu(x)
        v = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        return jnp.squeeze(v, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    world_state: jnp.ndarray


def split_teams(env) -> Dict[str, List[str]]:
    preds = [a for a in env.agents if a.startswith("adversary")]
    prey = [a for a in env.agents if a.startswith("agent")]
    assert preds and prey
    return {"pred": preds, "prey": prey}


def make_train(config, env):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    teams = split_teams(env)
    team_names = list(teams.keys())
    all_agents = env.agents

    obs_sizes = {a: env.observation_space(a).shape[0] for a in all_agents}
    max_obs = max(obs_sizes.values())
    continuous = config.get("ACTION_TYPE", DISCRETE) == CONTINUOUS
    if continuous:
        # Learned/planned action is (a_x, a_y) in [-1, 1]^2; the env receives
        # the redundant [0, 1]^5 vector via to_mpe_action at the boundary.
        action_dim = CONTINUOUS_ACTION_DIM
    else:
        action_dim = env.action_space(all_agents[0]).n
    world_state_dim = sum(obs_sizes[a] for a in all_agents)

    def pad_obs(obs_dict):
        out = {}
        for a in all_agents:
            o = obs_dict[a]
            if o.shape[-1] < max_obs:
                pad_width = max_obs - o.shape[-1]
                o = jnp.concatenate(
                    [o, jnp.zeros(o.shape[:-1] + (pad_width,))], axis=-1
                )
            out[a] = o
        return out

    def get_world_state(obs_dict):
        return jnp.concatenate([obs_dict[a] for a in all_agents], axis=-1)

    def batchify_team(x: dict, agents: List[str]):
        return jnp.stack([x[a] for a in agents], axis=0)

    clip_eps = config["CLIP_EPS"]

    def linear_schedule(count):
        frac = (
            1.0
            - (
                count
                // (
                    config.get("NUM_MINIBATCHES", 1)
                    * config.get("UPDATE_EPOCHS", 4)
                )
            )
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        original_seed = rng[0]
        rng, _rng = jax.random.split(rng)

        actor_cls = ContinuousActor if continuous else Actor
        actors = {
            t: actor_cls(action_dim=action_dim, hidden_dim=config["HIDDEN_SIZE"])
            for t in team_names
        }
        critics = {
            t: Critic(hidden_dim=config["HIDDEN_SIZE"]) for t in team_names
        }

        def create_train_states(rng):
            states = {}
            for t in team_names:
                rng, k_a, k_c = jax.random.split(rng, 3)
                dummy_obs = jnp.zeros((1, max_obs))
                dummy_ws = jnp.zeros((1, world_state_dim))
                actor_params = actors[t].init(k_a, dummy_obs)
                critic_params = critics[t].init(k_c, dummy_ws)

                if config.get("ANNEAL_LR", True):
                    actor_tx = optax.chain(
                        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                        optax.adam(learning_rate=linear_schedule, eps=1e-5),
                    )
                    critic_tx = optax.chain(
                        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                        optax.adam(learning_rate=linear_schedule, eps=1e-5),
                    )
                else:
                    actor_tx = optax.chain(
                        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                        optax.adam(config["LR"], eps=1e-5),
                    )
                    critic_tx = optax.chain(
                        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                        optax.adam(config["LR"], eps=1e-5),
                    )

                states[t] = (
                    TrainState.create(
                        apply_fn=actors[t].apply, params=actor_params, tx=actor_tx
                    ),
                    TrainState.create(
                        apply_fn=critics[t].apply, params=critic_params, tx=critic_tx
                    ),
                )
            return states

        train_states = create_train_states(rng)

        rng, _rng = jax.random.split(rng)
        reset_rngs = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset)(reset_rngs)

        def _update_step(runner_state, _):
            train_states, env_state, last_obs, rng = runner_state

            def _env_step(carry, _):
                train_states, env_state, last_obs, rng = carry
                rng, rng_act, rng_step = jax.random.split(rng, 3)

                padded = pad_obs(last_obs)
                ws = get_world_state(last_obs)

                all_actions = {}  # what the env receives
                stored_actions = {}  # what PPO stores (pre-tanh u if continuous)
                all_log_probs = {}
                all_values = {}
                for t in team_names:
                    ags = teams[t]
                    obs_t = batchify_team(padded, ags)
                    ws_t = jnp.broadcast_to(ws[None], (len(ags),) + ws.shape)

                    actor_state, critic_state = train_states[t]
                    rng, k = jax.random.split(rng)
                    keys = jax.random.split(k, len(ags) * config["NUM_ENVS"]).reshape(
                        len(ags), config["NUM_ENVS"], 2
                    )
                    if continuous:
                        mean_t, log_std_t = actors[t].apply(actor_state.params, obs_t)
                        u_t, a_t = jax.vmap(jax.vmap(tanh_gaussian_sample))(
                            keys, mean_t, log_std_t
                        )
                        lp_t = tanh_gaussian_log_prob(mean_t, log_std_t, u_t)
                        env_actions_t = to_mpe_action(a_t)
                        stored_t = u_t
                    else:
                        pi = jax.vmap(
                            jax.vmap(actors[t].apply, in_axes=(None, 0)),
                            in_axes=(None, 0),
                        )(actor_state.params, obs_t)
                        actions_t = jax.vmap(
                            jax.vmap(lambda pi, k: pi.sample(seed=k))
                        )(pi, keys)
                        lp_t = jax.vmap(jax.vmap(lambda pi, a: pi.log_prob(a)))(
                            pi, actions_t
                        )
                        env_actions_t = actions_t
                        stored_t = actions_t

                    v_t = jax.vmap(
                        jax.vmap(critics[t].apply, in_axes=(None, 0)),
                        in_axes=(None, 0),
                    )(critic_state.params, ws_t)

                    for i, a in enumerate(ags):
                        all_actions[a] = env_actions_t[i]
                        stored_actions[a] = stored_t[i]
                        all_log_probs[a] = lp_t[i]
                        all_values[a] = v_t[i]

                step_rngs = jax.random.split(rng_step, config["NUM_ENVS"])
                new_obs, new_env_state, rewards, dones, infos = jax.vmap(env.step)(
                    step_rngs, env_state, all_actions
                )

                transition_per_team = {}
                for t in team_names:
                    ags = teams[t]
                    transition_per_team[t] = Transition(
                        done=batchify_team(dones, ags),
                        action=batchify_team(stored_actions, ags),
                        value=batchify_team(all_values, ags),
                        reward=batchify_team(rewards, ags),
                        log_prob=batchify_team(all_log_probs, ags),
                        obs=batchify_team(padded, ags),
                        world_state=jnp.broadcast_to(ws[None], (len(ags),) + ws.shape),
                    )

                return (train_states, new_env_state, new_obs, rng), (
                    transition_per_team,
                    infos,
                )

            rng, _rng = jax.random.split(rng)
            (train_states, env_state, last_obs, rng), (traj_batch, infos) = jax.lax.scan(
                _env_step,
                (train_states, env_state, last_obs, _rng),
                None,
                config["NUM_STEPS"],
            )

            ws_last = get_world_state(last_obs)

            def _update_team(t, actor_state, critic_state, traj):
                ags = teams[t]
                n_agents = len(ags)

                ws_last_t = jnp.broadcast_to(
                    ws_last[None], (n_agents,) + ws_last.shape
                )
                last_val = jax.vmap(
                    jax.vmap(critics[t].apply, in_axes=(None, 0)),
                    in_axes=(None, 0),
                )(critic_state.params, ws_last_t)

                def _gae(traj, last_val):
                    def _step(carry, transition):
                        gae, next_val = carry
                        delta = (
                            transition.reward
                            + config["GAMMA"] * next_val * (1 - transition.done)
                            - transition.value
                        )
                        gae = (
                            delta
                            + config["GAMMA"]
                            * config["GAE_LAMBDA"]
                            * (1 - transition.done)
                            * gae
                        )
                        return (gae, transition.value), gae

                    _, advantages = jax.lax.scan(
                        _step,
                        (jnp.zeros_like(last_val), last_val),
                        traj,
                        reverse=True,
                    )
                    return advantages, advantages + traj.value

                advantages, targets = _gae(traj, last_val)

                def _ppo_epoch(carry, _):
                    actor_state, critic_state, rng = carry
                    rng, perm_key = jax.random.split(rng)

                    T = config["NUM_STEPS"]
                    N = config["NUM_ENVS"]
                    flat_size = n_agents * T * N

                    flat_traj = jax.tree.map(
                        lambda x: (
                            x.reshape(flat_size, *x.shape[3:])
                            if x.ndim > 3
                            else x.reshape(flat_size)
                        ),
                        traj,
                    )
                    flat_adv = advantages.reshape(flat_size)
                    flat_tgt = targets.reshape(flat_size)

                    num_mb = config.get("NUM_MINIBATCHES", 4)
                    mb_size = flat_size // num_mb
                    perm = jax.random.permutation(perm_key, flat_size)

                    def _update_mb(carry, idx_start):
                        actor_state, critic_state = carry
                        mb_idx = jax.lax.dynamic_slice(perm, (idx_start,), (mb_size,))
                        mb_traj = jax.tree.map(lambda x: x[mb_idx], flat_traj)
                        mb_adv = flat_adv[mb_idx]
                        mb_tgt = flat_tgt[mb_idx]

                        mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                        def actor_loss_fn(params):
                            if continuous:
                                mean, log_std = actors[t].apply(params, mb_traj.obs)
                                # Joint log-probability: summed over the two action
                                # dimensions, with the tanh Jacobian correction, at
                                # the stored pre-squash sample u.
                                log_prob = tanh_gaussian_log_prob(
                                    mean, log_std, mb_traj.action
                                )
                                entropy = gaussian_entropy(log_std).mean()
                            else:
                                pi = actors[t].apply(params, mb_traj.obs)
                                log_prob = pi.log_prob(mb_traj.action)
                                entropy = pi.entropy().mean()
                            ratio = jnp.exp(log_prob - mb_traj.log_prob)
                            l1 = ratio * mb_adv
                            l2 = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * mb_adv
                            pg_loss = -jnp.minimum(l1, l2).mean()
                            return pg_loss - config.get("ENT_COEF", 0.01) * entropy, (
                                pg_loss,
                                entropy,
                            )

                        def critic_loss_fn(params):
                            v = critics[t].apply(params, mb_traj.world_state)
                            v_clipped = mb_traj.value + (v - mb_traj.value).clip(
                                -clip_eps, clip_eps
                            )
                            vl1 = jnp.square(v - mb_tgt)
                            vl2 = jnp.square(v_clipped - mb_tgt)
                            return (
                                0.5
                                * jnp.maximum(vl1, vl2).mean()
                                * config.get("VF_COEF", 0.5)
                            )

                        (a_loss, (pg_loss, entropy)), a_grads = jax.value_and_grad(
                            actor_loss_fn, has_aux=True
                        )(actor_state.params)
                        c_loss, c_grads = jax.value_and_grad(critic_loss_fn)(
                            critic_state.params
                        )

                        actor_state = actor_state.apply_gradients(grads=a_grads)
                        critic_state = critic_state.apply_gradients(grads=c_grads)
                        return (actor_state, critic_state), {
                            "actor_loss": a_loss,
                            "critic_loss": c_loss,
                            "entropy": entropy,
                        }

                    starts = jnp.arange(num_mb) * mb_size
                    (actor_state, critic_state), loss_info = jax.lax.scan(
                        _update_mb, (actor_state, critic_state), starts
                    )
                    return (actor_state, critic_state, rng), loss_info

                rng_team = jax.random.fold_in(rng, zlib.crc32(t.encode()))
                (actor_state, critic_state, _), loss_info = jax.lax.scan(
                    _ppo_epoch,
                    (actor_state, critic_state, rng_team),
                    None,
                    config.get("UPDATE_EPOCHS", 4),
                )
                return actor_state, critic_state, jax.tree.map(lambda x: x.mean(), loss_info)

            new_train_states = {}
            metrics = {}
            for t in team_names:
                actor_s, critic_s = train_states[t]
                a_s, c_s, t_loss = _update_team(t, actor_s, critic_s, traj_batch[t])
                new_train_states[t] = (a_s, c_s)
                for k, v in t_loss.items():
                    metrics[f"{t}/{k}"] = v

            flat_infos = jax.tree.map(lambda x: x.mean(), infos)
            metrics.update({f"all/{k}": v for k, v in flat_infos.items()})
            n_agents = len(env.agents)
            for t in team_names:
                idxs = jnp.array([env.agents.index(a) for a in teams[t]])

                def _team_mean(x, idxs=idxs):
                    if x.ndim >= 1 and x.shape[-1] == n_agents:
                        return x[..., idxs].mean()
                    return x.mean()

                tm = jax.tree.map(_team_mean, infos)
                metrics.update({f"{t}/{k}": v for k, v in tm.items()})

            if config["WANDB_MODE"] != "disabled":

                def cb(m, seed):
                    wandb.log(m)

                jax.debug.callback(cb, metrics, original_seed)

            return (new_train_states, env_state, last_obs, rng), metrics

        rng, _rng = jax.random.split(rng)
        runner_state = (train_states, env_state, obsv, _rng)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train


_OBJ_KEYS = {
    "PRED_TYPE": "pred_type",
    "ARENA": "arena",
    "NUM_ADVERSARIES": "num_adversaries",
    "NUM_RESOURCES": "num_resources",
    "M_NEAREST_RESOURCES": "m_nearest_resources",
    "NUM_LAVA": "num_lava",
    "N_NEAREST_LAVA": "n_nearest_lava",
    "LAVA_RADIUS_MIN": "lava_radius_min",
    "LAVA_RADIUS_MAX": "lava_radius_max",
    "FRAC_RESOURCES_NEAR_LAVA": "frac_resources_near_lava",
    "LAVA_PENALTY": "lava_penalty",
    "BASE_LAVA_PENALTY": "base_lava_penalty",
    "PREY_LAVA_PENALTY": "prey_lava_penalty",
    "NOVELTY_BONUS": "novelty_bonus",
    "CAPTURE_BONUS": "capture_bonus",
    "DENSE_CHASE_COEF": "dense_chase_coef",
    "GRID_SIZE": "grid_size",
    "MIN_START_DIST": "min_start_dist",
    "COLLECT_RADIUS": "collect_radius",
    "COLLECT_REWARD": "collect_reward",
    "PRED_MAX_SPEED": "pred_max_speed",
    "PREY_MAX_SPEED": "prey_max_speed",
    "PRED_ACCEL": "pred_accel",
    "PREY_ACCEL": "prey_accel",
    "OBSTACLE_POSITIONS": "obstacle_positions",
    "MAX_STEPS": "max_steps",
}


def env_from_config(config):
    if not config.get("USE_OBJECTIVES", False):
        raise ValueError("This trainer only supports USE_OBJECTIVES=True")
    _kw = {py: config[k] for k, py in _OBJ_KEYS.items() if k in config}
    action_type = config.get("ACTION_TYPE", DISCRETE)
    if action_type not in (DISCRETE, CONTINUOUS):
        raise ValueError(f"ACTION_TYPE must be {DISCRETE!r} or {CONTINUOUS!r}")
    env = SimpleTagObjectivesMPE(
        **_kw, action_type=action_type, **config["ENV_KWARGS"]
    )
    env = MPELogWrapper(env)
    return env, config["ENV_NAME"]


def single_run(config):
    config = {**config, **config["alg"]}
    print("Config:\n", OmegaConf.to_yaml(config))

    alg_name = config.get("ALG_NAME", "mappo_objectives")
    env, env_name = env_from_config(copy.deepcopy(config))

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=[alg_name.upper(), env_name.upper(), f"jax_{jax.__version__}"],
        name=f"{alg_name}_{env_name}",
        config=config,
        mode=config["WANDB_MODE"],
    )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])

    t0 = time.time()
    train_vjit = jax.jit(jax.vmap(make_train(config, env)))
    outs = jax.block_until_ready(train_vjit(rngs))
    dt = time.time() - t0
    print(f"Training done in {dt:.1f}s")

    if config.get("SAVE_PATH", None) is not None:
        from jaxmarl.wrappers.baselines import save_params

        save_dir = os.path.join(config["SAVE_PATH"], env_name)
        os.makedirs(save_dir, exist_ok=True)
        OmegaConf.save(
            config,
            os.path.join(
                save_dir, f'{alg_name}_{env_name}_seed{config["SEED"]}_config.yaml'
            ),
        )
        states = outs["runner_state"][0]
        for t in states:
            actor_s, critic_s = states[t]
            for i in range(config["NUM_SEEDS"]):
                a_params = jax.tree.map(lambda x: x[i], actor_s.params)
                c_params = jax.tree.map(lambda x: x[i], critic_s.params)
                save_params(
                    a_params,
                    os.path.join(
                        save_dir,
                        f'{alg_name}_{env_name}_{t}_actor_seed{config["SEED"]}_vmap{i}.safetensors',
                    ),
                )
                save_params(
                    c_params,
                    os.path.join(
                        save_dir,
                        f'{alg_name}_{env_name}_{t}_critic_seed{config["SEED"]}_vmap{i}.safetensors',
                    ),
                )

        metrics = outs["metrics"]
        metrics_np = jax.tree.map(lambda x: np.asarray(x), metrics)
        np.savez_compressed(
            os.path.join(
                save_dir, f'{alg_name}_{env_name}_seed{config["SEED"]}_metrics.npz'
            ),
            **{
                k.replace("/", "__"): v
                for k, v in metrics_np.items()
                if isinstance(v, np.ndarray)
            },
        )


@hydra.main(version_base=None, config_path=_CONFIG_DIR, config_name="config")
def main(config):
    config = OmegaConf.to_container(config, resolve=True)
    single_run(config)


if __name__ == "__main__":
    # Ensure editable package imports work when launched as a script.
    src = str(_REPO_ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    main()
