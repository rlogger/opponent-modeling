"""P07/P10/A01 frozen diagnostics; simulator state is never inferred from a model.

The observation-only imagined-state template below is not a simulator state
claim. It is never stepped. All real branches use the complete supplied state.
"""
from __future__ import annotations

import numpy as np


def discounted_rewards(reward, valid, discount):
    reward, valid = np.asarray(reward), np.asarray(valid, bool)
    if reward.shape != valid.shape or reward.ndim != 2 or not 0 <= discount <= 1:
        raise ValueError("aligned candidate-by-time rewards/masks and bounded discount required")
    if not np.isfinite(reward).all():
        raise ValueError("nonfinite rewards")
    return (np.where(valid, reward, 0) * discount ** np.arange(reward.shape[1])).sum(-1)


def ranking_agreement(predicted, realized):
    """Tie-aware pairwise ordering, with uninformative true-return ties excluded."""
    predicted, realized = np.asarray(predicted), np.asarray(realized)
    if predicted.ndim != 1 or predicted.shape != realized.shape or len(predicted) < 2:
        raise ValueError("two aligned candidate return vectors required")
    if not np.isfinite(predicted).all() or not np.isfinite(realized).all():
        raise ValueError("nonfinite candidate returns")
    left, right = np.triu_indices(len(predicted), 1)
    true_sign = np.sign(realized[left] - realized[right])
    pred_sign = np.sign(predicted[left] - predicted[right])
    informative = true_sign != 0
    agreement = np.where(pred_sign == true_sign, 1., np.where(pred_sign == 0, .5, 0.))
    selected = int(np.argmax(predicted))
    return {"pairwise_agreement": float(agreement[informative].mean()) if informative.any() else None,
            "informative_pairs": int(informative.sum()), "true_tied_pairs": int((~informative).sum()),
            "selected_candidate": selected, "selected_realized_return": float(realized[selected]),
            "realized_best_return": float(realized.max()),
            "selection_regret": float(realized.max() - realized[selected])}


def physical_invariants(states, valid, tolerance=1e-5):
    """Known66D static/event/clock contracts; outside-arena position is legal."""
    states, valid = np.asarray(states), np.asarray(valid, bool)
    if states.shape != (len(valid), valid.shape[1] + 1, 66):
        raise ValueError("aligned66D predicted states and transition masks required")
    current, following = states[:, :-1], states[:, 1:]
    static = np.r_[8:40, 56:65]
    flags = following[..., 40:56]
    violations = {
        "static_geometry": np.any(np.abs(following[..., static] - states[:, 0][:, None, static]) > tolerance, axis=-1),
        "nonbinary_collected": np.any(np.minimum(np.abs(flags), np.abs(flags - 1)) > tolerance, axis=-1),
        "decreasing_collected": np.any(flags < current[..., 40:56] - tolerance, axis=-1),
        "invalid_radii": np.any(following[..., 62:65] <= 0, axis=-1),
        "wrong_clock_increment": np.abs(following[..., 65] - current[..., 65] - .01) > tolerance,
        "nonfinite": ~np.isfinite(following).all(axis=-1),
    }
    denominator = int(valid.sum())
    return {"valid_predicted_transitions": denominator,
            "counts": {name: int((value & valid).sum()) for name, value in violations.items()},
            "rates": {name: float(value[valid].mean()) if denominator else None for name, value in violations.items()}}


def _repeat_state(state, count):
    import jax
    import jax.numpy as jnp
    return jax.tree.map(lambda x: jnp.broadcast_to(x, (count, *np.shape(x))), state)


def specialist_on_imagined_observation(env, template, physical, red_params):
    """Query exact frozen policy at imagined fields, without stepping this object.

    Preserve predicted flags/radii as floats rather than projecting away model
    violations. get_obs uses them only to construct observations. The unknown
    simulator bookkeeping remains the complete prefix template and is unused
    by this predator's observation function.
    """
    import jax
    import jax.numpy as jnp

    from mopa.continuous_data import deterministic_specialist_action

    physical = jnp.asarray(physical)
    repeated = _repeat_state(template, len(physical))
    observation_only = repeated.replace(
        p_pos=repeated.p_pos.at[:, :2].set(physical[:, :4].reshape(-1, 2, 2)),
        p_vel=repeated.p_vel.at[:, :2].set(physical[:, 4:8].reshape(-1, 2, 2)),
        resource_pos=physical[:, 8:40].reshape(-1, 16, 2), collected=physical[:, 40:56],
        lava_pos=physical[:, 56:62].reshape(-1, 3, 2), lava_rad=physical[:, 62:65],
        step=physical[:, 65] * env.max_steps)
    observations = jax.vmap(env.get_obs)(observation_only)
    return deterministic_specialist_action(red_params, observations[env.adversaries[0]], 35)


def simulator_candidates(env, complete_state, actions, red_params, step_seed):
    """Run every fixed blue command sequence from one identical complete state."""
    import jax
    import jax.numpy as jnp

    from mopa.continuous_data import deterministic_specialist_action, markov_state
    from tag_objectives import joint_action_dict
    from tag_objectives.teams import freeze_tree

    actions = np.asarray(actions, np.float32)
    if actions.ndim != 3 or actions.shape[-1] != 2 or not np.isfinite(actions).all() or np.any(np.abs(actions) > 1):
        raise ValueError("bounded candidate-by-horizon blue actions required")
    n, horizon, _ = actions.shape
    state = _repeat_state(complete_state, n)
    observations = jax.vmap(env.get_obs)(state)
    step = jax.jit(jax.vmap(env.step_env))
    done = np.asarray(state.done).any(-1)
    states, red_actions, rewards, valid, captures, continues = [np.asarray(markov_state(env, state))], [], [], [], [], []
    start = int(np.asarray(complete_state.step))
    for t in range(horizon):
        active = ~done
        red = deterministic_specialist_action(red_params, observations[env.adversaries[0]], 35)
        command = joint_action_dict(env, jnp.asarray(actions[:, t]), red[:, None])
        key = jax.random.fold_in(jnp.asarray(step_seed, jnp.uint32), start + t)
        new_obs, new_state, reward, _, info = step(jnp.broadcast_to(key, (n, 2)), state, command)
        captured = np.asarray(info["captured"][:, 1], bool)
        new_done = np.asarray(new_state.done).any(-1)
        valid.append(active)
        red_actions.append(np.where(active[:, None], red, 0))
        rewards.append(np.where(active, reward[env.good_agents[0]], 0))
        captures.append(active & captured)
        continues.append(active & ~new_done)
        state = freeze_tree(jnp.asarray(active), new_state, state)
        observations = freeze_tree(jnp.asarray(active), new_obs, observations)
        states.append(np.asarray(markov_state(env, state)))
        done |= new_done
    return {"state": np.stack(states, 1), "red_action": np.stack(red_actions, 1),
            "reward": np.stack(rewards, 1), "valid": np.stack(valid, 1),
            "captured": np.stack(captures, 1), "continues": np.stack(continues, 1),
            "complete_final_state": state}


def simulator_policy_tail(agent, env, complete_states, red_params, context, step_seed,
                          start_step, remaining_steps, *, key):
    """One real finite-task tail sample under the same frozen-context prior.

    This measures a declared conditional continuation, not the online-context
    closed-loop controller. Shared action noise pairs candidates. The first
    action uses the same tail sampling key as factored estimate_value.
    """
    import jax
    import jax.numpy as jnp

    from mopa.continuous_data import deterministic_specialist_action, markov_state
    from tag_objectives import joint_action_dict
    from tag_objectives.teams import freeze_tree

    state = complete_states
    n = len(state.step)
    context = jnp.broadcast_to(context, (n, agent.model.context_dim))
    observations = jax.vmap(env.get_obs)(state)
    step = jax.jit(jax.vmap(env.step_env))
    key, _ = jax.random.split(key)
    action_key, _ = jax.random.split(key)
    rewards, masks = [], []
    for t in range(remaining_steps):
        active = ~np.asarray(state.done).any(-1)
        raw = markov_state(env, state)
        x = agent.model.encode(raw, agent.model.encoder.params, jax.random.PRNGKey(0))
        use_key = action_key if t == 0 else jax.random.fold_in(action_key, t)
        blue = agent.model.sample_actions(agent.model.policy_inputs(x, context),
                    agent.model.policy_model.params, deterministic=False, key=use_key)[0]
        red = deterministic_specialist_action(red_params, observations[env.adversaries[0]], 35)
        rng = jax.random.fold_in(jnp.asarray(step_seed, jnp.uint32), start_step + t)
        new_obs, new_state, reward, _, _ = step(jnp.broadcast_to(rng, (n, 2)), state,
                                               joint_action_dict(env, blue, red[:, None]))
        rewards.append(np.where(active, reward[env.good_agents[0]], 0))
        masks.append(active)
        state = freeze_tree(jnp.asarray(active), new_state, state)
        observations = freeze_tree(jnp.asarray(active), new_obs, observations)
    if not rewards:
        return {"reward": np.zeros((n, 0)), "valid": np.zeros((n, 0), bool), "return": np.zeros(n)}
    reward, valid = np.stack(rewards, 1), np.stack(masks, 1)
    return {"reward": reward, "valid": valid,
            "return": discounted_rewards(reward, valid, agent.discount)}


def imagined_candidates(agent, initial, actions, context, mean, std, red_source, *, key):
    """Same learned world/Q/policy and fixed context; vary only red action source.

    red_source(raw_state, fixed_context, step) may query learned/BC/oracle policy
    or inject actions from matched real branches. Tail keys match deployed
    factored estimate_value, including its reserved opponent-key split.
    Implicit dynamics requires no red source and supplies only schema placeholders
    for red actions. All saved evidence must be finite, including inactive tails;
    that diagnostic check is stricter than masking inactive planner contributions.
    """
    import jax
    import jax.numpy as jnp

    model = agent.model
    if model.opponent_mode not in {"implicit", "factored"} or model.encoder_type != "identity" or model.latent_dim != 66:
        raise ValueError("diagnostic requires an implicit or factored66D identity world model")
    if model.opponent_mode == "factored" and red_source is None:
        raise ValueError("factored diagnostics require an explicit red action source")
    actions = jnp.asarray(actions)
    n, horizon, _ = actions.shape
    initial = jnp.broadcast_to(jnp.asarray(initial), (n, 66))
    context = jnp.broadcast_to(jnp.asarray(context), (n, model.context_dim))
    x = model.encode(initial, model.encoder.params, jax.random.PRNGKey(0))
    states, red_actions, rewards, probabilities, masks = [np.asarray(initial)], [], [], [], []
    alive = (np.asarray(model.known_continuation(x)).copy() if model.transition_contract != "none"
             else np.ones(n, bool))
    if model.opponent_mode == "factored":
        key, _ = jax.random.split(key)
    for t in range(horizon):
        raw = x * std + mean
        # Implicit dynamics has no explicit red prediction. Zeros only preserve
        # the array schema; callers must not interpret them as an action model.
        red = (jnp.asarray(red_source(raw, context, t)) if model.opponent_mode == "factored"
               else jnp.zeros((n, model.action_dim)))
        transition = model.transition_inputs(actions[:, t], context, red)
        reward, _ = model.reward(x, transition, model.reward_model.params)
        continuation = (jax.nn.sigmoid(model.continue_logits(x, transition, model.continue_model.params))
                        if model.predict_continues else jnp.ones(n))
        x = model.next(x, transition, model.dynamics_model.params)
        if model.transition_contract != "none":
            continuation *= model.known_continuation(x)
        masks.append(alive.copy())
        states.append(np.asarray(x * std + mean))
        red_actions.append(np.asarray(red))
        rewards.append(np.asarray(reward))
        probabilities.append(np.asarray(continuation))
        alive &= np.asarray(continuation) > .5
    action_key, value_key = jax.random.split(key)
    tail_action = model.sample_actions(model.policy_inputs(x, context), model.policy_model.params,
                                      deterministic=False, key=action_key)[0]
    q, _ = model.Q(x, model.value_inputs(tail_action, context), model.value_model.params, value_key)
    reward = np.stack(rewards, 1)
    valid = np.stack(masks, 1)
    finite_return = discounted_rewards(reward, valid, agent.discount)
    tail = agent.discount ** horizon * alive * np.asarray(q).mean(0)
    if model.transition_contract != "none":
        tail = np.where(alive, tail, 0.)
    result = {"state": np.stack(states, 1), "red_action": np.stack(red_actions, 1),
              "reward": reward, "continues_probability": np.stack(probabilities, 1),
              "valid": valid, "finite_return": finite_return, "tail": tail,
              "planner_return": finite_return + tail}
    if not all(np.isfinite(v).all() for v in result.values()):
        raise FloatingPointError("nonfinite imagined evidence; retain failed diagnostic")
    return result



def imagined_policy_commands(agent, initial, context, mean, std, remaining_steps, *, key):
    """A01 full remainder commands from the frozen world's deterministic prior.

    Generate at the learned world's own imagined states with fixed original
    context and the installed learned red head. No simulator or alternative
    opponent source enters command generation. Later predictions after a hard
    predicted stop are explicitly extrapolation. Nonfinite arithmetic leaves a
    failed, partial record rather than substituting invented commands.
    """
    import jax
    import jax.numpy as jnp

    if not 1 <= remaining_steps <= 100:
        raise ValueError("remaining physical budget must lie in1..100")
    model = agent.model
    if model.opponent_mode != "factored" or model.encoder_type != "identity" or model.latent_dim != 66:
        raise ValueError("full remainder requires frozen factored66D identity world model")
    x = model.encode(jnp.asarray(initial)[None], model.encoder.params, jax.random.PRNGKey(0))
    context = jnp.asarray(context).reshape(1, model.context_dim)
    states, actions, red_actions, probabilities, valid = [np.asarray(initial)], [], [], [], []
    alive, status = True, "complete"
    for t in range(remaining_steps):
        blue = model.sample_actions(model.policy_inputs(x, context), model.policy_model.params,
                    deterministic=True, key=jax.random.fold_in(key, t))[0]
        red = model.red_action(x, context, model.red_model.params)
        transition = model.transition_inputs(blue, context, red)
        probability = (jax.nn.sigmoid(model.continue_logits(x, transition, model.continue_model.params))
                       if model.predict_continues else jnp.ones(1))
        following = model.next(x, transition, model.dynamics_model.params)
        raw = np.asarray(following * std + mean)[0]
        if not all(np.isfinite(v).all() for v in (blue, red, probability, raw)):
            status = "failed_nonfinite_generation"
            break
        actions.append(np.asarray(blue)[0])
        red_actions.append(np.asarray(red)[0])
        probabilities.append(float(probability[0]))
        valid.append(alive)
        states.append(raw)
        alive = alive and bool(probability[0] > .5)
        x = following
    return {"status": status, "requested_steps": remaining_steps, "generated_steps": len(actions),
            "state": np.asarray(states), "blue_action": np.asarray(actions).reshape(-1, 2),
            "red_action": np.asarray(red_actions).reshape(-1, 2),
            "continues_probability": np.asarray(probabilities), "valid": np.asarray(valid, bool)}

def compare_candidates(predicted, actual, std, discount, *, capture_distance, arena, actual_tail=None):
    valid = np.asarray(actual["valid"], bool)
    both = valid & np.asarray(predicted["valid"], bool)
    state_difference = predicted["state"][:, 1:] - actual["state"][:, 1:]
    after = predicted["state"][:, 1:]
    true_after = actual["state"][:, 1:]
    predicted_capture = np.linalg.norm(after[..., :2] - after[..., 2:4], axis=-1) < capture_distance
    predicted_boundary = (np.abs(after[..., 2:4]) > arena).any(-1)
    actual_boundary = (np.abs(true_after[..., 2:4]) > arena).any(-1)

    def average(values, mask=both):
        return float(np.asarray(values)[mask].mean()) if np.any(mask) else None

    realized_return = discounted_rewards(actual["reward"], valid, discount)

    def ranking(prediction, outcome):
        return ranking_agreement(prediction, outcome) if len(outcome) >= 2 else {"status": "one supplied trajectory; no candidate ranking comparison"}
    result = {
        "valid_real_transitions": int(valid.sum()), "valid_in_both": int(both.sum()),
        "red_action_squared_2d_error": average(np.square(predicted["red_action"] - actual["red_action"]).sum(-1)),
        "state_normalized_mse": average(np.square(state_difference / std).mean(-1)),
        "position_rmse": float(np.sqrt(average(np.square(state_difference[..., :4]).mean(-1)))) if both.any() else None,
        "reward_mse": average(np.square(predicted["reward"] - actual["reward"])),
        "capture_geometry_error_rate": average(predicted_capture != actual["captured"]),
        "boundary_error_rate": average(predicted_boundary != actual_boundary),
        "continuation_brier": average(np.square(predicted["continues_probability"] - actual["continues"])),
        "false_predicted_stop_fraction": average(~predicted["valid"], valid),
        "finite_return_mse": float(np.square(predicted["finite_return"] - realized_return).mean()),
        "finite_return_bias": float((predicted["finite_return"] - realized_return).mean()),
        "tail_mean": float(np.asarray(predicted["tail"]).mean()),
        "ranking_finite_horizon": ranking(predicted["finite_return"], realized_return),
        "ranking_with_Q_tail_against_finite_outcome": ranking(predicted["planner_return"], realized_return),
        "Q_comparison_limit": "Tail-augmented value and finite-horizon simulator outcome have different endpoints; no tail-accuracy claim",
        "invariants": physical_invariants(predicted["state"], predicted["valid"]),
    }
    if actual_tail is not None:
        horizon = valid.shape[1]
        realized_full = realized_return + discount ** horizon * actual_tail
        result.update({
            "ranking_full_finite_task": ranking(predicted["planner_return"], realized_full),
            "full_finite_task_return_mse": float(np.square(predicted["planner_return"] - realized_full).mean()),
            "full_finite_task_return_bias": float((predicted["planner_return"] - realized_full).mean()),
            "discounted_tail_mse": float(np.square(predicted["tail"] - discount ** horizon * actual_tail).mean()),
            "Q_tail_reference": "one real stochastic-policy tail sample with prefix context fixed; no online inference in this reference"})
    return result


def frozen_prefix(env, agent, opponent, red_params, reset_key, step_seed, length, policy_key):
    """Observed trajectory-local prefix; captures are retained without replacement."""
    import jax
    import jax.numpy as jnp

    from mopa.continuous_data import deterministic_specialist_action, markov_state
    from tag_objectives import joint_action_dict

    observations, state = env.reset(jnp.asarray(reset_key, jnp.uint32))
    carry = opponent.initial_context(1)
    states, actions, red_actions, rewards = [np.asarray(markov_state(env, state))], [], [], []
    for t in range(length):
        if np.asarray(state.done).any():
            break
        raw = markov_state(env, state)[None]
        blue, _ = agent.act(raw, mpc=False, deterministic=True, train=False,
                            context=carry.context, key=jax.random.fold_in(policy_key, t))
        red = deterministic_specialist_action(red_params, observations[env.adversaries[0]][None], 35)
        next_obs, next_state, reward, _, _ = env.step_env(
            jax.random.fold_in(jnp.asarray(step_seed, jnp.uint32), t), state,
            {name: value[0] for name, value in joint_action_dict(env, blue, red[:, None]).items()})
        carry = opponent.update_context(carry, raw, red, jnp.array([True]))
        actions.append(np.asarray(blue[0]))
        red_actions.append(np.asarray(red[0]))
        rewards.append(float(reward[env.good_agents[0]]))
        observations, state = next_obs, next_state
        states.append(np.asarray(markov_state(env, state)))
    return {"complete_state": state, "context_carry": carry, "state": np.asarray(states),
            "blue_action": np.asarray(actions).reshape(-1, 2), "red_action": np.asarray(red_actions).reshape(-1, 2),
            "reward": np.asarray(rewards), "requested_length": length, "valid_length": len(rewards)}


def closed_loop_suffix(env, agent, opponent, red_params, complete_state, context_carry,
                       step_seed, policy_key, *, mpc):
    """Real policy-only/MPPI branch; inference updates, all model weights frozen."""
    import time

    import jax
    import jax.numpy as jnp

    from mopa.continuous_data import deterministic_specialist_action, markov_state
    from tag_objectives import joint_action_dict

    state, carry, plan = complete_state, context_carry, None
    observations = env.get_obs(state)
    states, actions, red_actions, rewards, seconds, contexts = [np.asarray(markov_state(env, state))], [], [], [], [], []
    while not np.asarray(state.done).any():
        t = int(np.asarray(state.step))
        raw = markov_state(env, state)[None]
        started = time.monotonic()
        blue, plan = agent.act(raw, prev_plan=plan, mpc=mpc, deterministic=True, train=False,
                               context=carry.context, key=jax.random.fold_in(policy_key, t))
        blue = np.asarray(blue)
        seconds.append(time.monotonic() - started)
        red = deterministic_specialist_action(red_params, observations[env.adversaries[0]][None], 35)
        next_obs, next_state, reward, _, _ = env.step_env(
            jax.random.fold_in(jnp.asarray(step_seed, jnp.uint32), t), state,
            {name: value[0] for name, value in joint_action_dict(env, blue, red[:, None]).items()})
        contexts.append(np.asarray(carry.context[0]))
        carry = opponent.update_context(carry, raw, red, jnp.array([True]))
        actions.append(blue[0])
        red_actions.append(np.asarray(red[0]))
        rewards.append(float(reward[env.good_agents[0]]))
        observations, state = next_obs, next_state
        states.append(np.asarray(markov_state(env, state)))
    return {"state": np.asarray(states), "blue_action": np.asarray(actions).reshape(-1, 2),
            "red_action": np.asarray(red_actions).reshape(-1, 2), "reward": np.asarray(rewards),
            "context": np.asarray(contexts).reshape(-1, agent.model.context_dim),
            "final_context": np.asarray(carry.context[0]), "controller_seconds": np.asarray(seconds),
            "captured": bool(np.asarray(state.capture_t) >= 0), "resources_collected": int(np.asarray(state.collected).sum())}
