# Handoff: Continuous Opponent-Aware TD-MPC

## Snapshot and evidence boundary

- Base checkout when this handoff was written: `main` at `8c24db3`.
- TD-MPC is not implemented. It remains deferred in
  [`docs/STATUS.md`](docs/STATUS.md).
- Existing MAPPO specialists, rollout actions, and BC are discrete.
- Existing discrete checkpoints, datasets, `0s` results, and BC results cannot
  be presented as continuous-control results.
- Upstream TD-MPC2-JAX was inspected at commit
  `5b05ff452424896d709848e1f249bd67e269b8a1` from July 28, 2026.
- A smoke test proves code execution only. It is not evidence of TD-MPC,
  opponent controllability, or improved return.

## Chosen upstream implementation

Source: [ShaneFlandermeyer/tdmpc2-jax](https://github.com/ShaneFlandermeyer/tdmpc2-jax),
pinned to [`5b05ff4`](https://github.com/ShaneFlandermeyer/tdmpc2-jax/commit/5b05ff452424896d709848e1f249bd67e269b8a1).
The repository is MIT-licensed.

Use it as a reviewed source port, not as an unpinned dependency on `main`.
Direct installation is not the recommended integration because:

- `setup.py` pins `gymnasium[mujoco]==1.0.0`, while this project currently
  resolves Gymnasium 1.3.0 and does not need the upstream MuJoCo stack;
- the core imports TensorFlow Probability, but it is absent from upstream
  `install_requires`; the trainer also imports Hydra, TensorFlow, Orbax, and
  DMControl outside that declared dependency list;
- the upstream repository has no automated test suite;
- its trainer is written for Gymnasium/DMControl vector environments, not the
  functional JaxMARL API;
- its world model, replay schema, planner, and update loop assume one agent's
  action is the complete transition action.

### What to reuse

- `world_model.py`: SimNorm latent encoder, latent dynamics, two-hot reward and
  Q heads, Q ensemble, stochastic policy prior, and continuation head;
- `tdmpc2.py`: joint world-model loss, TD target, target-Q EMA, policy update,
  MPPI warm start, policy-prior candidates, elite weighting, and terminal Q;
- `common/` and `networks/`: symlog/two-hot utilities, normalized MLP layers,
  and ensemble structure;
- `config.yaml`: initial single-task model and planner defaults.

### What not to reuse

- `train.py`, because environment stepping, auto-reset, logging, and checkpoint
  ownership conflict with the existing JaxMARL pipeline;
- `data/`, because its sequential buffer deliberately permits samples to cross
  episode boundaries and does not carry opponent context or joint actions;
- `envs/`, MuJoCo dependencies, TensorFlow/TensorBoard setup, media, or the
  upstream packaging metadata.

### How to bring it in

1. Record the upstream URL, exact commit, copied files, and local deviations in
   `third_party/tdmpc2-jax/UPSTREAM.md`.
2. Preserve the upstream MIT license in `third_party/tdmpc2-jax/LICENSE` and
   retain attribution in adapted source headers.
3. Port only the required core into the local modules listed under Gate 4.
4. Make the first port commit behavior-preserving: formatting, imports, local
   configuration, and tests only. Add opponent conditioning in a later commit.
5. Never update from upstream `main` implicitly. Re-pin and re-run baseline
   parity tests for every upstream refresh.

Small compatibility substitutions are acceptable and should be documented:

- replace `einops` SimNorm reshaping with `jax.numpy.reshape`;
- replace `jaxtyping` annotations with the project's JAX/PyTree annotations;
- replace TensorFlow Probability's diagonal Normal with the existing pinned
  Distrax/JAX implementation, while preserving tanh log-probability correction.

These substitutions avoid adding a second environment/training dependency
stack. They must pass a fixed-seed output comparison against the pinned source.

## Fixed decisions

1. Blue is the controlled prey; red is the frozen predator.
2. The new TD-MPC path uses continuous actions only.
3. Expose each physical action as `(x, y) in [-1, 1]^2`.
4. Keep the three opponent types fixed within an episode.
5. Infer opponent context only from history available before the current action.
6. Train and validate opponent BC before training TD-MPC.
7. Implement the three slide formulations as matched modes. Equation 3 is the
   recommended model; Equations 1 and 2 are controls.
8. Use the pinned TD-MPC2-JAX core with JAX/Flax and the existing locked
   environment. Do not import either full upstream training repository.
9. Do not begin online co-training, switching opponents, cyclic tasks, or
   Spatial Blotto in this implementation.
10. Use each frozen specialist's deterministic policy mean for the initial
    rollout protocol. Add opponent sampling only as a separately declared arm.

## Notation

Do not use `z` for both representations.

- `s_t`: environment state or observation available to the world model.
- `x_t = h(s_t)`: TD-MPC world-state latent.
- `c_t = g(H_<t)`: causal opponent context.
- `u_t`: blue/prey action.
- `v_t`: red/predator action.

Define `c_0` as a zero/prior vector. `c_t` may use transitions only through
time `t - 1`. Never attach a full-episode latent to an earlier action.

## The three world-model modes

Use one implementation with `opponent_mode` rather than three copied models.

| Mode | Transition | Interpretation | Role |
|---|---|---|---|
| `implicit` | `x_next = d(x, u)` | Red behavior is marginalized into dynamics. | No-opponent baseline |
| `conditioned` | `x_next = d(x, u, c)` | Context selects a transition regime, but red action stays implicit. | Direct-conditioning ablation |
| `factored` | `v ~ pi_red(v | x, c)`; `x_next = d(x, u, v)` | Opponent behavior and physics are separate. | Recommended model |

The factored probabilistic model is

```text
p(x_next, v | x, u, c)
  = pi_red(v | x, c) * p_dynamics(x_next | x, u, v).
```

The slide's `pi_red(v | c)` is incomplete: a strategy does not determine a
directional action without the current situation. Because SimpleTag actions
are simultaneous, `pi_red` must not receive the candidate blue action from the
same timestep. Blue influences later red actions through later states.

### Conditioning contract

For `factored` mode:

| Component | Inputs |
|---|---|
| Opponent policy | `pi_red(x, c) -> distribution over v` |
| Joint-action dynamics | `d(x, u, v) -> x_next` |
| Blue reward | `R_blue(x, u, v)` |
| Termination/continuation | `T(x, u, v)` |
| Blue Q-functions | `Q_blue(x, u, c)` |
| Blue policy prior | `pi_blue(x, c)` |

Do not feed `c` into factored physics, blue reward, or termination. This is the
central controllability invariant:

> Holding `x`, `u`, and `v` fixed while changing `c` must leave predicted
> physics, reward, and termination unchanged.

In `implicit` mode, reward/termination/Q/policy receive no opponent context. In
`conditioned` mode, dynamics, reward, termination, Q, and the blue policy prior
receive `c`, because red action is being marginalized rather than modeled.

### Mapping onto TD-MPC2-JAX

The upstream code calls its world latent `z`; the adapted code should call it
`x` so it cannot be confused with opponent context.

| Upstream interface | Local interpretation/change |
|---|---|
| `WorldModel.encode(obs)` | World encoder `h(s) -> x`; it does not produce opponent context. |
| `WorldModel.next(z, action)` | Equation 1 uses blue action; Equation 2 adds `c`; Equation 3 uses concatenated blue and red actions. |
| `WorldModel.reward(z, action)` | Predict blue reward using the same mode-specific inputs as the transition. |
| `WorldModel.sample_actions(z)` | This remains the blue policy prior; it receives `c` in opponent-aware modes. |
| `WorldModel.Q(z, action)` | Blue Q-value; it receives blue action and `c`, not an action selected for red. |
| `TDMPC2.plan(...)` | MPPI continues to optimize blue actions only; factored rollouts generate red actions before each transition. |
| `TDMPC2.update(...)` | Add context sequences and recorded red actions; train factored physics with recorded joint actions. |

Do not repurpose the upstream blue policy prior as the opponent model. The
frozen red BC is an additional callable component with separately identified
parameters and provenance.

## Continuous-action contract

JaxMARL exposes continuous MPE actions as a redundant vector in `[0, 1]^5` and
decodes physical force as

```text
(a[2] - a[1], a[4] - a[3]).
```

The learning and planning interface should instead use the identifiable action
`(a_x, a_y) in [-1, 1]^2`. Convert it at the environment boundary:

```text
[a_x, a_y]
  -> [0, max(-a_x, 0), max(a_x, 0), max(-a_y, 0), max(a_y, 0)].
```

Add tests for zero, both axes, diagonals, bounds, batches, `jit`, and `vmap`.
The decoded force must equal the original two-dimensional action before the
environment applies agent acceleration.

## Current repository mapping

| Existing component | Reuse | Required change |
|---|---|---|
| [`src/tag_objectives`](src/tag_objectives) | Environment, rewards, capture termination, reset controls | Pass continuous action mode and add the 2-D boundary adapter |
| [`scripts/train_mappo.py`](scripts/train_mappo.py) | Specialist training and checkpoint conventions | Add a bounded continuous actor and continuous PPO log-probability/entropy handling |
| [`src/mopa/data.py`](src/mopa/data.py) | Matched resets and frozen-episode rollout logic | Store continuous joint actions and complete Markov transitions |
| [`src/mopa/bc.py`](src/mopa/bc.py) | Two-layer MLP, normalization, artifacts | Add a continuous head/loss without breaking the discrete experiment |
| [`scripts/run_bc.py`](scripts/run_bc.py) | Held-out folds, causal timing, four conditioning arms, closed-loop evaluation | Add continuous metrics and checkpoints |
| [`src/mopa/replay.py`](src/mopa/replay.py) | `T+1` observation alignment, masks, provenance | Separate termination from truncation and store time-indexed context |
| [`src/mopa/encoders.py`](src/mopa/encoders.py) | Causal GRU-JEPA context encoder | Freeze it for the first TD-MPC experiment |
| [`src/mopa/manifest.py`](src/mopa/manifest.py) | Hash-bound experiment provenance | Record action contract, model mode, planner config, and every source checkpoint |

There is currently no learned world encoder, transition model, reward/done
model, Q ensemble, blue policy prior, MPPI planner, or TD-MPC update loop.

## Data contract

Regenerate trajectories after retraining continuous specialists. Each episode
must contain:

```text
state_or_observation     float32 [T + 1, state_dim]
red_observation          float32 [T + 1, red_obs_dim]
blue_action              float32 [T, 2]
red_action               float32 [T, 2]
blue_reward              float32 [T]
terminated_capture       bool    [T]
truncated_timeout        bool    [T]
valid_mask               bool    [T]
causal_context           float32 [T + 1, context_dim]
objective_label          int32   []       # evaluation/oracle arm only
checkpoint_seed          int32   []
environment_seed         uint32  [2]
```

Use the complete `ObjectiveState` or a documented Markov-equivalent vector for
world-model targets. Positions alone are insufficient: velocity, resources,
collection state, lava, time, and termination-relevant state affect transitions
or reward.

Keep capture termination separate from timeout truncation. Bootstrap through a
timeout unless remaining horizon/time is explicitly part of the modeled state.

Preserve checkpoint-held-out splits and use identical reset keys across the
three opponent types. Check continuous action saturation and state-action
coverage before fitting a world model.

## Implementation order and gates

### Gate 0: upstream compatibility and parity spike

Before changing the algorithm for multiple agents:

1. Port the pinned upstream core and its default configuration.
2. Instantiate a two-dimensional single-agent model under this repository's
   locked JAX 0.4.38, Flax 0.10.4, and Optax 0.2.5 environment.
3. Run fixed-seed synthetic `encode`, `next`, `reward`, `Q`, policy, `plan`, and
   `update` calls. Assert shapes, finite losses, bounded actions, and parameter
   updates.
4. Compare outputs before and after each compatibility substitution.
5. Run a small single-agent continuous-control smoke with no opponent changes.

Use `opponent_mode=implicit` and `context_dim=0` for this baseline. Preserve
upstream defaults initially: horizon 3, 512 candidates, 24 policy-prior samples,
64 elites, six MPPI iterations, five Q networks, and 0.99 discount. A smaller
config may be used for CI but must be named `smoke`; it cannot replace the
reference configuration.

Pass when the source-port parity tests pass and a checkpoint can be saved and
reloaded with identical deterministic actions. Do not begin the JaxMARL or
opponent changes before this gate passes.

### Gate 1: continuous environment and specialists

1. Add the action adapter at the environment boundary.
2. Generalize MAPPO to a two-dimensional tanh-squashed diagonal Gaussian.
3. Retrain capture, risk, and curious predator checkpoint families.
4. Regenerate matched continuous trajectories.

PPO must compute the joint action log-probability by summing across action
dimensions and include the tanh change-of-variables correction. Do not train on
clipped Gaussian actions with an uncorrected Gaussian log-probability.

Pass when:

- actions and stored targets are finite `float32`, shaped `[..., 2]`, and
  bounded;
- environment adapter tests pass;
- the three continuous specialist families remain behaviorally distinguishable
  on capture, lava, coverage, and resource metrics;
- trajectory replay exactly reproduces source episodes.

### Gate 2: continuous opponent BC

Implement the requested vanilla baseline first:

```text
v_hat = tanh(MLP(red_observation))
loss = mean_squared_error(v_hat, v)
```

Use the same two-layer trunk for four matched arms:

1. `no_c`: current observation only;
2. `real_c`: current observation plus causal context;
3. `shuffled_c`: split-local shuffled context;
4. `oracle`: current observation plus true type one-hot.

Report held-out action MSE/MAE, direction cosine, and action saturation. Retain
closed-loop ADE/FDE, action agreement within a tolerance, capture, survival,
lava, coverage, and resource diagnostics.

The deterministic MSE model is the declared vanilla baseline. Add a diagonal
tanh-Gaussian output using the same trunk only when the planner needs calibrated
opponent sampling; report NLL/calibration when it is added.

Pass when:

- `real_c` beats both `no_c` and `shuffled_c` on held-out action likelihood or
  error and closed-loop behavior;
- the oracle arm establishes that type information is action-relevant;
- every action uses `c_t` computed before that action.

If the oracle arm does not beat `no_c`, stop and revisit the task/data rather
than adding TD-MPC complexity.

### Gate 3: simulator-backed planner

Validate planning before learning physics:

1. true simulator + true red specialist;
2. true simulator + observation-based red BC;
3. correct, zero, shuffled, and wrong context controls.

This stage can use exact simulated future red observations, so the current
observation-based BC interface is valid. It isolates planner and opponent-model
errors from world-model error.

MPPI optimizes only a blue action sequence. At each real step:

```text
x = encode(current_state)
c = infer_context(real_history_before_current_action)
warm-start the previous blue sequence shifted by one step
sample bounded blue candidate sequences
for every candidate and opponent scenario:
    generate/sample v from pi_red(current_imagined_state, c)
    advance joint dynamics with (u, v)
    add predicted blue reward until termination
add terminal blue Q when available; use zero for the planner-only smoke
update the blue Gaussian sequence from weighted elites
execute only the first blue action
observe the real transition, update c, and replan
```

Hold `c` fixed inside the short imagined horizon initially. If a categorical
belief over fixed episode types is sampled, sample one type per imagined
trajectory, not independently at every step.

Pass when the oracle-opponent simulator planner behaves sensibly, respects
bounds/termination, and its return exceeds random and fixed-policy controls.

### Gate 4: TD-MPC world model

Port the reviewed upstream core into these local runtime modules:

- `src/mopa/tdmpc.py`: adapted `world_model.py` plus the update portion of
  `tdmpc2.py`—encoder, dynamics modes, reward/continuation heads, Q ensemble,
  blue policy prior, losses, EMA targets, and update step;
- `src/mopa/mppi.py`: adapted `TDMPC2.plan` and `estimate_value`, retaining the
  continuous MPPI algorithm while making opponent generation explicit;
- `scripts/run_tdmpc.py`: data collection, training, evaluation, manifests, and
  matched mode comparison;
- `configs/tdmpc2.yaml`: source-pinned defaults plus the local action, context,
  opponent-mode, and smoke settings.

Do not import the upstream package at runtime and do not keep a second local
TD-MPC implementation. The source map and fixed-seed parity tests are what make
this an attributable adaptation rather than a new implementation from memory.

After Gate 0 parity, make these local correctness changes before adding the
three opponent modes:

- derive batch size from sampled tensors instead of allocating with a configured
  fixed batch size;
- sample sequences within episode/valid-mask boundaries using local replay;
- preserve upstream's useful distinction: termination stops TD bootstrap,
  whereas truncation ends the sampled rollout but does not automatically zero
  the target;
- make continuation timing consistent. For this task, predict
  `continue(x, u, v)` for the same transition and train it on
  `1 - terminated_capture`; do not train on one latent location and query it at
  a different location during planning;
- keep every JIT-visible opponent policy parameter, input normalization value,
  and PRNG key inside explicit JAX PyTrees. The current NumPy-oriented
  `BCPolicy` methods cannot be called from the jitted planner unchanged.

Start with an identity/normalized explicit Markov-state encoder using the same
interfaces. This makes the factorization and invariance tests debuggable. Label
that stage a **TD-MPC-style state-space baseline**, not decoder-free TD-MPC2.

Then enable the learned encoder and latent consistency target:

```text
L_world = sum_k temporal_weight[k] * (
    latent_consistency(predicted_x_next, stop_gradient(encoded_next_state))
    + reward_loss
    + termination_loss
    + TD_value_loss
)
```

Use the capture-aware TD target

```text
q_target = blue_reward
    + gamma * (1 - terminated_capture)
      * min_target_Q(x_next, pi_blue(x_next, c_next), c_next)
```

Do not zero the bootstrap merely because a dataset segment ended at a time
limit; handle truncation according to the declared finite-horizon state.

The TD-MPC core should include:

- joint-embedding/latent consistency;
- blue reward prediction;
- capture continuation/termination prediction;
- an EMA target-Q ensemble with a minimum-of-two target;
- a bounded stochastic blue policy prior;
- temporal loss weighting, SimNorm, and gradient clipping;
- short-horizon MPPI with terminal Q.

Retain the upstream no-opponent path as a regression target. For the three
modes, change only the model inputs and rollout factorization described above:

- `implicit`: upstream action is the two-dimensional blue action;
- `conditioned`: append causal context to dynamics/reward/Q/policy inputs;
- `factored`: dynamics/reward/continuation receive the four-dimensional joint
  action, while Q and the blue prior receive blue action/context as declared.

In factored planning, split PRNG keys explicitly for red actions at every
imagined step. Use the deterministic red mean for the first experiment. If red
uncertainty is enabled later, evaluate multiple coherent red scenarios per blue
candidate rather than letting uncontrolled sampling noise determine MPPI elites.

Omit pixels, multitask task embeddings, action-space padding, distributed
training, and large-model infrastructure.

#### Opponent-BC interface inside learned rollouts

A decoder-free rollout does not produce future raw red observations. Do not
silently call the observation-based BC on unavailable features.

- Simulator-backed planning uses the validated observation-based BC.
- Learned latent planning trains the same continuous BC architecture on
  `(x_t, c_t)` samples, or adds a narrowly scoped predicted-red-observation
  head.
- Prefer latent-input BC for the first decoder-free implementation because it
  avoids adding a decoder. Freeze the opponent-history encoder `g`; train or
  freeze the world encoder according to an explicit config and record it.

Pass when one-step and multi-step model error beat persistence, reward and
termination predictions are calibrated, and performance survives held-out
reset states and opponent checkpoints.

### Gate 5: matched equation comparison

Train `implicit`, `conditioned`, and `factored` with identical datasets, seeds,
capacity, update counts, planning horizons, candidate counts, and reset keys.

Run the component ladder:

1. simulator + true red policy;
2. simulator + modeled red policy;
3. learned physics + true red policy, with recorded joint actions for model-error
   evaluation;
4. learned physics + modeled red policy;
5. full end-to-end MPPI controller.

For Equation 3, additionally require:

- fixed `(x, u, v)` + varied `c` leaves dynamics/reward/termination unchanged;
- fixed `x` + varied `c` changes red action distributions when behavior should
  differ;
- varied red actions change next-state predictions in the expected direction;
- the planner never optimizes or chooses red actions;
- correct context outperforms zero, shuffled, and wrong context on matched
  resets.

Primary end-to-end metrics are blue return, resource collection, capture rate,
survival time, model rollout error, and planner latency. Report per-opponent
results and aggregate paired results across at least three controller seeds and
held-out opponent checkpoint families.

Do not claim success unless `factored` improves real-environment outcomes over
`implicit` and passes the action-clamp invariance test. Better latent clustering
alone is not sufficient.

## Minimal test plan

- `tests/test_tdmpc_upstream.py`: fixed-seed source-port parity, two-hot/symlog,
  SimNorm, tanh policy, MPPI, synthetic update, and checkpoint round trip;
- `tests/test_objectives_env.py`: continuous action adapter and environment
  boundary.
- `tests/test_pipeline.py`: continuous trajectory shapes, causal alignment,
  termination versus truncation, and exact replay.
- `tests/test_bc_experiment.py`: continuous losses, artifacts, conditioning
  arms, and closed-loop controls.
- `tests/test_tdmpc.py`: all three transition modes, target-network update,
  causal inputs, action-clamp invariance, termination masking, and finite loss.
- `tests/test_mppi.py`: bounds, blue-only optimization, warm start, terminal Q,
  type coherence, and execute-first-action behavior.

Use the standard verification commands:

```bash
uv run --locked ruff check .
uv run --locked pytest -q
uv lock --check
git diff --check
```

## Definition of done

The initial TD-MPC integration is complete only when:

- the pinned TD-MPC2-JAX source port passes its no-opponent parity gate and its
  license/source provenance is committed;
- three continuous red specialist families and hash-bound rollout data exist;
- continuous `no_c`, `real_c`, `shuffled_c`, and oracle BC results are recorded;
- the simulator-backed planner passes before learned physics is enabled;
- all three equation modes run through one shared implementation;
- Equation 3 passes the physics-invariance and causal-timing tests;
- matched, multi-seed real-environment outcomes are saved with manifests;
- [`docs/STATUS.md`](docs/STATUS.md) and the root README accurately separate
  implemented code, smoke validation, completed experiments, and open claims.

## Non-goals for this handoff

- No discrete TD-MPC implementation.
- No `0s` reproduction or comparison.
- No unpinned dependency on upstream `main`.
- No upstream `train.py`, replay buffers, environment wrappers, MuJoCo stack,
  TensorFlow logging, media, or pixel pipeline.
- No online weight updates for the opponent encoder or BC.
- No opponent-type switches within an episode.
- No red/blue co-training or self-play.
- No cyclic predator-prey or Spatial Blotto environment yet.
- No broad hyperparameter sweep.
- No SOTA, TD-MPC2 replication, or controllability claim from smoke tests.

## References

- [TD-MPC2 paper](https://arxiv.org/abs/2310.16828)
- [Official TD-MPC2 implementation](https://github.com/nicklashansen/tdmpc2)
- [TD-MPC2-JAX source](https://github.com/ShaneFlandermeyer/tdmpc2-jax)
- [Pinned TD-MPC2-JAX commit](https://github.com/ShaneFlandermeyer/tdmpc2-jax/commit/5b05ff452424896d709848e1f249bd67e269b8a1)
- [`docs/STATUS.md`](docs/STATUS.md): repository evidence boundary
- [`experiments/bc/README.md`](experiments/bc/README.md): current discrete BC
  experiment and result provenance
