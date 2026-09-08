# opponent-modeling

Opponent behavior modeling and continuous control in a predator–prey environment.
The experiments use capture, risk-averse, and curious MAPPO opponents, behavior
cloning, trajectory encoders, and TD-MPC-style controllers.

The continuous workflow is:

1. Train MAPPO specialists.
2. Collect a matched trajectory dataset.
3. Train either the `0s` opponent model and factored controller, or an implicit
   controller without opponent context.
4. Collect controller experience, update the controller, and evaluate on held-out
   opponents.

BC is a separate experiment; it is not a prerequisite for the `0s` workflow.
The older discrete experiments remain available but are not part of this path.

For collaborators, start with the [handoff and artifact guide](docs/COLLABORATOR_GUIDE.md).
The [results and development record](docs/RESULTS.md) explains what was run,
how the implementation changed, and what the evidence supports. This README
covers execution from scratch.

## Installation

Requires Git, [uv](https://docs.astral.sh/uv/), and Python 3.11 or 3.12. The examples
below use Python 3.11 and the `td-mpc2` branch. Run all commands from the repository
root, in the same shell.

`uv` manages the Python environment and packages—roughly the job of `venv` plus
`pip`. `uv sync` installs the locked dependencies; `uv run` runs a command in that
environment. It is not part of the learning algorithm. Install it once using the
[uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone --branch td-mpc2 https://github.com/rlogger/opponent-modeling.git
cd opponent-modeling

export UV_PROJECT_ENVIRONMENT=venv
uv sync --locked --all-extras --python 3.11
```

The lockfile supplies the training, plotting, and test dependencies, including
JAX 0.4.38, JaxMARL 0.1.0, and Distrax 0.1.5. Do not upgrade these independently.
Keep `--all-extras` on subsequent `uv run` commands so optional dependencies remain
installed. For a reproducible CPU run:

```bash
export JAX_PLATFORM_NAME=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
```

The specialist checkpoints and datasets required for this workflow are not
distributed with the repository. Generate them with the commands below, or obtain
a complete run directory together with its
dataset, manifest, and original specialist checkpoints. The reports under
`experiments/` do not replace those files.

Use fresh output directories for new experiments. MAPPO uses fixed checkpoint
filenames, so repeating a training command with the same output path replaces its
previous files.

## 1. Train continuous MAPPO specialists

```bash
for objective in capture risk curious; do
  uv run --locked --all-extras python scripts/train_mappo.py \
    alg=mappo_continuous_${objective} \
    SEED=0 NUM_SEEDS=3 WANDB_MODE=disabled
done
```

Each preset trains separate predator and prey actor–critic pairs with 128-unit,
two-layer MLPs. Actions are two-dimensional, tanh-bounded vectors in `[-1, 1]`.
The five-component vector passed to JaxMARL is a continuous force adapter, not a
discrete action choice.

| Setting | Default |
|---|---:|
| Environment transitions per training seed | 2,000,000 |
| Parallel environments | 32 |
| Steps per rollout | 100 |
| PPO updates per training seed | 625 |
| PPO epochs / minibatches | 4 / 4 |
| Learning rate | 0.0003 |
| Discount / GAE lambda | 0.99 / 0.95 |

Three objectives and three seeds require 18 million environment transitions in
total. Both teams train on each joint transition; this is not a per-agent count.
Training saves final weights, configuration, and loss/episode metrics to
`logs/MPE_simple_tag_v3_continuous/`. For example:

```text
mappo_continuous_capture_MPE_simple_tag_v3_continuous_pred_actor_seed0_vmap0.safetensors
mappo_continuous_capture_MPE_simple_tag_v3_continuous_prey_actor_seed0_vmap0.safetensors
mappo_continuous_capture_MPE_simple_tag_v3_continuous_seed0_config.yaml
mappo_continuous_capture_MPE_simple_tag_v3_continuous_seed0_metrics.npz
```

Critic weights are saved alongside the actors. `SEED=0` is the master random seed;
`vmap0`, `vmap1`, and `vmap2` are the three independently initialized runs. The
dataset readers expect this master seed and 128-unit actors. Keep both unchanged
for the workflow below.

Algorithm overrides use the `alg.` prefix, for example
`alg.TOTAL_TIMESTEPS=5000000`. `SAVE_PATH` changes the parent of the output directory;
pass the corresponding environment subdirectory through downstream `--logdir`
arguments. Omitting `alg=mappo_continuous_*` selects the legacy discrete preset.

The continuous risk preset explicitly sets `DENSE_CHASE_COEF: 0.1`; the environment
constructor defaults to `0.0`. To train risk with the constructor's reward instead,
add `alg.DENSE_CHASE_COEF=0.0` to that training command. Preserve this choice in the
saved configuration. The dataset collector uses environment defaults and does not
reload reward overrides from the training configuration.

## 2. Collect the dataset

```bash
uv run --locked --all-extras python scripts/make_continuous_dataset.py \
  --logdir logs/MPE_simple_tag_v3_continuous \
  --artifact-dir artifacts/continuous \
  --n-eps 200 --num-steps 100 \
  --ckpt-seeds 0,1,2 --rollout-seed 0 --prey-type capture
```

`--ckpt-seeds` selects the `vmap` indices, not the master random seed. Collection
uses each predator family against the capture-family prey at the corresponding
checkpoint index, with matched reset keys across objectives. Actions are
deterministic policy means by default; `--sampled` selects stochastic actions.

`--n-eps 200` means 200 episodes **per objective per checkpoint**: 1,800 episodes
in total. Episodes stop at capture or 100 steps, so the number of valid transitions
is generally below 180,000.

| File | Contents |
|---|---|
| `artifacts/continuous/dataset.npz` | States, observations, actions, rewards, terminal flags, masks, labels, and seeds |
| `artifacts/continuous/dataset.manifest.json` | Collection settings, source versions, and checkpoint hashes |
| `artifacts/continuous/report.json` | Dataset/manifest hashes, data checks, simulator replay, and behavior statistics |

The default collection command replays all episodes to check simulator agreement.
Array layout is `[episode, time, ...]`: states and observations include the final
state (`T+1`), while actions and rewards have `T` entries. Red and blue actions
have width 2. Use `valid_mask` or `valid_length` when analyzing data; padding after
termination is not additional experience. Capture and timeout have separate flags.

Keep the dataset unchanged after training. Checkpoint loading and controller
adaptation verify its hash, not just its filename or array shapes.

## 3. Train the `0s` opponent model and Equation 3 controller

```bash
uv run --locked --all-extras python scripts/run_0s_world_model.py \
  --dataset artifacts/continuous/dataset.npz \
  --logdir logs/MPE_simple_tag_v3_continuous \
  --out artifacts/continuous_0s/seed_0 \
  --encoder-steps 1500 --updates 2000 --seed 0 --heldout 2
```

This command fits one shared `0s` action-decoder VAE, freezes its opponent decoder,
and trains the factored world model, reward, value, policy-prior, and continuation
heads. Checkpoints 0 and 1 supply training data; checkpoint 2 is held out. With the
dataset above, that is 1,200 training episodes and 600 held-out episodes.

The encoder uses eight-step windows, a 64-unit GRU, and an eight-dimensional
opponent context `z`. At decision `t`, context includes only state–opponent-action
pairs observed before `t`. The controller's world state `x` is a normalized 66D
state vector, not another learned `0s` latent:

```text
predicted_red_action = decoder(x, z)
predicted_next_state = dynamics(x, blue_action, predicted_red_action)
```

This is the identity-state Equation 3 implementation, not the upstream
learned-latent TD-MPC2 baseline. `--updates` counts gradient updates, not environment
steps. Specialist checkpoint files are still required for the script's matched-state
diagnostics, even when the dataset already exists.

The output directory contains `agent.msgpack`, `opponent.msgpack`,
`state_stats.npz`, `config.json`, and `manifest.json` for reloading the model.
`training_history.json`, `metrics.json`, `latents.npz`, and `REPORT.md` contain the
training and representation diagnostics. This command does not evaluate the
controller in the real environment.

## 4. Continue training with controller experience

```bash
uv run --locked --all-extras python scripts/run_tdmpc.py adapt-0s \
  artifacts/continuous_0s/seed_0 \
  --dataset artifacts/continuous/dataset.npz \
  --logdir logs/MPE_simple_tag_v3_continuous \
  --out artifacts/tdmpc_0s_online/seed_0 \
  --rounds 6 --episodes-per-group 8 --updates-per-round 1000
```

Each round collects experience against the frozen training opponents and updates
the controller from mixed offline/online replay. The example adds 288 episodes
and 6,000 gradient updates. Actual transition counts depend on episode length.
The `0s` encoder and decoder, state normalization, planner settings, and environment
rewards remain unchanged.

This command requires checkpoint families `{0,1,2}`, held-out checkpoint 2, and
the exact dataset used for the initial fit. The output directory must be new or
empty. Updated checkpoints and training logs are saved after each round; collected
episodes are stored under `online_round_000/`, `online_round_001/`, and so on.
To continue an adapted run, use it as the positional input and choose another
fresh output directory. Retain its recorded experience files. Historical replay
paths are absolute, so moving an adapted run to another machine needs the
[resume precautions](docs/COLLABORATOR_GUIDE.md#continuing-an-adapted-run).

## 5. Evaluate the controller

```bash
uv run --locked --all-extras python scripts/run_tdmpc.py evaluate \
  artifacts/tdmpc_0s_online/seed_0 \
  --dataset artifacts/continuous/dataset.npz \
  --logdir logs/MPE_simple_tag_v3_continuous \
  --out artifacts/tdmpc_0s_online/seed_0/closed_loop \
  --n-eps 24 --context-modes online --controls --record
```

`--n-eps` is the number of held-out episodes per opponent type. This example runs
72 episodes per controller. `--controls` adds the fixed MAPPO prey and a random
policy on matched resets. It does not train either control. `--record` saves
transition traces and a GIF/PNG replay of the first episode from each evaluation
group; the trace arrays retain validity masks for terminal padding.

Read `evaluation.json` for returns, capture rates, episode lengths, resource
collection, timing, and provenance. `evaluation_per_episode.npz` retains individual
outcomes. To evaluate the offline model, use `artifacts/continuous_0s/seed_0` as
the input run and give its results a separate output directory.

Supported context ablations are `online,zero,shuffled,oracle,wrong_oracle`.
Oracle contexts are training-set prototype means, not access to the real opponent's
next action. A zero-context decoder is not a separately trained vanilla BC model.
Keep held-out opponents out of model selection; use a separate validation split
for tuning. Representation probes and ARI do not establish closed-loop control
performance.

## Other training paths

### Equation 1: implicit world model

This baseline predicts `x_next = dynamics(x, blue_action)` without an opponent
model or context. It requires only the continuous dataset and specialists above.

```bash
uv run --locked --all-extras python scripts/run_tdmpc.py train \
  --dataset artifacts/continuous/dataset.npz \
  --out artifacts/tdmpc_equation1 \
  --profile equation1 --mode implicit --encoder identity --features relative \
  --heldout 2 --seed 0 --updates 10000

uv run --locked --all-extras python scripts/run_tdmpc.py evaluate \
  artifacts/tdmpc_equation1/implicit__identity__ctx-none__h2__s0__relative \
  --dataset artifacts/continuous/dataset.npz \
  --logdir logs/MPE_simple_tag_v3_continuous \
  --n-eps 24 --context-modes zero --controls --record
```

Unlike the `0s` producer, `train` creates a named run subdirectory under `--out`.
In this name, `h2` identifies held-out checkpoint 2, not the planning horizon.
Add `--online-rounds 6 --online-episodes 8 --updates-per-round 1000` to the training
command for controller-generated experience. `--encoder mlp` selects a learned
SimNorm state encoder; it also changes the generated directory name.

The planner uses horizon 3, 512 candidates, 24 policy proposals, 64 elites, and
six MPPI iterations. The value ensemble has five heads and discount 0.99. See
[`configs/tdmpc2.yaml`](configs/tdmpc2.yaml) for network, loss, and profile settings.
Match training budgets, inputs, and opponent splits before comparing methods.

### Continuous behavior cloning

The existing BC experiment compares a vanilla opponent policy (`no_c`) with causal
GRU-JEPA context (`real_c`), shuffled context, and an objective-label oracle. It uses
MLPs with continuous action outputs. This runner uses the older three-column
context interface, **not `0s`**.

```bash
uv run --locked --all-extras python scripts/run_bc_continuous.py \
  --dataset artifacts/continuous/dataset.npz \
  --logdir logs/MPE_simple_tag_v3_continuous \
  --artifact-dir artifacts/bc_continuous \
  --seeds 0,1,2 --encoder-steps 5000 --bc-steps 4000 \
  --lat 2 --hid 32 --ctx 10
```

All three checkpoint indices become leave-one-out folds. Outputs include
`results.json` and `fold_<heldout>/seed_<seed>/`, containing `context_encoder.npz`
and `no_c.npz`, `real_c.npz`, `shuffled_c.npz`, and `oracle.npz`. The default also
evaluates closed-loop continuations; add `--skip-closed-loop` for offline evaluation
only, or `--closed-loop-eps 24` to limit continuation episodes per opponent type.

The generic `run_tdmpc.py train --mode conditioned` and `--mode factored` paths
consume these BC/context artifacts through `--bc-artifacts`. They are not substitutes
for the eight-dimensional `0s` training command in section 3.

### Exact-simulator planner

After the BC experiment, run the simulator-based diagnostic with:

```bash
uv run --locked --all-extras python scripts/run_sim_planner.py \
  --dataset artifacts/continuous/dataset.npz \
  --bc-artifacts artifacts/bc_continuous \
  --logdir logs/MPE_simple_tag_v3_continuous \
  --artifact-dir artifacts/sim_planner \
  --heldout 2 --bc-seed 0 --n-eps 24
```

This uses exact simulator dynamics and zero terminal value, not the learned TD-MPC
world model. It writes `results.json` and `per_episode.npz`. The runner loads BC
and context files even when its selected controller subset does not use every
model, so the preceding BC fit is required.

### Legacy discrete experiments

For reproduction of the earlier discrete pipeline only:

```bash
for objective in capture risk curious; do
  uv run --locked --all-extras python scripts/train_mappo.py \
    alg=mappo_objectives_${objective} SEED=0 NUM_SEEDS=3 WANDB_MODE=disabled
done

uv run --locked --all-extras python scripts/run_part1.py \
  --logdir logs/MPE_simple_tag_v3 --n-eps 200 --ckpt-seeds 0,1,2 \
  --encoder-steps 1500 --action-decoder-steps 1500 --bc-steps 4000 \
  --run-kind full --out artifacts/legacy/part1.json \
  --dataset-cache artifacts/legacy/dataset.npz

uv run --locked --all-extras python scripts/run_bc.py \
  --logdir logs/MPE_simple_tag_v3 \
  --artifact-dir artifacts/legacy/bc \
  --dataset-cache artifacts/legacy/bc/dataset.npz
```

Do not pass these integer-action datasets or categorical checkpoints to the
continuous runners. A full Part 1 rerun requires a new `--dataset-cache` path;
an existing cache is rejected. The discrete BC protocol is documented in
[`experiments/bc/README.md`](experiments/bc/README.md).

## Tests and smoke runs

```bash
uv run --locked --all-extras ruff check .
uv run --locked --all-extras pytest -q
```

A single continuous MAPPO update, without specialist checkpoint dependencies:

```bash
uv run --locked --all-extras python scripts/train_mappo.py \
  alg=mappo_continuous_capture SEED=0 NUM_SEEDS=1 \
  alg.TOTAL_TIMESTEPS=3200 \
  SAVE_PATH=artifacts/mappo_smoke WANDB_MODE=disabled
```

The legacy synthetic pipeline can also run without checkpoints:

```bash
uv run --locked --all-extras python scripts/run_part1.py \
  --synthetic --encoder-steps 1 --bc-steps 1 \
  --out artifacts/part1_synthetic.json
```

Smoke outputs check execution, not policy quality, and cannot replace the trained
specialists required by the dataset collector. Use a script's `--help` to inspect
its arguments; for the Hydra-based MAPPO trainer, `--cfg job` prints the configuration
without training.

## Repository layout

| Path | Purpose |
|---|---|
| `src/tag_objectives/` | Environment, rewards, observations, continuous action adapter |
| `src/mopa/` | Encoders, BC, replay, evaluation, TD-MPC, and MPPI |
| `scripts/` | Training, collection, and evaluation entry points |
| `configs/` | MAPPO presets and TD-MPC settings |
| `tests/` | Environment, data, model, and integration tests |
| `third_party/` | Upstream licenses and source-port provenance |
| `experiments/` | Recorded protocols, reports, and diagnostic summaries |
| `logs/`, `artifacts/` | Locally generated checkpoints and run data; git-ignored |

Keep each run's configuration, manifest, checkpoints, dataset, and specialist
weights together when archiving or sharing results. Large arrays and weights are
local artifacts; pushing a report does not upload its training data.

## Reference

- [Collaborator handoff, artifact transfer, and reproduction](docs/COLLABORATOR_GUIDE.md)
- [Results, implementation history, and limitations](docs/RESULTS.md)
- [Environment and reward definitions](docs/ENVIRONMENT.md)
- [TD-MPC and `0s` implementation](docs/TD_MPC_0S.md)
- [Historical implementation and gate status](docs/STATUS.md)
- [Earlier continuous GRU-JEPA experiment protocols](experiments/continuous/README.md)
- [Completed main-environment rerun](experiments/main_env_20260908/REPORT.md)
- [TD-MPC2-JAX source provenance and license](third_party/tdmpc2-jax/UPSTREAM.md)

The TD-MPC core is adapted from
[ShaneFlandermeyer/tdmpc2-jax](https://github.com/ShaneFlandermeyer/tdmpc2-jax),
pinned at `5b05ff4`. The `0s` action-decoder model follows
[Shashank's trajectory-encoding implementation](https://github.com/hegde95/opponent-modeling/tree/opponent-trajectory-encodings/encoding_viz).
Environment and training infrastructure use [JaxMARL](https://github.com/bold-lab-ai/JaxMARL).
