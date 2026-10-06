# opponent-modeling

Continuous predator–prey control with MAPPO specialists, `0s` opponent modeling,
and TD-MPC. Opponents follow capture, risk-averse, or curious objectives.
The repository also includes the separate three-zone [Spatial Blotto environment](docs/spatial-blotto.md).

[Implementation status](docs/STATUS.md) · [Collaborator guide](docs/COLLABORATOR_GUIDE.md) · [Results](docs/RESULTS.md)

Project requirements and adopted decisions are governed by the private
`rlogger/marl-private` repository. This repository contains the executable
implementation and public reproduction evidence; private meeting material stays
in that repository. Runs identify the specification commit and protocol version
alongside the executable commit, configuration and artifact hashes.

The October 5 audit has completed a local feasibility pilot. The prescribed main
comparisons and final publication are still pending. The pilot is infrastructure
and runtime evidence, not evidence that opponent modeling improves control.
Historical experiments keep their original directories and claims are qualified
in [Results](docs/RESULTS.md).

## Installation

Requires Git, Python 3.11 or 3.12, and
[uv](https://docs.astral.sh/uv/getting-started/installation/) to manage Python
packages and the virtual environment. Run commands from the repository root.

```bash
git clone https://github.com/rlogger/opponent-modeling.git
cd opponent-modeling

export UV_PROJECT_ENVIRONMENT=venv
uv sync --locked --all-extras --python 3.11
```

Dependencies are locked to JAX 0.4.38, JaxMARL 0.1.0, and Distrax 0.1.5.
Keep `--locked --all-extras`; do not upgrade them independently. For CPU runs:

```bash
export JAX_PLATFORM_NAME=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
```

For reproduction, check out the exact executable commit in the selected run
manifest before installing dependencies. `td-mpc2`, `new-env`, and the fork
encoding experiments preserve historical implementations; they are not
interchangeable with the consolidated implementation.

Datasets and trained weights are not included in Git. Generate them below or
follow the [artifact-transfer guide](docs/COLLABORATOR_GUIDE.md#artifact-transfer).
Use fresh output directories: repeating MAPPO training at the same path overwrites
its checkpoints.

## 1. Train continuous MAPPO specialists

```bash
for objective in capture risk curious; do
  uv run --locked --all-extras python scripts/train_mappo.py \
    alg=mappo_continuous_${objective} \
    SEED=0 NUM_SEEDS=3 WANDB_MODE=disabled
done
```

Each preset trains two-layer, 128-unit predator/prey MLPs with 2D actions in
`[-1, 1]`. Budget: 2 million joint environment transitions per seed and objective,
18 million total. Weights, configs, and metrics go to `logs/MPE_simple_tag_v3_continuous/`.

Keep master `SEED=0` and 128-unit actors for compatibility with the dataset readers.
`vmap0`, `vmap1`, and `vmap2` identify the three independent training runs.

Algorithm overrides use the `alg.` prefix, for example
`alg.TOTAL_TIMESTEPS=5000000`. `SAVE_PATH` changes the parent of the output directory;
pass the corresponding environment subdirectory through downstream `--logdir`
arguments. Omitting `alg=mappo_continuous_*` selects the legacy discrete preset.

The risk preset uses `DENSE_CHASE_COEF: 0.1`; collection uses the environment
default `0.0`. Add `alg.DENSE_CHASE_COEF=0.0` to risk training if you want matching
rewards. Collection does not reload training reward overrides.

## 2. Collect the dataset

```bash
uv run --locked --all-extras python scripts/make_continuous_dataset.py \
  --logdir logs/MPE_simple_tag_v3_continuous \
  --artifact-dir artifacts/continuous \
  --n-eps 200 --num-steps 100 \
  --ckpt-seeds 0,1,2 --rollout-seed 0 --prey-type capture
```

Collects 200 episodes per objective and checkpoint: 1,800 total, ending at capture
or 100 steps. `--ckpt-seeds` selects `vmap` indices. Each predator faces the
corresponding capture-prey checkpoint on matched resets. Actions use policy means;
add `--sampled` for stochastic actions.

Writes `dataset.npz`, `dataset.manifest.json`, and `report.json` under
`artifacts/continuous/`, including hashes and exact-replay checks. Use `valid_mask`
or `valid_length` to exclude terminal padding. Keep the dataset unchanged after
training; model loading checks its hash.

## 3. Train the `0s` opponent model and Equation 3 controller

```bash
uv run --locked --all-extras python scripts/run_0s_world_model.py \
  --dataset artifacts/continuous/dataset.npz \
  --logdir logs/MPE_simple_tag_v3_continuous \
  --out artifacts/continuous_0s/seed_0 \
  --encoder-steps 1500 --updates 2000 --seed 0 --heldout 2
```

Fits `0s`, freezes it, then trains the identity-state Equation 3 controller.
Checkpoints 0/1 provide 1,200 training episodes; checkpoint 2 supplies 600 held-out
episodes. The 8D opponent context uses only completed history. BC is not required.
See [model details](docs/RESULTS.md#what-the-current-model-represents).

`--updates` counts gradient steps. Keep the specialist weights available for
diagnostics. Retain `agent.msgpack`, `opponent.msgpack`, `state_stats.npz`,
`config.json`, and `manifest.json` together for reloading. Evaluation is separate.

## 4. Continue training with controller experience

```bash
uv run --locked --all-extras python scripts/run_tdmpc.py adapt-0s \
  artifacts/continuous_0s/seed_0 \
  --dataset artifacts/continuous/dataset.npz \
  --logdir logs/MPE_simple_tag_v3_continuous \
  --out artifacts/tdmpc_0s_online/seed_0 \
  --rounds 6 --episodes-per-group 8 --updates-per-round 1000
```

Adds 288 episodes and 6,000 updates against training specialists 0/1; checkpoint 2
stays held out. `0s`, normalization, rewards, and planner settings stay fixed.
Requires all three checkpoint families, the original dataset, and a new or empty
output directory. Keep the saved `online_round_*` replay files for continuation.
Moving an adapted run between machines requires the
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

Runs 24 held-out episodes per opponent type, 72 per controller. `--controls` adds
MAPPO and random policies on matched resets; `--record` saves traces and replays.
Results: `evaluation.json` and `evaluation_per_episode.npz`.

For the offline model, replace the input with `artifacts/continuous_0s/seed_0`
and use a separate output directory. Do not use held-out opponents for tuning.
Context ablations and their limitations are in the [controller guide](docs/TD_MPC_0S.md).

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

Settings: [`configs/tdmpc2.yaml`](configs/tdmpc2.yaml).

### Continuous behavior cloning

MLP opponent policies with vanilla, GRU-JEPA, shuffled, and oracle conditioning.
This runner uses the older 3D context interface, not `0s`.

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

## Spatial Blotto

The separate [Spatial Blotto specification and commands](docs/spatial-blotto.md)
cover three-zone rules, mathematical and scripted controllers, terminal-state
handling, tests and exports. Its environment and scripted payoff examples do not
establish learned Blotto control or transfer of predator–prey results.

## Reference

- [Environment and reward definitions](docs/ENVIRONMENT.md)
- [TD-MPC and `0s` implementation](docs/TD_MPC_0S.md)
- [Earlier continuous GRU-JEPA experiment protocols](experiments/continuous/README.md)
- [TD-MPC2-JAX source provenance and license](third_party/tdmpc2-jax/UPSTREAM.md)

The TD-MPC core is adapted from
[ShaneFlandermeyer/tdmpc2-jax](https://github.com/ShaneFlandermeyer/tdmpc2-jax),
pinned at `5b05ff4`. The `0s` action-decoder model follows
[Shashank's trajectory-encoding implementation](https://github.com/hegde95/opponent-modeling/tree/opponent-trajectory-encodings/encoding_viz).
Environment and training infrastructure use [JaxMARL](https://github.com/bold-lab-ai/JaxMARL).
