# opponent-modeling

Objective-typed multi-agent reinforcement learning plus a compact,
uncertainty-aware opponent-modeling pipeline. The repository includes:

- capture, risk-averse, and curious predator objectives;
- MAPPO specialist training and capture-aware rollouts;
- variable-length GRU-JEPA, the short-window `0s` action-decoder VAE,
  fixed-window JEPA, beta-VAE, random, and supervised controls;
- online Bayesian strategy beliefs and belief-mixture action prediction;
- causal latent-conditioned behavior cloning; and
- validated replay, counterfactual, and CPL foundations for a future planner.

## Install

Use Python 3.11 or 3.12 and the checked-in lockfile:

```bash
export UV_PROJECT_ENVIRONMENT=venv
uv sync --locked --all-extras
```

The lock pins the tested JaxMARL/JAX/Distrax combination; no separate JaxMARL
checkout is required. The explicit non-hidden environment path also keeps
editable package imports reliable on macOS. Keep `UV_PROJECT_ENVIRONMENT`
exported while running the repository commands below.

## Verify the repository

```bash
uv run ruff check .
uv run pytest -q
```

Run the deterministic end-to-end infrastructure smoke without checkpoints:

```bash
uv run python scripts/run_part1.py \
  --synthetic \
  --n-eps 3 \
  --ckpt-seeds 0,1 \
  --encoder-seeds 0 \
  --bc-seeds 0 \
  --ctx 2 \
  --prefixes 2,4 \
  --hid 8 \
  --encoder-steps 1 \
  --bc-steps 1 \
  --out artifacts/part1_synthetic.json
```

This writes a schema-v2 smoke manifest. It tests plumbing only and is not an
experiment result.

## Train MAPPO specialists

```bash
uv run python scripts/train_mappo.py alg=mappo_objectives_capture NUM_SEEDS=3
uv run python scripts/train_mappo.py alg=mappo_objectives_risk NUM_SEEDS=3
uv run python scripts/train_mappo.py alg=mappo_objectives_curious NUM_SEEDS=3
```

Actor checkpoints are written under `logs/MPE_simple_tag_v3/`. Validate the
expected three-seed checkpoint set before a real pipeline run:

```bash
uv run python scripts/run_part1.py \
  --dry-run-checks \
  --prey-objective capture \
  --out artifacts/part1_dry.json
```

The reference protocol holds out whole checkpoint seeds and uses one fixed prey
checkpoint family across all predator objectives. This prevents the strategy
label from merely identifying three separately co-trained prey policies.

## Export a trajectory dataset and representative figure

After training, generate a share-ready bundle containing 1,800 fixed-prey
episodes, a matched-reset PNG/PDF figure, checkpoint hashes, and schema metadata:

```bash
uv run python scripts/export_trajectory_bundle.py \
  --out-dir artifacts/trajectory_share \
  --n-eps 200 \
  --ckpt-seeds 0,1,2 \
  --rollout-seed 0 \
  --num-steps 100 \
  --prey-objective capture
```

The figure selects one deterministic matched reset: all three panels use the
same initial scene, checkpoint seed, and fixed capture-prey policy. Paths stop
at capture rather than drawing the frozen padding tail. To re-render a dataset
from a completed pipeline run, pass both `--dataset` and `--source-manifest`;
their SHA-256 binding is checked before export.

## Run the real pipeline

Start with `--run-kind smoke` and small declared budgets. A full execution must
use a clean committed worktree, a fixed prey objective, fresh rollouts, a
checkpoint-held-out split, and at least three distinct checkpoint, encoder, and
BC seeds:

```bash
N_EPISODES=...
ENCODER_STEPS=...
BC_STEPS=...
uv run python scripts/run_part1.py \
  --run-kind full \
  --prey-objective capture \
  --n-eps "$N_EPISODES" \
  --encoder-steps "$ENCODER_STEPS" \
  --bc-steps "$BC_STEPS" \
  --dataset-cache artifacts/part1_rollouts.npz \
  --out artifacts/part1_full.json
```

The source documents do not settle the full budgets or scientific success
thresholds. Accordingly, a successful full execution is recorded as
`full_run_finished`, not as proof of strategy recovery or robustness.

`0s` defaults to causal past-displacement state features. Its legacy forward-
displacement mode is available only for parity checks and is rejected by full
runs because it exposes the transition caused by the current action. Shashank's
model budget is 1,500 updates, window 8, latent size 8, and hidden size 64.

## Layout

```text
src/tag_objectives/   objective-typed environment and evaluation
src/mopa/             rollouts, encoders, beliefs, policies, BC, CPL, manifests
scripts/              MAPPO training and the end-to-end Part 1 driver
configs/alg/          capture, risk, and curious MAPPO configurations
docs/                 environment and implementation status
tests/                unit, regression, synthetic pipeline, and env tests
.github/workflows/    locked lint, test, and MAPPO smoke CI
```
