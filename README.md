# opponent-modeling

## Setup

Python 3.11 or 3.12:

```bash
export UV_PROJECT_ENVIRONMENT=venv
uv sync --locked --all-extras
```

## Verify

```bash
uv run --locked ruff check .
uv run --locked pytest -q
```

## Train

```bash
uv run --locked python scripts/train_mappo.py alg=mappo_objectives_capture NUM_SEEDS=3
uv run --locked python scripts/train_mappo.py alg=mappo_objectives_risk NUM_SEEDS=3
uv run --locked python scripts/train_mappo.py alg=mappo_objectives_curious NUM_SEEDS=3
```

## Continuous opponent-aware TD-MPC

The gated pipeline from [`handoff.md`](handoff.md); results and status are in
[`experiments/continuous/README.md`](experiments/continuous/README.md).

```bash
# Gate 1: continuous tanh-Gaussian specialists and the matched dataset
for t in capture risk curious; do
  uv run --locked --all-extras python scripts/train_mappo.py alg=mappo_continuous_$t NUM_SEEDS=3
done
uv run --locked --all-extras python scripts/make_continuous_dataset.py
# Gate 2: four-arm continuous opponent BC with a frozen causal context encoder
uv run --locked --all-extras python scripts/run_bc_continuous.py
# Gate 3: simulator-backed MPPI ladder with opponent-context controls
uv run --locked --all-extras python scripts/run_sim_planner.py --n-eps 48
# Gates 4-5: TD-MPC world models (implicit | conditioned | factored) and comparison.
# Reward-relevant relative features plus online collection rounds are the recorded
# configuration; purely offline training on the raw state leaves the arena.
for mode in implicit conditioned factored; do
  uv run --locked --all-extras python scripts/run_tdmpc.py train --mode $mode --encoder identity \
    --features relative --seed 0 --updates 10000 --online-rounds 6 --online-episodes 8 \
    --updates-per-round 2000 --out artifacts/tdmpc
  uv run --locked --all-extras python scripts/run_tdmpc.py evaluate \
    artifacts/tdmpc/${mode}__identity__ctx-causal__h2__s0__relative --n-eps 32 --controls
done
uv run --locked --all-extras python scripts/run_tdmpc.py compare --root artifacts/tdmpc
```

## Smoke test

```bash
uv run --locked python scripts/run_part1.py \
  --synthetic \
  --encoder-steps 1 \
  --bc-steps 1 \
  --out artifacts/part1_synthetic.json
```

## Documentation

- [Environment](docs/ENVIRONMENT.md)
- [BC experiment](experiments/bc/README.md)
- [Implementation status](docs/STATUS.md)
