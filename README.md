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

## Continuous TD-MPC: Equation 1

Start with `x_next = dynamics(x, blue_action)`. This baseline needs continuous
specialists and data; it runs without opponent BC or context-encoder artifacts.
Historical comparisons are in [`experiments/continuous`](experiments/continuous/README.md).

```bash
# Gate 1: continuous tanh-Gaussian specialists and the matched dataset
for t in capture risk curious; do
  uv run --locked --all-extras python scripts/train_mappo.py alg=mappo_continuous_$t NUM_SEEDS=3
done
uv run --locked --all-extras python scripts/make_continuous_dataset.py
# Equation 1: normalized state baseline, relative features, then online data.
uv run --locked --all-extras python scripts/run_tdmpc.py train \
  --seed 0 --online-rounds 6 --out artifacts/tdmpc_equation1
uv run --locked --all-extras python scripts/run_tdmpc.py evaluate \
  artifacts/tdmpc_equation1/implicit__identity__ctx-none__h2__s0__relative \
  --n-eps 32 --controls
```

Use `--encoder mlp` to train the learned SimNorm encoder with frozen training-set
input normalization. Both encoders use Equation 1; no new training results are
implied by this implementation update.

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
- [Continuous 0s experiment](experiments/continuous_0s/REPORT.md)
- [Implementation status](docs/STATUS.md)
