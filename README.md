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
- [Implementation status](docs/STATUS.md)
