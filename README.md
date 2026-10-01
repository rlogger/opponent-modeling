# opponent-modeling

Multi-agent research environments: objective-typed predator–prey and
[three-zone Spatial Blotto](docs/spatial-blotto.md). Blotto currently uses
mathematical allocation/routing controllers; RL and opponent-strategy learning
for this game are future work.

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

## Play Spatial Blotto

```bash
uv run --locked spatial-blotto --scenario reactive \
  --output artifacts/blotto/replay.html \
  --trajectory artifacts/blotto/episode.npz
```

Open the HTML replay to inspect movement, changing targets, ownership and scores.
Use `--scenario cyclic` for the fixed 3v3 allocation example, `--scenario balanced`
for even coverage, or `--scenario rotating --team-size 2` for rotating 2v2 teams.
The [game and integration guide](docs/spatial-blotto.md) covers rules, mathematical
controllers, JaxMARL interfaces, transition exports and tests.

## Train predator–prey specialists

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

- [Predator–prey environment](docs/ENVIRONMENT.md)
- [Spatial Blotto game and integration](docs/spatial-blotto.md)
- [BC experiment](experiments/bc/README.md)
- [Implementation status](docs/STATUS.md)
