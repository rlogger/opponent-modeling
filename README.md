# opponent-modeling

## Install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e /path/to/JaxMARL
pip install -e ".[dev,train]"
```

## Environment quick start

```python
import jax
from tag_objectives import make_env, evaluate_policy, random_policy

env = make_env("curious")
metrics = evaluate_policy(
    env, random_policy(env), n_eps=64, key=jax.random.PRNGKey(0)
)
print(metrics.means())
```

## Train MAPPO specialists

```bash
python scripts/train_mappo.py alg=mappo_objectives_capture NUM_SEEDS=3
python scripts/train_mappo.py alg=mappo_objectives_risk NUM_SEEDS=3
python scripts/train_mappo.py alg=mappo_objectives_curious NUM_SEEDS=3
```

Checkpoints land under `logs/MPE_simple_tag_v3/`.

## Layout

```text
src/tag_objectives/   # env
src/mopa/             # Part 1: encoders, belief, bc, data, metrics
scripts/train_mappo.py
scripts/run_part1.py
configs/alg/mappo_objectives_{capture,risk,curious}.yaml
docs/ENVIRONMENT.md
tests/
```

## Test

```bash
PYTHONPATH=src pytest -q
```
