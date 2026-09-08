# Main-branch environment rerun

Backup created first: `new-env` at `180571b`. Restore the environment core from
this repository's fetched `origin/main`, `8c24db3`. Keep the TD-MPC implementation
on `td-mpc2`; this is not a whole-repository reset or branch merge.

`objectives.py` and `resources.py` must be byte-identical to main. Keep only the
external API/action adapters needed to run existing continuous callers. Do not
alter reward coefficients, observation order, geometry, resets, or termination.

## Rerun matrix

- Native-main reference: regenerate 1,800 discrete specialist episodes (200 per
  objective/checkpoint, checkpoints 0/1/2, matched reset seed 0, fixed capture-prey
  family). Rerun source-compatible `0s` and the existing three causal fit seeds.
- Continuous compatibility arm: regenerate all 1,800 continuous episodes with
  the same source environment, continuous action adapter, and frozen specialists.
- Controller arm: rerun the saved six-round adapted `0s` controller against
  checkpoint 2, 24 matched episodes per objective, alongside MAPPO/random
  controls. Preserve all previous results and checkpoints; use new output paths.
- Compare data arrays/hashes and matched episode outcomes with `new-env`'s saved
  artifacts. If inputs and outputs are identical, report that explicitly instead
  of attributing a gain to the source rollback.

Native discrete specialists and continuous specialists are different policies.
Discrete results are a reference, not continuous TD-MPC validation. Source
restoration is not a claim that `0s`'s learned opponent predictions reproduce
all three intended behaviors. No new architecture, reward shaping, specialist
training, or controller tuning is included in this environment-only rerun.

## Reproduce

From the repository root, use the locked train/plot environment with
`PYTHONPATH=src`. Output directories must be new; preserve previous artifacts.

```bash
uv run --locked --extra train --extra plot python experiments/main_env_20260908/discrete/run.py
uv run --locked --extra train --extra plot python experiments/main_env_20260908/discrete/visualize.py
uv run --locked --extra train --extra plot python scripts/make_continuous_dataset.py \
  --artifact-dir experiments/main_env_20260908/continuous_data
uv run --locked --extra train --extra plot python scripts/run_tdmpc.py evaluate \
  experiments/original_env_20260908/adapted \
  --dataset experiments/main_env_20260908/continuous_data/dataset.npz \
  --n-eps 24 --controls --record --out experiments/main_env_20260908/controller
uv run --locked --extra train --extra plot python experiments/main_env_20260908/verify.py
```

The native fit wrapper reuses the pinned-source comparison driver and needs the
source checkout documented there (`7bae281`). Existing specialist checkpoints
and adapted controller weights must match their recorded hashes. No old
checkpoint or result is overwritten.
