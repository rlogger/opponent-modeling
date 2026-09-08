# Original environment and online-controller rerun

Pre-restoration code and reports: `archive/pre-original-env-20260908`, commit
`6721cf0`. Original environment: `rlogger/marl-opp-aware` at
`aecbab5daf6da402029e953be114a52e62a46c26` (remote main verified).

The original environment already has soft boundary penalties rather than hard
walls. Source restoration must not alter rewards, observations, reset geometry,
capture rules or the continuous action mapping to manufacture positive scores.
Original/current parity was exact across 360 matched discrete/continuous
transitions, including every observation, state, reward, done and info field.

## Fixed comparison

- Regenerate the full 1,800-episode continuous dataset using the original source,
  existing frozen specialist checkpoints, matched resets, and fixed capture prey.
  Verify its numerical compatibility with the earlier dataset before reuse.
- Baseline: saved seed-0 `0s` + Equation 3 checkpoint (2,000 offline updates).
  Rerun real-environment evaluation with fixed MAPPO-prey and random controls.
- Improvement arm: resume that same controller using fresh planner-collected
  episodes against training specialist checkpoints 0/1 only. Keep `0s`, state
  normalization, rewards, physics and planner settings frozen. Use uniform replay
  of original and newly collected transitions; record new data and update counts.
- Initial adaptation budget: six rounds, eight episodes per training
  checkpoint/objective group, and 1,000 updates after each round. Model selection
  must not use the held-out specialist's returns.
- Final comparison: 24 matched episodes per objective against checkpoint 2.
  Report all three objectives, episode-level metrics and the first matched replay;
  no success-selected clips. This is one model seed, not a multi-seed SOTA result.

Positive return is a measured outcome, not a reporting requirement. Separate
source parity, control improvement, absolute return and comparison with MAPPO.
Retain unsuccessful baselines and all training rounds. Do not transfer `0s`
probe/ARI scores into a claim about control performance.

## Execution

Use the locked environment with `PYTHONPATH=src` and the train/plot extras.

```bash
uv run --locked --extra train --extra plot python scripts/make_continuous_dataset.py \
  --artifact-dir experiments/original_env_20260908/data
uv run --locked --extra train --extra plot python scripts/run_tdmpc.py evaluate \
  experiments/shashank_comparison_20260908/continuous/seed_0 \
  --dataset experiments/original_env_20260908/data/dataset.npz \
  --n-eps 24 --controls --record --out experiments/original_env_20260908/baseline
uv run --locked --extra train --extra plot python scripts/run_tdmpc.py adapt-0s \
  experiments/shashank_comparison_20260908/continuous/seed_0 \
  --dataset experiments/original_env_20260908/data/dataset.npz \
  --out experiments/original_env_20260908/adapted \
  --rounds 6 --episodes-per-group 8 --updates-per-round 1000 --seed 0
uv run --locked --extra train --extra plot python scripts/run_tdmpc.py evaluate \
  experiments/original_env_20260908/adapted \
  --dataset experiments/original_env_20260908/data/dataset.npz \
  --n-eps 24 --record --out experiments/original_env_20260908/final
```

The actual adaptation invocation uses a byte-identical local temporary dataset
mirror to avoid iCloud I/O. Both it and the regenerated dataset have SHA256
`928181027a9e5e86b106190b1487cda90c09e2cc5df08b569f7b93ba8350b5f5`.
All 26 regenerated arrays are exactly equal to the earlier dataset. All 1,800
episodes (130,869 valid transitions) replay exactly, including observations,
rewards and termination flags. This is source/environment parity, not a new
`0s` probe or controller-performance claim.
