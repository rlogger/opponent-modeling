# Run a saved `0s` controller

Run the frozen continuous `0s` + factored Equation 3 checkpoint in the real
environment. No retraining or planner changes are needed.

```bash
uv run --locked --extra train --extra plot python scripts/run_tdmpc.py evaluate \
  experiments/shashank_comparison_20260908/continuous/seed_0 \
  --dataset experiments/shashank_comparison_20260908/continuous_data/dataset.npz \
  --n-eps 2 --record
```

The run directory must contain `manifest.json`, `config.json`, `state_stats.npz`,
`opponent.msgpack`, and `agent.msgpack`. The dataset and frozen specialist weights
must match their recorded hashes. Use `--logdir` if specialist files were moved.
The decoder is attached before restoring the world-model checkpoint.

Outputs default to `RUN/closed_loop/` (override with `--out`):

- `evaluation.json`: per-opponent outcomes, action bounds, planning latency, and provenance.
- `evaluation_per_episode.npz`: individual episode metrics.
- With `--record`: each arm's transition/context trace, first matched episode GIF,
  and final-frame PNG. Recordings are actual simulator transitions, not model predictions.

Replays use a fixed arena-scale camera (±2.5 at current defaults), with edge
pointers and actual coordinates for off-screen agents. `--replay-camera full`
restores the full-trajectory diagnostic view. Neither view changes agent positions
or adds walls; a late escape no longer zooms the default view out from frame zero.

`0s` defaults to `--context-modes online`. At decision `t`, its 8D latent contains
only completed `(state_s, observed_red_action_s)` pairs with `s < t`. It starts
at zero, includes the terminating transition, and freezes after termination.
Context stays fixed within each imagined planning horizon; actual specialist
actions are never supplied to imagined rollouts. This assumes opponent actions
are observable after each real transition.

Optional `--controls` adds matched MAPPO-prey and random controls. Existing
`zero`, `shuffled`, `oracle`, and `wrong_oracle` context modes remain available;
for `0s`, oracle modes mean train-only class prototypes, not access to the true
specialist policy. Zero context is not separately trained vanilla BC.

The saved planner settings are preserved. Reported steady planning latency
excludes the first two calls (which may compile), context inference, and simulator
stepping, and measures a whole episode batch. Two episodes per opponent are an
execution smoke test, not evidence of performance improvement. This remains the
identity-state Equation 3 variant, not standard learned-latent TD-MPC2 validation.

## Continue controller training

```bash
uv run --locked --extra train --extra plot python scripts/run_tdmpc.py adapt-0s \
  experiments/shashank_comparison_20260908/continuous/seed_0 \
  --dataset experiments/original_env_20260908/data/dataset.npz \
  --out artifacts/tdmpc_0s_online \
  --rounds 6 --episodes-per-group 8 --updates-per-round 1000
```

This resumes the controller and optimizer, collecting fresh planner experience
against training specialists 0/1 only; checkpoint 2 stays held out. `0s`, state
normalization, planner settings and environment rewards remain frozen. Training
samples uniformly from the original plus new replay. Each completed round saves
the latest controller, raw transitions, causal contexts, random keys and hashes.
An adapted run can itself be resumed into another new output directory; replay
and RNG state continue with it. Nonempty output directories are rejected.

See the [original-environment rerun protocol](../experiments/original_env_20260908/PROTOCOL.md)
for the fixed-budget comparison. The pre-restoration version is preserved on
`archive/pre-original-env-20260908` at `6721cf0`.
