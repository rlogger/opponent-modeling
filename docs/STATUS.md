# Implementation status

This file is the source of truth for what the repository implements and what
still needs experiment evidence. It consolidates the historical research
requirements and later implementation updates.

## Current lean scope

The latest update narrowed the active repository to the objective-typed
environment, MAPPO specialists, trajectory representations, uncertainty-aware
opponent policies, behavior cloning, CPL foundations, evaluation, and
reproducible manifests. Within that scope, the implementation is complete:

| Requirement | Implementation | Verification |
|---|---|---|
| Capture/risk/curious task with resources, lava, first-capture termination | `tag_objectives` | Reward, observation, termination, determinism, and multi-predator tests |
| MAPPO specialist training and checkpoints | `scripts/train_mappo.py`, `configs/alg/` | One-update CI smoke plus checkpoint save/load paths |
| Capture-aware labeled rollouts with a fixed prey control | `mopa.data`, `mopa.types` | Shape, mask, split, reset-key, valid-length, and causal-sample tests |
| Variable-length GRU-JEPA | `mopa.encoders` | Padding invariance, short-episode, EMA/frozen-encode, LayerNorm/no-BatchNorm, and squared-L2 tests |
| Window JEPA and beta-VAE baselines | `mopa.encoders` | Frozen encode round trips, moving-window evaluation, and valid future-target filtering |
| Short-window `0s` action-decoder VAE | `mopa.action_decoder`, `scripts/run_part1.py` | Masked windows, frozen encoding, causal/legacy feature modes, and exact same-data upstream parity |
| Probe, three-component GMM ARI, oracle, anytime curves | `mopa.metrics`, `scripts/run_part1.py` | Train-only fit and held-out scoring |
| Posterior, entropy, ECE/NLL/Brier/reliability | `mopa.strategy`, `mopa.metrics` | Numerical and adapting-strategy tests |
| Vanilla and latent-conditioned BC | `mopa.bc`, `scripts/run_bc.py` | Exact observations, reusable MLP artifacts, four matched arms, checkpoint-held-out scoring, and closed-loop replay |
| CPL and counterfactual pair integrity | `mopa.cpl`, `mopa.replay` | Bradley-Terry numeric/gradient tests, prefix and planner-provenance checks |
| Reproducible experiment artifacts | `scripts/run_part1.py`, `scripts/run_bc.py`, `mopa.manifest` | Git/checkpoint/dataset hashes, versions, split membership, reset keys, lengths, and metrics |
| Dependency and regression checks | `uv.lock`, `.github/workflows/ci.yml` | Locked tests, lint, synthetic pipeline, and MAPPO smoke |

## Continuous opponent-aware TD-MPC (handoff gates)

The continuous-action programme from [`handoff.md`](../handoff.md) is
implemented end to end; its experiment record with all numbers is
[`experiments/continuous/README.md`](../experiments/continuous/README.md).

| Gate | Implementation | Verification | Status |
|---|---|---|---|
| 0 upstream parity | `mopa.tdmpc`, `mopa.mppi`, `configs/tdmpc2.yaml`, `third_party/tdmpc2-jax/` | Bitwise comparison against pinned `5b05ff4` for every deterministic path; golden fixed-seed values in `tests/test_tdmpc_upstream.py` | pass |
| 1 continuous env + specialists | `tag_objectives.actions`, `mopa.continuous`, `mopa.nets.ContinuousActor`, `scripts/train_mappo.py` (`ACTION_TYPE: Continuous`), `mopa.continuous_data`, `scripts/make_continuous_dataset.py` | Adapter tests (zero/axes/diagonals/bounds/batch/jit/vmap), tanh-Gaussian log-prob vs Distrax, data-contract tests, bitwise exact replay of 1,800 episodes, 96% family probe | pass |
| 2 continuous opponent BC | `mopa.context`, `mopa.bc_continuous`, `scripts/run_bc_continuous.py` | Unit tests; 3 folds × 3 seeds experiment | partial: oracle and closed-loop criteria pass, strict offline `real_c` criterion fails on one fold |
| 3 simulator-backed planner | `mopa.sim_planner`, `mopa.evaluation`, `scripts/run_sim_planner.py` | `tests/test_mppi.py`; 9 fold × opponent groups | pass |
| 4 TD-MPC world model | `mopa.tdmpc` (implicit / conditioned / factored, identity or learned encoder, same-transition continuation head), `mopa.tdmpc_data`, `scripts/run_tdmpc.py` | `tests/test_tdmpc.py` (modes, EMA target, causal inputs, context invariance, termination masking); model error vs persistence and calibration reports per run | see experiment record |
| 5 matched comparison | `scripts/run_tdmpc.py compare` | Paired matched-reset deltas across modes × seeds, factored action-clamp invariance, claim gates | see experiment record |

Deferred within this programme: sampled (non-deterministic) red opponents in
factored rollouts, a calibrated tanh-Gaussian red head, online data collection
with the learned planner, opponent switching, co-training, and the cyclic /
Spatial Blotto tasks.

## Evidence boundary

Implementation is not a scientific result. The repository includes one
source-bound, multi-seed [BC experiment](../experiments/bc/README.md), but not
the source checkpoints, raw rollout dataset, or learning curves. Its evidence
supports only the reported BC comparison. Other comparative claims require a
full run with at least three specialist, encoder, and policy seeds, one fixed
prey checkpoint family, fresh rollouts, fixed checkpoint-held-out splits, and
manifests that bind every result to its checkpoints and clean commit.

The following must be true before claiming that strategy was recovered:

1. The held-out latent probe beats chance, the random-encoder control, and the
   survival-time shortcut with uncertainty intervals.
2. Held-out GMM ARI is positive and stable across encoder seeds.
3. GRU-JEPA is compared with window JEPA, beta-VAE, and the supervised oracle
   on the same examples, feature schema, headline context prefix, and split;
   longer anytime points report each fixed-window model's effective prefix.
4. Belief-mixture action NLL is evaluated from the predictive belief before
   the current action, never from a full-episode latent.
5. Smoke stages remain `smoke_passed`. `full_run_finished` records that the
   declared computation ended; it is not a scientific success label.
6. `0s` legacy-forward runs are parity checks only. Scientific comparisons use
   past-only displacement, report every declared encoder seed, and distinguish
   full-episode post-hoc scores from prefix-time evidence.

## Deferred research integrations

These appeared in older or below-the-divider planning material but conflict
with the latest explicit removal of planners "for now," are mutually exclusive
alternatives, or lack an implementable protocol:

- DreamerV3, MA-TDMPC, and MuZero/MCTS planners (TD-MPC2 is now implemented;
  see the continuous programme above).
- End-to-end model-based co-training.
- GAMMS capture-the-flag and SMAC/SMAX adapters.
- Three-predator coordinated-trapping experiments.
- A real CPL preference dataset generated by a trained planner.
- Open-set and learning-opponent experiments with a declared switch schedule.

The CPL counterfactual API is implemented so a planner can plug in without
changing data integrity rules; the TD-MPC controllers have not yet been wired
into it. Nothing here is presented as completed co-training.

## Open experiment decisions

The current protocol does not specify the preference ordering for CPL, success
thresholds for “clearly distinguishable,” full training budgets, confidence
interval convention, adaptive-opponent schedule, or exact planner choice.
Those decisions must be recorded in a run config before the corresponding
experiment can be called complete.
