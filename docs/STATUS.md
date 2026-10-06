# Implementation status

Updated October 5, 2026. `marl-private` governs adopted project requirements and
documented audit remedies. This file records implementation and evidence status;
older implementation notes do not supersede those private instructions.

## Current audit and experiment status

The integration branch combines the continuous-control implementation with the
existing [Spatial Blotto environment](spatial-blotto.md). Historical branches,
experiments and unpublished source snapshots are preserved. Reviewed changes and evidence are published on the audit branch. Canonical
consolidation and the prescribed main experiments remain paused.

Before the new local pilot, the locked suite passed **787 tests, with no skips**,
and Ruff passed. The pilot used executable commit
`99bfaa7a8c4b3e83ecbd8a55a3f481325eef2506` and specification commit
`0f86e0dce7b622e40968474a859790a71eaf9072`, protocol `RESL-20261005-P2`.
It completed eight small causal-prediction fits and ten controller arms across
two fitting seeds. These runs establish feasibility; they do not satisfy the
main experiment budgets or establish any performance hypothesis. Post-pilot
corrections require their own locked checks before subsequent runs.

The broad campaign remains on hold. The targeted continuation completed six
controllers through 8,000 updates and 864 evaluation episodes. MPPI boundary costs
fell substantially, but capture and several policy-only/calibration outcomes
worsened. [Results](RESULTS.md#targeted-control-continuation) retain the failed
hypotheses; fully repaired control is not established. The frozen run source
passed 875 locked tests with no skips locally and in remote CI before fitting.
The exact-protocol-pin correction is separate from that frozen source and does
not change its valid bindings.

The deep current-tree review of `marl-opp-aware` at
`aecbab5daf6da402029e953be114a52e62a46c26` covered 163 files/24,900 lines.
Original/current numerical parity passed 384 transitions across 42 scenes,
including boundary, capture, collection and timeout. The game retains soft
boundary penalties. Historical discrete actors and true-simulator planners have
different task/action/budget contracts; they are not matched TD-MPC baselines.
See [source provenance](../third_party/marl-opp-aware/UPSTREAM.md).

The optional P11 static/clock contract is implemented and independently reviewed;
it has focused mathematical, gradient, timeout and checkpoint tests. It has no
fitted comparative result yet and is disabled in the unchanged-model fitting
diagnostic. [The contract and diagnostic commands](control-pipeline.md#targeted-control-diagnostic)
separate these two interventions. The fitting diagnostic produced mixed results; it does not certify a full
repair or substitute for the prescribed main comparison.

Implemented audit remedies include episode-bounded causal context and replay
checks, explicit finite-update checks, saved optimizer/RNG state, separately
trained causal MLP/GRU prediction comparisons, matched controller collection,
and withdrawal of automatic scientific success labels. Unit tests establish
these contracts; empirical acceptance remains tied to the corresponding run.

| Private instruction | Current evidence boundary |
|---|---|
| A01: frozen trained TD-MPC prefix/continuation diagnostic | Prescribed numerical and visual comparison remains pending; historical prototype swaps are different evidence. |
| A02: world representation on/off, global state when off | Identity and learned MLP paths have implementation tests. A matched trained comparison remains pending; opponent-encoder comparisons do not close this instruction. |
| A03: randomly sample supplied pretrained opponents | New controller campaign implements per-episode selection and records identities; pilot execution is not the prescribed main result. |
| A04/A05: trajectory history and component integration | Causal timing, replay, attachment and checkpoint tests exist; final simulator and result verification remains required. |
| A06: actual opponent-weight updates through replay | Continuation of the original `0s` optimizer is implemented and tested; real-episode before/after validation remains pending. |
| A07: three-zone Spatial Blotto | Environment, mathematical/scripted controllers and terminal/export contracts are present. Learned Blotto control is not claimed. |
| A08: opponent uncertainty | Sampling and uncertainty diagnostics are explicit; calibrated uncertainty and novel-opponent claims require empirical evidence. |
| A09 and code-audit remedies | New comparisons remain distinct from the unavailable exact source/artifacts of the reported MLP-VAE experiment. |

No SOTA claim is supported. No historical failed criterion is changed to passed
by a smoke test, a later code correction, or completion of a different run.

## Historical September implementation snapshot

The following sections preserve earlier implementation and experiment records.
Their gate labels and deferrals describe those historical protocols. Current
scientific interpretation is given above and in [Results](RESULTS.md).

## Historical lean scope

The earlier scaffold update narrowed that stage to the objective-typed
environment, MAPPO specialists, trajectory representations, uncertainty-aware
opponent policies, behavior cloning, CPL foundations, evaluation, and
reproducible manifests. The following implementation checks were recorded for that stage:

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

## Historical continuous opponent-aware TD-MPC (handoff gates)

The continuous-action programme from [`handoff.md`](../handoff.md) is
recorded in the earlier implementation; its experiment record with all numbers is
[`experiments/continuous/README.md`](../experiments/continuous/README.md).

| Gate | Implementation | Verification | Status |
|---|---|---|---|
| 0 upstream parity | `mopa.tdmpc`, `mopa.mppi`, `configs/tdmpc2.yaml`, `third_party/tdmpc2-jax/` | Bitwise comparison against pinned `5b05ff4` for every deterministic path; golden fixed-seed values in `tests/test_tdmpc_upstream.py` | pass |
| 1 continuous env + specialists | `tag_objectives.actions`, `mopa.continuous`, `mopa.nets.ContinuousActor`, `scripts/train_mappo.py` (`ACTION_TYPE: Continuous`), `mopa.continuous_data`, `scripts/make_continuous_dataset.py` | Adapter tests (zero/axes/diagonals/bounds/batch/jit/vmap), tanh-Gaussian log-prob vs Distrax, data-contract tests, bitwise exact replay of 1,800 episodes, 96% family probe | pass |
| 2 continuous opponent BC | `mopa.context`, `mopa.bc_continuous`, `scripts/run_bc_continuous.py` | Unit tests; 3 folds × 3 seeds experiment | failed strict offline criterion: oracle and closed-loop signs were favorable, but do not satisfy the full requirement |
| 3 simulator-backed planner | `mopa.sim_planner`, `mopa.evaluation`, `scripts/run_sim_planner.py` | `tests/test_mppi.py`; 9 fold × opponent groups | historical pass label withdrawn as current acceptance; nine fold × opponent groups are not nine independent fits |
| 4 TD-MPC world model | `mopa.tdmpc` (implicit / conditioned / factored, identity or learned encoder, same-transition continuation head), `mopa.tdmpc_data` (episode-bounded replay, `relative` reward-relevant features, online append), `scripts/run_tdmpc.py` (offline updates plus online collection rounds against training families) | `tests/test_tdmpc.py` (modes, EMA target, causal inputs, context invariance, termination masking); model error vs persistence and calibration reports per run; 3 modes × 3 seeds experiment | pass for the identity-encoder state-space baseline: every mode beats persistence at 1 / 3 / 10 steps on the held-out family, reward EV 0.46–0.54, capture AUROC 0.99; learned `mlp` encoder not yet run |
| 5 matched comparison | `scripts/run_tdmpc.py compare` | Paired matched-reset deltas across modes × seeds, factored action-clamp invariance, claim gates; 32 matched held-out resets × 3 opponents × 3 seeds | historical nominal-pass label withdrawn as current acceptance: all modes beat the fixed prey and random; `factored` − `implicit` = +1.19 ± 4.77 return (6 / 9 pairs, seed-consistent only vs the capture predator); invariance exact in all factored seeds |

Historically deferred within that programme (not a current instruction to defer adopted A01–A08): the learned-encoder (`mlp`) comparison,
sampled (non-deterministic) red opponents in factored rollouts, a calibrated
tanh-Gaussian red head, an oracle-context training arm, opponent switching,
co-training, and the cyclic / Spatial Blotto tasks.

## Historical evidence boundary

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

## Historical deferred research integrations

The earlier scaffold listed the following deferrals. Later adopted private
instructions and frozen protocols govern their current status:

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

## Historical open experiment decisions

The current protocol does not specify the preference ordering for CPL, success
thresholds for “clearly distinguishable,” full training budgets, confidence
interval convention, adaptive-opponent schedule, or exact planner choice.
Those decisions must be recorded in a run config before the corresponding
experiment can be called complete.
