# Results and development record

Updated October 5, 2026. The targeted control continuation is complete; the
broad experiment campaign remains paused. **The original MPPI runaway behavior was
substantially reduced, but a fully repaired controller is not established.**

## Targeted control continuation

Protocol `resl_control_diagnostic_v2` is bound to specification
`44915520b507df5e0ba78b54da0ca4e7d3b81724` and executable
`f0973e690ed462a55e516d707dc3248dfe1c69f4`. Six original P2 controllers (implicit,
BC and recurrent `0s`, fitting seeds 11/22) were restored at 128 updates. They
received 1,872 updates on identical replay, then six rounds of 1,800 training-family
transitions and 1,000 updates: 8,000 total updates per controller. Rewards, soft
boundaries, opponent models and planner settings stayed fixed. P11 was off.
The later phase combines data and fitting; it cannot isolate either cause.

Each row below is one fit, averaged over 24 episodes (three specialists × four
reused and four fresh resets). Entries show 128→8,000. Episodes are not independent
fitting seeds. Fresh resets reuse the inspected opponent family.

| Arm / seed | MPPI return | Boundary cost | Capture fraction | Resources per episode |
| --- | --- | --- | --- | --- |
| implicit /11 | −1037.49→−8.07 | 1038.74→8.70 | .042→.500 | .33→1.12 |
| implicit /22 | −951.90→−9.63 | 952.73→16.72 | .042→.333 | .25→2.08 |
| BC /11 | −996.00→−31.53 | 997.25→37.36 | .042→.417 | .33→2.00 |
| BC /22 | −1003.14→−18.08 | 1004.39→20.79 | .042→.500 | .33→1.54 |
| 0s /11 | −975.41→−48.79 | 976.66→55.25 | .042→.417 | .33→2.12 |
| 0s /22 | −900.02→−11.41 | 901.27→14.33 | .000→.292 | .25→1.17 |

All 36 MPPI objective/reset groups improved return, boundary cost and outside
fraction, but capture worsened in 23 groups; one group collected fewer resources.
The unchanged-replay 2,000-update stage worsened policy-only boundary cost in all 36
groups. At 8,000, policy-only boundary cost remained worse in 12 groups, including
worse pooled cost for implicit/22, 0s/11 and 0s/22. These failed comparisons remain
explicit failures.

On the fixed original failure traces, overall/outside reward MAE and RMSE
improved for all six models. With deployed implicit/learned-red input, inside-bin
MAE and RMSE worsened in all six; edge-bin errors worsened in five (BC/11 is the
exception). With recorded-red input, 0s/11 interior MAE improves slightly while
its RMSE worsens; this exception remains in the full table. Fixed edge-scene H3 return MSE
also worsened in five fits (0s/22 is the exception). These calibration failures
prevent a general repair claim. Initial candidate scenes have tied true returns;
their rankings are unavailable. Learned Q tails are reported separately from
realized three-step rewards. No Q-calibration or opponent-model benefit follows.

Independent verification reconstructed raw aggregates from all 216 evaluation
trace files (864 episodes). Its predeclared numerical sample replayed 72 files
(288 episodes, 24,418 transitions), plus 36 training traces (2,728 transitions).
Within that sample, the maximum absolute differences for actions, states and
rewards were zero; the maximum causal-context difference was `7.1525574e-7`.
Blue-policy reproduction applies to the evaluation sample; training replay used
recorded blue actions. It also checked all 54 reward outputs, including fixed-row forward predictions,
and all 105 candidate arrays with their complete 512-score populations. The
largest candidate-score difference was approximately `6.546e-5`, within the
original tolerances. These checks verify the declared evidence and sampled
numerical paths; they do not establish bug-free or fully repaired control.

The [private result record](https://github.com/rlogger/marl-private/blob/codex/resl-audit-20261005/experiments/2026-10-05-comprehensive-audit/CONTROL.md)
retains every endpoint, objective/reset/mode, reward bin, failure and independent
review. [Artifacts](https://github.com/rlogger/marl-private/releases/tag/resl-control-diagnostic-20261005)
require private-repository access. [Reproduction](control-pipeline.md#targeted-control-diagnostic)
uses the exact source/protocol pair, not the latest checkout. The first attempt
remains failed and preserved separately; the second retains its original budgets
and numerical tolerances.

This is targeted diagnostic evidence, not the prescribed main comparison,
statistical significance, fresh-opponent generalization, calibrated uncertainty,
adaptive co-training, learned Blotto control or SOTA. It does not reconstruct the
unavailable source/artifacts of the reported MLP-VAE experiment.

## Historical September results

Historical gate labels are not current acceptance certificates. The strict
continuous-BC offline criterion failed on one fold; favorable closed-loop signs
do not repair it. Nine fold × opponent groups are not nine independent fits.
Historical outputs remain attached to their actual source versions.

Snapshot: September 8, 2026, published `td-mpc2` implementation at `fc0dc64`
(code unchanged by the README commit `84a38de`). All numbers below come from
saved reports and JSON artifacts. This documentation audit did not rerun training.

The discrete `0s` port matches the source implementation on identical data.
Continuous `0s` separates trajectory labels well, but its decoder does not yet
reproduce three distinct specialist behaviors. Controller adaptation improved
held-out return substantially; it still trails the saved MAPPO control.

## How the implementation developed

| Stage | Implementation and decision | Evidence |
|---|---|---|
| Specialist environment and data | Capture, risk-averse, and curious MAPPO policies; matched resets, checkpoint-held-out splits, exact replay | [Environment](ENVIRONMENT.md), [main-environment report](../experiments/main_env_20260908/REPORT.md) |
| Discrete `0s` source port | Ported Shashank's short-window state/action VAE and action decoder; compared source and port on the same data before using causal history | [Source comparison](../experiments/shashank_comparison_20260908/REPORT.md) |
| Vanilla and conditioned BC | Shared two-layer MLP with no context, causal context, shuffled context, and oracle controls; these earlier studies used GRU-JEPA, not `0s` | [Discrete BC](../experiments/bc/README.md), [earlier continuous gates](../experiments/continuous/README.md) |
| TD-MPC2 compatibility port | Pinned `ShaneFlandermeyer/tdmpc2-jax@5b05ff4`; consolidated core into `tdmpc.py` and `mppi.py`, replaced SimNorm reshaping and TFP distribution dependencies | [Source mapping and parity](../third_party/tdmpc2-jax/UPSTREAM.md) |
| Continuous data and control | Two-dimensional bounded actions; episode-bounded replay, termination handling, state normalization, and identity-state dynamics | [README execution path](../README.md), [continuous gates](../experiments/continuous/README.md) |
| Continuous `0s` + Equation 3 | Frozen 8D causal opponent context and action decoder; train the controller's dynamics, reward, continuation, Q, and policy heads | [Three-fit comparison](../experiments/shashank_comparison_20260908/REPORT.md) |
| Diagnose failed control | Recorded real rollouts and exact replay; identified poor coverage and underestimated boundary penalties as a model-bias hypothesis | [Original-environment report](../experiments/original_env_20260908/REPORT.md) |
| Adapt the controller | Add controller-generated data against training specialists; keep `0s`, normalization, environment, and planner configuration fixed | [Adaptation protocol](../experiments/original_env_20260908/PROTOCOL.md), [manifest](../experiments/original_env_20260908/adapted/manifest.json) |
| Restore main's environment and rerun | Preserve `new-env`, restore the two core files byte-for-byte from `main@8c24db3`, regenerate datasets and re-evaluate the controller | [Restoration provenance](../third_party/marl-opp-aware/UPSTREAM.md), [rerun](../experiments/main_env_20260908/REPORT.md) |

The restoration produced identical data and controller outcomes. It did not
cause the return improvement. The environment has a soft boundary penalty, not
a wall; a controller can leave the visible arena and incur large negative reward.

The TD-MPC source port is not an unmodified upstream training system. Gate 0
matched deterministic expressions and parameter initialization. Distrax and TFP
consume random keys differently, so stochastic policy samples and downstream
planning/update outputs are not claimed bitwise identical. Later identity-state,
continuation, replay, and opponent-mode extensions are documented separately.

## What the September model represents

`x` is the normalized 66D Markov state. `z` is an 8D vector inferred by the frozen
`0s` encoder from completed state/action history; it is zero before any history
is available. At action time `t`, it contains no action from time `t` or later.
Capture, risk, and curious prototypes are three training-set mean vectors in
this same space, not three encoders or class probabilities.

| Equation | Transition | Published execution path |
|---|---|---|
| 1: implicit | `x_next = dynamics(x, blue_action)` | Standalone `run_tdmpc.py train --mode implicit`; no opponent model required |
| 2: context-conditioned | `x_next = dynamics(x, blue_action, z)` | Generic conditioned driver uses the older 3D GRU-JEPA context, not the 8D `0s` checkpoint |
| 3: factored | `red_action = decoder(x, z)`; `x_next = dynamics(x, blue_action, red_action)` | Current `run_0s_world_model.py` and `adapt-0s` workflow |

Only the Equation 3 `0s` workflow produced the historical adaptation results below.
The real red opponent is still a frozen MAPPO policy. Changing `z` changes the
model's prediction, not the real opponent's policy. Context is held fixed within
each imagined planning horizon and refreshed after real transitions.

## Actual specialist behavior

Fresh main-environment datasets contain 1,800 episodes each: three objectives ×
three checkpoint seeds × 200 resets. Capture rate is the fraction of episodes
ending in capture; lava steps and coverage are per-episode predator averages.

| Action space | Specialist | Capture rate | Lava steps | Coverage, cells |
|---|---|---:|---:|---:|
| Discrete | Capture | 95.17% | 8.777 | 14.34 |
| Discrete | Risk | 4.50% | 0.342 | 8.92 |
| Discrete | Curious | 13.33% | 10.693 | 39.73 |
| Continuous | Capture | 99.50% | 6.577 | 13.65 |
| Continuous | Risk | 4.00% | 0.255 | 12.18 |
| Continuous | Curious | 14.50% | 11.592 | 49.18 |

Sources: [discrete report](../experiments/main_env_20260908/discrete/REPORT.md),
[continuous data report](../experiments/main_env_20260908/continuous_data/report.json).
These policies show aggregate pursuit, avoidance, and exploration differences.
This is evidence about the real specialists, not fidelity of the learned decoder.

The datasets contain 137,698 discrete and 130,869 continuous valid transitions.
The regenerated continuous dataset matched all 26 stored arrays, with exact
simulator replay for all 1,800 episodes. Native discrete regeneration also
matched the prior dataset and source/port metrics.

## Discrete `0s`: source comparison

| Verification | Episode probe | GMM ARI | Window/unit probe | Decoder accuracy |
|---|---:|---:|---:|---:|
| Shashank published | 0.785 | 0.435 | 0.547 | 0.829 |
| Same current data: upstream and integrated port | 0.755 | 0.538 | 0.566 | 0.789 |
| Causal protocol, three fit seeds | 0.734 ± 0.046 | 0.525 ± 0.017 | 0.581 ± 0.031 | 0.763 ± 0.001 |

Source: [comparison JSON](../experiments/shashank_comparison_20260908/comparison.json)
and its [protocol report](../experiments/shashank_comparison_20260908/REPORT.md).
The source-compatible upstream and integrated runs had zero error in all four
reported metrics. The published numbers were not reproduced exactly on changed
data; that is distinct from a port discrepancy.

The audited source was `hegde95/opponent-modeling@7bae281091a96fc83cadad3671bef070bd371675`.
Its original dataset and checkpoint binaries were unavailable. The rerun used
current specialists, a fixed capture-prey policy, and resets matched across
objectives; the published collection used objective-specific prey policies and
label-offset resets. See the [source and dataset provenance](../experiments/shashank_comparison_20260908/upstream/provenance.json).

The final row reports mean ± population SD over three fits. Full-episode probes
are post-hoc, not online strategy estimates. The entire decoder-accuracy column,
including the causal-features row, measures posterior reconstruction using
target actions and pools training and held-out windows. It is not held-out
causal prediction accuracy. The separate held-out reconstruction metric for the
causal-features fits was 0.796 ± 0.002; that reconstruction also sees target
actions and is not a next-action forecast. Do not exchange these decoder metrics.

## Continuous `0s`: representation and action prediction

Three fit seeds, 1,200 training episodes and 600 held-out episodes; checkpoint
seed 2 held out. Each fit used 1,500 encoder updates and 2,000 world-model updates.
All figures are mean ± population SD over fit seeds, not confidence intervals.

| Metric | Result |
|---|---:|
| Full-episode strategy probe | 0.931 ± 0.031 |
| GMM ARI | 0.465 ± 0.035 |
| Window probe | 0.832 ± 0.025 |
| Eight-step causal prefix probe | 0.674 ± 0.016 |
| Episode-length-only probe | 0.648 |
| Causal-context squared action error | 0.491 ± 0.004 |
| Zero-context squared action error | 0.495 ± 0.002 |
| Correct-prototype squared action error | 0.477 ± 0.003 |
| Wrong-prototype squared action errors | 0.510 ± 0.005 / 0.510 ± 0.007 |

Action error is squared Euclidean error for the 2D opponent action, averaged
within episodes and then equally across objective types, not discrete accuracy.
The correct-prototype arm is an oracle that uses the known objective label.
Source: [continuous comparison and definitions](../experiments/shashank_comparison_20260908/REPORT.md).

In matched-state prototype tests, every fit preferred the risk prototype for
all three specialist targets: only one of three target labels matched its own
best prototype. Representation separation therefore has not established faithful
three-way behavior control. The zero-context result is an ablation of the same
decoder, not a separately trained vanilla-BC baseline.

## Closed-loop controller

The completed experiment trained controller seed 0, then evaluated on 24 matched
reset keys per opponent type against held-out specialist checkpoint 2. Each arm
has 72 evaluation episodes. The MAPPO control is the saved capture-prey policy,
not a newly trained equal-budget opponent-aware baseline.

| Held-out opponent | Offline `0s` TD-MPC | Adapted `0s` TD-MPC | MAPPO | Random |
|---|---:|---:|---:|---:|
| Capture | −598.35 | −2.43 | −0.61 | −10.58 |
| Risk | −597.05 | +8.70 | +26.57 | −13.83 |
| Curious | −569.25 | +6.04 | +25.85 | −8.52 |
| Mean across types | −588.22 | +4.10 | +17.27 | −10.98 |

These are undiscounted real-environment blue returns. Sources:
[before/after report](../experiments/original_env_20260908/REPORT.md) and
[main-environment evaluation JSON](../experiments/main_env_20260908/controller/evaluation.json).
The main-source rerun reproduced the adapted results exactly.

| Budget | Completed run |
|---|---:|
| Offline training episodes / valid transitions | 1,200 / 86,899 |
| Additional controller episodes / valid transitions | 288 / 24,703 |
| Total replay transitions after adaptation | 111,602 |
| Initial controller gradient updates | 2,000 |
| Additional updates | 6 rounds × 1,000 = 6,000 |
| Collection per round | 8 episodes × 3 types × 2 training checkpoints |
| Batch size / planning horizon | 128 / 3 |
| MPPI candidates / policy-prior proposals / elites / iterations | 512 / 24 / 64 / 6 |
| Q networks / discount | 5 / 0.99 |

Adaptation froze `0s`, normalization, rewards, and planner settings. Its recorded
wall time was 2,133.2 seconds (35.6 minutes), including host pauses; this is not a
portable speed estimate. Specialist training was a separate budget of 2 million
joint environment transitions per seed and objective, 18 million across the
nine fits. The controller is not an 111,602-sample end-to-end system when its
specialist-data generation cost is included.

Adaptation improved 69 of 72 matched episodes. The adapted mean's 95%
reset-cluster bootstrap interval was [0.37, 7.71], conditional on this single
fitted controller. This is not uncertainty across training seeds. The experiment
supports improvement from additional controller experience and updates, not
superiority over MAPPO, a causal `0s` advantage, or state-of-the-art performance.

### Training and planning diagnostics

Final minibatch loss snapshots, before and after adaptation:

| Loss | Offline end | Adapted end |
|---|---:|---:|
| State consistency | 0.012452 | 0.006778 |
| Reward | 0.412139 | 0.327160 |
| Value | 0.926750 | 1.045815 |
| Policy prior | −1.720626 | +0.083199 |
| Continuation | 0.013326 | 0.019489 |

Sources: [offline training history](../experiments/shashank_comparison_20260908/continuous/seed_0/training_history.json),
[adaptation manifest](../experiments/original_env_20260908/adapted/manifest.json),
and [report](../experiments/original_env_20260908/REPORT.md). These are training
snapshots on changing replay distributions, not held-out errors. Consistency is
normalized-state prediction error in this identity-state model. Reward and value
use categorical two-hot losses; their values are not errors in reward units.
The policy prior maximizes entropy-regularized learned Q, not imitation of MPPI
action sequences. Its loss need not decrease monotonically or remain negative.

A diagnostic on the old six-episode smoke traces found lower reward error
outside the arena after adaptation. These are not the 72-episode evaluation
traces. This supports improved calibration on those states, not a measurement
of full planner-return bias. See the
[before](../experiments/original_env_20260908/reward_diagnostic.json) and
[after](../experiments/original_env_20260908/reward_diagnostic_adapted.json) records.

Saved evaluation contains episode lengths, capture/timeouts, action traces, and
planning latency. It does not contain proposal-origin labels or the complete
predicted return of each executed candidate sequence. Those diagnostics require
instrumentation and a new authorized run. Policy proposals number 24 while there
are 64 elites, so their maximum elite share is 37.5%, not 100%.

## Earlier studies: do not combine with current `0s` results

The [discrete BC experiment](../experiments/bc/README.md) used causal GRU-JEPA
features, three checkpoint folds, and three paired seeds—not the `0s` encoder.

| BC arm | Offline NLL ↓ | Accuracy ↑ | Closed-loop ADE ↓ | Expert agreement ↑ |
|---|---:|---:|---:|---:|
| Vanilla | 0.890 | 0.723 | 1.209 | 0.368 |
| GRU-JEPA latent | 0.902 | 0.727 | 1.084 | 0.412 |
| Shuffled | 0.893 | 0.723 | 1.218 | 0.364 |
| Oracle | 0.829 | 0.796 | 0.697 | 0.686 |

The full gate did not pass: latent conditioning improved closed-loop behavior
but worsened offline NLL on two folds. The source report gives dispersion and
the exact aggregation protocol; these means are not a new rerun.

The [earlier continuous gate study](../experiments/continuous/README.md) also
uses a different representation and protocol: 3D GRU-JEPA context, relative
features, and later controller experiments with 22,000 updates and three seeds.
Its stronger returns must not be presented as results for the September 8D `0s`,
66D-state, 8,000-update, one-seed controller.

## Remaining evidence gaps

- No completed equal-budget comparison establishes a return benefit from `0s`
  over implicit dynamics or a separately trained vanilla-BC opponent model.
- The September adaptation result is one training seed and one held-out checkpoint
  family, not a generalization benchmark across environments.
- Open-loop imagined trajectories and prototype plots are model diagnostics,
  not executed control. Local three-equation inspection work used 2,000-update
  offline Equation 1/2 models and the original Equation 3 base, not the adapted
  controller; that work was not part of the published evaluation.
- The newer matched-control benchmark and standalone numerical verifier were
  paused/incomplete at this documentation snapshot.
- The risk MAPPO preset uses `DENSE_CHASE_COEF: 0.1`, while collection uses the
  restored environment defaults, including `0.0`. Preserve and report effective
  train/evaluation configurations rather than calling them identical.

For transferring data or rerunning a fitted model, use the
[collaborator guide](COLLABORATOR_GUIDE.md), not the historical output paths in
old command logs.
