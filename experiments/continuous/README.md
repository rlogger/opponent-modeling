# Continuous opponent-aware TD-MPC: gate results

This directory records the experiments run for the continuous-action,
opponent-aware TD-MPC programme described in [`handoff.md`](../../handoff.md).
Every number below comes from the JSON files in this directory, which were
produced by the scripts named in each section on one Apple M4 Pro (CPU JAX
0.4.38). Raw datasets, checkpoints, and per-episode arrays live under
`artifacts/` and `logs/` (not committed); manifests bind each result to their
SHA-256 hashes and to the git commit.

Evidence boundary: everything here is a single-machine, three-seed,
one-held-out-checkpoint-family study on the objective-typed SimpleTag task. It
supports the gate decisions stated below and nothing broader (no SOTA or
TD-MPC2-replication claim).

## Gate 0 — upstream parity

Behaviour-preserving port of `ShaneFlandermeyer/tdmpc2-jax @ 5b05ff4`; see
[`third_party/tdmpc2-jax/UPSTREAM.md`](../../third_party/tdmpc2-jax/UPSTREAM.md)
for the bitwise comparison table. Passed. Fixed-seed golden values from the
verified port are asserted in `tests/test_tdmpc_upstream.py` so later gates
cannot silently alter the implicit single-agent path.

## Gate 1 — continuous environment and specialists

Files: [`gate1_training_curves.json`](gate1_training_curves.json),
[`gate1_dataset_report.json`](gate1_dataset_report.json).
Scripts: `scripts/train_mappo.py alg=mappo_continuous_{capture,risk,curious} NUM_SEEDS=3`,
`scripts/make_continuous_dataset.py`.

- Action contract: `(a_x, a_y) ∈ [-1, 1]²` converted at the boundary by
  `tag_objectives.to_mpe_action`; the decoded force equals the original action
  (tests cover zero, axes, diagonals, bounds, batches, `jit`, `vmap`).
- MAPPO: tanh-squashed diagonal Gaussian, joint log-probability summed over the
  two dimensions with the tanh Jacobian, pre-squash samples stored for exact
  PPO ratios. 2M steps × 3 seeds per family, ~100 s per family.
- Dataset: 1,800 episodes (3 objectives × 3 checkpoint seeds × 200 matched
  resets), horizon 100, 130,869 valid transitions, 66-D Markov state, exact
  17-D red and 35-D blue observations, separate `terminated_capture` /
  `truncated_timeout`. Deterministic policy means for both teams.
- Exact replay: all 1,800 episodes re-stepped from their reset and step keys
  with the stored joint actions reproduce state, observations, and reward with
  max abs error **0.0** and matching termination flags.

| Family | Capture rate | Survival (steps) | Predator lava steps | Predator coverage (cells) | Prey resources |
|---|---:|---:|---:|---:|---:|
| capture | 0.995 | 27.4 | 6.58 | 13.6 | 2.22 |
| risk | 0.040 | 98.2 | 0.26 | 12.2 | 6.24 |
| curious | 0.145 | 92.4 | 11.59 | 49.2 | 6.30 |

A leave-one-checkpoint-out logistic probe from the per-episode behaviour vector
to the objective label scores **0.963** (chance 0.333; folds 0.948 / 0.965 /
0.977): the three continuous families are behaviourally distinguishable on
capture, lava, coverage, and resource metrics. Saturation check before fitting
any model: 87.1% of blue and 65.6% of red transitions have at least one action
axis with |a| > 0.95 (tanh policies are close to bang-bang); both teams visit
every cell of a 16×16 arena grid.

**Gate 1: pass.**

## Gate 2 — continuous opponent behaviour cloning

File: [`gate2_bc_results.json`](gate2_bc_results.json).
Script: `scripts/run_bc_continuous.py` (defaults).

Model `v̂ = tanh(MLP(red_observation, c))`, MSE loss, 2×128 trunk, 4,000 steps,
three-column condition slot; GRU-JEPA context encoder (latent 2, hidden 32,
5,000 steps) trained per fold on training episodes only; `c_0 = 0`, `c_t` from
transitions before `t`. Three leave-one-checkpoint-out folds × three paired
seeds. Closed loop replays the exact 10-step expert prefix and then lets the
cloned predator act against the frozen prey on all matched held-out episodes
(≈ 200 per objective per fold). Values are mean ± std over fold means.

| Arm | Held-out MSE ↓ | Direction cosine ↑ | Agreement (L2 ≤ 0.25) ↑ | Closed-loop ADE ↓ | FDE ↓ | Expert agreement ↑ | Capture gap ↓ | Survival gap ↓ | Lava gap ↓ | Coverage gap ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `no_c` | 0.513 ± 0.232 | 0.790 ± 0.096 | 0.404 ± 0.095 | 1.231 ± 0.064 | 2.361 ± 0.110 | 0.191 ± 0.052 | 0.225 ± 0.013 | 15.63 ± 1.27 | 2.54 ± 0.55 | 11.48 ± 1.46 |
| `real_c` | 0.514 ± 0.261 | 0.793 ± 0.104 | 0.408 ± 0.093 | 1.135 ± 0.088 | 2.188 ± 0.164 | 0.221 ± 0.063 | 0.216 ± 0.013 | 14.94 ± 1.09 | 2.37 ± 0.28 | 10.07 ± 1.83 |
| `shuffled_c` | 0.523 ± 0.231 | 0.787 ± 0.095 | 0.400 ± 0.092 | 1.232 ± 0.063 | 2.345 ± 0.114 | 0.186 ± 0.050 | 0.231 ± 0.020 | 16.09 ± 1.49 | 2.53 ± 0.48 | 11.94 ± 1.51 |
| `oracle` | 0.332 ± 0.182 | 0.870 ± 0.061 | 0.572 ± 0.081 | 0.883 ± 0.188 | 1.651 ± 0.193 | 0.479 ± 0.101 | 0.026 ± 0.009 | 2.24 ± 0.30 | 1.07 ± 0.17 | 1.21 ± 0.25 |

Paired fold deltas (MSE, lower is better): `real_c − no_c` = −0.020 / +0.043 /
−0.020; `real_c − shuffled_c` = −0.025 / +0.034 / −0.034; `oracle − no_c` =
−0.147 / −0.251 / −0.145. Closed-loop ADE: `real_c − no_c` = −0.146 / −0.037 /
−0.104; `real_c − shuffled_c` = −0.119 / −0.058 / −0.116.

- Oracle beats `no_c` on every fold offline and closed-loop: **type information
  is action-relevant**, so the handoff's stop rule is not triggered.
- `real_c` beats `no_c` and `shuffled_c` closed-loop on every fold (ADE and
  expert agreement) but offline only on folds 0 and 2; on the hard fold 1
  (held-out checkpoint 1) it is worse.

**Gate 2: partial pass** — closed-loop and oracle criteria met, strict
offline criterion not met (same pattern as the discrete BC experiment). The
2-D unsupervised GRU-JEPA context is the weak component; it was not tuned
(no hyper-parameter sweep by handoff decision).

## Gate 3 — simulator-backed planner

File: [`gate3_sim_planner_results.json`](gate3_sim_planner_results.json).
Script: `scripts/run_sim_planner.py --n-eps 48 --population 128 --elites 16 --iterations 3 --horizon 10`.

MPPI over blue action sequences with the exact JaxMARL simulator as the model,
red actions from an opponent model at every imagined step, zero terminal
value, context held fixed inside the horizon and recomputed before every real
step. 48 matched resets per opponent type per held-out checkpoint (3 × 3 = 9
groups). Mean blue return (sum of prey reward; capture rate in parentheses),
averaged over the three checkpoint folds:

| Blue controller | vs capture | vs risk | vs curious |
|---|---:|---:|---:|
| frozen MAPPO prey (fixed policy) | 0.49 (1.00) | 27.96 (0.03) | 27.79 (0.14) |
| random | −20.05 (0.54) | −14.01 (0.02) | −13.82 (0.26) |
| planner + true red | **24.72 (0.00)** | **33.13 (0.00)** | **36.48 (0.00)** |
| planner + oracle BC, correct one-hot | 19.16 (0.26) | 33.17 (0.00) | 35.60 (0.07) |
| planner + oracle BC, zero context | 15.17 (0.47) | 32.78 (0.01) | 35.16 (0.11) |
| planner + oracle BC, wrong one-hot | 15.24 (0.45) | 33.02 (0.01) | 31.82 (0.19) |
| planner + `real_c` BC, online context | 15.48 (0.44) | 33.20 (0.00) | 34.66 (0.07) |
| planner + `real_c` BC, zero context | 16.56 (0.46) | 32.96 (0.01) | 36.25 (0.07) |
| planner + `real_c` BC, shuffled context | 16.30 (0.45) | 32.96 (0.00) | 34.90 (0.07) |
| planner + `no_c` BC | 15.90 (0.44) | 33.20 (0.00) | 34.31 (0.08) |

- Planner + true red beats the fixed policy in all 9 groups (mean +12.7
  return; against the capture predator it is never caught where the MAPPO prey
  is always caught) and random in all 9 groups (+47.4). Actions stayed within
  bounds and no controller acted after termination.
- Opponent-model error matters most against the capture predator: the oracle
  BC with the correct one-hot recovers most of the true-red return (19.2 vs
  24.7) and beats the wrong one-hot in every fold against capture and curious
  predators (+2.6 mean); the causal `real_c` context gives no consistent
  benefit over zero/shuffled/`no_c` (−0.8 / −0.3 / −0.0), consistent with
  Gate 2.

**Gate 3: pass** (planner + true red exceeds both controls; bounds and
termination respected). The context controls show that only *correct* type
information helps planning here.

## Gate 4 / Gate 5 — learned world model and matched comparison

See [`gate45_identity_comparison.json`](gate45_identity_comparison.json) and
[`gate45_mlp_comparison.json`](gate45_mlp_comparison.json) once present; the
sections below are filled from those files.
