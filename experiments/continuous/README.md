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

## Gate 4 — TD-MPC-style state-space world model (identity encoder)

File: [`gate45_identity_comparison.json`](gate45_identity_comparison.json)
(`summary.<mode>__identity.model_quality`, aggregated from the nine run
manifests). Scripts, per mode ∈ {implicit, conditioned, factored} and seed ∈
{0, 1, 2}:

```bash
scripts/run_tdmpc.py train --mode $mode --encoder identity --features relative --heldout 2 \
    --seed $seed --updates 10000 --online-rounds 6 --online-episodes 8 --updates-per-round 2000
scripts/run_tdmpc.py evaluate $run --n-eps 32 [--controls]
scripts/run_tdmpc.py compare --root artifacts/tdmpc_identity_relative_online
```

Setup (`configs/tdmpc2.yaml` profile `gate4`; upstream planner defaults
unchanged: horizon 3, 512 candidates, 24 policy-prior samples, 64 elites, six
MPPI iterations, five Q networks, discount 0.99):

- **Encoder.** Identity over the normalised 111-D `relative` observation: the
  66-D Markov state plus prey-relative resource / predator / lava offsets,
  nearest-uncollected-resource offset and distance, predator distance, and
  boundary proximity — all deterministic functions of the state, so the vector
  stays Markov-equivalent. Per the handoff this stage is the **TD-MPC-style
  state-space baseline**, not decoder-free TD-MPC2. Dynamics are residual
  (`x + f(x, a)`); reward and Q heads are two-hot with 101 bins; the
  continuation head predicts `1 − terminated_capture` for the same transition.
- **Context.** Frozen Gate 2 GRU-JEPA causal context (fold 2, seed 0; 2-D
  latent zero-padded to the 3-D context slot), `c_0 = 0`, `c_t` from
  transitions before `t`. `implicit` receives zeros; `conditioned` feeds `c`
  to dynamics / reward / continuation / Q / policy prior; `factored` feeds the
  recorded 4-D joint action to dynamics / reward / continuation, learns
  `red(x, c) → v` on a detached latent (MSE), and gives `c` only to Q and the
  blue prior.
- **Data.** Checkpoint family 2 held out; 1,200 training episodes
  (86.9 k transitions) from Gate 1. After 10 k offline updates, six online
  rounds each collect 8 episodes × 3 opponents × 2 *training* checkpoint
  families (48 episodes) with the current planner (upstream training-time
  exploration, online causal context) and are followed by 2 k updates: 22 k
  updates and ≈ 24.3 k added transitions per run (≈ 111 k final). The held-out
  family is never collected.
- **Why relative features and online rounds.** The first attempt (raw 66-D
  Markov features, 20 k purely offline updates) is kept in
  [`gate45_v1_markov_features_implicit_s0_evaluation.json`](gate45_v1_markov_features_implicit_s0_evaluation.json):
  its reward head had *negative* held-out explained variance (−0.5) and the
  planner drove the prey out of the arena — return −660 / −529 / −476 against
  the capture / risk / curious predators, versus +0.8 / +25.7 / +26.4 for the
  fixed MAPPO prey and −19 / −6 / −27 for random. Relative features lift the
  reward head to +0.5 held-out explained variance after 4 k updates, and the
  online rounds put the planner's own state distribution into the replay.

Held-out-family model quality, mean ± std over the three seeds
(open-loop rollouts with the recorded joint actions from 4,096 held-out starts
per horizon; ratio = model MSE / persistence MSE in normalised-state units;
position RMSE in arena units, model / persistence):

| Mode | k = 1 ratio | k = 3 ratio | k = 10 ratio | k = 1 position RMSE | k = 3 | k = 10 |
|---|---:|---:|---:|---:|---:|---:|
| implicit | 0.698 ± 0.016 | 0.467 ± 0.001 | 0.315 ± 0.002 | 0.021 / 0.080 | 0.065 / 0.239 | 0.300 / 0.697 |
| conditioned | 0.706 ± 0.019 | 0.465 ± 0.002 | 0.311 ± 0.005 | 0.021 / 0.080 | 0.065 / 0.239 | 0.281 / 0.697 |
| factored | 0.673 ± 0.003 | 0.412 ± 0.002 | 0.269 ± 0.007 | 0.017 / 0.080 | 0.048 / 0.239 | 0.147 / 0.697 |

| Mode | Reward explained variance ↑ | Reward MAE ↓ | Continue Brier ↓ | Continue ECE ↓ | Capture AUROC ↑ | Capture recall @ 0.5 |
|---|---:|---:|---:|---:|---:|---:|
| implicit | 0.502 ± 0.024 | 0.247 ± 0.014 | 0.0205 ± 0.0006 | 0.0193 ± 0.0006 | 0.988 ± 0.000 | 0.27 ± 0.03 |
| conditioned | 0.456 ± 0.020 | 0.257 ± 0.002 | 0.0192 ± 0.0018 | 0.0175 ± 0.0022 | 0.987 ± 0.001 | 0.36 ± 0.07 |
| factored | 0.535 ± 0.006 | 0.235 ± 0.007 | 0.0198 ± 0.0006 | 0.0184 ± 0.0007 | 0.985 ± 0.001 | 0.34 ± 0.02 |

On the training families the same heads reach reward explained variance
0.81 / 0.79 / 0.79, capture AUROC 1.000, and capture recall 0.80 / 0.84 / 0.88
(implicit / conditioned / factored), so the held-out gap is a generalisation
gap to the unseen specialist checkpoints, not underfitting.

- Every mode beats persistence at every horizon on both splits.
- `factored` has the lowest multi-step error: with the recorded red action as
  an input, its 10-step held-out position RMSE (0.147) is half that of
  `implicit` (0.300), and its reward head is the best calibrated. This is the
  component-ladder rung "learned physics + true red actions".
- Capture is rare (226 of the 43,970 held-out transitions) and the
  continuation head ranks it well (AUROC ≈ 0.99) but is under-confident at the
  0.5 threshold (recall 0.27–0.36 held-out), i.e. it only partially stops the
  TD bootstrap and reward accumulation at capture during planning.

Online collection (mean collected return of the exploring planner over the six
training groups, then over seeds; the exploration noise makes these lower than
the deterministic evaluation below):

| Mode | round 0 | round 1 | round 2 | round 3 | round 4 | round 5 |
|---|---:|---:|---:|---:|---:|---:|
| implicit | −110.4 ± 40.5 | −2.6 ± 26.3 | 4.8 ± 22.6 | 2.1 ± 16.6 | 24.0 ± 1.9 | 24.7 ± 2.4 |
| conditioned | −88.3 ± 5.4 | −21.8 ± 13.8 | 24.3 ± 2.6 | 16.2 ± 5.3 | 18.0 ± 7.3 | 18.1 ± 0.6 |
| factored | −80.5 ± 16.5 | −42.5 ± 32.8 | 1.9 ± 3.8 | 17.8 ± 8.1 | 17.7 ± 3.7 | 23.3 ± 1.2 |

The purely offline model (round 0) still leaves the arena; two to three rounds
of the planner's own data are needed before the collected return turns
positive, which is the same failure the v1 record shows.

**Gate 4: pass** for the state-space baseline (model error beats persistence
at 1 / 3 / 10 steps, reward and termination heads are calibrated on held-out
resets and held-out checkpoints). The learned-encoder (`--encoder mlp`,
decoder-free) stage has not been run yet.

## Gate 5 — matched comparison (identity encoder)

File: [`gate45_identity_comparison.json`](gate45_identity_comparison.json)
(`summary`, `comparisons`, `factored_invariance`, `claim_gates`), produced by
`scripts/run_tdmpc.py compare --root artifacts/tdmpc_identity_relative_online`
from the nine `evaluation.json` files.

Protocol. The three modes were trained on identical data, seeds, capacity,
update counts, online-collection schedule, and planner settings. Each
controller acts deterministically (best elite) in the real environment against
the *held-out* checkpoint family on the first 32 matched dataset resets per
opponent type (same reset and step keys for every controller and mode). For
`conditioned` and `factored`, `online` recomputes the causal context from the
real history before every action; `zero`, `shuffled` (the online context of a
different episode in the same matched batch), and `wrong_oracle` (the one-hot
of a different type) are the controls. The `oracle` / `wrong_oracle` rows feed one-hot
vectors to models trained only on the 2-D causal latent, so they are
out-of-distribution probes, not a trained oracle arm.

Blue return (sum of prey reward), mean ± std over the three seeds of the
per-seed mean; capture rate in parentheses:

| Blue controller | vs capture | vs risk | vs curious |
|---|---:|---:|---:|
| frozen MAPPO prey | 0.78 (1.00) | 25.72 (0.00) | 26.44 (0.16) |
| random | −19.32 (0.34) | −6.23 (0.00) | −26.58 (0.19) |
| `implicit` | 3.41 ± 3.04 (0.67) | 29.91 ± 1.97 (0.02) | 33.92 ± 0.55 (0.06) |
| `conditioned`, online context | 3.30 ± 3.28 (0.62) | 33.82 ± 2.11 (0.03) | 33.46 ± 2.40 (0.12) |
| `conditioned`, zero context | 3.14 ± 0.91 (0.73) | 30.72 ± 1.93 (0.02) | 28.04 ± 5.24 (0.09) |
| `conditioned`, shuffled context | 3.77 ± 2.01 (0.59) | 33.14 ± 1.74 (0.02) | 35.51 ± 2.91 (0.09) |
| `conditioned`, wrong one-hot | 2.16 ± 1.38 (0.67) | 28.04 ± 3.00 (0.01) | 30.43 ± 1.93 (0.07) |
| `factored`, online context | **5.62 ± 1.76 (0.49)** | 33.29 ± 4.74 (0.01) | 31.91 ± 5.89 (0.09) |
| `factored`, zero context | 3.63 ± 2.35 (0.55) | 31.63 ± 5.03 (0.00) | 31.93 ± 4.73 (0.11) |
| `factored`, shuffled context | 4.27 ± 2.41 (0.54) | 32.11 ± 4.32 (0.01) | 31.28 ± 4.64 (0.09) |
| `factored`, wrong one-hot | 5.50 ± 3.18 (0.54) | 30.68 ± 3.49 (0.06) | 30.72 ± 2.54 (0.15) |

Survival (steps) and resources collected against the capture predator: MAPPO
prey 28.0 / 2.25; `implicit` 63.6 ± 1.6 / 2.73 ± 0.19; `conditioned` online
65.9 ± 7.3 / 2.85 ± 0.50; `factored` online 72.7 ± 5.4 / 2.69 ± 0.29. Against
the risk and curious predators every mode with online or zero context collects
6.1–7.2 resources (MAPPO prey 5.78) and is caught no more often than the MAPPO
prey (≤ 0.03 vs risk, ≤ 0.12 vs curious).

Paired matched-reset deltas of blue return (nine pairs = 3 seeds × 3
opponents; per-opponent values are the three seeds):

| Comparison | Δ mean ± std | pairs > 0 | vs capture | vs risk | vs curious |
|---|---:|---:|---|---|---|
| `factored` online − `implicit` | +1.19 ± 4.77 | 6 / 9 | +4.01, +1.44, +1.17 | −1.18, +4.80, +6.51 | −7.71, +6.76, −5.09 |
| `conditioned` online − `implicit` | +1.12 ± 4.67 | 4 / 9 | +7.60, −0.72, −7.19 | +5.87, −1.23, +7.09 | −1.55, −2.02, +2.20 |
| `factored` online − zero context | +1.20 ± 2.04 | 7 / 9 | | | |
| `factored` online − shuffled context | +1.05 ± 1.94 | 7 / 9 | | | |
| `factored` online − wrong one-hot | +1.30 ± 3.31 | 6 / 9 | | | |

Equation 3 requirements, evaluated on 64 real held-out latents per seed
(`factored_invariance`):

- fixed `(x, u, v)` + swapped `c` (each one-hot and zero): max absolute change
  of predicted dynamics, reward, and continuation logits = **0.0** for all
  three seeds (the context never enters those heads);
- fixed `x` + swapped `c`: mean red-action change 0.23–0.53 (actions in
  `[−1, 1]²`), so the opponent head does depend on context;
- flipping the red action changes the predicted next state by 0.83–0.91
  (normalised units) on average;
- MPPI optimises blue action sequences only (`tests/test_mppi.py`); all
  executed actions satisfy `|a| ≤ 1`, and episodes are frozen at capture so
  no later action affects any metric.

Planner latency: 21.1 ms per environment step on average (max 25.2 ms) for
512 candidates × 6 MPPI iterations × horizon 3 with five Q networks, CPU JAX
on the M4 Pro.

Reading:

- **Learned-model planning works.** Every mode beats the fixed MAPPO prey and
  random against all three opponents on return; against the capture predator
  the planners are caught in 49–73 % of episodes instead of 100 % and survive
  2–2.6× longer. This is the state-space (identity-encoder) baseline's main
  result. Against the capture predator the learned-model planners (3–6 return)
  remain far below the Gate 3 simulator-backed planner with the true red
  policy (24.7, horizon 10, no terminal value), so world-model error still
  costs most where opponent behaviour matters most.
- **`factored` vs `implicit`: nominal pass, weak evidence.** The mean paired
  gain is +1.19 return (6 / 9 pairs) but its spread across pairs (std 4.77,
  standard error ≈ 1.6) is larger than the effect. The gain is
  seed-consistent only against the capture predator (+1.2 to +4.0 in all
  three seeds, capture rate 0.49 vs 0.67, survival 72.7 vs 63.6 steps) —
  the opponent for which Gate 3 also showed that opponent-model quality
  matters — and mixed against the risk and curious predators, where the red
  head's error (final MSE 0.24 on near-bang-bang targets) is not compensated
  by any capture risk.
- **Context controls.** `factored` with online causal context beats its own
  zero, shuffled, and wrong-one-hot controls on the mean (7 / 9, 7 / 9,
  6 / 9 pairs; +1.0 to +1.3 return), which is the direction the handoff
  requires, but the margins are again within one standard deviation.
  `conditioned` shows no consistent benefit of context (4 / 9 pairs vs
  `implicit`, and its shuffled-context arm is as good as online context),
  in line with the Gate 2 finding that the 2-D unsupervised causal context is
  the weak component.
- Physics / reward / termination invariance to context holds exactly in all
  factored runs, and the red head and dynamics respond to context and red
  action respectively, so the factored model is structurally the model the
  handoff specifies; the limiting factor is the quality of the opponent head
  and of the 2-D context, not the factorisation.

**Gate 5: nominal pass** for the identity-encoder stage — `factored` improves
mean real-environment return over `implicit` and passes the action-clamp
invariance test with three seeds, which is the claim gate written in
`scripts/run_tdmpc.py compare`. It is **not** evidence of a robust
controllability effect: with nine pairs the improvement is not
distinguishable from zero, and the handoff's stronger reading ("correct
context outperforms zero, shuffled, and wrong context") holds only on the
mean. A better opponent head (Gate 2's oracle arm shows type information is
action-relevant) or an oracle-context training arm would be the next
discriminating experiment.

Provenance note: `implicit` seed 0 was evaluated from the clean commit
`ab9b70d`; the other eight evaluations and the comparison ran from the same
commit while the working tree contained the uncommitted compare-aggregation
and documentation edits of the following commit (`git_dirty: true` in those
files). The `evaluate` code path was not changed by those edits.

## Gate 4 / Gate 5 — learned encoder (`--encoder mlp`)

See [`gate45_mlp_comparison.json`](gate45_mlp_comparison.json) once present;
this section is filled from that file.
