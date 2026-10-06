# Matched continuous-control benchmark

**Paused by user: implementation and explanatory visualizations only.**
Do not execute this proposed experiment or publish new metrics until requested.
Earlier preparation artifacts are retained, not a completed benchmark. Source
changes after the pause intentionally invalidate their old experiment binding;
any future authorized run must use a fresh output directory.

Predeclared before held-out evaluation. Environment and rewards remain unchanged
from `main@8c24db3`, currently integrated on `td-mpc2@fc0dc64`.
Use the freshly verified continuous dataset in `experiments/main_env_20260908/continuous_data`;
its SHA256 is `928181027a9e5e86b106190b1487cda90c09e2cc5df08b569f7b93ba8350b5f5`.
The main-source rollback changed neither its arrays nor the environment mechanics.

## Questions and information contract

| Arm | Current state | Causal history input | Opponent prediction |
|---|---|---|---|
| implicit | Same normalized 66D | None | Implicit dynamics |
| bc | Same normalized 66D | None | Frozen vanilla MLP, current 8D features |
| 0s | Same normalized 66D | Frozen causal 8D 0s context | Frozen 0s decoder |
| ppo | Same normalized 66D | None | No world model; PPO response |
| ppo_z | Same normalized 66D | Exactly the same frozen 0s context | No world model; PPO response |

State-only equal-information comparisons are BC/implicit versus PPO. History
equal-information comparison is 0s versus PPO-z. Both PPO actor and critic receive
the same inputs. The old 35D MAPPO policy is not an equal-information control and
is excluded from the primary table. With one controlled blue agent and frozen
red opponents, the new baseline is a blue-only PPO response using the existing
MAPPO actor architecture, not reproduction of the old MAPPO training run or joint
co-training. Neither PPO arm receives privileged critic-only information.

0s versus BC/implicit intentionally changes history use. It does not isolate
latent representation quality from access to history. BC and the 0s decoder
receive the same eight current position/velocity features; only 0s additionally
receives context. Neither decoder changes during controller learning. Context
uses completed real state/action pairs only and is fixed inside imagined plans.
No strategy labels, current red action, future states, or held-out outcomes are
controller inputs. This is simulator-state/action-access research, not a
native-observation-only benchmark.

## Fixed budgets and split

- Training seeds 0, 1, 2; all controllers start from scratch. Independently fit
  0s and BC for each seed, 1,500 updates, batch 128, hidden width 64. The fitting
  objectives/architectures differ; this is not a parameter-count-matched claim.
- Same verified continuous dataset: 1,200 training episodes / 86,899 valid
  transitions from specialist checkpoints 0/1. Checkpoint 2 is held out.
  Training-only normalization and 0s artifacts are shared by paired arms.
- All controllers receive the same offline experience. TD-MPC gets 2,000 replay
  updates. PPO gets 2,000 explicit blue-action BC and finite-episode-return value
  warm-start updates; this is not an off-the-shelf, online-only PPO comparison.
- Six rounds, exactly **600 valid real transitions per objective/checkpoint
  group**: 3,600 per round; **21,600 additional interactions per controller**.
  Total available replay/experience is **108,499 recorded real transitions** per
  arm/seed, including the shared offline data; replay samples are not new data.
  Padded post-terminal steps do not count. Administrative quota cuts are recorded
  as truncations and bootstrapped, never mislabeled captures. Reset streams and
  group quotas match; resulting on-policy trajectories naturally differ.
- TD-MPC: 1,000 replay updates after each round, 8,000 total. PPO: four epochs of
  each fresh on-policy batch, minibatch 128, GAE lambda 0.95. Optimization and
  computation are reported separately, not claimed equal.
- All TD-MPC arms retain horizon 3, population 512, policy-prior samples 24,
  elites 64, six MPPI iterations, five Q networks, discount 0.99, identity
  normalized 66D state and hidden width 128. No planner weakening for results.
- Final checkpoint only: 24 fresh, predeclared matched reset keys per objective and seed,
  specialist checkpoint 2. No held-out evaluation is used to select a checkpoint,
  alter budgets, tune settings, or drop seeds. These resets differ from the earlier
  24-episode evaluation, but the held-out specialist family has been inspected
  before. This is not a never-seen opponent-family or strategy generalization claim.

## Artifacts and claims

Save each seed/arm's configuration, source and artifact hashes, frozen opponent
identity, actual interaction counts, latest complete round checkpoint and all
per-round trajectory recordings. Bind the nine frozen red checkpoints to the
dataset provenance, including checkpoint 2 used only for final evaluation.
Resuming checks hashes/configuration and restores optimizer and replay RNG state.
Before aggregation, verify common source/data/budgets/split, within-seed shared
encoder/statistics, completed quotas, and hashes of models and recorded outcomes.
Reject incomplete or mixed smoke/scientific runs; do not label two completed seeds
as the predeclared three-seed benchmark. Record actual compute and elapsed time
separately from real-transition budgets, including resumed execution segments.
Final table shows return, capture and resources, with sample SD over training
seed means. Paired comparisons resample seed and matched reset clusters together
across objectives. Three seeds provide limited, descriptive uncertainty, not a
definitive SOTA claim. Report unfavorable outcomes and all predeclared arms.
The three fit seeds share the same frozen specialist population and offline data;
their SD is not variation over independently trained opponent populations. Learned
history inference is online, but encoder/opponent weights remain frozen. Neither
positive return nor a probe/ARI score establishes faithful three-way behavior control.

```bash
uv run --locked --extra train --extra dev python scripts/run_control_benchmark.py
```

The command above is implementation-only and performs no experiments. Actual
execution additionally requires `--execute` and a fresh `--out` directory.

The `--smoke --out <separate-directory>` profile is only a plumbing test and may
never supply the scientific table. `--phase prepare`, `--phase train --seeds 0`,
and `--phase summarize` support separate workers using the same protocol.
