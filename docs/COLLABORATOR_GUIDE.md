# Collaborator guide

Updated October 5, 2026. This guide preserves the September `td-mpc2`
reproduction workflow and distinguishes it from the current consolidation.
`marl-private` governs project instructions; [Implementation status](STATUS.md)
records the current evidence boundary. The feasibility pilot and targeted
continuation are complete, with retained control/calibration failures. Reviewed
audit branches are published; the broad main campaign remains paused.

## Start here

- [README](../README.md): installation and commands, from specialist training to evaluation.
- [Results and development record](RESULTS.md): implementation decisions, budgets, comparisons, and limitations.
- [Environment](ENVIRONMENT.md): observations, objectives, rewards, and termination.
- [Saved `0s` controller](TD_MPC_0S.md): causal context, checkpoint loading, and real-environment execution.
- [TD-MPC2 source provenance](../third_party/tdmpc2-jax/UPSTREAM.md): source mapping, dependency substitutions, and parity boundaries.

`handoff.md` and the older gate sections in `docs/STATUS.md` retain historical
plans and experiments. They are not a current checklist of unimplemented work,
and their older continuous results are not the current `0s` results.

## Historical source snapshot

| Reference | Role |
|---|---|
| `td-mpc2` | Historical continuous-control branch; retain for source-bound reproduction |
| `fc0dc648bec5586fe32f79fb9cfc584b03d52342` | Published implementation used by the completed main-environment evaluation |
| `84a38de8b33daa2d6ca942690453408bc12a74fe` | Same implementation, with the expanded execution README; clean-source verification target |
| `main` at `8c24db3c7deff8bd32610b04c3f1ebc08bf5e429` | Source of the restored environment core; not the continuous TD-MPC implementation |
| `new-env` at `180571beed82326a0d11da8fad20c1ebb1886841` | Backup made before restoring the environment from main |

For historical reproduction, start from a separate clean checkout pinned to
`84a38de8b33daa2d6ca942690453408bc12a74fe`, then use this guide. For the new campaign,
use the exact clean executable commit and committed private protocol identified
in its manifest. The targeted result uses executable f0973e6 and private 4491552;
see the [full pins and commands](control-pipeline.md#targeted-control-diagnostic). Do not assume a local
working copy has the same implementation as the published result.

At the September documentation audit, the author's working tree also contained unpublished benchmark,
dashboard, inspector, and controller changes. They were not included in this
documentation commit or the clean-source tests. In particular, the newer matched
controller benchmark was paused, not completed.

## Choose the execution path

| Goal | Start with | What it establishes |
|---|---|---|
| Check installation and algorithm plumbing | README tests and CLI checks | Code execution, not control performance |
| Re-evaluate the completed fitted controller | Transfer the artifacts below, then run saved-model evaluation | Reproduction of one fitted model's held-out evaluation |
| Repeat training from scratch | README stages 1–5, with fresh output locations | An independent run; compare its manifests before comparing scores |
| Extend controller training | Complete replay transfer and resume-path checks below | Continuation of a fitted controller, not a fresh seed |

The normal path is continuous MAPPO → matched dataset → frozen `0s` model and
Equation 3 controller → controller-generated replay and updates → held-out
evaluation. Vanilla BC is a separate experiment, not a dependency of this path.

## Artifact transfer

A Git clone is not a trained experiment. Reports, configuration, manifests,
summary JSON, and selected visuals are committed. Model binaries, datasets,
specialist weights, and raw replay are git-ignored and must be transferred
separately or regenerated.

Preserve these repository-relative locations for the completed evaluation:

```text
experiments/original_env_20260908/adapted/
  agent.msgpack
  opponent.msgpack
  state_stats.npz
  config.json
  manifest.json
experiments/main_env_20260908/continuous_data/
  dataset.npz
  dataset.manifest.json
  report.json
logs/MPE_simple_tag_v3_continuous/
  mappo_continuous_capture_MPE_simple_tag_v3_continuous_pred_actor_seed0_vmap2.safetensors
  mappo_continuous_risk_MPE_simple_tag_v3_continuous_pred_actor_seed0_vmap2.safetensors
  mappo_continuous_curious_MPE_simple_tag_v3_continuous_pred_actor_seed0_vmap2.safetensors
  mappo_continuous_capture_MPE_simple_tag_v3_continuous_prey_actor_seed0_vmap2.safetensors
```

Also transfer the saved specialist training configurations and metrics if the
collaborator needs to audit training rather than just evaluate actors. Critics
are not loaded by this evaluation.

Additional inputs depend on the job:

| Job | Additional files |
|---|---|
| Resume the adapted controller | All 36 replay NPZs under `adapted/online_round_000/` through `online_round_005/`; capture/risk/curious predator actors for `vmap0` and `vmap1` |
| Regenerate the matched dataset | All nine predator actors and the capture-prey actors for `vmap0`, `vmap1`, and `vmap2` |
| Audit individual recorded episodes | `evaluation_per_episode.npz` and recorded transition NPZs from the relevant evaluation directory |
| Rerun historical restoration verifiers | Their original and regenerated datasets, parent checkpoints, recordings, source refs, and path dependencies; see the original protocols |

For a shared archive, include an inventory and SHA-256 digests, not just a folder
named after the experiment. Keep the original manifests unchanged. New specialist
training is not a substitute for the exact saved weights in a matched rerun.

### Verify the transferred bytes

These hashes were checked against the local files and completed evaluation:

| File | SHA-256 |
|---|---|
| `adapted/agent.msgpack` | `1559f6fbb55cc455cede3376134b08b716bfb3fc538790f0004f5226f0c82f16` |
| `adapted/opponent.msgpack` | `048233bf8c1f8165556db1e089b6e85e9110fbf2d67d8638031a9dcd206e0271` |
| `adapted/state_stats.npz` | `49246c7fd646630d30308284dd49fa3d3df1fea7693a6dae26ce4eb6fbca46b4` |
| `continuous_data/dataset.npz` | `928181027a9e5e86b106190b1487cda90c09e2cc5df08b569f7b93ba8350b5f5` |
| Capture-prey `seed0_vmap2` actor | `e20c31b26e8792b6cbb27a6b90aa39be4a11dc1153f1d14ceae3f033f6b075c7` |

From the repository root:

```bash
shasum -a 256 \
  experiments/original_env_20260908/adapted/agent.msgpack \
  experiments/original_env_20260908/adapted/opponent.msgpack \
  experiments/original_env_20260908/adapted/state_stats.npz \
  experiments/main_env_20260908/continuous_data/dataset.npz \
  logs/MPE_simple_tag_v3_continuous/mappo_continuous_capture_MPE_simple_tag_v3_continuous_prey_actor_seed0_vmap2.safetensors
```

Compare every output with the table. The evaluator checks dataset and held-out
predator hashes, but does not enforce the MAPPO prey-control hash. Check that
separately before claiming the same comparison. All nine predator hashes and all
36 online replay hashes were also checked against the saved manifest in this audit.

## Re-evaluate the completed controller

After installation and artifact transfer, run from the repository root:

```bash
uv run --locked --all-extras python scripts/run_tdmpc.py evaluate \
  experiments/original_env_20260908/adapted \
  --dataset experiments/main_env_20260908/continuous_data/dataset.npz \
  --logdir logs/MPE_simple_tag_v3_continuous \
  --out artifacts/reproductions/main_env_20260908/controller \
  --n-eps 24 --context-modes online --controls --record
```

Use a new output directory for each attempt. Do not write over the completed
`experiments/main_env_20260908/controller` result. Explicit `--dataset` and
`--logdir` arguments make evaluation usable in a relocated checkout despite
historical path strings in the manifest.

This evaluates one fitted controller on 24 matched resets per opponent type,
with specialist checkpoint 2 held out, plus MAPPO and random controls. It does
not train a new model. Compare `evaluation.json` and `evaluation_per_episode.npz`
with the [recorded protocol and results](RESULTS.md#closed-loop-controller).
Check the recorded controller, dataset, specialist, configuration, and source
hashes before attributing a score difference to the algorithm. Different
hardware can also introduce numerical differences; identical bytes are not a
promise of bitwise-identical trajectories on every backend.

## Continuing an adapted run

Continuation restores the model, optimizer, sampling state, and prior replay;
without a seed override it also restores the saved RNG state. The completed run
contains 86,899 offline and 24,703 online transitions: 111,602 in total.

There is a portability limitation: `adapt-0s` reads the absolute paths in
`manifest.json` → `online_adaptation.data[].path`. The saved 36 replay paths
start with `/Users/rajdeepsingh/Documents/Playground/opponent-modeling/`.
Supplying a new dataset or log directory does not relocate those replay paths.

Before resuming on another machine, either preserve the original layout or
rebase replay paths in a separate copied run and manifest. Preserve the original
manifest, verify every replay digest, and document the relocation. There is no
`--replay-root` or `--relocate` option. A small explicit relocation feature is
reasonable follow-up work; it is not implemented by this documentation update.

Once paths and hashes are valid, pass the adapted directory to the README's
`adapt-0s` command and choose a fresh `--out`. Verify the six training-opponent
weights against the saved online groups: their hashes are recorded, but replaced
training weights are not automatically rejected by the adaptation command.

## Recording a new experiment

1. Record the branch, commit, dirty state, dependency versions, hardware, and
   complete command. Use `uv.lock`; do not independently upgrade JAX or Distrax.
2. Use fresh output directories. MAPPO's fixed filenames can overwrite older
   weights; follow the README's root-level `SAVE_PATH` and `--logdir` conventions.
3. For the historical September `0s` protocol, keep specialist checkpoint 2 held
   out and fit encoders, normalization, and prototypes on checkpoints 0/1 only.
   New runs follow their frozen protocol; targeted v2 trains on family 0 and
   evaluates family 1.
4. Record valid transitions separately from padded timesteps, gradient updates,
   and episodes. Record collection and evaluation seed keys.
5. Preserve manifests, resolved configs, input hashes, logs, checkpoints, and
   per-episode outcomes. State whether wall time includes compilation or pauses.
6. Compare controls on matched resets and explicitly describe unequal training
   budgets or information. A zero-context ablation is not trained vanilla BC.
7. Write failures and incomplete runs into the report. Do not select a favorable
   seed and describe it as a multi-seed result.

Historical `PROTOCOL.md` command blocks document what happened; some target
completed output directories. The historical `verify.py` scripts write
`verification.json` and need local data and paths. Neither is a safe clone-only
smoke test. Use the README's unit tests for installation checks.

## Historical clean-source verification

The audit used a clean `git archive` export of `84a38de`, with imports explicitly
resolved to the exported source rather than the author's dirty working tree.
It used Python 3.11, JAX 0.4.38, Flax 0.10.4, and Distrax 0.1.5 on CPU.
The existing pinned environment was reused with `--no-sync`; this was not a
fresh dependency installation.

| Check | Result |
|---|---|
| README help/config commands, including all three MAPPO presets | 13 passed |
| Focused unit and integration tests | 83 passed; no failures or skips |
| Ruff on the clean source | Passed |
| Python compilation of `src`, `scripts`, and `tests` | Passed |

The focused test commands were:

```bash
uv run --locked --no-sync --all-extras pytest -q \
  tests/test_continuous.py tests/test_action_decoder.py \
  tests/test_tdmpc_upstream.py tests/test_zero_s.py

uv run --locked --no-sync --all-extras pytest -q \
  tests/test_original_environment.py tests/test_continuous_data.py \
  tests/test_tdmpc_implicit.py tests/test_zero_s_online.py \
  tests/test_zero_s_evaluation.py tests/test_tdmpc.py

uv run --locked --no-sync --all-extras ruff check .
uv run --locked --no-sync --all-extras python -m compileall -q src scripts tests
```

The two test groups took 31.34 and 65.51 seconds. Omit `--no-sync` when following
the README installation flow on a new machine; it is recorded here only to make
the audit's execution conditions explicit. The full test suite was not run in that September check. Before the October
pilot, the consolidated locked suite passed 787 tests without skips; subsequent
corrections require fresh checks against their own source commit.

This verification does not regenerate the experiment tables, transfer artifacts,
or establish GPU compatibility. Synthetic optimizer steps in unit tests are not
new experiment training.

## Current work and reproduction boundary

The new campaign binds every run to both specification and executable commits,
a frozen protocol, dataset/checkpoint hashes, split membership and raw traces.
Use fresh output directories and preserve the historical paused benchmark.
The pilot is a feasibility run; it cannot replace prescribed main budgets,
independent fitting seeds or raw-to-reported result verification.

World representation on/off means identity global-state input versus a learned
world encoder. It is distinct from changing the opponent encoder or setting an
opponent context to zero. Likewise, replay updates of controller weights and
replay updates of opponent-model weights are separate operations.

The [Spatial Blotto environment](spatial-blotto.md) is included in the
consolidated source. Its scripted and mathematical examples remain separate
from predator–prey experiments and from unclaimed learned Blotto control.

See [Implementation status](STATUS.md) for completed checks, pending experiments
and the limits of the historical reports. Private presentations and discussion
remain in `marl-private`; public artifacts identify their governing protocol and
specification commit without copying that material.
