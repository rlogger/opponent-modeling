# Original environment provenance

## Current source and metadata correction

On 2026-10-05, authenticated GitHub and Git inspection confirmed that the private
`rlogger/marl-opp-aware` repository still has one remote branch, `main`, at
`aecbab5daf6da402029e953be114a52e62a46c26`; no accessible forks or remote tags
were returned. That is the same upstream revision pinned below. A fresh clone
was inspected directly; an unauthenticated web 404 does not mean it is missing.

`objectives.py` remains byte-identical to the main-source restoration below.
`resources.py` now differs by a reviewed constructor correction to **declared
observation-space shapes** for supported agent/landmark counts. Its current
SHA256 is
`f335be9828212c02d6c6f5c8815fd2f2a7323a304514cef7e5e5e4751fe84ae0`.
The historical full-file hash is retained below rather than silently replaced.
Observation arrays, reward, reset and transition code were not changed by this
correction. `tests/test_resource_space_contract.py` checks the declared/actual
shapes, and `tests/test_original_environment.py` retains the historical
nonconstructor-method AST fingerprint and original numerical goldens.

Against the freshly fetched upstream source, all 12 objective-class methods
have equal ASTs after removing docstrings and the unused `n_preds` assignment.
Resource reset, step and observation methods also match. The remaining resource
differences are the declared-space correction and equivalent early-return
formatting in `_place_resources` (`if`/`elif`/`else` became returning `if`s).
These are source-comparison results; numerical runtime comparison is described
separately below and does not certify every historical dependency version.

A fresh 2026-10-05 runtime comparison imported the original files directly from
that clean upstream clone and compared them with the current implementation
under JAX/JAXlib 0.4.38, JaxMARL 0.1.0 and Flax 0.10.4. Three objectives, both
discrete and continuous actions, and reset seeds 0/7/22 produced **360 matched
transitions**. Another **24 controlled transitions** checked soft-boundary
movement, simultaneous capture/collection, collection alone and the 99-to-100
timeout. All reset/state/observation/reward/done/info leaves were exactly equal;
the controlled blue rewards also matched their analytic values. No source
weights, checkpoints or environment mechanics were changed for this check.

## Historical main-source restoration

On 2026-09-08, at the user's request, the two core modules were replaced by
byte-identical files from this repository's `main` commit
[`8c24db3c7deff8bd32610b04c3f1ebc08bf5e429`](https://github.com/rlogger/opponent-modeling/tree/8c24db3c7deff8bd32610b04c3f1ebc08bf5e429).
The previous version is preserved on `new-env` at `180571beed82326a0d11da8fad20c1ebb1886841`.

| Main file | SHA256 |
| --- | --- |
| `src/tag_objectives/objectives.py` | `71332590fc61490bfd6271ab72da01f41df952963c10d86b577035ac74796f84` |
| `src/tag_objectives/resources.py` | `72910059f63c01613b4ad3803236497723724e3c16525d9b21ca51b920b0d4d4` |

At that restoration, there were no deviations in these two core files.
The subsequent observation-space correction is documented above.
Continuous support remains in
the existing API and action adapter: `action_type="Continuous"` was already
accepted by main's JaxMARL parent. The API now inspects the declared `Box` action
space instead of requiring extra attributes on the environment. Main's default
and existing `mappo_objectives_*` configurations remain discrete; continuous
specialists use the explicitly separate `mappo_continuous_*` configurations.

Before replacement, main versus `new-env` was compared under locked JAX 0.4.38 /
JaxMARL 0.1.0: three reset seeds, 20 transitions, all three objectives, discrete
and continuous actions. Every reset/state/observation/reward/done/info field
matched exactly across **360 transitions**, maximum absolute error **0.0**.
All task defaults also matched. Replacing these sources does not itself change
the reward definitions or make learned specialist behavior more faithful.

At that restoration, `tests/test_original_environment.py` checked the exact
main-source hashes above while retaining the original numerical goldens below.
The current test preserves the exact objective-file hash and the historical
resource behavior fingerprint; only the declared resource spaces have changed.

Validation after main-source replacement: **72 passed** across
`test_original_environment.py`, `test_objectives_env.py`,
`test_continuous_data.py`, `test_zero_s_evaluation.py`, `test_mpc_replay.py`,
and `test_rendering.py`; Ruff and scoped `git diff --check` passed.

## Historical marl-opp-aware restoration audit

The remaining record describes the earlier restoration, before the main-source
replacement above; its local deviations and AST checks are historical.

Source: [rlogger/marl-opp-aware](https://github.com/rlogger/marl-opp-aware), pinned
at [`aecbab5daf6da402029e953be114a52e62a46c26`](https://github.com/rlogger/marl-opp-aware/tree/aecbab5daf6da402029e953be114a52e62a46c26).

| Original file | Local implementation | Original SHA256 |
| --- | --- | --- |
| `src/simple_tag_objectives.py` | `src/tag_objectives/objectives.py` | `efb9c9215b8269de34c0d8fa3809b3e55dc580f8d189d8fcd5c0a33d4363f9be` |
| `src/simple_tag_resources.py` | `src/tag_objectives/resources.py` | `60771df62b40073d82b58e7516bd89a36e9ec8fc7d70ff24a2e1e3f44d090760` |

The source bodies and comments were restored on 2026-09-08. There is one
environment implementation, at the local paths above; this directory contains
provenance, not another environment hierarchy.

## Local deviations

- Adapt the resource import to the `tag_objectives` package and sort imports.
- Omit the unused original `n_preds` variable and `State` import.
- Retain action-type validation, the `action_type` attribute, and the
  `continuous_actions` property needed by local callers. The original already
  accepts `action_type="Continuous"` through its JaxMARL parent.
- Retain `actions.py`'s two-dimensional `[-1, 1]^2` boundary adapter. Its five
  MPE channels decode to exactly the same force, before the original per-agent
  acceleration. This does not introduce different physics.

No changes to rewards, observation ordering, resource/lava placement, reset,
capture/timeout rules, speeds, acceleration, collision physics, or arena size.
The original arena has a **soft boundary penalty, not a physical wall**.

## Verification and claim boundary

Before restoration, original versus local code was tested under the same pinned
JAX 0.4.38 / JaxMARL 0.1.0 runtime: three reset seeds, 20 fixed-action transitions
per seed, all three objectives, both discrete and continuous actions. Across
360 transitions, reset/next states, observations, rewards, dones, and info were
exactly equal (maximum absolute difference **0.0**). Thus the renderer redesign
had not changed these game mechanics; restoration alone is not expected to
improve controller returns.

`tests/test_original_environment.py` is portable and needs no sibling checkout.
It fixes original-source method AST fingerprints (ignoring docstrings/imports
and Python-version metadata), original reset observations, a 12-step continuous
golden trajectory, continuous-force mapping, the soft boundary, and checkpoint
dimensions (17D predator, 35D prey, 66D Markov state, 8D `0s` features).
Goldens came from the pinned original, not from the restored implementation.
The objective constructor's API additions are excluded from its AST fingerprint;
its task defaults are checked numerically. Existing objective tests cover
capture, collection, lava, novelty, multi-predator, and timeout behavior.

Validation on 2026-09-08: **68 passed** across `test_original_environment.py`,
`test_objectives_env.py`, `test_continuous_data.py`, `test_mpc_replay.py`,
`test_rendering.py`, and `test_replay_cpl.py`, using `uv run --locked --no-sync`
with the train, plot, and dev extras. After strengthening default-parameter
assertions, the nine original-environment tests passed again. Ruff and
`git diff --check` also passed for the restored source and new tests.

Existing continuous checkpoints/data remain schema-compatible. Original discrete
policies are not continuous checkpoints. This comparison verifies source
equivalence under our locked runtime, not reconstruction of every historical
dependency version or a claim of positive TD-MPC control performance.
