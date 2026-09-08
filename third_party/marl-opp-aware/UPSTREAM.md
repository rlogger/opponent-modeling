# Original environment provenance

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
