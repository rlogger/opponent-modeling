# Saved `0s` → real-environment MPC

Frozen seed-0 Equation 3 checkpoint, held-out specialist checkpoint 2. Online
8D context uses only completed state-action history. Two matched episodes per
opponent, 100 real steps each; no retraining, parameter updates, or planner tuning.

| Opponent | Mean blue return | Captured | Mean resources |
|---|---:|---:|---:|
| capture | -755.756 | 0/2 | 0.5 |
| risk | -625.540 | 0/2 | 0.5 |
| curious | -716.321 | 0/2 | 0.5 |

These are six execution-smoke episodes, not evidence of strong control or a
comparison with baselines. The rendered prey leaves the soft arena boundary;
avoiding capture alone is insufficient when returns are strongly negative.

Verification: 68 focused regression tests passed. All recorded simulator states
and rewards replay exactly (maximum error 0). Streaming contexts match offline
causal contexts within 5.97e-7. Saved weights, normalization, configuration and
dataset hashes are unchanged. A repeat run reproduced identical trajectories,
contexts, actions and per-episode metrics. Planning retains horizon 3, 512 candidates,
24 prior samples, 64 elites, 6 MPPI iterations, 5 Q networks and discount 0.99.

First matched episode from each group, with no outcome-based selection:
[capture replay](arena_view/capture__tdmpc__online.gif), [risk replay](arena_view/risk__tdmpc__online.gif),
[curious replay](arena_view/curious__tdmpc__online.gif).

These corrected replays keep the camera at ±2.5 and mark off-screen agents with
their actual coordinates. They render the same recorded transitions; no policy,
physics, or results changed. [Rerender checks](arena_view/manifest.json). The
original zoomed-out GIFs remain beside this report as diagnostic overviews.

[Full metrics and provenance](evaluation.json). Transition NPZ files also store
the actual decision contexts, both actions, reset/step keys and terminal masks.
See [the command and protocol](../../../../../docs/TD_MPC_0S.md).
