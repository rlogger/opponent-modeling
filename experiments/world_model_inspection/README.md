# Mean-0s world-model inspector

[Interactive view](world-model-behaviors.html) · [Provenance](manifest.json)

These are learned, open-loop predictions under fixed blue commands, not real
environment rollouts or MPC-selected plans. `x` is the normalized 66D world
state; `z` is the separate eight-dimensional `0s` opponent representation.

| Equation | Transition | Changing the opponent mean |
|---|---|---|
| 1 · implicit | `x_next = d(x, u)` | Ignored; the three predicted paths coincide. |
| 2 · conditioned | `x_next = d(x, u, z)` | Enters dynamics directly; no explicit red-action predictor. |
| 3 · factored | `v = decoder(x, z)`; `x_next = d(x, u, v)` | Changes predicted red actions, which enter shared physics. |

`u` is the blue action; `v` is the red action. The frozen `0s` decoder receives
current imagined positions/velocities, not the same-step blue command.

## Data and checkpoints

The capture, risk and curious prototypes are three **train-only, episode-equal
8D vector means**, obtained from one shared frozen `0s` encoder. The cloud shows
training-episode latents; stars project those same means using the saved PCA.
Known objective colors are not discovered clusters or calibrated probabilities.

Equation 1 and Equation 2 were fitted offline for **2,000 updates each**, matching
the source training episodes, normalization, seed and update budget. Equation 2
was trained on the actual causal `0s` context cache: decision `t` uses only
completed state/red-action pairs before `t`, not a full-episode embedding or
class mean. Equation 1 has no context input. Equation 3 reuses the original
[`continuous_0s`](../continuous_0s/) base checkpoint and frozen decoder, **not the
adapted controller**. Models have their actual equation-specific input widths;
weights are never relabeled between equations.

Nine held-out starting scenes are selected deterministically: first, middle and
last episode for each source objective, without selection by predicted outcome.
Within each scene, all nine panels share the initial state and recorded blue
commands. Each prototype stays fixed for up to 30 imagined steps, capped by the
source episode length. These mean interventions are not online strategy inference.

## Reading the view

- The recorded source path is a reference, **not ground truth for latent swaps**.
- Equation 3's red-clamp toggle supplies the same recorded red commands across
  prototypes. Predicted states and continuation then coincide: `z` has no direct
  route into physics. Unclamped red actions are recomputed from each imagined state.
- A stop mark follows the first predicted continuation probability below 0.5.
  Later dashed paths are fixed-horizon extrapolation, not confirmed survival or
  capture. Resources/lava stay at their initial display positions; agent markers
  are not drawn to collision scale.

The export checks Equation 1's swap invariance and Equation 3's joint-action
clamp invariance. These structural diagnostics do not establish faithful
three-way behavior, improved control, benchmark superiority or SOTA.

## Reproduce

From the repository root, with the source artifacts and matching dataset present:

```bash
uv run --locked --extra train python scripts/inspect_world_models.py
```

The default restores compatible checkpoints and regenerates predictions; it
does **not train**. Add `--fit-missing` only to explicitly fit missing Equation 1
or Equation 2 checkpoints. It never retrains `0s`/MAPPO, runs an environment,
plans with MPC or fits probes. Incompatible saved bindings are rejected.

Outputs: `world-model-behaviors.html`, `world-model-data.json`, `rollouts.npz`,
and hash-bound `manifest.json`; fitted models have per-mode `binding.json` files.
Checkpoint binaries and logs are local-only.
