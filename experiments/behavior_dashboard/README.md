# Behavior dashboard

[Open the dashboard](index.html) · [Source and file hashes](manifest.json)

One browser file with saved interactive data, all 10 original PNGs and the GIF
embedded. Open `index.html` directly; no HTTP server is required. Interactive
charts still need network access to load D3 from its CDN.

Rebuild from the repository root using Python's standard library only:

```bash
python3 scripts/build_behavior_dashboard.py
```

The builder checks saved source hashes and writes a hash-bound manifest.
It performs no new analysis, model inference, training, probes or rollouts.
Original source files remain unchanged; historical plots retain their results.

## Five tabs

- **Recorded:** matched MAPPO/TD-MPC episodes, trajectory PCA and causal `0s` clouds.
- **World models:** the three-equation prototype inspector and red-action clamp.
- **Latents:** original representation overview, training curves, means and prefix plots.
- **Actions:** specialist matching, latent controls, action vectors and physics error.
- **Forecasts:** original Equation 3 paths, extra scenes, animation and final frame.

Recorded TD-MPC episodes use the **adapted** controller; the original `0s` figures
and three-equation inspector use the **base** Equation 3 checkpoint. They are not
one controller run. Zero-context controls are the same decoder with `z=0`, not
separately trained vanilla BC. Extra historical scenes are three risk-source
episodes; fixed-horizon forecasts are not simulator ground truth or control results.
