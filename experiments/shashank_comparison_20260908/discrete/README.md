# Fresh discrete 0s comparison

Collected 1,800 episodes again from the existing specialists: 200 matched starts
per objective and checkpoint seed, fixed capture-family prey, 100-step horizon.
Checkpoint seeds 0/1 train the encoder; checkpoint seed 2 is held out. Policies
were not retrained and the renderer did not change simulation dynamics.

| Fresh run | Episode probe | GMM ARI | Window probe | Reconstruction accuracy |
|---|---:|---:|---:|---:|
| Pinned source, legacy features | 0.755 | 0.538 | 0.566 | 0.789 |
| Integrated port, legacy features | 0.755 | 0.538 | 0.566 | 0.789 |
| Integrated causal features, 3 seeds | 0.734 ± 0.046 | 0.525 ± 0.017 | 0.581 ± 0.031 | 0.763 ± 0.001 |

All fits used 1,500 updates, hidden width 64, latent width 8, 8-step windows,
batch 128, and Adam at 0.001. Causal runs use logical seeds 0/1/2 (model keys
1000/1001/1002); the source comparison uses its default model key 1000.
The ± values are population standard deviations across encoder seeds, not
confidence intervals. Length alone gives a 0.617 held-out probe.

Source and port match exactly: zero maximum difference in final episode/window
latents, logged loss/MSE/KL, and decoder accuracy. Source weights are captured
at function return without changing the source training math.

The reconstruction column matches Shashank's definition: accuracy over **all
valid training and held-out windows**, using the window's observed actions in
its latent. It is not online next-action prediction. The separately measured
held-out reconstruction accuracy for causal runs is 0.796 ± 0.002. Episode
probes and ARI use the full valid episode and therefore remain post-hoc;
causal prefix probes are saved separately in [metrics.json](metrics.json).

Reproduce from the repository root, with the upstream checkout at commit
`7bae281091a96fc83cadad3671bef070bd371675`:

```bash
uv run --locked --extra train --extra plot --extra dev python \
  experiments/shashank_comparison_20260908/discrete/run.py \
  --source /path/to/hegde95-opponent-modeling
```

`--collect-only` writes the source-compatible dataset; `--fit-only` refits all
five models on that dataset. [dataset_manifest.json](dataset_manifest.json)
records checkpoint/core-code hashes, reset checks, and gameplay metrics;
[fit_manifest.json](fit_manifest.json) records saved-weight/latent hashes and
runtime versions. The old cache could not be loaded, so equality to its raw
arrays is **not** claimed; these are actual new rollouts and fits.
