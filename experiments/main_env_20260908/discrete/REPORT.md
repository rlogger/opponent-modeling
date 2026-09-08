# Native-main discrete rerun

Fresh collection and all five `0s` fits reproduce the previous discrete results exactly. Restoring the source did not change these trajectories or metrics.

Environment: exact `main` commit `8c24db3c7deff8bd32610b04c3f1ebc08bf5e429`. Collection: 1,800 episodes (200 per objective/checkpoint, three checkpoint seeds), matched starts, fixed capture-prey family, 100-step maximum. This yields 137,698 valid transitions; 91,712 belong to the two training checkpoints. Checkpoint 2 is held out for representation evaluation. Collection plus five 1,500-update fits took 54.64 seconds on CPU, excluding visualization.

## Actual specialist behavior

| Predator | Capture rate | Predator lava steps | Visited cells | Episode steps |
|---|---:|---:|---:|---:|
| Capture | 95.17% | 8.777 | 14.34 | 39.04 |
| Risk | 4.50% | 0.342 | 8.92 | 97.82 |
| Curious | 13.33% | 10.693 | 39.73 | 92.64 |

These native discrete specialists show the intended aggregate distinctions: capture catches prey, risk avoids lava, and curious visits more cells. This does not establish that a latent-conditioned world model or continuous controller reproduces those distinctions.

![Native specialist behavior](behavior.png)

First matched episode, checkpoint 0, with no outcome selection: [capture](replays/capture.gif), [risk](replays/risk.gif), [curious](replays/curious.gif). Replaying the recorded discrete actions reproduces every displayed position exactly; provenance is in [replay_manifest.json](replay_manifest.json).

## Fresh `0s` representation results

| Fit | Held-out probe | GMM ARI | Unit probe | Decoder accuracy |
|---|---:|---:|---:|---:|
| Pinned Shashank source, fresh main data | 0.755 | 0.538 | 0.566 | 0.789 |
| Integrated source-compatible port | 0.755 | 0.538 | 0.566 | 0.789 |
| Integrated causal features, three seeds | 0.734 ± 0.046 | 0.525 ± 0.017 | 0.581 ± 0.031 | 0.763 ± 0.001 |

All values were rerun, not copied. Source/port numerical parity errors are zero. Every listed metric has zero difference from the previous discrete experiment. The final row reports population standard deviation across encoder seeds, not a confidence interval. Episode probes/ARI use post-hoc full valid trajectories; decoder accuracy follows the source convention of all valid windows, including training windows. Causal held-out-only decoder accuracy is 0.796 ± 0.002. Prefix-time probes are recorded separately in [metrics.json](metrics.json).

## Reproduction and provenance

`run.py` imports the existing comparison driver's `collect()` and `fit()` without modifying it; it verifies exact main source bytes and refuses nonempty output. Run it from the repository root in the locked environment. `visualize.py` uses the existing shared renderer and exact recorded actions.

- [Environment, dataset equality, counts, and metric deltas](main_provenance.json)
- [Checkpoint hashes and collection schema](dataset_manifest.json)
- [Fit packages, pinned source, and artifact hashes](fit_manifest.json)

All 18 dataset arrays and the compressed dataset SHA256 match the previous collection: `e391221052e269fad8687b9fead4c7f2eca10df4a33e3e76d51d6cc7273b5b3b`. Weights and NPZ files remain local ignored artifacts; JSON reports and visuals are retained separately.
