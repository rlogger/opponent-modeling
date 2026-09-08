# Fresh Shashank comparison

Completed 2026-09-08. See [results](REPORT.md), [machine-readable comparison](comparison.json), and [validation](validation.json).

- Source: `hegde95/opponent-modeling`, `opponent-trajectory-encodings`, pinned `7bae281091a96fc83cadad3671bef070bd371675`.
- Current project: `134c455963b2b158d073963f980f2e2a7a8acc6a` plus the recorded uncommitted implementation. No checkpoint or old result is overwritten.
- The new environment renderer changes presentation, not physics/rewards/resets. Current dynamics were retained; reset fixes remain a separate decision.
- Specialist checkpoints are fixed. Discrete and continuous trajectories are recollected with the existing full-size protocols; this is not new specialist-policy training.

## Run matrix

| Arm | Data / fitting | Purpose |
|---|---|---|
| Shashank published | Original published metrics; original data absent from source tree | Historical reference, not a fresh reproduction |
| Source suite | All 13 encoders, fresh local discrete data, source full budgets, checkpoint 2 held out | Compare implementations on current data |
| Integrated source-compatible `0s` | Exactly the same discrete data, source features and model key 1000 | Check source-port parity |
| Integrated causal `0s` | Discrete data, pre-action features, three independent fit seeds | Recommended timing protocol |
| Continuous `0s` + Equation 3 | Fresh continuous data, three fit seeds; 1500 encoder and 2000 world-model updates each | Current continuous implementation, causal action prediction, prototype fidelity, physics error |

All latent models use training-only normalization and fitting. Report episode/window post-hoc probes separately from completed-history causal diagnostics. Source `--seed 0` maps to model key 1000 for `0s`. Continuous runs retain their existing explicit keys 0/1/2; their action MSE cannot be compared with discrete accuracy.

Source PCA plots are regenerated. Optional t-SNE/UMAP maps are not required for metric comparison; do not add dependencies solely for decorative projections. No architecture search, best-seed selection, additional equations, or new BC implementation is included. The existing continuous `zero` control is the same decoder at zero context, not a separately trained vanilla BC model.

Report individual seeds and mean ± population standard deviation, exact dataset/source hashes, all failed checks, and whether the intended prototype is the best match to each specialist. Continuous run seeds also vary GMM initialization and evaluation-state sampling; SD is not exclusively fit variability or a held-out-fold confidence interval. Distinct latent clusters or trajectories alone do not establish controllability or SOTA.
