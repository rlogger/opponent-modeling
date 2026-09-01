# Opponent behaviour cloning

This experiment compares one shared 2x128 MLP policy under four matched
conditioning inputs.

| Arm | Condition |
|---|---|
| Vanilla | three zeros |
| Latent | causal GRU-JEPA `z[t-1]`; zero at `t=0` |
| Shuffled | checkpoint-local episode derangement of the same latent |
| Oracle | strategy one-hot |

## Protocol

- 1,800 fixed-prey expert episodes, 137,698 valid actions, and exact 17-D
  predator observations.
- Three leave-one-checkpoint-out folds and paired encoder/BC seeds `0,1,2`.
- 5,000 encoder steps and 4,000 BC steps; all arms use a 3-D condition slot.
- Offline results average within episode, then strategy. Closed-loop results
  replay 10 expert steps, then use greedy BC on all 200 matched resets per
  strategy and fold.
- Values below are mean ± standard deviation across the three fold means,
  after averaging seeds within each fold.

## Offline results

| Arm | Macro NLL ↓ | Macro accuracy ↑ | Action-balanced accuracy ↑ |
|---|---:|---:|---:|
| Vanilla | 0.890 ± 0.107 | 0.723 ± 0.036 | 0.683 ± 0.046 |
| Latent | 0.902 ± 0.111 | 0.727 ± 0.040 | 0.685 ± 0.051 |
| Shuffled | 0.893 ± 0.111 | 0.723 ± 0.036 | 0.681 ± 0.048 |
| Oracle | 0.829 ± 0.167 | 0.796 ± 0.027 | 0.728 ± 0.034 |

## Closed-loop results

All gaps are absolute differences from the held-out expert; lower is better.

| Arm | Expert agreement ↑ | Predator ADE ↓ | Capture gap ↓ | Survival gap ↓ | Lava-step gap ↓ | Coverage gap ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Vanilla | 0.368 ± 0.026 | 1.209 ± 0.053 | 0.251 ± 0.009 | 15.284 ± 0.526 | 2.478 ± 0.328 | 12.565 ± 0.429 |
| Latent | 0.412 ± 0.052 | 1.084 ± 0.093 | 0.242 ± 0.009 | 14.586 ± 0.512 | 2.086 ± 0.121 | 11.363 ± 0.236 |
| Shuffled | 0.364 ± 0.029 | 1.218 ± 0.062 | 0.256 ± 0.007 | 15.248 ± 0.521 | 2.514 ± 0.367 | 12.750 ± 0.316 |
| Oracle | 0.686 ± 0.043 | 0.697 ± 0.079 | 0.024 ± 0.007 | 2.086 ± 0.256 | 1.270 ± 0.277 | 1.347 ± 0.359 |

## Result

The full gate did not pass. Latent conditioning improved closed-loop agreement
and ADE on every fold, while shuffled latents did not reproduce those gains.
However, latent offline NLL was worse than vanilla on two folds. The oracle
improved accuracy on every fold but missed the stricter NLL gate on fold 0.
These results validate the BC baseline and a closed-loop latent benefit, but
not a stable offline-likelihood or SOTA claim.

## Reproduce

```bash
uv run --locked python scripts/run_bc.py --require-clean
```

The published [result manifest](results.json) records all folds, seeds, hashes,
metrics, and gates. It was produced from commit
`9d991fd38cc393afd0af792eff41f3ed36c9194c`; the raw result SHA-256 is
`1bf41b9babc8e86fa230a3cb00748142018b7a28a266b2168333acc518f7190c`.
