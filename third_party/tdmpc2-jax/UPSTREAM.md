# Upstream provenance: tdmpc2-jax

This directory records provenance for the TD-MPC2 core ported into this
repository. It contains **no executable runtime code**; the port lives in
`src/mopa/tdmpc.py` and `src/mopa/mppi.py`.

| Field | Value |
|---|---|
| Source | `ShaneFlandermeyer/tdmpc2-jax` |
| URL | <https://github.com/ShaneFlandermeyer/tdmpc2-jax> |
| Pinned commit | `5b05ff452424896d709848e1f249bd67e269b8a1` (2026-07-28, "Add MIT License to the project") |
| License | MIT, copied verbatim from that commit into [`LICENSE`](LICENSE) (git blob `376000a72d4afe2be7b7e897d0ff3603dfc959b2`, sha256 `f7f203e51863b173a2cd420a2db0e4fe56550f8ad596d18e0bd8a6258909688f`) |
| Integration | Reviewed source port, not a package dependency. The upstream package is never imported at runtime. |
| Gate | Gate 0 (`handoff.md`): behavior-preserving single-agent baseline |

Never refresh from upstream `main` implicitly. Any re-pin must update this file
and re-run `tests/test_tdmpc_upstream.py`.

## Upstream-to-local source mapping

| Upstream file (`tdmpc2_jax/…`) | Component | Local destination |
|---|---|---|
| `common/activations.py` | `mish`, `simnorm` | `src/mopa/tdmpc.py` (helpers section) |
| `common/util.py` | `symlog`, `symexp`, `two_hot`, `two_hot_inv`, `sg` | `src/mopa/tdmpc.py` |
| `common/loss.py` | `soft_crossentropy` | `src/mopa/tdmpc.py` |
| `common/scale.py` | `percentile_normalization` | `src/mopa/tdmpc.py` (`mean_std_normalization` is unused upstream and was not ported) |
| `networks/mlp.py` | `NormedLinear` | `src/mopa/tdmpc.py` |
| `networks/ensemble.py` | `Ensemble` | `src/mopa/tdmpc.py` |
| `world_model.py` | `WorldModel` (`create`, `encode`, `next`, `reward`, `sample_actions`, `Q`) | `src/mopa/tdmpc.py` |
| `tdmpc2.py` | `TDMPC2` fields, `create`, `act`, `update` (world-model loss, TD target, ensemble subsampling, target-Q EMA, policy loss, optimizer steps) | `src/mopa/tdmpc.py` |
| `tdmpc2.py` | `TDMPC2.plan`, `TDMPC2.estimate_value` | `src/mopa/mppi.py` (`plan`, `estimate_value` as module functions taking the agent PyTree; `TDMPC2.plan`/`estimate_value` delegate to them) |
| `train.py` lines 85–99, 134–153 only | Encoder module construction (`NormedLinear` stack, `optax.adam`), `WorldModel.create` / `TDMPC2.create` wiring, PRNG split order, `action_dim >= 20` MPPI adjustment | `src/mopa/tdmpc.py` (`build_encoder`, `create_agent`) |
| `config.yaml` | `encoder`, `world_model`, `tdmpc2` sections | `configs/tdmpc2.yaml` |

### Not ported (by decision)

`train.py` (training loop, env stepping, logging, Orbax checkpointing),
`data/` (replay buffers), `envs/` (DMControl/MuJoCo wrappers), `media/`,
`setup.py`, `README.md`, `common/scale.py::mean_std_normalization`, and the
`tabulate` debug-print flags in `WorldModel.create` / `train.py`. The Hydra,
env, buffer, logging, and tabulate keys of `config.yaml` were dropped with them.

## Local compatibility substitutions

Each substitution replaces a dependency absent from (or incompatible with) the
locked environment (JAX 0.4.38, Flax 0.10.4, Optax 0.2.5, distrax 0.1.5).
Dependency versions were not changed.

| # | Upstream | Local | Preserved semantics | Verification |
|---|---|---|---|---|
| 1 | `einops.rearrange(x, '...(L V) -> ... L V', V=simplex_dim)` / inverse in `simnorm` | `jnp.reshape(x, (*batch, D // V, V))` → softmax(axis=-1) → `jnp.reshape(x, original_shape)` | Row-major grouping (`index = l*V + v`), softmax axis, output shape | `test_simnorm_matches_reshape_softmax_reshape` (exact against an explicit NumPy reshape-softmax-reshape) |
| 2 | `tensorflow_probability.substrates.jax.distributions.MultivariateNormalDiag(loc, scale_diag)` in `sample_actions` | `distrax.MultivariateNormalDiag(loc, scale_diag)` | Diagonal Gaussian; reparameterized `loc + scale * eps` sampling; mean/`exp(log_std)` parameterization; tanh squashing; `[-1, 1]` bounds; tanh log-prob Jacobian `2*(log 2 - u - softplus(-2u))` summed over action dims (unchanged upstream expression) | `test_policy_log_prob_matches_explicit_diagonal_normal_with_tanh_correction` (closed-form diagonal-normal log-density + Jacobian, `atol=1e-5`) |
| 3 | `jaxtyping.PRNGKeyArray`, `jaxtyping.PyTree` annotations | `jax.Array` / `typing.Any` aliases (`PRNGKey`, `Params`, `PyTree`) | Annotations only; no runtime effect | n/a |
| 4 | Encoder built inline in `train.py` with `env.observation_space` | `build_encoder(obs_dim, …)` with `jnp.zeros(obs_dim)` init input | Identical module stack and optimizer; Dense/LayerNorm parameter shapes do not depend on the init batch shape | covered by the model tests |
| 5 | Upstream variable name `z` for the world latent | `x` (handoff notation, to avoid confusion with opponent context `c`) | Rename only | n/a |

Stochastic caveat: distrax and TensorFlow Probability may consume the PRNG key
differently inside `sample`. Fixed-seed sampled actions are therefore
reproducible **within** this port but are not claimed to be bitwise identical
to the upstream TFP samples. Deterministic paths (`encode`, `next`, `reward`,
`Q`, policy mean, log-probability of a given pre-squash action, planning with a
fixed noise realization) are unchanged expressions.

## Fixed-seed comparison against the pinned source (Gate 0 spike)

Performed once, outside the repository, on 2026-09-03 with the locked `.venv`
plus a throwaway `/tmp` directory holding only `einops` and `jaxtyping` so that
the upstream checkout at `5b05ff4` could be imported side by side with the port
(`smoke` profile, `obs_dim=4`, `action_dim=2`, identical PRNG keys). The
upstream package is not installed in the repository environment.

| Comparison (same inputs, same keys) | Result |
|---|---|
| `simnorm`, `mish`, `two_hot`, `two_hot_inv`, `soft_crossentropy` | bitwise identical (max abs diff 0.0) |
| Encoder, dynamics, reward, policy, value parameter trees after `create` | bitwise identical |
| `encode`, `next`, `reward` (+logits), `Q` (+logits, shape `(5, B)`) | bitwise identical |
| `sample_actions(deterministic=True)`: action, mean, log_std, log_prob | bitwise identical |
| Distrax vs TFP `log_prob` at the same pre-squash sample | max abs diff 4.8e-7 (float32) |
| Distrax vs TFP `sample` with the same key | differs (different PRNG consumption); mean/std over 4000 keys agree to 4.5e-4 / 4.1e-4 |
| `plan(deterministic=True)` with the same key | action max abs diff 1.8e-2, MPPI `std` identical (difference traces to the policy-prior samples above) |
| `estimate_value` with the same key | identical at initialization (zero-initialized reward/Q heads) |
| `update`: consistency, reward, value, total loss | identical to printed float32 precision |
| `update`: encoder, dynamics, reward, value, target-value parameters | bitwise identical |
| `update`: policy loss / policy parameters | 3.4e-5 / 5.7e-4 max abs diff (stochastic policy samples only) |

Conclusion: every deterministic expression is unchanged; the only divergence is
the sampler's internal key handling, which is the documented Distrax
substitution. Bitwise equality is **not** claimed for stochastic paths.

## Preserved upstream behavior worth knowing

These are upstream properties kept verbatim for parity. They are candidates
for the post-Gate-0 local correctness changes listed in `handoff.md` (Gate 4),
not for this port:

- `update` allocates `finished` / `latent_xs` with the configured `batch_size`
  rather than the sampled batch.
- The value loss sums `lam[:, None] * soft_crossentropy(...)` over `axis=1`
  (batch) whereas the reward loss sums over `axis=0` (time).
- Bootstrapping uses `(1 - terminated)`; truncation only ends the imagined
  rollout mask (`finished`).
- The encoder uses `optax.adam`; all other heads use `optax.adamw`.
- `predict_continues=False` by default, so the continuation head is `None` and
  its loss term is `0.0`.

## Behavioral confirmation

- No algorithmic changes: equations, loss coefficients, temporal weights
  (`rho`), optimizer chains (`zero_nans -> clip_by_global_norm -> adam(w)`),
  target-Q EMA (`tau`), ensemble subsampling, MPPI update rule, and PRNG
  splitting order are copied from the pinned commit.
- No opponent-aware changes: the transition remains Equation 1,
  `x_next = dynamics(x, agent_action)`. There are no opponent-policy
  parameters, predicted opponent actions, context tensors, context-conditioned
  or joint-action dynamics, or opponent-aware replay fields.
  `opponent_mode=implicit` and `context_dim=0` are validated at construction
  only (`validate_gate0_config`) and do not enter any computation.
- Forbidden imports (`hydra`, `tensorflow`, `tensorflow_probability`, `orbax`,
  `dm_control`, `einops`, `jaxtyping`) are absent from the ported files and
  asserted by `tests/test_tdmpc_upstream.py::test_ported_files_have_no_forbidden_imports`.
