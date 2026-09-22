# TD-MPC2 latent uncertainty and BC continuation

Status: planned; this branch adds the implementation and test plan only.
No new model, experiment, or test result is claimed.

## Branch baseline

Remote heads were fetched and checked on 2026-09-22:

| Branch | Tip | Relationship to `td-mpc2` |
| --- | --- | --- |
| `main` | `8c24db3` | Ancestor, 17 commits behind |
| `new-env` | `180571b` | Ancestor, 4 commits behind |
| `td-mpc2` | `b46f31b` | Base for this work |

The new branch is `codex/tdmpc2-latent-uncertainty`, based on
`b46f31b304a4a8cc9721d7ff3e28f23be7fb2e70`. No cross-branch merge is needed.
The local `archive/pre-original-env-20260908` branch is an older ancestor.

The original checkout has uncommitted benchmark and visualization changes.
This branch uses a separate worktree and does not include those changes.
In particular, its baseline does not contain `FrozenBCOpponent`, the new
controller/world-model inspectors, or the matched-control benchmark driver.

## Existing components to reuse

- [Continuous BC continuation](../scripts/run_bc_continuous.py):
  `closed_loop_continuation` replays expert actions for `t < ctx`, switches
  the **red predator** to BC at `t = ctx`, and retains the frozen prey.
  It validates prefix replay and returns aggregate metrics, but does not
  export its trajectory histories. This path uses 3D JEPA context.
- [Action-decoder VAE](../src/mopa/action_decoder.py): `SeqGaussian`
  predicts a mean and log-variance and samples during training.
  Offline encoding and [online 0s context](../src/mopa/zero_s.py) retain
  posterior means. The 8D `0s` context is distinct from the 3D BC context.
- [TD-MPC](../src/mopa/tdmpc.py): implicit, conditioned, and factored modes
  already exist. In factored mode, opponent action is predicted first,
  then dynamics receives both physical actions.
- `ZeroSOpponent.attach` installs a frozen action decoder with
  `optax.set_to_zero()` and `red_loss_scale=0`. It currently requires the
  identity 66D world-state encoder; learned SimNorm latents are not a
  drop-in replacement.

## 1. Model uncertainty in learned representations

Keep the two representations explicit: `x` is the TD-MPC world latent;
`c` is the opponent context inferred from completed history.

The proposed first experiment fits a separate VAE to detached, causal
opponent contexts `c_t`, using training episodes only. A VAE on TD-MPC
world latents `x_t` should be a separately named experiment, with a frozen
source encoder and a recorded checkpoint. Under the current identity
encoder, `x` is normalized physical state, not a learned representation.

1. Export representations with episode IDs, prefix lengths, valid masks,
   split membership, source checkpoint hashes, and normalization.
2. Fit normalization, PCA/PPCA, and a small VAE on training representations.
   Compare held-out reconstruction and predictive-density diagnostics.
3. Separately expose the existing trajectory VAE's posterior mean and
   log-variance at causal prefixes. Sample contexts, decode opponent
   actions, and propagate them through fixed-blue-action rollouts.
4. Record sampling keys and the temporal sampling rule. Do not average
   window variances or resample each step without specifying the
   resulting probabilistic model.
5. Measure held-out action/position interval coverage and width by horizon.
   Treat VAE reconstruction error and sample spread as diagnostics until
   their relationship to prediction error has been measured.

A VAE over context vectors models their distribution; it does not by itself
provide a calibrated posterior over opponent strategy or epistemic
uncertainty. Keep that density model separate from the existing
history-conditioned posterior.

PCA is a deterministic linear projection. Under the relevant linear-Gaussian
assumptions, a linear VAE corresponds to **probabilistic PCA**, which is the
appropriate probabilistic baseline. See
[Tipping and Bishop](https://www.microsoft.com/en-us/research/publication/probabilistic-principal-component-analysis/)
and [Lucas et al.](https://arxiv.org/abs/1911.02469).

## 2. Visualize expert-to-BC continuation

Start from the existing red-policy continuation path. The interpretation
here is that the trained red specialist supplies the prefix and red BC
takes over. Switching a blue TD-MPC controller to blue BC would require a
separate blue cloning policy and training target.

- Make the handoff step configurable, with a small default such as `k=2`.
  Expert actions control transitions `0 .. k-1`; BC first acts at `k`.
- Export expert and BC paths, executed actions, action source, reset/step
  keys, valid lengths, terminal causes, and exact checkpoint identities.
- Plot matched expert and BC trajectories with the shared prefix and
  handoff state visibly marked. Use actual simulator positions.
- Label replayed prefixes as replayed; a fresh live-specialist prefix can
  be added as an explicitly selected mode.
- Keep imagined world-model trajectories in separate panels, label
  sample bands, and stop or mask them according to continuation semantics.
- Report episodes that end before takeover rather than silently selecting
  only long episodes. The current evaluator filters for surviving prefixes.

## 3. Decoder and TD-MPC integration contracts

Three different meanings of "detached decoder" must remain explicit:

1. **Frozen opponent decoder:** TD-MPC updates cannot change decoder
   parameters. Freezing weights alone does not remove derivatives with
   respect to its inputs.
2. **Detached auxiliary red-head input:** existing `red_detach_latent`
   applies `stop_gradient` to world latents for the auxiliary red-action
   loss. The red head may still train; that loss must not train the world
   encoder or dynamics when detachment is enabled.
3. **Optional diagnostic observation decoder:** if a decoder is added to
   inspect learned world latents, train it as
   `D(stop_gradient(x)) -> observation` with its own optimizer.
   Its reconstruction loss must not alter TD-MPC encoder/dynamics/actor
   parameters or enter the control objective. This is separate from
   the existing opponent-action decoder.

For the identity-state baseline, positions already come from inverse
normalization; an observation decoder is unnecessary. A learned-encoder
integration needs an explicit bridge before attaching the existing 0s
decoder.

## Tests to add and coverage to retain

| Contract | Existing coverage / new assertion |
| --- | --- |
| Frozen decoder ownership | Keep `tests/test_zero_s.py`: exact decoder parameter equality after an update while dynamics changes, finite losses, bounded actions, checkpoint restore. Extend to repeated updates if optimizer ownership changes. |
| Detached input gradients | Add an isolated auxiliary-red-loss test with a learned encoder and multi-step rollout. With detachment, encoder/dynamics gradients are zero; without it, a deliberately non-degenerate fixture produces nonzero gradients. The trainable red head still receives gradients. Compare gradients directly to avoid weight-decay confounding. |
| Diagnostic decoder isolation | If introduced, reconstruction updates change only diagnostic-decoder parameters; source latents, TD-MPC parameters, and same-key controller actions remain unchanged. |
| Posterior validity and sampling | Check shapes, finite means/log-variances, positive variances, same-key reproducibility, different-key variation, and convergence to the deterministic mean path as variance tends to zero. |
| Causal uncertainty | Changing current/future actions or future observations cannot change the current posterior. Preserve zero-history behavior, episode resets, terminal freezing, and online/offline prefix equivalence. Existing mean-context tests are in `tests/test_zero_s_online.py`. |
| Split isolation | Changing held-out examples cannot change fit parameters, training normalization, PCA/PPCA axes, or the representation VAE. Ignore padded samples. |
| Exact takeover | Exercise the existing continuation function for `k=0`, `k=1`, and `k=2`; check prefix state equality, the first BC action at `k`, and no extra expert-controlled transition. Test invalid cutoffs and termination before/at takeover. |
| Trace correctness | Exported actions replay to saved positions with the recorded keys; switch markers refer to the handoff state; no padded states contribute to metrics or plots. |
| Factored input contract | Preserve clamped-action context invariance from `tests/test_tdmpc.py`; reject incompatible schemas/normalization and test missing required factored inputs instead of silently fabricating training data. |
| Round trip and end-to-end smoke | Save/reload uncertainty and decoder metadata, then reproduce a short same-key sampled rollout and handoff. Check masks, finite losses, and bounded actions. |

Implement small deterministic tests first, then a short simulator integration
smoke with saved checkpoints. Unit tests establish wiring and numerical
contracts; held-out calibration and control comparisons require separate
experiments.

## Suggested implementation order

1. Export and visualize the existing BC continuation; add switch/replay tests.
2. Add explicit gradient-boundary tests for the current decoder integration.
3. Add train-only representation export, PCA/PPCA and VAE fitting, and
   uncertainty-preserving inference with deterministic seed tests.
4. Propagate sampled contexts through diagnostic rollouts and measure
   held-out coverage before selecting any uncertainty-aware planner rule.
5. Add a detached observation decoder only for a declared diagnostic of
   learned world latents.
