#!/usr/bin/env python3
"""Tiny-set reconstruction checks only: no controller, rollout, or default changes."""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mopa.action_decoder import (  # noqa: E402
    ActionDecoderConfig,
    MLPHead,
    SeqGaussian,
    _beta,
    _free_bits_kl,
    _last_valid,
    _masked_mse,
    decode_action_decoder,
)
from mopa.manifest import (  # noqa: E402
    file_sha256,
    git_dirty,
    git_sha,
    package_versions,
)
from mopa.zero_s import zero_s_features  # noqa: E402

BEHAVIORS = ("capture", "risk", "curious")
MODES = ("vae", "stochastic_no_kl", "deterministic_no_kl", "bc")
LABELS = ("VAE", "Stochastic, no KL", "Deterministic, no KL", "MLP BC")


def select_windows(lengths, labels, checkpoints, behavior, count, width, seed, checkpoint=0):
    """One random complete, nonoverlapping-grid window per training episode."""
    if checkpoint not in (0, 1):
        raise ValueError("only training checkpoints 0 or 1 may be selected")
    eligible = np.flatnonzero(
        (labels == behavior) & (checkpoints == checkpoint) & (lengths >= width)
    )
    if count < 2 or width < 2 or len(eligible) < count:
        raise ValueError("need at least two eligible full windows from distinct episodes")
    rng = np.random.default_rng(seed)
    episode = rng.choice(eligible, size=count, replace=False)
    start = np.array([rng.integers(lengths[e] // width) * width for e in episode])
    return episode, start


def initialize(state, action, cfg, seed):
    """Shared AE/VAE initialization; BC inherits identical state-to-action weights."""
    _, ek, dk = jax.random.split(jax.random.PRNGKey(seed), 3)
    encoder = SeqGaussian(lat=cfg.lat, hid=cfg.hid)
    e = encoder.init(ek, jnp.concatenate([state[:1], action[:1]], -1),
                     jnp.ones(state.shape[:2], bool)[:1])
    d = MLPHead(out=cfg.action_dim, hid=cfg.hid).init(
        dk, jnp.zeros((1, cfg.lat + state.shape[-1]), jnp.float32)
    )
    bc = copy.deepcopy(d)
    first = bc["params"]["MLPTrunk_0"]["Dense_0"]
    first["kernel"] = first["kernel"][cfg.lat:]
    return {"e": e, "d": d}, bc


def train_arm(state, action, cfg, seed, mode, log_every=50):
    """Full-batch overfit with post-update, posterior-mean reconstruction metrics.

    BC gets only the eight state features. All other arms see target actions
    through the final window posterior: this is reconstruction, NOT forecasting.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode}")
    if cfg.action_type != "continuous" or cfg.steps < 1 or log_every < 1:
        raise ValueError("positive steps/log interval and continuous actions required")
    state, action = jnp.asarray(state), jnp.asarray(action)
    mask = jnp.ones(state.shape[:2], bool)
    sequence = jnp.concatenate([state, action], axis=-1)
    encoder = SeqGaussian(lat=cfg.lat, hid=cfg.hid)
    decoder = MLPHead(out=cfg.action_dim, hid=cfg.hid)
    params, bc_params = initialize(state, action, cfg, seed)
    if mode == "bc":
        params = bc_params
    optimizer = optax.adam(cfg.learning_rate)
    opt_state = optimizer.init(params)

    def posterior(p):
        mu, logvar = encoder.apply(p["e"], sequence, mask)
        return _last_valid(mu, mask), _last_valid(logvar, mask)

    def decode(p, latent):
        repeated = jnp.broadcast_to(latent[:, None], (*state.shape[:2], cfg.lat))
        return decode_action_decoder(p["d"], state, repeated, cfg)

    def loss_fn(p, key, beta):
        if mode == "bc":
            return _masked_mse(jnp.tanh(decoder.apply(p, state)), action, mask)
        mu, logvar = posterior(p)
        latent = mu
        if mode != "deterministic_no_kl":
            latent = mu + jnp.exp(0.5 * logvar) * jax.random.normal(key, mu.shape)
        loss = _masked_mse(decode(p, latent), action, mask)
        if mode == "vae":
            loss = loss + beta * _free_bits_kl(mu, logvar, cfg.free_bits)[0]
        return loss

    @jax.jit
    def update(p, opt, key, beta):
        loss, grad = jax.value_and_grad(loss_fn)(p, key, beta)
        delta, opt = optimizer.update(grad, opt, p)
        return optax.apply_updates(p, delta), opt, loss

    @jax.jit
    def predict(p):
        if mode == "bc":
            return jnp.tanh(decoder.apply(p, state))
        return decode(p, posterior(p)[0])

    history = []

    def record(step):
        prediction = predict(params)
        mse = float(_masked_mse(prediction, action, mask))
        if not np.isfinite(mse):
            raise FloatingPointError(f"nonfinite reconstruction at {mode}/{step}")
        history.append({"step": step, "mean_latent_mse": mse})

    record(0)
    start = time.perf_counter()
    rng = jax.random.PRNGKey(seed + 1000)
    for step in range(cfg.steps):
        rng, key = jax.random.split(rng)
        params, opt_state, loss = update(
            params, opt_state, key, _beta(step, cfg.steps, cfg.beta_max)
        )
        if (step + 1) % log_every == 0 or step + 1 == cfg.steps:
            record(step + 1)
            if not np.isfinite(float(loss)):
                raise FloatingPointError(f"nonfinite objective at {mode}/{step}")
    prediction = np.asarray(predict(params))
    if not np.all(np.abs(prediction) <= 1):
        raise AssertionError("unbounded action")
    if not all(np.isfinite(np.asarray(p)).all() for p in jax.tree_util.tree_leaves(params)):
        raise FloatingPointError("nonfinite parameters")
    result = {
        "initial_mse": history[0]["mean_latent_mse"],
        "final_mse": history[-1]["mean_latent_mse"],
        "max_vector_squared_error": float(np.sum((prediction - np.asarray(action)) ** 2, -1).max()),
        "parameter_count": sum(p.size for p in jax.tree_util.tree_leaves(params)),
        "training_loop_seconds_including_update_compile": time.perf_counter() - start,
        "history": history,
    }
    if mode != "bc":
        mu, logvar = posterior(params)
        result["zero_latent_mse"] = float(_masked_mse(decode(params, jnp.zeros_like(mu)), action, mask))
        result["shifted_window_latent_mse"] = float(_masked_mse(decode(params, jnp.roll(mu, 1, axis=0)), action, mask))
        result["posterior_mean_variance_per_dim"] = np.asarray(mu.var(0)).tolist()
        # The deterministic arm's log-variance head is untrained; do not interpret it.
        if mode != "deterministic_no_kl":
            keys = jax.random.split(jax.random.PRNGKey(seed + 2000), 32)
            losses = jax.jit(jax.vmap(lambda k: _masked_mse(
                decode(params, mu + jnp.exp(0.5 * logvar) * jax.random.normal(k, mu.shape)),
                action, mask)))(keys)
            result["sampled_mse_32_draws"] = float(losses.mean())
            result["sampled_mse_draw_std"] = float(losses.std())
            penalty, kl = _free_bits_kl(mu, logvar, cfg.free_bits)
            result.update(raw_kl=float(kl), floored_kl=float(penalty),
                          mean_posterior_std=float(jnp.exp(0.5 * logvar).mean()))
    return result, prediction


def plots(out, results, arrays):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ("#b33a3a", "#b47716", "#257347", "#2865a1")
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
    for ax, behavior in zip(axs, BEHAVIORS):
        for mode, label, color in zip(MODES, LABELS, colors):
            h = results[behavior][mode]["history"]
            ax.semilogy([v["step"] for v in h], [max(v["mean_latent_mse"], 1e-8) for v in h],
                        label=label, color=color)
        ax.set(title=behavior.capitalize(), xlabel="Full-batch updates", ylabel="Training vector MSE")
        ax.grid(alpha=0.15)
    axs[0].legend(fontsize=8)
    fig.suptitle("Tiny-set memorization · same recorded windows · not held-out performance")
    fig.savefig(out / "learning_curves.png", dpi=160)
    plt.close(fig)
    fig, axs = plt.subplots(3, 4, figsize=(11, 8), constrained_layout=True, sharex=True, sharey=True)
    for row, behavior in enumerate(BEHAVIORS):
        target = arrays[f"{behavior}_action"].reshape(-1, 2)
        for col, (mode, label, color) in enumerate(zip(MODES, LABELS, colors)):
            ax = axs[row, col]
            predicted = arrays[f"{behavior}_{mode}"].reshape(-1, 2)
            for dim, marker in enumerate(("o", "x")):
                ax.scatter(target[:, dim], predicted[:, dim], marker=marker, s=12, alpha=0.5,
                           color=color, label=("x action", "y action")[dim])
            ax.plot([-1, 1], [-1, 1], "k--", lw=0.8)
            ax.set(title=f"{behavior} · {label}\nMSE {results[behavior][mode]['final_mse']:.5f}",
                   xlim=(-1.08, 1.08), ylim=(-1.08, 1.08), aspect="equal")
            if row == 2:
                ax.set_xlabel("Recorded action component")
            if col == 0:
                ax.set_ylabel("Reconstructed component")
    axs[0, 0].legend(fontsize=7)
    fig.suptitle("Final training-set actions · dashed line = exact reconstruction")
    fig.savefig(out / "action_reconstruction.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "artifacts/continuous/dataset.npz")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--windows", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    # Never replace a previous diagnostic run or production checkpoint.
    args.out.mkdir(parents=True, exist_ok=False)
    cfg = ActionDecoderConfig(action_type="continuous", steps=args.steps, batch=args.windows)
    with np.load(args.dataset, allow_pickle=False) as raw:
        lengths, labels, checkpoints = (raw[k] for k in ("valid_length", "objective_label", "checkpoint_seed"))
        all_state, all_action = raw["state"], raw["red_action"]
    results, arrays, selection = {}, {}, {}
    for k, behavior in enumerate(BEHAVIORS):
        episode, start = select_windows(lengths, labels, checkpoints, k, args.windows, cfg.window, args.seed, args.checkpoint)
        time_idx = start[:, None] + np.arange(cfg.window)
        state = np.asarray(zero_s_features(all_state[episode[:, None], time_idx]))
        action = all_action[episode[:, None], time_idx]
        if not np.isfinite(state).all() or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
            raise ValueError("invalid selected state/action pairs")
        mean, std = state.mean((0, 1)), state.std((0, 1)) + 1e-6
        normalized = (state - mean) / std
        arrays.update({f"{behavior}_state": state, f"{behavior}_action": action,
                       f"{behavior}_mean": mean, f"{behavior}_std": std})
        selection[behavior] = {"episode": episode.tolist(), "start": start.tolist(),
                               "checkpoint": checkpoints[episode].tolist()}
        results[behavior] = {
            "constant_mean_action_mse": float(np.sum((action - action.mean((0, 1))) ** 2, -1).mean())
        }
        for mode in MODES:
            print(f"{behavior}/{mode}: {args.windows * cfg.window} actions, {cfg.steps} updates", flush=True)
            result, prediction = train_arm(normalized, action, cfg, args.seed, mode)
            results[behavior][mode] = result
            arrays[f"{behavior}_{mode}"] = prediction
            print(f"  final training MSE={result['final_mse']:.6f}; {result['training_loop_seconds_including_update_compile']:.1f}s", flush=True)
    output = {
        "kind": "tiny_training_set_reconstruction_diagnostic",
        "config": asdict(cfg), "seed": args.seed, "selection": selection,
        "dataset": {"path": str(args.dataset.resolve()), "sha256": file_sha256(args.dataset)},
        "git_sha": git_sha(ROOT), "git_dirty": git_dirty(ROOT), "dependencies": package_versions(),
        "source_sha256": {str(p.relative_to(ROOT)): file_sha256(p) for p in
                          (Path(__file__), ROOT / "src/mopa/action_decoder.py", ROOT / "src/mopa/zero_s.py")},
        "protocol": [
            f"One complete {cfg.window}-step window per selected checkpoint-{args.checkpoint} episode; checkpoint 2 excluded.",
            "Each behavior fitted separately; labels select data but are not model inputs.",
            "Identical full batches, train-only normalization, optimizer, step count, decoder width, and state features.",
            "AE/VAE share exact initial parameters; BC copies decoder state rows and all remaining weights, dropping latent inputs.",
            "BC is a plain two-layer ReLU/tanh MLP with the VAE head initialization, not the production BC default initializer.",
            "BC lacks the GRU and target-action history: equal examples/updates are not equal parameter count or compute.",
            "VAE beta ramps from 0 to 1 over half the run with 0.2 free bits per latent dimension.",
            "No-KL controls remove the penalty; deterministic control also removes latent sampling.",
            f"MSE sums over both action coordinates then averages all {args.windows * cfg.window} valid transitions; posterior-mean evaluation after updates.",
            "Full-batch tiny-set training differs from production window sampling and full-dataset normalization.",
            "No held-out evaluation, causal inference, controller training, environment rollouts, or checkpoint replacement.",
        ], "results": results,
    }
    (args.out / "results.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(args.out / "predictions.npz", **arrays)
    plots(args.out, results, arrays)
    print(f"Saved diagnostics to {args.out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
