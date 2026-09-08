#!/usr/bin/env python3
"""Train continuous 0s and one factored world model; audit held-out latent swaps."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import flax.serialization
import jax
import matplotlib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, adjusted_rand_score, balanced_accuracy_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mopa.action_decoder import (  # noqa: E402
    ActionDecoderConfig,
    decode_action_decoder,
    fit_action_decoder_vae,
    gather_windows,
)
from mopa.continuous_data import (  # noqa: E402
    DEFAULT_CONTINUOUS_LOGDIR,
    OBJECTIVE_TYPES,
    continuous_checkpoint_path,
    deterministic_specialist_action,
    load_continuous_actor_params,
    load_continuous_dataset,
)
from mopa.manifest import (  # noqa: E402
    file_sha256,
    git_dirty,
    git_sha,
    package_versions,
)
from mopa.tdmpc import create_agent, load_config  # noqa: E402
from mopa.tdmpc_data import (  # noqa: E402
    SequenceReplay,
    multistep_model_error,
    state_statistics,
)
from mopa.zero_s import ZeroSOpponent, rollout_zero_s, zero_s_features  # noqa: E402

SOURCE_PIN = "7bae281091a96fc83cadad3671bef070bd371675"
SOURCE = f"https://github.com/hegde95/opponent-modeling/blob/{SOURCE_PIN}/encoding_viz/REPORT.md"
COLORS = ("#dc5545", "#3177b6", "#40965e")


def write_json(path, value):
    def convert(item):
        return item.tolist() if isinstance(item, (np.ndarray, jax.Array)) else item.item()
    path.write_text(json.dumps(value, indent=2, default=convert, allow_nan=False) + "\n")


def probe(features, labels, train, test, seed):
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed))
    model.fit(features[train], labels[train])
    prediction = model.predict(features[test])
    return {
        "accuracy": float(accuracy_score(labels[test], prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(labels[test], prediction)),
        "n_train": int(len(train)), "n_heldout": int(len(test)),
    }


def balanced_action_error(errors, mask, labels):
    """Mean squared vector error, averaged within episodes then equally by type."""
    episode_error = (errors * mask).sum(axis=1) / mask.sum(axis=1)
    per_type = {name: float(episode_error[labels == k].mean()) for k, name in enumerate(OBJECTIVE_TYPES)}
    return {"macro_episode_mse": float(np.mean(list(per_type.values()))), "per_type": per_type}


def prototype_match_summary(matrix):
    """For each fixed specialist target, compare all prototype predictions."""
    best = np.argmin(matrix, axis=0)
    return {"best_prototype_per_specialist": [OBJECTIVE_TYPES[k] for k in best],
            "correct_best_prototypes": int(np.sum(best == np.arange(3)))}


def evaluate_latents(fit, opponent, ds, train, test, context, seed):
    labels, windows = ds.objective_label, fit.encoding.windows
    wt = np.flatnonzero(np.isin(windows.episode, train))
    we = np.flatnonzero(np.isin(windows.episode, test))
    z = fit.episode_latents
    scaler = StandardScaler().fit(z[train])
    scaled = scaler.transform(z)
    mixture = GaussianMixture(3, covariance_type="full", n_init=8, random_state=seed).fit(scaled[train])
    prefix_train = train[ds.valid_length[train] >= 8]
    prefix_test = test[ds.valid_length[test] >= 8]
    metrics = {
        "episode_probe": probe(z, labels, train, test, seed),
        "window_probe": probe(fit.window_latents, labels[windows.episode], wt, we, seed),
        "gmm_ari": float(adjusted_rand_score(labels[test], mixture.predict(scaled[test]))),
        "length_only_probe": probe(ds.valid_length[:, None], labels, train, test, seed),
        "prefix8_probe": probe(context[:, 8], labels, prefix_train, prefix_test, seed),
    }
    # Reconstruction has seen the target action through its window posterior.
    features = np.asarray(zero_s_features(ds.state[:, :-1]))
    standardized = (features - fit.encoder.state_mean) / fit.encoder.state_std
    states_w = gather_windows(standardized, windows)[we]
    targets_w = gather_windows(ds.red_action, windows)[we]
    latent_w = np.broadcast_to(fit.window_latents[we, None], states_w.shape[:2] + (z.shape[-1],))
    decoded = np.asarray(decode_action_decoder(fit.decoder_params, states_w, latent_w, fit.encoder.config))
    error_w = np.sum((decoded - targets_w) ** 2, axis=-1)
    reconstruction = np.zeros((len(labels), ds.red_action.shape[1]), np.float32)
    for row, wi in enumerate(we):
        length, start, episode = windows.lengths[wi], windows.start[wi], windows.episode[wi]
        reconstruction[episode, start:start + length] = error_w[row, :length]
    mask, y, target = ds.valid_mask[test], labels[test], ds.red_action[test]
    metrics["posthoc_reconstruction"] = balanced_action_error(reconstruction[test], mask, y)
    states = ds.state[test, :-1]
    controls = {"causal_history": context[test, :-1], "zero": np.zeros_like(context[test, :-1])}
    for shift, name in ((0, "correct_prototype"), (1, "wrong_prototype_1"), (2, "wrong_prototype_2")):
        controls[name] = np.broadcast_to(opponent.prototypes[(y + shift) % 3, None], context[test, :-1].shape)
    metrics["action_prediction"] = {}
    for name, ctx in controls.items():
        prediction = np.asarray(opponent.actions(states, ctx))
        metrics["action_prediction"][name] = balanced_action_error(np.sum((prediction - target) ** 2, axis=-1), mask, y)
    return metrics, scaler, we


def latent_figure(out, fit, ds, train, test, scaler, we, metrics, physics_history, seed):
    pca = PCA(n_components=2, random_state=seed).fit(scaler.transform(fit.episode_latents[train]))
    xy = pca.transform(scaler.transform(fit.episode_latents))
    fig, axs = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for ax, idx, title in ((axs[0, 0], train, "Training episodes"), (axs[0, 1], test, "Held-out checkpoint episodes")):
        for k, name in enumerate(OBJECTIVE_TYPES):
            use = idx[ds.objective_label[idx] == k]
            ax.scatter(*xy[use].T, s=10, alpha=0.6, color=COLORS[k], label=name)
        ax.set(title=title, xlabel="PC1", ylabel="PC2")
        ax.legend(fontsize=8)
    for ckpt in np.unique(ds.checkpoint_seed):
        idx = ds.checkpoint_seed == ckpt
        axs[0, 2].scatter(*xy[idx].T, s=8, alpha=0.45, label=f"checkpoint {ckpt}")
    axs[0, 2].set(title="Checkpoint identity (same PCA)", xlabel="PC1", ylabel="PC2")
    axs[0, 2].legend(fontsize=8)
    for name in ("loss", "action_mse", "kl"):
        axs[1, 0].plot([r["step"] for r in fit.history], [r[name] for r in fit.history], label=f"0s {name}")
    if physics_history:
        axs[1, 0].plot([r["step"] for r in physics_history], [r["consistency_loss"] for r in physics_history], label="physics consistency")
    axs[1, 0].set(title="Training losses", xlabel="Update", ylabel="Loss")
    axs[1, 0].legend(fontsize=8)
    selected = we[::max(1, len(we) // 3000)]
    window_xy = pca.transform(scaler.transform(fit.window_latents[selected]))
    window_y = ds.objective_label[fit.encoding.windows.episode[selected]]
    for k, name in enumerate(OBJECTIVE_TYPES):
        axs[1, 1].scatter(*window_xy[window_y == k].T, s=7, alpha=0.3, color=COLORS[k], label=name)
    axs[1, 1].set(title="Held-out windows (episode-trained PCA)", xlabel="PC1", ylabel="PC2")
    names = ["Episode", "Window", "Prefix 8", "Length", "GMM ARI"]
    values = [metrics[k]["accuracy"] for k in ("episode_probe", "window_probe", "prefix8_probe", "length_only_probe")] + [metrics["gmm_ari"]]
    axs[1, 2].bar(names, values, color=["#526d91"] * 4 + ["#9a698d"])
    axs[1, 2].set(title="Held-out representation diagnostics", ylim=(min(0, min(values) - 0.05), 1.07))
    axs[1, 2].tick_params(axis="x", labelrotation=20)
    for i, value in enumerate(values):
        axs[1, 2].text(i, max(value, 0) + 0.02, f"{value:.3f}", ha="center", fontsize=9)
    fig.suptitle("Continuous 0s · shared frozen encoder · held-out specialist checkpoint")
    fig.savefig(out / "encoding_viz.png", dpi=170)
    plt.close(fig)
    return pca


def specialist_action_check(out, opponent, ds, test, logdir, heldout, seed):
    """Compare all three prototypes with all specialists on identical held-out scenes."""
    rng = np.random.default_rng(seed)
    pairs = [(e, t) for e in test for t in rng.choice(int(ds.valid_length[e]), min(4, int(ds.valid_length[e])), replace=False)]
    ep, time = np.asarray(pairs).T
    states, observations = ds.state[ep, time], ds.red_observation[ep, time]
    width = max(ds.red_observation.shape[-1], ds.blue_observation.shape[-1])
    targets, predictions, checkpoints = [], [], []
    for k, name in enumerate(OBJECTIVE_TYPES):
        path = continuous_checkpoint_path(logdir, name, "pred", heldout)
        params = load_continuous_actor_params(path)
        targets.append(np.asarray(deterministic_specialist_action(params, observations, width)))
        predictions.append(np.asarray(opponent.actions(states, np.broadcast_to(opponent.prototypes[k], (len(states), opponent.prototypes.shape[-1])))))
        checkpoints.append({"type": name, "path": str(path.resolve()), "sha256": file_sha256(path)})
    targets, predictions = np.stack(targets, axis=1), np.stack(predictions, axis=1)
    matrix = np.sum((predictions[:, :, None] - targets[:, None, :]) ** 2, axis=-1).mean(axis=0)
    diagonal, off_diagonal = float(np.trace(matrix) / 3), float(matrix[~np.eye(3, dtype=bool)].mean())
    np.savez(out / "matched_state_actions.npz", state=states, episode=ep, time=time,
             specialist_action=targets, prototype_action=predictions, mse_matrix=matrix)
    return {"n_states": len(states), "row_prototypes": OBJECTIVE_TYPES, "column_specialists": OBJECTIVE_TYPES,
            "prototype_vs_specialist_mse": matrix, "diagonal_mse": diagonal,
            "off_diagonal_mse": off_diagonal, "off_diagonal_minus_diagonal": off_diagonal - diagonal,
            **prototype_match_summary(matrix), "checkpoints": checkpoints,
            "sampling": "up to four seeded valid steps per held-out episode; same states for every type"}


def rollout_figure(out, agent, opponent, ds, test, mean, std):
    episode = int(test[ds.objective_label[test] == 1][0])
    horizon = min(30, int(ds.valid_length[episode]))
    initial = np.repeat(ds.state[episode, 0, None], 3, axis=0)
    blue = np.repeat(ds.blue_action[episode, :horizon, None], 3, axis=1)
    states, actions = rollout_zero_s(agent, initial, blue, opponent.prototypes, mean, std)
    states, actions = np.asarray(states), np.asarray(actions)
    np.savez(out / "latent_swap_rollouts.npz", state=states, red_action=actions, blue_action=blue,
             context=opponent.prototypes, initial_episode=episode)
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True, sharex=True, sharey=True)
    positions = states[..., :4].reshape(-1, 2)
    bound = max(1.1, float(np.max(np.abs(positions))) * 1.05)
    for k, ax in enumerate(axs):
        resources = initial[k, 8:40].reshape(16, 2)
        remaining = initial[k, 40:56] < 0.5
        ax.scatter(*resources[remaining].T, s=16, color="#e1ac2b", marker="*", label="resource")
        for center, radius in zip(initial[k, 56:62].reshape(3, 2), initial[k, 62:65]):
            ax.add_patch(plt.Circle(center, radius, color="#dc5545", alpha=0.12))
        ax.plot(states[:, k, 2], states[:, k, 3], color="#2669ae", label="blue prediction")
        ax.plot(states[:, k, 0], states[:, k, 1], color=COLORS[0], label="red prediction")
        ix = np.arange(0, horizon, 4)
        ax.quiver(states[ix, k, 0], states[ix, k, 1], actions[ix, k, 0], actions[ix, k, 1],
                  color=COLORS[0], angles="xy", scale_units="xy", scale=8, alpha=0.8)
        ax.scatter(states[0, k, [0, 2]], states[0, k, [1, 3]], color=[COLORS[0], "#2669ae"], marker="o", s=32)
        ax.set(title=f"{OBJECTIVE_TYPES[k]} prototype", xlabel="x", ylabel="y",
               xlim=(-bound, bound), ylim=(-bound, bound), aspect="equal")
    axs[0].legend(fontsize=7)
    fig.suptitle(f"Learned Eq. 3 rollouts · same initial state and blue actions · {horizon} steps\nNo simulator ground truth for latent swaps; arrows are predicted red actions")
    fig.savefig(out / "latent_swap_rollouts.png", dpi=170)
    plt.close(fig)
    return {"episode": episode, "horizon": horizon, "simulator_ground_truth": False,
            "max_absolute_position": float(np.max(np.abs(positions))),
            "mean_pairwise_red_position_distance": float(np.mean([np.linalg.norm(states[:, i, :2] - states[:, j, :2], axis=-1).mean() for i, j in ((0, 1), (0, 2), (1, 2))]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "artifacts/continuous/dataset.npz")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/continuous_0s")
    parser.add_argument("--encoder-steps", type=int, default=1500)
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--heldout", type=int, default=2)
    parser.add_argument("--logdir", type=Path, default=ROOT / DEFAULT_CONTINUOUS_LOGDIR)
    args = parser.parse_args()
    if args.encoder_steps < 1 or args.updates < 1:
        parser.error("encoder-steps and updates must be positive")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    ds = load_continuous_dataset(args.dataset)
    train = np.flatnonzero(ds.checkpoint_seed != args.heldout)
    test = np.flatnonzero(ds.checkpoint_seed == args.heldout)
    if not len(train) or not len(test) or any(not np.any(ds.objective_label[idx] == k) for idx in (train, test) for k in range(3)):
        raise ValueError("each split must contain all three strategy types")
    enc_cfg = ActionDecoderConfig(action_type="continuous", action_dim=2, steps=args.encoder_steps)
    print(f"Fit shared continuous 0s: {len(train)} train / {len(test)} held-out episodes", flush=True)
    fit = fit_action_decoder_vae(np.asarray(zero_s_features(ds.state[:, :-1])), ds.red_action,
                                ds.valid_length, train, jax.random.PRNGKey(args.seed), config=enc_cfg)
    opponent = ZeroSOpponent.from_fit(fit, ds.objective_label, train)
    opponent.save(out / "opponent.msgpack")
    context = opponent.context(ds.state, ds.red_action, ds.valid_length)
    metrics, scaler, we = evaluate_latents(fit, opponent, ds, train, test, context, args.seed)
    metrics["matched_state_specialists"] = specialist_action_check(out, opponent, ds, test, args.logdir, args.heldout, args.seed)
    print(f"Held-out episode probe {metrics['episode_probe']['accuracy']:.3f}; ARI {metrics['gmm_ari']:.3f}", flush=True)

    cfg = load_config(profile="gate4")
    cfg.update(opponent_mode="factored", context_dim=enc_cfg.lat)
    cfg["encoder"]["type"] = "identity"
    cfg["factored"]["red_loss_scale"] = 0.0
    mean, std = state_statistics(ds.state[train], ds.valid_mask[train], feature_map="markov")
    agent = create_agent(cfg, ds.state.shape[-1], key=jax.random.PRNGKey(args.seed), obs_mean=mean, obs_std=std)
    agent = opponent.attach(agent, mean, std)
    replays = {name: SequenceReplay.from_dataset(ds.as_dict(), idx, agent.horizon, context, feature_map="markov")
               for name, idx in (("train", train), ("heldout", test))}
    rng, key = np.random.default_rng(args.seed), jax.random.PRNGKey(10000 + args.seed)
    history = []
    for step in range(1, args.updates + 1):
        key, update_key = jax.random.split(key)
        agent, info = agent.update(**replays["train"].sample(rng, agent.batch_size), key=update_key)
        if step == 1 or step % 100 == 0 or step == args.updates:
            row = {"step": step, **{k: float(np.asarray(info[k])) for k in
                   ("total_loss", "consistency_loss", "reward_loss", "value_loss", "continue_loss", "red_loss", "policy_loss")}}
            if not np.isfinite(list(row.values())).all():
                raise RuntimeError(f"nonfinite world-model loss at update {step}")
            history.append(row)
            print(f"Physics {step}/{args.updates}: consistency={row['consistency_loss']:.4f}", flush=True)
    metrics["physics_recorded_joint_actions"] = {name: multistep_model_error(agent, replay, horizons=(1, 3, 10), obs_std=std, seed=args.seed)
                                                 for name, replay in replays.items()}
    metrics["latent_swap"] = rollout_figure(out, agent, opponent, ds, test, mean, std)
    pca = latent_figure(out, fit, ds, train, test, scaler, we, metrics, history, args.seed)
    (out / "agent.msgpack").write_bytes(flax.serialization.to_bytes(agent))
    np.savez(out / "state_stats.npz", mean=mean, std=std)
    np.savez(out / "latents.npz", episode=fit.episode_latents, window=fit.window_latents,
             prefix=fit.prefix_latents, context=context, prototypes=opponent.prototypes,
             labels=ds.objective_label, checkpoint_seed=ds.checkpoint_seed, train=train, heldout=test,
             window_episode=fit.encoding.windows.episode, window_start=fit.encoding.windows.start,
             pca_components=pca.components_, pca_mean=pca.mean_, latent_mean=scaler.mean_, latent_scale=scaler.scale_)
    write_json(out / "config.json", {"encoder": asdict(enc_cfg), "world_model": cfg})
    write_json(out / "training_history.json", {"encoder": fit.history, "world_model": history})
    write_json(out / "metrics.json", metrics)
    rows = [("Episode probe (posthoc)", metrics["episode_probe"]["accuracy"]),
            ("GMM ARI (posthoc)", metrics["gmm_ari"]), ("Window probe (posthoc)", metrics["window_probe"]["accuracy"]),
            ("8-completed-step prefix probe", metrics["prefix8_probe"]["accuracy"]),
            ("Length-only probe", metrics["length_only_probe"]["accuracy"])]
    table = "\n".join(f"| {name} | {value:.4f} |" for name, value in rows)
    matrix = metrics["matched_state_specialists"]["prototype_vs_specialist_mse"]
    matrix_table = "\n".join(f"| {name} | " + " | ".join(f"{v:.4f}" for v in matrix[k]) + " |" for k, name in enumerate(OBJECTIVE_TYPES))
    action_table = "\n".join(f"| {name} | {row['macro_episode_mse']:.4f} |" for name, row in metrics["action_prediction"].items())
    physics_table = "\n".join(f"| {h} | {row['position_rmse_model']:.4f} | {row['position_rmse_persistence']:.4f} |"
                              for h, row in metrics["physics_recorded_joint_actions"]["heldout"]["per_horizon"].items())
    matched = metrics["matched_state_specialists"]
    match_count = matched["correct_best_prototypes"]
    behavior_conclusion = ("All three named prototypes are the lowest-error choice for their matching specialist in this test. "
                          if match_count == 3 else
                          f"Only {match_count}/3 named prototypes are the lowest-error choice for their matching specialist. "
                          "Three-way behavior fidelity is not established by this run. ")
    (out / "REPORT.md").write_text(
        "# Continuous 0s and Equation 3\n\n"
        f"[Source 0s report]({SOURCE}); encoder seed {args.seed}, held-out specialist checkpoint {args.heldout}.\n\n"
        "One shared GRU64/window8/latent8 action-decoder VAE; train-only normalization and prototypes. "
        "Continuous adaptation uses recorded pre-action velocities (causal) and 2D tanh action reconstruction. "
        "Physics uses normalized 66D state and both actions; its frozen opponent decoder uses the selected 8D prototype.\n\n"
        "`v_t = 0s_decoder(state_t, z); x_(t+1) = dynamics(x_t, blue_action_t, v_t)`\n\n"
        f"{len(train)} training episodes; {len(test)} held out. {args.encoder_steps} VAE updates and {args.updates} world-model updates. "
        "The three means are three 8D vectors, not three scalar latent coordinates. "
        "All labels are excluded from VAE fitting. This is a continuous adaptation, not a matched reproduction of the discrete source scores.\n\n"
        f"| Held-out diagnostic | Value |\n|---|---:|\n{table}\n\n"
        "![Representation diagnostics](encoding_viz.png)\n\n![Latent swaps](latent_swap_rollouts.png)\n\n"
        "Episode/window probes and reconstruction are posthoc; target actions enter their posterior. "
        "Prefix probes and causal action prediction use only completed history. Strategy means use training episodes; "
        "correct-prototype evaluation uses known type labels, so it is an oracle control, not online inference. "
        "Zero context is an ablation of this same trained decoder, not separately trained vanilla BC. "
        "PCA, scaling, probes, and full-covariance 3-component GMM fit training data only. "
        "Action errors in metrics.json are squared 2D-vector errors averaged within episodes and equally across types. "
        "Continuous errors are not comparable to the source's discrete accuracy.\n\n"
        f"| Held-out action prediction | MSE (lower is better) |\n|---|---:|\n{action_table}\n\n"
        "These predictions are evaluated on recorded held-out states. They do not measure closed-loop performance.\n\n"
        "The 8D agent-only input omits lava geometry visible to specialists, limiting attainable risk-policy fidelity.\n\n"
        "Matched-state action MSE below evaluates each prototype against each held-out specialist "
        "on the same held-out scenes (up to four seeded valid steps per episode). Rows are prototypes, columns are specialist targets; "
        "compare rows within each fixed specialist column, since target difficulty differs across specialists. "
        "Queries on another specialist's visited states may be out of that policy's training distribution.\n\n"
        f"| Prototype | capture | risk | curious |\n|---|---:|---:|---:|\n{matrix_table}\n\n"
        f"Best prototypes for capture/risk/curious targets: {', '.join(matched['best_prototype_per_specialist'])}. "
        f"{behavior_conclusion}Different trajectories under swapped z alone are insufficient.\n\n"
        "Latent-swap plots use identical initial state and blue action sequence across all three prototypes. "
        "They are autoregressive learned rollouts with no simulator ground truth for the swaps. "
        "The fixed plot horizon does not reset or stop on predicted capture; post-capture states are extrapolations. "
        "Recorded-joint-action horizon 1/3/10 errors compare physics to persistence separately. "
        "Different paths alone do not establish correct strategy behavior. "
        "This run establishes component diagnostics, not control performance, online adaptation, or SOTA.\n\n"
        f"| Held-out physics horizon | Position RMSE | Persistence RMSE |\n|---|---:|---:|\n{physics_table}\n\n"
        "## Reproduce\n\nFrom the repository root, with continuous data and specialist checkpoints present:\n\n"
        "```bash\nuv run --locked --extra train --extra plot python scripts/run_0s_world_model.py \\\n"
        f"  --encoder-steps {args.encoder_steps} --updates {args.updates} --seed {args.seed} --heldout {args.heldout} \\\n"
        f"  --out {out}\n```\n\n"
        "Reload opponent.msgpack with ZeroSOpponent.load; construct the identity/factored agent using config.json "
        "and state_stats.npz, attach the opponent decoder, then restore agent.msgpack with flax.serialization.from_bytes. "
        "The attached apply function is static and must be reconstructed before restoring parameter bytes.\n"
    )
    source_files = [Path(__file__).resolve(), *sorted((ROOT / "src").rglob("*.py")),
                    ROOT / "configs/tdmpc2.yaml", ROOT / "uv.lock"]
    manifest = {"source_0s_commit": SOURCE_PIN, "tdmpc_upstream_commit": "5b05ff452424896d709848e1f249bd67e269b8a1",
                "git_sha": git_sha(ROOT), "git_dirty": git_dirty(ROOT), "dependencies": package_versions(),
                "seed": args.seed, "heldout_checkpoint": args.heldout,
                "train_episodes": train, "heldout_episodes": test,
                "specialist_checkpoints": metrics["matched_state_specialists"]["checkpoints"],
                "dataset": {"path": str(args.dataset.resolve()), "sha256": file_sha256(args.dataset)},
                "code": {str(p.relative_to(ROOT)): file_sha256(p) for p in source_files},
                "artifacts": {p.name: file_sha256(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != "manifest.json"}}
    write_json(out / "manifest.json", manifest)
    print(f"Report: {out / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
