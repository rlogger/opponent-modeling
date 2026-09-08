#!/usr/bin/env python3
"""Visual diagnostics from saved 0s arrays; no encoder/world-model retraining."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mopa.manifest import file_sha256  # noqa: E402

TYPES = ("capture", "risk", "curious")
COLORS = ("#d45c46", "#367bb5", "#3f9369")
NAVY = "#243f58"
FIGURES = (
    ("latent_prototypes.png", "Where the three means sit", "Held-out episode latents use the saved training-fitted PCA. Stars are training-only means. Heatmap values are the eight raw latent coordinates, not strategy probabilities."),
    ("specialist_match.png", "Does each mean produce the right behavior?", "All prototypes and specialists see identical scenes. Compare rows within a specialist column. Outlined cells are column minima; lower error is better. The diagonal cells correspond to the intended strategy matches."),
    ("causal_prefix.png", "How much can be inferred early?", "Fresh linear diagnostic probes use saved causal prefixes, fitting only training episodes. The encoder is frozen. All plotted times use the same episodes, with no survivor filtering. The full-episode comparison has access to later behavior."),
    ("action_controls.png", "Does latent conditioning help prediction?", "Errors are averaged within episodes and then across types. Known-type means are an oracle control; zero context is the same decoder with z=0, not separately trained vanilla BC. Wrong means average the two fixed cyclic swaps."),
    ("action_vectors.png", "What actions do the means produce?", "One deterministic example from each recorded behavior: first saved scene per type. Each column compares a prototype with its corresponding specialist on the identical state. Solid arrows are the decoder; dashed arrows are the specialist. These are actions, not trajectories."),
    ("physics_error.png", "Can the learned physics predict motion?", "Recorded joint actions isolate the physics model. Both curves use identical held-out start states. This does not measure the quality of the learned opponent or closed-loop control."),
)


def save(fig, out, name):
    fig.savefig(out / name, dpi=175, facecolor="white")
    plt.close(fig)


def project(latent, data):
    return ((latent - data["latent_mean"]) / data["latent_scale"] - data["pca_mean"]) @ data["pca_components"].T


def latent_prototypes(data, out):
    query = data["heldout"]
    xy, means = project(data["episode"][query], data), project(data["prototypes"], data)
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True, gridspec_kw={"width_ratios": [1.1, 1]})
    for k, name in enumerate(TYPES):
        use = data["labels"][query] == k
        axs[0].scatter(*xy[use].T, color=COLORS[k], alpha=0.33, s=18, label=name)
        axs[0].scatter(*means[k], color=COLORS[k], marker="*", s=320, edgecolor="black", linewidth=0.8, zorder=4)
        axs[0].annotate(name + " mean", means[k], xytext=(10, 12 + 14 * (k == 2)), textcoords="offset points", fontsize=9)
    axs[0].set(title="Held-out trajectories + training means", xlabel="PC1", ylabel="PC2")
    axs[0].legend(frameon=False)
    values = data["prototypes"]
    bound = max(float(np.max(np.abs(values))), 1e-6)
    chart = axs[1].imshow(values, cmap="RdBu_r", vmin=-bound, vmax=bound, aspect="auto")
    axs[1].set(xticks=range(8), xticklabels=[f"z{i + 1}" for i in range(8)], yticks=range(3), yticklabels=TYPES,
               title="Three means, each an 8D vector", xlabel="Latent coordinate")
    for (i, j), value in np.ndenumerate(values):
        axs[1].text(j, i, f"{value:.2f}", ha="center", va="center", color="white" if abs(value) > 0.6 * bound else NAVY, fontsize=9)
    fig.colorbar(chart, ax=axs[1], shrink=0.7, label="Raw posterior-mean coordinate")
    fig.suptitle("0s: a shared latent space, not three separately trained encoders", fontsize=15)
    save(fig, out, "latent_prototypes.png")


def specialist_match(metrics, out):
    matrix = np.asarray(metrics["matched_state_specialists"]["prototype_vs_specialist_mse"])
    regret = matrix - matrix.min(axis=0, keepdims=True)
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    for ax, values, title in zip(axs, (matrix, regret), ("Action MSE on identical scenes", "Extra error above each column's best")):
        chart = ax.imshow(values, cmap="YlOrRd", vmin=0, vmax=max(float(values.max()), 1e-6))
        ax.set(xticks=range(3), xticklabels=TYPES, yticks=range(3), yticklabels=TYPES,
               xlabel="Fixed specialist target", ylabel="Selected prototype", title=title)
        for (i, j), value in np.ndenumerate(values):
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", color="white" if value > 0.65 * values.max() else NAVY)
        for col, row in enumerate(np.argmin(matrix, axis=0)):
            ax.add_patch(Rectangle((col - 0.47, row - 0.47), 0.94, 0.94, fill=False, edgecolor="#147e71", linewidth=3))
        fig.colorbar(chart, ax=ax, shrink=0.75, label="Squared 2D action error")
    winners = int(np.sum(np.argmin(matrix, axis=0) == np.arange(3)))
    fig.suptitle(f"Behavior matching: {winners} of 3 intended prototypes win their target columns", fontsize=15)
    save(fig, out, "specialist_match.png")


def causal_prefix(data, lengths, metrics, out):
    train, query, labels = data["train"], data["heldout"], data["labels"]
    max_prefix = min(10, int(np.min(lengths[np.r_[train, query]])))
    times = np.arange(max_prefix + 1)
    accuracy = []
    for t in times:
        probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=0))
        probe.fit(data["context"][train, t], labels[train])
        accuracy.append(float(accuracy_score(labels[query], probe.predict(data["context"][query, t]))))
    if 8 in times:
        np.testing.assert_allclose(accuracy[8], metrics["prefix8_probe"]["accuracy"], atol=1e-8)
    fig, ax = plt.subplots(figsize=(9.5, 4.8), constrained_layout=True)
    ax.plot(times, accuracy, "o-", color=NAVY, linewidth=2.5, label="Causal prefix probe")
    ax.axhline(1 / 3, linestyle=":", color="#999999", label="Chance (3 types)")
    ax.axhline(metrics["episode_probe"]["accuracy"], linestyle="--", color=COLORS[2], label="Full episode (posthoc)")
    for i in (0, min(8, max_prefix), max_prefix):
        ax.annotate(f"{accuracy[i]:.3f}", (times[i], accuracy[i]), xytext=(0, 9), textcoords="offset points", ha="center")
    ax.set(xlabel="Completed state-action pairs available at decision time", ylabel="Held-out objective accuracy",
           title=f"Early inference: the same {len(query)} held-out episodes at every time", ylim=(0, 1.08), xticks=times)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    save(fig, out, "causal_prefix.png")
    return {"completed_steps": times.tolist(), "heldout_accuracy": accuracy,
            "n_train": len(train), "n_heldout": len(query), "same_cohort": True,
            "encoder_retrained": False, "diagnostic_probe": "train-only StandardScaler + LogisticRegression at each prefix"}


def action_controls(metrics, out):
    controls = metrics["action_prediction"]
    rows = [("Causal history", controls["causal_history"], NAVY),
            ("Correct mean (known type)", controls["correct_prototype"], COLORS[2]),
            ("Zero context", controls["zero"], "#999999")]
    wrong = {"per_type": {k: np.mean([controls[f"wrong_prototype_{j}"]["per_type"][k] for j in (1, 2)]) for k in TYPES},
             "macro_episode_mse": np.mean([controls[f"wrong_prototype_{j}"]["macro_episode_mse"] for j in (1, 2)])}
    rows.append(("Wrong means (average)", wrong, "#c08a34"))
    fig, ax = plt.subplots(figsize=(11.5, 5), constrained_layout=True)
    x, width = np.arange(4), 0.19
    for i, (name, row, color) in enumerate(rows):
        values = [row["per_type"][k] for k in TYPES] + [row["macro_episode_mse"]]
        bars = ax.bar(x + (i - 1.5) * width, values, width, color=color, label=name)
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=3)
    ax.set(xticks=x, xticklabels=(*TYPES, "Equal-type mean"), ylabel="Mean squared 2D action error",
           title="Action prediction on recorded held-out states (lower is better)", ylim=(0, 0.92))
    ax.legend(ncol=2, frameon=False, fontsize=9)
    save(fig, out, "action_controls.png")


def action_vectors(data, scenes, out):
    origins = data["labels"][scenes["episode"]]
    selected = [int(np.flatnonzero(origins == k)[0]) for k in range(3)]
    fig, axs = plt.subplots(3, 3, figsize=(10.5, 10), constrained_layout=True, sharex=True, sharey=True)
    for row, scene in enumerate(selected):
        for col, name in enumerate(TYPES):
            ax = axs[row, col]
            expert, predicted = scenes["specialist_action"][scene, col], scenes["prototype_action"][scene, col]
            ax.axhline(0, color="#dedede", lw=0.6)
            ax.axvline(0, color="#dedede", lw=0.6)
            ax.annotate("", expert, (0, 0), arrowprops={"arrowstyle": "-|>", "color": "#333333", "linestyle": "--", "lw": 2})
            ax.annotate("", predicted, (0, 0), arrowprops={"arrowstyle": "-|>", "color": COLORS[col], "lw": 2.5})
            ax.set(xlim=(-1.15, 1.15), ylim=(-1.15, 1.15), aspect="equal", xticks=[-1, 0, 1], yticks=[-1, 0, 1])
            ax.text(0.04, 0.05, f"MSE {np.sum((predicted - expert) ** 2):.2f}", transform=ax.transAxes, fontsize=9)
            if row == 0:
                ax.set_title(name + " prototype / specialist")
            if col == 0:
                ax.set_ylabel(f"Scene from {TYPES[row]}\nepisode {scenes['episode'][scene]}, t={scenes['time'][scene]}\nAction y")
            if row == 2:
                ax.set_xlabel("Action x")
    fig.suptitle("Actual actions: solid = 0s decoder, dashed = specialist\nSame state across each row; examples are not aggregate evidence", fontsize=15)
    save(fig, out, "action_vectors.png")
    return selected


def physics_error(metrics, out):
    values = metrics["physics_recorded_joint_actions"]["heldout"]["per_horizon"]
    horizons = np.array(sorted(map(int, values)))
    fig, ax = plt.subplots(figsize=(9, 4.7), constrained_layout=True)
    for field, label, color in (("position_rmse_model", "Learned physics + recorded joint actions", NAVY),
                                ("position_rmse_persistence", "Persistence (keep starting position)", "#a37e5c")):
        y = [values[str(h)][field] for h in horizons]
        ax.plot(horizons, y, "o-", color=color, linewidth=2.5, label=label)
        for x, v in zip(horizons, y):
            ax.annotate(f"{v:.3f}", (x, v), xytext=(0, 9), textcoords="offset points", ha="center")
    ax.set(xticks=horizons, xlabel="Open-loop forecast horizon (steps)", ylabel="Position RMSE (arena coordinates)",
           title="Physics prediction: 4,096 held-out starts per horizon", ylim=(0, 0.83))
    ax.legend(frameon=False, fontsize=9)
    save(fig, out, "physics_error.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "experiments/continuous_0s")
    parser.add_argument("--dataset", type=Path, default=ROOT / "artifacts/continuous/dataset.npz")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    out = args.out or args.run / "visuals"
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.run / "manifest.json").read_text())
    names = ("metrics.json", "latents.npz", "matched_state_actions.npz")
    inputs = {name: file_sha256(args.run / name) for name in names}
    for name, digest in inputs.items():
        if digest != manifest["artifacts"][name]:
            raise ValueError(f"saved run artifact hash mismatch: {name}")
    dataset_hash = file_sha256(args.dataset)
    if dataset_hash != manifest["dataset"]["sha256"]:
        raise ValueError("dataset differs from the saved run")
    inputs["dataset.npz"] = dataset_hash
    metrics = json.loads((args.run / "metrics.json").read_text())
    with np.load(args.run / "latents.npz") as source:
        data = {k: source[k] for k in source.files}
    with np.load(args.run / "matched_state_actions.npz") as source:
        scenes = {k: source[k] for k in source.files}
    with np.load(args.dataset) as source:
        lengths = source["valid_length"]
        np.testing.assert_array_equal(source["objective_label"], data["labels"])
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 10, "axes.titlepad": 12, "axes.labelcolor": NAVY, "text.color": NAVY})
    latent_prototypes(data, out)
    specialist_match(metrics, out)
    prefix = causal_prefix(data, lengths, metrics, out)
    action_controls(metrics, out)
    selected = action_vectors(data, scenes, out)
    physics_error(metrics, out)
    (out / "prefix_metrics.json").write_text(json.dumps(prefix, indent=2) + "\n")
    gallery = f"# 0s visual gallery\n\nSaved encoder seed {manifest['seed']}; specialist checkpoint {manifest['heldout_checkpoint']} held out. No encoder or world-model retraining.\n\n"
    gallery += "## Watch the latent swap\n\nSame starting state and blue action sequence; only the 8D prototype changes. Red is the opponent, blue is the controlled agent. These are learned model predictions, not simulator ground truth. Fixed-horizon paths can continue past capture.\n\n![Animated learned rollouts](latent_swap_animation.gif)\n\n"
    for name, title, caption in FIGURES:
        gallery += f"## {title}\n\n{caption}\n\n![{title}]({name})\n\n"
    gallery += "## Does the effect repeat in other scenes?\n\nThree deterministic held-out scenes, three means each. Same initial state and blue actions within each row. No success-based selection; these are qualitative forecasts, not validated strategy control.\n\n![Additional learned rollouts](additional_scenes.png)\n\n"
    gallery += "## Reproduce\n\nFrom the repository root:\n\n```bash\nuv run --locked --extra train --extra plot python scripts/visualize_0s.py\nuv run --locked --extra train --extra plot python scripts/visualize_0s_rollouts.py\n```\n\nThe two manifests here bind the figures to the saved arrays/checkpoints. The original run report, models and manifest are unchanged.\n"
    (out / "README.md").write_text(gallery)
    outputs = [x[0] for x in FIGURES] + ["prefix_metrics.json", "README.md"]
    provenance = {"input_hashes": inputs, "source_sha256": file_sha256(Path(__file__)),
                  "selected_action_scene_rows": selected, "run": str(args.run.resolve()),
                  "output_hashes": {name: file_sha256(out / name) for name in outputs}}
    (out / "diagnostics_manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Created six diagnostic figures: {out / 'README.md'}", flush=True)


if __name__ == "__main__":
    main()
