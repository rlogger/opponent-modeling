"""Summarize the fixed matched evaluation; no checkpoint or episode selection."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent
TYPES = ("capture", "risk", "curious")
ARMS = (("Offline TD-MPC", "baseline", "tdmpc", "online"),
        ("Adapted TD-MPC", "final", "tdmpc", "online"),
        ("MAPPO prey", "baseline", "mappo_prey", "zero"),
        ("Random", "baseline", "random", "zero"))


def main():
    manifests = {name: json.loads((ROOT / name / "evaluation.json").read_text())
                 for name in ("baseline", "final")}
    adaptation = json.loads((ROOT / "adapted/manifest.json").read_text())["online_adaptation"]
    values, rows, traces = {}, [], {}
    for title, folder, controller, context in ARMS:
        values[title] = []
        for typ in TYPES:
            key = f"{typ}__{controller}__{context}"
            with np.load(ROOT / folder / f"{key}.npz") as data:
                d = dict(data)
            traces[title, typ] = d
            returns = (d["blue_reward"] * d["valid_mask"]).sum(1)
            values[title].append(returns)
            row = next(r for r in manifests[folder]["runs"]
                       if (r["opponent"], r["controller"], r["context_mode"]) == (typ, controller, context))
            np.testing.assert_allclose(returns.mean(), row["blue_return"]["mean"], atol=1e-4)
            valid = d["valid_mask"]
            outside = (np.abs(d["state"][:, 1:, 2:4]).max(-1) > 2) & valid
            rows.append({"arm": title, "opponent": typ, "n": len(returns),
                         "return_mean": float(returns.mean()), "return_std": float(returns.std()),
                         "capture_rate": row["captured"]["mean"],
                         "resources_mean": row["resources_collected"]["mean"],
                         "outside_transition_fraction": float(outside.sum() / valid.sum())})
        values[title] = np.asarray(values[title])
    for typ in TYPES:
        reference = traces[ARMS[0][0], typ]
        for title, *_ in ARMS[1:]:
            for key in ("environment_seed", "step_seed", "dataset_episode"):
                np.testing.assert_array_equal(reference[key], traces[title, typ][key])
    delta = values["Adapted TD-MPC"] - values["Offline TD-MPC"]
    # Reset-cluster bootstrap: each sampled reset keeps all three opponent types.
    boot = np.random.default_rng(0).integers(0, delta.shape[1], (10000, delta.shape[1]))
    ci = np.quantile(delta[:, boot].mean((0, 2)), [0.025, 0.975]).tolist()
    training = [{"round": r["round"], "updates_before_collection": 1000 * r["round"],
                 "return_mean": float(np.mean([g["blue_return_mean"] for g in r["groups"]]))}
                for r in adaptation["rounds"]]
    result = {"scope": "One controller seed; 24 matched held-out resets per opponent. Not a SOTA claim.",
              "metrics": rows, "training_collection": training,
              "overall_returns": {title: {"mean": float(v.mean()),
                                   "reset_bootstrap_95ci": np.quantile(v[:, boot].mean((0, 2)), [.025, .975]).tolist()}
                                  for title, v in values.items()},
              "paired_improvement": {"mean": float(delta.mean()), "reset_bootstrap_95ci": ci,
                                     "fraction_positive": float((delta > 0).mean())},
              "online_transitions": adaptation["online_transitions"],
              "updates_added": adaptation["updates_completed"],
              "outside_protocol": "Fraction of valid transitions whose next blue position has max absolute coordinate > 2."}
    (ROOT / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    axes[0].plot([r["updates_before_collection"] for r in training],
                 [r["return_mean"] for r in training], "o-", color="#217a76")
    axes[0].set(title="Training collection (not held-out evaluation)",
                xlabel="Additional updates before collection", ylabel="Mean undiscounted return")
    colors = ("#c44e52", "#217a76", "#4c72b0", "#999999")
    x = np.arange(3)
    for i, (title, *_rest) in enumerate(ARMS):
        axes[1].bar(x + (i - 1.5) * .2, values[title].mean(1), .19, label=title, color=colors[i])
    axes[1].set(xticks=x, xticklabels=TYPES, title="Held-out checkpoint 2 · 24 matched resets",
                ylabel="Mean return (symmetric-log scale)", yscale="symlog")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.axhline(0, color="#777777", linewidth=.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
    fig.savefig(ROOT / "comparison.png", dpi=180)
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
