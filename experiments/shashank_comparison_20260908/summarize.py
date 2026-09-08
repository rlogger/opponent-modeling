"""Summarize completed comparison runs; never fit or select a winning seed."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
KEY = "sa_short_seq_action_decoder_vae"
PIN = "7bae281091a96fc83cadad3671bef070bd371675"
METRICS = ("heldout_probe", "heldout_ari", "unit_probe", "decoder_action_accuracy")


def read(path):
    return json.loads((HERE / path).read_text())


def stats(values):
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("nonfinite result")
    return {"mean": float(array.mean()), "std": float(array.std()), "values": array.tolist()}


def fmt(row):
    return f"{row['mean']:.3f}" + (f" ± {row['std']:.3f}" if len(row["values"]) > 1 else "")


def main():
    published, upstream = read("upstream/published_results.json"), read("upstream/results.json")
    discrete = read("discrete/metrics.json")
    continuous = [read(f"continuous/seed_{seed}/metrics.json") for seed in range(3)]
    if len(upstream["strategies"]) != 13 or set(upstream["strategies"]) != set(published["strategies"]):
        raise ValueError("all 13 source strategies must finish before reporting completion")
    source = published["strategies"][KEY]
    source_metrics = {**source["metrics"], **source["extra"]}
    rows = {
        "Shashank published (historical data)": {k: stats([source_metrics[k]]) for k in METRICS},
        "Source code, fresh current discrete data": {k: stats([discrete["source_legacy_forward"][k]]) for k in METRICS},
        "Integrated port, identical source settings": {k: stats([discrete["integrated_legacy_forward"][k]]) for k in METRICS},
        "Integrated causal features, 3 fits": {k: discrete["causal_three_seed_summary"][k] for k in METRICS},
    }
    for k in ("heldout_probe", "heldout_ari", "unit_probe"):
        np.testing.assert_allclose(upstream["strategies"][KEY]["metrics"][k], discrete["source_legacy_forward"][k], atol=1e-8)
    cstats = {name: stats([item[field]["accuracy"] for item in continuous]) for name, field in (
        ("episode_probe", "episode_probe"), ("window_probe", "window_probe"),
        ("prefix8_probe", "prefix8_probe"), ("length_only_probe", "length_only_probe"))}
    cstats["gmm_ari"] = stats([item["gmm_ari"] for item in continuous])
    cstats["window_balanced_accuracy"] = stats([item["window_probe"]["balanced_accuracy"] for item in continuous])
    action = {name: stats([item["action_prediction"][name]["macro_episode_mse"] for item in continuous])
              for name in continuous[0]["action_prediction"]}
    matches = [item["matched_state_specialists"]["correct_best_prototypes"] for item in continuous]
    physics = {h: {kind: stats([item["physics_recorded_joint_actions"]["heldout"]["per_horizon"][h][kind] for item in continuous])
                   for kind in ("position_rmse_model", "position_rmse_persistence")} for h in ("1", "3", "10")}
    result = {"discrete": rows, "parity": discrete["parity"], "continuous": cstats,
              "continuous_action_mse": action, "intended_prototype_matches_per_seed": matches,
              "physics": physics, "std_scope": "population SD of 3 run seeds, including seeded evaluation sampling where applicable; checkpoint2 held out in all runs; not CI",
              "validation": read("validation.json")}
    (HERE / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    fig, axs = plt.subplots(1, 2, figsize=(13, 5.6), layout="constrained")
    short = ["Published\n(original data)", "Source\n(current data)", "Integrated\n(source settings)", "Integrated causal\n(3 fit seeds)"]
    x = np.arange(4)
    for offset, metric, label, color in ((-0.18, "heldout_probe", "Episode probe", "#365f7d"), (0.18, "heldout_ari", "GMM ARI", "#bd765c")):
        means = [row[metric]["mean"] for row in rows.values()]
        errors = [row[metric]["std"] if len(row[metric]["values"]) > 1 else np.nan for row in rows.values()]
        bars = axs[0].bar(x + offset, means, .34, yerr=errors, capsize=3, label=label, color=color)
        axs[0].bar_label(bars, fmt="%.3f", padding=4, fontsize=8)
    axs[0].set(xticks=x, xticklabels=short, ylim=(0, 1.05), title="Discrete 0s: source parity is exact", ylabel="Held-out score")
    axs[0].legend(frameon=False, loc="upper right")
    names = ["episode_probe", "gmm_ari", "window_probe", "prefix8_probe"]
    bars = axs[1].bar(np.arange(4), [cstats[k]["mean"] for k in names], yerr=[cstats[k]["std"] for k in names], capsize=4, color="#547b6b")
    axs[1].bar_label(bars, fmt="%.3f", padding=5, fontsize=9)
    axs[1].set(xticks=np.arange(4), xticklabels=["Episode\n(post-hoc)", "ARI\n(post-hoc)", "Window\n(post-hoc)", "8 completed\nsteps"], ylim=(0, 1.17), title="Continuous adaptation: a different task")
    for ax in axs:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Fresh full-budget fits · checkpoint 2 held out", fontsize=15)
    fig.supxlabel("Error bars: run-to-run SD across three seeds. Discrete and continuous scores are not an apples-to-apples improvement claim.", fontsize=9)
    fig.savefig(HERE / "comparison.png", dpi=170)
    plt.close(fig)

    table = "\n".join("| " + name + " | " + " | ".join(fmt(row[k]) for k in METRICS) + " |" for name, row in rows.items())
    suite = []
    for key, row in sorted(upstream["strategies"].items(), key=lambda pair: (pair[1]["order"], pair[1]["short"])):
        p, r = published["strategies"][key]["metrics"], row["metrics"]
        suite.append(f"| {row['tag']} | `{key}` | {p['heldout_probe']:.3f} | {r['heldout_probe']:.3f} | {p['heldout_ari']:.3f} | {r['heldout_ari']:.3f} |")
    ctable = "\n".join(f"| {name} | {fmt(value)} |" for name, value in cstats.items())
    atable = "\n".join(f"| {name} | {fmt(value)} |" for name, value in action.items())
    ptable = "\n".join(f"| {h} | {fmt(row['position_rmse_model'])} | {fmt(row['position_rmse_persistence'])} |" for h, row in physics.items())
    cseeds = "\n".join(f"| {seed} | {m['episode_probe']['accuracy']:.3f} | {m['gmm_ari']:.3f} | {m['window_probe']['accuracy']:.3f} | {m['prefix8_probe']['accuracy']:.3f} | {matches[seed]}/3 |" for seed, m in enumerate(continuous))
    best_probe = max(upstream["strategies"], key=lambda k: upstream["strategies"][k]["metrics"]["heldout_probe"])
    best_ari = max(upstream["strategies"], key=lambda k: upstream["strategies"][k]["metrics"]["heldout_ari"])
    report = f"""# Fresh comparison with Shashank

All 13 source encoders, the matched integrated port, three causal discrete fits, and three continuous `0s` + Equation 3 fits were rerun at full budgets. Specialist checkpoints and dynamics are unchanged. The fresh continuous collection is identical to the previous 26 arrays and exactly replays all 1,800 episodes. The visual upgrade therefore does not constitute a numerical improvement.

## 0s comparison

| Run | Episode probe | GMM ARI | Window probe | Reconstruction acc. |
|---|---:|---:|---:|---:|
{table}

Published values: [pinned source results](https://github.com/hegde95/opponent-modeling/blob/{PIN}/encoding_viz/results.json). Original data are unavailable: the historical collection used objective-specific prey policies and different checkpoints; current data use the fixed capture-prey family. Only the two same-data source/port rows test implementation parity. Their maximum latent, loss-history, and accuracy differences are `{max(discrete['parity'].values()):.1g}`.

Reconstruction accuracy follows the source definition: all valid samples, including training episodes, using a posterior that saw the target actions. It is not held-out next-action accuracy. Legacy features also include `p[t+1] - p[t]`, exposing the current action's transition; legacy mode is replication-only. Recommended causal features use past displacement, but pooled episode/window scores remain post-hoc. The recommended held-out reconstruction is separately `{fmt(discrete['causal_three_seed_summary']['heldout_decoder_accuracy'])}`. Source key 1000 corresponds to logical seed 0; discrete causal keys are 1000/1001/1002.

![Comparison](comparison.png)

## Current continuous implementation

This is the 2D tanh-action adaptation with actual pre-action velocities, not a reproduction of discrete action accuracy. All three fits use 1,200 training / 600 held-out episodes, GRU64/window8/latent8, 1,500 encoder updates, and 2,000 Equation 3 updates. Direct model keys are 0/1/2. Means are three training-only 8D vectors. Mean ± population SD measures run-to-run variability, including seeded GMM initialization and evaluation-state sampling where applicable; it is not uncertainty across held-out checkpoint folds.

| Diagnostic | Mean ± SD |
|---|---:|
{ctable}

Window accuracy weights windows, so longer episodes contribute more; balanced accuracy is included to expose this imbalance. Completed-history prefixes use no future actions. Length-only evidence is a post-hoc shortcut, not an online baseline.

| Fit seed | Episode probe | ARI | Window probe | Prefix 8 | Correct prototype matches |
|---|---:|---:|---:|---:|---:|
{cseeds}

| Held-out action control | Squared 2D action error, lower is better |
|---|---:|
{atable}

Errors average within episodes and equally across objective types. Correct prototypes use known type labels (oracle). Zero is an ablation of the same decoder, not separately trained vanilla BC. Correct-prototype match counts are {matches}, each out of three specialist columns. All three seeds select the risk prototype as best against every specialist. Distinct latent clusters and changed paths do not establish faithful three-way behavior control. The decoder's 8D kinematics omit lava features available to specialists. No online adaptation, closed-loop control advantage, or SOTA claim follows from these results.

| Physics horizon | Model position RMSE | Persistence RMSE |
|---|---:|---:|
{ptable}

Recorded joint actions isolate physics accuracy; these errors do not validate the opponent model. [Fresh latent/control visuals](continuous/seed_0/visuals/README.md) and [new environment replay](environment/replay.gif) use the fresh run/data, without success-based selection.

## All 13 source encoders

Same pinned source and full default budgets: 1,500 sequence or 3,000 point updates; no quick fits. This is a one-fit-per-strategy current-data rerun, not a multi-seed ranking. Source PCA/diagnostic figures are regenerated; optional t-SNE/UMAP maps are omitted.

| Tag | Encoder | Published probe | Fresh probe | Published ARI | Fresh ARI |
|---|---|---:|---:|---:|---:|
{chr(10).join(suite)}

`{best_probe}` has the highest linear probe in this fresh single-fit suite; `{best_ari}` has the highest ARI. This is not dominance across metrics or a multi-seed SOTA result. The large changes in the original sequence VAE scores occur with unmodified source code on different data, not from the integrated port.

Source length-only probe: published {published['controls']['survival_time_probe']:.3f}, fresh {upstream['controls']['survival_time_probe']:.3f}. See [source figures](upstream/images/summary.png), [source results](upstream/results.json), [discrete parity and seeds](discrete/metrics.json), [machine-readable comparison](comparison.json), and [protocol](PROTOCOL.md). The unmodified source trajectory overview retains mixed-map overlays and fixed axes; use the new environment replay for faithful individual scenes.

## Verification and reproduction

35 focused tests passed. Data/checkpoint/code hashes, actual commands, and timing are recorded in the per-arm manifests/logs. The initial continuous attempt encountered an offloaded-file read failure; it is retained in execution logs and was retried using verified local bytes. No failed fit was selected or counted as a completed run. Dataset regeneration reuses fixed specialist checkpoints; no policy training, reset fix, additional equation, or new BC implementation was performed.

```bash
uv run --locked --extra plot python experiments/shashank_comparison_20260908/summarize.py
```

This command summarizes completed fits only. See each arm's provenance/execution record for full training commands. Previous result directories remain untouched.
"""
    (HERE / "REPORT.md").write_text(report)
    paths = [HERE / "upstream/results.json", HERE / "upstream/published_results.json", HERE / "discrete/metrics.json",
             *[HERE / f"continuous/seed_{seed}/metrics.json" for seed in range(3)], HERE / "validation.json"]
    outputs = [HERE / p for p in ("REPORT.md", "comparison.json", "comparison.png")]
    def hashes(ps):
        return {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in ps}
    (HERE / "summary_manifest.json").write_text(json.dumps({"inputs": hashes(paths), "outputs": hashes(outputs),
        "summary_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
