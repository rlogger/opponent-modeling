#!/usr/bin/env python3
"""Combine verified saved views and figures; no model calls or new analysis."""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGURES = (
    ("encoding_viz.png", "0s representation overview",
     "Saved training/held-out episode PCA, checkpoint-colored PCA, training losses, "
     "held-out window PCA, and historical probe/length-control/ARI results. No refitting.", "latents"),
    ("visuals/latent_prototypes.png", "Latent point clouds and three mean vectors",
     "Held-out episode latents in training-fitted PCA; stars are train-only means. "
     "Heatmap cells are eight raw latent coordinates, not strategy probabilities.", "latents"),
    ("visuals/causal_prefix.png", "Causal-prefix inference",
     "Previously computed train-only diagnostic probes on saved causal prefixes, using the same "
     "held-out cohort. The full-episode reference sees later behavior. No probe was fitted for this dashboard.", "latents"),
    ("visuals/specialist_match.png", "Prototype versus specialist matching",
     "Saved action errors on identical states. Outlined cells are each target column's best match; "
     "the diagonal is the intended behavior match, not a guaranteed outcome.", "actions"),
    ("visuals/action_controls.png", "Action-prediction controls",
     "Historical causal-history, correct-mean, wrong-mean and zero-context controls. "
     "Zero context uses the same 0s decoder with z=0; it is not separately trained vanilla BC.", "actions"),
    ("visuals/action_vectors.png", "Predicted and specialist action vectors",
     "Saved same-state actions: solid arrows are the 0s decoder; dashed arrows are the specialist. "
     "One deterministic example per recorded objective; these are commands, not trajectories.", "actions"),
    ("visuals/physics_error.png", "Physics prediction across horizons",
     "Previously measured position error with recorded joint actions. This isolates learned physics, "
     "not opponent prediction quality or closed-loop controller performance.", "actions"),
    ("latent_swap_rollouts.png", "Original Equation 3 prototype rollouts",
     "Saved base-model predictions under identical initial states and blue commands. "
     "These are imagined paths, not simulator counterfactual ground truth.", "forecasts"),
    ("visuals/additional_scenes.png", "Additional held-out starting scenes",
     "Three saved risk-source episodes (1000, 1100, 1199), each with three prototype swaps. "
     "Not one scene per objective. Fixed-horizon predictions lack the new inspector's stop markings.", "forecasts"),
    ("visuals/latent_swap_animation.gif", "Original latent-swap animation",
     "Saved fixed-horizon Equation 3 forecasts. Playback starts only on request; "
     "states beyond capture are extrapolations, not confirmed episode continuations.", "forecasts"),
    ("visuals/latent_swap_animation_final.png", "Animation endpoint",
     "The saved final frame of the same original animation, retained for static inspection.", "forecasts"),
)


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def json_bytes(content: bytes):
    def reject(value):
        raise ValueError(f"Nonfinite JSON value: {value}")
    return json.loads(content, parse_constant=reject)


def verified(path: Path, expected: str) -> bytes:
    content = path.read_bytes()
    if digest(content) != expected:
        raise ValueError(f"Saved artifact hash mismatch: {path}")
    return content


def inline_json(value) -> str:
    return json.dumps(value, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")


def substitute(template: str, marker: str, value: str) -> str:
    if template.count(marker) != 1:
        raise ValueError(f"Expected exactly one template marker: {marker}")
    return template.replace(marker, value)


def build_dashboard(controller_dir, world_dir, source_dir, out):
    controller_dir, world_dir, source_dir, out = map(Path, (controller_dir, world_dir, source_dir, out))
    inputs = {}

    def manifest_at(directory, name="manifest.json"):
        path = directory / name
        content = path.read_bytes()
        inputs[str(path.resolve())] = digest(content)
        return json_bytes(content), digest(content)

    controller, _ = manifest_at(controller_dir)
    world, _ = manifest_at(world_dir)
    source, source_hash = manifest_at(source_dir)
    diagnostics, _ = manifest_at(source_dir / "visuals", "diagnostics_manifest.json")
    forecasts, _ = manifest_at(source_dir / "visuals", "rollout_manifest.json")
    if world["source_manifest_sha256"] != source_hash or forecasts["original_manifest_sha256"] != source_hash:
        raise ValueError("World-model or forecast source manifest does not match the original 0s run")
    for name, expected in diagnostics["input_hashes"].items():
        original = source["dataset"]["sha256"] if name == "dataset.npz" else source["artifacts"][name]
        if expected != original:
            raise ValueError(f"Historical diagnostic source mismatch: {name}")

    controller_path, world_path = controller_dir / "trajectories.json", world_dir / "world-model-data.json"
    controller_data = verified(controller_path, controller["output_sha256"])
    world_data = verified(world_path, world["artifacts"][world_path.name])
    inputs.update({str(controller_path.resolve()): digest(controller_data), str(world_path.resolve()): digest(world_data)})
    templates = ROOT / "scripts"
    controller_template = templates / "controller_inspector.html"
    world_template = templates / "world_model_inspector.html"
    shell_template = templates / "behavior_dashboard.html"
    controller_view = substitute(controller_template.read_text(), "__TRAJECTORY_DATA__", inline_json(json_bytes(controller_data)))
    world_view = substitute(world_template.read_text(), "__WORLD_MODEL_DATA__", inline_json(json_bytes(world_data)))

    figures, images = [], {}
    for name, title, caption, group in FIGURES:
        if name.startswith("visuals/"):
            short = Path(name).name
            diagnostic = short in diagnostics["output_hashes"]
            expected = diagnostics["output_hashes"][short] if diagnostic else forecasts["outputs"][short]
            source_manifest = "visuals/diagnostics_manifest.json" if diagnostic else "visuals/rollout_manifest.json"
        else:
            expected, source_manifest = source["artifacts"][name], "manifest.json"
        content = verified(source_dir / name, expected)
        kind = "gif" if name.endswith(".gif") else "png"
        images[name] = f"data:image/{kind};base64," + base64.b64encode(content).decode("ascii")
        figures.append({"name": name, "sha256": digest(content), "source_manifest": source_manifest,
                        "title": title, "caption": caption, "group": group})

    galleries = {group: [] for group in ("latents", "actions", "forecasts")}
    for figure in figures:
        name, title, caption, group = (figure[k] for k in ("name", "title", "caption", "group"))
        source_label = f"Saved original continuous_0s / {name}"
        identifier = "bd-figure-" + Path(name).stem.replace("_", "-")
        alt = html.escape(title + ". " + caption, quote=True)
        if name.endswith(".gif"):
            poster = images["visuals/latent_swap_animation_final.png"]
            media = (f'<button type="button" class="btn" data-animation-toggle aria-pressed="false">Play saved animation</button>'
                     f'<img class="bd-animation" src="{poster}" data-poster="{poster}" '
                     f'data-animation="{images[name]}" alt="{alt}" loading="lazy">')
        else:
            media = f'<img src="{images[name]}" alt="{alt}" loading="lazy" decoding="async">'
        galleries[group].append(
            f'<figure id="{identifier}"><figcaption><h3>{html.escape(title)}</h3>'
            f'<p>{html.escape(caption)}</p><small>{html.escape(source_label)}</small></figcaption>{media}</figure>')
    document = shell_template.read_text()
    for marker, value in (("__CONTROLLER_VIEW__", controller_view), ("__WORLD_VIEW__", world_view),
                          ("__LATENT_FIGURES__", "\n".join(galleries["latents"])),
                          ("__ACTION_FIGURES__", "\n".join(galleries["actions"])),
                          ("__FORECAST_FIGURES__", "\n".join(galleries["forecasts"]))):
        document = substitute(document, marker, value)
    result = {
        "schema": 1, "inputs": inputs, "figures": figures,
        "source_roles": {"recorded": "Adapted factored 0s controller and MAPPO; recorded real episodes",
                         "world": "Offline Eq1/Eq2 and original base Eq3; fixed-action imaginations",
                         "figures": "Original continuous_0s seed0 historical diagnostics; not newly computed"},
        "operations": {"training": False, "new_rollouts": False, "model_inference": False, "probe_fitting": False},
        "code": {str(path.relative_to(ROOT)): digest(path.read_bytes()) for path in
                 (Path(__file__), controller_template, world_template, shell_template)},
        "artifacts": {"index.html": digest(document.encode())},
    }
    # Validate every input before creating output. Existing research artifacts are never rewritten.
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(document)
    (out / "manifest.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controllers", type=Path, default=ROOT / "experiments/controller_inspection")
    parser.add_argument("--world-models", type=Path, default=ROOT / "experiments/world_model_inspection")
    parser.add_argument("--source", type=Path, default=ROOT / "experiments/continuous_0s")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/behavior_dashboard")
    args = parser.parse_args()
    build_dashboard(args.controllers, args.world_models, args.source, args.out)
    print(f"Saved dashboard: {args.out / 'index.html'}. No training, inference, rollouts, or probe fitting.")


if __name__ == "__main__":
    main()
