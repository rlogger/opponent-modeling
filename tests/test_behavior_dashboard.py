"""Static dashboard packaging with tiny saved-data fixtures; no model runtime."""
import base64
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path

import pytest

PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=")
GIF = base64.b64decode("R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7")
ROOT_PLOTS = ("encoding_viz.png", "latent_swap_rollouts.png")
DIAGNOSTICS = ("latent_prototypes.png", "specialist_match.png", "causal_prefix.png",
               "action_controls.png", "action_vectors.png", "physics_error.png")
ROLLOUTS = ("latent_swap_animation.gif", "latent_swap_animation_final.png", "additional_scenes.png")
TABS = ("recorded", "world", "latents", "actions", "forecasts")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False) + "\n")


class Document(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.elements, self.scripts, self.current = [], {}, None
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.elements.append((tag, attrs))
        if tag == "script" and attrs.get("type") == "application/json":
            self.current = attrs["id"]
            self.scripts[self.current] = ""

    def handle_data(self, data):
        if self.current is not None:
            self.scripts[self.current] += data

    def handle_endtag(self, tag):
        if tag == "script":
            self.current = None


@pytest.fixture
def exporter(monkeypatch):
    # Import and execute packaging with the scientific runtime unavailable.
    for name in ("jax", "flax", "numpy", "sklearn", "matplotlib"):
        monkeypatch.setitem(sys.modules, name, None)

    def forbidden(*args, **kwargs):
        raise AssertionError("Static packaging must not invoke a subprocess")

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    path = Path(__file__).resolve().parents[1] / "scripts/build_behavior_dashboard.py"
    spec = importlib.util.spec_from_file_location("behavior_dashboard_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def saved(tmp_path):
    controller, world, source, out = (tmp_path / name for name in ("controller", "world", "source", "out"))
    for path in (controller, world, source / "visuals"):
        path.mkdir(parents=True)
    synthetic_hash = hashlib.sha256(b"saved diagnostic input").hexdigest()
    source_manifest = {"artifacts": {"latents.npz": synthetic_hash, "metrics.json": synthetic_hash},
                       "dataset": {"sha256": synthetic_hash}}
    for name in ROOT_PLOTS:
        (source / name).write_bytes(PNG)
        source_manifest["artifacts"][name] = digest(source / name)
    write_json(source / "manifest.json", source_manifest)
    diagnostics, rollouts = {}, {}
    for names, hashes in ((DIAGNOSTICS, diagnostics), (ROLLOUTS, rollouts)):
        for name in names:
            path = source / "visuals" / name
            path.write_bytes(GIF if name.endswith(".gif") else PNG)
            hashes[name] = digest(path)
    write_json(source / "visuals/diagnostics_manifest.json", {
        "input_hashes": {name: synthetic_hash for name in ("latents.npz", "metrics.json", "dataset.npz")},
        "output_hashes": diagnostics})
    write_json(source / "visuals/rollout_manifest.json", {
        "original_manifest_sha256": digest(source / "manifest.json"), "outputs": rollouts})

    episodes = []
    for label, objective in enumerate(("capture", "risk", "curious")):
        for name in ("mappo", "tdmpc"):
            row = {"controller": name, "objective": objective, "pair": 0, "episode": label,
                   "map": label, "length": 2, "checkpoint": 2, "stop": "timeout",
                   "xy": [[0, 0, 1, 1], [0.1, 0, 0.9, 1], [0.2, 0, 0.8, 1]],
                   "blue": [[-0.2, 0], [-0.2, 0]], "red": [[0.2, 0], [0.2, 0]],
                   "collected": [0, 1, 1], "pc": [label, 0]}
            if name == "tdmpc":
                row["zpc"] = [[0, 0], [label + 0.5, 1]]
            episodes.append(row)
    controller_data = {"episodes": episodes, "maps": [
        {"resources": [[i * 0.1, 0] for i in range(16)], "lava": [[0, 0.5, 0.2]] * 3} for _ in range(3)]}
    write_json(controller / "trajectories.json", controller_data)
    write_json(controller / "manifest.json", {"output_sha256": digest(controller / "trajectories.json")})

    scene = {"episode": 2, "objective": "capture", "state0": [0.0] * 66,
             "blue": [[0.1, 0], [0.2, 0]], "reference": episodes[0]["xy"], "models": []}
    for mode in ("implicit", "conditioned", "factored"):
        paths = [{"xy": episodes[0]["xy"], "red": [[0.2, 0], [0.3, 0]] if mode == "factored" else None,
                  "stop": 3} for _ in range(3)]
        model = {"mode": mode, "paths": paths}
        if mode == "factored":
            model["clamped_paths"] = paths
        scene["models"].append(model)
    world_data = {"prototypes": [[float(k)] * 8 for k in range(3)], "prototype_pc": [[k, k] for k in range(3)],
                  "latent_cloud": [{"point": [i * 0.1, i * 0.2], "label": i % 3, "episode": i} for i in range(9)],
                  "scenes": [scene]}
    write_json(world / "world-model-data.json", world_data)
    write_json(world / "manifest.json", {"source_manifest_sha256": digest(source / "manifest.json"),
        "artifacts": {"world-model-data.json": digest(world / "world-model-data.json")}})
    return controller, world, source, out, controller_data, world_data


def test_build_preserves_all_saved_points_and_embeds_all_original_figures(exporter, saved):
    controller, world, source, out, controller_data, world_data = saved
    manifest = exporter.build_dashboard(controller, world, source, out)
    index = out / "index.html"
    text = index.read_text()
    document = Document(text)
    assert "<!doctype html>" in text.lower() and "<html" in text.lower()
    assert json.loads(document.scripts["ci-data"]) == controller_data
    assert json.loads(document.scripts["wm-data"]) == world_data
    assert json.loads((out / "manifest.json").read_text()) == manifest
    assert manifest["artifacts"]["index.html"] == digest(index)
    assert manifest["operations"] == {"training": False, "new_rollouts": False,
                                      "model_inference": False, "probe_fitting": False}
    assert manifest["inputs"]
    figures = manifest["figures"]
    assert len(figures) == 11
    assert {Path(row["name"]).name for row in figures} == set(ROOT_PLOTS + DIAGNOSTICS + ROLLOUTS)
    assert all(row["sha256"] and row["source_manifest"] for row in figures)
    images = [attrs for tag, attrs in document.elements if tag == "img" and attrs.get("src", "").startswith("data:")]
    poster = "data:image/png;base64," + base64.b64encode(PNG).decode()
    animation = "data:image/gif;base64," + base64.b64encode(GIF).decode()
    assert len(images) == 11 and all(attrs["src"] == poster for attrs in images)
    animated = [attrs for attrs in images if "data-animation" in attrs]
    assert len(animated) == 1
    assert animated[0]["data-animation"] == animation
    assert animated[0]["data-poster"] == poster  # no automatic GIF playback


def test_tabs_have_aria_relationships_and_keyboard_navigation(exporter, saved):
    out = saved[3]
    exporter.build_dashboard(*saved[:4])
    text = (out / "index.html").read_text()
    document = Document(text)
    elements = {attrs["id"]: attrs for _, attrs in document.elements if "id" in attrs}
    assert any(attrs.get("role") == "tablist" for _, attrs in document.elements)
    selected = []
    for name in TABS:
        tab_id, panel_id = f"bd-tab-{name}", f"bd-panel-{name}"
        tab, panel = elements[tab_id], elements[panel_id]
        assert tab["role"] == "tab" and tab["aria-controls"] == panel_id
        assert panel["role"] == "tabpanel" and panel["aria-labelledby"] == tab_id
        assert tab["aria-selected"] in {"true", "false"}
        selected.append(tab["aria-selected"] == "true")
    assert sum(selected) == 1
    for key in ("keydown", "ArrowLeft", "ArrowRight", "Home", "End"):
        assert key in text


def test_plot_labels_have_accessible_context_specific_explanations(exporter, saved):
    exporter.build_dashboard(*saved[:4])
    text = (saved[3] / "index.html").read_text()
    document = Document(text)
    elements = {attrs["id"]: attrs for _, attrs in document.elements if "id" in attrs}
    for chart, descriptions in (("ci-cloud", ("ci-axis-help", "ci-point-help")),
                                ("wm-cloud", ("wm-axis-help",))):
        assert elements[chart]["aria-describedby"].split() == list(descriptions)
        assert all(description in elements for description in descriptions)
    assert sum(tag == "details" for tag, _ in document.elements) >= 2
    for explanation in ("not arena positions or probabilities", "not z[0] and z[1]",
                        "mean-centered, unscaled", "standardized training latents",
                        "not physical x/y coordinates", "not independent samples",
                        "original eight entries, not the two PCA scores",
                        "all context points belong to TD-MPC2"):
        assert explanation in text
    assert "find('axis-help').textContent=isContext?" in text
    assert "find('point-help').textContent=view==='context'?" in text


def test_page_uses_browser_defaults_but_retains_plot_primitives(exporter, saved):
    exporter.build_dashboard(*saved[:4])
    text = (saved[3] / "index.html").read_text()
    assert "bd-theme" not in text
    assert "light-dark(" not in text
    assert "font-family:" not in text
    assert "body {" not in text
    assert "--foreground:CanvasText" in text
    assert "--blue:" in text and "--red:" in text
    assert "[hidden] { display:none !important; }" in text


def test_responsive_layout_keeps_figures_and_controls_within_containers(exporter, saved):
    exporter.build_dashboard(*saved[:4])
    text = (saved[3] / "index.html").read_text()
    for rule in ("max-width:80rem", "minmax(min(100%,22rem),1fr)",
                 "minmax(min(100%,18rem),1fr)", "max-width:56rem", "font-size:1rem",
                 "max-height:min(75svh,48rem)", "max-width:none; max-height:none",
                 "overflow:auto", "@media (pointer:coarse)", "min-height:44px"):
        assert rule in text
    assert "media.setAttribute('role', 'region')" in text
    assert "if (fullSize) media.tabIndex = 0" in text
    assert "rect.width-tip.offsetWidth" in text


@pytest.mark.parametrize("inspector", ["controller", "world_model"])
def test_chart_geometry_uses_svg_width_and_preserves_square_coordinates(inspector):
    # Execute the actual frame functions with a minimal D3 drawing stub, not a
    # browser or scientific runtime. The SVG can be narrower than its parent.
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for chart geometry checks")
    source = (Path(__file__).resolve().parents[1] / "scripts" / f"{inspector}_inspector.html").read_text()
    frame = re.search(r"  +function frame\(.*?(?=\n  +function )", source, re.S).group()
    call = ("frame(node, 300, [0, 10], [0, 10], 'x', 'y', square)"
            if inspector == "controller" else "frame(node, [[0, 10], [0, 10]], ['x', 'y'], square)")
    script = """
const assert = require('node:assert/strict');
const selection = new Proxy({}, {get: () => () => selection});
const axis = () => ({ticks() {return this;}, tickFormat() {return this;}});
const d3 = {select: () => selection, axisBottom: axis, axisLeft: axis,
  scaleLinear() {
    let domain, range;
    const scale = v => range[0] + (v-domain[0])/(domain[1]-domain[0])*(range[1]-range[0]);
    scale.domain = v => {domain=v; return scale;};
    scale.range = v => {range=v; return scale;};
    return scale;
  }};
const format = String;
""" + frame + """
for (const width of [240, 280, 320, 375, 416, 512, 768, 896, 1280, 2560]) {
  const node = {id:'test', getBoundingClientRect: () => ({width}),
    parentElement: {getBoundingClientRect: () => ({width:width+200})}};
  for (const square of [true, false]) {
    const p = CALL;
    assert.equal(p.width, width);
    assert.ok(p.left >= 0 && p.right <= width && p.right > p.left);
    assert.ok(p.top >= 0 && p.bottom < p.height && p.bottom > p.top);
    assert.ok(p.height <= 428, 'large screens must not produce giant arenas');
    assert.ok(Number.isFinite(p.x(5)) && Number.isFinite(p.y(5)));
    if (square) {
      assert.ok(Math.abs((p.right-p.left)-(p.bottom-p.top)) < 1e-8);
      assert.ok(Math.abs((p.x(10)-p.x(0))+(p.y(10)-p.y(0))) < 1e-8);
    }
  }
}
""".replace("CALL", call)
    subprocess.run([node, "-e", script], check=True, capture_output=True, text=True, timeout=15)


@pytest.mark.parametrize("which", ["controller", "world"])
def test_refuses_changed_or_missing_json_before_writing(exporter, saved, which):
    directory, name = ((saved[0], "trajectories.json") if which == "controller"
                       else (saved[1], "world-model-data.json"))
    path = directory / name
    path.write_text(path.read_text() + " ")
    with pytest.raises((ValueError, FileNotFoundError)):
        exporter.build_dashboard(*saved[:4])
    assert not (saved[3] / "index.html").exists()
    path.unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        exporter.build_dashboard(*saved[:4])


@pytest.mark.parametrize("relative", ["encoding_viz.png", "visuals/action_vectors.png", "visuals/latent_swap_animation.gif"])
def test_refuses_changed_or_missing_figure_from_each_manifest(exporter, saved, relative):
    path = saved[2] / relative
    path.write_bytes(b"tampered")
    with pytest.raises((ValueError, FileNotFoundError)):
        exporter.build_dashboard(*saved[:4])
    assert not (saved[3] / "index.html").exists()
    path.unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        exporter.build_dashboard(*saved[:4])


@pytest.mark.parametrize("which", ["world", "rollout", "diagnostics"])
def test_refuses_stale_original_manifest_binding(exporter, saved, which):
    path = (saved[1] / "manifest.json" if which == "world" else saved[2] / "visuals" /
            ("rollout_manifest.json" if which == "rollout" else "diagnostics_manifest.json"))
    manifest = json.loads(path.read_text())
    if which == "diagnostics":
        manifest["input_hashes"]["latents.npz"] = "0" * 64
    else:
        manifest["source_manifest_sha256" if which == "world" else "original_manifest_sha256"] = "0" * 64
    write_json(path, manifest)
    with pytest.raises(ValueError):
        exporter.build_dashboard(*saved[:4])
    assert not (saved[3] / "index.html").exists()


def test_json_script_escape_preserves_values_without_executable_markup(exporter, saved):
    hostile = '</script><img id="unexpected-executable-markup" src=x onerror=alert(1)>'
    for directory, name, manifest_key in ((saved[0], "trajectories.json", "output_sha256"),
                                           (saved[1], "world-model-data.json", "artifacts")):
        data = json.loads((directory / name).read_text())
        data["inspection_note"] = hostile
        write_json(directory / name, data)
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest_key == "artifacts":
            manifest["artifacts"][name] = digest(directory / name)
        else:
            manifest[manifest_key] = digest(directory / name)
        write_json(directory / "manifest.json", manifest)
    exporter.build_dashboard(*saved[:4])
    text = (saved[3] / "index.html").read_text()
    document = Document(text)
    assert hostile not in text
    assert not any(attrs.get("id") == "unexpected-executable-markup" for _, attrs in document.elements)
    for script in ("ci-data", "wm-data"):
        assert json.loads(document.scripts[script])["inspection_note"] == hostile
