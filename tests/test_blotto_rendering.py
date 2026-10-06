"""Portable replay export and the viewer's recorded-frame behavior."""

from __future__ import annotations

import json
import shutil
import subprocess
from html.parser import HTMLParser

import pytest

from spatial_blotto.rendering import render_html, write_replay


class ReplayDocument(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.ids = set()
        self.scripts = []
        self.external_assets = []
        self._script = None
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        if tag == "script":
            self._script = {"attrs": attrs, "text": ""}
            self.scripts.append(self._script)
        if "src" in attrs or tag == "link":
            self.external_assets.append(attrs)

    def handle_data(self, data):
        if self._script is not None:
            self._script["text"] += data

    def handle_endtag(self, tag):
        if tag == "script":
            self._script = None


def replay_payload(team_size=3):
    count = 2 * team_size
    return {
        "schemaVersion": 1,
        "seed": 12,
        "steps": 1,
        "teamSize": team_size,
        "scenario": "rotating",
        "rewardMode": "zero_sum",
        "arena": 1.0,
        "dt": 0.1,
        "radius": 0.2,
        "centers": [[-0.5, 0.2], [0, -0.3], [0.5, 0.2]],
        "agents": [f"red_{i}" for i in range(team_size)]
        + [f"blue_{i}" for i in range(team_size)],
        "controllers": {"red": {"name": "cyclic"}, "blue": {"name": "reactive"}},
        "frames": {
            "positions": [[[0.0, 0.0]] * count, [[0.1, 0.1]] * count],
            "counts": [[[0, 0, 0], [0, 0, 0]], [[team_size, 0, 0], [0, team_size, 0]]],
            "owners": [[0, 0, 0], [1, -1, 0]],
            "scores": [[0, 0], [1, 1]],
            "rewards": [[0, 0], [0, 0]],
            "done": [False, True],
            "targetZones": [[0] * team_size + [1] * team_size, [2] * count],
            "targets": [
                [[-0.5, 0.2]] * team_size + [[0, -0.3]] * team_size,
                [[0.5, 0.2]] * count,
            ],
        },
    }


@pytest.mark.parametrize("team_size", [2, 3, 6])
def test_standalone_html_has_controls_and_roundtrips_recorded_payload(team_size):
    payload = replay_payload(team_size)
    document = ReplayDocument(render_html(payload))
    assert {
        "arena", "play", "restart", "speed", "frame", "routes", "trails",
        "redController", "blueController", "redTarget", "blueTarget",
        "redScore", "blueScore", "rewardTitle", "rewardValue",
    } <= document.ids
    assert not document.external_assets
    assert len(document.scripts) == 2
    embedded, program = document.scripts
    assert embedded["attrs"]["type"] == "application/json"
    assert json.loads(embedded["text"]) == payload
    assert program["text"].strip()


def test_metadata_cannot_close_script_or_create_markup():
    payload = replay_payload()
    hostile = '</script><script>alert("unexpected")</script><img src=x>'
    payload["scenario"] = hostile
    payload["controllers"]["red"]["name"] = hostile
    html = render_html(payload)
    document = ReplayDocument(html)
    assert hostile not in html
    assert len(document.scripts) == 2
    assert not document.external_assets
    assert json.loads(document.scripts[0]["text"]) == payload


def test_write_replay_creates_portable_file_and_returns_resolved_path(tmp_path):
    path = tmp_path / "nested" / "replay.html"
    payload = replay_payload(2)
    assert write_replay(path, payload) == path.resolve()
    assert path.read_text(encoding="utf-8") == render_html(payload)


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_data_is_rejected_without_overwriting_output(tmp_path, nonfinite):
    payload = replay_payload()
    payload["frames"]["positions"][0][0][0] = nonfinite
    path = tmp_path / "replay.html"
    path.write_text("existing replay", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON"):
        write_replay(path, payload)
    assert path.read_text(encoding="utf-8") == "existing replay"


def run_viewer_javascript(tmp_path, payload, checks, canvas_setup="const context = null;"):
    """Execute the exported program against a small DOM and optional Canvas fixture."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required to exercise replay JavaScript")
    document = ReplayDocument(render_html(payload))
    script = tmp_path / "viewer-check.cjs"
    fixture = r'''
const assert = require("node:assert/strict");
const elements = new Map();
function element(id) {
  if (!elements.has(id)) elements.set(id, {
    textContent: "", value: "0", checked: false, hidden: false,
    width: 800, height: 800, listeners: {}, attributes: {},
    addEventListener(name, fn) { this.listeners[name] = fn; },
    setAttribute(name, value) { this.attributes[name] = value; },
    getContext() { return context; }
  });
  return elements.get(id);
}
const document = {getElementById: element};
let nextAnimation;
function requestAnimationFrame(callback) { nextAnimation = callback; }
element("replay-data").textContent = JSON.stringify(PAYLOAD);
element("speed").value = "1";
'''.replace("PAYLOAD", json.dumps(payload, allow_nan=False))
    script.write_text(
        canvas_setup + fixture + document.scripts[1]["text"] + checks, encoding="utf-8"
    )
    result = subprocess.run([node, str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("team_size", [2, 3, 6])
def test_viewer_scrubbing_updates_targets_scores_and_labels(tmp_path, team_size):
    payload = replay_payload(team_size)
    checks = r'''
assert.equal(element("matchup").textContent, `${teamSize} vs ${teamSize} · ${teamSize*2} agents.`);
assert.equal(element("redController").textContent, "cyclic");
assert.equal(element("blueController").textContent, "reactive");
assert.equal(element("scenarioNote").textContent, "Scenario: rotating");
assert.equal(element("redTarget").textContent, `${teamSize} / 0 / 0`);
assert.equal(element("blueTarget").textContent, `0 / ${teamSize} / 0`);
assert.match(element("rewardTitle").textContent, /zero_sum/);
assert.equal(element("redScore").textContent, 0);
element("frame").value = "1";
element("frame").listeners.input();
assert.equal(element("redTarget").textContent, `0 / 0 / ${teamSize}`);
assert.equal(element("blueTarget").textContent, `0 / 0 / ${teamSize}`);
assert.equal(element("redScore").textContent, 1);
assert.equal(element("count-0").textContent, `${teamSize} : 0`);
assert.equal(element("owner-0").textContent, "Red");
assert.equal(element("owner-1").textContent, "Blue");
assert.equal(element("status").textContent, "Episode complete");
element("restart").listeners.click();
assert.equal(element("frame").value, 0);
assert.equal(element("status").textContent, "Initial state");
assert.equal(element("play").textContent, "Play");
element("play").listeners.click();
assert.equal(element("play").textContent, "Pause");
nextAnimation(0);
nextAnimation(100);
assert.equal(element("status").textContent, "Episode complete");
assert.equal(element("play").textContent, "Play");
'''
    run_viewer_javascript(tmp_path, payload, checks)


@pytest.mark.parametrize("team_size", [2, 3, 6])
def test_coincident_markers_remain_visible_with_true_position_anchors(tmp_path, team_size):
    payload = replay_payload(team_size)
    canvas_setup = r'''
const labels = [], segments = [], anchors = [];
let path = [], dash = [];
const context = {
  beginPath() { path = []; },
  moveTo(x, y) { path.push([x, y]); },
  lineTo(x, y) { path.push([x, y]); },
  stroke() { segments.push({points: path.slice(), dash: dash.slice()}); },
  arc(x, y, radius) { if (radius === 3) anchors.push([x, y]); },
  fillText(text, x, y) { if (/^\d+$/.test(text)) labels.push([x, y-4]); },
  setLineDash(value) { dash = value; },
  clearRect() {}, strokeRect() {}, fill() {}, closePath() {}
};
'''
    checks = r'''
// Every recorded agent gets a separate visible glyph at the converged location.
assert.equal(labels.length, agentCount);
for (let i=0; i<labels.length; i++) for (let j=i+1; j<labels.length; j++) {
  assert.ok(Math.hypot(labels[i][0]-labels[j][0], labels[i][1]-labels[j][1]) >= 36);
}
assert.equal(anchors.length, agentCount);
anchors.forEach(p => assert.deepEqual(p, [400, 400]));
labels.forEach(p => assert.ok(segments.some(s => s.dash.length === 0 &&
  JSON.stringify(s.points) === JSON.stringify([[400, 400], p]))));
// Offsets are display-only: recorded data, counts and owners are unchanged.
assert.ok(frames.positions[0].every(p => p[0] === 0 && p[1] === 0));
assert.equal(element("count-0").textContent, "0 : 0");
assert.equal(element("owner-0").textContent, "Neutral");
element("routes").checked = true;
element("trails").checked = true;
segments.length = 0;
element("frame").value = "1";
element("frame").listeners.input();
// Target routes and trails still originate and terminate at physical coordinates.
const routes = segments.filter(s => s.dash.length > 0);
assert.equal(routes.length, agentCount);
const near = (p, q) => Math.hypot(p[0]-q[0], p[1]-q[1]) < 1e-8;
routes.forEach(s => assert.ok(near(s.points[0], [434, 366])));
assert.equal(segments.filter(s => s.points.length === 2 &&
  near(s.points[0], [400, 400]) && near(s.points[1], [434, 366])).length, agentCount);
assert.equal(element("count-0").textContent, `${teamSize} : 0`);
assert.equal(element("owner-0").textContent, "Red");
'''
    run_viewer_javascript(tmp_path, payload, checks, canvas_setup)
