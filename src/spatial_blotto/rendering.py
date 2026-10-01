"""Self-contained HTML replays for spatial Blotto rollouts.

The viewer reads recorded states and controller diagnostics. Rendering never
runs a controller or predicts future targets, so it also supports policy replays.
"""

from __future__ import annotations

import json
from pathlib import Path


def render_html(payload: dict) -> str:
    """Return a standalone viewer with strict, script-safe embedded JSON.

    ``payload`` follows the version 1 rollout schema. Targets and target-zone
    allocations belong to each recorded frame; ``-1`` marks unknown assignments.
    Non-finite values are rejected before any output is written.
    """
    serialized = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    serialized = serialized.replace("<", "\\u003c")
    return HTML.replace("__REPLAY_DATA__", serialized)


def write_replay(path: Path, payload: dict) -> Path:
    """Write a portable replay and return its resolved output path."""
    html = render_html(payload)
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Spatial Blotto · controller replay</title>
<style>
:root { color-scheme: light; font: 15px/1.5 system-ui, sans-serif; color: #213246; background: #f2f5f7; }
* { box-sizing: border-box; }
body { margin: 0; }
main { max-width: 1200px; margin: auto; padding: 32px 24px; }
h1 { font-size: clamp(26px, 4vw, 38px); letter-spacing: -.035em; margin: 0 0 6px; }
h2 { font-size: 17px; margin: 0 0 12px; }
p { margin: 0 0 14px; }
.eyebrow { text-transform: uppercase; letter-spacing: .11em; color: #596c7b; font-size: 11px; font-weight: 750; }
.intro { max-width: 780px; color: #596c7b; }
.layout { display: grid; grid-template-columns: minmax(0, 1.5fr) minmax(260px, 1fr); gap: 20px; margin-top: 24px; }
.panel { min-width: 0; padding: 20px; border: 1px solid #d9e2e8; border-radius: 14px; background: white; }
.stage { padding: 12px; }
canvas { display: block; width: 100%; aspect-ratio: 1; background: #fcfdfd; border-radius: 8px; }
.sidebar { display: grid; align-content: start; gap: 16px; }
.team { display: flex; justify-content: space-between; align-items: center; margin: 10px 0; gap: 8px; }
.controller { display: block; overflow-wrap: anywhere; font-weight: 500; font-size: 12px; }
.red { color: #ad3e48; }
.blue { color: #225fa9; }
.score { font-size: 30px; font-weight: 750; line-height: 1.1; font-variant-numeric: tabular-nums; }
.sub { color: #5d6c77; font-size: 12px; }
.zones { display: grid; gap: 8px; }
.zone { display: grid; grid-template-columns: 1fr 1.5fr 1fr; align-items: center; padding: 10px 12px; background: #f2f5f7; border-radius: 8px; }
.zone span:last-child { text-align: right; font-size: 12px; font-weight: 650; }
.controls { margin-top: 14px; display: flex; flex-wrap: wrap; align-items: center; gap: 10px; }
button, select { font: inherit; border: 1px solid #bdcbd6; border-radius: 6px; padding: 6px 12px; background: white; color: #213246; }
button { cursor: pointer; }
button.primary { background: #213246; color: white; border-color: #213246; }
button:focus-visible, input:focus-visible, select:focus-visible { outline: 3px solid #8cadc8; outline-offset: 3px; }
input[type="range"] { width: 100%; accent-color: #213246; }
.timeline { width: 100%; display: flex; justify-content: space-between; font-variant-numeric: tabular-nums; }
label { cursor: pointer; }
.checks { display: flex; flex-wrap: wrap; gap: 16px; margin-top: 14px; }
.badge { display: inline-block; padding: 3px 8px; border-radius: 5px; background: #e9eef2; font-size: 12px; font-weight: 650; }
.reward { padding-top: 12px; border-top: 1px solid #e1e7ec; margin-top: 14px; }
.mono { font-variant-numeric: tabular-nums; }
.summary { margin: 18px 0 0; font-size: 13px; color: #596c7b; }
.unknown { color: #596c7b; font-size: 12px; margin-top: 10px; }
@media (max-width: 780px) { main { padding: 20px 12px; } .layout { grid-template-columns: 1fr; } .panel { padding: 16px; } .stage { padding: 8px; } }
@media (prefers-reduced-motion: reduce) { * { scroll-behavior: auto; } }
</style>
</head>
<body>
<main>
  <p class="eyebrow">JaxMARL environment · recorded controller replay</p>
  <h1>Three zones. Two teams.</h1>
  <p class="intro"><span id="matchup"></span> A team owns a zone when it has strictly more agents inside its radius. Equal counts, including empty zones, are neutral. Positions, scores, and targets below come from the recorded episode.</p>
  <div class="layout">
    <section class="panel stage" aria-label="Replay arena and controls">
      <canvas id="arena" width="800" height="800" role="img" aria-label="Red and blue agents moving among zones A, B, and C. Counts and scores appear alongside the arena.">Your browser does not support Canvas. Zone counts and scores remain available in the adjacent panels.</canvas>
      <div class="controls">
        <button id="play" class="primary" type="button">Play</button>
        <button id="restart" type="button">Restart</button>
        <label for="speed">Speed</label>
        <select id="speed"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option><option value="4">4×</option></select>
        <div class="timeline"><label for="frame">Episode step</label><span id="time"></span></div>
        <input id="frame" type="range" min="0" value="0" step="1" aria-label="Episode step">
      </div>
      <div class="checks">
        <label><input id="routes" type="checkbox" checked> Target routes</label>
        <label><input id="trails" type="checkbox"> Trails</label>
      </div>
      <p class="sub" style="margin:12px 0 0">Overlapping markers are spread; thin solid lines point to their actual positions.</p>
    </section>
    <aside class="sidebar">
      <section class="panel">
        <h2>Controllers &amp; targets</h2>
        <p class="sub">Target allocation: A / B / C. Routes show the targets recorded at the selected frame; a controller may change them during the episode.</p>
        <div class="team red"><strong>● Red<span id="redController" class="controller"></span></strong><span id="redTarget" class="mono"></span></div>
        <div class="team blue"><strong>◆ Blue<span id="blueController" class="controller"></span></strong><span id="blueTarget" class="mono"></span></div>
        <p id="unknownTargets" class="unknown" hidden></p>
        <p class="sub" id="scenarioNote"></p>
        <span id="seed" class="badge"></span>
        <span id="status" class="badge">Initial state</span>
      </section>
      <section class="panel">
        <h2>Zone occupancy</h2>
        <div class="zones" id="zones"></div>
        <p class="sub" style="margin:12px 0 0">Counts show Red : Blue. Ownership is measured after each movement step.</p>
      </section>
      <section class="panel">
        <h2>Cumulative ownership score</h2>
        <p class="sub">One raw point per owned zone per step, counted once per team.</p>
        <div class="team"><strong class="red">● Red</strong><span id="redScore" class="score red">0</span></div>
        <div class="team"><strong class="blue">◆ Blue</strong><span id="blueScore" class="score blue">0</span></div>
        <p id="currentScore" class="sub mono"></p>
        <div class="reward">
          <strong id="rewardTitle"></strong>
          <p id="rewardRule" class="sub"></p>
          <span id="rewardValue" class="mono"></span>
        </div>
      </section>
    </aside>
  </div>
  <p id="summary" class="summary" aria-live="off"></p>
</main>
<script type="application/json" id="replay-data">__REPLAY_DATA__</script>
<script>
"use strict";
const data = JSON.parse(document.getElementById("replay-data").textContent);
const byId = id => document.getElementById(id);
const canvas = byId("arena"), ctx = canvas.getContext("2d"), frames = data.frames;
const colors = ["#bd4651", "#2b6ab8"], names = ["A", "B", "C"];
const slider = byId("frame");
const teamSize = data.teamSize, agentCount = data.agents.length;
const finalFrame = frames.positions.length - 1;
slider.max = finalFrame;
let frame = 0, playing = false, lastTime = null, elapsed = 0;
const allocation = (zoneIds, offset) => names.map((_, z) => zoneIds.slice(offset, offset+teamSize).filter(v => v === z).length).join(" / ");
byId("matchup").textContent = `${teamSize} vs ${teamSize} · ${agentCount} agents.`;
canvas.setAttribute("aria-label", `${agentCount} agents in a ${teamSize} vs ${teamSize} game among zones A, B, and C. Counts and scores appear alongside the arena.`);
byId("redController").textContent = data.controllers.red.name;
byId("blueController").textContent = data.controllers.blue.name;
byId("seed").textContent = "Seed " + data.seed;
byId("scenarioNote").textContent = data.scenario ? "Scenario: " + data.scenario : "";
byId("scenarioNote").hidden = !data.scenario;
byId("rewardTitle").textContent = "Per-agent reward · " + data.rewardMode;
byId("rewardRule").textContent = data.rewardMode === "zero_sum"
  ? "Each agent receives its team's owned zones minus the opponent's owned zones."
  : "Each agent receives its team's number of owned zones. These team rewards are not zero-sum.";
byId("zones").innerHTML = names.map((name, z) => `<div class="zone"><strong>Zone ${name}</strong><span class="mono" id="count-${z}"></span><span id="owner-${z}"></span></div>`).join("");
const margin = 60, span = canvas.width - 2*margin;
const xy = p => [margin + (p[0]+data.arena)/(2*data.arena)*span, canvas.height-margin-(p[1]+data.arena)/(2*data.arena)*span];
const drawCircle = (p, radius, fill, stroke) => { ctx.beginPath(); ctx.arc(...p, radius, 0, 2*Math.PI); if (fill) { ctx.fillStyle=fill; ctx.fill(); } if (stroke) { ctx.strokeStyle=stroke; ctx.stroke(); } };
function markerPositions(points) {
  const parents = points.map((_, i) => i);
  const root = i => { while (parents[i] !== i) i = parents[i]; return i; };
  let displayed = points.map(p => [...p]);
  // Merge overlapping glyph groups, including any new overlaps caused by fanning.
  for (let pass=0; pass<points.length; pass++) {
    let merged = false;
    for (let i=0; i<points.length; i++) for (let j=i+1; j<points.length; j++) {
      if (root(i) !== root(j) && Math.hypot(displayed[i][0]-displayed[j][0], displayed[i][1]-displayed[j][1]) < 36) {
        parents[root(j)] = root(i); merged = true;
      }
    }
    if (!merged) break;
    const groups = new Map();
    points.forEach((_, i) => { const key = root(i); if (!groups.has(key)) groups.set(key, []); groups.get(key).push(i); });
    groups.forEach(group => {
      if (group.length === 1) return;
      const spread = Math.max(26, 38/(2*Math.sin(Math.PI/group.length)));
      const center = [0, 1].map(axis => group.reduce((sum, i) => sum+points[i][axis], 0)/group.length);
      center[0] = Math.max(spread+20, Math.min(canvas.width-spread-20, center[0]));
      center[1] = Math.max(spread+20, Math.min(canvas.height-spread-20, center[1]));
      group.forEach((i, slot) => {
        const angle = -Math.PI/2 + slot*2*Math.PI/group.length;
        displayed[i] = [center[0]+spread*Math.cos(angle), center[1]+spread*Math.sin(angle)];
      });
    });
  }
  return displayed;
}
function draw() {
  slider.value = frame;
  byId("time").textContent = `${frame} / ${finalFrame} · ${(frame*data.dt).toFixed(1)} s`;
  slider.setAttribute("aria-valuetext", `Step ${frame} of ${finalFrame}`);
  byId("status").textContent = frames.done[frame] ? "Episode complete" : frame === 0 ? "Initial state" : "Episode in progress";
  const counts = frames.counts[frame], owners = frames.owners[frame];
  const targetZones = frames.targetZones[frame], targets = frames.targets[frame];
  byId("redTarget").textContent = allocation(targetZones, 0);
  byId("blueTarget").textContent = allocation(targetZones, teamSize);
  const unknown = targetZones.filter(zone => zone < 0).length;
  byId("unknownTargets").hidden = unknown === 0;
  byId("unknownTargets").textContent = `${unknown} agent${unknown === 1 ? " has" : "s have"} no recorded zone target.`;
  const owned = [owners.filter(v => v === 1).length, owners.filter(v => v === -1).length];
  names.forEach((name, z) => {
    byId(`count-${z}`).textContent = `${counts[0][z]} : ${counts[1][z]}`;
    const owner = byId(`owner-${z}`);
    owner.textContent = owners[z] === 1 ? "Red" : owners[z] === -1 ? "Blue" : "Neutral";
    owner.className = owners[z] === 1 ? "red" : owners[z] === -1 ? "blue" : "";
  });
  byId("redScore").textContent = frames.scores[frame][0];
  byId("blueScore").textContent = frames.scores[frame][1];
  byId("currentScore").textContent = `Currently owned: Red ${owned[0]} · Blue ${owned[1]}`;
  byId("rewardValue").textContent = `This step: Red ${frames.rewards[frame][0]} · Blue ${frames.rewards[frame][1]}`;
  byId("summary").textContent = `Step ${frame}. ` + names.map((name,z) => `Zone ${name}: ${counts[0][z]} red, ${counts[1][z]} blue`).join(". ") + `. Raw cumulative score: Red ${frames.scores[frame][0]}, Blue ${frames.scores[frame][1]}.`;
  if (!ctx) return;
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.lineWidth = 1;
  ctx.strokeStyle = "#e6edf1";
  for (let i=0; i<=6; i++) {
    const p=margin+i*span/6;
    ctx.beginPath(); ctx.moveTo(p,margin); ctx.lineTo(p,canvas.height-margin); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(margin,p); ctx.lineTo(canvas.width-margin,p); ctx.stroke();
  }
  ctx.strokeStyle="#c2d0db";
  ctx.strokeRect(margin,margin,span,span);
  const radius=data.radius/(2*data.arena)*span;
  data.centers.forEach((center,z) => {
    const p=xy(center), color=owners[z]===1?colors[0]:owners[z]===-1?colors[1]:"#8fa0ac";
    ctx.lineWidth=2; drawCircle(p,radius,color+"1a",color);
    ctx.fillStyle="#435669"; ctx.font="600 18px system-ui"; ctx.textAlign="center";
    ctx.fillText(`Zone ${names[z]}`,p[0],p[1]-radius-14);
    ctx.font="14px system-ui"; ctx.fillText(`${counts[0][z]} red : ${counts[1][z]} blue`,p[0],p[1]+radius+24);
  });
  const actualPositions = frames.positions[frame].map(xy);
  const displayedPositions = markerPositions(actualPositions);
  for (let i=0; i<agentCount; i++) {
    const team=i<teamSize?0:1, p=actualPositions[i], target=xy(targets[i]);
    if (byId("trails").checked && frame>0) {
      ctx.strokeStyle=colors[team]+"55"; ctx.lineWidth=2; ctx.beginPath();
      for (let t=0; t<=frame; t++) { const v=xy(frames.positions[t][i]); if(t===0) ctx.moveTo(...v); else ctx.lineTo(...v); }
      ctx.stroke();
    }
    if (byId("routes").checked && targetZones[i] >= 0) {
      ctx.strokeStyle=colors[team]+"70"; ctx.lineWidth=1.5; ctx.setLineDash([5,6]); ctx.beginPath(); ctx.moveTo(...p); ctx.lineTo(...target); ctx.stroke(); ctx.setLineDash([]);
      ctx.beginPath(); ctx.moveTo(target[0]-4,target[1]); ctx.lineTo(target[0]+4,target[1]); ctx.moveTo(target[0],target[1]-4); ctx.lineTo(target[0],target[1]+4); ctx.stroke();
    }
  }
  for (let i=0; i<agentCount; i++) {
    const actual=actualPositions[i], displayed=displayedPositions[i];
    if (Math.hypot(actual[0]-displayed[0], actual[1]-displayed[1]) > 0.1) {
      ctx.strokeStyle="#748593"; ctx.lineWidth=1; ctx.beginPath(); ctx.moveTo(...actual); ctx.lineTo(...displayed); ctx.stroke();
      drawCircle(actual,3,"white","#748593");
    }
  }
  for (let i=0; i<agentCount; i++) {
    const team=i<teamSize?0:1, p=displayedPositions[i];
    ctx.lineWidth=2; ctx.strokeStyle="white"; ctx.fillStyle=colors[team];
    if (team===0) drawCircle(p,13,colors[team],"white");
    else { ctx.beginPath(); ctx.moveTo(p[0],p[1]-16); ctx.lineTo(p[0]+16,p[1]); ctx.lineTo(p[0],p[1]+16); ctx.lineTo(p[0]-16,p[1]); ctx.closePath(); ctx.fill(); ctx.stroke(); }
    ctx.font="700 12px system-ui"; ctx.textAlign="center"; ctx.fillStyle="white"; ctx.fillText(String(i%teamSize+1),p[0],p[1]+4);
  }
}
function setPlaying(value) { playing=value; lastTime=null; elapsed=0; byId("play").textContent=playing?"Pause":"Play"; }
byId("play").addEventListener("click",()=> { if(frame===finalFrame) frame=0; setPlaying(!playing); draw(); });
byId("restart").addEventListener("click",()=> { frame=0; setPlaying(false); draw(); });
slider.addEventListener("input",()=> { frame=Number(slider.value); setPlaying(false); draw(); });
byId("routes").addEventListener("change",draw);
byId("trails").addEventListener("change",draw);
function tick(now) {
  if (playing) {
    if (lastTime!==null) elapsed+=Math.min(now-lastTime,250)*Number(byId("speed").value);
    lastTime=now;
    const duration=data.dt*1000;
    if (elapsed>=duration) { const advance=Math.floor(elapsed/duration); elapsed-=advance*duration; frame=Math.min(finalFrame,frame+advance); if(frame===finalFrame) setPlaying(false); draw(); }
  }
  requestAnimationFrame(tick);
}
draw();
requestAnimationFrame(tick);
</script>
</body>
</html>
"""
