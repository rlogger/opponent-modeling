#!/usr/bin/env python3
"""Run scripted spatial Blotto agents and export a self-contained HTML replay.

Run from an editable project install, or use ``PYTHONPATH=src python
scripts/demo_spatial_blotto.py``. No policies are trained by this demo.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp

from spatial_blotto import SpatialBlotto


def scripted_targets(env: SpatialBlotto, scenario: str) -> tuple[jax.Array, list[int]]:
    """Choose destinations and small in-zone offsets that keep agents legible."""
    zone_ids = [0, 0, 1, 1, 1, 2] if scenario == "cyclic" else [0, 1, 2, 0, 1, 2]
    offsets = [[0.0, 0.0] for _ in zone_ids]
    for zone in range(3):
        occupants = [i for i, target in enumerate(zone_ids) if target == zone]
        for slot, agent in enumerate(occupants):
            angle = 2 * math.pi * slot / len(occupants)
            radius = 0.42 * float(env.zone_radius) if len(occupants) > 1 else 0.0
            offsets[agent] = [radius * math.cos(angle), radius * math.sin(angle)]
    return env.zone_centers[jnp.asarray(zone_ids)] + jnp.asarray(offsets), zone_ids


def simulate(seed: int, steps: int, scenario: str, reward_mode: str) -> dict:
    """Record a finite episode without the public step method's automatic reset."""
    env = SpatialBlotto(team_size=3, max_steps=steps, reward_mode=reward_mode)
    targets, zone_ids = scripted_targets(env, scenario)
    reset_key, rollout_key = jax.random.split(jax.random.PRNGKey(seed))
    _, initial_state = env.reset(reset_key)

    def record(state, team_reward):
        return {
            "positions": state.p_pos,
            "counts": env.zone_counts(state),
            "owners": env.zone_owners(state),
            "scores": state.team_scores,
            "rewards": team_reward,
            "done": state.done,
        }

    def advance(carry, _):
        key, state = carry
        key, step_key = jax.random.split(key)
        actions = (targets - state.p_pos) / (env.dt * env.max_speed)
        actions /= jnp.maximum(1.0, jnp.linalg.norm(actions, axis=-1, keepdims=True))
        action_dict = {agent: actions[i] for i, agent in enumerate(env.agents)}
        _, state, rewards, _, _ = env.step_env(step_key, state, action_dict)
        team_reward = jnp.stack([rewards[env.agents[0]], rewards[env.agents[3]]])
        return (key, state), record(state, team_reward)

    def rollout():
        _, frames = jax.lax.scan(advance, (rollout_key, initial_state), None, length=steps)
        first = record(initial_state, jnp.zeros(2))
        return jax.tree_util.tree_map(
            lambda start, rest: jnp.concatenate([start[None], rest], axis=0), first, frames
        )

    frames = jax.device_get(jax.jit(rollout)())
    return {
        "seed": seed,
        "steps": steps,
        "scenario": scenario,
        "rewardMode": reward_mode,
        "arena": float(env.arena),
        "dt": float(env.dt),
        "radius": float(env.zone_radius),
        "centers": jax.device_get(env.zone_centers).tolist(),
        "targets": jax.device_get(targets).tolist(),
        "targetZones": zone_ids,
        "agents": list(env.agents),
        "frames": {name: value.tolist() for name, value in frames.items()},
    }


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Spatial Blotto · scripted replay</title>
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
.panel { padding: 20px; border: 1px solid #d9e2e8; border-radius: 14px; background: white; }
.stage { padding: 12px; }
canvas { display: block; width: 100%; aspect-ratio: 1; background: #fcfdfd; border-radius: 8px; }
.sidebar { display: grid; align-content: start; gap: 16px; }
.team { display: flex; justify-content: space-between; align-items: center; margin: 10px 0; gap: 8px; }
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
.checks { display: flex; gap: 16px; margin-top: 14px; }
.badge { display: inline-block; padding: 3px 8px; border-radius: 5px; background: #e9eef2; font-size: 12px; font-weight: 650; }
.reward { padding-top: 12px; border-top: 1px solid #e1e7ec; margin-top: 14px; }
.mono { font-variant-numeric: tabular-nums; }
.summary { margin: 18px 0 0; font-size: 13px; color: #596c7b; }
@media (max-width: 780px) { main { padding: 20px 12px; } .layout { grid-template-columns: 1fr; } .panel { padding: 16px; } .stage { padding: 8px; } }
@media (prefers-reduced-motion: reduce) { * { scroll-behavior: auto; } }
</style>
</head>
<body>
<main>
  <p class="eyebrow">JaxMARL environment · deterministic scripted control</p>
  <h1>Three zones. Two teams.</h1>
  <p class="intro">Six agents move toward fixed destinations. A team owns a zone when it has strictly more agents inside its radius. Equal counts, including empty zones, are neutral. This replay demonstrates the environment; the agents are not learned policies.</p>
  <div class="layout">
    <section class="panel stage" aria-label="Replay arena and controls">
      <canvas id="arena" width="800" height="800" role="img" aria-label="Six agents moving among zones A, B, and C. Counts and scores appear alongside the arena.">Your browser does not support Canvas. Zone counts and scores remain available in the adjacent panels.</canvas>
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
    </section>
    <aside class="sidebar">
      <section class="panel">
        <h2>Scripted destinations</h2>
        <p class="sub">Allocation order: A / B / C. All agents aim directly at a fixed point inside their assigned zone.</p>
        <div class="team red"><strong>● Red</strong><span id="redTarget" class="mono"></span></div>
        <div class="team blue"><strong>◆ Blue</strong><span id="blueTarget" class="mono"></span></div>
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
slider.max = data.steps;
let frame = 0, playing = false, lastTime = null, elapsed = 0;
const allocation = offset => names.map((_, z) => data.targetZones.slice(offset, offset+3).filter(v => v === z).length).join(" / ");
byId("redTarget").textContent = allocation(0);
byId("blueTarget").textContent = allocation(3);
byId("seed").textContent = "Seed " + data.seed;
byId("scenarioNote").textContent = data.scenario === "cyclic"
  ? "At their destinations: Red owns A; Blue owns B and C. Scores accumulated during travel can differ from the final ownership split."
  : "At their destinations: each zone contains one agent from each team, so all three zones are neutral. Travel can still earn points.";
byId("rewardTitle").textContent = "Training reward · " + data.rewardMode;
byId("rewardRule").textContent = data.rewardMode === "zero_sum"
  ? "Each agent receives its team's owned zones minus the opponent's owned zones."
  : "Each agent receives its team's number of owned zones. These team rewards are not zero-sum.";
byId("zones").innerHTML = names.map((name, z) => `<div class="zone"><strong>Zone ${name}</strong><span class="mono" id="count-${z}"></span><span id="owner-${z}"></span></div>`).join("");
const margin = 60, span = canvas.width - 2*margin;
const xy = p => [margin + (p[0]+data.arena)/(2*data.arena)*span, canvas.height-margin-(p[1]+data.arena)/(2*data.arena)*span];
const drawCircle = (p, radius, fill, stroke) => { ctx.beginPath(); ctx.arc(...p, radius, 0, 2*Math.PI); if (fill) { ctx.fillStyle=fill; ctx.fill(); } if (stroke) { ctx.strokeStyle=stroke; ctx.stroke(); } };
function draw() {
  slider.value = frame;
  byId("time").textContent = `${frame} / ${data.steps} · ${(frame*data.dt).toFixed(1)} s`;
  slider.setAttribute("aria-valuetext", `Step ${frame} of ${data.steps}`);
  byId("status").textContent = frames.done[frame] ? "Episode complete" : frame === 0 ? "Initial state" : "Episode in progress";
  const counts = frames.counts[frame], owners = frames.owners[frame];
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
  for (let i=0; i<6; i++) {
    const team=i<3?0:1, p=xy(frames.positions[frame][i]), target=xy(data.targets[i]);
    if (byId("trails").checked && frame>0) {
      ctx.strokeStyle=colors[team]+"55"; ctx.lineWidth=2; ctx.beginPath();
      for (let t=0; t<=frame; t++) { const v=xy(frames.positions[t][i]); if(t===0) ctx.moveTo(...v); else ctx.lineTo(...v); }
      ctx.stroke();
    }
    if (byId("routes").checked) {
      ctx.strokeStyle=colors[team]+"70"; ctx.lineWidth=1.5; ctx.setLineDash([5,6]); ctx.beginPath(); ctx.moveTo(...p); ctx.lineTo(...target); ctx.stroke(); ctx.setLineDash([]);
      ctx.beginPath(); ctx.moveTo(target[0]-4,target[1]); ctx.lineTo(target[0]+4,target[1]); ctx.moveTo(target[0],target[1]-4); ctx.lineTo(target[0],target[1]+4); ctx.stroke();
    }
    ctx.lineWidth=2; ctx.strokeStyle="white"; ctx.fillStyle=colors[team];
    if (team===0) drawCircle(p,13,colors[team],"white");
    else { ctx.beginPath(); ctx.moveTo(p[0],p[1]-16); ctx.lineTo(p[0]+16,p[1]); ctx.lineTo(p[0],p[1]+16); ctx.lineTo(p[0]-16,p[1]); ctx.closePath(); ctx.fill(); ctx.stroke(); }
    ctx.font="700 12px system-ui"; ctx.textAlign="center"; ctx.fillStyle="white"; ctx.fillText(String(i%3+1),p[0],p[1]+4);
  }
}
function setPlaying(value) { playing=value; lastTime=null; elapsed=0; byId("play").textContent=playing?"Pause":"Play"; }
byId("play").addEventListener("click",()=> { if(frame===data.steps) frame=0; setPlaying(!playing); draw(); });
byId("restart").addEventListener("click",()=> { frame=0; setPlaying(false); draw(); });
slider.addEventListener("input",()=> { frame=Number(slider.value); setPlaying(false); draw(); });
byId("routes").addEventListener("change",draw);
byId("trails").addEventListener("change",draw);
function tick(now) {
  if (playing) {
    if (lastTime!==null) elapsed+=Math.min(now-lastTime,250)*Number(byId("speed").value);
    lastTime=now;
    const duration=data.dt*1000;
    if (elapsed>=duration) { const advance=Math.floor(elapsed/duration); elapsed-=advance*duration; frame=Math.min(data.steps,frame+advance); if(frame===data.steps) setPlaying(false); draw(); }
  }
  requestAnimationFrame(tick);
}
draw();
requestAnimationFrame(tick);
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/tmp/spatial-blotto.html"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--scenario", choices=("cyclic", "balanced"), default="cyclic")
    parser.add_argument("--reward-mode", choices=("ownership", "zero_sum"), default="ownership")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if not 0 <= args.seed <= 2**32 - 1:
        parser.error("--seed must be between 0 and 4294967295")
    replay = simulate(args.seed, args.steps, args.scenario, args.reward_mode)
    # Escaping '<' also prevents a future metadata field from closing the script tag.
    payload = json.dumps(replay, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(HTML.replace("__REPLAY_DATA__", payload), encoding="utf-8")
    scores = replay["frames"]["scores"][-1]
    print(f"Saved {output}")
    print(f"Raw team scores: red={scores[0]:g}, blue={scores[1]:g}; scripted agents, no training.")


if __name__ == "__main__":
    main()
