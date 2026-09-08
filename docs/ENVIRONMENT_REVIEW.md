# Environment and visualization review

Reviewed 2026-09-08. Original reference: [`rlogger/marl-opp-aware` at aecbab5](https://github.com/rlogger/marl-opp-aware/tree/aecbab5daf6da402029e953be114a52e62a46c26).
Its remote HEAD and local checkout HEAD both resolve to `aecbab5daf6da402029e953be114a52e62a46c26`.
The local environment, demo, gallery, environment-card, and trajectory-panel sources were inspected alongside this repository's `tag_objectives` implementation.
This pass improves presentation only. Dynamics, rewards, resets, checkpoints, and experimental results are unchanged; no training or evaluation experiment was rerun.
See [visualization usage and outputs](VISUALIZATION.md).

## Rendering must follow the environment

| Environment contract | Rendering consequence |
|---|---|
| Predator/prey radii are `0.075`/`0.05`; capture is center distance below their sum. | Read actual radii from the environment; show capture from recorded terminal information. |
| Nominal arena half-width is `2.0`; prey bounds shaping begins at `1.8`. There is no hard position wall. | Mark the nominal arena as a reference; preserve or explicitly flag out-of-arena paths. |
| Lava contact uses agent-center distance below the stored disc radius. Lava has no collision force. | Draw exact disc geometry; do not imply an obstacle or add agent radius to the contact test. |
| Defaults penalize lava only for the risk predator (`-100`); other predators and prey have zero lava penalty. | Describe the region as lava, not universal damage or a lethal zone. |
| Collection occurs after a transition when prey-center/resource-center distance is below `0.15`. | Use recorded `collected` changes for collection events; do not infer collection from marker overlap. |
| Position advances using existing velocity; the action changes velocity. | Distinguish action-command arrows from velocity arrows. |
| Capture ends an episode; timeout is a separate terminal condition. | Stop recorded trajectories at their valid end and display the correct event. |

Contracts: [resource collection](../src/tag_objectives/resources.py#L150), [lava contact](../src/tag_objectives/objectives.py#L258), [capture and rewards](../src/tag_objectives/objectives.py#L376), [termination](../src/tag_objectives/objectives.py#L430).
Actual environment attributes were inspected under the existing JaxMARL runtime; rendering does not change their values.
Learned world-model paths must retain a prediction label. A predicted capture, resource value, or rollout after capture is not a simulator event.

## Minimal features retained from the original

The useful design in [objectives_demo.py](https://github.com/rlogger/marl-opp-aware/blob/aecbab5daf6da402029e953be114a52e62a46c26/src/objectives_demo.py) is a shared frame function: translucent lava and outlines, resource status, agent trails/discs, and a small step/event annotation.
Use consistent agent colors across strategy panels and identical reset scenes for comparisons. Optional observation links can clarify the nearest-resource/lava inputs.
One shared Python renderer can serve static previews, replay animations, and learned-trajectory plots; a dashboard framework is unnecessary.

Old rendering choices that should not be copied:

- Lines 160–161 and 260–261 draw the prey with radius `0.055`, rather than the environment's `0.05`.
- Lines 158–159 reveal entire future paths during animation; lines 188–191 and 269–272 reveal eventual outcomes before those events occur. Replay should display information through the current frame.
- [Trajectory panels, lines 63–74](https://github.com/rlogger/marl-opp-aware/blob/aecbab5daf6da402029e953be114a52e62a46c26/src/objectives_traj_panels.py#L63) place many unrelated-reset paths over the first episode's lava. Such a backdrop cannot demonstrate hazard avoidance.
- [Gallery selection, lines 43–52](https://github.com/rlogger/marl-opp-aware/blob/aecbab5daf6da402029e953be114a52e62a46c26/src/objectives_gallery.py#L43) can label a fallback median episode “a capture” when none were captured. Labels must follow the selected episode's events.
- A fixed `±2.15` viewport can hide valid recorded paths. The audited dataset reaches `|agent coordinate| = 11.450`; lava discs extend to `2.388`.

## Replay-camera regression corrected

The first TD-MPC replay used the entire episode's furthest position to choose
its initial zoom. Blue later reached `x = 8.160448`, so a nominal ±2 arena looked
tiny inside roughly ±8.4 axes even at step zero. This was a presentation regression.
The earlier gallery used roughly ±2.7; the original demo used a fixed ±2.15.

The corrected replay defaults to fixed ±2.5 axes, with a clearer dashed arena
reference and labeled off-screen pointers showing true agent coordinates.
`--replay-camera full` retains the old diagnostic overview. No positions are
clamped, and no walls, rewards, physics, resets or policies were changed.

All five core environment modules match HEAD and the earlier experiment hashes.
The capture replay's initial 66D state exactly matches dataset episode 400.
The other important difference is the **controller**: the old gallery used a
fixed MAPPO prey; the new replay uses the saved TD-MPC prey. Restoring the camera
does not change that controller's poor out-of-arena behavior.

## Inherited environment issues: documented, not changed

These observations come from the existing dataset, not newly generated resets:

`artifacts/continuous/dataset.npz` — 1,800 episodes; SHA-256:

`928181027a9e5e86b106190b1487cda90c09e2cc5df08b569f7b93ba8350b5f5`

| Issue | Current code and verified observation |
|---|---|
| Minimum spawn distance is applied before jitter. | [objectives.py:324–332](../src/tag_objectives/objectives.py#L324): 123/1,800 initial center separations are below `2.2`; minimum `2.0835`. |
| Lava clearance has an unchecked fallback. | [objectives.py:287–299](../src/tag_objectives/objectives.py#L287): when all four candidates fail, `argmax` selects candidate zero. 57/1,800 episodes have initial clearance below the intended `0.35`; **21/1,800 actually begin with an agent center inside lava**. Minimum signed edge clearance is `-0.3705`. |
| Fixed nearest-resource slots can contain collected entries. | [objectives.py:275–278](../src/tag_objectives/objectives.py#L275): collected entries are ranked last, but fixed-size top-k still returns them when too few resources remain; there is no availability mask in this observation. |

The first two counts measure episodes, not independent layouts: the three strategy sets share reset scenes.
Distance uses initial predator/prey centers. Lava clearance is the minimum, across both agents and all discs, of center distance minus disc radius; clearance below `0.35` does not itself mean lava contact.
These issues require a separate environment decision because fixing them changes data generation and checkpoint comparability. Visualization must not conceal them or silently repair recorded states.
