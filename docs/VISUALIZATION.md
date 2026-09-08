# Visualization

One shared [Matplotlib renderer](../src/tag_objectives/rendering.py), one [replay script](../scripts/render_environment.py). No dashboard, new dependencies, or changes to environment dynamics.

## View the environment

[Recorded replay and snapshots](../experiments/environment/README.md) compare capture, risk, and curious opponents from identical initial states. Colors identify roles, not policy-visible strategy labels. Trails contain only the past; captured panels freeze at their actual terminal frame. Agent and lava sizes follow the environment.

```bash
uv run --locked --extra train --extra plot python scripts/render_environment.py
```

Defaults select checkpoint 2, group episode 0, without filtering for success. `--episode`, `--checkpoint`, and `--steps` change the selection. This reads saved data; it does not train or generate trajectories. The output manifest records dataset/source/output hashes and matched-reset checks.

For an existing simulator state:

```python
from tag_objectives.rendering import ArenaRenderer, frame_from_state

view = ArenaRenderer(ax, env, source="simulator", observations=False)
view.draw(frame_from_state(env, state))
```

Pass `history` as positions through the current frame, shaped `(time, agents, 2)`. Use `frame_from_markov` for recorded or predicted state vectors. `source="model"` explicitly labels predictions and suppresses observed-score/capture claims. The default camera stays at arena scale; edge pointers flag off-screen agents with actual coordinates. An explicit shared `limit` can cover a full-trajectory comparison. The nominal ±2 arena is not a physical wall. Unsupported landmark geometry is rejected, not silently hidden.

Saved TD-MPC replays use the same fixed arena-scale default. `--replay-camera full`
keeps the older zoomed-out diagnostic view. This avoids shrinking the entire game
at frame zero because an agent escapes late in the episode. See [the evaluation command](TD_MPC_0S.md).

## Useful techniques

| Question | Visualization | Status / required evidence |
|---|---|---|
| What is happening in the game? | Replay, fading trails, velocity direction ticks, collection counts, current terminal event | Added; recorded GIF and PNGs |
| How do objectives change behavior? | Synchronized matched-reset panels | Added; same reset and prey checkpoint, not necessarily identical blue actions |
| What can each agent observe? | Nearest available resource/lava links | Added debug view; not an exact observation-slot dump or new policy input |
| Are the three `0s` representations separable? | PCA scatter, prototype-coordinate heatmap, held-out probe/ARI | [Existing gallery](../experiments/continuous_0s/visuals/README.md); projection appearance alone is not evidence |
| Can strategy be inferred online? | Prefix-time probe curves; later, switch timelines and belief entropy/calibration | Prefix curves exist; switch/belief plots need causal sequential evaluation |
| Does the latent improve action prediction? | Matched-specialist error matrix, action arrows, context controls | Existing; zero context is not a separately trained vanilla BC baseline |
| Does changing z control predicted behavior? | Same-state/action latent swaps, additional-scene small multiples | Existing qualitative forecasts; three-way controllability is not established |
| Is learned physics accurate? | Truth/model overlay, residual vectors, error versus horizon | Horizon errors exist; overlays are a next addition using recorded joint actions |
| Is behavior consistently different? | Occupancy/novelty heatmaps; distance-to-prey and signed-lava-clearance traces | Next; use matched maps or normalized geometry, never unrelated paths over one lava map |
| Does performance improve? | Capture rate, survival curves, resources, lava exposure, coverage; seed learning curves | Next evaluation; paired resets and seed-level uncertainty, not selected successful clips |
| Is uncertainty useful? | Ensemble trajectory fans, error-versus-uncertainty and calibration plots | Needs held-out predictions and ensemble outputs |
| What does TD-MPC plan? | Candidate rollouts, elites, chosen action, predicted versus realized return | Later; requires planner logging, not part of this visual-only pass |
| Is latent behavior smooth? | Prototype interpolation and counterfactual action fields | Later; hold state/blue actions fixed and validate against specialists; interpolation need not be on-distribution |

Start with the replay, observation view, and matched panels. For research, prioritize action-prediction controls, rollout error, and multi-seed task metrics before adding more plots. Do not compare raw rewards across different objectives. Fit projections/prototypes on training data and preserve episode/checkpoint splits; report causal-prefix and full-episode results separately.

The [environment review](ENVIRONMENT_REVIEW.md) records the original repository comparison and inherited reset issues. No experiment was rerun for these presentation changes.
