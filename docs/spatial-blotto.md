# Spatial Blotto: game, mathematical controllers, and integration

A from-scratch JAX environment using the JaxMARL interface, alongside the
existing predator–prey task. This release supplies game mechanics, simple
mathematical controllers, complete match collection, validated transition
exports, and an interactive replay. It does not train policies or implement
opponent-strategy inference.

The precise configuration is **three zones, equal team budgets, symmetric and
homogeneous zone values, spatial movement, and indivisible units**. Continuous
movement does not make resource allocations continuous. This is a sequential
territorial-control extension of Blotto, not the static continuous-resource game.

## Run a match

From a checkout of `main`, with Python 3.11 or 3.12:

```bash
export UV_PROJECT_ENVIRONMENT=venv
uv sync --locked --all-extras

# Source allocation example: red (2,1,0), blue (0,2,1).
uv run --locked spatial-blotto --scenario cyclic \
  --output artifacts/blotto/example.html

# React to current enemy occupancy, versus balanced fixed coverage.
uv run --locked spatial-blotto --scenario reactive \
  --output artifacts/blotto/reactive.html \
  --json artifacts/blotto/reactive.json \
  --trajectory artifacts/blotto/reactive.npz

# Moving target allocations, with two units per team.
uv run --locked spatial-blotto --scenario rotating --team-size 2 --period 40 \
  --reward-mode zero_sum --output artifacts/blotto/rotating.html

# Choose each controller and allocation explicitly.
uv run --locked spatial-blotto --red-policy fixed --red-allocation 2,1,0 \
  --blue-policy reactive --steps 200 --seed 7 \
  --output artifacts/blotto/custom.html
```

`python -m spatial_blotto` and `python scripts/demo_spatial_blotto.py` expose the
same CLI. Open the generated HTML in a browser; it embeds its data and needs no
network assets. Output defaults to `artifacts/blotto/replay.html`. Generated
artifacts are ignored by Git. No checkpoints or training dataset are needed.

For backward compatibility with the initial demo, **scenario `cyclic` means a
fixed pair from the static allocation cycle**. Scenario `rotating` actually
uses the time-varying `cyclic` controller. The viewer names each controller and
shows its targets at the selected frame. `--help` lists all options, including
`--dt`, `--max-speed`, `--zone-radius`, and `--reward-mode`.

## Game rules

| Component | Rule |
| --- | --- |
| Teams | Red and blue, three units each by default; 2v2 also tested. Environment accepts positive team sizes; exact baseline controllers cap size at six. |
| Arena | Square `[-1.5,1.5]^2`, hard walls. |
| Zones | Three disjoint equal circles, radius 0.22, equilateral triangle of circumradius 0.75. A is top, B lower left, C lower right. |
| Action | Each unit supplies two velocity commands in `[-1,1]`; components are clipped, then vector length is capped at one. |
| Movement | All units move simultaneously, maximum speed 1, timestep 0.1. No inertia, collision, combat, elimination or obstacles. Units may overlap. |
| Ownership | Strict majority of units whose endpoint lies inside a zone. Boundary counts; empty or contested ties are neutral. |
| Persistence | Recomputed after every move. Leaving a zone can immediately lose it. Passing through without ending inside earns no point. |
| Default reward | Number of owned zones after movement, shared with every teammate. |
| Alternative reward | `zero_sum`: own owned zones minus opponent owned zones. |
| Reset | Red near `(-1.05,0)` with seeded jitter; blue is its horizontal mirror. |
| Horizon | 200 transitions by default. This is the finite game duration, not a collection cutoff. |
| Information | Full current physical positions/velocities, geometry, ownership, time and own teammate ID. No opponent targets, future actions or policy labels. |

Budgets are equal because team sizes are equal. Values are symmetric because
both teams receive the same value for any given zone; they are homogeneous
because all three zones have equal value. Symmetry holds under team exchange
and horizontal reflection, which also exchanges B and C. The map does not
promise arbitrary rotational symmetry of its square arena and starting positions.

The default reward is **not constant-sum**: tied zones award neither team a
point. Score difference is a different objective, not merely a rescaling of
ownership. Both modes retain raw ownership scores for inspection.

## Movement and scoring math

For each unit, first component-clip its action to obtain `u`, then move by:

```math
\hat u_{i,t}=\frac{u_{i,t}}{\max(1,\lVert u_{i,t}\rVert_2)},\qquad
p_{i,t+1}=\Pi_{[-L,L]^2}(p_{i,t}+\Delta t\,v_{\max}\hat u_{i,t}).
```

Reported velocity is realized displacement divided by `dt`, including wall
clipping. Zero action stops immediately. Counts and raw scores are:

```math
n_{q,k}(t)=\sum_{i\in q}\mathbf{1}\{\lVert p_i(t)-c_k\rVert_2\leq\rho\},
\qquad
s_R(t)=\sum_{k=1}^{3}\mathbf{1}\{n_{R,k}(t)\gt n_{B,k}(t)\}.
```

Blue's score swaps the teams. Per-step rewards use the post-move scores:

```text
ownership: r_R = s_R;       r_B = s_B
zero_sum:  r_R = s_R - s_B; r_B = s_B - s_R
```

For red `(2,1,0)` and blue `(0,2,1)`, raw scores are 1 and 2; zero-sum rewards
are -1 and +1. Travel changes cumulative returns: endpoint allocation scores
are not episode scores.

Agent-keyed rewards duplicate the team reward for each teammate. **Average
within a team, or select one representative, to recover the team reward.**
Summing teammates multiplies it by team size. Summing both teams in zero-sum
mode cancels the signal. The collector stores exactly one reward per team.
`team_scores` accumulates raw ownership as exact int32 integers; the horizon is
bounded by `2**24` so scores cannot overflow and successive elapsed-time inputs
remain distinguishable in float32.

## Mathematical controllers

All four controllers consume the same current physical state. They do not
access the other controller object, target assignments, hidden intentions,
future actions, or learned models.

| Controller | Allocation rule |
| --- | --- |
| `fixed` | Three nonnegative integer counts summing to team size. |
| `balanced` | Counts differ by at most one. Extra units favor A, then the team's starting side; 3v3 is `(1,1,1)` for both teams. |
| `cyclic` | Rotate counts every `period` steps. Default red allocation is `(n-1,1,0)` for `n>=2`; blue mirrors it and rotates oppositely. |
| `reactive` | Enumerate feasible assignments; maximize the static payoff against current opponent zone counts, then minimize travel distance. |

For each allocation, enumerate agent-to-zone assignments and minimize the sum
of Euclidean distances to zone centers. There are `3**n` candidates: 27 for
3v3, at most 729 under the explicit controller limit `n<=6`. This is a small
exact assignment calculation, not a neural policy. It does not solve makespan,
first-entry time, or accumulated game return. Deterministic ties use mirrored
zone order, preserving horizontal symmetry between the teams.

There are no obstacles, so the shortest route to a selected center is a line.
For displacement `d = center - position`, the controller commands:

```math
u=\frac{d}{\max(\lVert d\rVert_2,\Delta t\,v_{\max})}.
```

It travels at maximum speed when far away and reaches the center without
overshooting on the last step. The center is a deliberate destination;
minimum-distance entry into the scoring circle would be a different routing
objective. Reassignment can occur each step, especially for reactive control.

**Reactive is a heuristic for the dynamic game.** Its static calculation holds
current opponent counts fixed, including fewer than `n` occupied units when
opponents are in transit. It neither predicts their next actions nor accounts
for scoring during its own travel. Against changing opponents it can chase,
switch targets, or perform worse than a fixed allocation. That is appropriate
baseline behavior to measure, not an optimality claim.

Controller periods are positive integers no larger than `2147483647`. The demo
rejects unsupported team sizes before constructing the environment.

## Python integration

```python
import jax
from spatial_blotto import SpatialBlotto, make_controller, rollout_episode
from spatial_blotto.rollout import validate_episode

env = SpatialBlotto(team_size=3, max_steps=200, reward_mode="ownership")
red = make_controller(env, "red", "reactive")
blue = make_controller(env, "blue", "balanced")
run = jax.jit(lambda key: rollout_episode(env, red, blue, key))
episode = run(jax.random.PRNGKey(0))
validate_episode(env, episode)  # Host-side validation before artifact use.
print(episode.team_rewards.sum(axis=0))

# Vectorized independent matches, using exactly the same policy configuration.
keys = jax.random.split(jax.random.PRNGKey(10), 8)
batch = jax.jit(jax.vmap(run))(keys)
```

The team callable contract is `controller(state) -> (team_size,2)` in unit-ID
order. `joint_actions(env, red_actions, blue_actions)` adapts these arrays to
JaxMARL's agent dictionary. Both teams choose from the same pre-transition
state before movement. Optional `controller.target_zones(state)` supplies
viewer diagnostics; ordinary policies can omit it. Unknown targets are `-1`.

Instantiate this environment directly. Local classes are not automatically
registered in upstream `jaxmarl.make`, and no installed JaxMARL files are
modified. `reset`, `get_obs`, `step_env`, inherited `step`, `observation_space`,
`action_space`, and `agent_classes` follow the JaxMARL API. `state_space()`
describes the centralized vector. Generic `LogWrapper` is tested; MPE-specific
wrappers/trainers make additional assumptions and should not be used unchanged.

### Observations and centralized state

For `n` units per team and `N=2*n` total units, each observation has
`4*N+10+n` float32 channels: 37 for 3v3, 28 for 2v2.

1. Own position (2) and own realized velocity (2).
2. Other relative positions (`2*(N-1)`) and velocities (`2*(N-1)`).
3. Relative zone centers (6), own-perspective signed owners (3).
4. Elapsed fraction (1), own within-team one-hot ID (`n`).

Others are ordered as teammates by ID, then opponents by ID. Positions,
velocities and time are normalized; all observation channels lie in `[-1,1]`.
No policy names or target diagnostics enter observations. Centralized
`get_state` is positions, velocities and elapsed fraction (`4*N+1` channels;
25 for 3v3). Geometry is fixed environment configuration. Cumulative score is
logging state; it does not affect future rewards.

The environment validates static configuration and action keys/shapes/dtypes.
Finite actions are a precondition of compiled physics. Use
`env.validate_actions(actions)` for host-side rejection of NaN/Inf before a
custom controller enters JIT; this validator is intentionally not jittable.
The episode exporter also rejects nonfinite outputs. Invalid actions are not
silently converted into valid no-ops.

Physics uses float32. For numerical stability, `dt`, `max_speed`, and their
product must each lie in `[2**-126, 2**63]`. Radius must be at least `2**-126`;
its float32 value must be less than `sqrt(3)*0.375 - 8*float32.eps` (about
0.6495181). The small gap prevents rounding from putting a unit in two zones.
Stable Euclidean norms avoid squared-distance underflow. As with all float32
simulations, movements below local positional precision can round away; the
defaults avoid these extreme scales.

### Terminal and trajectory contracts

`step_env` returns the actual next state, which becomes absorbing after the
finite horizon. Later calls give zero reward and preserve it. Inherited `step`
automatically resets after the terminal transition: its returned state and
observation belong to the next episode, while rewards/dones belong to the old
one. `info['terminal_state']` and `info['terminal_observation']` preserve the
actual next inputs on every step, including termination.

The collector uses `step_env`. With `T=max_steps`, it records:

| Arrays | Shape/meaning |
| --- | --- |
| observations / world_states | `T+1` states, including the actual terminal state. |
| positions / velocities / team_scores | `T+1` physical/logging states. |
| actions | `(T,N,2)`, chosen at the corresponding pre-transition state. |
| team_rewards / ownership_scores | `(T,2)`, post-move reward/raw score counted once per team. |
| terminated / truncated | `(T,)`; only the last transition terminates, no administrative truncations. |
| zone_counts / zone_owners | `T+1` diagnostic states. |

For future transition learning, pair `observations[t]`, `actions[t]`,
`team_rewards[t]`, and `observations[t+1]`. This finite task has zero bootstrap
value after its terminal horizon. Future shorter collection quotas must be
represented separately as truncations; never use an autoreset observation as
the preceding episode's next-state target.

`save_trajectory(path, episode, metadata, env=env)` validates before writing
NPZ. It checks shapes, finite values, bounds, terminal flags, scores/rewards,
observations, and whether recorded actions reproduce next states. Load with
`np.load(path, allow_pickle=False)`. Targets are omitted from learning arrays;
policy configuration remains explicitly labeled metadata. HTML/JSON replay
frames have `T+1` entries, with zero reward in the initial display frame.
Replay/NPZ metadata contain seed, environment/controller configuration, package
versions, source hashes and the explicit statement that these are mathematical
baselines. In the viewer, overlapping unit markers are spread for visibility;
thin connectors mark their actual positions. Routes, trails, counts, and scores
always use the recorded physical positions.

## Code and verification map

| Module | Responsibility |
| --- | --- |
| `environment.py` | Game dynamics, rewards, spaces and terminal semantics. |
| `allocations.py` | Integer allocation enumeration and static payoffs. |
| `controllers.py` | Assignment selection and direct mathematical routing. |
| `rollout.py` | Simultaneous team adapter, JAX collection, validation and NPZ export. |
| `rendering.py` | Portable HTML view of recorded states and targets. |
| `cli.py`, `__main__.py`, demo script | Installed command, module invocation and compatibility script. |

```bash
uv run --locked ruff check .
uv run --locked pytest -q
uv run --locked pytest tests/test_spatial_blotto.py \
  tests/test_blotto_controllers.py tests/test_blotto_rollout.py \
  tests/test_blotto_rendering.py -q
```

The tests cover independent NumPy dynamics/scoring, every default-budget
allocation pair, observation structure/bounds, mirrored resets, team exchange,
unit permutation, walls, boundaries, overlaps, terminal/absorbing/autoreset
behavior, JIT/vmap/scan and `LogWrapper`. Controller tests use independent
exhaustive payoff/assignment oracles, routing convergence, mirrored tie-breaking,
reward-mode differences and switching schedules. Integration tests reconstruct
transitions, test vectorized matches, reject corrupt exports and run the CLI
outside the repository root through the installed package. Viewer tests execute
its JavaScript with Node when available and cover changing targets, playback,
script-safe JSON and dynamic team sizes. CI also runs a complete Blotto export
smoke in addition to the existing predator–prey smoke.

## Research boundaries and next integration

The allocations `(2,1,0)`, `(0,2,1)`, `(1,0,2)` contain a two-zone win cycle.
That does not establish inevitable learning cycles or absence of equilibrium.
Independent enumeration confirms that the static three-unit, three-zone game
has a pure equilibrium `(1,1,1)` versus itself under score-difference reward.
Under raw ownership with neutral ties, `(3,0,0)` versus `(1,1,1)` is a pure
Nash equilibrium with scores 1 and 2; zone permutations and team swaps yield
six. These statements concern the static allocation game, not a solution of
the sequential spatial game.

Next work is to choose and implement the RL control architecture: centralized
team actions or a shared per-unit policy, with consistent observation and reward
aggregation. The current predator–prey MAPPO trainer assumes different roles,
observations and actions; it is not a drop-in Blotto trainer. World models would
need Blotto's state and joint-action schema rather than predator–prey features.

Only after that baseline should strategy/intent models, fixed and held-out
opponent populations, switching opponents, latent-conditioned planning and
co-training be evaluated. Full physical observations may reveal movement
quickly. No particular required history length, RL benefit, strategy recovery,
robustness, or adaptive learning cycle is claimed by these mathematical demos.
