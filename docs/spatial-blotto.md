# Spatial Blotto v0

A from-scratch JAX environment implementing the JaxMARL interface. It lives in
`src/spatial_blotto/` alongside the existing predator–prey environment.

The project proposal is [NEXT:09 in the private research archive](https://github.com/rlogger/marl-private/blob/main/sources/mbct-next-steps-2026-09-03/slide-09.md):
three zones in a triangle, 2v2 or 3v3 teams, per-step majority ownership, and an
opponent-modeling hypothesis. This implementation defaults to the illustrated
3v3 case. Movement, neutral ties, observations, reset geometry and horizon below
are **v0 design choices**, not additional claims from the slide.

## Rules before algorithms

| Component | v0 rule |
| --- | --- |
| Units | Three red and three blue; `team_size=2` also supported |
| Arena | Square `[-1.5, 1.5]^2`, hard walls |
| Zones | Three disjoint radius-0.22 circles in an equilateral triangle |
| Action | Two continuous velocity commands per unit in `[-1,1]` |
| Movement | All units move simultaneously; speed capped at 1, `dt=0.1` |
| Interaction | Units can overlap; no collisions, combat, elimination or momentum |
| Ownership | Strictly more units inside a zone; ties and empty zones are neutral |
| Persistence | Recomputed every step; leaving a zone can lose ownership immediately |
| Reward | One point per owned zone after movement, shared with every teammate |
| Reset | Red starts near `(-1.05,0)`; blue is its mirror image; seeded jitter |
| Horizon | 200 transitions, configurable; this is the finite task duration |
| Information | All physical positions/velocities, zone ownership, time and own unit ID |

The mirrored reset makes the teams geometrically equivalent after exchanging
the left/right zones. Units are points for scoring; touching a zone boundary
counts. An agent can count in at most one zone. Scoring uses endpoint positions,
so crossing through a zone without ending inside does not earn points.

## Mathematics and implementation

For unit position `p`, first clip the action components to obtain `u`, then cap
its Euclidean length. The arena projection clips each position component:

```math
\hat u_{i,t} = \frac{u_{i,t}}{\max(1,\lVert u_{i,t}\rVert_2)},\qquad
p_{i,t+1} = \Pi_{[-L,L]^2}\left(p_{i,t}+\Delta t\,v_{\max}\hat u_{i,t}\right).
```

The reported velocity is actual displacement divided by `dt`, including wall
clipping. With direct velocity control, zero action stops immediately. This is
deliberately simpler than MPE acceleration and collision dynamics.

Counts in zone `k`, with center `c_k` and radius `rho`, are:

```math
n_{q,k}(t)=\sum_{i\in q}\mathbf{1}\{\lVert p_i(t)-c_k\rVert_2\leq\rho\},
\qquad q\in\{R,B\}.
```

The raw score for red is the number of zones where its count exceeds blue's;
blue's definition is symmetric:

```math
s_R(t)=\sum_{k=1}^{3}\mathbf{1}\{n_{R,k}(t)>n_{B,k}(t)\}.
```

With default `reward_mode="ownership"`, every red unit receives `s_R(t+1)` and
every blue unit receives `s_B(t+1)`. For the slide's red `(2,1,0)` against blue
`(0,2,1)`, the rewards are exactly **1 and 2**. These are shared team rewards:
averaging across teammates recovers the team score; summing multiplies it by
team size. State `team_scores` accumulates raw scores once per team.

Optional `reward_mode="zero_sum"` gives red `s_R-s_B` and blue `s_B-s_R`, so the
same example gives **−1 and +1**. This changes the optimization objective:
neutral zones make total raw ownership variable. Raw ownership and difference
rewards are not interchangeable experiments. Both modes expose raw scores.

## Run it

From the repository root, use a Python environment with this project's `train`
and `dev` extras installed (`jaxmarl==0.1.0`, JAX 0.4.38 in the current lock).
No new dependencies or upstream JaxMARL source edits are required. A fresh
environment can install these with `python -m pip install -e '.[train,dev]'`.

```python
import jax
from spatial_blotto import SpatialBlotto

env = SpatialBlotto(team_size=3, max_steps=200)
key_reset, key_action, key_step = jax.random.split(jax.random.PRNGKey(0), 3)
obs, state = env.reset(key_reset)
keys = jax.random.split(key_action, env.num_agents)
actions = {
    agent: env.action_space(agent).sample(keys[i])
    for i, agent in enumerate(env.agents)
}
obs, state, rewards, dones, info = env.step(key_step, state, actions)
```

Instantiate `SpatialBlotto` directly: `jaxmarl.make` does not register local
environment classes automatically. The public dictionaries follow the
[JaxMARL parallel API](https://jaxmarl.foersterlab.com/), with keys `red_0` through
`red_2`, `blue_0` through `blue_2`, and `dones["__all__"]`.

```sh
PYTHONPATH=src python scripts/demo_spatial_blotto.py --output /tmp/spatial-blotto.html
PYTHONPATH=src python -m pytest tests/test_spatial_blotto.py -q
```

On the current workstation, these commands were verified with
`/Users/rajdeepsingh/venvs/marl/bin/python` (JAX 0.4.38, JaxMARL 0.1.0).
All 21 environment checks pass, including `jit`, `vmap`, `lax.scan`, the source
allocation example, terminal transitions and generic `LogWrapper` compatibility.

The HTML replay is a deterministic scripted-policy demonstration, not a trained
agent result. Its controls let you inspect movement, counts and scores.

### Observations and transitions

Each observation has `4*N + 10 + n` values: **37 for 3v3**, **28 for 2v2**.
Here `N=2*n` is total units. All channels are normalized to `[-1,1]`:

1. Own position (2), own velocity (2).
2. Relative positions of other units (`2*(N-1)`), then their velocities (`2*(N-1)`).
3. Relative zone centers (6), own-perspective signed ownership (3).
4. Elapsed fraction of horizon (1), own within-team ID as a one-hot vector (`n`).

Other units are ordered as teammates by ID, then opponents by ID. Velocities
are physical, already observed velocities. Opponents' target zones, future
actions and policy identities are never observation channels. An agent sees
the full current physical state, but another controller's intentions may remain
unknown. This is not yet a local-sensing or occlusion task.

`env.get_state(state)` supplies a normalized physical Markov vector of length
`4*N+1` (**25 for 3v3**) for a centralized critic or world model. Geometry is
fixed environment configuration. Accumulated score is only a logging variable;
it does not affect subsequent dynamics or rewards.

`step_env` returns the actual next state. Inherited `step` automatically resets
after the horizon and returns **reset observations/state together with terminal
rewards/dones**. Use `info["terminal_observation"]` and
`info["terminal_state"]` to recover the actual next inputs on that transition.
These fields exist on every step. The horizon is task termination, with time
included in observations; `terminated=True`, `truncated=False` at the last step.
Further `step_env` calls on a finished state leave it unchanged and give zero
reward. Do not join the next episode to a world-model training target.

## What to build next, in order

1. **Inspect scripted behavior.** Reproduce the slide, uniform allocations,
   relocation and ties. Check episode returns as well as endpoint ownership:
   travel time changes the dynamic game.
2. **Add team control and a PPO baseline.** Choose a centralized controller
   producing `n*2` actions, or one shared policy per team producing 2 per unit.
   Keep this choice fixed across baselines. Adapt the training interface before
   running learning experiments.
3. **Build a fixed opponent population.** Include fixed allocations and
   reactive reallocators; separate training policies from held-out opponents.
   Record physical observations, joint actions, counts and rewards with seeds.
4. **Test whether history helps.** Compare a current-state policy, a history
   policy, an explicit intent model, and a target-aware oracle under matched
   interaction/training budgets. The oracle is an evaluation upper comparator;
   its targets must not enter ordinary policy inputs. Full-state observation
   can reveal direction quickly, so a 10–15-step inference requirement is not
   established by these rules.
5. **Integrate world models and planning.** Give the model all six units and
   their actions; train dynamics separately from discontinuous ownership
   rewards. Validate multi-step prediction and closed-loop returns before
   adding co-training, partial observations, inertia or more complex capture.

The existing `scripts/train_mappo.py` assumes predator/prey roles and maps 2D
commands to MPE's 5D action encoding. The current TD-MPC pipeline also assumes
the predator–prey state schema. Neither is a drop-in trainer for this module.
Generic JaxMARL `LogWrapper` is suitable; `MPELogWrapper` applies an extra reward
scale, and some centralized rollout helpers assume discrete actions or sum all
agents' rewards. In zero-sum mode, summing both teams cancels the signal.

## The strategic claim still needs testing

The allocations `(2,1,0)`, `(0,2,1)`, `(1,0,2)` form a cycle of strict two-zone
wins. That establishes a cycle among those strategies, not the absence of an
equilibrium or inevitable cycling of a learning algorithm.

As a check, the static game with three indivisible units and three equal zones
has only ten allocations. Under score-difference rewards, `(1,1,1)` against
itself is a pure Nash equilibrium. Under raw ownership with neutral ties,
`(3,0,0)` against `(1,1,1)` is a pure equilibrium with scores 1 and 2: neither
side can increase its own score by changing allocation. Zone permutations and
team swaps give six such pure equilibria. These are statements about the static
allocation game, not a solution of this sequential spatial environment.

The environment is a testbed for the project's opponent-modeling hypothesis.
A passing environment test or an appealing replay is not evidence of a learned
advantage, successful intent inference, or co-training cycles.
