# Objective-typed predator environment

`SimpleTagObjectivesMPE` is a research instrument for latent strategy inference:
a predator's hidden objective shapes its behaviour, a prey must infer and
exploit it, and terrain (lava plus resources) makes the objectives observable.

The environment source is restored from `rlogger/marl-opp-aware@aecbab5`;
[provenance and parity tests](../third_party/marl-opp-aware/UPSTREAM.md) document
the package/API adaptations. Continuous actions remain supported.

## Task summary

P predators (default 1) vs 1 prey with a nominal `[-2, 2]^2` arena (soft bounds shaping, not walls). 16 resources (half
attached to lava discs), 3 lava discs (radius 0.35–0.60), no obstacles.
The prey collects resources (+5 each); predators cannot see resources. Both
sides observe the 3 nearest lava discs `(dx, dy, radius)`; the prey also sees
10 resource slots `(dx, dy)`, prioritizing uncollected resources. When fewer
remain, collected entries fill the remaining slots.

See [visualization](VISUALIZATION.md) for replay and [environment review](ENVIRONMENT_REVIEW.md) for inherited reset issues.

At the current defaults, lava is -100/step for the risk-averse predator and 0 for
every other agent (`lava_penalty=100`, `base_lava_penalty=0`,
`prey_lava_penalty=0`). The episode ends at the first capture (any predator)
or at 100 steps; survival time to first capture is a primary behavior metric.

## Extension point 1: new predator objectives

```python
from tag_objectives import ObjectiveSpec, register_objective

register_objective("ambusher", lambda env: ObjectiveSpec(
    capture=env.capture_bonus, lava=-env.base_lava_penalty, still=0.05))

def guard_reward(env, ctx):
    import jax.numpy as jnp
    d_res = jnp.linalg.norm(
        ctx["new_state"].resource_pos.mean(0) - ctx["pred_pos"], axis=-1)
    return env.capture_bonus * ctx["capture_f"] - 0.5 * d_res

register_objective("guard", lambda env: guard_reward)
```

`ctx` channels: `capture_f`, `dist`, `in_lava`, `new_cell`, `pred_pos`,
`prey_pos`, `prey_lava`, `state`, `new_state`.

| name | reward per step (current defaults) |
|------|-------------------------------|
| `capture` | `-dist_i` |
| `risk` | `+10*capture - 100*in_lava_i - dense_chase_coef*dist_i` |
| `curious` | `+10*capture + 0.5*new_cell_i - dense_chase_coef*dist_i` |

Prey reward is fixed: `+5/resource, -10 if captured, arena-bounds shaping`.

## Extension point 2: team size and mixed teams

`num_adversaries=P` enables multi-predator teams. Capture is any-predator
(shared bonus); `dist` / `lava` / `novelty` are per-predator; novelty is
team-shared. Mixed types:

```python
SimpleTagObjectivesMPE(
    pred_type=("capture", "risk", "curious"),
    num_adversaries=3,
)
```

## Extension point 3: evaluate any policy

```python
import jax
from tag_objectives import make_env, evaluate_policy, random_policy

env = make_env("risk")
metrics = evaluate_policy(env, random_policy(env), n_eps=300, key=jax.random.PRNGKey(0))
print(metrics.means())
```

## Reliability

- Golden reward tests pin reward math with explicit v3-era constants.
- Contract tests cover obs shapes, capture termination, lava/novelty, determinism,
  multi-predator / mixed teams, and `evaluate_policy`.
- `reset` / `step_env` are jit-compiled and vmap-safe; observations are fixed-shape.
