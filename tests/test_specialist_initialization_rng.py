"""Execute the real nested initializer without a costly specialist training run."""
import ast
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np


def test_specialist_initializer_returns_unconsumed_stream_without_changing_parameter_keys():
    path = Path(__file__).resolve().parents[1] / "scripts/train_mappo.py"
    tree = ast.parse(path.read_text())
    function = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "create_train_states")
    actor = SimpleNamespace(init=lambda key, dummy: {"initialization_key": np.asarray(jax.random.key_data(key))}, apply=None)
    scope = dict(jax=jax, jnp=np, team_names=["pred", "prey"], max_obs=35, world_state_dim=52,
                 actors={"pred": actor, "prey": actor}, critics={"pred": actor, "prey": actor},
                 config={"ANNEAL_LR": False, "MAX_GRAD_NORM": .5, "LR": .001},
                 optax=SimpleNamespace(chain=lambda *a: None, clip_by_global_norm=lambda *a: None, adam=lambda *a, **kw: None),
                 TrainState=SimpleNamespace(create=lambda **kw: kw["params"]))
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, str(path), "exec"), scope)
    with jax.debug_key_reuse(True):
        states, remaining = scope["create_train_states"](jax.random.key(11))
        downstream, reset = jax.random.split(remaining)
        jax.random.split(reset, 2)
        jax.random.split(downstream, 2)
    # Captured keys are the actual actor/critic init inputs from the source.
    actual = [tuple(v.tolist()) for v in jax.tree.leaves(states)]
    assert len(actual) == 4 and len(set(actual)) == 4
    assert tuple(np.asarray(jax.random.key_data(remaining)).tolist()) not in actual
    # Parameter initialization must retain the historical sequence of split calls.
    historical = []
    key = jax.random.PRNGKey(11)
    for _ in range(2):
        key, actor_key, critic_key = jax.random.split(key, 3)
        historical.extend([tuple(actor_key.tolist()), tuple(critic_key.tolist())])
    assert set(actual) == set(historical)
    assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                   and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                   and node.value.func.id == "create_train_states"]
    assert len(assignments) == 1
    assert isinstance(assignments[0].targets[0], ast.Tuple)
    assert [n.id for n in assignments[0].targets[0].elts] == ["train_states", "rng"]
