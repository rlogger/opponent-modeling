"""Checks that reporting preserves physical reward bins and original planning RNG."""
import importlib.util
from pathlib import Path

import numpy as np


def module():
    path = Path(__file__).resolve().parents[1] / 'scripts/summarize_control_diagnostic.py'
    spec = importlib.util.spec_from_file_location('summary_probe', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_reward_bins_use_reward_bearing_next_state_and_ignore_padding():
    states = np.zeros((1, 5, 66))
    states[0, 1:, 2] = [1.8, 1.9, 2.1, 100.]
    trace = {'state': states, 'valid_mask': np.array([[1, 1, 1, 0]], bool),
             'blue_reward': np.array([[0., -.5, -1., 900.]])}
    result = module().error_bins([0., 0., 0.], trace)
    assert result['all']['count'] == 3
    assert result['inside_le_1.8']['mae'] == 0.
    assert result['edge_1.8_to_2']['mae'] == .5
    assert result['outside_gt_2']['mae'] == 1.


def test_saved_candidate_key_matches_original_planner_observer():
    import jax
    import jax.numpy as jnp

    from mopa import mppi
    from mopa.tdmpc import create_agent, load_config

    cfg = load_config(profile='gate4')
    cfg.update(opponent_mode='implicit', context_dim=0)
    cfg['encoder']['type'] = 'identity'
    cfg['world_model'].update(hidden_dim=16, predict_continues=True)
    agent = create_agent(cfg, 66, key=jax.random.PRNGKey(0),
                         obs_mean=np.zeros(66), obs_std=np.ones(66))
    _, key = jax.random.split(jax.random.PRNGKey(20261005 + 17))
    _, _, seen = mppi.plan(agent, jnp.zeros((1, 66)), 3, jnp.zeros((1, 0)),
                          deterministic=True, return_diagnostics=True, key=key)
    np.testing.assert_array_equal(module().candidate_key(17), seen['estimate_value_key'])
