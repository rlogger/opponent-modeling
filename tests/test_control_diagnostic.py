"""Exact checkpoint continuation and prevention of protocol/data contamination."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture(scope='module')
def driver():
    path = Path(__file__).resolve().parents[1] / 'scripts/run_control_diagnostic.py'
    spec = importlib.util.spec_from_file_location('control_diagnostic_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def protocol():
    return {'status': 'frozen', 'fitting_seeds': [11, 22], 'arms': ['implicit', 'bc', '0s'],
            'initial_optimizer_step': 128,
            'same_replay_stage': {'additional_updates': 1872, 'total_updates': 2000, 'new_real_transitions': 0},
            'online_stage': {'rounds': 6, 'additional_updates_per_round': 1000,
                             'valid_transitions_per_type_per_family_per_round': 600,
                             'training_checkpoint_families': [0], 'total_new_transitions_per_controller': 10800,
                             'total_updates_at_end': 8000, 'collection_batch_episodes': 1,
                             'collection_round_indices': [1, 2, 3, 4, 5, 6]},
            'evaluation': {'checkpoint_family': 1, 'model_steps': [128, 2000, 8000],
                           'modes': ['MPPI', 'policy-only'], 'objectives': ['capture', 'risk', 'curious'],
                           'episodes_per_fit_per_type_per_mode_per_checkpoint': 8, 'fresh_key_base': 900001},
            'initial_artifacts': [{'seed': s, 'arm': a} for s in (11, 22) for a in ('implicit', 'bc', '0s')],
            'fixed': {'horizon': 3, 'population_size': 512, 'mppi_iterations': 6,
                      'policy_prior_samples': 24, 'num_elites': 64, 'discount': .99,
                      'termination_contract': 'finite_100_step_v1'}}


@pytest.mark.parametrize('fault', ['draft', 'duplicate_artifact', 'heldout_training', 'stage_budget',
                                 'old_training_reset', 'evaluation_after_selection', 'horizon', 'fresh_base'])
def test_protocol_rejects_changed_comparisons_and_leakage(driver, fault):
    p = protocol()
    assert driver.protocol_configuration(p)['training_checkpoints'] == [0]
    if fault == 'draft':
        p['status'] = 'draft'
    elif fault == 'duplicate_artifact':
        p['initial_artifacts'][-1] = p['initial_artifacts'][0]
    elif fault == 'heldout_training':
        p['online_stage']['training_checkpoint_families'] = [0, 1]
    elif fault == 'stage_budget':
        p['same_replay_stage']['additional_updates'] = 2000
    elif fault == 'old_training_reset':
        p['online_stage']['collection_round_indices'][0] = 0
    elif fault == 'evaluation_after_selection':
        p['evaluation']['model_steps'] = [128, 8000]
    elif fault == 'horizon':
        p['fixed']['horizon'] = 10
    else:
        p['evaluation']['fresh_key_base'] = 900000
    with pytest.raises(ValueError):
        driver.protocol_configuration(p)


@pytest.mark.parametrize('bad', [[1], [1., 2.], [-1, 2], [1, 2**40]])
def test_restore_rng_rejects_corrupt_update_key(driver, bad):
    manifest = {'replay_rng': np.random.default_rng(12).bit_generator.state, 'update_key': bad}
    with pytest.raises(ValueError):
        driver.restore_rng(manifest)


def test_complete_reset_identity_catches_different_key_same_state(driver, monkeypatch):
    monkeypatch.setattr(driver, 'fingerprints', lambda env, keys: ['same-state'] * len(keys))
    keys, states = set(), set()
    driver.register_resets(None, np.array([[1, 2]], np.uint32), keys, states)
    with pytest.raises(ValueError, match='complete initial state'):
        driver.register_resets(None, np.array([[3, 4]], np.uint32), keys, states)
    assert keys == {(1, 2)} and states == {'same-state'}


def test_checkpoint_fork_preserves_genuine_optimizer_replay_and_jax_rng(driver, tmp_path):
    import flax.serialization as serialization
    import jax

    from mopa.tdmpc import create_agent, load_config
    from mopa.tdmpc_data import SequenceReplay

    cfg = load_config(profile='smoke')
    cfg['encoder']['type'] = 'identity'
    cfg['world_model'].update(hidden_dim=8, predict_continues=True, num_value_nets=2, num_bins=11)
    cfg['tdmpc2'].update(batch_size=2, horizon=1, continue_loss_scale=1.)
    base = create_agent(cfg, 66, key=jax.random.PRNGKey(19), obs_mean=np.zeros(66), obs_std=np.ones(66))
    rng = np.random.default_rng(23)
    state = rng.normal(size=(3, 5, 66)).astype(np.float32)
    data = {'state': state, 'blue_action': np.zeros((3, 4, 2), np.float32),
            'red_action': np.zeros((3, 4, 2), np.float32), 'blue_reward': np.array([[0, 5, 0, -10]] * 3, np.float32),
            'terminated_capture': np.array([[False, False, False, True]] * 3),
            'truncated_timeout': np.zeros((3, 4), bool), 'valid_length': np.full(3, 4)}
    replay = SequenceReplay.from_dataset(data, np.arange(3), 1, np.zeros((3, 5, 0), np.float32))
    initial_rng = np.random.default_rng(41)
    one, key, _ = driver.advance(base, replay, initial_rng, jax.random.PRNGKey(7), 1, tmp_path / 'first.jsonl')
    saved = driver.save_state(one, initial_rng, key, tmp_path, 'fork')
    restored = serialization.from_bytes(base, (tmp_path / saved['checkpoint']).read_bytes())
    driver.assert_state_equal(serialization.to_state_dict(restored), serialization.to_state_dict(one))
    resumed_rng, resumed_key = driver.restore_rng(saved)
    uninterrupted, final_key, _ = driver.advance(one, replay, initial_rng, key, 3, tmp_path / 'uninterrupted.jsonl')
    resumed, resume_final_key, history = driver.advance(restored, replay, resumed_rng, resumed_key, 3, tmp_path / 'resumed.jsonl')
    driver.assert_state_equal(serialization.to_state_dict(resumed), serialization.to_state_dict(uninterrupted))
    assert resumed_rng.bit_generator.state == initial_rng.bit_generator.state
    np.testing.assert_array_equal(final_key, resume_final_key)
    assert [r['step'] for r in history] == [2, 3]
    assert all(r['world_gradients_finite'] for r in history)
    assert not np.array_equal(np.asarray(resumed.model.reward_model.params['layers_2']['kernel']),
                              np.asarray(one.model.reward_model.params['layers_2']['kernel']))
    with pytest.raises(FileExistsError):
        driver.save_state(resumed, resumed_rng, resume_final_key, tmp_path, 'fork')
    with pytest.raises(ValueError, match='increasing'):
        driver.advance(resumed, replay, resumed_rng, resume_final_key, 3, tmp_path / 'invalid.jsonl')
    assert not (tmp_path / 'invalid.jsonl').exists()


def test_invalid_gradient_cannot_be_logged_as_completed_update(driver, tmp_path):
    import jax

    states = {name: SimpleNamespace(step=0) for name in driver.TRAIN_STATES}
    agent = SimpleNamespace(model=SimpleNamespace(**states), batch_size=1,
                            update=lambda **kwargs: (None, {'world_gradients_finite': False, 'policy_gradients_finite': True}))
    replay = SimpleNamespace(sample=lambda rng, count: {})
    with pytest.raises(FloatingPointError):
        driver.advance(agent, replay, np.random.default_rng(0), jax.random.PRNGKey(0), 1, tmp_path / 'failed.jsonl')
    assert (tmp_path / 'failed.jsonl').read_text() == ''


def test_old_and_fresh_evaluation_keys_are_distinct_and_repeatable(driver):
    for seed in (11, 22):
        old = driver.evaluation_keys(seed, False)
        fresh = driver.evaluation_keys(seed, True)
        assert not set(map(tuple, old[0])) & set(map(tuple, fresh[0]))
        repeated = driver.evaluation_keys(seed, True)
        np.testing.assert_array_equal(fresh[0], repeated[0])
        np.testing.assert_array_equal(fresh[1], repeated[1])
        assert fresh[2] == repeated[2]


def test_reward_components_use_next_position_and_mask_padding(driver):
    trace = {'state': np.zeros((1, 3, 66), np.float32), 'valid_mask': np.array([[True, False]]),
             'blue_reward': np.array([[-.5, 1000]], np.float32), 'terminated_capture': np.zeros((1, 2), bool)}
    trace['state'][0, 1, 2] = 1.9
    trace['state'][0, 2, 2] = 20
    result = driver.metrics(trace)
    assert result['blue_return'][0] == pytest.approx(-.5)
    assert result['boundary_cost'][0] == pytest.approx(.5, abs=1e-6)
    assert result['outside_transitions'][0] == 0
    assert result['valid_transitions'][0] == 1
    assert json.loads(json.dumps(result['blue_return'].tolist())) == [-.5]


def test_checkpoint_leaf_guard_ignores_mapping_order_but_catches_changed_moment(driver):
    before = {'b': np.array([1., 2.], np.float32), 'a': {'count': np.asarray(128, np.int32)}}
    after = {'a': {'count': np.asarray(128, np.int32)}, 'b': np.array([1., 2.], np.float32)}
    driver.assert_state_equal(before, after)
    after['b'][1] += 1e-6
    with pytest.raises(ValueError, match='optimizer state'):
        driver.assert_state_equal(before, after)


def test_baseline_comparison_rejects_changed_controller_or_terminal_path(driver):
    trace = {name: np.zeros((2, 2), dtype=bool) for name in ['valid_mask', 'terminated_capture', 'truncated_timeout']}
    trace.update(valid_length=np.zeros(2, np.int32), environment_seed=np.zeros((2, 2), np.uint32), step_seed=np.ones((2, 2), np.uint32))
    trace.update({name: np.zeros((2, 2, 1), np.float32) for name in ['state', 'blue_action', 'red_action', 'blue_reward', 'context', 'final_context']})
    reference = {name: values.copy() for name, values in trace.items()}
    assert set(driver.verify_baseline(trace, reference).values()) == {0.}
    trace['blue_action'][0, 0, 0] = .01
    with pytest.raises(AssertionError):
        driver.verify_baseline(trace, reference)
    trace['blue_action'][:] = 0
    trace['terminated_capture'][0, 0] = True
    with pytest.raises(AssertionError):
        driver.verify_baseline(trace, reference)


def test_diagnostic_evidence_is_required_and_hash_checked_before_fitting(driver, tmp_path):
    artifact = tmp_path / 'diagnostic.raw'
    artifact.write_bytes(b'preserved candidate or trace')
    item = {'path': str(artifact), 'sha256': driver.file_sha256(artifact)}
    identities = [{'seed': s, 'arm': a, 'objective': o}
                  for s in (11, 22) for a in driver.ARMS for o in driver.OBJECTIVE_TYPES]
    scenes = [{**identity, 'step': 0, 'candidates': item, 'complete_initial_state': item}
              for identity in identities]
    scenes += [{**identity, 'step': 12, 'candidates': item, 'complete_initial_state': item}
               for identity in identities[:-1]]
    p = {'diagnostic_inputs': {'candidate_inventory': item, 'scenes': scenes,
         'original_validation_traces': [{**identity, **item} for identity in identities],
         'model_steps': [128, 2000, 8000]}}
    assert driver.diagnostic_input_hashes(p) == {str(artifact): item['sha256']}
    p['diagnostic_inputs']['scenes'] = scenes[:-1]
    with pytest.raises(ValueError, match='fixed35'):
        driver.diagnostic_input_hashes(p)
    p['diagnostic_inputs']['scenes'] = scenes
    artifact.write_bytes(b'changed diagnostic bytes')
    with pytest.raises(ValueError, match='changed'):
        driver.diagnostic_input_hashes(p)
