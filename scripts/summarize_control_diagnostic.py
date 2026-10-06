#!/usr/bin/env python3
"""Reconstruct the frozen control diagnostic and compare its unchanged probe states."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import flax.serialization as serialization
import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
import run_control_diagnostic as driver  # noqa: E402

from mopa import mppi  # noqa: E402
from mopa import planning_diagnostics as diagnostics  # noqa: E402
from mopa.bc_continuous import FrozenBCOpponent  # noqa: E402
from mopa.continuous_data import (  # noqa: E402
    OBJECTIVE_TYPES,
    load_continuous_actor_params,
    markov_state,
)
from mopa.manifest import file_sha256  # noqa: E402
from mopa.tdmpc import create_agent  # noqa: E402
from mopa.zero_s import ZeroSOpponent  # noqa: E402


def error_bins(prediction, trace):
    valid = np.asarray(trace['valid_mask'], bool)
    actual = np.asarray(trace['blue_reward'][valid], np.float64)
    size = np.abs(trace['state'][:, 1:, 2:4]).max(-1)[valid]
    prediction = np.asarray(prediction, np.float64).reshape(-1)
    if prediction.shape != actual.shape or not np.isfinite(prediction).all():
        raise ValueError('aligned finite predictions on all valid transitions required')
    result = {}
    for name, mask in [('all', np.ones(len(size), bool)), ('inside_le_1.8', size <= 1.8),
                       ('edge_1.8_to_2', (size > 1.8) & (size <= 2)), ('outside_gt_2', size > 2)]:
        error = prediction[mask] - actual[mask]
        result[name] = {'count': int(mask.sum()),
                        'mean_prediction': float(prediction[mask].mean()) if mask.any() else None,
                        'mean_actual': float(actual[mask].mean()) if mask.any() else None,
                        'mean_error': float(error.mean()) if mask.any() else None,
                        'mae': float(np.abs(error).mean()) if mask.any() else None,
                        'rmse': float(np.sqrt(np.square(error).mean())) if mask.any() else None}
    return result


def restore_controller(item, endpoint, step):
    source = json.loads(Path(item['manifest']['path']).read_text())
    shared = Path(item['shared_manifest']['path']).parent
    shared_manifest = json.loads((shared / 'manifest.json').read_text())
    driver.check_files({str(shared / name): digest for name, digest in shared_manifest['artifacts'].items()})
    with np.load(shared / 'stats.npz', allow_pickle=False) as arrays:
        mean, std = arrays['mean'], arrays['std']
    agent = create_agent(source['model_configuration'], 66, key=jax.random.PRNGKey(item['seed']),
                         obs_mean=mean, obs_std=std)
    opponent = None
    if item['arm'] == '0s':
        opponent = ZeroSOpponent.load(shared / '0s.msgpack')
        agent = opponent.attach(agent, mean, std)
    elif item['arm'] == 'bc':
        agent = FrozenBCOpponent.load(shared / 'bc.npz').attach(agent, mean, std)
    red = None if item['arm'] == 'implicit' else serialization.to_state_dict(agent.model.red_model.params)
    agent = serialization.from_bytes(agent, endpoint.read_bytes())
    driver.assert_step(agent, step)
    if red is not None:
        driver.assert_state_equal(serialization.to_state_dict(agent.model.red_model.params), red)
    return agent, mean, std


def candidate_key(step):
    """The pre-existing P10 probe's final estimate_value key, independent of weights."""
    _, planning_key = jax.random.split(jax.random.PRNGKey(20261005 + step))
    prior_key = jax.random.split(planning_key, 4)[0]  # H3 prior proposals
    return jax.random.split(prior_key, 8)[-1]  # two keys plus six MPPI iterations


def _summarize(args, report):
    protocol, sources = driver.bind_protocol(args)
    manifest_path = args.run / 'manifest.json'
    manifest_digest = file_sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if (manifest['status'] != 'complete' or manifest['specification_commit'] != args.spec_commit
            or manifest['executable_commit'] != args.code_commit
            or manifest['protocol_sha256'] != file_sha256(args.protocol)):
        raise ValueError('complete run under the identical frozen protocol/source required')
    identities = {(r['seed'], r['arm']) for r in manifest['controllers']}
    expected = {(s, a) for s in protocol['fitting_seeds'] for a in protocol['arms']}
    if identities != expected or len(manifest['controllers']) != len(expected):
        raise ValueError('exactly all declared independent fitting runs required')
    if any(r['status'] != 'complete' or [e['step'] for e in r['endpoints']] != [128, 2000, 8000]
           for r in manifest['controllers']):
        raise ValueError('all fixed controller endpoints must be complete')
    files = {str(args.run / name): digest for name, digest in manifest['artifacts'].items()}
    files[str(manifest_path)] = manifest_digest
    files.update(manifest['input_hashes'])
    probes = protocol['diagnostic_inputs']
    for row in protocol['initial_artifacts']:
        for name in ('manifest', 'checkpoint', 'shared_manifest'):
            files[row[name]['path']] = row[name]['sha256']
    for row in probes['original_validation_traces']:
        files[row['path']] = row['sha256']
    for row in probes['scenes']:
        for name in ('candidates', 'complete_initial_state'):
            files[row[name]['path']] = row[name]['sha256']
    files[probes['candidate_inventory']['path']] = probes['candidate_inventory']['sha256']
    driver.check_files(files)
    args.output.mkdir(parents=True, exist_ok=False)
    report.update(status='running', scope='Fixed P2 probe states and candidate actions; no fitting or model selection.',
                  protocol_sha256=file_sha256(args.protocol), specification_commit=args.spec_commit,
                  executable_commit=args.code_commit, rollouts=[], reward=[], candidates=[])
    driver.write_json(args.output / 'report.json', report)
    env = driver.campaign.historical.benchmark_env()
    binding = json.loads((Path(protocol['initial_campaign']) / 'campaign.json').read_text())
    params = {objective: load_continuous_actor_params(binding['specialists']['1:' + objective]['path'])
              for objective in OBJECTIVE_TYPES}
    for item in protocol['initial_artifacts']:
        seed, arm = item['seed'], item['arm']
        directory = args.run / f'seed_{seed}' / arm
        original_traces = [r for r in probes['original_validation_traces'] if (r['seed'], r['arm']) == (seed, arm)]
        scenes = [r for r in probes['scenes'] if (r['seed'], r['arm']) == (seed, arm)]
        for step in probes['model_steps']:
            name = f'step_{step:05d}'
            endpoint = directory / f'{name}.msgpack'
            if str(endpoint) not in files:
                raise ValueError('unbound controller endpoint')
            agent, mean, std = restore_controller(item, endpoint, step)
            evaluation = directory / f'{name}_evaluation'
            rows = json.loads((evaluation / 'metrics.json').read_text())
            seen = set()
            for row in rows:
                key = (row['reset_set'], row['mode'], row['objective'])
                if key in seen or (row['seed'], row['arm'], row['step']) != (seed, arm, step):
                    raise ValueError('duplicated or incompatible evaluation row')
                seen.add(key)
                with np.load(evaluation / row['trace'], allow_pickle=False) as raw:
                    trace = dict(raw)
                values = driver.metrics(trace)
                for metric, value in values.items():
                    np.testing.assert_allclose(value, row['metrics'][metric], atol=1e-10, rtol=0)
                report['rollouts'].append({**{k: row[k] for k in ('seed', 'arm', 'step', 'reset_set', 'mode', 'objective')},
                                           'episode_keys': trace['environment_seed'], 'metrics': values})
            expected_rows = {(reset, mode, objective) for reset in ('reused', 'fresh')
                             for mode in ('MPPI', 'policy-only') for objective in OBJECTIVE_TYPES}
            if seen != expected_rows:
                raise ValueError('missing prespecified rollout comparison')
            for row in original_traces:
                with np.load(row['path'], allow_pickle=False) as raw:
                    trace = dict(raw)
                valid = trace['valid_mask']
                x = agent.model.encode(jnp.asarray(trace['state'][:, :-1][valid]), agent.model.encoder.params,
                                       jax.random.PRNGKey(0))
                blue, context = jnp.asarray(trace['blue_action'][valid]), jnp.asarray(trace['context'][valid])
                model = agent.model
                true_red = jnp.asarray(trace['red_action'][valid])
                recorded = model.reward(x, model.transition_inputs(blue, context, true_red), model.reward_model.params)[0]
                predictions = {'implicit' if arm == 'implicit' else 'recorded_red': recorded}
                if arm != 'implicit':
                    red = model.red_action(x, context, model.red_model.params)
                    predictions['learned_red'] = model.reward(x, model.transition_inputs(blue, context, red), model.reward_model.params)[0]
                path = args.output / f'{seed}_{arm}_{step}_{row["objective"]}_reward.npz'
                np.savez_compressed(path, episode_step=np.argwhere(valid), actual=trace['blue_reward'][valid], **predictions)
                report['reward'].append({'seed': seed, 'arm': arm, 'step': step, 'objective': row['objective'],
                                         'errors': {k: error_bins(v, trace) for k, v in predictions.items()}})
            for scene in scenes:
                row = next(r for r in original_traces if r['objective'] == scene['objective'])
                with np.load(row['path'], allow_pickle=False) as arrays:
                    trace = dict(arrays)
                with np.load(scene['candidates']['path'], allow_pickle=False) as arrays:
                    old = dict(arrays)
                _, template = env.reset(jnp.asarray(trace['environment_seed'][0]))
                state = serialization.from_bytes(template, Path(scene['complete_initial_state']['path']).read_bytes())
                initial = np.asarray(markov_state(env, state))
                np.testing.assert_allclose(initial, trace['state'][0, scene['step']], atol=2e-6, rtol=0)
                actual = diagnostics.simulator_candidates(env, state, old['actions'], params[scene['objective']], trace['step_seed'][0])
                actual.pop('complete_final_state')
                for key, value in actual.items():
                    np.testing.assert_allclose(value, old['actual_' + key], atol=1e-5, rtol=0)
                context = jnp.asarray(trace['context'][0, scene['step']])
                def learned_red(raw, c, time):
                    del time
                    return agent.model.red_action((raw - mean) / std, c, agent.model.red_model.params)
                predicted = diagnostics.imagined_candidates(agent, initial, old['actions'], context, mean, std,
                    learned_red if arm != 'implicit' else None, key=candidate_key(scene['step']))
                x = agent.model.encode(jnp.broadcast_to(initial, (len(old['actions']), 66)), agent.model.encoder.params, jax.random.PRNGKey(0))
                c = jnp.broadcast_to(context, (len(old['actions']), model.context_dim))
                score = mppi.estimate_value(agent, x, jnp.asarray(old['actions']), c, 3, candidate_key(scene['step']))
                np.testing.assert_allclose(predicted['planner_return'], score, atol=2e-4, rtol=2e-5)
                result = diagnostics.compare_candidates(predicted, actual, std, agent.discount,
                    capture_distance=float(np.asarray(env.rad)[:2].sum()), arena=2.)
                if arm == 'implicit':
                    result['red_action_squared_2d_error'] = None
                both = actual['valid'] & predicted['valid']
                pred_continue = predicted['continues_probability'] > .5
                result['false_continue_count'] = int((both & pred_continue & ~actual['continues']).sum())
                result['false_stop_count'] = int((both & ~pred_continue & actual['continues']).sum())
                out = args.output / f'{seed}_{arm}_{step}_{scene["objective"]}_{scene["step"]}_candidates.npz'
                np.savez_compressed(out, **predicted)
                report['candidates'].append({'seed': seed, 'arm': arm, 'step': step,
                    'objective': scene['objective'], 'scene_step': scene['step'], 'metrics': result})
            print(f'verified {arm} seed {seed} step {step}', flush=True)
            driver.write_json(args.output / 'report.json', report)
    driver.check_files({**sources, **files})
    report.update(status='complete', source_hashes=sources, input_hashes=files,
                  artifacts={p.name: file_sha256(p) for p in args.output.iterdir()
                             if p.is_file() and p.name != 'report.json'})
    driver.write_json(args.output / 'report.json', report)


def summarize(args):
    report = {}
    try:
        _summarize(args, report)
    except BaseException as exc:
        if report:  # Only this invocation's freshly created output may be changed.
            report.update(status='failed', error=repr(exc))
            driver.write_json(args.output / 'report.json', report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'spec-repo', 'run', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('spec-commit', 'code-commit'):
        parser.add_argument('--' + name, required=True)
    summarize(parser.parse_args())


if __name__ == '__main__':
    main()
