#!/usr/bin/env python3
"""Continue reviewed P2 TD checkpoints under a committed boundary diagnostic protocol."""
from __future__ import annotations

import argparse
import fcntl
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import flax.serialization as serialization
import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
import run_resl_campaign as campaign  # noqa: E402
from run_opponent_updates import (  # noqa: E402
    check_files,
    fingerprints,
    load_verifier,
    source_binding,
)

from mopa.bc_continuous import FrozenBCOpponent  # noqa: E402
from mopa.continuous_data import (  # noqa: E402
    OBJECTIVE_TYPES,
    load_continuous_actor_params,
    markov_state,
    replay_episodes,
)
from mopa.evaluation import specialist_action_function  # noqa: E402
from mopa.manifest import file_sha256, package_versions  # noqa: E402
from mopa.tdmpc import check_update, create_agent  # noqa: E402
from mopa.tdmpc_data import SequenceReplay, state_statistics  # noqa: E402
from mopa.zero_s import ZeroSOpponent  # noqa: E402

ARMS = ('implicit', 'bc', '0s')
TRAIN_STATES = ('encoder', 'dynamics_model', 'reward_model', 'value_model', 'policy_model', 'continue_model')
write_json = campaign.write_json


def protocol_configuration(p):
    """This diagnostic has explicit endpoint/data budgets, never smoke defaults."""
    if p['status'] != 'frozen' or p['fitting_seeds'] != [11, 22] or tuple(p['arms']) != ARMS:
        raise ValueError('frozen protocol with the six declared controllers required')
    if p['initial_optimizer_step'] != 128:
        raise ValueError('the exact saved128-update starting point is required')
    same, online, evaluation = p['same_replay_stage'], p['online_stage'], p['evaluation']
    expected = {'additional_updates': 1872, 'total_updates': 2000, 'new_real_transitions': 0}
    if any(same[k] != v for k, v in expected.items()):
        raise ValueError('same-replay stage must isolate additional fitting to2000')
    expected = {'rounds': 6, 'additional_updates_per_round': 1000,
                'valid_transitions_per_type_per_family_per_round': 600,
                'training_checkpoint_families': [0], 'total_new_transitions_per_controller': 10800,
                'total_updates_at_end': 8000, 'collection_batch_episodes': 1,
                'collection_round_indices': [1, 2, 3, 4, 5, 6]}
    if any(online[k] != v for k, v in expected.items()):
        raise ValueError('online stage differs from the declared existing procedure')
    if (evaluation['checkpoint_family'] != 1 or evaluation['model_steps'] != [128, 2000, 8000]
            or evaluation['modes'] != ['MPPI', 'policy-only']
            or evaluation['objectives'] != list(OBJECTIVE_TYPES)
            or evaluation['episodes_per_fit_per_type_per_mode_per_checkpoint'] != 8
            or evaluation['fresh_key_base'] != 900001):
        raise ValueError('paired validation keys, modes and endpoints must remain fixed')
    identities = [(r['seed'], r['arm']) for r in p['initial_artifacts']]
    if len(identities) != 6 or set(identities) != {(s, a) for s in (11, 22) for a in ARMS}:
        raise ValueError('exactly one source artifact per declared seed/arm required')
    fixed = p['fixed']
    for key, value in {'horizon': 3, 'population_size': 512, 'mppi_iterations': 6,
                       'policy_prior_samples': 24, 'num_elites': 64, 'discount': .99,
                       'termination_contract': 'finite_100_step_v1'}.items():
        if fixed[key] != value:
            raise ValueError('the saved task and planner must remain unchanged')
    return {'training_checkpoints': [0], 'transitions_per_training_group': 600,
            'terminal_contract': 'finite_100_step_v1'}


def bind_protocol(args):
    relative = args.protocol.resolve().relative_to(args.spec_repo.resolve())
    original = subprocess.check_output(['git', 'show', f'{args.spec_commit}:{relative}'], cwd=args.spec_repo)
    if len(args.spec_commit) != 40 or original != args.protocol.read_bytes():
        raise ValueError('protocol must equal the committed private specification')
    p = json.loads(original)
    protocol_configuration(p)
    if args.code_commit != p.get('frozen_executable_commit'):
        raise ValueError('executable commit differs from frozen protocol')
    bound = source_binding(args.code_commit)
    bound[str(args.protocol.resolve())] = file_sha256(args.protocol)
    return p, bound


def diagnostic_input_hashes(p):
    """Bind the prescribed calibration evidence before fitting any controller."""
    probes = p['diagnostic_inputs']
    traces, scenes = probes['original_validation_traces'], probes['scenes']
    expected = {(s, a, o) for s in (11, 22) for a in ARMS for o in OBJECTIVE_TYPES}
    def identity(row):
        return row['seed'], row['arm'], row['objective']

    if (len(traces) != 18 or {identity(row) for row in traces} != expected
            or len(scenes) != 35 or len({(*identity(row), row['step']) for row in scenes}) != 35
            or {identity(row) for row in scenes if row['step'] == 0} != expected
            or any(identity(row) not in expected or row['step'] < 0 for row in scenes)
            or probes['model_steps'] != [128, 2000, 8000]):
        raise ValueError('fixed35 candidate scenes and18 validation traces required')
    bindings = [probes['candidate_inventory'], *traces]
    for row in scenes:
        bindings.extend([row['candidates'], row['complete_initial_state']])
    result = {}
    for item in bindings:
        if item['path'] in result and result[item['path']] != item['sha256']:
            raise ValueError('conflicting diagnostic artifact hashes')
        result[item['path']] = item['sha256']
    check_files(result)
    return result


def assert_step(agent, expected):
    for name in TRAIN_STATES:
        if int(getattr(agent.model, name).step) != expected:
            raise ValueError(f'{name} optimizer step differs from{expected}')



def assert_state_equal(actual, expected):
    """Exact parameter/optimizer leaves; mapping serialization order is immaterial."""
    if isinstance(actual, dict) and isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise ValueError('checkpoint state keys differ')
        for key in actual:
            assert_state_equal(actual[key], expected[key])
        return
    if actual is None or expected is None:
        if actual is not expected:
            raise ValueError('checkpoint state None differs')
        return
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape or a.dtype != b.dtype or a.tobytes() != b.tobytes():
        raise ValueError('checkpoint parameter/optimizer state differs')


def advance(agent, replay, rng, key, target, path):
    """Continue the existing optimizer and RNG; persist every finite check."""
    start = int(agent.model.reward_model.step)
    if target <= start:
        raise ValueError('strictly increasing update endpoint required')
    assert_step(agent, start)
    history = []
    with path.open('x') as log:
        for number in range(start + 1, target + 1):
            key, update_key = jax.random.split(key)
            begun = time.monotonic()
            candidate, info = agent.update(**replay.sample(rng, agent.batch_size), key=update_key)
            check_update(info, number)
            agent = candidate
            row = {k: float(np.asarray(v)) for k, v in info.items() if np.ndim(v) == 0}
            row.update(step=number, seconds=time.monotonic() - begun)
            log.write(json.dumps(row, allow_nan=False) + '\n')
            log.flush()
            history.append(row)
    assert_step(agent, target)
    return agent, key, history


def restore_rng(manifest):
    rng = np.random.default_rng()
    rng.bit_generator.state = manifest['replay_rng']
    key = np.asarray(manifest['update_key'])
    if key.shape != (2,) or not np.issubdtype(key.dtype, np.integer) or np.any(key < 0) or np.any(key > 2**32 - 1):
        raise ValueError('saved legacy JAX key must contain two uint32 words')
    return rng, jnp.asarray(key, jnp.uint32)


def evaluation_keys(seed, fresh):
    if not fresh:
        return campaign.historical.keys(seed, 0, 0, 0, 4, evaluation=True)
    key = jax.random.fold_in(jax.random.PRNGKey(900001), seed)
    reset, step = jax.random.split(key)
    return (np.asarray(jax.random.split(reset, 4)), np.asarray(jax.random.split(step, 4)),
            int(np.asarray(key)[0]) & 0x7FFFFFF0)


def register_resets(env, keys, used_keys, used_states):
    keys = np.asarray(keys)
    hashes = fingerprints(env, keys)
    new_keys = {tuple(k) for k in keys.tolist()}
    if len(new_keys) != len(keys) or len(set(hashes)) != len(keys) or new_keys & used_keys or set(hashes) & used_states:
        raise ValueError('new reset overlaps a reserved key or complete initial state')
    used_keys.update(new_keys)
    used_states.update(hashes)
    return hashes


def metrics(trace):
    valid = trace['valid_mask']
    positions = trace['state'][:, 1:, 2:4]
    x = np.abs(positions) / 2
    boundary = np.where(x < .9, 0, np.where(x < 1, (x - .9) * 10,
                         np.minimum(np.exp(np.minimum(2 * x - 2, 80)), 10))).sum(-1)
    resources = np.diff(trace['state'][..., 40:56].sum(-1), axis=1)
    reconstructed = 5 * resources - 10 * trace['terminated_capture'] - boundary
    np.testing.assert_allclose(reconstructed[valid], trace['blue_reward'][valid], rtol=0, atol=2e-5)
    return {'blue_return': np.where(valid, trace['blue_reward'], 0).sum(1, dtype=np.float64),
            'boundary_cost': np.where(valid, boundary, 0).sum(1, dtype=np.float64),
            'outside_transitions': ((np.abs(positions) > 2).any(-1) & valid).sum(1),
            'valid_transitions': valid.sum(1), 'captured': (trace['terminated_capture'] & valid).any(1),
            'resources_collected': np.where(valid, resources, 0).sum(1)}


def evaluate(agent, opponent, params, seed, arm, directory):
    directory.mkdir(exist_ok=False)
    env = campaign.historical.benchmark_env()
    policy = jax.jit(lambda s, c, k: agent.act(s, mpc=False, deterministic=True, train=False, context=c, key=k)[0])

    def blue(state, obs, context, carry, key, t):
        del obs, t
        return policy(markov_state(env, state), context, key), carry

    controllers = {'MPPI': campaign.historical.Controller(agent, env, training=False), 'policy-only': blue}
    records = []
    for fresh in (False, True):
        reset, step, base = evaluation_keys(seed, fresh)
        for mode, controller in controllers.items():
            for label, objective in enumerate(OBJECTIVE_TYPES):
                result = campaign.historical.run_matched_episodes(
                    env, params[1, label], controller, reset, step, horizon=100,
                    context_mode='online' if opponent else 'zero', label=label,
                    zero_s=opponent, context_width=None if opponent else 0,
                    shuffle_seed=base, record_transitions=True)
                trace = result['transitions']
                filename = f'{"fresh" if fresh else "reused"}_{mode}_{objective}.npz'
                path = directory / filename
                np.savez_compressed(path, **trace, environment_seed=reset, step_seed=step,
                                    terminal_for_value=campaign.value_terminals(trace, 'finite_100_step_v1'))
                values = metrics(trace)
                np.testing.assert_allclose(result['blue_return'], values['blue_return'], rtol=1e-6, atol=2e-5)
                records.append({'seed': seed, 'arm': arm, 'step': int(agent.model.reward_model.step),
                                'reset_set': 'fresh' if fresh else 'reused', 'mode': mode,
                                'objective': objective, 'episode_keys': reset, 'metrics': values,
                                'trace': filename, 'sha256': file_sha256(path)})
    write_json(directory / 'metrics.json', records)
    return records



def verify_baseline(trace, original):
    """Same saved128 checkpoint and reset keys must reproduce its actual MPPI path."""
    exact = ('valid_mask', 'valid_length', 'terminated_capture', 'truncated_timeout', 'environment_seed', 'step_seed')
    for name in exact:
        np.testing.assert_array_equal(trace[name], original[name])
    errors = {}
    for name, tolerance in {'state': 2e-6, 'blue_action': 1e-6, 'red_action': 1e-6,
                            'blue_reward': 1e-6, 'context': 2e-5, 'final_context': 2e-5}.items():
        np.testing.assert_allclose(trace[name], original[name], atol=tolerance, rtol=0)
        errors[name] = float(np.abs(trace[name] - original[name]).max(initial=0.))
    return errors


def save_state(agent, rng, key, directory, name):
    checkpoint = directory / f'{name}.msgpack'
    with checkpoint.open('xb') as handle:
        handle.write(serialization.to_bytes(agent))
    state = {'checkpoint': checkpoint.name, 'sha256': file_sha256(checkpoint),
             'step': int(agent.model.reward_model.step), 'replay_rng': rng.bit_generator.state,
             'update_key': np.asarray(key).tolist()}
    write_json(directory / f'{name}.json', state)
    return state


def run(args):
    p, source = bind_protocol(args)
    cfg = protocol_configuration(p)
    args.output.mkdir(parents=True, exist_ok=False)
    lock_path = ROOT / 'artifacts/control_diagnostic.lock'
    lock_path.parent.mkdir(exist_ok=True)
    lock = lock_path.open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = {'status': 'running', 'protocol_id': p['protocol_id'], 'specification_commit': args.spec_commit,
                'protocol_sha256': file_sha256(args.protocol), 'executable_commit': args.code_commit,
                'dependencies': package_versions(), 'controllers': []}
    write_json(args.output / 'manifest.json', manifest)
    try:
        pilot = Path(p['initial_campaign'])
        binding = json.loads((pilot / 'campaign.json').read_text())
        if file_sha256(pilot / 'campaign.json') != p['initial_campaign_sha256']:
            raise ValueError('original campaign binding differs from frozen protocol')
        if binding['executable_commit'] != p['initial_executable_commit'] or binding['specification_commit'] != p['initial_specification_commit']:
            raise ValueError('original executable/specification differs')
        inputs = diagnostic_input_hashes(p)
        inputs.update({str(pilot / 'campaign.json'): p['initial_campaign_sha256'], binding['dataset_path']: binding['dataset_sha256']})
        sidecar = Path(binding['dataset_path']).with_suffix('.manifest.json')
        inputs[str(sidecar)] = binding['dataset_manifest_sha256']
        for row in p['initial_artifacts']:
            for name in ('manifest', 'checkpoint', 'shared_manifest'):
                item = row[name]
                inputs[item['path']] = item['sha256']
        for item in binding['specialists'].values():
            inputs[item['path']] = item['sha256']
        check_files(inputs)
        manifest.update(source_hashes=source, input_hashes=inputs)
        write_json(args.output / 'manifest.json', manifest)
        print('Initial source, controller, dataset and diagnostic hashes verified.', flush=True)
        with np.load(binding['dataset_path'], allow_pickle=False) as raw:
            all_data = dict(raw)
        rows = np.flatnonzero(all_data['checkpoint_seed'] == 0)
        if len(rows) != 600:
            raise ValueError('the original600 family0 fitting episodes are required')
        data = {k: v[rows] for k, v in all_data.items()}
        mean, std = state_statistics(data['state'], data['valid_mask'])
        env = campaign.historical.benchmark_env()
        reserved = {tuple(k) for k in all_data['environment_seed'].tolist()}
        # Every prior controller trace is reserved, including inspected validation.
        for item in p['initial_artifacts']:
            old_dir = Path(item['manifest']['path']).parent
            old = json.loads(Path(item['manifest']['path']).read_text())
            inputs.update({str(old_dir / name): digest for name, digest in old['evaluation_artifacts'].items()})
            check_files(inputs)
            for round_row in old['rounds']:
                for record in round_row['data']:
                    trace_path = old_dir / record['file']
                    inputs[str(trace_path)] = record['sha256']
                    check_files({str(trace_path): record['sha256']})
                    with np.load(trace_path) as trace:
                        reserved.update(map(tuple, trace['environment_seed'].tolist()))
        for seed in (11, 22):
            reserved.update(map(tuple, evaluation_keys(seed, False)[0].tolist()))
        reserved_states = set(fingerprints(env, np.asarray(sorted(reserved), np.uint32)))
        fresh_bindings = {'reused': {}, 'fresh': {}}
        for seed in (11, 22):
            fresh, step, base = evaluation_keys(seed, True)
            hashes = register_resets(env, fresh, reserved, reserved_states)
            fresh_bindings['fresh'][str(seed)] = {'reset': fresh, 'step': step, 'shuffle_seed': base, 'complete_initial_state_sha256': hashes}
            old_reset, old_step, old_base = evaluation_keys(seed, False)
            fresh_bindings['reused'][str(seed)] = {'reset': old_reset, 'step': old_step, 'shuffle_seed': old_base,
                                                 'complete_initial_state_sha256': fingerprints(env, old_reset)}
        write_json(args.output / 'validation_keys.json', fresh_bindings)
        params = {(family, label): load_continuous_actor_params(binding['specialists'][f'{family}:{objective}']['path'])
                  for family in (0, 1) for label, objective in enumerate(OBJECTIVE_TYPES)}
        initial_replay = replay_episodes(all_data, rows)
        if any(v > 1e-5 for k, v in initial_replay.items() if k.startswith('max_abs_error_')) or initial_replay['termination_flags_match_fraction'] != 1:
            raise ValueError('initial offline simulator replay failed')
        write_json(args.output / 'initial_replay.json', initial_replay)
        verifier = load_verifier()
        step_fn = jax.jit(jax.vmap(env.step_env))
        state_fn = jax.jit(lambda state: markov_state(env, state))
        red_fns = {pair: specialist_action_function(par, 35)
                   for pair, par in params.items()}
        for label in range(3):
            subset = np.flatnonzero(data['objective_label'] == label)
            valid = data['valid_mask'][subset]
            predicted = red_fns[0, label](jnp.asarray(data['red_observation'][subset, :-1][valid]))
            np.testing.assert_allclose(predicted, data['red_action'][subset][valid], atol=1e-6, rtol=0)
        jobs = []

        def guards(job):
            check_files({**source, **inputs})
            if job['frozen_red'] is not None:
                assert_state_equal(serialization.to_state_dict(job['agent'].model.red_model.params), job['frozen_red'])
            if shutil.disk_usage(args.output).free < 2 * 1024**3:
                raise RuntimeError('less than2GiB disk remains; preserve partial run')

        def endpoint(job, name):
            guards(job)
            print(f"seed={job['seed']} arm={job['arm']} endpoint={name}: evaluating reused and fresh reset sets", flush=True)
            snapshot = save_state(job['agent'], job['rng'], job['key'], job['directory'], name)
            eval_dir = job['directory'] / f'{name}_evaluation'
            evaluate(job['agent'], job['opponent'], params, job['seed'], job['arm'], eval_dir)
            certificates = []
            baseline = []
            if name == 'step_00128':
                for objective in OBJECTIVE_TYPES:
                    with np.load(eval_dir / f'reused_MPPI_{objective}.npz') as new, np.load(job['original_evaluation'] / f'{objective}.npz') as old:
                        baseline.append({'objective': objective, 'max_errors': verify_baseline(dict(new), dict(old))})
                write_json(eval_dir / 'original_baseline_comparison.json', baseline)
            for file in sorted(eval_dir.glob('*.npz')):
                objective = file.stem.rsplit('_', 1)[-1]
                certificate = verifier(file, env, step_fn, state_fn, red_fns[1, OBJECTIVE_TYPES.index(objective)], job['opponent'])
                certificates.append({'trace': file.name, **certificate})
            write_json(eval_dir / 'replay_verification.json', certificates)
            job['record']['endpoints'].append(snapshot)
            guards(job)
            write_json(args.output / 'manifest.json', manifest)
            print(f"seed={job['seed']} arm={job['arm']} endpoint={name}: saved and replay verified", flush=True)

        for item in p['initial_artifacts']:
            seed, arm = item['seed'], item['arm']
            original = json.loads(Path(item['manifest']['path']).read_text())
            shared_path = Path(item['shared_manifest']['path'])
            shared = json.loads(shared_path.read_text())
            if (original['seed'] != seed or original['arm'] != arm or original['status'] != 'complete'
                    or shared['seed'] != seed or shared['status'] != 'complete'
                    or original['binding'] != binding or shared['binding'] != binding
                    or original['shared_manifest_sha256'] != item['shared_manifest']['sha256']
                    or original['checkpoint_sha256'] != item['checkpoint']['sha256']):
                raise ValueError('initial controller/shared artifact identity differs')
            if Path(item['checkpoint']['path']).resolve() != (Path(item['manifest']['path']).parent / original['checkpoint']).resolve():
                raise ValueError('checkpoint is not the manifest-selected128-update source')
            for name, digest in shared['artifacts'].items():
                inputs[str(shared_path.parent / name)] = digest
            check_files(inputs)
            with np.load(shared_path.parent / 'stats.npz') as stats:
                np.testing.assert_array_equal(mean, stats['mean'])
                np.testing.assert_array_equal(std, stats['std'])
            model_cfg = original['model_configuration']
            if (model_cfg != campaign.controller_configuration(arm, 8 if arm == '0s' else 0)
                    or binding['configuration']['terminal_contract'] != 'finite_100_step_v1'):
                raise ValueError('saved method/task differs from the existing implementation')
            agent = create_agent(model_cfg, 66, key=jax.random.PRNGKey(seed), obs_mean=mean, obs_std=std)
            opponent = ZeroSOpponent.load(shared_path.parent / '0s.msgpack') if arm == '0s' else None
            attached = opponent if opponent else FrozenBCOpponent.load(shared_path.parent / 'bc.npz') if arm == 'bc' else None
            if attached:
                agent = attached.attach(agent, mean, std)
            expected_red = serialization.to_state_dict(agent.model.red_model.params) if arm != 'implicit' else None
            checkpoint_bytes = Path(item['checkpoint']['path']).read_bytes()
            agent = serialization.from_bytes(agent, checkpoint_bytes)
            assert_state_equal(serialization.to_state_dict(agent), serialization.msgpack_restore(checkpoint_bytes))
            if expected_red is not None:
                assert_state_equal(serialization.to_state_dict(agent.model.red_model.params), expected_red)
            assert_step(agent, 128)
            with np.load(shared_path.parent / 'context.npz') as context_file:
                context = context_file['context'] if opponent else np.zeros((*data['state'].shape[:2], 0), np.float32)
            if opponent is not None:
                reconstructed_context = opponent.context(data['state'], data['red_action'], data['valid_length'])
                np.testing.assert_allclose(context, reconstructed_context, atol=2e-5, rtol=0)
            replay = SequenceReplay.from_dataset(campaign.training_trace(data, 'finite_100_step_v1'),
                                                np.arange(len(rows)), agent.horizon, context)
            for round_row in original['rounds']:
                for record in round_row['data']:
                    with np.load(Path(item['manifest']['path']).parent / record['file']) as archive:
                        trace = dict(archive)
                    if not np.all(trace['checkpoint_seed'] == 0):
                        raise ValueError('nontraining specialist in initial replay')
                    verifier(Path(item['manifest']['path']).parent / record['file'], env, step_fn, state_fn,
                             red_fns[0, OBJECTIVE_TYPES.index(record['objective'])], opponent)
                    replay.append(campaign.training_trace(trace, 'finite_100_step_v1'), campaign.historical.replay_context(trace))
            if replay.n_transitions != 42974 + 72:
                raise ValueError('initial replay does not exactly recover the original transition count')
            rng, key = restore_rng(original)
            frozen_red = serialization.to_state_dict(agent.model.red_model.params) if arm != 'implicit' else None
            directory = args.output / f'seed_{seed}' / arm
            directory.mkdir(parents=True, exist_ok=False)
            record = {'seed': seed, 'arm': arm, 'status': 'running', 'initial_source': item, 'rounds': [], 'endpoints': []}
            manifest['controllers'].append(record)
            used_keys, used_states = set(reserved), set(reserved_states)

            with (directory / 'state_stats.npz').open('xb') as handle:
                handle.write((shared_path.parent / 'stats.npz').read_bytes())
            job = dict(agent=agent, replay=replay, rng=rng, key=key, opponent=opponent,
                       seed=seed, arm=arm, directory=directory, record=record,
                       used_keys=used_keys, used_states=used_states, frozen_red=frozen_red,
                       original_evaluation=Path(item['manifest']['path']).parent / original['evaluation_directory'])
            jobs.append(job)
            endpoint(job, 'step_00128')
        # All six restoration, initial replay, reset and128-evaluation gates finish
        # before any fitted parameter is changed.
        manifest['global_preflight'] = 'complete'
        write_json(args.output / 'manifest.json', manifest)
        print('All six restoration and step128 evaluation gates passed; starting continuation.', flush=True)
        for job in jobs:
            agent, replay, rng, key = job['agent'], job['replay'], job['rng'], job['key']
            directory, record = job['directory'], job['record']
            seed, arm, opponent = job['seed'], job['arm'], job['opponent']
            used_keys, used_states = job['used_keys'], job['used_states']
            guards(job)
            print(f'seed={seed} arm={arm}: continuing unchanged replay from128 to2000 updates', flush=True)
            agent, key, _ = advance(agent, replay, rng, key, 2000, directory / 'same_replay_updates.jsonl')
            job.update(agent=agent, key=key)
            endpoint(job, 'step_02000')
            for round_index in range(1, 7):
                guards(job)
                print(f'seed={seed} arm={arm} round={round_index}: collecting1800 training transitions', flush=True)
                attempt = directory / f'round_{round_index:03d}' / 'attempt_000'
                attempt.mkdir(parents=True, exist_ok=False)
                traces, _, records = campaign.collect(cfg, agent, opponent, params, seed, round_index, attempt)
                if sum(r['valid_transitions'] for r in records) != 1800:
                    raise ValueError('incorrect new real interaction quota')
                for trace_record, trace in zip(records, traces, strict=True):
                    verifier(directory / trace_record['file'], env, step_fn, state_fn,
                             red_fns[0, OBJECTIVE_TYPES.index(trace_record['objective'])], opponent)
                    with np.load(directory / trace_record['file']) as saved:
                        register_resets(env, saved['environment_seed'], used_keys, used_states)
                        if not np.all(saved['checkpoint_seed'] == 0):
                            raise ValueError('validation family leaked into training')
                    replay.append(campaign.training_trace(trace, 'finite_100_step_v1'), campaign.historical.replay_context(trace))
                print(f'seed={seed} arm={arm} round={round_index}: collected and verified; fitting1000 updates', flush=True)
                agent, key, _ = advance(agent, replay, rng, key, 2000 + 1000 * round_index,
                                        attempt / 'updates.jsonl')
                job.update(agent=agent, key=key)
                state = save_state(agent, rng, key, directory, f'round_{round_index:03d}')
                record['rounds'].append({'round': round_index, 'data': records, 'state': state,
                                         'replay_transitions': replay.n_transitions})
                guards(job)
                write_json(args.output / 'manifest.json', manifest)
                print(f'seed={seed} arm={arm} round={round_index}: checkpoint saved at{state["step"]} updates', flush=True)
            endpoint(job, 'step_08000')
            record['status'] = 'complete'
            write_json(args.output / 'manifest.json', manifest)
        check_files({**source, **inputs})
        manifest.update(status='complete', source_hashes=source, input_hashes=inputs,
                        artifacts={str(p.relative_to(args.output)): file_sha256(p) for p in args.output.rglob('*')
                                   if p.is_file() and p.name != 'manifest.json'})
        write_json(args.output / 'manifest.json', manifest)
    except BaseException as exc:
        manifest.update(status='failed', error=repr(exc))
        write_json(args.output / 'manifest.json', manifest)
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--spec-repo', type=Path, required=True)
    parser.add_argument('--spec-commit', required=True)
    parser.add_argument('--code-commit', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.execute:
        parser.error('explicit --execute is required; no automatic fitting')
    run(args)


if __name__ == '__main__':
    main()
