"""Continuous joint-action data contract: shapes, flags, matched resets, replay."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("jaxmarl")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from mopa.continuous_data import (  # noqa: E402
    MARKOV_STATE_FIELDS,
    OBJECTIVE_TYPES,
    ContinuousTrajectoryDataset,
    continuous_checkpoint_path,
    continuous_objective_dataset,
    family_behavior_summary,
    markov_state,
    markov_state_dim,
    replay_episodes,
    state_action_coverage,
    validate_continuous_dataset,
)
from mopa.nets import ContinuousActor  # noqa: E402
from tag_objectives import make_env  # noqa: E402


def _write_random_checkpoints(logdir, seeds=(0, 1)):
    """Randomly initialized continuous actors in the trainer's file layout."""
    from jaxmarl.wrappers.baselines import save_params

    env = make_env("capture", continuous=True)
    width = max(env.observation_space(a).shape[0] for a in env.agents)
    actor = ContinuousActor(action_dim=2, hidden_dim=128)
    logdir.mkdir(parents=True, exist_ok=True)
    k = 0
    for seed in seeds:
        for team in ("pred", "prey"):
            for objective in OBJECTIVE_TYPES:
                params = actor.init(jax.random.PRNGKey(k), jnp.zeros((1, width)))
                # Make the specialists move: a nonzero bias on the mean head.
                params["params"]["Dense_2"]["bias"] = jnp.asarray([0.6, -0.4]) * (k % 3 - 1)
                save_params(
                    params, str(continuous_checkpoint_path(logdir, objective, team, seed))
                )
                k += 1


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    logdir = tmp_path_factory.mktemp("ckpts")
    _write_random_checkpoints(logdir)
    return continuous_objective_dataset(
        n_eps=4, ckpt_seeds=(0, 1), rng0=0, num_steps=12, logdir=logdir
    )


def test_markov_state_dimension_and_content():
    env = make_env("risk", continuous=True)
    _, state = env.reset(jax.random.PRNGKey(0))
    x = np.asarray(markov_state(env, state))
    assert x.shape == (markov_state_dim(env),)
    assert x.shape[0] == 4 * 2 + 3 * 16 + 3 * 3 + 1 == 66
    assert len(MARKOV_STATE_FIELDS) == 7
    np.testing.assert_allclose(x[:4], np.asarray(state.p_pos[:2]).reshape(-1))
    np.testing.assert_allclose(x[4:8], 0.0)  # zero initial velocity
    assert x[-1] == 0.0  # time fraction at reset
    batched = markov_state(env, jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(1), 3))[1])
    assert batched.shape == (3, 66)


def test_dataset_matches_contract_and_replays_exactly(dataset):
    ds = dataset
    assert isinstance(ds, ContinuousTrajectoryDataset)
    n, horizon = 3 * 2 * 4, 12
    assert ds.state.shape == (n, horizon + 1, 66)
    assert ds.blue_observation.shape == (n, horizon + 1, 35)
    assert ds.red_observation.shape == (n, horizon + 1, 17)
    assert ds.blue_action.shape == ds.red_action.shape == (n, horizon, 2)
    assert ds.blue_action.dtype == ds.red_action.dtype == np.float32
    assert ds.blue_reward.shape == (n, horizon)
    for name in ("terminated_capture", "truncated_timeout", "valid_mask"):
        assert getattr(ds, name).dtype == bool and getattr(ds, name).shape == (n, horizon)
    assert ds.causal_context.shape == (n, horizon + 1, 0)
    assert ds.environment_seed.dtype == np.uint32 and ds.environment_seed.shape == (n, 2)
    assert (np.abs(ds.blue_action) <= 1.0).all() and (np.abs(ds.red_action) <= 1.0).all()
    assert np.isfinite(ds.state).all() and np.isfinite(ds.blue_reward).all()
    # Aliases for the legacy encoder/BC utilities.
    np.testing.assert_array_equal(ds.label, ds.objective_label)
    np.testing.assert_array_equal(ds.ckpt_seed, ds.checkpoint_seed)

    summary = validate_continuous_dataset(ds)
    assert summary["n_episodes"] == n and summary["horizon"] == horizon
    assert summary["replay"]["termination_flags_match_fraction"] == 1.0
    assert summary["replay"]["max_abs_error_state"] <= 1e-5
    assert summary["replay"]["max_abs_error_blue_reward"] <= 1e-5

    # Every episode ends with exactly one flag at its last valid transition.
    ends = ds.terminated_capture | ds.truncated_timeout
    assert (ends.sum(1) == 1).all()
    assert np.array_equal(np.argmax(ends, 1), ds.valid_length - 1)
    # Matched resets across the three objective types within a checkpoint.
    for seed in (0, 1):
        groups = [
            ds.environment_seed[(ds.checkpoint_seed == seed) & (ds.objective_label == lab)]
            for lab in range(3)
        ]
        assert all(np.array_equal(g, groups[0]) for g in groups)
    # Time fraction advances by 1/max_steps per valid transition.
    valid_rows = np.flatnonzero(ds.valid_length == horizon)
    if len(valid_rows):
        tf = ds.state[valid_rows[0], :, -1]
        np.testing.assert_allclose(np.diff(tf), 1.0 / 100.0, atol=1e-6)


def test_replay_detects_tampered_actions(dataset):
    ds = dataset.as_dict()
    tampered = dict(ds)
    tampered["blue_action"] = ds["blue_action"].copy()
    tampered["blue_action"][0, 0] = np.clip(tampered["blue_action"][0, 0] + 0.5, -1, 1)
    out = replay_episodes(tampered, indices=[0])
    assert out["max_abs_error_state"] > 1e-4
    with pytest.raises(ValueError, match="exact replay failed"):
        validate_continuous_dataset(tampered, replay_indices=[0])


def test_validator_rejects_inconsistent_flags(dataset):
    bad = dataset.as_dict()
    bad["truncated_timeout"] = bad["truncated_timeout"].copy()
    bad["truncated_timeout"][0, 0] = True
    with pytest.raises(ValueError):
        validate_continuous_dataset(bad, replay_indices=[0])


def test_coverage_and_family_summaries_run(dataset):
    cov = state_action_coverage(dataset)
    assert cov["n_valid_transitions"] == int(dataset.valid_mask.sum())
    assert 0.0 <= cov["blue_action"]["either_axis_saturated_fraction"] <= 1.0
    assert 0.0 < cov["prey_grid_cells_visited_fraction"] <= 1.0
    fam = family_behavior_summary(dataset)
    assert set(fam["per_type"]) == set(OBJECTIVE_TYPES)
    assert fam["behaviour_probe"]["chance"] == pytest.approx(1 / 3)
    assert len(fam["behaviour_probe"]["folds"]) == 2
