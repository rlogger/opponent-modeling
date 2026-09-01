"""Causal split and metric checks for behaviour cloning."""

import numpy as np
import pytest

from mopa.bc import (
    BCPolicy,
    bc_metrics_from_logits,
    evaluate_bc,
    fit_bc,
    train_eval_bc_metrics,
)


def test_bc_uses_explicit_manifest_validation_mask():
    rng = np.random.default_rng(0)
    states = rng.normal(size=(30, 4)).astype(np.float32)
    actions = (states[:, 0] > 0).astype(np.int32)
    episodes = np.repeat(np.arange(10), 3).astype(np.int32)
    validation = episodes >= 8

    result = train_eval_bc_metrics(
        states,
        actions,
        episodes,
        rng_seed=7,
        steps=4,
        validation_mask=validation,
    )

    assert result["n_train"] == 24
    assert result["n_val"] == 6
    assert 0.0 <= result["accuracy"] <= 1.0
    assert result["nll"] >= 0.0


def test_bc_rejects_empty_or_misaligned_manifest_split():
    states = np.zeros((6, 2), dtype=np.float32)
    actions = np.zeros(6, dtype=np.int32)
    episodes = np.arange(6, dtype=np.int32)

    with pytest.raises(ValueError, match="align"):
        train_eval_bc_metrics(
            states,
            actions,
            episodes,
            rng_seed=0,
            steps=1,
            validation_mask=np.zeros(5, dtype=bool),
        )
    with pytest.raises(ValueError, match="both train and validation"):
        train_eval_bc_metrics(
            states,
            actions,
            episodes,
            rng_seed=0,
            steps=1,
            validation_mask=np.zeros(6, dtype=bool),
        )


def test_bc_policy_is_reusable_and_pickle_free(tmp_path):
    rng = np.random.default_rng(2)
    features = rng.normal(size=(40, 4)).astype(np.float32)
    actions = (features[:, 0] > 0.0).astype(np.int32)
    policy = fit_bc(
        features,
        actions,
        rng_seed=3,
        steps=5,
        n_actions=2,
        metadata={"arm": "vanilla", "fold": 1},
    )

    query = features[:12]
    logits = policy.logits(query)
    probabilities = policy.probabilities(query)
    assert logits.shape == probabilities.shape == (12, 2)
    np.testing.assert_allclose(probabilities.sum(axis=-1), 1.0, atol=1e-6)
    np.testing.assert_array_equal(policy.greedy_action(query), logits.argmax(-1))
    np.testing.assert_array_equal(
        policy.sample_action(query, 17), policy.sample_action(query, 17)
    )

    artifact = tmp_path / "vanilla.npz"
    policy.save(artifact)
    restored = BCPolicy.load(artifact)
    np.testing.assert_allclose(restored.logits(query), logits, atol=1e-6)
    np.testing.assert_allclose(restored.probabilities(query), probabilities, atol=1e-6)
    assert dict(restored.metadata)["arm"] == "vanilla"
    assert restored.input_size == 4
    assert restored.n_actions == 2

    metrics = evaluate_bc(restored, query, actions[:12])
    assert metrics["n_samples"] == 12
    assert 0.0 <= metrics["action_balanced_accuracy"] <= 1.0


def test_bc_episode_then_strategy_macro_metrics():
    logits = np.asarray(
        [
            [3.0, 0.0],
            [0.0, 3.0],
            [0.0, 3.0],
            [0.0, 3.0],
            [3.0, 0.0],
            [3.0, 0.0],
        ],
        dtype=np.float32,
    )
    actions = np.asarray([0, 0, 1, 1, 1, 0], dtype=np.int32)
    episodes = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int32)
    strategies = np.asarray([0, 0, 0, 0, 1, 1], dtype=np.int32)

    metrics = bc_metrics_from_logits(
        logits,
        actions,
        episode_ids=episodes,
        strategy_labels=strategies,
    )

    assert metrics["accuracy"] == pytest.approx(4 / 6)
    assert metrics["action_balanced_accuracy"] == pytest.approx(2 / 3)
    assert metrics["episode_macro_accuracy"] == pytest.approx(2 / 3)
    assert metrics["episode_strategy_macro_accuracy"] == pytest.approx(0.625)
    assert metrics["n_episodes"] == 3
    assert metrics["n_strategies"] == 2
    assert metrics["per_strategy"]["0"]["n_episodes"] == 2
    assert metrics["episode_strategy_macro_nll"] >= 0.0

    mixed = strategies.copy()
    mixed[1] = 1
    with pytest.raises(ValueError, match="exactly one strategy"):
        bc_metrics_from_logits(
            logits,
            actions,
            episode_ids=episodes,
            strategy_labels=mixed,
        )
