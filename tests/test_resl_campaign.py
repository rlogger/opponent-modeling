"""Source-bound five-arm campaign and intrinsic/administrative horizon contracts."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture(scope="module")
def campaign():
    path = Path(__file__).resolve().parents[1] / "scripts/run_resl_campaign.py"
    spec = importlib.util.spec_from_file_location("resl_campaign_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def protocol(stage="pilot"):
    main = stage == "main"
    cfg = dict(arms=["implicit", "bc", "0s", "ppo", "ppo_z"], encoder_steps=1500 if main else 2,
        offline_updates=2000 if main else 2, rounds=6 if main else 1, updates_per_round=1000 if main else 2,
        transitions_per_training_group=600 if main else 5, batch_episodes=1,
        eval_episodes_per_objective=24 if main else 4, evaluation_batch_episodes=24 if main else 4,
        ppo_epochs=4, ppo_minibatch=128, terminal_contract="finite_100_step_v1",
        training_checkpoints=[0, 1] if main else [0], evaluation_checkpoint=2 if main else 1,
        planner_profile="gate4", horizon=3, population_size=512, policy_prior_samples=24,
        num_elites=64, mppi_iterations=6,
        opponent_sampling="random next eligible training specialist at each episode under exact per-type transition quotas")
    return {f"control_{stage}": cfg, f"{stage}_seeds": [0, 1, 2] if main else [11, 22]}


def trace(terminal_time=False, capture=False):
    state = np.zeros((1, 3, 66), np.float32)
    state[0, :, 65] = [0.98, 0.99, 1.] if terminal_time else [0., .01, .02]
    return {"state": state, "valid_mask": np.ones((1, 2), bool),
            "terminated_capture": np.array([[False, capture]]),
            "truncated_timeout": np.array([[False, not capture]])}


@pytest.mark.parametrize("capture,timeout,expected", [(False, False, 2.7), (False, True, 0.), (True, False, 0.)])
def test_analytic_finite_horizon_and_administrative_cut_targets(campaign, capture, timeout, expected):
    tr = trace(timeout, capture)
    original = {k: v.copy() for k, v in tr.items()}
    target = campaign.training_trace(tr, "finite_100_step_v1")
    advantage, returns = campaign.historical.advantage_targets(
        np.zeros((1, 2)), np.array([[1., 2., 3.]]), target["terminated_capture"], tr["valid_mask"],
        discount=.9, gae_lambda=1.)
    assert returns[0, 1] == pytest.approx(expected)
    assert returns[0, 0] == pytest.approx(.9 * expected, abs=1e-6)
    assert np.isfinite(advantage).all()
    for key, value in original.items():
        np.testing.assert_array_equal(tr[key], value)
    assert not np.any(target["terminated_capture"] & target["truncated_timeout"])


def test_historical_timeout_target_is_explicit_and_unchanged(campaign):
    np.testing.assert_array_equal(campaign.value_terminals(trace(True), "capture_only_bootstrap_timeouts"), False)


@pytest.mark.parametrize("corruption", ["sixth_arm", "main_budget", "duplicate_seed", "pilot_test", "training_overlap", "weaken_planner"])
def test_protocol_rejects_silent_changes(campaign, corruption):
    stage = "main" if corruption == "main_budget" else "pilot"
    value = protocol(stage)
    cfg = value[f"control_{stage}"]
    if corruption == "sixth_arm":
        cfg["arms"].append("history")
    elif corruption == "main_budget":
        cfg["offline_updates"] = 1
    elif corruption == "duplicate_seed":
        value["pilot_seeds"] = [11, 11]
    elif corruption == "pilot_test":
        cfg["evaluation_checkpoint"] = 2
    elif corruption == "training_overlap":
        cfg["training_checkpoints"] = [0, 1]
    else:
        cfg["population_size"] = 8
    with pytest.raises(ValueError):
        campaign.protocol_configuration(value, stage)


@pytest.mark.parametrize("stage", ["pilot", "main"])
def test_documented_five_arm_budget_is_accepted(campaign, stage):
    cfg, seeds = campaign.protocol_configuration(protocol(stage), stage)
    assert len(cfg["arms"]) == 5
    assert len(seeds) == (3 if stage == "main" else 2)
    if stage == "main":
        assert cfg["rounds"] * cfg["transitions_per_training_group"] * 3 * len(cfg["training_checkpoints"]) == 21_600


def test_world_representation_toggle_retains_global_inputs_and_no_opponent_context(campaign, tmp_path):
    value = protocol()
    cfg = value["control_pilot"]
    cfg.update(study="world_representation", arms=list(campaign.WORLD_REPRESENTATION_ARMS))
    with pytest.raises(ValueError, match="global-state inputs"):
        campaign.protocol_configuration(value, "pilot")
    cfg["world_encoders"] = {"implicit": "identity", "implicit_mlp": "mlp"}
    campaign.protocol_configuration(value, "pilot")
    binding = {"configuration": cfg, "source_hashes": campaign.source_hashes()}
    shared = campaign.prepare_shared(tmp_path, 11, {}, np.zeros(66), np.ones(66), binding)
    assert shared["models"] == shared["contexts"] == {}
    assert not list(tmp_path.rglob("*.msgpack"))
    for arm, encoder in cfg["world_encoders"].items():
        model = campaign.controller_configuration(arm, 0)
        assert model["opponent_mode"] == "implicit" and model["context_dim"] == 0
        assert model["encoder"]["type"] == encoder
        if encoder == "mlp":
            assert model["encoder"]["normalize_inputs"] is True
        assert model["world_model"]["hidden_dim"] == 128
        assert model["tdmpc2"]["population_size"] == 512


def test_collection_randomizes_episode_specialists_and_keeps_exact_quotas(campaign, monkeypatch, tmp_path):
    cfg = protocol()["control_pilot"]
    observed = []
    monkeypatch.setattr(campaign.historical, "benchmark_env", lambda: object())
    monkeypatch.setattr(campaign.historical, "Controller", lambda *a, **k: SimpleNamespace(is_ppo=False, samples={}))

    def episodes(env, red, controller, reset, step_keys, **kwargs):
        del env, controller, step_keys
        assert len(reset) == 1
        observed.append(red)
        count = min(2, kwargs["max_transitions"])
        tr = trace()
        tr["valid_mask"][0, count:] = False
        tr["valid_length"] = np.array([count])
        tr["context"] = np.zeros((1, 2, 0), np.float32)
        tr["final_context"] = np.zeros((1, 0), np.float32)
        return {"transitions": tr, "controller_seconds_per_batch": np.array([0.1, .01])}

    monkeypatch.setattr(campaign.historical, "run_matched_episodes", episodes)
    destination = tmp_path / "arm" / "round_000" / "attempt_000"
    destination.mkdir(parents=True)
    params = {(0, label): label for label in range(3)}
    _, batches, records = campaign.collect(cfg, object(), None, params, 11, 0, destination)
    assert not batches and len(records) == 9
    assert sum(r["valid_transitions"] for r in records) == 15
    assert observed != sorted(observed)
    for objective in campaign.OBJECTIVE_TYPES:
        assert sum(r["valid_transitions"] for r in records if r["objective"] == objective) == 5
    for record in records:
        assert (destination.parents[1] / record["file"]).is_file()


def test_saved_artifact_changes_are_rejected(campaign, tmp_path):
    path = tmp_path / "artifact"
    path.write_bytes(b"original")
    hashes = {path.name: campaign.file_sha256(path)}
    campaign.verify_files(tmp_path, hashes)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed artifact"):
        campaign.verify_files(tmp_path, hashes)


def test_representation_controls_are_a_separate_explicit_protocol(campaign):
    value = protocol()
    cfg = value["control_pilot"]
    cfg["study"] = "representation"
    cfg["arms"] = list(campaign.REPRESENTATION_ARMS)
    with pytest.raises(ValueError, match="predictor configuration"):
        campaign.protocol_configuration(value, "pilot")
    cfg["predictor_config"] = dict(method="recurrent_vae", feature_schema="expert17_v1",
        history=8, latent_dim=8, hid=64, batch=128, learning_rate=.001, beta=1.,
        free_bits=.2, sample_training=True, objective="past_next_action")
    from dataclasses import asdict, replace

    selected = campaign.CausalOpponentConfig(**{**cfg["predictor_config"], "steps": cfg["encoder_steps"], "max_history": 4})
    cfg["predictor_config"]["max_history"] = 4
    with pytest.raises(ValueError, match="per-arm predictor configurations"):
        campaign.protocol_configuration(value, "pilot")
    history, gap = campaign.match_encoder_capacity(replace(selected, method="deterministic_history",
        beta=0., sample_training=False), replace(selected, method="recurrent_vae", encoder_hid=None))
    cfg["predictor_configurations"] = {
        "bc": asdict(replace(selected, method="bc", history=0, max_history=None,
            encoder_hid=None, beta=0., sample_training=False)),
        "history": asdict(history), "causal_vae": asdict(selected)}
    cfg["history_encoder_parameter_gap"] = gap
    campaign.protocol_configuration(value, "pilot")
    assert cfg["predictor_configurations"]["history"]["encoder_hid"] == 101
    assert cfg["predictor_configurations"]["bc"]["max_history"] is None
    cfg["predictor_config"]["objective"] = "target_inclusive_reconstruction"
    with pytest.raises(ValueError, match="past-only"):
        campaign.protocol_configuration(value, "pilot")


def test_ppo_key_separation_checkpoint_resume_and_interrupted_evaluation(campaign, monkeypatch, tmp_path):
    """Mock updates isolate persistence/RNG; this is not a controller fit."""
    import jax

    cfg = protocol()["control_pilot"]
    cfg.update(offline_updates=1, rounds=1)
    binding = {"configuration": cfg, "source_hashes": {"code": "fixed"}}
    monkeypatch.setattr(campaign, "source_hashes", lambda: {"code": "fixed"})
    agent = SimpleNamespace(context_dim=0, hidden=128, learning_rate=.0003, clip_epsilon=.2,
        entropy_coefficient=.01, value_coefficient=.5, max_grad_norm=.5)
    monkeypatch.setattr(campaign, "create_response", lambda *a, **k: agent)
    keys = []

    def warm(agent, batch, key, **kwargs):
        del batch, kwargs
        keys.append(np.asarray(key))
        return agent, {"loss": 1.}

    monkeypatch.setattr(campaign, "warm_start_response", warm)
    monkeypatch.setattr(campaign, "update_response", warm)
    monkeypatch.setattr(campaign.flax.serialization, "to_bytes", lambda _: b"optimizer and params")
    monkeypatch.setattr(campaign.ResponsePPO, "load", lambda _: agent)
    monkeypatch.setattr(campaign.historical, "benchmark_env", lambda: object())
    monkeypatch.setattr(campaign.historical, "Controller", lambda *a, **k: object())

    def collect(cfg, agent, opponent, params, seed, round_index, directory):
        del agent, opponent, params, seed, round_index
        records = []
        for objective in campaign.OBJECTIVE_TYPES:
            path = directory / f"{objective}.npz"
            path.write_bytes(b"mock trace only")
            records.append({"file": str(path.relative_to(directory.parents[1])),
                "sha256": campaign.file_sha256(path), "valid_transitions": cfg["transitions_per_training_group"],
                "checkpoint": 0, "objective": objective})
        return [], [{"unused": np.zeros((1,))}], records

    monkeypatch.setattr(campaign, "collect", collect)
    calls = [0]

    def evaluate(*args, **kwargs):
        del args, kwargs
        calls[0] += 1
        if calls[0] == 2:
            raise RuntimeError("simulated evaluation interruption")
        return {"transitions": trace(), "blue_return": np.zeros(1), "captured": np.zeros(1),
                "resources_collected": np.zeros(1)}

    monkeypatch.setattr(campaign.historical, "run_matched_episodes", evaluate)
    data = {**trace(), "blue_action": np.zeros((1, 2, 2)), "blue_reward": np.zeros((1, 2))}
    shared = {"models": {}, "contexts": {}, "manifest_sha256": "shared"}
    params = {(1, label): object() for label in range(3)}
    args = (tmp_path, 11, "ppo", data, np.zeros(66), np.ones(66), shared, params, binding)
    with pytest.raises(RuntimeError, match="interruption"):
        campaign.run_arm(*args)
    directory = tmp_path / "seed_11" / "ppo"
    snapshots = {p.name: p.read_bytes() for p in directory.glob("checkpoint_*.msgpack")}
    assert len(snapshots) == 2 and len(keys) == 2
    assert not np.array_equal(keys[0], keys[1])
    parent_key, offline_key = jax.random.split(jax.random.PRNGKey(10011))
    _, online_key = jax.random.split(parent_key)
    np.testing.assert_array_equal(keys[0], offline_key)
    np.testing.assert_array_equal(keys[1], online_key)
    campaign.run_arm(*args)
    assert len(keys) == 2  # Resume did not refit the completed warm-start/round.
    assert (directory / "evaluation_attempt_000/capture.npz").is_file()
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["evaluation_directory"] == "evaluation_attempt_001"
    assert {p.name: p.read_bytes() for p in directory.glob("checkpoint_*.msgpack")} == snapshots
