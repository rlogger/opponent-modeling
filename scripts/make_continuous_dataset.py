#!/usr/bin/env python3
"""Generate, validate, and summarize the matched continuous trajectory dataset.

Gate 1 deliverable (handoff): fresh rollouts of the three continuous predator
specialist families against the fixed ``capture`` prey family, matched reset
keys across objective types, the full data contract, exact simulator replay,
action-saturation / state-coverage diagnostics, and a behavioural
distinguishability probe. Writes ``dataset.npz``, ``dataset.manifest.json``,
and ``report.json`` under ``--artifact-dir``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mopa.continuous_data import (  # noqa: E402
    DEFAULT_CONTINUOUS_LOGDIR,
    MARKOV_STATE_FIELDS,
    OBJECTIVE_TYPES,
    continuous_checkpoint_path,
    continuous_objective_dataset,
    family_behavior_summary,
    state_action_coverage,
    validate_continuous_dataset,
)
from mopa.manifest import (  # noqa: E402
    file_sha256,
    git_dirty,
    git_sha,
    package_versions,
)


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def checkpoint_manifest(logdir: Path, seeds: tuple[int, ...], prey_type: str) -> list[dict]:
    rows = []
    sources = [(o, "pred", s) for s in seeds for o in OBJECTIVE_TYPES] + [
        (prey_type, "prey", s) for s in seeds
    ]
    for objective, team, seed in sources:
        path = continuous_checkpoint_path(logdir, objective, team, seed)
        if not path.is_file():
            raise FileNotFoundError(f"missing continuous checkpoint: {path}")
        rows.append(
            {
                "objective": objective,
                "team": team,
                "seed": int(seed),
                "path": str(path),
                "sha256": file_sha256(path),
            }
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-dir", type=Path, default=Path("artifacts/continuous"))
    p.add_argument("--logdir", type=Path, default=DEFAULT_CONTINUOUS_LOGDIR)
    p.add_argument("--n-eps", type=int, default=200)
    p.add_argument("--num-steps", type=int, default=100)
    p.add_argument("--rollout-seed", type=int, default=0)
    p.add_argument("--ckpt-seeds", type=_parse_ints, default=(0, 1, 2))
    p.add_argument("--prey-type", default="capture")
    p.add_argument(
        "--sampled",
        action="store_true",
        help="declared arm: sample specialist actions instead of the deterministic mean",
    )
    p.add_argument("--replay-episodes", type=int, default=0, help="0 = replay all")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.n_eps < 1 or args.num_steps < 1:
        raise ValueError("--n-eps and --num-steps must be positive")
    checkpoints = checkpoint_manifest(args.logdir, args.ckpt_seeds, args.prey_type)
    ds = continuous_objective_dataset(
        n_eps=args.n_eps,
        ckpt_seeds=args.ckpt_seeds,
        rng0=args.rollout_seed,
        num_steps=args.num_steps,
        logdir=args.logdir,
        prey_type=args.prey_type,
        deterministic=not args.sampled,
    )
    data = ds.as_dict()
    replay_indices = None
    if args.replay_episodes:
        replay_indices = np.arange(min(args.replay_episodes, len(ds.objective_label)))
    summary = validate_continuous_dataset(data, replay_indices=replay_indices)
    coverage = state_action_coverage(data)
    behaviour = family_behavior_summary(data)

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = args.artifact_dir / "dataset.npz"
    np.savez_compressed(dataset_path, **data)
    manifest = {
        "schema_version": 1,
        "git_sha": git_sha(_ROOT),
        "git_dirty": git_dirty(_ROOT),
        "dependencies": package_versions(),
        "rollout": {
            "n_eps": args.n_eps,
            "num_steps": args.num_steps,
            "rollout_seed": args.rollout_seed,
            "checkpoint_seeds": list(args.ckpt_seeds),
            "prey_type": args.prey_type,
            "policy_action": "sampled" if args.sampled else "deterministic_tanh_mean",
            "num_adversaries": 1,
            "action_contract": "(a_x, a_y) in [-1, 1]^2 -> to_mpe_action at env boundary",
            "markov_state_fields": list(MARKOV_STATE_FIELDS),
        },
        "source_checkpoints": [
            {k: row[k] for k in ("objective", "team", "seed", "sha256")} for row in checkpoints
        ],
    }
    manifest_path = args.artifact_dir / "dataset.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    report = {
        "dataset_sha256": file_sha256(dataset_path),
        "manifest_sha256": file_sha256(manifest_path),
        "summary": summary,
        "coverage": coverage,
        "behaviour": behaviour,
        "checkpoints": checkpoints,
        "git_sha": manifest["git_sha"],
        "git_dirty": manifest["git_dirty"],
    }
    report_path = args.artifact_dir / "report.json"
    report_path.write_text(json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n")
    print(json.dumps(_jsonable({"summary": summary, "behaviour": behaviour["per_type"], "probe": behaviour["behaviour_probe"], "coverage": coverage}), indent=1))
    print(f"Wrote {dataset_path}, {manifest_path}, {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
