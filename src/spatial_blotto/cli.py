#!/usr/bin/env python3
"""Run mathematical Blotto controllers and export replay/transition artifacts."""

from __future__ import annotations

import argparse
import json
from numbers import Integral
from pathlib import Path

import jax

from spatial_blotto import SpatialBlotto
from spatial_blotto.controllers import MAX_EXACT_TEAM_SIZE, POLICIES, make_controller
from spatial_blotto.rendering import write_replay
from spatial_blotto.rollout import episode_payload, rollout_episode, save_trajectory


def allocation(text):
    try:
        values = tuple(int(x) for x in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "use three integer counts, e.g. 2,1,0"
        ) from exc
    if len(values) != 3 or min(values) < 0:
        raise argparse.ArgumentTypeError("use three nonnegative counts, e.g. 2,1,0")
    return values


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/blotto/replay.html")
    )
    parser.add_argument(
        "--json", type=Path, dest="json_path", help="optional full replay JSON"
    )
    parser.add_argument(
        "--trajectory", type=Path, help="optional aligned transition NPZ"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--team-size", type=int, default=3)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--max-speed", type=float, default=1.0)
    parser.add_argument("--zone-radius", type=float, default=0.22)
    parser.add_argument(
        "--scenario",
        choices=("cyclic", "balanced", "reactive", "rotating"),
        default="cyclic",
        help="cyclic reproduces the fixed slide allocations; rotating changes allocations over time",
    )
    parser.add_argument("--red-policy", choices=POLICIES)
    parser.add_argument("--blue-policy", choices=POLICIES)
    parser.add_argument("--red-allocation", type=allocation)
    parser.add_argument("--blue-allocation", type=allocation)
    parser.add_argument(
        "--period", type=int, default=40, help="steps between cyclic-policy rotations"
    )
    parser.add_argument(
        "--reward-mode", choices=("ownership", "zero_sum"), default="ownership"
    )
    return parser


def simulate(
    seed=0,
    steps=200,
    scenario="cyclic",
    reward_mode="ownership",
    *,
    team_size=3,
    red_policy=None,
    blue_policy=None,
    red_allocation=None,
    blue_allocation=None,
    period=40,
    dt=0.1,
    max_speed=1.0,
    zone_radius=0.22,
):
    """Return a replay payload and the unreset, aligned JAX episode."""
    if (
        isinstance(seed, bool)
        or not isinstance(seed, Integral)
        or not 0 <= seed <= 2**32 - 1
    ):
        raise ValueError("seed must be an integer between 0 and 4294967295")
    # Reject unsupported demo sizes before the environment constructs per-agent
    # observation indices. The environment itself supports larger teams.
    if (
        isinstance(team_size, bool)
        or not isinstance(team_size, Integral)
        or not 1 <= team_size <= MAX_EXACT_TEAM_SIZE
    ):
        raise ValueError(
            f"demo team_size must be an integer between 1 and {MAX_EXACT_TEAM_SIZE}"
        )
    if scenario not in ("cyclic", "balanced", "reactive", "rotating"):
        raise ValueError("unknown scenario")
    env = SpatialBlotto(
        team_size=team_size,
        max_steps=steps,
        reward_mode=reward_mode,
        dt=dt,
        max_speed=max_speed,
        zone_radius=zone_radius,
    )
    presets = {
        "cyclic": ("fixed", "fixed"),
        "balanced": ("balanced", "balanced"),
        "reactive": ("reactive", "balanced"),
        "rotating": ("cyclic", "cyclic"),
    }
    policies = (red_policy or presets[scenario][0], blue_policy or presets[scenario][1])
    # Named fixed example is scaled to the selected budget; 3v3 gives (2,1,0)/(0,2,1).
    defaults = (
        ((team_size - 1, 1, 0), (0, team_size - 1, 1))
        if team_size > 1
        else ((1, 0, 0), (0, 1, 0))
    )
    configs, controllers = {}, []
    for i, (team, policy, counts) in enumerate(
        zip(("red", "blue"), policies, (red_allocation, blue_allocation), strict=True)
    ):
        if counts is None and policy == "fixed":
            counts = defaults[i]
        controller = make_controller(
            env, team, policy, allocation=counts, period=period
        )
        controllers.append(controller)
        configs[team] = {
            "name": policy,
            "allocation": list(controller.allocation)
            if policy in ("fixed", "cyclic")
            else None,
            "period": period if policy == "cyclic" else None,
        }
    episode = jax.jit(lambda key: rollout_episode(env, *controllers, key))(
        jax.random.PRNGKey(int(seed))
    )
    return episode_payload(
        env, episode, seed=int(seed), controllers=configs, scenario=scenario
    ), episode


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    outputs = [
        p.expanduser().resolve()
        for p in (args.output, args.json_path, args.trajectory)
        if p is not None
    ]
    if len(set(outputs)) != len(outputs):
        parser.error("HTML, JSON and trajectory output paths must be different")
    if args.trajectory is not None and args.trajectory.suffix != ".npz":
        parser.error("--trajectory must end in .npz")
    try:
        replay, episode = simulate(
            args.seed,
            args.steps,
            args.scenario,
            args.reward_mode,
            team_size=args.team_size,
            red_policy=args.red_policy,
            blue_policy=args.blue_policy,
            red_allocation=args.red_allocation,
            blue_allocation=args.blue_allocation,
            period=args.period,
            dt=args.dt,
            max_speed=args.max_speed,
            zone_radius=args.zone_radius,
        )
    except ValueError as exc:
        parser.error(str(exc))
    output = write_replay(args.output, replay)
    if args.json_path is not None:
        target = args.json_path.expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(replay, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    if args.trajectory is not None:
        metadata = {k: v for k, v in replay.items() if k != "frames"}
        save_trajectory(
            args.trajectory,
            episode,
            metadata,
            env=SpatialBlotto(
                team_size=args.team_size,
                max_steps=args.steps,
                dt=args.dt,
                max_speed=args.max_speed,
                zone_radius=args.zone_radius,
                reward_mode=args.reward_mode,
            ),
        )
    print(f"Saved {output}")
    print(json.dumps(replay["summary"], allow_nan=False))
    print(
        "Mathematical baseline controllers; no policy training or opponent-strategy inference."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
