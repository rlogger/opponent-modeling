"""Validation, deterministic selection, and plotting for trajectory bundles."""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from mopa.types import ObjectiveDataset, ObjectiveObservationDataset

LABEL_NAMES = {
    0: "prey-seeking (capture)",
    1: "risk-averse",
    2: "curious",
}
DISPLAY_TITLES = {
    0: "Prey-seeking",
    1: "Risk-averse",
    2: "Curious",
}
COLORS = {
    0: "#0d6e7a",
    1: "#c45c14",
    2: "#5b4fa0",
}
AXIS_SEMANTICS = {
    "prey_pos": ["episode", "time_including_initial", "xy"],
    "pred_pos": ["episode", "time_including_initial", "predator", "xy"],
    "pred_obs": [
        "episode",
        "time_including_initial",
        "predator",
        "observation_feature",
    ],
    "lava_pos": ["episode", "lava_region", "xy"],
    "lava_rad": ["episode", "lava_region"],
    "prey_act": ["episode", "action_time"],
    "pred_act": ["episode", "action_time", "predator"],
    "capture_t": ["episode"],
    "captured": ["episode"],
    "survival_time": ["episode"],
    "pred_lava_steps": ["episode"],
    "prey_lava_steps": ["episode"],
    "resources_collected": ["episode"],
    "pred_coverage": ["episode"],
    "label": ["episode"],
    "ckpt_seed": ["episode"],
    "env_seed": ["episode", "jax_key_word"],
    "valid_length": ["episode"],
}


def _arrays(
    dataset: ObjectiveDataset | Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    if isinstance(dataset, ObjectiveDataset):
        return dataset.as_dict()
    names = list(ObjectiveDataset.__dataclass_fields__)
    if "pred_obs" in dataset:
        names.append("pred_obs")
    return {
        name: np.asarray(dataset[name])
        for name in names
    }


def load_objective_dataset(path: Path | str) -> ObjectiveDataset:
    """Load the exact lean schema without pickle-backed arrays."""
    with np.load(path, allow_pickle=False) as raw:
        missing = set(ObjectiveDataset.__dataclass_fields__) - set(raw.files)
        if missing:
            raise ValueError(f"dataset is missing fields: {sorted(missing)}")
        values: dict[str, np.ndarray] = {
            name: np.asarray(raw[name])
            for name in ObjectiveDataset.__dataclass_fields__
        }
        if "pred_obs" in raw.files:
            values["pred_obs"] = np.asarray(raw["pred_obs"])
    if "pred_obs" in values:
        return ObjectiveObservationDataset(**values)
    return ObjectiveDataset(**values)


def matched_groups(
    dataset: ObjectiveDataset | Mapping[str, np.ndarray],
) -> dict[tuple[int, int, int], dict[int, int]]:
    """Index episodes by checkpoint/reset key and hidden-objective label."""
    data = _arrays(dataset)
    groups: dict[tuple[int, int, int], dict[int, int]] = {}
    for index, (seed, key, label) in enumerate(
        zip(data["ckpt_seed"], data["env_seed"], data["label"])
    ):
        group_key = (int(seed), int(key[0]), int(key[1]))
        objective = int(label)
        if objective in groups.setdefault(group_key, {}):
            raise ValueError(
                f"duplicate label {objective} in matched group {group_key}"
            )
        groups[group_key][objective] = index
    return groups


def validate_objective_dataset(
    dataset: ObjectiveDataset | Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Validate shapes, capture padding, and controlled reset matching."""
    data = _arrays(dataset)
    n = int(data["label"].shape[0])
    if n == 0:
        raise ValueError("dataset must contain at least one episode")
    if data["prey_act"].ndim != 2:
        raise ValueError("prey_act must have shape (episode, time)")
    horizon = int(data["prey_act"].shape[1])
    if data["pred_act"].ndim != 3:
        raise ValueError("pred_act must have shape (episode, time, predator)")
    predators = int(data["pred_act"].shape[2])

    expected = {
        "prey_pos": (n, horizon + 1, 2),
        "pred_pos": (n, horizon + 1, predators, 2),
        "prey_act": (n, horizon),
        "pred_act": (n, horizon, predators),
        "env_seed": (n, 2),
    }
    if "pred_obs" in data:
        pred_obs = data["pred_obs"]
        if pred_obs.ndim != 4 or pred_obs.shape[-1] < 1:
            raise ValueError(
                "pred_obs must have shape "
                "(episode, time_including_initial, predator, observation_feature)"
            )
        expected["pred_obs"] = (
            n,
            horizon + 1,
            predators,
            int(pred_obs.shape[-1]),
        )
    for name in (
        "capture_t",
        "captured",
        "survival_time",
        "pred_lava_steps",
        "prey_lava_steps",
        "resources_collected",
        "pred_coverage",
        "label",
        "ckpt_seed",
        "valid_length",
    ):
        expected[name] = (n,)
    for name, shape in expected.items():
        if data[name].shape != shape:
            raise ValueError(f"{name} has shape {data[name].shape}; expected {shape}")
    if data["lava_pos"].ndim != 3 or data["lava_pos"].shape[:1] != (n,):
        raise ValueError("lava_pos must have shape (episode, lava_region, 2)")
    if data["lava_pos"].shape[-1:] != (2,):
        raise ValueError("lava_pos must have shape (episode, lava_region, 2)")
    if data["lava_rad"].shape != data["lava_pos"].shape[:-1]:
        raise ValueError("lava_rad must align with lava_pos")

    float_fields = {
        "prey_pos",
        "pred_pos",
        "pred_obs",
        "lava_pos",
        "lava_rad",
        "survival_time",
        "pred_lava_steps",
        "prey_lava_steps",
        "resources_collected",
        "pred_coverage",
    }
    integer_fields = {
        "prey_act",
        "pred_act",
        "capture_t",
        "captured",
        "label",
        "ckpt_seed",
        "valid_length",
    }
    for name in float_fields & data.keys():
        if data[name].dtype != np.float32:
            raise ValueError(f"{name} must use float32, found {data[name].dtype}")
    for name in integer_fields:
        if data[name].dtype != np.int32:
            raise ValueError(f"{name} must use int32, found {data[name].dtype}")
    if data["env_seed"].dtype != np.uint32:
        raise ValueError(f"env_seed must use uint32, found {data['env_seed'].dtype}")

    for name in (
        "prey_pos",
        "pred_pos",
        "pred_obs",
        "lava_pos",
        "lava_rad",
        "survival_time",
        "pred_lava_steps",
        "prey_lava_steps",
        "resources_collected",
        "pred_coverage",
    ):
        if name not in data:
            continue
        if not np.isfinite(data[name]).all():
            raise ValueError(f"{name} contains non-finite values")

    labels = set(np.unique(data["label"]).tolist())
    if labels != {0, 1, 2}:
        raise ValueError(f"expected labels 0, 1, and 2; found {sorted(labels)}")
    if not np.isin(data["captured"], [0, 1]).all():
        raise ValueError("captured must contain only 0 or 1")
    lengths = np.asarray(data["valid_length"], dtype=np.int64)
    if np.any(lengths < 1) or np.any(lengths > horizon):
        raise ValueError("valid_length must be between 1 and the action horizon")
    captured = data["captured"].astype(bool)
    if not np.array_equal(data["capture_t"][captured], lengths[captured]):
        raise ValueError("captured episodes must satisfy capture_t == valid_length")
    if np.any(data["capture_t"][~captured] != -1):
        raise ValueError("uncaptured episodes must have capture_t == -1")
    if np.any(lengths[~captured] != horizon):
        raise ValueError("uncaptured episodes must use the full horizon")
    if not np.array_equal(
        data["survival_time"], lengths.astype(np.float32, copy=False)
    ):
        raise ValueError("survival_time must equal valid_length")

    for index, length_value in enumerate(lengths):
        length = int(length_value)
        if length < horizon:
            if np.any(data["prey_act"][index, length:] != 0):
                raise ValueError(
                    f"prey actions after valid_length are nonzero at {index}"
                )
            if np.any(data["pred_act"][index, length:] != 0):
                raise ValueError(
                    f"predator actions after valid_length are nonzero at {index}"
                )
            if not np.all(
                data["prey_pos"][index, length:] == data["prey_pos"][index, length]
            ):
                raise ValueError(f"prey position tail is not frozen at {index}")
            if not np.all(
                data["pred_pos"][index, length:] == data["pred_pos"][index, length]
            ):
                raise ValueError(f"predator position tail is not frozen at {index}")
            if "pred_obs" in data and not np.all(
                data["pred_obs"][index, length:]
                == data["pred_obs"][index, length]
            ):
                raise ValueError(
                    f"predator observation tail is not frozen at {index}"
                )

    groups = matched_groups(data)
    for group_key, members in groups.items():
        if set(members) != {0, 1, 2}:
            raise ValueError(
                f"matched group {group_key} has labels {sorted(members)}; expected 0,1,2"
            )
        reference = members[0]
        for label in (1, 2):
            candidate = members[label]
            for name, value in (
                ("initial prey position", data["prey_pos"][:, 0]),
                ("initial predator position", data["pred_pos"][:, 0]),
                ("lava positions", data["lava_pos"]),
                ("lava radii", data["lava_rad"]),
            ):
                if not np.array_equal(value[reference], value[candidate]):
                    raise ValueError(
                        f"{name} differs across labels in matched group {group_key}"
                    )
            if "pred_obs" in data and not np.array_equal(
                data["pred_obs"][reference, 0], data["pred_obs"][candidate, 0]
            ):
                raise ValueError(
                    "initial predator observation differs across labels in "
                    f"matched group {group_key}"
                )

    summary = {
        "episodes": n,
        "horizon": horizon,
        "predators": predators,
        "matched_groups": len(groups),
        "episodes_per_label": {
            str(label): int(np.sum(data["label"] == label)) for label in range(3)
        },
    }
    if "pred_obs" in data:
        summary["predator_observation_dim"] = int(data["pred_obs"].shape[-1])
    return summary


def behavior_vector(
    dataset: ObjectiveDataset | Mapping[str, np.ndarray], index: int
) -> np.ndarray:
    """Bounded behavior summary used only to choose a display episode."""
    data = _arrays(dataset)
    horizon = int(data["prey_act"].shape[1])
    length = max(int(data["valid_length"][index]), 1)
    return np.asarray(
        [
            float(data["captured"][index]),
            length / horizon,
            float(data["pred_lava_steps"][index]) / length,
            float(data["pred_coverage"][index]) / 256.0,
        ],
        dtype=np.float64,
    )


def select_representative_group(
    dataset: ObjectiveDataset | Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Choose the matched group closest to all three classwise medians."""
    data = _arrays(dataset)
    groups = matched_groups(data)
    medians = {
        label: np.median(
            np.stack(
                [
                    behavior_vector(data, int(index))
                    for index in np.flatnonzero(data["label"] == label)
                ]
            ),
            axis=0,
        )
        for label in range(3)
    }
    scored: list[tuple[float, tuple[int, int, int], dict[int, int]]] = []
    for group_key, members in groups.items():
        if set(members) != {0, 1, 2}:
            continue
        score = sum(
            float(np.sum((behavior_vector(data, members[label]) - medians[label]) ** 2))
            for label in range(3)
        )
        scored.append((score, group_key, members))
    if not scored:
        raise ValueError("no complete matched objective group is available")
    score, group_key, members = min(scored, key=lambda item: (item[0], item[1]))
    return {
        "algorithm": "matched-class-median-v1",
        "formula": (
            "min_g sum_label ||v(g,label)-median_label(v)||_2^2; "
            "v=[captured,L/T,pred_lava_steps/max(L,1),pred_coverage/256]"
        ),
        "score": score,
        "group_key": {
            "checkpoint_seed": group_key[0],
            "environment_key": [group_key[1], group_key[2]],
        },
        "indices": {str(label): int(members[label]) for label in range(3)},
        "class_medians": {str(label): medians[label].tolist() for label in range(3)},
        "selected_vectors": {
            str(label): behavior_vector(data, members[label]).tolist()
            for label in range(3)
        },
    }


def reconstruct_resources(
    dataset: ObjectiveDataset | Mapping[str, np.ndarray],
    selection: Mapping[str, Any],
) -> np.ndarray:
    """Recreate the common resource layout from the recorded JAX reset key."""
    import jax.numpy as jnp

    from tag_objectives import SimpleTagObjectivesMPE

    data = _arrays(dataset)
    index = int(selection["indices"]["0"])
    key = jnp.asarray(data["env_seed"][index], dtype=jnp.uint32)
    env = SimpleTagObjectivesMPE(num_adversaries=int(data["pred_pos"].shape[2]))
    _, state = env.reset(key)
    initial = np.asarray(state.p_pos)
    predators = int(data["pred_pos"].shape[2])
    if not np.allclose(initial[:predators], data["pred_pos"][index, 0]):
        raise ValueError("reset key does not reproduce the selected predator start")
    if not np.allclose(initial[predators], data["prey_pos"][index, 0]):
        raise ValueError("reset key does not reproduce the selected prey start")
    if not np.allclose(np.asarray(state.lava_pos), data["lava_pos"][index]):
        raise ValueError("reset key does not reproduce the selected lava positions")
    if not np.allclose(np.asarray(state.lava_rad), data["lava_rad"][index]):
        raise ValueError("reset key does not reproduce the selected lava radii")
    return np.asarray(state.resource_pos, dtype=np.float32)


def resource_visibility(
    prey_pos: np.ndarray,
    resource_pos: np.ndarray,
    collect_radius: float,
    *,
    expected_collected: int | None = None,
) -> np.ndarray:
    """Return whether each resource is visible after every saved state."""
    prey = np.asarray(prey_pos, dtype=np.float32)
    resources = np.asarray(resource_pos, dtype=np.float32)
    if prey.ndim != 2 or prey.shape[1:] != (2,) or len(prey) < 1:
        raise ValueError("prey_pos must have shape (T, 2) with T >= 1")
    if resources.ndim != 2 or resources.shape[1:] != (2,):
        raise ValueError("resource_pos must have shape (R, 2)")
    if not np.isfinite(collect_radius) or collect_radius <= 0:
        raise ValueError("collect_radius must be positive and finite")

    collected = np.zeros((len(prey), len(resources)), dtype=bool)
    if len(prey) > 1 and len(resources) > 0:
        distance = np.linalg.norm(
            prey[1:, None, :] - resources[None, :, :], axis=-1
        )
        collected[1:] = np.logical_or.accumulate(
            distance < np.float32(collect_radius), axis=0
        )

    if expected_collected is not None:
        expected = int(expected_collected)
        if expected != expected_collected or not 0 <= expected <= len(resources):
            raise ValueError("expected_collected must be a valid integer count")
        actual = int(collected[-1].sum())
        if actual != expected:
            raise ValueError(
                f"resource trace implies {actual} collections; expected {expected}"
            )
    return ~collected


def render_representative_figure(
    dataset: ObjectiveDataset | Mapping[str, np.ndarray],
    selection: Mapping[str, Any],
    png_path: Path | str,
    pdf_path: Path | str,
    *,
    resource_pos: np.ndarray,
    prey_objective: str,
    collect_radius: float = 0.15,
    dpi: int = 220,
) -> dict[str, Any]:
    """Render one controlled, matched-reset trajectory for each objective."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle

    data = _arrays(dataset)
    png_path = Path(png_path)
    pdf_path = Path(pdf_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    prey_color = "#2f6db3"
    lava_color = "#c0392b"
    resource_color = "#2f8f4e"
    muted = "#5a5a5a"
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 5.0), sharex=True, sharey=True)
    line_point_counts: dict[str, int] = {}
    visible_resource_counts: dict[str, int] = {}

    selected_points = [np.asarray(resource_pos, dtype=np.float64)]
    selected_lava_extent = []
    for label in range(3):
        index = int(selection["indices"][str(label)])
        end = min(int(data["valid_length"][index]) + 1, data["prey_pos"].shape[1])
        selected_points.extend(
            [
                np.asarray(data["prey_pos"][index, :end], dtype=np.float64),
                np.asarray(data["pred_pos"][index, :end, 0], dtype=np.float64),
            ]
        )
        selected_lava_extent.append(
            np.abs(np.asarray(data["lava_pos"][index], dtype=np.float64))
            + np.asarray(data["lava_rad"][index], dtype=np.float64)[:, None]
        )
    max_abs = max(
        2.0,
        float(np.max(np.abs(np.concatenate(selected_points, axis=0)))),
        float(np.max(np.concatenate(selected_lava_extent, axis=0))),
    )
    axis_limit = max_abs + max(0.08, 0.03 * max_abs)
    ticks = np.arange(np.ceil(-axis_limit), np.floor(axis_limit) + 1)

    for label, ax in enumerate(axes):
        index = int(selection["indices"][str(label)])
        length = int(data["valid_length"][index])
        end = min(length + 1, data["prey_pos"].shape[1])
        line_point_counts[str(label)] = end
        prey = data["prey_pos"][index, :end]
        predator = data["pred_pos"][index, :end, 0]
        visible_resources = resource_visibility(
            prey,
            resource_pos,
            collect_radius,
            expected_collected=int(data["resources_collected"][index]),
        )[-1]
        visible_resource_counts[str(label)] = int(visible_resources.sum())

        for center, radius in zip(data["lava_pos"][index], data["lava_rad"][index]):
            ax.add_patch(
                Circle(
                    center,
                    float(radius),
                    facecolor=lava_color,
                    edgecolor="#6c2d2d",
                    alpha=0.18,
                    linewidth=0.8,
                    zorder=1,
                )
            )
        ax.scatter(
            resource_pos[visible_resources, 0],
            resource_pos[visible_resources, 1],
            s=13,
            c=resource_color,
            marker="s",
            alpha=0.72,
            linewidths=0,
            zorder=2,
        )
        ax.plot(
            prey[:, 0],
            prey[:, 1],
            color=prey_color,
            linewidth=1.8,
            linestyle=(0, (4, 2)),
            alpha=0.92,
            zorder=3,
        )
        ax.plot(
            predator[:, 0],
            predator[:, 1],
            color=COLORS[label],
            linewidth=2.2,
            solid_capstyle="round",
            zorder=4,
        )
        ax.scatter(
            predator[0, 0],
            predator[0, 1],
            s=34,
            facecolors="white",
            edgecolors=COLORS[label],
            linewidths=1.4,
            zorder=5,
        )
        ax.scatter(
            predator[-1, 0],
            predator[-1, 1],
            s=30,
            c=COLORS[label],
            linewidths=0,
            zorder=5,
        )
        ax.scatter(
            prey[0, 0],
            prey[0, 1],
            s=23,
            c=prey_color,
            marker="^",
            zorder=5,
        )

        mask = data["label"] == label
        ax.text(
            0.5,
            1.14,
            DISPLAY_TITLES[label],
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
            color=COLORS[label],
        )
        ax.text(
            0.5,
            1.055,
            f"n={int(mask.sum())}   capture={data['captured'][mask].mean():.1%}   "
            f"median T={np.median(data['valid_length'][mask]):.0f}   "
            f"coverage={data['pred_coverage'][mask].mean():.1f}",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=8.2,
            color=muted,
        )
        ax.text(
            0.5,
            -0.15,
            f"shown: T={length} · lava steps={data['pred_lava_steps'][index]:.0f} · "
            f"coverage={data['pred_coverage'][index]:.0f}",
            transform=ax.transAxes,
            ha="center",
            fontsize=8.5,
            color=muted,
        )
        ax.set_xlim(-axis_limit, axis_limit)
        ax.set_ylim(-axis_limit, axis_limit)
        ax.set_aspect("equal")
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.tick_params(labelsize=8, length=2.5, color="#bbbbbb")
        ax.set_xlabel("x", fontsize=9, color=muted)
        if label == 0:
            ax.set_ylabel("y", fontsize=9, color=muted)
        for spine in ax.spines.values():
            spine.set_color("#d0d0d0")
            spine.set_linewidth(0.8)

    handles = [
        Line2D([0], [0], color="#333333", linewidth=2.2, label="predator path"),
        Line2D(
            [0],
            [0],
            color=prey_color,
            linewidth=1.8,
            linestyle="--",
            label="fixed-policy prey path",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=resource_color,
            markersize=5,
            label="resources",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=lava_color,
            alpha=0.35,
            markersize=7,
            label="lava",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=4,
        frameon=False,
        fontsize=8.5,
    )
    fig.suptitle(
        "Representative trajectories by hidden objective",
        fontsize=15,
        fontweight="bold",
        y=1.01,
        color="#1a1a1a",
    )
    fig.text(
        0.5,
        0.015,
        f"Controlled comparison: same checkpoint seed, reset key, initial scene, and {prey_objective}-prey policy; paths stop at capture or horizon.",
        ha="center",
        fontsize=8.7,
        color=muted,
    )
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.18, top=0.76, wspace=0.12)
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    pixels = mpimg.imread(png_path)
    return {
        "dpi": dpi,
        "pixel_width": int(pixels.shape[1]),
        "pixel_height": int(pixels.shape[0]),
        "common_abs_axis_limit": float(axis_limit),
        "line_point_counts": line_point_counts,
        "visible_resource_counts": visible_resource_counts,
    }


def dataset_schema(
    dataset: ObjectiveDataset | Mapping[str, np.ndarray],
) -> dict[str, Any]:
    data = _arrays(dataset)
    return {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "axes": AXIS_SEMANTICS[name],
        }
        for name, value in data.items()
    }


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_dataset_atomic(path: Path | str, dataset: ObjectiveDataset) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(suffix=".npz", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **dataset.as_dict())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
