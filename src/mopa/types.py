"""Typed containers for Part 1 datasets and BC comparisons."""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Iterator

import numpy as np
from numpy.typing import NDArray

Arrayf = NDArray[np.floating[Any]]
Arrayi = NDArray[np.integer[Any]]


class _ArrayMapping:
    """Mixin: dataclass of arrays that also behaves like ``Mapping[str, ndarray]``."""

    def as_dict(self) -> dict[str, np.ndarray]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def __iter__(self) -> Iterator[str]:
        return (f.name for f in fields(self))

    def __len__(self) -> int:
        return len(fields(self))

    def __getitem__(self, key: str) -> np.ndarray:
        return getattr(self, key)

    def keys(self) -> Iterator[str]:
        return iter(self)

    def values(self) -> Iterator[np.ndarray]:
        return (getattr(self, f.name) for f in fields(self))

    def items(self) -> Iterator[tuple[str, np.ndarray]]:
        return ((f.name, getattr(self, f.name)) for f in fields(self))


@dataclass
class ObjectiveDataset(_ArrayMapping):
    """Objective-typed predator rollouts with lava layouts and first-episode metrics."""

    prey_pos: Arrayf
    pred_pos: Arrayf
    lava_pos: Arrayf
    lava_rad: Arrayf
    prey_act: Arrayi
    pred_act: Arrayi
    capture_t: Arrayi
    captured: Arrayi
    survival_time: Arrayf
    pred_lava_steps: Arrayf
    prey_lava_steps: Arrayf
    resources_collected: Arrayf
    pred_coverage: Arrayf
    label: Arrayi
    ckpt_seed: Arrayi
    env_seed: Arrayi
    valid_length: Arrayi


@dataclass
class ObjectiveObservationDataset(ObjectiveDataset):
    """Objective dataset plus exact pre-action predator observations."""

    pred_obs: Arrayf


@dataclass(frozen=True)
class BCRunStats:
    """Mean / std / per-seed accuracies for one BC conditioning variant."""

    mean: float
    std: float
    runs: tuple[float, ...]

    def as_tuple(self) -> tuple[float, float, list[float]]:
        return self.mean, self.std, list(self.runs)


@dataclass(frozen=True)
class CheckpointRef:
    """Pointer to a saved MAPPO actor under ``logs/``."""

    alg: str
    team: str
    seed: int
    path: str
