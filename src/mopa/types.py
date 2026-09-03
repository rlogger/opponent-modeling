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


@dataclass
class ContinuousTrajectoryDataset(_ArrayMapping):
    """Continuous joint-action episodes from frozen specialists (handoff data contract).

    Per episode ``i`` with horizon ``T`` (``valid_length[i]`` real transitions):

    - ``state``            float32 ``[N, T + 1, state_dim]`` Markov-equivalent vector
    - ``blue_observation`` float32 ``[N, T + 1, blue_obs_dim]``
    - ``red_observation``  float32 ``[N, T + 1, red_obs_dim]``
    - ``blue_action``      float32 ``[N, T, 2]`` in ``[-1, 1]``
    - ``red_action``       float32 ``[N, T, 2]`` in ``[-1, 1]``
    - ``blue_reward``      float32 ``[N, T]``
    - ``terminated_capture`` / ``truncated_timeout`` / ``valid_mask`` bool ``[N, T]``
      (``truncated_timeout`` marks the env time limit *or* the recording
      horizon without capture; exactly one flag ends every episode)
    - ``causal_context``   float32 ``[N, T + 1, context_dim]`` (``c_0`` is zero)
    - ``objective_label`` / ``checkpoint_seed`` int32 ``[N]``
    - ``environment_seed`` uint32 ``[N, 2]`` reset key; ``step_seed`` uint32 ``[N, 2]``

    Positions, lava geometry, and first-episode metrics are retained so the
    existing feature / encoder / plotting utilities keep working.
    """

    state: Arrayf
    blue_observation: Arrayf
    red_observation: Arrayf
    blue_action: Arrayf
    red_action: Arrayf
    blue_reward: Arrayf
    terminated_capture: NDArray[np.bool_]
    truncated_timeout: NDArray[np.bool_]
    valid_mask: NDArray[np.bool_]
    causal_context: Arrayf
    objective_label: Arrayi
    checkpoint_seed: Arrayi
    environment_seed: Arrayi
    step_seed: Arrayi
    prey_pos: Arrayf
    pred_pos: Arrayf
    lava_pos: Arrayf
    lava_rad: Arrayf
    capture_t: Arrayi
    captured: Arrayi
    survival_time: Arrayf
    pred_lava_steps: Arrayf
    prey_lava_steps: Arrayf
    resources_collected: Arrayf
    pred_coverage: Arrayf
    valid_length: Arrayi

    @property
    def label(self) -> Arrayi:
        """Alias so encoder/BC utilities written for ``ObjectiveDataset`` work."""
        return self.objective_label

    @property
    def ckpt_seed(self) -> Arrayi:
        return self.checkpoint_seed

    @property
    def env_seed(self) -> Arrayi:
        return self.environment_seed


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
