"""Spatial Blotto environment, mathematical controllers, and JAX match collection."""

from spatial_blotto.allocations import allocation_scores, enumerate_allocations
from spatial_blotto.controllers import TeamController, make_controller
from spatial_blotto.environment import BlottoState, SpatialBlotto
from spatial_blotto.rollout import Episode, joint_actions, rollout_episode

__all__ = [
    "BlottoState",
    "Episode",
    "SpatialBlotto",
    "TeamController",
    "allocation_scores",
    "enumerate_allocations",
    "joint_actions",
    "make_controller",
    "rollout_episode",
]
