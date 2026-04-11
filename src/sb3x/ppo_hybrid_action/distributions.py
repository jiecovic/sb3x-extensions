"""Hybrid action distribution for independent Box and MultiDiscrete branches."""

from __future__ import annotations

from stable_baselines3.common.distributions import (
    MultiCategoricalDistribution,
)

from sb3x.common.hybrid_action import (
    BaseHybridActionDistribution,
    HybridActionSpec,
)


class HybridActionDistribution(BaseHybridActionDistribution):
    """Independent Gaussian and MultiCategorical distribution pair."""

    discrete_dist: MultiCategoricalDistribution

    def __init__(self, spec: HybridActionSpec) -> None:
        super().__init__(
            spec,
            MultiCategoricalDistribution(spec.discrete_action_dims),
        )
