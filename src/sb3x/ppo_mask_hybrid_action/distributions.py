"""Maskable hybrid action distribution."""

from __future__ import annotations

from sb3x.common.hybrid_action import (
    BaseHybridActionDistribution,
    HybridActionSpec,
)
from sb3x.common.maskable import MaskableMultiCategoricalDistribution, MaybeMasks


class MaskableHybridActionDistribution(BaseHybridActionDistribution):
    """Independent Gaussian branch plus masked MultiCategorical branch."""

    discrete_dist: MaskableMultiCategoricalDistribution

    def __init__(self, spec: HybridActionSpec) -> None:
        super().__init__(
            spec,
            MaskableMultiCategoricalDistribution(spec.discrete_action_dims),
        )

    def apply_masking(self, masks: MaybeMasks) -> None:
        """Apply masks only to the discrete branch."""
        self.discrete_dist.apply_masking(masks)
