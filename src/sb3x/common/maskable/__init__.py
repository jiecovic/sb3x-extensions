"""Shared invalid-action masking helpers."""

from .distributions import (
    MaskableBernoulliDistribution,
    MaskableCategorical,
    MaskableCategoricalDistribution,
    MaskableDistribution,
    MaskableMultiCategoricalDistribution,
    MaybeMasks,
    make_masked_proba_distribution,
)
from .utils import (
    EXPECTED_METHOD_NAME,
    get_action_masks,
    is_masking_supported,
    make_all_valid_action_masks,
    mask_dims_for_action_space,
    reshape_action_masks,
)

__all__ = [
    "EXPECTED_METHOD_NAME",
    "MaskableBernoulliDistribution",
    "MaskableCategorical",
    "MaskableCategoricalDistribution",
    "MaskableDistribution",
    "MaskableMultiCategoricalDistribution",
    "MaybeMasks",
    "get_action_masks",
    "is_masking_supported",
    "make_masked_proba_distribution",
    "make_all_valid_action_masks",
    "mask_dims_for_action_space",
    "reshape_action_masks",
]
