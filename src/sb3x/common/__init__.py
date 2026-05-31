"""Shared implementation helpers for sb3x algorithms."""

from sb3x.common.auxiliary_losses import (
    PolicyActionEvaluation,
    PolicyAuxiliaryLoss,
    combine_policy_auxiliary_losses,
    evaluate_actions_with_optional_aux,
    evaluate_policy_actions_with_optional_aux,
)
from sb3x.common.entropy import entropy_loss, normalize_entropy_group_weights

__all__ = [
    "PolicyActionEvaluation",
    "PolicyAuxiliaryLoss",
    "combine_policy_auxiliary_losses",
    "entropy_loss",
    "evaluate_actions_with_optional_aux",
    "evaluate_policy_actions_with_optional_aux",
    "normalize_entropy_group_weights",
]
