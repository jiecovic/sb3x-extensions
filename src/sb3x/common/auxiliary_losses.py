"""Generic optional policy-side auxiliary loss hooks for PPO-family trainers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch as th


@dataclass(frozen=True, slots=True)
class PolicyAuxiliaryLoss:
    """One minibatch auxiliary loss bundle returned by an opt-in policy hook."""

    total_loss: th.Tensor
    metrics: dict[str, float]


def evaluate_actions_with_optional_aux(
    policy: Any,
    *args: Any,
    auxiliary_mask: th.Tensor | None = None,
    **kwargs: Any,
) -> tuple[th.Tensor, th.Tensor, th.Tensor | None, PolicyAuxiliaryLoss | None]:
    """Call a policy aux hook when present, else fall back to evaluate_actions()."""

    evaluate_with_aux = _callable_attr(policy, "evaluate_actions_with_aux")
    if evaluate_with_aux is not None:
        values, log_prob, entropy, aux_loss = evaluate_with_aux(
            *args,
            auxiliary_mask=auxiliary_mask,
            **kwargs,
        )
        return values, log_prob, entropy, aux_loss

    evaluate_actions = _callable_attr(policy, "evaluate_actions")
    if evaluate_actions is None:
        raise TypeError("Policy does not expose evaluate_actions(...)")
    values, log_prob, entropy = evaluate_actions(*args, **kwargs)
    return values, log_prob, entropy, None


def _callable_attr(instance: Any, name: str) -> Callable[..., Any] | None:
    attribute = getattr(instance, name, None)
    return attribute if callable(attribute) else None
