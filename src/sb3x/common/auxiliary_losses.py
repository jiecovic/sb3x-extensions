"""Generic optional policy-side auxiliary loss hooks for PPO-family trainers."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch as th


@dataclass(frozen=True, slots=True)
class PolicyAuxiliaryLoss:
    """One minibatch auxiliary loss bundle returned by an opt-in policy hook."""

    total_loss: th.Tensor
    metrics: dict[str, float]


@dataclass(frozen=True, slots=True)
class PolicyActionEvaluation:
    """Policy action-evaluation result with optional extension payloads."""

    values: th.Tensor
    log_prob: th.Tensor
    entropy: th.Tensor | None
    aux_loss: PolicyAuxiliaryLoss | None = None
    entropy_components: Mapping[str, th.Tensor] = field(default_factory=dict)


def evaluate_actions_with_optional_aux(
    policy: Any,
    *args: Any,
    auxiliary_mask: th.Tensor | None = None,
    **kwargs: Any,
) -> tuple[th.Tensor, th.Tensor, th.Tensor | None, PolicyAuxiliaryLoss | None]:
    """Call a policy aux hook when present, else fall back to evaluate_actions()."""
    evaluation = evaluate_policy_actions_with_optional_aux(
        policy,
        *args,
        auxiliary_mask=auxiliary_mask,
        **kwargs,
    )
    return (
        evaluation.values,
        evaluation.log_prob,
        evaluation.entropy,
        evaluation.aux_loss,
    )


def evaluate_policy_actions_with_optional_aux(
    policy: Any,
    *args: Any,
    auxiliary_mask: th.Tensor | None = None,
    **kwargs: Any,
) -> PolicyActionEvaluation:
    """Call the richest available policy action-evaluation hook."""

    evaluate_with_aux = _callable_attr(policy, "evaluate_actions_with_aux")
    if evaluate_with_aux is not None:
        return _normalize_policy_action_evaluation(
            evaluate_with_aux(
                *args,
                auxiliary_mask=auxiliary_mask,
                **kwargs,
            )
        )

    evaluate_actions = _callable_attr(policy, "evaluate_actions")
    if evaluate_actions is None:
        raise TypeError("Policy does not expose evaluate_actions(...)")
    return _normalize_policy_action_evaluation(evaluate_actions(*args, **kwargs))


def combine_policy_auxiliary_losses(
    losses: Iterable[PolicyAuxiliaryLoss | None],
) -> PolicyAuxiliaryLoss | None:
    """Combine optional policy-side losses into one optimizer term."""
    present = [loss for loss in losses if loss is not None]
    if not present:
        return None

    total_loss = sum((loss.total_loss for loss in present), present[0].total_loss * 0)
    metrics: dict[str, float] = {}
    for loss in present:
        metrics.update(loss.metrics)
    metrics["__total__"] = float(total_loss.detach().cpu().item())
    return PolicyAuxiliaryLoss(total_loss=total_loss, metrics=metrics)


def _normalize_policy_action_evaluation(result: Any) -> PolicyActionEvaluation:
    if isinstance(result, PolicyActionEvaluation):
        return result
    if not isinstance(result, tuple):
        raise TypeError("Policy action evaluation must return a tuple or dataclass")
    if len(result) == 3:
        values, log_prob, entropy = result
        return PolicyActionEvaluation(values=values, log_prob=log_prob, entropy=entropy)
    if len(result) == 4:
        values, log_prob, entropy, aux_loss = result
        return PolicyActionEvaluation(
            values=values,
            log_prob=log_prob,
            entropy=entropy,
            aux_loss=aux_loss,
        )
    if len(result) == 5:
        values, log_prob, entropy, aux_loss, entropy_components = result
        return PolicyActionEvaluation(
            values=values,
            log_prob=log_prob,
            entropy=entropy,
            aux_loss=aux_loss,
            entropy_components=entropy_components,
        )
    raise TypeError(
        f"Policy action evaluation must return 3, 4, or 5 values, got {len(result)}"
    )


def _callable_attr(instance: Any, name: str) -> Callable[..., Any] | None:
    attribute = getattr(instance, name, None)
    return attribute if callable(attribute) else None
