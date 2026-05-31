"""Entropy-loss helpers shared by PPO variants with named action groups."""

from __future__ import annotations

from collections.abc import Mapping

import torch as th


def entropy_loss(
    *,
    log_prob: th.Tensor,
    entropy: th.Tensor | None,
    entropy_components: Mapping[str, th.Tensor],
    entropy_group_weights: Mapping[str, float],
    sample_mask: th.Tensor | None = None,
) -> tuple[th.Tensor, dict[str, float]]:
    """Compute PPO entropy loss with optional per-action-group weights."""
    if entropy_group_weights:
        loss_tensor = _weighted_entropy_loss(
            entropy_components=entropy_components,
            entropy_group_weights=entropy_group_weights,
            sample_mask=sample_mask,
        )
    elif entropy is None:
        loss_tensor = -_masked_mean(-log_prob, sample_mask)
    else:
        loss_tensor = -_masked_mean(entropy, sample_mask)

    metrics = {
        name: float(_masked_mean(component, sample_mask).detach().cpu().item())
        for name, component in entropy_components.items()
    }
    return loss_tensor, metrics


def normalize_entropy_group_weights(
    weights: Mapping[str, float] | None,
) -> dict[str, float]:
    """Normalize user-provided entropy weights once at algorithm construction."""
    if not weights:
        return {}

    normalized: dict[str, float] = {}
    for name, weight in weights.items():
        value = float(weight)
        if value < 0:
            raise ValueError(f"entropy weight for {name!r} must be non-negative")
        normalized[str(name)] = value
    return normalized


def _weighted_entropy_loss(
    *,
    entropy_components: Mapping[str, th.Tensor],
    entropy_group_weights: Mapping[str, float],
    sample_mask: th.Tensor | None,
) -> th.Tensor:
    if not entropy_components:
        raise ValueError(
            "entropy_group_weights require policies to return entropy components"
        )

    weighted_entropy: th.Tensor | None = None
    for name, weight in entropy_group_weights.items():
        component = entropy_components.get(name)
        if component is None:
            continue
        weighted_component = component * weight
        weighted_entropy = (
            weighted_component
            if weighted_entropy is None
            else weighted_entropy + weighted_component
        )

    if weighted_entropy is None:
        known_names = ", ".join(sorted(entropy_components))
        requested_names = ", ".join(sorted(entropy_group_weights))
        raise ValueError(
            "entropy_group_weights do not match any policy entropy components "
            f"(requested: {requested_names}; available: {known_names})"
        )
    return -_masked_mean(weighted_entropy, sample_mask)


def _masked_mean(values: th.Tensor, sample_mask: th.Tensor | None) -> th.Tensor:
    if sample_mask is None:
        return values.mean()
    return values[sample_mask].mean()
