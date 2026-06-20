"""Small tensor metric helpers shared by training loops."""

from __future__ import annotations

from collections.abc import Mapping

import torch as th


def append_component_means(
    history: dict[str, list[float]],
    components: Mapping[str, th.Tensor],
    *,
    sample_mask: th.Tensor | None = None,
) -> None:
    """Append scalar means for named per-sample tensor components."""
    for name, component in components.items():
        history.setdefault(name, []).append(
            float(_masked_mean(component, sample_mask).detach().cpu().item())
        )


def _masked_mean(values: th.Tensor, sample_mask: th.Tensor | None) -> th.Tensor:
    if sample_mask is None:
        return values.mean()
    return values[sample_mask].mean()
