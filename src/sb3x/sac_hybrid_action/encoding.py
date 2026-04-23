"""Action encodings used by hybrid-action SAC."""

from __future__ import annotations

import torch as th
import torch.nn.functional as F

from sb3x.common.hybrid_action import HybridActionSpec


def validate_sac_discrete_dims(spec: HybridActionSpec) -> None:
    """Reject degenerate dimensions that SB3's Box scaling cannot represent."""
    if any(action_dim <= 1 for action_dim in spec.discrete_action_dims):
        raise ValueError(
            "HybridActionSAC requires each MultiDiscrete branch dimension to "
            "have at least two actions"
        )


def scale_discrete_actions(
    spec: HybridActionSpec,
    discrete_actions: th.Tensor,
) -> th.Tensor:
    """Map zero-based discrete indices to SB3's scaled Box coordinates."""
    nvec = _nvec_tensor(spec, discrete_actions).reshape(1, -1)
    return 2.0 * discrete_actions.float() / (nvec - 1.0) - 1.0


def unscale_discrete_actions(
    spec: HybridActionSpec,
    scaled_discrete_actions: th.Tensor,
) -> th.Tensor:
    """Map scaled Box coordinates back to valid zero-based discrete indices."""
    nvec = _nvec_tensor(spec, scaled_discrete_actions).reshape(1, -1)
    discrete_actions = 0.5 * (scaled_discrete_actions + 1.0) * (nvec - 1.0)
    return discrete_actions.round().clamp(min=0).minimum(nvec - 1.0).long()


def one_hot_discrete_actions(
    spec: HybridActionSpec,
    discrete_actions: th.Tensor,
) -> th.Tensor:
    """One-hot encode every ``MultiDiscrete`` branch and concatenate them."""
    encoded_branches = [
        F.one_hot(discrete_actions[:, branch_idx], num_classes=action_dim).float()
        for branch_idx, action_dim in enumerate(spec.discrete_action_dims)
    ]
    return th.cat(encoded_branches, dim=1)


def encode_scaled_hybrid_actions_for_critic(
    spec: HybridActionSpec,
    scaled_actions: th.Tensor,
) -> th.Tensor:
    """Encode scaled flat SAC actions as continuous plus one-hot discrete input."""
    continuous_actions, scaled_discrete_actions = th.split(
        scaled_actions,
        [spec.continuous_dim, spec.discrete_dim],
        dim=1,
    )
    discrete_actions = unscale_discrete_actions(spec, scaled_discrete_actions)
    return th.cat(
        [continuous_actions, one_hot_discrete_actions(spec, discrete_actions)],
        dim=1,
    )


def _nvec_tensor(spec: HybridActionSpec, reference: th.Tensor) -> th.Tensor:
    dtype = reference.dtype if th.is_floating_point(reference) else th.float32
    return th.as_tensor(
        spec.discrete_action_dims,
        device=reference.device,
        dtype=dtype,
    )
