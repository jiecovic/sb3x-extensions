"""Shared tensor helpers and base distribution for hybrid actions."""

from __future__ import annotations

from typing import Literal, TypeVar

import torch as th
from stable_baselines3.common.distributions import (
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
)
from torch import nn

from sb3x.common.maskable import MaskableMultiCategoricalDistribution, MaybeMasks

from .spaces import (
    HybridActionGroupNames,
    HybridActionSpec,
    make_hybrid_action_group_names,
)

SelfBaseHybridActionDistribution = TypeVar(
    "SelfBaseHybridActionDistribution",
    bound="BaseHybridActionDistribution",
)
ContinuousLogStdMode = Literal["parameter", "state_dependent"]
CONTINUOUS_LOG_STD_BOUNDS = (-20.0, 2.0)


class HybridActionNet(nn.Module):
    """Policy head that emits continuous means, optional stds, and discrete logits."""

    def __init__(
        self,
        *,
        latent_dim: int,
        continuous_dim: int,
        discrete_logits_dim: int,
        continuous_log_std_mode: ContinuousLogStdMode = "parameter",
        log_std_init: float = 0.0,
        log_std_bounds: tuple[float, float] = CONTINUOUS_LOG_STD_BOUNDS,
    ) -> None:
        super().__init__()
        if continuous_log_std_mode not in ("parameter", "state_dependent"):
            raise ValueError(
                "continuous_log_std_mode must be 'parameter' or 'state_dependent'"
            )
        if log_std_bounds[0] >= log_std_bounds[1]:
            raise ValueError("continuous log std bounds must be ordered min < max")
        self.continuous_log_std_mode: ContinuousLogStdMode = continuous_log_std_mode
        self.log_std_init = float(log_std_init)
        self.log_std_bounds = (float(log_std_bounds[0]), float(log_std_bounds[1]))
        self.continuous_net = _branch_head(latent_dim, continuous_dim)
        self.continuous_log_std_net = (
            _branch_head(latent_dim, continuous_dim)
            if continuous_log_std_mode == "state_dependent"
            else None
        )
        self.discrete_net = _branch_head(latent_dim, discrete_logits_dim)

    def forward(self, latent: th.Tensor) -> th.Tensor:
        continuous_mean = self.continuous_net(latent)
        if self.continuous_log_std_net is None:
            continuous_params = continuous_mean
        else:
            min_log_std, max_log_std = self.log_std_bounds
            continuous_log_std = th.clamp(
                self.continuous_log_std_net(latent) + self.log_std_init,
                min=min_log_std,
                max=max_log_std,
            )
            continuous_params = th.cat([continuous_mean, continuous_log_std], dim=1)
        return th.cat(
            [continuous_params, self.discrete_net(latent)],
            dim=1,
        )


class _EmptyBranchHead(nn.Module):
    """Return an empty branch tensor without creating zero-size parameters."""

    in_features: int
    out_features: int = 0

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.in_features = int(latent_dim)

    def forward(self, latent: th.Tensor) -> th.Tensor:
        return latent.new_empty((*latent.shape[:-1], 0))


def _branch_head(latent_dim: int, output_dim: int) -> nn.Module:
    if output_dim <= 0:
        return _EmptyBranchHead(latent_dim)
    return nn.Linear(latent_dim, output_dim)


def split_hybrid_action_params(
    spec: HybridActionSpec,
    action_params: th.Tensor,
    *,
    continuous_log_std_mode: ContinuousLogStdMode = "parameter",
) -> tuple[th.Tensor, th.Tensor | None, th.Tensor]:
    """Split flat policy-head output into continuous params and discrete logits."""
    if continuous_log_std_mode == "parameter":
        continuous_mean, discrete_logits = th.split(
            action_params,
            [spec.continuous_dim, spec.discrete_logits_dim],
            dim=1,
        )
        return continuous_mean, None, discrete_logits
    if continuous_log_std_mode != "state_dependent":
        raise ValueError(
            "continuous_log_std_mode must be 'parameter' or 'state_dependent'"
        )
    continuous_mean, continuous_log_std, discrete_logits = th.split(
        action_params,
        [spec.continuous_dim, spec.continuous_dim, spec.discrete_logits_dim],
        dim=1,
    )
    return continuous_mean, continuous_log_std, discrete_logits


def split_hybrid_actions(
    spec: HybridActionSpec,
    actions: th.Tensor,
) -> tuple[th.Tensor, th.Tensor]:
    """Split flat action tensors into continuous and discrete branches."""
    continuous_actions, discrete_actions = th.split(
        actions,
        [spec.continuous_dim, spec.discrete_dim],
        dim=1,
    )
    return continuous_actions, discrete_actions.long()


def combine_hybrid_actions(
    continuous_actions: th.Tensor,
    discrete_actions: th.Tensor,
) -> th.Tensor:
    """Combine branch actions into the flat action tensor used by SB3 buffers."""
    return th.cat([continuous_actions, discrete_actions.float()], dim=1)


def _empty_branch_tensor(reference: th.Tensor) -> th.Tensor:
    return reference.new_zeros((reference.shape[0], 0))


class BaseHybridActionDistribution(Distribution):
    """Shared Gaussian-plus-discrete distribution behavior for hybrid actions."""

    discrete_dist: Distribution

    def __init__(
        self,
        spec: HybridActionSpec,
        discrete_dist: Distribution,
        group_names: HybridActionGroupNames | None = None,
        continuous_log_std_mode: ContinuousLogStdMode = "parameter",
        log_std_bounds: tuple[float, float] = CONTINUOUS_LOG_STD_BOUNDS,
    ) -> None:
        super().__init__()
        if continuous_log_std_mode not in ("parameter", "state_dependent"):
            raise ValueError(
                "continuous_log_std_mode must be 'parameter' or 'state_dependent'"
            )
        self.spec = spec
        self.continuous_log_std_mode: ContinuousLogStdMode = continuous_log_std_mode
        self.log_std_bounds = (float(log_std_bounds[0]), float(log_std_bounds[1]))
        self.group_names = (
            group_names
            if group_names is not None
            else make_hybrid_action_group_names(spec)
        )
        self.continuous_dist = DiagGaussianDistribution(spec.continuous_dim)
        self.discrete_dist = discrete_dist

    def proba_distribution_net(
        self,
        latent_dim: int,
        log_std_init: float = 0.0,
    ) -> tuple[nn.Module, nn.Parameter | None]:
        action_net = HybridActionNet(
            latent_dim=latent_dim,
            continuous_dim=self.spec.continuous_dim,
            discrete_logits_dim=self.spec.discrete_logits_dim,
            continuous_log_std_mode=self.continuous_log_std_mode,
            log_std_init=log_std_init,
            log_std_bounds=self.log_std_bounds,
        )
        if self.continuous_log_std_mode == "state_dependent":
            return action_net, None
        log_std = nn.Parameter(
            th.ones(self.spec.continuous_dim) * log_std_init,
            requires_grad=True,
        )
        return action_net, log_std

    def proba_distribution(
        self: SelfBaseHybridActionDistribution,
        action_params: th.Tensor,
        log_std: th.Tensor | None,
    ) -> SelfBaseHybridActionDistribution:
        continuous_mean, continuous_log_std, discrete_logits = (
            split_hybrid_action_params(
                self.spec,
                action_params,
                continuous_log_std_mode=self.continuous_log_std_mode,
            )
        )
        if continuous_log_std is not None:
            log_std = continuous_log_std
        if log_std is None:
            raise TypeError("hybrid action distribution requires continuous log std")
        self.continuous_dist.proba_distribution(continuous_mean, log_std)
        self.discrete_dist.proba_distribution(discrete_logits)
        return self

    def continuous_log_std(self) -> th.Tensor:
        """Return per-sample continuous log std for the current distribution."""

        return self.continuous_dist.distribution.scale.log()

    def continuous_std(self) -> th.Tensor:
        """Return per-sample continuous std for the current distribution."""

        return self.continuous_dist.distribution.scale

    def std_components(self) -> dict[str, th.Tensor]:
        """Return one per-sample std tensor for each named continuous action."""
        return {
            name: std
            for name, std in zip(
                self.group_names.continuous,
                th.unbind(self.continuous_std(), dim=1),
                strict=True,
            )
        }

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        continuous_actions, discrete_actions = split_hybrid_actions(
            self.spec,
            actions,
        )
        discrete_log_prob = (
            actions.new_zeros((actions.shape[0],))
            if not self.spec.discrete_action_dims
            else self.discrete_dist.log_prob(discrete_actions)
        )
        return self.continuous_dist.log_prob(continuous_actions) + discrete_log_prob

    def entropy(self) -> th.Tensor | None:
        continuous_entropy = self.continuous_dist.entropy()
        discrete_entropy = (
            None
            if continuous_entropy is None
            else continuous_entropy.new_zeros(continuous_entropy.shape)
        )
        if self.spec.discrete_action_dims:
            discrete_entropy = self.discrete_dist.entropy()
        if continuous_entropy is None or discrete_entropy is None:
            return None
        return continuous_entropy + discrete_entropy

    def entropy_components(self) -> dict[str, th.Tensor] | None:
        """Return one per-sample entropy tensor for each named action group."""
        continuous_entropy = self.continuous_dist.distribution.entropy()
        discrete_entropy = (
            continuous_entropy.new_zeros((continuous_entropy.shape[0], 0))
            if not self.spec.discrete_action_dims
            else _discrete_entropy_components(self.discrete_dist)
        )
        if continuous_entropy is None or discrete_entropy is None:
            return None

        components: dict[str, th.Tensor] = {}
        for name, entropy in zip(
            self.group_names.continuous,
            th.unbind(continuous_entropy, dim=1),
            strict=True,
        ):
            components[name] = entropy
        for name, entropy in zip(
            self.group_names.discrete,
            th.unbind(discrete_entropy, dim=1),
            strict=True,
        ):
            components[name] = entropy
        return components

    def sample(self) -> th.Tensor:
        continuous_actions = self.continuous_dist.sample()
        return combine_hybrid_actions(
            continuous_actions,
            (
                _empty_branch_tensor(continuous_actions)
                if not self.spec.discrete_action_dims
                else self.discrete_dist.sample()
            ),
        )

    def mode(self) -> th.Tensor:
        continuous_actions = self.continuous_dist.mode()
        return combine_hybrid_actions(
            continuous_actions,
            (
                _empty_branch_tensor(continuous_actions)
                if not self.spec.discrete_action_dims
                else self.discrete_dist.mode()
            ),
        )

    def actions_from_params(
        self,
        action_params: th.Tensor,
        log_std: th.Tensor | None,
        deterministic: bool = False,
    ) -> th.Tensor:
        self.proba_distribution(action_params, log_std)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(
        self,
        action_params: th.Tensor,
        log_std: th.Tensor | None,
    ) -> tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(action_params, log_std)
        return actions, self.log_prob(actions)


class HybridActionDistribution(BaseHybridActionDistribution):
    """Independent Gaussian and MultiCategorical distribution pair."""

    discrete_dist: MultiCategoricalDistribution

    def __init__(
        self,
        spec: HybridActionSpec,
        group_names: HybridActionGroupNames | None = None,
        continuous_log_std_mode: ContinuousLogStdMode = "parameter",
        log_std_bounds: tuple[float, float] = CONTINUOUS_LOG_STD_BOUNDS,
    ) -> None:
        super().__init__(
            spec,
            MultiCategoricalDistribution(spec.discrete_action_dims),
            group_names=group_names,
            continuous_log_std_mode=continuous_log_std_mode,
            log_std_bounds=log_std_bounds,
        )


class MaskableHybridActionDistribution(BaseHybridActionDistribution):
    """Independent Gaussian branch plus masked MultiCategorical branch."""

    discrete_dist: MaskableMultiCategoricalDistribution

    def __init__(
        self,
        spec: HybridActionSpec,
        group_names: HybridActionGroupNames | None = None,
        continuous_log_std_mode: ContinuousLogStdMode = "parameter",
        log_std_bounds: tuple[float, float] = CONTINUOUS_LOG_STD_BOUNDS,
    ) -> None:
        super().__init__(
            spec,
            MaskableMultiCategoricalDistribution(spec.discrete_action_dims),
            group_names=group_names,
            continuous_log_std_mode=continuous_log_std_mode,
            log_std_bounds=log_std_bounds,
        )

    def apply_masking(self, masks: MaybeMasks) -> None:
        """Apply masks only to the discrete branch."""
        self.discrete_dist.apply_masking(masks)


def _discrete_entropy_components(distribution: Distribution) -> th.Tensor | None:
    if isinstance(distribution, MaskableMultiCategoricalDistribution):
        return distribution.entropy_components()
    if isinstance(distribution, MultiCategoricalDistribution):
        return th.stack(
            [categorical.entropy() for categorical in distribution.distribution],
            dim=1,
        )
    entropy = distribution.entropy()
    return None if entropy is None else entropy.unsqueeze(dim=1)
