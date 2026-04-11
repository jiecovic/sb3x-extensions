"""Hybrid action distribution for independent Box and MultiDiscrete branches."""

from __future__ import annotations

from typing import TypeVar

import torch as th
from stable_baselines3.common.distributions import (
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
)
from torch import nn

from sb3x.common.hybrid_action import HybridActionSpec

SelfHybridActionDistribution = TypeVar(
    "SelfHybridActionDistribution",
    bound="HybridActionDistribution",
)


class HybridActionDistribution(Distribution):
    """Independent Gaussian and MultiCategorical distribution pair."""

    def __init__(self, spec: HybridActionSpec) -> None:
        super().__init__()
        self.spec = spec
        self.continuous_dist = DiagGaussianDistribution(spec.continuous_dim)
        self.discrete_dist = MultiCategoricalDistribution(spec.discrete_action_dims)

    def proba_distribution_net(
        self,
        latent_dim: int,
        log_std_init: float = 0.0,
    ) -> tuple[nn.Module, nn.Parameter]:
        action_net = HybridActionNet(
            latent_dim=latent_dim,
            continuous_dim=self.spec.continuous_dim,
            discrete_logits_dim=self.spec.discrete_logits_dim,
        )
        log_std = nn.Parameter(
            th.ones(self.spec.continuous_dim) * log_std_init,
            requires_grad=True,
        )
        return action_net, log_std

    def proba_distribution(
        self: SelfHybridActionDistribution,
        action_params: th.Tensor,
        log_std: th.Tensor,
    ) -> SelfHybridActionDistribution:
        continuous_mean, discrete_logits = self._split_action_params(action_params)
        self.continuous_dist.proba_distribution(continuous_mean, log_std)
        self.discrete_dist.proba_distribution(discrete_logits)
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        continuous_actions, discrete_actions = self._split_actions(actions)
        return self.continuous_dist.log_prob(
            continuous_actions
        ) + self.discrete_dist.log_prob(discrete_actions)

    def entropy(self) -> th.Tensor | None:
        continuous_entropy = self.continuous_dist.entropy()
        discrete_entropy = self.discrete_dist.entropy()
        if continuous_entropy is None:
            return None
        return continuous_entropy + discrete_entropy

    def sample(self) -> th.Tensor:
        return self._combine_actions(
            self.continuous_dist.sample(),
            self.discrete_dist.sample(),
        )

    def mode(self) -> th.Tensor:
        return self._combine_actions(
            self.continuous_dist.mode(),
            self.discrete_dist.mode(),
        )

    def actions_from_params(
        self,
        action_params: th.Tensor,
        log_std: th.Tensor,
        deterministic: bool = False,
    ) -> th.Tensor:
        self.proba_distribution(action_params, log_std)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(
        self,
        action_params: th.Tensor,
        log_std: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(action_params, log_std)
        return actions, self.log_prob(actions)

    def _split_action_params(
        self,
        action_params: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor]:
        continuous_mean, discrete_logits = th.split(
            action_params,
            [self.spec.continuous_dim, self.spec.discrete_logits_dim],
            dim=1,
        )
        return continuous_mean, discrete_logits

    def _split_actions(self, actions: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        continuous_actions, discrete_actions = th.split(
            actions,
            [self.spec.continuous_dim, self.spec.discrete_dim],
            dim=1,
        )
        return continuous_actions, discrete_actions.long()

    @staticmethod
    def _combine_actions(
        continuous_actions: th.Tensor,
        discrete_actions: th.Tensor,
    ) -> th.Tensor:
        return th.cat([continuous_actions, discrete_actions.float()], dim=1)


class HybridActionNet(nn.Module):
    """Policy head that emits continuous means and discrete logits."""

    def __init__(
        self,
        *,
        latent_dim: int,
        continuous_dim: int,
        discrete_logits_dim: int,
    ) -> None:
        super().__init__()
        self.continuous_net = nn.Linear(latent_dim, continuous_dim)
        self.discrete_net = nn.Linear(latent_dim, discrete_logits_dim)

    def forward(self, latent: th.Tensor) -> th.Tensor:
        return th.cat(
            [self.continuous_net(latent), self.discrete_net(latent)],
            dim=1,
        )
