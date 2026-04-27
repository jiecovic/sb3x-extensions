"""Categorical actor used by discrete SAC."""

from __future__ import annotations

from typing import Any

import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, create_mlp
from stable_baselines3.common.type_aliases import PyTorchObs
from torch import nn


class DiscreteSACActor(BasePolicy):
    """Categorical SAC actor for ``spaces.Discrete`` action spaces."""

    action_space: spaces.Discrete

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        net_arch: list[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        activation_fn: type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )
        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn
        logits_net = create_mlp(
            features_dim,
            int(action_space.n),
            net_arch,
            activation_fn,
        )
        self.logits_net = nn.Sequential(*logits_net)

    def forward(
        self,
        obs: PyTorchObs,
        deterministic: bool = False,
    ) -> th.Tensor:
        logits = self.action_logits(obs)
        if deterministic:
            return logits.argmax(dim=1).reshape(-1)
        return th.distributions.Categorical(logits=logits).sample().reshape(-1)

    def action_logits(self, obs: PyTorchObs) -> th.Tensor:
        """Return unnormalized action logits for one observation batch."""
        features = self.extract_features(obs, self.features_extractor)
        return self.logits_net(features)

    def action_probabilities(self, obs: PyTorchObs) -> tuple[th.Tensor, th.Tensor]:
        """Return action probabilities and log-probabilities for all actions."""
        logits = self.action_logits(obs)
        log_probs = F.log_softmax(logits, dim=1)
        return log_probs.exp(), log_probs

    def _predict(
        self,
        observation: PyTorchObs,
        deterministic: bool = False,
    ) -> th.Tensor:
        return self(observation, deterministic=deterministic)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            {
                "net_arch": self.net_arch,
                "features_dim": self.features_dim,
                "activation_fn": self.activation_fn,
                "features_extractor": self.features_extractor,
            }
        )
        return data


__all__ = ["DiscreteSACActor"]
