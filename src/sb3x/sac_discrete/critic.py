"""Discrete-action Q critics used by discrete SAC."""

from __future__ import annotations

from typing import Any

import torch as th
from gymnasium import spaces
from stable_baselines3.common.policies import BaseModel
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, create_mlp
from stable_baselines3.common.type_aliases import PyTorchObs
from torch import nn


class DiscreteSACCritic(BaseModel):
    """Twin Q-network critic that outputs one value per discrete action."""

    action_space: spaces.Discrete
    features_extractor: BaseFeaturesExtractor

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        net_arch: list[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        activation_fn: type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
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
        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.q_networks: list[nn.Module] = []
        for idx in range(n_critics):
            q_net = nn.Sequential(
                *create_mlp(
                    features_dim,
                    int(action_space.n),
                    net_arch,
                    activation_fn,
                )
            )
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def forward(self, obs: PyTorchObs) -> tuple[th.Tensor, ...]:
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs, self.features_extractor)
        return tuple(q_net(features) for q_net in self.q_networks)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            {
                "net_arch": self.net_arch,
                "features_dim": self.features_dim,
                "activation_fn": self.activation_fn,
                "features_extractor": self.features_extractor,
                "n_critics": self.n_critics,
                "share_features_extractor": self.share_features_extractor,
            }
        )
        return data


__all__ = ["DiscreteSACCritic"]
