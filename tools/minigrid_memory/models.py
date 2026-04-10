"""Repo-local MiniGrid model components used by benchmark runs.

The CNN extractor follows the shape recommended by the official MiniGrid
training guide: image-only observations with a custom feature extractor rather
than flattening the symbolic grid into one long vector.
"""

from __future__ import annotations

import gymnasium as gym
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    """Small CNN for MiniGrid image observations.

    MiniGrid's compact observation image is only ``7x7x3`` by default, so the
    standard NatureCNN used by SB3 is not a great fit. This extractor keeps the
    spatial structure intact, then projects into one fixed feature vector before
    the recurrent core.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 128,
    ) -> None:
        super().__init__(observation_space, features_dim)
        if not isinstance(observation_space, gym.spaces.Box):
            raise TypeError(
                "MinigridFeaturesExtractor requires a Box observation space"
            )
        if len(observation_space.shape) != 3:
            raise ValueError(
                "MinigridFeaturesExtractor expects channel-first image observations"
            )

        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """Extract one dense feature vector per MiniGrid observation."""
        return self.linear(self.cnn(observations))


__all__ = ["MinigridFeaturesExtractor"]
