"""Mask-aware SAC policies for hybrid continuous/discrete action spaces."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    NatureCNN,
)
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from torch import nn

from sb3x.common.maskable import MaybeMasks
from sb3x.sac_hybrid_action.policies import HybridActionSACPolicy


class MaskableHybridActionSACPolicy(HybridActionSACPolicy):
    """Hybrid SAC policy whose discrete branch can consume action masks."""

    def _predict(
        self,
        observation: PyTorchObs,
        deterministic: bool = False,
        action_masks: MaybeMasks = None,
    ) -> th.Tensor:
        return self.actor(
            observation,
            deterministic=deterministic,
            action_masks=action_masks,
        )

    def predict(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
        action_masks: MaybeMasks = None,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None]:
        del episode_start
        self.set_training_mode(False)

        raw_observation: object = observation
        if isinstance(raw_observation, tuple) and len(raw_observation) == 2:
            _, maybe_info = raw_observation
            if isinstance(maybe_info, dict):
                raise ValueError(
                    "You passed a Gym reset tuple to predict(); pass only the "
                    "observation."
                )

        obs_tensor, vectorized_env = self.obs_to_tensor(observation)
        with th.no_grad():
            actions = self._predict(
                obs_tensor,
                deterministic=deterministic,
                action_masks=action_masks,
            )

        if self.action_space.shape is None:
            raise ValueError("Action space must define a shape")
        actions = actions.cpu().numpy().reshape((-1, *self.action_space.shape))
        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                actions = self.unscale_action(actions)
            else:
                actions = np.clip(
                    actions,
                    self.action_space.low,
                    self.action_space.high,
                )

        if not vectorized_env:
            actions = actions.squeeze(axis=0)

        return actions, state


class MaskableHybridActionSACCnnPolicy(MaskableHybridActionSACPolicy):
    """CNN policy entrypoint for maskable hybrid-action SAC."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: list[int] | dict[str, list[int]] | None = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: dict[str, Any] | None = None,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
        hybrid_action_space: spaces.Dict | None = None,
        max_discrete_combinations: int = 1024,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            use_sde=use_sde,
            log_std_init=log_std_init,
            use_expln=use_expln,
            clip_mean=clip_mean,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            n_critics=n_critics,
            share_features_extractor=share_features_extractor,
            hybrid_action_space=hybrid_action_space,
            max_discrete_combinations=max_discrete_combinations,
        )


class MaskableHybridActionSACMultiInputPolicy(MaskableHybridActionSACPolicy):
    """Multi-input policy entrypoint for maskable hybrid-action SAC."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: list[int] | dict[str, list[int]] | None = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: type[BaseFeaturesExtractor] = CombinedExtractor,
        features_extractor_kwargs: dict[str, Any] | None = None,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
        hybrid_action_space: spaces.Dict | None = None,
        max_discrete_combinations: int = 1024,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            use_sde=use_sde,
            log_std_init=log_std_init,
            use_expln=use_expln,
            clip_mean=clip_mean,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            n_critics=n_critics,
            share_features_extractor=share_features_extractor,
            hybrid_action_space=hybrid_action_space,
            max_discrete_combinations=max_discrete_combinations,
        )


MlpPolicy = MaskableHybridActionSACPolicy
CnnPolicy = MaskableHybridActionSACCnnPolicy
MultiInputPolicy = MaskableHybridActionSACMultiInputPolicy

__all__ = [
    "CnnPolicy",
    "MaskableHybridActionSACCnnPolicy",
    "MaskableHybridActionSACMultiInputPolicy",
    "MaskableHybridActionSACPolicy",
    "MlpPolicy",
    "MultiInputPolicy",
]
