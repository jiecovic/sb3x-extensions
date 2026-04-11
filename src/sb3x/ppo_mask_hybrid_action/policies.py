"""Mask-aware actor-critic policies for hybrid-action PPO."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    NatureCNN,
)
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from torch import nn

from sb3x.common.hybrid_action import MaskableHybridActionDistribution
from sb3x.common.maskable import MaybeMasks
from sb3x.ppo_hybrid_action.policies import HybridActionActorCriticPolicy


class MaskableHybridActionActorCriticPolicy(HybridActionActorCriticPolicy):
    """Hybrid action policy with masks applied to the discrete branch only."""

    action_dist: MaskableHybridActionDistribution

    def _make_action_dist(self) -> MaskableHybridActionDistribution:
        return MaskableHybridActionDistribution(self.hybrid_action_spec)

    def forward(
        self,
        obs: PyTorchObs,
        deterministic: bool = False,
        action_masks: MaybeMasks = None,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        latent_pi, latent_vf = self._extract_actor_critic_latents(obs)

        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        distribution.apply_masking(action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self._flat_action_shape()))
        return actions, values, log_prob

    def _get_action_dist_from_latent(
        self,
        latent_pi: th.Tensor,
    ) -> MaskableHybridActionDistribution:
        action_params = self.action_net(latent_pi)
        return self.action_dist.proba_distribution(action_params, self.log_std)

    def _predict(
        self,
        observation: PyTorchObs,
        deterministic: bool = False,
        action_masks: MaybeMasks = None,
    ) -> th.Tensor:
        return self.get_distribution(
            observation,
            action_masks=action_masks,
        ).get_actions(deterministic=deterministic)

    def predict(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
        action_masks: MaybeMasks = None,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None]:
        self.set_training_mode(False)

        if isinstance(observation, tuple):
            raise ValueError(
                "You have passed a tuple to predict() instead of an observation. "
                "Use `obs, info = env.reset()` with Gymnasium envs, but pass only "
                "`obs` to model.predict()."
            )

        obs_tensor, vectorized_env = self.obs_to_tensor(observation)
        with th.no_grad():
            actions = self._predict(
                obs_tensor,
                deterministic=deterministic,
                action_masks=action_masks,
            )
        actions = actions.cpu().numpy().reshape((-1, *self._flat_action_shape()))

        if isinstance(self.action_space, spaces.Box):
            actions = np.clip(actions, self.action_space.low, self.action_space.high)

        if not vectorized_env:
            actions = actions.squeeze(axis=0)

        return actions, state

    def evaluate_actions(
        self,
        obs: PyTorchObs,
        actions: th.Tensor,
        action_masks: MaybeMasks = None,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor | None]:
        latent_pi, latent_vf = self._extract_actor_critic_latents(obs)

        distribution = self._get_action_dist_from_latent(latent_pi)
        distribution.apply_masking(action_masks)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def _extract_actor_critic_latents(
        self,
        obs: PyTorchObs,
    ) -> tuple[th.Tensor, th.Tensor]:
        features = self.extract_features(obs)
        if self.share_features_extractor:
            if not isinstance(features, th.Tensor):
                raise TypeError("Expected shared feature extractor to return a tensor")
            return self.mlp_extractor(features)

        if not isinstance(features, tuple) or len(features) != 2:
            raise TypeError("Expected separate actor and critic feature tensors")
        pi_features, vf_features = features
        return (
            self.mlp_extractor.forward_actor(pi_features),
            self.mlp_extractor.forward_critic(vf_features),
        )

    def get_distribution(
        self,
        obs: PyTorchObs,
        action_masks: MaybeMasks = None,
    ) -> MaskableHybridActionDistribution:
        features = BasePolicy.extract_features(
            self,
            obs,
            self.pi_features_extractor,
        )
        latent_pi = self.mlp_extractor.forward_actor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        distribution.apply_masking(action_masks)
        return distribution

    def _flat_action_shape(self) -> tuple[int, ...]:
        action_shape = self.action_space.shape
        if action_shape is None:
            raise ValueError("MaskableHybridActionPPO requires a flat Box action space")
        return tuple(int(dim) for dim in action_shape)


class MaskableHybridActionActorCriticCnnPolicy(MaskableHybridActionActorCriticPolicy):
    """CNN policy entrypoint for maskable hybrid-action PPO."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: list[int] | dict[str, list[int]] | None = None,
        activation_fn: type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: dict[str, Any] | None = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
        hybrid_action_space: spaces.Dict | None = None,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
            use_sde=use_sde,
            log_std_init=log_std_init,
            full_std=full_std,
            use_expln=use_expln,
            squash_output=squash_output,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            share_features_extractor=share_features_extractor,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            hybrid_action_space=hybrid_action_space,
        )


class MaskableHybridActionMultiInputActorCriticPolicy(
    MaskableHybridActionActorCriticPolicy
):
    """Multi-input policy entrypoint for maskable hybrid-action PPO."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: list[int] | dict[str, list[int]] | None = None,
        activation_fn: type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: type[BaseFeaturesExtractor] = CombinedExtractor,
        features_extractor_kwargs: dict[str, Any] | None = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
        hybrid_action_space: spaces.Dict | None = None,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
            use_sde=use_sde,
            log_std_init=log_std_init,
            full_std=full_std,
            use_expln=use_expln,
            squash_output=squash_output,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            share_features_extractor=share_features_extractor,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            hybrid_action_space=hybrid_action_space,
        )


MlpPolicy = MaskableHybridActionActorCriticPolicy
CnnPolicy = MaskableHybridActionActorCriticCnnPolicy
MultiInputPolicy = MaskableHybridActionMultiInputActorCriticPolicy

__all__ = [
    "CnnPolicy",
    "MaskableHybridActionActorCriticCnnPolicy",
    "MaskableHybridActionActorCriticPolicy",
    "MaskableHybridActionMultiInputActorCriticPolicy",
    "MlpPolicy",
    "MultiInputPolicy",
]
