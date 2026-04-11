"""Actor-critic policies for ``HybridActionPPO``."""

from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
)
from stable_baselines3.common.type_aliases import Schedule
from torch import nn

from sb3x.common.hybrid_action import HybridActionSpec, make_hybrid_action_spec

from .distributions import HybridActionDistribution


class HybridActionActorCriticPolicy(ActorCriticPolicy):
    """Actor-critic policy with a hybrid continuous/discrete action head."""

    action_dist: HybridActionDistribution

    def _make_action_dist(self) -> HybridActionDistribution:
        return HybridActionDistribution(self.hybrid_action_spec)

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
        features_extractor_class: type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: dict[str, Any] | None = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
        hybrid_action_space: spaces.Dict | None = None,
    ) -> None:
        if hybrid_action_space is None:
            raise ValueError("HybridActionPPO policies require hybrid_action_space")
        if use_sde:
            raise ValueError("HybridActionPPO does not support gSDE")

        self.hybrid_action_spec = make_hybrid_action_spec(hybrid_action_space)
        _validate_flat_action_space(action_space, self.hybrid_action_spec)

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
        )

    def _build(self, lr_schedule: Schedule) -> None:
        """Create SB3 actor-critic modules with a hybrid action distribution."""
        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi
        self.action_dist = self._make_action_dist()
        self.action_net, self.log_std = self.action_dist.proba_distribution_net(
            latent_dim=latent_dim_pi,
            log_std_init=self.log_std_init,
        )
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        if self.ortho_init:
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            if not self.share_features_extractor:
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        optimizer_kwargs = dict(self.optimizer_kwargs)
        optimizer_kwargs["lr"] = lr_schedule(1)
        self.optimizer = self.optimizer_class(
            self.parameters(),
            **optimizer_kwargs,
        )

    def _get_action_dist_from_latent(
        self,
        latent_pi: th.Tensor,
    ) -> HybridActionDistribution:
        action_params = self.action_net(latent_pi)
        return self.action_dist.proba_distribution(action_params, self.log_std)


class HybridActionActorCriticCnnPolicy(HybridActionActorCriticPolicy):
    """CNN policy entrypoint for hybrid action PPO."""

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


class HybridActionMultiInputActorCriticPolicy(HybridActionActorCriticPolicy):
    """Multi-input policy entrypoint for hybrid action PPO."""

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


def _validate_flat_action_space(
    action_space: spaces.Box,
    spec: HybridActionSpec,
) -> None:
    if not isinstance(action_space, spaces.Box):
        raise TypeError("HybridActionPPO policy expects the internal flat Box space")
    if action_space.shape != (spec.flat_dim,):
        raise ValueError(
            f"Flat action space shape {action_space.shape} does not match "
            f"hybrid action size {(spec.flat_dim,)}"
        )


MlpPolicy = HybridActionActorCriticPolicy
CnnPolicy = HybridActionActorCriticCnnPolicy
MultiInputPolicy = HybridActionMultiInputActorCriticPolicy

__all__ = [
    "CnnPolicy",
    "HybridActionActorCriticCnnPolicy",
    "HybridActionActorCriticPolicy",
    "HybridActionMultiInputActorCriticPolicy",
    "MlpPolicy",
    "MultiInputPolicy",
]
