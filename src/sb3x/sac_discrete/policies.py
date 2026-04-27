"""SAC policies for finite discrete action spaces."""

from __future__ import annotations

from typing import Any

import torch as th
from gymnasium import spaces
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    get_actor_critic_arch,
)
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from torch import nn

from .actor import DiscreteSACActor
from .critic import DiscreteSACCritic


class DiscreteSACPolicy(BasePolicy):
    """Policy container with categorical actor and twin discrete-action critics."""

    action_space: spaces.Discrete
    actor: DiscreteSACActor
    critic: DiscreteSACCritic
    critic_target: DiscreteSACCritic

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        net_arch: list[int] | dict[str, list[int]] | None = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        features_extractor_class: type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: dict[str, Any] | None = None,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            normalize_images=normalize_images,
        )
        if net_arch is None:
            net_arch = [256, 256]

        actor_arch, critic_arch = get_actor_critic_arch(net_arch)
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.actor_kwargs = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": actor_arch,
            "activation_fn": activation_fn,
            "normalize_images": normalize_images,
        }
        self.critic_kwargs = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": critic_arch,
            "activation_fn": activation_fn,
            "normalize_images": normalize_images,
            "n_critics": n_critics,
            "share_features_extractor": share_features_extractor,
        }
        self.share_features_extractor = share_features_extractor
        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        self.actor = self.make_actor()
        actor_optimizer_kwargs = dict(self.optimizer_kwargs)
        actor_optimizer_kwargs["lr"] = lr_schedule(1)
        self.actor.optimizer = self.optimizer_class(
            self.actor.parameters(),
            **actor_optimizer_kwargs,
        )

        if self.share_features_extractor:
            self.critic = self.make_critic(
                features_extractor=self.actor.features_extractor,
            )
            critic_parameters = [
                param
                for name, param in self.critic.named_parameters()
                if "features_extractor" not in name
            ]
        else:
            self.critic = self.make_critic(features_extractor=None)
            critic_parameters = list(self.critic.parameters())

        self.critic_target = self.make_critic(features_extractor=None)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.set_training_mode(False)

        critic_optimizer_kwargs = dict(self.optimizer_kwargs)
        critic_optimizer_kwargs["lr"] = lr_schedule(1)
        self.critic.optimizer = self.optimizer_class(
            critic_parameters,
            **critic_optimizer_kwargs,
        )

    def make_actor(
        self,
        features_extractor: BaseFeaturesExtractor | None = None,
    ) -> DiscreteSACActor:
        actor_kwargs = self._update_features_extractor(
            self.actor_kwargs,
            features_extractor,
        )
        return DiscreteSACActor(**actor_kwargs).to(self.device)

    def make_critic(
        self,
        features_extractor: BaseFeaturesExtractor | None = None,
    ) -> DiscreteSACCritic:
        critic_kwargs = self._update_features_extractor(
            self.critic_kwargs,
            features_extractor,
        )
        return DiscreteSACCritic(**critic_kwargs).to(self.device)

    def forward(self, obs: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def _predict(
        self,
        observation: PyTorchObs,
        deterministic: bool = False,
    ) -> th.Tensor:
        return self.actor(observation, deterministic=deterministic)

    def set_training_mode(self, mode: bool) -> None:
        self.actor.set_training_mode(mode)
        self.critic.set_training_mode(mode)
        self.training = mode

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            {
                "net_arch": self.net_arch,
                "activation_fn": self.activation_fn,
                "lr_schedule": self._dummy_schedule,
                "optimizer_class": self.optimizer_class,
                "optimizer_kwargs": self.optimizer_kwargs,
                "features_extractor_class": self.features_extractor_class,
                "features_extractor_kwargs": self.features_extractor_kwargs,
                "n_critics": self.critic_kwargs["n_critics"],
                "share_features_extractor": self.share_features_extractor,
            }
        )
        return data


class DiscreteSACCnnPolicy(DiscreteSACPolicy):
    """CNN policy entrypoint for DiscreteSAC."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        net_arch: list[int] | dict[str, list[int]] | None = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        features_extractor_class: type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: dict[str, Any] | None = None,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            n_critics=n_critics,
            share_features_extractor=share_features_extractor,
        )


class DiscreteSACMultiInputPolicy(DiscreteSACPolicy):
    """Multi-input policy entrypoint for DiscreteSAC."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        net_arch: list[int] | dict[str, list[int]] | None = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        features_extractor_class: type[BaseFeaturesExtractor] = CombinedExtractor,
        features_extractor_kwargs: dict[str, Any] | None = None,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            n_critics=n_critics,
            share_features_extractor=share_features_extractor,
        )


MlpPolicy = DiscreteSACPolicy
CnnPolicy = DiscreteSACCnnPolicy
MultiInputPolicy = DiscreteSACMultiInputPolicy

__all__ = [
    "CnnPolicy",
    "DiscreteSACActor",
    "DiscreteSACCnnPolicy",
    "DiscreteSACCritic",
    "DiscreteSACMultiInputPolicy",
    "DiscreteSACPolicy",
    "MlpPolicy",
    "MultiInputPolicy",
]
