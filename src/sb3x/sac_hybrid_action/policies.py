"""SAC policies for ``Dict(continuous=Box, discrete=MultiDiscrete)`` actions."""

from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
)
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from stable_baselines3.sac.policies import LOG_STD_MAX, LOG_STD_MIN, SACPolicy
from torch import nn

from sb3x.common.hybrid_action import (
    HybridActionSpec,
    combine_hybrid_actions,
    make_hybrid_action_spec,
)
from sb3x.common.maskable import MaybeMasks
from sb3x.common.maskable.distributions import MASKED_LOGIT_VALUE

from .encoding import (
    encode_scaled_hybrid_actions_for_critic,
    scale_discrete_actions,
    validate_sac_discrete_dims,
)


class HybridActionSACActor(BasePolicy):
    """SAC actor with a squashed Gaussian branch and categorical branches."""

    action_space: spaces.Box

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: list[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        full_std: bool = True,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        normalize_images: bool = True,
        hybrid_action_space: spaces.Dict | None = None,
        max_discrete_combinations: int = 1024,
    ) -> None:
        if hybrid_action_space is None:
            raise ValueError("HybridActionSAC policies require hybrid_action_space")
        if use_sde:
            raise ValueError("HybridActionSAC does not support gSDE")
        del full_std, use_expln, clip_mean

        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )

        self.hybrid_action_space = hybrid_action_space
        self.hybrid_action_spec = make_hybrid_action_spec(hybrid_action_space)
        validate_sac_discrete_dims(self.hybrid_action_spec)
        _validate_flat_action_space(action_space, self.hybrid_action_spec)

        self.use_sde = False
        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn
        self.log_std_init = log_std_init
        self.max_discrete_combinations = max_discrete_combinations

        latent_pi_net = create_mlp(features_dim, -1, net_arch, activation_fn)
        self.latent_pi = nn.Sequential(*latent_pi_net)
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else features_dim

        self.action_dist = SquashedDiagGaussianDistribution(
            self.hybrid_action_spec.continuous_dim,
        )
        self.mu = nn.Linear(last_layer_dim, self.hybrid_action_spec.continuous_dim)
        self.log_std = nn.Linear(
            last_layer_dim,
            self.hybrid_action_spec.continuous_dim,
        )
        self.discrete_logits = nn.Linear(
            last_layer_dim,
            self.hybrid_action_spec.discrete_logits_dim,
        )

        discrete_actions = _enumerate_discrete_actions(
            self.hybrid_action_spec,
            max_discrete_combinations=max_discrete_combinations,
        )
        self.register_buffer(
            "_all_discrete_actions",
            discrete_actions,
            persistent=False,
        )

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            {
                "net_arch": self.net_arch,
                "features_dim": self.features_dim,
                "activation_fn": self.activation_fn,
                "use_sde": False,
                "log_std_init": self.log_std_init,
                "full_std": True,
                "use_expln": False,
                "features_extractor": self.features_extractor,
                "clip_mean": 0.0,
                "hybrid_action_space": self.hybrid_action_space,
                "max_discrete_combinations": self.max_discrete_combinations,
            }
        )
        return data

    def get_action_dist_params(
        self,
        obs: PyTorchObs,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Return continuous Gaussian params and discrete branch logits."""
        features = self.extract_features(obs, self.features_extractor)
        latent_pi = self.latent_pi(features)
        mean_actions = self.mu(latent_pi)
        log_std = th.clamp(self.log_std(latent_pi), LOG_STD_MIN, LOG_STD_MAX)
        discrete_logits = self.discrete_logits(latent_pi)
        return mean_actions, log_std, discrete_logits

    def forward(
        self,
        obs: PyTorchObs,
        deterministic: bool = False,
        action_masks: MaybeMasks = None,
    ) -> th.Tensor:
        mean_actions, log_std, discrete_logits = self.get_action_dist_params(obs)
        continuous_actions = self.action_dist.actions_from_params(
            mean_actions,
            log_std,
            deterministic=deterministic,
        )
        discrete_actions, _ = self._discrete_actions_and_log_prob(
            discrete_logits,
            deterministic=deterministic,
            action_masks=action_masks,
        )
        scaled_discrete_actions = scale_discrete_actions(
            self.hybrid_action_spec,
            discrete_actions,
        )
        return combine_hybrid_actions(continuous_actions, scaled_discrete_actions)

    def action_log_prob(
        self,
        obs: PyTorchObs,
        action_masks: MaybeMasks = None,
    ) -> tuple[th.Tensor, th.Tensor]:
        mean_actions, log_std, discrete_logits = self.get_action_dist_params(obs)
        continuous_actions, continuous_log_prob = (
            self.continuous_action_log_prob_from_params(mean_actions, log_std)
        )
        discrete_actions, discrete_log_prob = self._discrete_actions_and_log_prob(
            discrete_logits,
            deterministic=False,
            action_masks=action_masks,
        )
        scaled_discrete_actions = scale_discrete_actions(
            self.hybrid_action_spec,
            discrete_actions,
        )
        actions = combine_hybrid_actions(continuous_actions, scaled_discrete_actions)
        return actions, continuous_log_prob + discrete_log_prob

    def continuous_action_log_prob_from_params(
        self,
        mean_actions: th.Tensor,
        log_std: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor]:
        """Sample the continuous branch with SAC's reparameterized Gaussian."""
        return self.action_dist.log_prob_from_params(mean_actions, log_std)

    def all_scaled_discrete_actions(self) -> th.Tensor:
        """Return every discrete action combination in scaled Box coordinates."""
        return scale_discrete_actions(
            self.hybrid_action_spec,
            self._all_discrete_action_tensor(),
        )

    def discrete_log_prob_matrix(
        self,
        discrete_logits: th.Tensor,
        action_masks: MaybeMasks = None,
    ) -> th.Tensor:
        """Return log-probabilities for all enumerated discrete combinations."""
        all_discrete_actions = self._all_discrete_action_tensor()
        branch_log_probs = self._branch_log_probs(discrete_logits, action_masks)
        log_probs = th.zeros(
            (discrete_logits.shape[0], all_discrete_actions.shape[0]),
            device=discrete_logits.device,
            dtype=discrete_logits.dtype,
        )
        for branch_idx, branch_probs in enumerate(branch_log_probs):
            branch_actions = all_discrete_actions[:, branch_idx]
            log_probs = log_probs + branch_probs[:, branch_actions]
        return log_probs

    def _predict(
        self, observation: PyTorchObs, deterministic: bool = False
    ) -> th.Tensor:
        return self(observation, deterministic)

    def _all_discrete_action_tensor(self) -> th.Tensor:
        all_discrete_actions = self._all_discrete_actions
        if not isinstance(all_discrete_actions, th.Tensor):
            raise TypeError("Expected discrete-action buffer to be a tensor")
        return all_discrete_actions

    def _discrete_actions_and_log_prob(
        self,
        discrete_logits: th.Tensor,
        *,
        deterministic: bool,
        action_masks: MaybeMasks = None,
    ) -> tuple[th.Tensor, th.Tensor]:
        branch_logits = th.split(
            discrete_logits,
            self.hybrid_action_spec.discrete_action_dims,
            dim=1,
        )
        branch_masks = self._split_action_masks(discrete_logits, action_masks)
        actions: list[th.Tensor] = []
        log_probs: list[th.Tensor] = []
        for logits, branch_mask in zip(branch_logits, branch_masks, strict=True):
            masked_logits = self._masked_logits(logits, branch_mask)
            if deterministic:
                branch_actions = th.argmax(masked_logits, dim=1)
            else:
                branch_actions = th.distributions.Categorical(
                    logits=masked_logits
                ).sample()
            actions.append(branch_actions)
            log_probs.append(
                F.log_softmax(masked_logits, dim=1)
                .gather(1, branch_actions.reshape(-1, 1))
                .squeeze(1)
            )
        return th.stack(actions, dim=1), th.stack(log_probs, dim=1).sum(dim=1)

    def _branch_log_probs(
        self,
        discrete_logits: th.Tensor,
        action_masks: MaybeMasks,
    ) -> list[th.Tensor]:
        branch_logits = th.split(
            discrete_logits,
            self.hybrid_action_spec.discrete_action_dims,
            dim=1,
        )
        branch_masks = self._split_action_masks(discrete_logits, action_masks)
        return [
            F.log_softmax(self._masked_logits(logits, branch_mask), dim=1)
            for logits, branch_mask in zip(branch_logits, branch_masks, strict=True)
        ]

    def _split_action_masks(
        self,
        discrete_logits: th.Tensor,
        action_masks: MaybeMasks,
    ) -> list[th.Tensor | None]:
        if action_masks is None:
            return [None for _ in self.hybrid_action_spec.discrete_action_dims]

        mask_tensor = th.as_tensor(
            action_masks,
            dtype=th.bool,
            device=discrete_logits.device,
        ).reshape(discrete_logits.shape[0], self.hybrid_action_spec.discrete_logits_dim)
        return list(
            th.split(mask_tensor, self.hybrid_action_spec.discrete_action_dims, dim=1)
        )

    def _masked_logits(
        self,
        logits: th.Tensor,
        action_mask: th.Tensor | None,
    ) -> th.Tensor:
        if action_mask is None:
            return logits
        if not action_mask.any(dim=1).all():
            raise ValueError("Each discrete action branch must have a valid action")

        huge_negative = th.tensor(
            MASKED_LOGIT_VALUE,
            dtype=logits.dtype,
            device=logits.device,
        )
        return th.where(action_mask, logits, huge_negative)


class HybridActionContinuousCritic(ContinuousCritic):
    """Continuous SAC critic that receives one-hot encoded discrete actions."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: list[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        activation_fn: type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
        hybrid_action_space: spaces.Dict | None = None,
    ) -> None:
        if hybrid_action_space is None:
            raise ValueError("HybridActionSAC critics require hybrid_action_space")
        self.hybrid_action_space = hybrid_action_space
        self.hybrid_action_spec = make_hybrid_action_spec(hybrid_action_space)
        validate_sac_discrete_dims(self.hybrid_action_spec)
        _validate_flat_action_space(action_space, self.hybrid_action_spec)

        critic_action_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(
                self.hybrid_action_spec.continuous_dim
                + self.hybrid_action_spec.discrete_logits_dim,
            ),
            dtype=np.float32,
        )
        super().__init__(
            observation_space=observation_space,
            action_space=critic_action_space,
            net_arch=net_arch,
            features_extractor=features_extractor,
            features_dim=features_dim,
            activation_fn=activation_fn,
            normalize_images=normalize_images,
            n_critics=n_critics,
            share_features_extractor=share_features_extractor,
        )

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> tuple[th.Tensor, ...]:
        encoded_actions = encode_scaled_hybrid_actions_for_critic(
            self.hybrid_action_spec,
            actions,
        )
        return super().forward(obs, encoded_actions)

    def q1_forward(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        encoded_actions = encode_scaled_hybrid_actions_for_critic(
            self.hybrid_action_spec,
            actions,
        )
        return super().q1_forward(obs, encoded_actions)


class HybridActionSACPolicy(SACPolicy):
    """SAC policy for hybrid continuous/discrete action spaces."""

    actor: HybridActionSACActor
    critic: HybridActionContinuousCritic
    critic_target: HybridActionContinuousCritic

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
        features_extractor_class: type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: dict[str, Any] | None = None,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
        hybrid_action_space: spaces.Dict | None = None,
        max_discrete_combinations: int = 1024,
    ) -> None:
        if hybrid_action_space is None:
            raise ValueError("HybridActionSAC policies require hybrid_action_space")
        if use_sde:
            raise ValueError("HybridActionSAC does not support gSDE")

        self.hybrid_action_space = hybrid_action_space
        self.hybrid_action_spec = make_hybrid_action_spec(hybrid_action_space)
        validate_sac_discrete_dims(self.hybrid_action_spec)
        _validate_flat_action_space(action_space, self.hybrid_action_spec)
        self.max_discrete_combinations = max_discrete_combinations

        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            use_sde=False,
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
        )

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            {
                "hybrid_action_space": self.hybrid_action_space,
                "max_discrete_combinations": self.max_discrete_combinations,
            }
        )
        return data

    def make_actor(
        self,
        features_extractor: BaseFeaturesExtractor | None = None,
    ) -> HybridActionSACActor:
        actor_kwargs = self._update_features_extractor(
            self.actor_kwargs,
            features_extractor,
        )
        actor_kwargs["hybrid_action_space"] = self.hybrid_action_space
        actor_kwargs["max_discrete_combinations"] = self.max_discrete_combinations
        return HybridActionSACActor(**actor_kwargs).to(self.device)

    def make_critic(
        self,
        features_extractor: BaseFeaturesExtractor | None = None,
    ) -> HybridActionContinuousCritic:
        critic_kwargs = self._update_features_extractor(
            self.critic_kwargs,
            features_extractor,
        )
        critic_kwargs["hybrid_action_space"] = self.hybrid_action_space
        return HybridActionContinuousCritic(**critic_kwargs).to(self.device)


class HybridActionSACCnnPolicy(HybridActionSACPolicy):
    """CNN policy entrypoint for hybrid-action SAC."""

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


class HybridActionSACMultiInputPolicy(HybridActionSACPolicy):
    """Multi-input policy entrypoint for hybrid-action SAC."""

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


def _enumerate_discrete_actions(
    spec: HybridActionSpec,
    *,
    max_discrete_combinations: int,
) -> th.Tensor:
    num_combinations = int(np.prod(spec.discrete_action_dims, dtype=np.int64))
    if num_combinations > max_discrete_combinations:
        raise ValueError(
            "HybridActionSAC enumerates the discrete branch exactly during "
            f"training, but this action space has {num_combinations} "
            f"combinations and max_discrete_combinations={max_discrete_combinations}"
        )
    return th.as_tensor(
        list(product(*(range(action_dim) for action_dim in spec.discrete_action_dims))),
        dtype=th.long,
    )


def _validate_flat_action_space(
    action_space: spaces.Box,
    spec: HybridActionSpec,
) -> None:
    if not isinstance(action_space, spaces.Box):
        raise TypeError("HybridActionSAC policy expects the internal flat Box space")
    if action_space.shape != (spec.flat_dim,):
        raise ValueError(
            f"Flat action space shape {action_space.shape} does not match "
            f"hybrid action size {(spec.flat_dim,)}"
        )


MlpPolicy = HybridActionSACPolicy
CnnPolicy = HybridActionSACCnnPolicy
MultiInputPolicy = HybridActionSACMultiInputPolicy

__all__ = [
    "CnnPolicy",
    "HybridActionContinuousCritic",
    "HybridActionSACActor",
    "HybridActionSACCnnPolicy",
    "HybridActionSACMultiInputPolicy",
    "HybridActionSACPolicy",
    "MlpPolicy",
    "MultiInputPolicy",
]
