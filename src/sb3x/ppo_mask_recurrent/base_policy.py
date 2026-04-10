"""Base recurrent actor-critic policy with invalid-action masking support."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch as th
from gymnasium import spaces
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
)
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from torch import nn

from sb3x.common.maskable import (
    MaskableDistribution,
    MaybeMasks,
    make_masked_proba_distribution,
)

FeatureTensor = th.Tensor | tuple[th.Tensor, th.Tensor]
LSTMState = tuple[th.Tensor, th.Tensor]


def _split_actor_critic_features(
    features: FeatureTensor,
    *,
    share_features_extractor: bool,
) -> tuple[th.Tensor, th.Tensor]:
    """Narrow SB3's feature-extractor output to explicit actor/value tensors."""
    if share_features_extractor:
        if not isinstance(features, th.Tensor):
            raise TypeError("Expected shared feature extractor to return one tensor")
        return features, features

    if isinstance(features, tuple) and len(features) == 2:
        return features
    raise TypeError("Expected separate actor and critic feature tensors")


def _require_lstm_state(state: tuple[th.Tensor, ...] | th.Tensor) -> LSTMState:
    """Narrow a recurrent state tuple to the expected hidden/cell pair."""
    if isinstance(state, th.Tensor):
        raise TypeError("Expected hidden/cell state pair, received one tensor")
    if len(state) != 2:
        raise TypeError(
            f"Expected hidden/cell state pair, received {len(state)} tensors"
        )
    return (state[0], state[1])


def _require_linear(module: nn.Linear | None) -> nn.Linear:
    """Require an initialized linear projection."""
    if module is None:
        raise TypeError("Expected critic projection to be initialized")
    return module


def _count_vectorized_envs(observation: PyTorchObs) -> int:
    """Return the number of environments represented by an observation batch."""
    if isinstance(observation, dict):
        return observation[next(iter(observation.keys()))].shape[0]
    return observation.shape[0]


def _action_shape(action_space: spaces.Space) -> tuple[int, ...]:
    """Return a non-optional action shape tuple."""
    shape = action_space.shape
    if shape is None:
        raise TypeError("Expected action space shape to be defined")
    return shape


class MaskableRecurrentActorCriticPolicy(RecurrentActorCriticPolicy):
    """Recurrent actor-critic policy with invalid-action masking support.

    This keeps the upstream recurrent actor-critic layout, but replaces the
    action distribution with a mask-aware variant and threads `action_masks`
    through inference and action evaluation.
    """

    action_dist: MaskableDistribution

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
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
        lstm_hidden_size: int = 256,
        n_lstm_layers: int = 1,
        shared_lstm: bool = False,
        enable_critic_lstm: bool = True,
        lstm_kwargs: dict[str, Any] | None = None,
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
            lstm_hidden_size=lstm_hidden_size,
            n_lstm_layers=n_lstm_layers,
            shared_lstm=shared_lstm,
            enable_critic_lstm=enable_critic_lstm,
            lstm_kwargs=lstm_kwargs,
        )
        self.action_dist = make_masked_proba_distribution(action_space)

    def _get_action_dist_from_latent(
        self,
        latent_pi: th.Tensor,
    ) -> MaskableDistribution:
        action_logits = self.action_net(latent_pi)
        return self.action_dist.proba_distribution(action_logits=action_logits)

    def forward(
        self,
        obs: PyTorchObs,
        lstm_states: RNNStates,
        episode_starts: th.Tensor,
        deterministic: bool = False,
        action_masks: MaybeMasks = None,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, RNNStates]:
        """Compute masked actions, values, and updated recurrent state."""
        features = self.extract_features(obs)
        pi_features, vf_features = _split_actor_critic_features(
            features,
            share_features_extractor=self.share_features_extractor,
        )

        latent_pi, raw_lstm_states_pi = self._process_sequence(
            pi_features,
            _require_lstm_state(lstm_states.pi),
            episode_starts,
            self.lstm_actor,
        )
        lstm_states_pi = _require_lstm_state(raw_lstm_states_pi)
        if self.lstm_critic is not None:
            latent_vf, raw_lstm_states_vf = self._process_sequence(
                vf_features,
                _require_lstm_state(lstm_states.vf),
                episode_starts,
                self.lstm_critic,
            )
            lstm_states_vf = _require_lstm_state(raw_lstm_states_vf)
        elif self.shared_lstm:
            latent_vf = latent_pi.detach()
            lstm_states_vf = (
                lstm_states_pi[0].detach(),
                lstm_states_pi[1].detach(),
            )
        else:
            latent_vf = _require_linear(self.critic)(vf_features)
            lstm_states_vf = lstm_states_pi

        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *_action_shape(self.action_space)))
        return actions, values, log_prob, RNNStates(lstm_states_pi, lstm_states_vf)

    def get_distribution(
        self,
        obs: PyTorchObs,
        lstm_states: LSTMState,
        episode_starts: th.Tensor,
        action_masks: MaybeMasks = None,
    ) -> tuple[MaskableDistribution, LSTMState]:
        """Return the masked policy distribution and next actor LSTM state."""
        features = super(ActorCriticPolicy, self).extract_features(
            obs,
            self.pi_features_extractor,
        )
        if not isinstance(features, th.Tensor):
            raise TypeError("Expected actor feature extractor to return a tensor")

        latent_pi, raw_next_lstm_states = self._process_sequence(
            features,
            lstm_states,
            episode_starts,
            self.lstm_actor,
        )
        next_lstm_states = _require_lstm_state(raw_next_lstm_states)
        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        distribution = self._get_action_dist_from_latent(latent_pi)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        return distribution, next_lstm_states

    def evaluate_actions(
        self,
        obs: PyTorchObs,
        actions: th.Tensor,
        lstm_states: RNNStates,
        episode_starts: th.Tensor,
        action_masks: MaybeMasks = None,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor | None]:
        """Evaluate masked log-probs and entropy for a training minibatch."""
        features = self.extract_features(obs)
        pi_features, vf_features = _split_actor_critic_features(
            features,
            share_features_extractor=self.share_features_extractor,
        )

        latent_pi, _ = self._process_sequence(
            pi_features,
            _require_lstm_state(lstm_states.pi),
            episode_starts,
            self.lstm_actor,
        )
        if self.lstm_critic is not None:
            latent_vf, _ = self._process_sequence(
                vf_features,
                _require_lstm_state(lstm_states.vf),
                episode_starts,
                self.lstm_critic,
            )
        elif self.shared_lstm:
            latent_vf = latent_pi.detach()
        else:
            latent_vf = _require_linear(self.critic)(vf_features)

        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        distribution = self._get_action_dist_from_latent(latent_pi)
        if action_masks is not None:
            distribution.apply_masking(action_masks)

        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()

    def _predict(
        self,
        observation: PyTorchObs,
        lstm_states: LSTMState,
        episode_starts: th.Tensor,
        deterministic: bool = False,
        action_masks: MaybeMasks = None,
    ) -> tuple[th.Tensor, LSTMState]:
        """Predict the next action tensor and recurrent state."""
        distribution, next_lstm_states = self.get_distribution(
            observation,
            lstm_states,
            episode_starts,
            action_masks=action_masks,
        )
        return distribution.get_actions(deterministic=deterministic), next_lstm_states

    def predict(  # type: ignore[override]
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
        action_masks: np.ndarray | None = None,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None]:
        """Predict one action, optionally constrained by invalid-action masks."""
        self.set_training_mode(False)

        obs_tensor, vectorized_env = self.obs_to_tensor(observation)
        n_envs = _count_vectorized_envs(obs_tensor)

        if state is None:
            zeros = np.concatenate(
                [np.zeros(self.lstm_hidden_state_shape) for _ in range(n_envs)],
                axis=1,
            )
            state = (zeros, zeros)

        if episode_start is None:
            episode_start = np.zeros(n_envs, dtype=bool)

        with th.no_grad():
            tensor_states = (
                th.tensor(state[0], dtype=th.float32, device=self.device),
                th.tensor(state[1], dtype=th.float32, device=self.device),
            )
            episode_starts = th.tensor(
                episode_start,
                dtype=th.float32,
                device=self.device,
            )
            actions, next_states = self._predict(
                obs_tensor,
                lstm_states=tensor_states,
                episode_starts=episode_starts,
                deterministic=deterministic,
                action_masks=action_masks,
            )
            next_states_np = (
                next_states[0].cpu().numpy(),
                next_states[1].cpu().numpy(),
            )

        actions_np = (
            actions.cpu().numpy().reshape((-1, *_action_shape(self.action_space)))
        )
        if not vectorized_env:
            actions_np = actions_np.squeeze(axis=0)

        return actions_np, next_states_np


__all__ = ["LSTMState", "MaskableRecurrentActorCriticPolicy"]
