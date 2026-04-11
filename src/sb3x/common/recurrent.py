"""Shared typed helpers for recurrent policies."""

from __future__ import annotations

from typing import Protocol

import torch as th
from gymnasium import spaces
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from stable_baselines3.common.type_aliases import PyTorchObs
from torch import nn

FeatureTensor = th.Tensor | tuple[th.Tensor, th.Tensor]
LSTMState = tuple[th.Tensor, th.Tensor]
PolicyObs = th.Tensor | dict[str, th.Tensor]


class RecurrentValuePolicy(Protocol):
    """Minimal policy surface needed for recurrent terminal value bootstrapping."""

    def predict_values(
        self,
        obs: th.Tensor,
        lstm_states: LSTMState,
        episode_starts: th.Tensor,
    ) -> th.Tensor: ...


def split_actor_critic_features(
    features: FeatureTensor,
    *,
    share_features_extractor: bool,
) -> tuple[th.Tensor, th.Tensor]:
    """Narrow SB3's feature-extractor output to actor/value tensors."""
    if share_features_extractor:
        if not isinstance(features, th.Tensor):
            raise TypeError("Expected shared feature extractor to return one tensor")
        return features, features

    if isinstance(features, tuple) and len(features) == 2:
        return features
    raise TypeError("Expected separate actor and critic feature tensors")


def require_lstm_state(state: tuple[th.Tensor, ...] | th.Tensor) -> LSTMState:
    """Narrow a recurrent state tuple to the expected hidden/cell pair."""
    if isinstance(state, th.Tensor):
        raise TypeError("Expected hidden/cell state pair, received one tensor")
    if len(state) != 2:
        raise TypeError(
            f"Expected hidden/cell state pair, received {len(state)} tensors"
        )
    return (state[0], state[1])


def require_linear(module: nn.Linear | None) -> nn.Linear:
    """Require an initialized linear projection."""
    if module is None:
        raise TypeError("Expected critic projection to be initialized")
    return module


def count_vectorized_envs(observation: PyTorchObs) -> int:
    """Return the number of environments represented by an observation batch."""
    if isinstance(observation, dict):
        return observation[next(iter(observation.keys()))].shape[0]
    return observation.shape[0]


def action_shape(action_space: spaces.Space) -> tuple[int, ...]:
    """Return a non-optional action shape tuple."""
    shape = action_space.shape
    if shape is None:
        raise TypeError("Expected action space shape to be defined")
    return shape


def make_recurrent_states(
    lstm: nn.LSTM,
    *,
    n_envs: int,
    device: th.device | str,
) -> RNNStates:
    """Create zero actor/value LSTM states for rollout collection."""
    state_shape = (lstm.num_layers, n_envs, lstm.hidden_size)
    return RNNStates(
        (
            th.zeros(state_shape, device=device),
            th.zeros(state_shape, device=device),
        ),
        (
            th.zeros(state_shape, device=device),
            th.zeros(state_shape, device=device),
        ),
    )


def recurrent_hidden_state_buffer_shape(
    lstm: nn.LSTM,
    *,
    n_steps: int,
    n_envs: int,
) -> tuple[int, int, int, int]:
    """Return the recurrent rollout-buffer hidden-state storage shape."""
    return (n_steps, lstm.num_layers, n_envs, lstm.hidden_size)


def predict_recurrent_values(
    policy: RecurrentValuePolicy,
    obs: PyTorchObs,
    lstm_states: LSTMState,
    episode_starts: th.Tensor,
) -> th.Tensor:
    """Call the weakly typed SB3-Contrib recurrent value boundary."""
    return policy.predict_values(
        obs,  # pyright: ignore[reportArgumentType]
        lstm_states,
        episode_starts,
    )
