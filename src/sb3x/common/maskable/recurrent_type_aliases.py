"""Type aliases for recurrent rollout samples with action masks."""

from __future__ import annotations

from typing import NamedTuple

import torch as th
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from stable_baselines3.common.type_aliases import TensorDict


class MaskableRecurrentRolloutBufferSamples(NamedTuple):
    """Minibatch sampled from a recurrent rollout buffer with action masks."""

    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    lstm_states: RNNStates
    episode_starts: th.Tensor
    sequence_mask: th.Tensor
    action_masks: th.Tensor


class MaskableRecurrentDictRolloutBufferSamples(NamedTuple):
    """Dict-observation minibatch sampled from a recurrent maskable buffer."""

    observations: TensorDict
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    lstm_states: RNNStates
    episode_starts: th.Tensor
    sequence_mask: th.Tensor
    action_masks: th.Tensor
