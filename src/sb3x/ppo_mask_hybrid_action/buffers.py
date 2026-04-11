"""Rollout buffers for maskable hybrid-action PPO."""

from __future__ import annotations

from collections.abc import Generator
from typing import NamedTuple

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.vec_env import VecNormalize

from sb3x.common.maskable import make_all_valid_action_masks, reshape_action_masks


class MaskableHybridActionRolloutBufferSamples(NamedTuple):
    """Rollout samples plus masks for the discrete hybrid-action branch."""

    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    action_masks: th.Tensor


class MaskableHybridActionDictRolloutBufferSamples(NamedTuple):
    """Dict-observation rollout samples plus discrete-branch masks."""

    observations: TensorDict
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    action_masks: th.Tensor


class MaskableHybridActionRolloutBuffer(RolloutBuffer):
    """Rollout buffer that stores masks for the discrete action branch."""

    action_masks: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: th.device | str = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
        mask_dims: int = 0,
    ) -> None:
        if mask_dims <= 0:
            raise ValueError("mask_dims must be positive")
        self.mask_dims = mask_dims
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            gae_lambda,
            gamma,
            n_envs,
        )

    def reset(self) -> None:
        self.action_masks = make_all_valid_action_masks(
            buffer_size=self.buffer_size,
            n_envs=self.n_envs,
            mask_dims=self.mask_dims,
        )
        super().reset()

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
        action_masks: np.ndarray | None = None,
    ) -> None:
        """Store masks for the current rollout position when provided."""
        if action_masks is not None:
            self.action_masks[self.pos] = reshape_action_masks(
                action_masks,
                n_envs=self.n_envs,
                mask_dims=self.mask_dims,
            )
        super().add(obs, action, reward, episode_start, value, log_prob)

    def get(
        self,
        batch_size: int | None = None,
    ) -> Generator[MaskableHybridActionRolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)

        if not self.generator_ready:
            for tensor in [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "action_masks",
            ]:
                setattr(self, tensor, self.swap_and_flatten(getattr(self, tensor)))
            self.generator_ready = True

        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: VecNormalize | None = None,
    ) -> MaskableHybridActionRolloutBufferSamples:
        del env
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.action_masks[batch_inds].reshape(-1, self.mask_dims),
        )
        return MaskableHybridActionRolloutBufferSamples(*map(self.to_torch, data))


class MaskableHybridActionDictRolloutBuffer(DictRolloutBuffer):
    """Dict-observation rollout buffer with discrete-branch action masks."""

    action_masks: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        device: th.device | str = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
        mask_dims: int = 0,
    ) -> None:
        if mask_dims <= 0:
            raise ValueError("mask_dims must be positive")
        self.mask_dims = mask_dims
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            gae_lambda,
            gamma,
            n_envs=n_envs,
        )

    def reset(self) -> None:
        self.action_masks = make_all_valid_action_masks(
            buffer_size=self.buffer_size,
            n_envs=self.n_envs,
            mask_dims=self.mask_dims,
        )
        super().reset()

    def add(
        self,
        obs: dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
        action_masks: np.ndarray | None = None,
    ) -> None:
        """Store masks for the current rollout position when provided."""
        if action_masks is not None:
            self.action_masks[self.pos] = reshape_action_masks(
                action_masks,
                n_envs=self.n_envs,
                mask_dims=self.mask_dims,
            )
        super().add(obs, action, reward, episode_start, value, log_prob)

    def get(
        self,
        batch_size: int | None = None,
    ) -> Generator[MaskableHybridActionDictRolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)

        if not self.generator_ready:
            for key, obs in self.observations.items():
                self.observations[key] = self.swap_and_flatten(obs)

            for tensor in [
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "action_masks",
            ]:
                setattr(self, tensor, self.swap_and_flatten(getattr(self, tensor)))
            self.generator_ready = True

        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: VecNormalize | None = None,
    ) -> MaskableHybridActionDictRolloutBufferSamples:
        del env
        return MaskableHybridActionDictRolloutBufferSamples(
            observations={
                key: self.to_torch(obs[batch_inds])
                for key, obs in self.observations.items()
            },
            actions=self.to_torch(self.actions[batch_inds]),
            old_values=self.to_torch(self.values[batch_inds].flatten()),
            old_log_prob=self.to_torch(self.log_probs[batch_inds].flatten()),
            advantages=self.to_torch(self.advantages[batch_inds].flatten()),
            returns=self.to_torch(self.returns[batch_inds].flatten()),
            action_masks=self.to_torch(
                self.action_masks[batch_inds].reshape(-1, self.mask_dims)
            ),
        )
