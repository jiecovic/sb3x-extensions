"""Replay buffers that store discrete-branch action masks for hybrid SAC."""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import torch as th
from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.vec_env import VecNormalize

from sb3x.common.maskable import make_all_valid_action_masks, reshape_action_masks


class MaskableHybridActionReplayBufferSamples(NamedTuple):
    """Array-observation replay samples with current and next action masks."""

    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    action_masks: th.Tensor
    next_action_masks: th.Tensor
    discounts: th.Tensor | None = None


class MaskableHybridActionDictReplayBufferSamples(NamedTuple):
    """Dict-observation replay samples with current and next action masks."""

    observations: TensorDict
    actions: th.Tensor
    next_observations: TensorDict
    dones: th.Tensor
    rewards: th.Tensor
    action_masks: th.Tensor
    next_action_masks: th.Tensor
    discounts: th.Tensor | None = None


class MaskableHybridActionReplayBuffer(ReplayBuffer):
    """Replay buffer for array observations and flattened discrete masks."""

    def __init__(self, *args: Any, mask_dims: int, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.mask_dims = int(mask_dims)
        self.action_masks = make_all_valid_action_masks(
            buffer_size=self.buffer_size,
            n_envs=self.n_envs,
            mask_dims=self.mask_dims,
        )
        self.next_action_masks = make_all_valid_action_masks(
            buffer_size=self.buffer_size,
            n_envs=self.n_envs,
            mask_dims=self.mask_dims,
        )

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
        *,
        action_masks: np.ndarray | None = None,
        next_action_masks: np.ndarray | None = None,
    ) -> None:
        self.action_masks[self.pos] = _normalize_masks(
            action_masks,
            n_envs=self.n_envs,
            mask_dims=self.mask_dims,
        )
        self.next_action_masks[self.pos] = _normalize_masks(
            next_action_masks,
            n_envs=self.n_envs,
            mask_dims=self.mask_dims,
        )
        super().add(obs, next_obs, action, reward, done, infos)

    def sample(
        self,
        batch_size: int,
        env: VecNormalize | None = None,
    ) -> MaskableHybridActionReplayBufferSamples:
        if not self.optimize_memory_usage:
            upper_bound = self.buffer_size if self.full else self.pos
            batch_inds = np.random.randint(0, upper_bound, size=batch_size)
            return self._get_samples(batch_inds, env=env)

        if self.full:
            batch_inds = (
                np.random.randint(1, self.buffer_size, size=batch_size) + self.pos
            ) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: VecNormalize | None = None,
    ) -> MaskableHybridActionReplayBufferSamples:
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_observations = _as_array_observation(
                self._normalize_obs(
                    self.observations[
                        (batch_inds + 1) % self.buffer_size,
                        env_indices,
                        :,
                    ],
                    env,
                )
            )
        else:
            next_observations = _as_array_observation(
                self._normalize_obs(
                    self.next_observations[batch_inds, env_indices, :],
                    env,
                )
            )
        observations = _as_array_observation(
            self._normalize_obs(
                self.observations[batch_inds, env_indices, :],
                env,
            )
        )

        return MaskableHybridActionReplayBufferSamples(
            observations=self.to_torch(observations),
            actions=self.to_torch(self.actions[batch_inds, env_indices, :]),
            next_observations=self.to_torch(next_observations),
            dones=self.to_torch(
                self.dones[batch_inds, env_indices]
                * (1 - self.timeouts[batch_inds, env_indices])
            ).reshape(-1, 1),
            rewards=self.to_torch(
                self._normalize_reward(
                    self.rewards[batch_inds, env_indices].reshape(-1, 1),
                    env,
                )
            ),
            action_masks=self.to_torch(
                self.action_masks[batch_inds, env_indices, :],
            ),
            next_action_masks=self.to_torch(
                self.next_action_masks[batch_inds, env_indices, :],
            ),
        )


class MaskableHybridActionDictReplayBuffer(DictReplayBuffer):
    """Replay buffer for dict observations and flattened discrete masks."""

    def __init__(self, *args: Any, mask_dims: int, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.mask_dims = int(mask_dims)
        self.action_masks = make_all_valid_action_masks(
            buffer_size=self.buffer_size,
            n_envs=self.n_envs,
            mask_dims=self.mask_dims,
        )
        self.next_action_masks = make_all_valid_action_masks(
            buffer_size=self.buffer_size,
            n_envs=self.n_envs,
            mask_dims=self.mask_dims,
        )

    def add(
        self,
        obs: dict[str, np.ndarray],
        next_obs: dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
        *,
        action_masks: np.ndarray | None = None,
        next_action_masks: np.ndarray | None = None,
    ) -> None:
        self.action_masks[self.pos] = _normalize_masks(
            action_masks,
            n_envs=self.n_envs,
            mask_dims=self.mask_dims,
        )
        self.next_action_masks[self.pos] = _normalize_masks(
            next_action_masks,
            n_envs=self.n_envs,
            mask_dims=self.mask_dims,
        )
        super().add(obs, next_obs, action, reward, done, infos)

    def sample(
        self,
        batch_size: int,
        env: VecNormalize | None = None,
    ) -> MaskableHybridActionDictReplayBufferSamples:
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: VecNormalize | None = None,
    ) -> MaskableHybridActionDictReplayBufferSamples:
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        observations_ = self._normalize_obs(
            {
                key: obs[batch_inds, env_indices, :]
                for key, obs in self.observations.items()
            },
            env,
        )
        next_observations_ = self._normalize_obs(
            {
                key: obs[batch_inds, env_indices, :]
                for key, obs in self.next_observations.items()
            },
            env,
        )
        if not isinstance(observations_, dict) or not isinstance(
            next_observations_,
            dict,
        ):
            raise TypeError("Dict replay buffer expected dict observations")

        return MaskableHybridActionDictReplayBufferSamples(
            observations={
                key: self.to_torch(obs) for key, obs in observations_.items()
            },
            actions=self.to_torch(self.actions[batch_inds, env_indices]),
            next_observations={
                key: self.to_torch(obs) for key, obs in next_observations_.items()
            },
            dones=self.to_torch(
                self.dones[batch_inds, env_indices]
                * (1 - self.timeouts[batch_inds, env_indices])
            ).reshape(-1, 1),
            rewards=self.to_torch(
                self._normalize_reward(
                    self.rewards[batch_inds, env_indices].reshape(-1, 1),
                    env,
                )
            ),
            action_masks=self.to_torch(
                self.action_masks[batch_inds, env_indices, :],
            ),
            next_action_masks=self.to_torch(
                self.next_action_masks[batch_inds, env_indices, :],
            ),
        )


def _as_array_observation(
    observation: np.ndarray | dict[str, np.ndarray],
) -> np.ndarray:
    if isinstance(observation, dict):
        raise TypeError("Array replay buffer expected array observations")
    return observation


def _normalize_masks(
    masks: np.ndarray | None,
    *,
    n_envs: int,
    mask_dims: int,
) -> np.ndarray:
    if masks is None:
        return np.ones((n_envs, mask_dims), dtype=np.float32)
    return reshape_action_masks(masks, n_envs=n_envs, mask_dims=mask_dims)
