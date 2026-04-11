"""Recurrent rollout buffers that carry invalid-action masks."""

from __future__ import annotations

from collections.abc import Generator

import numpy as np
from sb3_contrib.common.recurrent.buffers import (
    RecurrentDictRolloutBuffer,
    RecurrentRolloutBuffer,
    create_sequencers,
)
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from stable_baselines3.common.vec_env import VecNormalize

from sb3x.common.maskable import (
    make_all_valid_action_masks,
    mask_dims_for_action_space,
    reshape_action_masks,
)

from .type_aliases import (
    MaskableRecurrentDictRolloutBufferSamples,
    MaskableRecurrentRolloutBufferSamples,
)


class MaskableRecurrentRolloutBuffer(RecurrentRolloutBuffer):
    """Recurrent rollout buffer that also stores invalid-action masks."""

    action_masks: np.ndarray
    mask_dims: int

    def reset(self) -> None:
        super().reset()
        self.mask_dims = mask_dims_for_action_space(self.action_space)
        self.action_masks = make_all_valid_action_masks(
            buffer_size=self.buffer_size,
            n_envs=self.n_envs,
            mask_dims=self.mask_dims,
        )

    def add(
        self,
        *args: object,
        action_masks: np.ndarray | None = None,
        lstm_states: RNNStates,
        **kwargs: object,
    ) -> None:
        if action_masks is not None:
            self.action_masks[self.pos] = reshape_action_masks(
                action_masks,
                n_envs=self.n_envs,
                mask_dims=self.mask_dims,
            )
        super().add(*args, lstm_states=lstm_states, **kwargs)

    def get(
        self,
        batch_size: int | None = None,
    ) -> Generator[MaskableRecurrentRolloutBufferSamples, None, None]:
        if not self.full:
            raise AssertionError("Rollout buffer must be full before sampling from it")

        if not self.generator_ready:
            for tensor_name in [
                "hidden_states_pi",
                "cell_states_pi",
                "hidden_states_vf",
                "cell_states_vf",
            ]:
                setattr(
                    self,
                    tensor_name,
                    getattr(self, tensor_name).swapaxes(1, 2),
                )

            for tensor_name in [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "hidden_states_pi",
                "cell_states_pi",
                "hidden_states_vf",
                "cell_states_vf",
                "episode_starts",
                "action_masks",
            ]:
                setattr(
                    self,
                    tensor_name,
                    self.swap_and_flatten(getattr(self, tensor_name)),
                )
            self.generator_ready = True

        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        split_index = np.random.randint(self.buffer_size * self.n_envs)
        indices = np.arange(self.buffer_size * self.n_envs)
        indices = np.concatenate((indices[split_index:], indices[:split_index]))

        env_change = np.zeros(self.buffer_size * self.n_envs).reshape(
            self.buffer_size,
            self.n_envs,
        )
        env_change[0, :] = 1.0
        env_change = self.swap_and_flatten(env_change)

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            batch_inds = indices[start_idx : start_idx + batch_size]
            yield self._get_samples(batch_inds, env_change)
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env_change: np.ndarray,
        env: VecNormalize | None = None,
    ) -> MaskableRecurrentRolloutBufferSamples:
        del env
        seq_start_indices, pad_sequence, pad_and_flatten = create_sequencers(
            self.episode_starts[batch_inds],
            env_change[batch_inds],
            self.device,
        )

        n_seq = len(seq_start_indices)
        max_length = pad_sequence(self.actions[batch_inds]).shape[1]
        padded_batch_size = n_seq * max_length

        lstm_states_pi = (
            self.hidden_states_pi[batch_inds][seq_start_indices].swapaxes(0, 1),
            self.cell_states_pi[batch_inds][seq_start_indices].swapaxes(0, 1),
        )
        lstm_states_vf = (
            self.hidden_states_vf[batch_inds][seq_start_indices].swapaxes(0, 1),
            self.cell_states_vf[batch_inds][seq_start_indices].swapaxes(0, 1),
        )

        return MaskableRecurrentRolloutBufferSamples(
            observations=pad_sequence(self.observations[batch_inds]).reshape(
                (padded_batch_size, *self.obs_shape)
            ),
            actions=pad_sequence(self.actions[batch_inds]).reshape(
                (padded_batch_size, *self.actions.shape[1:])
            ),
            old_values=pad_and_flatten(self.values[batch_inds]),
            old_log_prob=pad_and_flatten(self.log_probs[batch_inds]),
            advantages=pad_and_flatten(self.advantages[batch_inds]),
            returns=pad_and_flatten(self.returns[batch_inds]),
            lstm_states=RNNStates(
                (
                    self.to_torch(lstm_states_pi[0]).contiguous(),
                    self.to_torch(lstm_states_pi[1]).contiguous(),
                ),
                (
                    self.to_torch(lstm_states_vf[0]).contiguous(),
                    self.to_torch(lstm_states_vf[1]).contiguous(),
                ),
            ),
            episode_starts=pad_and_flatten(self.episode_starts[batch_inds]),
            sequence_mask=pad_and_flatten(
                np.ones_like(self.returns[batch_inds], dtype=np.float32)
            ),
            action_masks=pad_sequence(self.action_masks[batch_inds]).reshape(
                (padded_batch_size, self.mask_dims)
            ),
        )


class MaskableRecurrentDictRolloutBuffer(RecurrentDictRolloutBuffer):
    """Dict-observation recurrent rollout buffer with action masks."""

    action_masks: np.ndarray
    mask_dims: int

    def reset(self) -> None:
        super().reset()
        self.mask_dims = mask_dims_for_action_space(self.action_space)
        self.action_masks = make_all_valid_action_masks(
            buffer_size=self.buffer_size,
            n_envs=self.n_envs,
            mask_dims=self.mask_dims,
        )

    def add(
        self,
        *args: object,
        action_masks: np.ndarray | None = None,
        lstm_states: RNNStates,
        **kwargs: object,
    ) -> None:
        if action_masks is not None:
            self.action_masks[self.pos] = reshape_action_masks(
                action_masks,
                n_envs=self.n_envs,
                mask_dims=self.mask_dims,
            )
        super().add(*args, lstm_states=lstm_states, **kwargs)

    def get(
        self,
        batch_size: int | None = None,
    ) -> Generator[MaskableRecurrentDictRolloutBufferSamples, None, None]:
        if not self.full:
            raise AssertionError("Rollout buffer must be full before sampling from it")

        if not self.generator_ready:
            for tensor_name in [
                "hidden_states_pi",
                "cell_states_pi",
                "hidden_states_vf",
                "cell_states_vf",
            ]:
                setattr(
                    self,
                    tensor_name,
                    getattr(self, tensor_name).swapaxes(1, 2),
                )

            for key, obs in self.observations.items():
                self.observations[key] = self.swap_and_flatten(obs)

            for tensor_name in [
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "hidden_states_pi",
                "cell_states_pi",
                "hidden_states_vf",
                "cell_states_vf",
                "episode_starts",
                "action_masks",
            ]:
                setattr(
                    self,
                    tensor_name,
                    self.swap_and_flatten(getattr(self, tensor_name)),
                )
            self.generator_ready = True

        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        split_index = np.random.randint(self.buffer_size * self.n_envs)
        indices = np.arange(self.buffer_size * self.n_envs)
        indices = np.concatenate((indices[split_index:], indices[:split_index]))

        env_change = np.zeros(self.buffer_size * self.n_envs).reshape(
            self.buffer_size,
            self.n_envs,
        )
        env_change[0, :] = 1.0
        env_change = self.swap_and_flatten(env_change)

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            batch_inds = indices[start_idx : start_idx + batch_size]
            yield self._get_samples(batch_inds, env_change)
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env_change: np.ndarray,
        env: VecNormalize | None = None,
    ) -> MaskableRecurrentDictRolloutBufferSamples:
        del env
        seq_start_indices, pad_sequence, pad_and_flatten = create_sequencers(
            self.episode_starts[batch_inds],
            env_change[batch_inds],
            self.device,
        )

        n_seq = len(seq_start_indices)
        max_length = pad_sequence(self.actions[batch_inds]).shape[1]
        padded_batch_size = n_seq * max_length

        lstm_states_pi = (
            self.hidden_states_pi[batch_inds][seq_start_indices].swapaxes(0, 1),
            self.cell_states_pi[batch_inds][seq_start_indices].swapaxes(0, 1),
        )
        lstm_states_vf = (
            self.hidden_states_vf[batch_inds][seq_start_indices].swapaxes(0, 1),
            self.cell_states_vf[batch_inds][seq_start_indices].swapaxes(0, 1),
        )
        observations = {
            key: pad_sequence(obs[batch_inds]).reshape(
                (padded_batch_size, *self.obs_shape[key])
            )
            for key, obs in self.observations.items()
        }

        return MaskableRecurrentDictRolloutBufferSamples(
            observations=observations,
            actions=pad_sequence(self.actions[batch_inds]).reshape(
                (padded_batch_size, *self.actions.shape[1:])
            ),
            old_values=pad_and_flatten(self.values[batch_inds]),
            old_log_prob=pad_and_flatten(self.log_probs[batch_inds]),
            advantages=pad_and_flatten(self.advantages[batch_inds]),
            returns=pad_and_flatten(self.returns[batch_inds]),
            lstm_states=RNNStates(
                (
                    self.to_torch(lstm_states_pi[0]).contiguous(),
                    self.to_torch(lstm_states_pi[1]).contiguous(),
                ),
                (
                    self.to_torch(lstm_states_vf[0]).contiguous(),
                    self.to_torch(lstm_states_vf[1]).contiguous(),
                ),
            ),
            episode_starts=pad_and_flatten(self.episode_starts[batch_inds]),
            sequence_mask=pad_and_flatten(
                np.ones_like(self.returns[batch_inds], dtype=np.float32)
            ),
            action_masks=pad_sequence(self.action_masks[batch_inds]).reshape(
                (padded_batch_size, self.mask_dims)
            ),
        )
