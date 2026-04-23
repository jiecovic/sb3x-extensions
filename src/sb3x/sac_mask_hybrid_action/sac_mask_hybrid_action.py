"""Maskable SAC wrapper for hybrid continuous/discrete action environments."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, ClassVar, TypeVar

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import MaybeCallback

from sb3x.common.hybrid_action import HybridAction, make_hybrid_action_spec
from sb3x.common.maskable import (
    MaybeMasks,
    get_action_masks,
    is_masking_supported,
    mask_dims_for_action_space,
    reshape_action_masks,
)
from sb3x.sac_hybrid_action.sac_hybrid_action import (
    HybridActionSAC,
    _HybridSACTrainBatch,
)

from .buffers import (
    MaskableHybridActionDictReplayBuffer,
    MaskableHybridActionReplayBuffer,
)
from .policies import (
    CnnPolicy,
    MaskableHybridActionSACPolicy,
    MlpPolicy,
    MultiInputPolicy,
)

SelfMaskableHybridActionSAC = TypeVar(
    "SelfMaskableHybridActionSAC",
    bound="MaskableHybridActionSAC",
)


class MaskableHybridActionSAC(HybridActionSAC):
    """HybridActionSAC with invalid-action masks for the discrete branch.

    Masks are applied only to the ``MultiDiscrete`` branch. The continuous
    branch remains the standard squashed Gaussian SAC actor.
    """

    policy_aliases: ClassVar[dict[str, type[MaskableHybridActionSACPolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }

    mask_dims: int
    use_masking: bool = True
    _current_action_masks: np.ndarray | None = None

    @property
    def maskable_policy(self) -> MaskableHybridActionSACPolicy:
        """Return the policy narrowed to the maskable hybrid SAC type."""
        if not isinstance(self.policy, MaskableHybridActionSACPolicy):
            raise TypeError("Policy must subclass MaskableHybridActionSACPolicy")
        return self.policy

    @property
    def maskable_replay_buffer(
        self,
    ) -> MaskableHybridActionReplayBuffer | MaskableHybridActionDictReplayBuffer:
        """Return the replay buffer narrowed to the maskable variants."""
        if not isinstance(
            self.replay_buffer,
            (MaskableHybridActionReplayBuffer, MaskableHybridActionDictReplayBuffer),
        ):
            raise TypeError(f"{self.replay_buffer} does not support action masking")
        return self.replay_buffer

    def _setup_model(self) -> None:
        if not hasattr(self, "hybrid_action_spec"):
            hybrid_action_space = self.policy_kwargs.get("hybrid_action_space")
            if hybrid_action_space is None:
                raise ValueError("Missing hybrid_action_space in policy kwargs")
            self.hybrid_action_spec = make_hybrid_action_spec(hybrid_action_space)

        self.mask_dims = mask_dims_for_action_space(
            self.hybrid_action_spec.discrete_space
        )
        replay_buffer_kwargs = dict(self.replay_buffer_kwargs)
        provided_mask_dims = replay_buffer_kwargs.get("mask_dims")
        if provided_mask_dims is not None and int(provided_mask_dims) != self.mask_dims:
            raise ValueError(
                f"replay_buffer_kwargs['mask_dims']={provided_mask_dims} does not "
                f"match the discrete branch mask size {self.mask_dims}"
            )
        replay_buffer_kwargs["mask_dims"] = self.mask_dims
        self.replay_buffer_kwargs = replay_buffer_kwargs

        if self.replay_buffer_class is None:
            self.replay_buffer_class = (
                MaskableHybridActionDictReplayBuffer
                if isinstance(self.observation_space, spaces.Dict)
                else MaskableHybridActionReplayBuffer
            )

        super()._setup_model()

        if not isinstance(self.policy, MaskableHybridActionSACPolicy):
            raise TypeError("Policy must subclass MaskableHybridActionSACPolicy")

    def learn(
        self: SelfMaskableHybridActionSAC,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "MaskableHybridActionSAC",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
        use_masking: bool = True,
    ) -> SelfMaskableHybridActionSAC:
        """Train with masks enabled by default, matching sb3-contrib convention."""
        self.use_masking = use_masking
        if use_masking and self.env is not None and not is_masking_supported(self.env):
            raise ValueError(
                "Environment does not support action masking. Expose an "
                "action_masks() method returning the discrete-branch mask."
            )
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _sample_action(
        self,
        learning_starts: int,
        action_noise: ActionNoise | None = None,
        n_envs: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        if action_noise is not None:
            raise ValueError("MaskableHybridActionSAC does not support action_noise")

        self._current_action_masks = self._get_current_action_masks(n_envs)
        if self.num_timesteps < learning_starts:
            unscaled_action = self._sample_warmup_actions(self._current_action_masks)
        else:
            if self._last_obs is None:
                raise RuntimeError("self._last_obs was not set")
            action_masks = self._current_action_masks if self.use_masking else None
            unscaled_action, _ = self.maskable_policy.predict(
                self._last_obs,
                deterministic=False,
                action_masks=action_masks,
            )

        if not isinstance(self.action_space, spaces.Box):
            raise TypeError("MaskableHybridActionSAC requires the internal flat Box")
        scaled_action = self.policy.scale_action(unscaled_action)
        return self.policy.unscale_action(scaled_action), scaled_action

    def _store_transition(
        self,
        replay_buffer: ReplayBuffer,
        buffer_action: np.ndarray,
        new_obs: np.ndarray | dict[str, np.ndarray],
        reward: np.ndarray,
        dones: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        if not isinstance(
            replay_buffer,
            (MaskableHybridActionReplayBuffer, MaskableHybridActionDictReplayBuffer),
        ):
            raise TypeError(f"{replay_buffer} does not support action masking")

        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

        next_obs = deepcopy(new_obs_)
        next_action_masks = self._get_next_action_masks(dones, infos)
        for idx, done in enumerate(dones):
            terminal_obs = infos[idx].get("terminal_observation")
            if done and terminal_obs is not None:
                if isinstance(next_obs, dict):
                    terminal_obs_ = terminal_obs
                    if self._vec_normalize_env is not None:
                        terminal_obs_ = self._vec_normalize_env.unnormalize_obs(
                            terminal_obs_
                        )
                    for key in next_obs:
                        next_obs[key][idx] = terminal_obs_[key]
                else:
                    next_obs[idx] = terminal_obs
                    if self._vec_normalize_env is not None:
                        next_obs[idx] = self._vec_normalize_env.unnormalize_obs(
                            next_obs[idx, :]
                        )

        current_action_masks = self._current_action_masks
        if current_action_masks is None:
            current_action_masks = self._all_valid_action_masks(len(dones))

        last_original_obs = self._last_original_obs
        if isinstance(replay_buffer, MaskableHybridActionDictReplayBuffer):
            if not isinstance(last_original_obs, dict) or not isinstance(
                next_obs,
                dict,
            ):
                raise TypeError("Dict replay buffer requires dict observations")
            replay_buffer.add(
                last_original_obs,
                next_obs,
                buffer_action,
                reward_,
                dones,
                infos,
                action_masks=current_action_masks,
                next_action_masks=next_action_masks,
            )
        else:
            if not isinstance(last_original_obs, np.ndarray) or not isinstance(
                next_obs, np.ndarray
            ):
                raise TypeError("Replay buffer requires array observations")
            replay_buffer.add(
                last_original_obs,
                next_obs,
                buffer_action,
                reward_,
                dones,
                infos,
                action_masks=current_action_masks,
                next_action_masks=next_action_masks,
            )

        self._last_obs = new_obs
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_

    def _sample_train_batch(self, batch_size: int) -> _HybridSACTrainBatch:
        replay_data = self.maskable_replay_buffer.sample(
            batch_size,
            env=self._vec_normalize_env,
        )
        discounts = (
            replay_data.discounts if replay_data.discounts is not None else self.gamma
        )
        return _HybridSACTrainBatch(
            observations=replay_data.observations,
            actions=replay_data.actions,
            next_observations=replay_data.next_observations,
            dones=replay_data.dones,
            rewards=replay_data.rewards,
            discounts=discounts,
            action_masks=replay_data.action_masks,
            next_action_masks=replay_data.next_action_masks,
        )

    def predict(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
        action_masks: MaybeMasks = None,
    ) -> tuple[HybridAction | list[HybridAction], tuple[np.ndarray, ...] | None]:
        """Return public hybrid actions, optionally masking the discrete branch."""
        flat_action, next_state = self.maskable_policy.predict(
            observation,
            state=state,
            episode_start=episode_start,
            deterministic=deterministic,
            action_masks=action_masks,
        )
        return self._unflatten_predicted_actions(flat_action), next_state

    def _get_current_action_masks(self, n_envs: int) -> np.ndarray:
        if not self.use_masking:
            return self._all_valid_action_masks(n_envs)
        if self.env is None:
            raise RuntimeError("Action masks require an environment")
        if not is_masking_supported(self.env):
            raise ValueError(
                "Environment does not support action masking. Expose an "
                "action_masks() method returning the discrete-branch mask."
            )
        return reshape_action_masks(
            get_action_masks(self.env),
            n_envs=n_envs,
            mask_dims=self.mask_dims,
        )

    def _get_next_action_masks(
        self,
        dones: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> np.ndarray:
        if self.use_masking:
            if self.env is None:
                raise RuntimeError("Action masks require an environment")
            next_action_masks = reshape_action_masks(
                get_action_masks(self.env),
                n_envs=len(dones),
                mask_dims=self.mask_dims,
            )
        else:
            next_action_masks = self._all_valid_action_masks(len(dones))

        for idx, done in enumerate(dones):
            terminal_action_masks = infos[idx].get("terminal_action_masks")
            if done and terminal_action_masks is not None:
                next_action_masks[idx] = np.asarray(
                    terminal_action_masks,
                    dtype=np.float32,
                ).reshape(self.mask_dims)
        return next_action_masks

    def _sample_warmup_actions(self, action_masks: np.ndarray) -> np.ndarray:
        if not self.use_masking:
            return np.array(
                [self.action_space.sample() for _ in range(action_masks.shape[0])]
            )

        mask_batch = np.asarray(action_masks, dtype=bool).reshape(
            action_masks.shape[0],
            self.mask_dims,
        )
        actions = []
        for masks in mask_batch:
            flat_action = np.asarray(self.action_space.sample(), dtype=np.float32)
            discrete_action = self._sample_discrete_branch(masks)
            flat_action[self.hybrid_action_spec.continuous_dim :] = (
                discrete_action.reshape(-1)
            )
            actions.append(flat_action)
        return np.asarray(actions, dtype=np.float32)

    def _sample_discrete_branch(self, action_masks: np.ndarray) -> np.ndarray:
        branch_actions: list[int] = []
        start = 0
        for action_dim in self.hybrid_action_spec.discrete_action_dims:
            branch_mask = action_masks[start : start + action_dim]
            valid_actions = np.flatnonzero(branch_mask)
            if valid_actions.size == 0:
                raise ValueError("Each discrete action branch must have a valid action")
            sampled_idx = int(self.action_space.np_random.integers(valid_actions.size))
            branch_actions.append(int(valid_actions[sampled_idx]))
            start += action_dim
        return np.asarray(
            branch_actions,
            dtype=self.hybrid_action_spec.discrete_space.dtype,
        ).reshape(self.hybrid_action_spec.discrete_shape)

    def _all_valid_action_masks(self, n_envs: int) -> np.ndarray:
        return np.ones((n_envs, self.mask_dims), dtype=np.float32)


__all__ = ["MaskableHybridActionSAC"]
