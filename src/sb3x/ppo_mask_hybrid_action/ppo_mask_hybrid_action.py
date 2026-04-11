"""Maskable PPO for hybrid continuous/discrete action environments."""

from __future__ import annotations

from typing import Any, ClassVar, TypeVar

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv
from torch.nn import functional as F

from sb3x.common.hybrid_action import HybridAction, make_hybrid_action_spec
from sb3x.common.maskable import (
    get_action_masks,
    is_masking_supported,
    mask_dims_for_action_space,
)
from sb3x.ppo_hybrid_action import HybridActionPPO

from .buffers import (
    MaskableHybridActionDictRolloutBuffer,
    MaskableHybridActionRolloutBuffer,
)
from .policies import (
    CnnPolicy,
    MaskableHybridActionActorCriticPolicy,
    MlpPolicy,
    MultiInputPolicy,
)

SelfMaskableHybridActionPPO = TypeVar(
    "SelfMaskableHybridActionPPO",
    bound="MaskableHybridActionPPO",
)


class MaskableHybridActionPPO(HybridActionPPO):
    """HybridActionPPO with invalid-action masks for the discrete branch."""

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }

    def __init__(
        self,
        policy: str | type[MaskableHybridActionActorCriticPolicy],
        env: GymEnv | str | None,
        learning_rate: float | Schedule = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float | Schedule = 0.2,
        clip_range_vf: None | float | Schedule = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: type[RolloutBuffer] | None = None,
        rollout_buffer_kwargs: dict[str, Any] | None = None,
        target_kl: float | None = None,
        stats_window_size: int = 100,
        tensorboard_log: str | None = None,
        policy_kwargs: dict[str, Any] | None = None,
        verbose: int = 0,
        seed: int | None = None,
        device: th.device | str = "auto",
        _init_setup_model: bool = True,
    ) -> None:
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            target_kl=target_kl,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )

    @property
    def maskable_policy(self) -> MaskableHybridActionActorCriticPolicy:
        """Return the policy narrowed to the maskable hybrid policy type."""
        if not isinstance(self.policy, MaskableHybridActionActorCriticPolicy):
            raise TypeError(
                "Policy must subclass MaskableHybridActionActorCriticPolicy"
            )
        return self.policy

    @property
    def maskable_rollout_buffer(
        self,
    ) -> MaskableHybridActionRolloutBuffer | MaskableHybridActionDictRolloutBuffer:
        """Return the rollout buffer narrowed to the maskable hybrid variants."""
        if not isinstance(
            self.rollout_buffer,
            (MaskableHybridActionRolloutBuffer, MaskableHybridActionDictRolloutBuffer),
        ):
            raise TypeError(f"{self.rollout_buffer} does not support action masking")
        return self.rollout_buffer

    def _setup_model(self) -> None:
        if not hasattr(self, "hybrid_action_spec"):
            hybrid_action_space = self.policy_kwargs.get("hybrid_action_space")
            if hybrid_action_space is None:
                raise ValueError("Missing hybrid_action_space in policy kwargs")
            self.hybrid_action_spec = make_hybrid_action_spec(hybrid_action_space)

        mask_dims = mask_dims_for_action_space(self.hybrid_action_spec.discrete_space)
        rollout_buffer_kwargs = dict(self.rollout_buffer_kwargs)
        provided_mask_dims = rollout_buffer_kwargs.get("mask_dims")
        if provided_mask_dims is not None and int(provided_mask_dims) != mask_dims:
            raise ValueError(
                f"rollout_buffer_kwargs['mask_dims']={provided_mask_dims} does not "
                f"match the discrete branch mask size {mask_dims}"
            )
        rollout_buffer_kwargs["mask_dims"] = mask_dims
        self.rollout_buffer_kwargs = rollout_buffer_kwargs

        if self.rollout_buffer_class is None:
            self.rollout_buffer_class = (
                MaskableHybridActionDictRolloutBuffer
                if isinstance(self.observation_space, spaces.Dict)
                else MaskableHybridActionRolloutBuffer
            )

        super()._setup_model()

        if not isinstance(self.policy, MaskableHybridActionActorCriticPolicy):
            raise TypeError(
                "Policy must subclass MaskableHybridActionActorCriticPolicy"
            )

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
        use_masking: bool = True,
    ) -> bool:
        """Collect rollouts, optionally applying discrete-branch action masks."""
        if not isinstance(
            rollout_buffer,
            (MaskableHybridActionRolloutBuffer, MaskableHybridActionDictRolloutBuffer),
        ):
            raise TypeError(f"{rollout_buffer} does not support action masking")

        last_obs = self._last_obs
        assert last_obs is not None, "No previous observation was provided"
        last_episode_starts = self._last_episode_starts
        assert last_episode_starts is not None, "Episode starts were not initialized"
        policy = self.maskable_policy
        policy.set_training_mode(False)

        n_steps = 0
        action_masks = None
        rollout_buffer.reset()

        if use_masking and not is_masking_supported(env):
            raise ValueError(
                "Environment does not support action masking. Expose an "
                "action_masks() method returning the discrete-branch mask."
            )

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            with th.no_grad():
                obs_tensor = obs_as_tensor(last_obs, self.device)
                if use_masking:
                    action_masks = get_action_masks(env)
                actions, values, log_probs = policy(
                    obs_tensor,
                    action_masks=action_masks,
                )

            actions = actions.cpu().numpy()
            clipped_actions = actions
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(
                    actions,
                    self.action_space.low,
                    self.action_space.high,
                )

            new_obs, rewards, dones, infos = env.step(clipped_actions)
            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = policy.obs_to_tensor(
                        infos[idx]["terminal_observation"]
                    )[0]
                    with th.no_grad():
                        terminal_value = policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            if isinstance(rollout_buffer, MaskableHybridActionDictRolloutBuffer):
                if not isinstance(last_obs, dict):
                    raise TypeError("Dict rollout buffer requires dict observations")
                rollout_buffer.add(
                    last_obs,
                    actions,
                    rewards,
                    last_episode_starts,
                    values,
                    log_probs,
                    action_masks=action_masks,
                )
            else:
                if isinstance(last_obs, dict):
                    raise TypeError("Rollout buffer requires array observations")
                rollout_buffer.add(
                    last_obs,
                    actions,
                    rewards,
                    last_episode_starts,
                    values,
                    log_probs,
                    action_masks=action_masks,
                )
            if isinstance(new_obs, tuple):
                raise TypeError("Tuple observations are not supported")
            self._last_obs = new_obs
            self._last_episode_starts = dones
            last_obs = new_obs
            last_episode_starts = dones

        with th.no_grad():
            values = policy.predict_values(obs_as_tensor(last_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        callback.on_rollout_end()
        return True

    def train(self) -> None:
        """Update policy using the currently gathered rollout buffer."""
        policy = self.maskable_policy
        rollout_buffer = self.maskable_rollout_buffer

        policy.set_training_mode(True)
        self._update_learning_rate(policy.optimizer)

        clip_range_schedule = self.clip_range
        assert callable(clip_range_schedule), "Clip range schedule was not initialized"
        clip_range = clip_range_schedule(self._current_progress_remaining)
        clip_range_vf_value: float | None = None
        if self.clip_range_vf is not None:
            clip_range_vf_schedule = self.clip_range_vf
            assert callable(clip_range_vf_schedule), (
                "Value clip range schedule was not initialized"
            )
            clip_range_vf_value = clip_range_vf_schedule(
                self._current_progress_remaining
            )

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        continue_training = True

        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in rollout_buffer.get(self.batch_size):
                values, log_prob, entropy = policy.evaluate_actions(
                    rollout_data.observations,
                    rollout_data.actions,
                    action_masks=rollout_data.action_masks,
                )
                values = values.flatten()

                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(
                    ratio,
                    1 - clip_range,
                    1 + clip_range,
                )
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if clip_range_vf_value is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values,
                        -clip_range_vf_value,
                        clip_range_vf_value,
                    )

                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                )

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = (
                        th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    )
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(
                            "Early stopping at step "
                            f"{epoch} due to reaching max kl: {approx_kl_div:.2f}"
                        )
                    break

                policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(policy.parameters(), self.max_grad_norm)
                policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(
            rollout_buffer.values.flatten(),
            rollout_buffer.returns.flatten(),
        )

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/std", th.exp(policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if clip_range_vf_value is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf_value)

    def predict(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
        action_masks: np.ndarray | th.Tensor | None = None,
    ) -> tuple[HybridAction | list[HybridAction], tuple[np.ndarray, ...] | None]:
        """Predict a public hybrid action, optionally using discrete masks."""
        flat_action, next_state = self.maskable_policy.predict(
            observation,
            state=state,
            episode_start=episode_start,
            deterministic=deterministic,
            action_masks=action_masks,
        )
        return self._unflatten_predicted_actions(flat_action), next_state

    def learn(  # pyright: ignore[reportIncompatibleMethodOverride]
        self: SelfMaskableHybridActionPPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "MaskableHybridActionPPO",
        reset_num_timesteps: bool = True,
        use_masking: bool = True,
        progress_bar: bool = False,
    ) -> SelfMaskableHybridActionPPO:
        """Learn with discrete-branch masking enabled by default."""
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )
        callback.on_training_start(locals(), globals())

        assert self.env is not None

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(
                self.env,
                callback,
                self.rollout_buffer,
                self.n_steps,
                use_masking=use_masking,
            )
            if not continue_training:
                break

            iteration += 1
            self._update_current_progress_remaining(
                self.num_timesteps,
                total_timesteps,
            )

            if log_interval is not None and iteration % log_interval == 0:
                self.dump_logs(iteration)

            self.train()

        callback.on_training_end()
        return self
