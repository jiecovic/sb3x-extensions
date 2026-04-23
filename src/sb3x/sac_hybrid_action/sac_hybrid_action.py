"""SAC wrapper for hybrid continuous/discrete action environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import GymEnv, PyTorchObs, Schedule
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from sb3x.common.hybrid_action import (
    HybridAction,
    HybridActionSpec,
    combine_hybrid_actions,
    has_public_hybrid_action_space,
    make_hybrid_action_spec,
    prepare_hybrid_action_env,
    wrap_hybrid_action_env,
)
from sb3x.common.maskable import MaybeMasks

from .policies import (
    CnnPolicy,
    HybridActionContinuousCritic,
    HybridActionSACActor,
    HybridActionSACPolicy,
    MlpPolicy,
    MultiInputPolicy,
)


class HybridActionSAC(SAC):
    """SAC for ``Dict(continuous=Box, discrete=MultiDiscrete)`` action spaces.

    The discrete branch is treated exactly in the SAC backup and policy loss by
    enumerating all ``MultiDiscrete`` combinations. The critic receives the
    continuous branch plus a one-hot representation of the discrete branch.
    """

    policy_aliases: ClassVar[dict[str, type[HybridActionSACPolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }

    actor: HybridActionSACActor
    critic: HybridActionContinuousCritic
    critic_target: HybridActionContinuousCritic
    hybrid_action_spec: HybridActionSpec

    def __init__(
        self,
        policy: str | type[HybridActionSACPolicy],
        env: GymEnv | str | None,
        learning_rate: float | Schedule = 3e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: int | tuple[int, str] = 1,
        gradient_steps: int = 1,
        action_noise: ActionNoise | None = None,
        replay_buffer_class: type[ReplayBuffer] | None = None,
        replay_buffer_kwargs: dict[str, Any] | None = None,
        optimize_memory_usage: bool = False,
        n_steps: int = 1,
        ent_coef: str | float = "auto",
        target_update_interval: int = 1,
        target_entropy: str | float = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: str | None = None,
        policy_kwargs: dict[str, Any] | None = None,
        verbose: int = 0,
        seed: int | None = None,
        device: th.device | str = "auto",
        _init_setup_model: bool = True,
    ) -> None:
        if use_sde or use_sde_at_warmup:
            raise ValueError("HybridActionSAC does not support gSDE")
        if action_noise is not None:
            raise ValueError(
                "HybridActionSAC does not support action_noise because it would "
                "corrupt the categorical branch"
            )

        wrapped_env, hybrid_action_spec, policy_kwargs = prepare_hybrid_action_env(
            env,
            {} if policy_kwargs is None else policy_kwargs,
            algorithm_name="HybridActionSAC",
            init_setup_model=_init_setup_model,
        )
        if hybrid_action_spec is not None:
            self.hybrid_action_spec = hybrid_action_spec

        super().__init__(
            policy=policy,
            env=wrapped_env,  # pyright: ignore[reportArgumentType]
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=None,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            n_steps=n_steps,
            ent_coef=ent_coef,
            target_update_interval=target_update_interval,
            target_entropy=target_entropy,
            use_sde=False,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=False,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )

    @classmethod
    def _wrap_env(
        cls,
        env: gym.Env | VecEnv,
        verbose: int = 0,
        monitor_wrapper: bool = True,
    ) -> VecEnv:
        """Let SB3 loading validate against the internal flat action space."""
        wrapped_env: GymEnv = env
        if has_public_hybrid_action_space(env):
            wrapped_env, _ = wrap_hybrid_action_env(env)
        return super()._wrap_env(
            wrapped_env,
            verbose=verbose,
            monitor_wrapper=monitor_wrapper,
        )

    def _setup_model(self) -> None:
        """Initialize SB3 SAC and refresh hybrid action metadata after loading."""
        super()._setup_model()
        hybrid_action_space = self.policy_kwargs.get("hybrid_action_space")
        if hybrid_action_space is not None:
            self.hybrid_action_spec = make_hybrid_action_spec(hybrid_action_space)

    def _sample_action(
        self,
        learning_starts: int,
        action_noise: ActionNoise | None = None,
        n_envs: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample internal flat actions while keeping public predict() ergonomic."""
        if action_noise is not None:
            raise ValueError("HybridActionSAC does not support action_noise")

        if self.num_timesteps < learning_starts:
            unscaled_action = np.array(
                [self.action_space.sample() for _ in range(n_envs)]
            )
        else:
            if self._last_obs is None:
                raise RuntimeError("self._last_obs was not set")
            unscaled_action, _ = super().predict(
                self._last_obs,
                deterministic=False,
            )

        if not isinstance(self.action_space, spaces.Box):
            raise TypeError("HybridActionSAC requires the internal flat Box space")
        scaled_action = self.policy.scale_action(unscaled_action)
        return self.policy.unscale_action(scaled_action), scaled_action

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        """Run SAC updates with exact expectation over the discrete branch."""
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers.append(self.ent_coef_optimizer)
        self._update_learning_rate(optimizers)

        ent_coef_losses: list[float] = []
        ent_coefs: list[float] = []
        actor_losses: list[float] = []
        critic_losses: list[float] = []

        for gradient_step in range(gradient_steps):
            del gradient_step
            train_batch = self._sample_train_batch(batch_size)

            policy_eval = self._evaluate_hybrid_policy(
                train_batch.observations,
                action_masks=train_batch.action_masks,
            )
            expected_log_prob = self._expected_joint_log_prob(policy_eval)

            ent_coef_loss: th.Tensor | None = None
            ent_coef: th.Tensor
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                if not isinstance(self.target_entropy, float):
                    raise TypeError("target_entropy must be a float after setup")
                ent_coef_loss = -(
                    self.log_ent_coef
                    * (expected_log_prob.reshape(-1, 1) + self.target_entropy).detach()
                ).mean()
                ent_coef_losses.append(float(ent_coef_loss.item()))
            else:
                ent_coef = th.as_tensor(self.ent_coef_tensor, device=self.device)

            ent_coefs.append(float(ent_coef.item()))

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                next_policy_eval = self._evaluate_hybrid_policy(
                    train_batch.next_observations,
                    action_masks=train_batch.next_action_masks,
                )
                next_min_q = self._min_q_for_all_discrete_actions(
                    self.critic_target,
                    train_batch.next_observations,
                    next_policy_eval.continuous_actions,
                )
                next_value = self._expected_soft_value(
                    next_policy_eval,
                    next_min_q,
                    ent_coef,
                )
                target_q_values = train_batch.rewards + (
                    1 - train_batch.dones
                ) * train_batch.discounts * next_value.reshape(-1, 1)

            current_q_values = self.critic(
                train_batch.observations,
                train_batch.actions,
            )
            critic_loss_terms = [
                F.mse_loss(current_q, target_q_values) for current_q in current_q_values
            ]
            critic_loss = 0.5 * th.stack(critic_loss_terms).sum()
            critic_losses.append(float(critic_loss.item()))

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            min_q = self._min_q_for_all_discrete_actions(
                self.critic,
                train_batch.observations,
                policy_eval.continuous_actions,
            )
            actor_loss = self._actor_loss(policy_eval, min_q, ent_coef)
            actor_losses.append(float(actor_loss.item()))

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            if self._n_updates % self.target_update_interval == 0:
                polyak_update(
                    self.critic.parameters(),
                    self.critic_target.parameters(),
                    self.tau,
                )
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)
            self._n_updates += 1

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def _sample_train_batch(self, batch_size: int) -> _HybridSACTrainBatch:
        if self.replay_buffer is None:
            raise RuntimeError("Replay buffer was not initialized")
        replay_data = self.replay_buffer.sample(
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
        )

    def predict(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
    ) -> tuple[HybridAction | list[HybridAction], tuple[np.ndarray, ...] | None]:
        """Return public hybrid actions instead of the internal flat actions."""
        flat_action, next_state = super().predict(
            observation,
            state=state,
            episode_start=episode_start,
            deterministic=deterministic,
        )
        return self._unflatten_predicted_actions(flat_action), next_state

    def _evaluate_hybrid_policy(
        self,
        observations: PyTorchObs,
        action_masks: MaybeMasks = None,
    ) -> _HybridPolicyEval:
        mean_actions, log_std, discrete_logits = self.actor.get_action_dist_params(
            observations,
        )
        continuous_actions, continuous_log_prob = (
            self.actor.continuous_action_log_prob_from_params(
                mean_actions,
                log_std,
            )
        )
        discrete_log_probs = self.actor.discrete_log_prob_matrix(
            discrete_logits,
            action_masks=action_masks,
        )
        return _HybridPolicyEval(
            continuous_actions=continuous_actions,
            continuous_log_prob=continuous_log_prob,
            discrete_log_probs=discrete_log_probs,
            discrete_probs=discrete_log_probs.exp(),
        )

    def _min_q_for_all_discrete_actions(
        self,
        critic: HybridActionContinuousCritic,
        observations: PyTorchObs,
        continuous_actions: th.Tensor,
    ) -> th.Tensor:
        scaled_discrete_actions = self.actor.all_scaled_discrete_actions()
        num_batches = continuous_actions.shape[0]
        num_discrete_actions = scaled_discrete_actions.shape[0]

        repeated_observations = _repeat_observations(
            observations,
            num_discrete_actions,
        )
        repeated_continuous_actions = (
            continuous_actions[:, None, :]
            .expand(num_batches, num_discrete_actions, -1)
            .reshape(num_batches * num_discrete_actions, -1)
        )
        repeated_discrete_actions = (
            scaled_discrete_actions[None, :, :]
            .expand(num_batches, num_discrete_actions, -1)
            .reshape(num_batches * num_discrete_actions, -1)
        )
        actions = combine_hybrid_actions(
            repeated_continuous_actions,
            repeated_discrete_actions,
        )
        q_values = th.cat(critic(repeated_observations, actions), dim=1)
        min_q, _ = th.min(q_values, dim=1)
        return min_q.reshape(num_batches, num_discrete_actions)

    def _expected_joint_log_prob(self, policy_eval: _HybridPolicyEval) -> th.Tensor:
        joint_log_prob = (
            policy_eval.continuous_log_prob[:, None] + policy_eval.discrete_log_probs
        )
        return (policy_eval.discrete_probs * joint_log_prob).sum(dim=1)

    def _expected_soft_value(
        self,
        policy_eval: _HybridPolicyEval,
        min_q: th.Tensor,
        ent_coef: th.Tensor,
    ) -> th.Tensor:
        joint_log_prob = (
            policy_eval.continuous_log_prob[:, None] + policy_eval.discrete_log_probs
        )
        return (policy_eval.discrete_probs * (min_q - ent_coef * joint_log_prob)).sum(
            dim=1,
        )

    def _actor_loss(
        self,
        policy_eval: _HybridPolicyEval,
        min_q: th.Tensor,
        ent_coef: th.Tensor,
    ) -> th.Tensor:
        joint_log_prob = (
            policy_eval.continuous_log_prob[:, None] + policy_eval.discrete_log_probs
        )
        return (
            (policy_eval.discrete_probs * (ent_coef * joint_log_prob - min_q))
            .sum(dim=1)
            .mean()
        )

    def _unflatten_predicted_actions(
        self,
        actions: np.ndarray,
    ) -> HybridAction | list[HybridAction]:
        actions_array = np.asarray(actions, dtype=np.float32)
        if actions_array.ndim == 1:
            return self.hybrid_action_spec.unflatten_action(actions_array)
        return self.hybrid_action_spec.unflatten_action_batch(actions_array)


@dataclass(frozen=True)
class _HybridPolicyEval:
    """Policy quantities for exact discrete SAC expectations."""

    continuous_actions: th.Tensor
    continuous_log_prob: th.Tensor
    discrete_log_probs: th.Tensor
    discrete_probs: th.Tensor


@dataclass(frozen=True)
class _HybridSACTrainBatch:
    """Replay data plus optional masks for the shared SAC update."""

    observations: PyTorchObs
    actions: th.Tensor
    next_observations: PyTorchObs
    dones: th.Tensor
    rewards: th.Tensor
    discounts: th.Tensor | float
    action_masks: MaybeMasks = None
    next_action_masks: MaybeMasks = None


def _repeat_observations(observations: PyTorchObs, repeats: int) -> PyTorchObs:
    if isinstance(observations, dict):
        return {
            key: value.repeat_interleave(repeats, dim=0)
            for key, value in observations.items()
        }
    return observations.repeat_interleave(repeats, dim=0)
