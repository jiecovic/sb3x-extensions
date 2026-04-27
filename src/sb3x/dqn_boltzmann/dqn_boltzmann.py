"""DQN variant with Boltzmann exploration."""

from __future__ import annotations

from typing import Any, TypeVar

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import LinearSchedule, polyak_update
from stable_baselines3.dqn.policies import DQNPolicy

SelfBoltzmannDQN = TypeVar("SelfBoltzmannDQN", bound="BoltzmannDQN")


class BoltzmannDQN(DQN):
    """DQN with softmax-over-Q exploration instead of epsilon-greedy actions.

    Training remains vanilla SB3 DQN. The only behavioral change is
    non-deterministic action selection during rollout collection and
    ``predict(..., deterministic=False)``.
    """

    temperature_schedule: Schedule

    def __init__(
        self,
        policy: str | type[DQNPolicy],
        env: GymEnv | str,
        learning_rate: float | Schedule = 1e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: int | tuple[int, str] = 4,
        gradient_steps: int = 1,
        replay_buffer_class: type[ReplayBuffer] | None = None,
        replay_buffer_kwargs: dict[str, Any] | None = None,
        optimize_memory_usage: bool = False,
        n_steps: int = 1,
        target_update_interval: int = 10000,
        temperature_initial: float = 1.0,
        temperature_final: float = 0.05,
        temperature_fraction: float = 0.1,
        max_grad_norm: float = 10,
        stats_window_size: int = 100,
        tensorboard_log: str | None = None,
        policy_kwargs: dict[str, Any] | None = None,
        verbose: int = 0,
        seed: int | None = None,
        device: th.device | str = "auto",
        _init_setup_model: bool = True,
    ) -> None:
        if temperature_initial <= 0.0:
            raise ValueError("temperature_initial must be greater than zero")
        if temperature_final <= 0.0:
            raise ValueError("temperature_final must be greater than zero")
        if temperature_fraction <= 0.0:
            raise ValueError("temperature_fraction must be greater than zero")

        self.temperature_initial = float(temperature_initial)
        self.temperature_final = float(temperature_final)
        self.temperature_fraction = float(temperature_fraction)
        self.temperature = self.temperature_initial

        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            n_steps=n_steps,
            target_update_interval=target_update_interval,
            exploration_fraction=temperature_fraction,
            exploration_initial_eps=0.0,
            exploration_final_eps=0.0,
            max_grad_norm=max_grad_norm,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )

    def _setup_model(self) -> None:
        super()._setup_model()
        self.temperature_schedule = LinearSchedule(
            self.temperature_initial,
            self.temperature_final,
            self.temperature_fraction,
        )
        self.temperature = self.temperature_schedule(self._current_progress_remaining)

    def _on_step(self) -> None:
        self._n_calls += 1
        if self._n_calls % max(self.target_update_interval // self.n_envs, 1) == 0:
            polyak_update(
                self.q_net.parameters(),
                self.q_net_target.parameters(),
                self.tau,
            )
            polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self.temperature = self.temperature_schedule(self._current_progress_remaining)
        self.logger.record("rollout/temperature", self.temperature)

    def predict(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None]:
        """Use argmax actions when deterministic and Boltzmann samples otherwise."""
        if deterministic:
            return self.policy.predict(
                observation,
                state=state,
                episode_start=episode_start,
                deterministic=True,
            )

        actions = self._sample_boltzmann_actions(observation)
        return actions, state

    def learn(
        self: SelfBoltzmannDQN,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "BoltzmannDQN",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfBoltzmannDQN:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _sample_boltzmann_actions(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
    ) -> np.ndarray:
        if not isinstance(self.action_space, spaces.Discrete):
            raise TypeError("BoltzmannDQN requires a Discrete action space")

        action_probs, vectorized_env = self._boltzmann_action_probabilities(
            observation,
        )
        sampled_actions = th.multinomial(action_probs, num_samples=1).squeeze(1)
        actions = sampled_actions.cpu().numpy().reshape((-1,))
        if not vectorized_env:
            actions = actions.squeeze(axis=0)
        return actions

    def _boltzmann_action_probabilities(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
    ) -> tuple[th.Tensor, bool]:
        if not isinstance(self.action_space, spaces.Discrete):
            raise TypeError("BoltzmannDQN requires a Discrete action space")

        self.policy.set_training_mode(False)
        obs_tensor, vectorized_env = self.policy.obs_to_tensor(observation)
        with th.no_grad():
            q_values = self.q_net(obs_tensor)
            action_probs = th.softmax(q_values / self.temperature, dim=1)
        return action_probs, vectorized_env


__all__ = ["BoltzmannDQN"]
