"""SAC for finite discrete action spaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, TypeVar

import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import (
    GymEnv,
    MaybeCallback,
    PyTorchObs,
    Schedule,
)
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update

from .actor import DiscreteSACActor
from .critic import DiscreteSACCritic
from .policies import (
    CnnPolicy,
    DiscreteSACPolicy,
    MlpPolicy,
    MultiInputPolicy,
)

SelfDiscreteSAC = TypeVar("SelfDiscreteSAC", bound="DiscreteSAC")


class DiscreteSAC(OffPolicyAlgorithm):
    """Soft Actor-Critic for finite ``spaces.Discrete`` action spaces.

    The actor is categorical and the critic outputs one Q-value per action. The
    soft value terms are computed exactly by summing over the finite action set.
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }

    actor: DiscreteSACActor
    critic: DiscreteSACCritic
    critic_target: DiscreteSACCritic
    policy: DiscreteSACPolicy

    def __init__(
        self,
        policy: str | type[DiscreteSACPolicy],
        env: GymEnv | str,
        learning_rate: float | Schedule = 3e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: int | tuple[int, str] = 1,
        gradient_steps: int = 1,
        replay_buffer_class: type[ReplayBuffer] | None = None,
        replay_buffer_kwargs: dict[str, Any] | None = None,
        optimize_memory_usage: bool = False,
        n_steps: int = 1,
        ent_coef: str | float = "auto",
        target_update_interval: int = 1,
        target_entropy: str | float = "auto",
        stats_window_size: int = 100,
        tensorboard_log: str | None = None,
        policy_kwargs: dict[str, Any] | None = None,
        verbose: int = 0,
        seed: int | None = None,
        device: th.device | str = "auto",
        _init_setup_model: bool = True,
    ) -> None:
        self.target_update_interval = int(target_update_interval)
        self.ent_coef = ent_coef
        self.target_entropy = target_entropy
        self.log_ent_coef: th.Tensor | None = None
        self.ent_coef_optimizer: th.optim.Optimizer | None = None
        self.ent_coef_tensor: th.Tensor

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
            action_noise=None,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            n_steps=n_steps,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            sde_support=False,
            supported_action_spaces=(spaces.Discrete,),
            support_multi_env=True,
        )

        if self.target_update_interval <= 0:
            raise ValueError("target_update_interval must be greater than zero")

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        self.batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(
            self.critic_target,
            ["running_"],
        )

        if not isinstance(self.action_space, spaces.Discrete):
            raise TypeError("DiscreteSAC requires a Discrete action space")
        if self.target_entropy == "auto":
            self.target_entropy = -float(np.log(int(self.action_space.n)))
        else:
            self.target_entropy = float(self.target_entropy)

        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                if init_value <= 0.0:
                    raise ValueError("The initial ent_coef value must be positive")
            self.log_ent_coef = th.log(
                th.ones(1, device=self.device) * init_value
            ).requires_grad_(True)
            self.ent_coef_optimizer = th.optim.Adam(
                [self.log_ent_coef],
                lr=self.lr_schedule(1),
            )
        else:
            self.ent_coef_tensor = th.tensor(float(self.ent_coef), device=self.device)

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        """Run SAC updates with exact action expectations."""
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers.append(self.ent_coef_optimizer)
        self._update_learning_rate(optimizers)

        ent_coef_losses: list[float] = []
        ent_coefs: list[float] = []
        actor_losses: list[float] = []
        critic_losses: list[float] = []

        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(  # type: ignore[union-attr]
                batch_size,
                env=self._vec_normalize_env,
            )
            discounts = replay_data.discounts
            if discounts is None:
                discounts = self.gamma
            policy_eval = self._evaluate_policy(replay_data.observations)
            expected_log_prob = self._expected_log_prob(policy_eval)

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
                next_policy_eval = self._evaluate_policy(replay_data.next_observations)
                next_min_q = self._min_q_values(
                    self.critic_target,
                    replay_data.next_observations,
                )
                next_value = self._expected_soft_value(
                    next_policy_eval,
                    next_min_q,
                    ent_coef,
                )
                target_q_values = replay_data.rewards + (
                    1 - replay_data.dones
                ) * discounts * next_value.reshape(-1, 1)

            action_indices = replay_data.actions.long()
            current_q_values = [
                q_values.gather(dim=1, index=action_indices)
                for q_values in self.critic(replay_data.observations)
            ]
            critic_loss = (
                0.5
                * th.stack(
                    [
                        F.mse_loss(current_q, target_q_values)
                        for current_q in current_q_values
                    ]
                ).sum()
            )
            critic_losses.append(float(critic_loss.item()))

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            min_q = self._min_q_values(self.critic, replay_data.observations)
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

    def learn(
        self: SelfDiscreteSAC,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "DiscreteSAC",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfDiscreteSAC:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _evaluate_policy(self, observations: PyTorchObs) -> _DiscretePolicyEval:
        action_probs, log_probs = self.actor.action_probabilities(observations)
        return _DiscretePolicyEval(
            action_probs=action_probs,
            log_probs=log_probs,
        )

    def _min_q_values(
        self,
        critic: DiscreteSACCritic,
        observations: PyTorchObs,
    ) -> th.Tensor:
        q_values = th.stack(critic(observations), dim=0)
        min_q, _ = th.min(q_values, dim=0)
        return min_q

    def _expected_log_prob(self, policy_eval: _DiscretePolicyEval) -> th.Tensor:
        return (policy_eval.action_probs * policy_eval.log_probs).sum(dim=1)

    def _expected_soft_value(
        self,
        policy_eval: _DiscretePolicyEval,
        min_q: th.Tensor,
        ent_coef: th.Tensor,
    ) -> th.Tensor:
        return (
            policy_eval.action_probs * (min_q - ent_coef * policy_eval.log_probs)
        ).sum(dim=1)

    def _actor_loss(
        self,
        policy_eval: _DiscretePolicyEval,
        min_q: th.Tensor,
        ent_coef: th.Tensor,
    ) -> th.Tensor:
        return (
            (policy_eval.action_probs * (ent_coef * policy_eval.log_probs - min_q))
            .sum(dim=1)
            .mean()
        )

    def _excluded_save_params(self) -> list[str]:
        return [*super()._excluded_save_params(), "actor", "critic", "critic_target"]

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        if self.ent_coef_optimizer is not None:
            state_dicts.append("ent_coef_optimizer")
            saved_pytorch_variables = ["log_ent_coef"]
        else:
            saved_pytorch_variables = ["ent_coef_tensor"]
        return state_dicts, saved_pytorch_variables


@dataclass(frozen=True)
class _DiscretePolicyEval:
    """Policy probabilities used by exact discrete SAC expectations."""

    action_probs: th.Tensor
    log_probs: th.Tensor


__all__ = ["DiscreteSAC"]
