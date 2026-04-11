"""Recurrent PPO wrapper for hybrid continuous/discrete actions."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, ClassVar, TypeVar

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium import spaces
from sb3_contrib.common.recurrent.buffers import (
    RecurrentDictRolloutBuffer,
    RecurrentRolloutBuffer,
)
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import (
    FloatSchedule,
    explained_variance,
    obs_as_tensor,
)
from stable_baselines3.common.vec_env import VecEnv
from torch.nn import functional as F

from sb3x.common.hybrid_action import (
    HybridAction,
    HybridActionSpec,
    has_public_hybrid_action_space,
    make_hybrid_action_spec,
    prepare_hybrid_action_env,
    wrap_hybrid_action_env,
)

from .policies import (
    CnnLstmPolicy,
    HybridRecurrentActorCriticPolicy,
    MlpLstmPolicy,
    MultiInputLstmPolicy,
)

SelfHybridRecurrentPPO = TypeVar(
    "SelfHybridRecurrentPPO",
    bound="HybridRecurrentPPO",
)
PolicyObs = th.Tensor | dict[str, th.Tensor]
LSTMState = tuple[th.Tensor, th.Tensor]


def _forward_recurrent_policy(
    policy: HybridRecurrentActorCriticPolicy,
    obs: PolicyObs,
    lstm_states: RNNStates,
    episode_starts: th.Tensor,
) -> tuple[th.Tensor, th.Tensor, th.Tensor, RNNStates]:
    """Isolate the weakly-typed recurrent policy call boundary."""
    return policy(  # pyright: ignore[reportArgumentType, reportCallIssue]
        obs,
        lstm_states,
        episode_starts,
    )


def _evaluate_recurrent_actions(
    policy: HybridRecurrentActorCriticPolicy,
    obs: PolicyObs,
    actions: th.Tensor,
    lstm_states: RNNStates,
    episode_starts: th.Tensor,
) -> tuple[th.Tensor, th.Tensor, th.Tensor | None]:
    """Isolate the weakly-typed upstream recurrent action-eval boundary."""
    return policy.evaluate_actions(
        obs,  # pyright: ignore[reportArgumentType]
        actions,
        lstm_states,
        episode_starts,
    )


def _predict_recurrent_values(
    policy: HybridRecurrentActorCriticPolicy,
    obs: PolicyObs,
    lstm_states: LSTMState,
    episode_starts: th.Tensor,
) -> th.Tensor:
    """Isolate the weakly-typed upstream recurrent value-prediction boundary."""
    return policy.predict_values(
        obs,  # pyright: ignore[reportArgumentType]
        lstm_states,
        episode_starts,
    )


class HybridRecurrentPPO(OnPolicyAlgorithm):
    """Recurrent PPO for ``Dict(continuous=Box, discrete=MultiDiscrete)`` actions."""

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpLstmPolicy": MlpLstmPolicy,
        "CnnLstmPolicy": CnnLstmPolicy,
        "MultiInputLstmPolicy": MultiInputLstmPolicy,
    }

    hybrid_action_spec: HybridActionSpec

    def __init__(
        self,
        policy: str | type[HybridRecurrentActorCriticPolicy],
        env: GymEnv | str | None,
        learning_rate: float | Schedule = 3e-4,
        n_steps: int = 128,
        batch_size: int | None = 128,
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
        target_kl: float | None = None,
        stats_window_size: int = 100,
        tensorboard_log: str | None = None,
        policy_kwargs: dict[str, Any] | None = None,
        verbose: int = 0,
        seed: int | None = None,
        device: th.device | str = "auto",
        _init_setup_model: bool = True,
    ) -> None:
        if use_sde:
            raise ValueError("HybridRecurrentPPO does not support gSDE")

        wrapped_env, hybrid_action_spec, policy_kwargs = prepare_hybrid_action_env(
            env,
            {} if policy_kwargs is None else policy_kwargs,
            algorithm_name="HybridRecurrentPPO",
            init_setup_model=_init_setup_model,
        )
        if hybrid_action_spec is not None:
            self.hybrid_action_spec = hybrid_action_spec

        super().__init__(
            policy,
            # SB3 load() passes env=None; _init_setup_model=False keeps that valid.
            env=wrapped_env,  # pyright: ignore[reportArgumentType]
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=False,
            sde_sample_freq=sde_sample_freq,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=False,
            supported_action_spaces=(spaces.Box,),
        )

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl
        self._last_lstm_states: RNNStates | None = None

        if _init_setup_model:
            self._setup_model()

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

    @property
    def recurrent_policy(self) -> HybridRecurrentActorCriticPolicy:
        """Return the policy narrowed to the hybrid recurrent policy type."""
        if not isinstance(self.policy, HybridRecurrentActorCriticPolicy):
            raise TypeError("Policy must subclass HybridRecurrentActorCriticPolicy")
        return self.policy

    @property
    def recurrent_rollout_buffer(
        self,
    ) -> RecurrentRolloutBuffer | RecurrentDictRolloutBuffer:
        """Return the rollout buffer narrowed to the recurrent variants."""
        if not isinstance(
            self.rollout_buffer,
            (RecurrentRolloutBuffer, RecurrentDictRolloutBuffer),
        ):
            raise TypeError(f"{self.rollout_buffer} doesn't support recurrent policy")
        return self.rollout_buffer

    def _setup_model(self) -> None:
        if not hasattr(self, "hybrid_action_spec"):
            hybrid_action_space = self.policy_kwargs.get("hybrid_action_space")
            if hybrid_action_space is None:
                raise ValueError("Missing hybrid_action_space in policy kwargs")
            self.hybrid_action_spec = make_hybrid_action_spec(hybrid_action_space)

        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        buffer_cls = (
            RecurrentDictRolloutBuffer
            if isinstance(self.observation_space, spaces.Dict)
            else RecurrentRolloutBuffer
        )

        policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=False,
            **self.policy_kwargs,
        )
        policy = policy.to(self.device)
        if not isinstance(policy, HybridRecurrentActorCriticPolicy):
            raise TypeError("Policy must subclass HybridRecurrentActorCriticPolicy")

        self.policy = policy
        lstm = policy.lstm_actor

        single_hidden_state_shape = (lstm.num_layers, self.n_envs, lstm.hidden_size)
        self._last_lstm_states = RNNStates(
            (
                th.zeros(single_hidden_state_shape, device=self.device),
                th.zeros(single_hidden_state_shape, device=self.device),
            ),
            (
                th.zeros(single_hidden_state_shape, device=self.device),
                th.zeros(single_hidden_state_shape, device=self.device),
            ),
        )

        hidden_state_buffer_shape = (
            self.n_steps,
            lstm.num_layers,
            self.n_envs,
            lstm.hidden_size,
        )

        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            hidden_state_buffer_shape,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

        self.clip_range = FloatSchedule(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, (
                    "`clip_range_vf` must be positive, "
                    "pass `None` to deactivate vf clipping"
                )

            self.clip_range_vf = FloatSchedule(self.clip_range_vf)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """Collect experiences using the current recurrent hybrid policy."""
        assert isinstance(
            rollout_buffer,
            (RecurrentRolloutBuffer, RecurrentDictRolloutBuffer),
        ), f"{rollout_buffer} doesn't support recurrent policy"

        last_obs = self._last_obs
        assert last_obs is not None, "No previous observation was provided"
        last_episode_starts = self._last_episode_starts
        assert last_episode_starts is not None, "Episode starts were not initialized"
        assert self._last_lstm_states is not None, "LSTM state was not initialized"

        policy = self.recurrent_policy
        policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        callback.on_rollout_start()

        lstm_states = deepcopy(self._last_lstm_states)
        dones = np.zeros(env.num_envs, dtype=bool)

        while n_steps < n_rollout_steps:
            with th.no_grad():
                obs_tensor = obs_as_tensor(last_obs, self.device)
                episode_starts = th.tensor(
                    last_episode_starts,
                    dtype=th.float32,
                    device=self.device,
                )
                actions, values, log_probs, lstm_states = _forward_recurrent_policy(
                    policy,
                    obs_tensor,
                    lstm_states,
                    episode_starts,
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
                        terminal_lstm_state = (
                            lstm_states.vf[0][:, idx : idx + 1, :].contiguous(),
                            lstm_states.vf[1][:, idx : idx + 1, :].contiguous(),
                        )
                        episode_starts = th.tensor(
                            [False],
                            dtype=th.float32,
                            device=self.device,
                        )
                        terminal_value = _predict_recurrent_values(
                            policy,
                            terminal_obs,
                            terminal_lstm_state,
                            episode_starts,
                        )[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                last_obs,
                actions,
                rewards,
                last_episode_starts,
                values,
                log_probs,
                lstm_states=self._last_lstm_states,
            )

            if isinstance(new_obs, tuple):
                raise TypeError("Tuple observations are not supported")

            self._last_obs = new_obs
            self._last_episode_starts = dones
            self._last_lstm_states = lstm_states
            last_obs = new_obs
            last_episode_starts = dones

        with th.no_grad():
            episode_starts = th.tensor(dones, dtype=th.float32, device=self.device)
            values = _predict_recurrent_values(
                policy,
                obs_as_tensor(last_obs, self.device),
                (lstm_states.vf[0], lstm_states.vf[1]),
                episode_starts,
            )

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        callback.on_rollout_end()
        return True

    def train(self) -> None:
        """Update policy using the currently gathered recurrent rollout buffer."""
        policy = self.recurrent_policy
        rollout_buffer = self.recurrent_rollout_buffer

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
                sequence_mask = rollout_data.mask > 1e-8

                values, log_prob, entropy = _evaluate_recurrent_actions(
                    policy,
                    rollout_data.observations,
                    rollout_data.actions,
                    rollout_data.lstm_states,
                    rollout_data.episode_starts,
                )
                values = values.flatten()

                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages[sequence_mask].mean()) / (
                        advantages[sequence_mask].std() + 1e-8
                    )

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(
                    ratio,
                    1 - clip_range,
                    1 + clip_range,
                )
                policy_loss = -th.mean(
                    th.min(policy_loss_1, policy_loss_2)[sequence_mask]
                )

                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean(
                    (th.abs(ratio - 1) > clip_range).float()[sequence_mask]
                ).item()
                clip_fractions.append(clip_fraction)

                if clip_range_vf_value is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values,
                        -clip_range_vf_value,
                        clip_range_vf_value,
                    )

                value_loss = F.mse_loss(
                    rollout_data.returns[sequence_mask],
                    values_pred[sequence_mask],
                )
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob[sequence_mask])
                else:
                    entropy_loss = -th.mean(entropy[sequence_mask])
                entropy_losses.append(entropy_loss.item())

                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                )

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = (
                        th.mean(((th.exp(log_ratio) - 1) - log_ratio)[sequence_mask])
                        .cpu()
                        .numpy()
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
    ) -> tuple[HybridAction | list[HybridAction], tuple[np.ndarray, ...] | None]:
        """Predict a public hybrid action and next recurrent state."""
        flat_action, next_state = self.recurrent_policy.predict(
            observation,
            state=state,
            episode_start=episode_start,
            deterministic=deterministic,
        )
        return self._unflatten_predicted_actions(flat_action), next_state

    def _unflatten_predicted_actions(
        self,
        actions: np.ndarray,
    ) -> HybridAction | list[HybridAction]:
        actions_array = np.asarray(actions, dtype=np.float32)
        if actions_array.ndim == 1:
            return self.hybrid_action_spec.unflatten_action(actions_array)
        return self.hybrid_action_spec.unflatten_action_batch(actions_array)

    def learn(  # pyright: ignore[reportIncompatibleMethodOverride]
        self: SelfHybridRecurrentPPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "HybridRecurrentPPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfHybridRecurrentPPO:
        """Learn and return ``self`` with the narrowed algorithm type."""
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

    def _excluded_save_params(self) -> list[str]:
        """Exclude recurrent rollout state from persistence."""
        return super()._excluded_save_params() + ["_last_lstm_states"]  # noqa: RUF005
