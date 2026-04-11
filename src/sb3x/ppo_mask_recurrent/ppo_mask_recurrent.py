"""Local implementation of ``MaskableRecurrentPPO``.

This module stays structurally close to ``sb3_contrib``'s recurrent PPO so the
maskable additions remain easy to diff and review. When action masks are not
used, the algorithm should keep matching recurrent PPO behavior.
"""

from copy import deepcopy
from typing import Any, ClassVar, TypeVar

import numpy as np
import torch as th
from gymnasium import spaces
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

from sb3x.common.maskable import get_action_masks, is_masking_supported
from sb3x.common.maskable.recurrent_buffers import (
    MaskableRecurrentDictRolloutBuffer,
    MaskableRecurrentRolloutBuffer,
)
from sb3x.common.recurrent import (
    PolicyObs,
    make_recurrent_states,
    predict_recurrent_values,
    recurrent_hidden_state_buffer_shape,
)

from .policies import (
    CnnLstmPolicy,
    MaskableRecurrentActorCriticPolicy,
    MlpLstmPolicy,
    MultiInputLstmPolicy,
)

SelfMaskableRecurrentPPO = TypeVar(
    "SelfMaskableRecurrentPPO", bound="MaskableRecurrentPPO"
)
ActionMasks = np.ndarray | th.Tensor | None


def _forward_recurrent_policy(
    policy: MaskableRecurrentActorCriticPolicy,
    obs: PolicyObs,
    lstm_states: RNNStates,
    episode_starts: th.Tensor,
    action_masks: ActionMasks = None,
) -> tuple[th.Tensor, th.Tensor, th.Tensor, RNNStates]:
    """Isolate the weakly-typed recurrent policy call boundary."""
    return policy.forward(
        obs,
        lstm_states,
        episode_starts,
        action_masks=action_masks,
    )


def _evaluate_recurrent_actions(
    policy: MaskableRecurrentActorCriticPolicy,
    obs: PolicyObs,
    actions: th.Tensor,
    lstm_states: RNNStates,
    episode_starts: th.Tensor,
    action_masks: ActionMasks = None,
) -> tuple[th.Tensor, th.Tensor, th.Tensor | None]:
    """Isolate the weakly-typed upstream recurrent action-eval boundary."""
    return policy.evaluate_actions(
        obs,
        actions,
        lstm_states,
        episode_starts,
        action_masks=action_masks,
    )


class MaskableRecurrentPPO(OnPolicyAlgorithm):
    """Recurrent PPO with optional invalid-action masking support."""

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpLstmPolicy": MlpLstmPolicy,
        "CnnLstmPolicy": CnnLstmPolicy,
        "MultiInputLstmPolicy": MultiInputLstmPolicy,
    }

    def __init__(
        self,
        policy: str | type[MaskableRecurrentActorCriticPolicy],
        env: GymEnv | str,
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
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
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

    @property
    def recurrent_policy(self) -> MaskableRecurrentActorCriticPolicy:
        """Return the policy narrowed to the recurrent policy type."""
        if not isinstance(self.policy, MaskableRecurrentActorCriticPolicy):
            raise TypeError("Policy must subclass MaskableRecurrentActorCriticPolicy")
        return self.policy

    @property
    def recurrent_rollout_buffer(
        self,
    ) -> MaskableRecurrentRolloutBuffer | MaskableRecurrentDictRolloutBuffer:
        """Return the rollout buffer narrowed to the recurrent variants."""
        if not isinstance(
            self.rollout_buffer,
            (MaskableRecurrentRolloutBuffer, MaskableRecurrentDictRolloutBuffer),
        ):
            raise TypeError(f"{self.rollout_buffer} doesn't support recurrent policy")
        return self.rollout_buffer

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        buffer_cls = (
            MaskableRecurrentDictRolloutBuffer
            if isinstance(self.observation_space, spaces.Dict)
            else MaskableRecurrentRolloutBuffer
        )

        policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs,
        )
        policy = policy.to(self.device)
        if not isinstance(policy, MaskableRecurrentActorCriticPolicy):
            raise TypeError("Policy must subclass MaskableRecurrentActorCriticPolicy")

        self.policy = policy
        lstm = policy.lstm_actor

        self._last_lstm_states = make_recurrent_states(
            lstm,
            n_envs=self.n_envs,
            device=self.device,
        )
        hidden_state_buffer_shape = recurrent_hidden_state_buffer_shape(
            lstm,
            n_steps=self.n_steps,
            n_envs=self.n_envs,
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
        use_masking: bool = True,
    ) -> bool:
        """Collect experiences using the current recurrent policy."""
        assert isinstance(
            rollout_buffer,
            (MaskableRecurrentRolloutBuffer, MaskableRecurrentDictRolloutBuffer),
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
        action_masks = None
        if self.use_sde:
            policy.reset_noise(env.num_envs)

        if use_masking and not is_masking_supported(env):
            raise ValueError(
                "Environment does not support action masking. Consider using an "
                "ActionMasker wrapper or exposing action_masks()."
            )

        callback.on_rollout_start()

        lstm_states = deepcopy(self._last_lstm_states)

        while n_steps < n_rollout_steps:
            if (
                self.use_sde
                and self.sde_sample_freq > 0
                and n_steps % self.sde_sample_freq == 0
            ):
                policy.reset_noise(env.num_envs)

            with th.no_grad():
                obs_tensor = obs_as_tensor(last_obs, self.device)
                episode_starts = th.tensor(
                    last_episode_starts,
                    dtype=th.float32,
                    device=self.device,
                )
                if use_masking:
                    action_masks = get_action_masks(env)
                actions, values, log_probs, lstm_states = _forward_recurrent_policy(
                    policy,
                    obs_tensor,
                    lstm_states,
                    episode_starts,
                    action_masks=action_masks,
                )

            actions = actions.cpu().numpy()
            new_obs, rewards, dones, infos = env.step(actions)

            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)

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
                        terminal_value = predict_recurrent_values(
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
                action_masks=action_masks,
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
            values = predict_recurrent_values(
                policy,
                obs_as_tensor(new_obs, self.device),
                (lstm_states.vf[0], lstm_states.vf[1]),
                episode_starts,
            )

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """Update the policy using the currently gathered rollout buffer."""
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
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                sequence_mask = rollout_data.sequence_mask > 1e-8

                values, log_prob, entropy = _evaluate_recurrent_actions(
                    policy,
                    rollout_data.observations,
                    actions,
                    rollout_data.lstm_states,
                    rollout_data.episode_starts,
                    action_masks=rollout_data.action_masks,
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

                value_loss = th.mean(
                    ((rollout_data.returns - values_pred) ** 2)[sequence_mask]
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
        if hasattr(policy, "log_std"):
            self.logger.record("train/std", th.exp(policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if clip_range_vf_value is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf_value)

    def predict(  # type: ignore[override]
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
        action_masks: np.ndarray | None = None,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None]:
        """Predict an action, optionally constrained by invalid-action masks."""
        return self.recurrent_policy.predict(
            observation,
            state=state,
            episode_start=episode_start,
            deterministic=deterministic,
            action_masks=action_masks,
        )

    def learn(  # type: ignore[override]
        self: SelfMaskableRecurrentPPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "MaskableRecurrentPPO",
        reset_num_timesteps: bool = True,
        use_masking: bool = True,
        progress_bar: bool = False,
    ) -> SelfMaskableRecurrentPPO:
        """Learn and return ``self`` with the narrowed algorithm type.

        Masking is enabled by default, matching ``MaskablePPO`` behavior.
        Pass ``use_masking=False`` to exercise the unmasked recurrent PPO path.
        """
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

    def _excluded_save_params(self) -> list[str]:
        """Exclude recurrent rollout state from persistence."""
        return super()._excluded_save_params() + ["_last_lstm_states"]  # noqa: RUF005
