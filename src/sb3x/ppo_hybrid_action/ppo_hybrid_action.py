"""PPO wrapper for hybrid continuous/discrete action environments."""

from __future__ import annotations

from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from sb3x.common.hybrid_action import (
    HybridAction,
    HybridActionSpec,
    has_public_hybrid_action_space,
    make_hybrid_action_spec,
    prepare_hybrid_action_env,
    wrap_hybrid_action_env,
)

from .policies import CnnPolicy, MlpPolicy, MultiInputPolicy


class HybridActionPPO(PPO):
    """PPO for ``Dict(continuous=Box, discrete=MultiDiscrete)`` action spaces.

    The public environment keeps its hybrid action dictionary. Internally the
    algorithm presents SB3's PPO rollout and training code with a flat ``Box``
    action space, then uses a custom policy distribution to split that flat
    action into independent Gaussian and MultiCategorical branches.
    """

    policy_aliases: ClassVar[dict[str, type[ActorCriticPolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }

    hybrid_action_spec: HybridActionSpec

    def __init__(
        self,
        policy: str | type[ActorCriticPolicy],
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
        if use_sde:
            raise ValueError("HybridActionPPO does not support gSDE")

        wrapped_env, hybrid_action_spec, policy_kwargs = prepare_hybrid_action_env(
            env,
            {} if policy_kwargs is None else policy_kwargs,
            algorithm_name="HybridActionPPO",
            init_setup_model=_init_setup_model,
        )
        if hybrid_action_spec is not None:
            self.hybrid_action_spec = hybrid_action_spec

        super().__init__(
            policy=policy,
            # SB3 load() passes env=None even though PPO.__init__ annotates it
            # as required; _init_setup_model=False keeps that path valid.
            env=wrapped_env,  # pyright: ignore[reportArgumentType]
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
            use_sde=False,
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
        """Initialize SB3 PPO and refresh hybrid action metadata after loading."""
        super()._setup_model()
        hybrid_action_space = self.policy_kwargs.get("hybrid_action_space")
        if hybrid_action_space is not None:
            self.hybrid_action_spec = make_hybrid_action_spec(hybrid_action_space)

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

    def _unflatten_predicted_actions(
        self,
        actions: np.ndarray,
    ) -> HybridAction | list[HybridAction]:
        actions_array = np.asarray(actions, dtype=np.float32)
        if actions_array.ndim == 1:
            return self.hybrid_action_spec.unflatten_action(actions_array)
        return self.hybrid_action_spec.unflatten_action_batch(actions_array)
