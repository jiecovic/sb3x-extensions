"""Environment wrappers that expose hybrid actions as flat SB3 actions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import gymnasium as gym
import numpy as np
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.vec_env import VecEnvWrapper
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnv,
    VecEnvObs,
    VecEnvStepReturn,
)

from .spaces import HybridAction, HybridActionSpec, make_hybrid_action_spec


class HybridActionEnvWrapper(gym.ActionWrapper):
    """Expose a hybrid ``Dict`` action env as a flat ``Box`` action env."""

    hybrid_action_spec: HybridActionSpec

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.hybrid_action_spec = make_hybrid_action_spec(env.action_space)
        self.action_space = self.hybrid_action_spec.flat_action_space

    def action(self, action: np.ndarray | HybridAction) -> HybridAction:
        """Convert flat model actions to the wrapped env's hybrid action."""
        if isinstance(action, Mapping):
            return self.hybrid_action_spec.normalize_action(action)
        return self.hybrid_action_spec.unflatten_action(
            np.asarray(action, dtype=np.float32)
        )


class HybridActionVecEnvWrapper(VecEnvWrapper):
    """Vectorized wrapper that converts flat batches to hybrid action dicts."""

    hybrid_action_spec: HybridActionSpec

    def __init__(self, venv: VecEnv) -> None:
        self.hybrid_action_spec = make_hybrid_action_spec(venv.action_space)
        super().__init__(venv, action_space=self.hybrid_action_spec.flat_action_space)

    def reset(self) -> VecEnvObs:
        return self.venv.reset()

    def step_async(
        self,
        actions: np.ndarray | Sequence[HybridAction] | HybridAction,
    ) -> None:
        converted_actions = self._convert_actions(actions)
        # SB3 types VecEnv actions as ndarray, but this wrapper targets a
        # Dict-action VecEnv below the flat-action model boundary.
        self.venv.step_async(converted_actions)  # pyright: ignore[reportArgumentType]

    def step_wait(self) -> VecEnvStepReturn:
        return self.venv.step_wait()

    def _convert_actions(
        self,
        actions: np.ndarray | Sequence[HybridAction] | HybridAction,
    ) -> Sequence[HybridAction] | HybridAction:
        if isinstance(actions, Mapping):
            return self.hybrid_action_spec.normalize_action(actions)
        if _is_hybrid_action_sequence(actions):
            return [
                self.hybrid_action_spec.normalize_action(action) for action in actions
            ]
        return self.hybrid_action_spec.unflatten_action_batch(
            np.asarray(actions, dtype=np.float32)
        )


def wrap_hybrid_action_env(env: GymEnv | str) -> tuple[GymEnv, HybridActionSpec]:
    """Wrap a hybrid-action env so SB3 PPO sees a flat ``Box`` action space."""
    if isinstance(env, str):
        env = gym.make(env)

    if isinstance(env, VecEnv):
        wrapped_vec_env = HybridActionVecEnvWrapper(env)
        return wrapped_vec_env, wrapped_vec_env.hybrid_action_spec
    if isinstance(env, gym.Env):
        wrapped_env = HybridActionEnvWrapper(env)
        return wrapped_env, wrapped_env.hybrid_action_spec

    raise TypeError(f"Unsupported environment type: {type(env)}")


def _is_hybrid_action_sequence(
    value: object,
) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, np.ndarray)
        and len(value) > 0
        and isinstance(value[0], Mapping)
    )
