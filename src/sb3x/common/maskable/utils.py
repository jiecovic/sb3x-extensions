"""Environment helpers for invalid-action masking."""

from __future__ import annotations

import numpy as np
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.vec_env import VecEnv

EXPECTED_METHOD_NAME = "action_masks"


def get_action_masks(env: GymEnv) -> np.ndarray:
    """Return the current invalid-action masks exposed by an env or wrapper."""
    if isinstance(env, VecEnv):
        return np.stack(env.env_method(EXPECTED_METHOD_NAME))
    return env.get_wrapper_attr(EXPECTED_METHOD_NAME)()


def is_masking_supported(env: GymEnv) -> bool:
    """Return whether an env or wrapper exposes an ``action_masks()`` method."""
    if isinstance(env, VecEnv):
        return env.has_attr(EXPECTED_METHOD_NAME)

    try:
        env.get_wrapper_attr(EXPECTED_METHOD_NAME)
    except AttributeError:
        return False
    return True
