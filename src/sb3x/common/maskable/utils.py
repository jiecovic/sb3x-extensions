"""Environment helpers for invalid-action masking."""

from __future__ import annotations

import numpy as np
from gymnasium import spaces
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


def mask_dims_for_action_space(action_space: spaces.Space) -> int:
    """Return the flattened mask width for an SB3-Contrib-style action space."""
    if isinstance(action_space, spaces.Discrete):
        return int(action_space.n)
    if isinstance(action_space, spaces.MultiDiscrete):
        return sum(int(action_dim) for action_dim in action_space.nvec.ravel())
    if isinstance(action_space, spaces.MultiBinary):
        if not isinstance(action_space.n, int):
            raise ValueError(
                "Multi-dimensional MultiBinary action spaces are not supported"
            )
        return 2 * action_space.n
    raise ValueError(f"Unsupported action space {type(action_space)}")


def make_all_valid_action_masks(
    *,
    buffer_size: int,
    n_envs: int,
    mask_dims: int,
) -> np.ndarray:
    """Create the default all-valid mask storage for rollout buffers."""
    return np.ones((buffer_size, n_envs, mask_dims), dtype=np.float32)


def reshape_action_masks(
    action_masks: np.ndarray,
    *,
    n_envs: int,
    mask_dims: int,
) -> np.ndarray:
    """Normalize one rollout step's masks to ``(n_envs, mask_dims)``."""
    return np.asarray(action_masks, dtype=np.float32).reshape((n_envs, mask_dims))
