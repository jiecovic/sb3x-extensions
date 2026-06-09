"""Action-space helpers for hybrid continuous/discrete algorithms."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from gymnasium import spaces

HybridAction: TypeAlias = dict[str, np.ndarray]

CONTINUOUS_ACTION_KEY = "continuous"
DISCRETE_ACTION_KEY = "discrete"
HYBRID_ACTION_KEYS = frozenset({CONTINUOUS_ACTION_KEY, DISCRETE_ACTION_KEY})


@dataclass(frozen=True)
class HybridActionSpec:
    """Validated shape metadata for one supported hybrid action space."""

    action_space: spaces.Dict
    continuous_space: spaces.Box
    discrete_space: spaces.MultiDiscrete

    @property
    def continuous_shape(self) -> tuple[int, ...]:
        return tuple(int(dim) for dim in self.continuous_space.shape)

    @property
    def discrete_shape(self) -> tuple[int, ...]:
        return tuple(int(dim) for dim in self.discrete_space.nvec.shape)

    @property
    def continuous_dim(self) -> int:
        return int(np.prod(self.continuous_shape, dtype=np.int64))

    @property
    def discrete_dim(self) -> int:
        return int(self.discrete_space.nvec.size)

    @property
    def discrete_action_dims(self) -> list[int]:
        return [int(action_dim) for action_dim in self.discrete_space.nvec.ravel()]

    @property
    def discrete_logits_dim(self) -> int:
        return sum(self.discrete_action_dims)

    @property
    def flat_dim(self) -> int:
        return self.continuous_dim + self.discrete_dim

    @property
    def flat_low(self) -> np.ndarray:
        discrete_low = np.zeros(self.discrete_dim, dtype=np.float32)
        return np.concatenate(
            [
                self.continuous_space.low.astype(np.float32).reshape(-1),
                discrete_low,
            ],
        )

    @property
    def flat_high(self) -> np.ndarray:
        discrete_high = self.discrete_space.nvec.astype(np.float32).reshape(-1) - 1.0
        return np.concatenate(
            [
                self.continuous_space.high.astype(np.float32).reshape(-1),
                discrete_high,
            ],
        )

    @property
    def flat_action_space(self) -> spaces.Box:
        """Return the flat action space used internally by SB3 algorithms."""
        return spaces.Box(
            low=self.flat_low,
            high=self.flat_high,
            shape=(self.flat_dim,),
            dtype=np.float32,
        )

    def flatten_action(self, action: HybridAction) -> np.ndarray:
        """Convert a public hybrid action dict to the internal flat action."""
        continuous = np.asarray(
            action[CONTINUOUS_ACTION_KEY],
            dtype=np.float32,
        ).reshape(-1)
        discrete = np.asarray(
            action[DISCRETE_ACTION_KEY],
            dtype=np.float32,
        ).reshape(-1)

        if continuous.size != self.continuous_dim:
            raise ValueError(
                "Continuous action has "
                f"{continuous.size} values, expected {self.continuous_dim}"
            )
        if discrete.size != self.discrete_dim:
            raise ValueError(
                f"Discrete action has {discrete.size} values, "
                f"expected {self.discrete_dim}"
            )

        return np.concatenate([continuous, discrete]).astype(np.float32)

    def normalize_action(self, action: Mapping[str, np.ndarray]) -> HybridAction:
        """Validate and dtype-normalize one public hybrid action dict."""
        return self.unflatten_action(self.flatten_action(dict(action)))

    def unflatten_action(self, action: np.ndarray) -> HybridAction:
        """Convert one internal flat action to the public hybrid action dict."""
        flat_action = np.asarray(action, dtype=np.float32).reshape(-1)
        if flat_action.size != self.flat_dim:
            raise ValueError(
                f"Flat action has {flat_action.size} values, expected {self.flat_dim}"
            )

        continuous_flat = flat_action[: self.continuous_dim]
        discrete_flat = flat_action[self.continuous_dim :]

        continuous = np.clip(
            continuous_flat,
            self.continuous_space.low.reshape(-1),
            self.continuous_space.high.reshape(-1),
        ).astype(self.continuous_space.dtype)
        discrete = np.clip(
            np.rint(discrete_flat),
            np.zeros(self.discrete_dim, dtype=np.float32),
            self.discrete_space.nvec.reshape(-1) - 1,
        ).astype(self.discrete_space.dtype)

        return {
            CONTINUOUS_ACTION_KEY: continuous.reshape(self.continuous_shape),
            DISCRETE_ACTION_KEY: discrete.reshape(self.discrete_shape),
        }

    def unflatten_action_batch(self, actions: np.ndarray) -> list[HybridAction]:
        """Convert a batch of internal flat actions to public hybrid actions."""
        flat_actions = np.asarray(actions, dtype=np.float32)
        if flat_actions.ndim == 1:
            flat_actions = flat_actions.reshape(1, -1)
        if flat_actions.ndim != 2 or flat_actions.shape[1] != self.flat_dim:
            raise ValueError(
                "Expected flat action batch with shape "
                f"(n_envs, {self.flat_dim}), got {flat_actions.shape}"
            )
        return [self.unflatten_action(action) for action in flat_actions]


@dataclass(frozen=True)
class HybridActionGroupNames:
    """Names for each entropy-producing branch in a hybrid action space."""

    continuous: tuple[str, ...]
    discrete: tuple[str, ...]

    @property
    def all(self) -> tuple[str, ...]:
        return (*self.continuous, *self.discrete)


def make_hybrid_action_group_names(
    spec: HybridActionSpec,
    names: Mapping[str, Sequence[str]] | None = None,
) -> HybridActionGroupNames:
    """Validate optional user-facing names for hybrid entropy components."""
    if names is None:
        return HybridActionGroupNames(
            continuous=tuple(
                f"continuous_{index}" for index in range(spec.continuous_dim)
            ),
            discrete=tuple(f"discrete_{index}" for index in range(spec.discrete_dim)),
        )

    continuous = tuple(str(name) for name in names.get(CONTINUOUS_ACTION_KEY, ()))
    discrete = tuple(str(name) for name in names.get(DISCRETE_ACTION_KEY, ()))
    if len(continuous) != spec.continuous_dim:
        raise ValueError(
            "continuous action group names must match the continuous action size "
            f"{spec.continuous_dim}, got {len(continuous)}"
        )
    if len(discrete) != spec.discrete_dim:
        raise ValueError(
            "discrete action group names must match the discrete action size "
            f"{spec.discrete_dim}, got {len(discrete)}"
        )

    all_names = (*continuous, *discrete)
    if len(set(all_names)) != len(all_names):
        raise ValueError("hybrid action group names must be unique")
    if any(not name for name in all_names):
        raise ValueError("hybrid action group names must be non-empty")

    return HybridActionGroupNames(continuous=continuous, discrete=discrete)


def make_hybrid_action_spec(action_space: spaces.Space) -> HybridActionSpec:
    """Validate and describe the supported ``Box + MultiDiscrete`` action space."""
    if not isinstance(action_space, spaces.Dict):
        raise TypeError(
            "Hybrid action algorithms require a gymnasium.spaces.Dict action space"
        )

    keys = frozenset(action_space.spaces.keys())
    if keys != HYBRID_ACTION_KEYS:
        raise ValueError(
            "Hybrid action spaces must have exactly the keys "
            f"{sorted(HYBRID_ACTION_KEYS)}, got {sorted(keys)}"
        )

    continuous_space = action_space.spaces[CONTINUOUS_ACTION_KEY]
    discrete_space = action_space.spaces[DISCRETE_ACTION_KEY]
    if not isinstance(continuous_space, spaces.Box):
        raise TypeError("Hybrid action 'continuous' branch must be spaces.Box")
    if not isinstance(discrete_space, spaces.MultiDiscrete):
        raise TypeError("Hybrid action 'discrete' branch must be spaces.MultiDiscrete")
    if not np.issubdtype(continuous_space.dtype, np.floating):
        raise TypeError("Hybrid action 'continuous' branch must use a floating dtype")
    if not np.all(np.isfinite(continuous_space.low)) or not np.all(
        np.isfinite(continuous_space.high)
    ):
        raise ValueError("Continuous hybrid action bounds must be finite")
    if np.any(discrete_space.nvec <= 0):
        raise ValueError("MultiDiscrete branch dimensions must be positive")
    if np.any(discrete_space.start != 0):
        raise ValueError("MultiDiscrete branch must use zero-based actions")
    if continuous_space.shape is not None:
        continuous_dim = int(np.prod(continuous_space.shape, dtype=np.int64))
    else:
        continuous_dim = 0
    if continuous_dim + int(discrete_space.nvec.size) == 0:
        raise ValueError("Hybrid action space must expose at least one action")

    return HybridActionSpec(
        action_space=action_space,
        continuous_space=continuous_space,
        discrete_space=discrete_space,
    )
