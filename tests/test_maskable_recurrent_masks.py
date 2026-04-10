"""Mask-specific checks for the local recurrent PPO implementation."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch as th
from gymnasium import spaces

from sb3x import MaskableRecurrentPPO


class MaskedBanditEnv(gym.Env[np.ndarray, int]):
    """Tiny env exposing a constant invalid-action mask."""

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)
        self._mask = np.array([False, True, False], dtype=bool)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        super().reset(seed=seed)
        del options
        return np.array([0.0], dtype=np.float32), {}

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        reward = 1.0 if int(action) == 1 else 0.0
        return np.array([0.0], dtype=np.float32), reward, True, False, {}

    def action_masks(self) -> np.ndarray:
        return self._mask.copy()


class UnmaskedBanditEnv(MaskedBanditEnv):
    """Same toy env without the action-mask API."""

    def __getattribute__(self, name: str) -> object:
        if name == "action_masks":
            raise AttributeError(name)
        return super().__getattribute__(name)


def _build_model(env: gym.Env[np.ndarray, int]) -> MaskableRecurrentPPO:
    return MaskableRecurrentPPO(
        "MlpLstmPolicy",
        env,
        seed=123,
        device="cpu",
        n_steps=4,
        batch_size=4,
        n_epochs=1,
        policy_kwargs={
            "lstm_hidden_size": 8,
            "n_lstm_layers": 1,
            "net_arch": [8],
        },
    )


def test_predict_respects_action_masks() -> None:
    """Masked predict() should eliminate forbidden discrete actions."""
    model = _build_model(MaskedBanditEnv())
    obs = np.array([0.0], dtype=np.float32)

    action_net = model.policy.action_net
    if not isinstance(action_net, th.nn.Linear):
        raise TypeError("Expected a linear action head for the toy discrete env")

    with th.no_grad():
        action_net.weight.fill_(0.0)
        if action_net.bias is None:
            raise TypeError("Expected a bias term on the action head")
        action_net.bias.copy_(th.tensor([3.0, 2.0, 1.0]))

    unmasked_action, _ = model.predict(obs, deterministic=True)
    masked_action, _ = model.predict(
        obs,
        deterministic=True,
        action_masks=np.array([False, True, True], dtype=bool),
    )

    assert int(np.asarray(unmasked_action).item()) == 0
    assert int(np.asarray(masked_action).item()) == 1


def test_learn_defaults_to_masking_and_populates_recurrent_action_masks() -> None:
    """Default learn() should store the env masks in the rollout buffer."""
    model = _build_model(MaskedBanditEnv())

    model.learn(total_timesteps=4)

    assert model.recurrent_rollout_buffer.action_masks.shape == (4, 3)
    np.testing.assert_array_equal(
        model.recurrent_rollout_buffer.action_masks[0],
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
    )


def test_learn_requires_env_support_when_masking_is_left_on() -> None:
    """Default learn() should fail fast when the env does not expose masks."""
    model = _build_model(UnmaskedBanditEnv())

    with pytest.raises(ValueError, match="support action masking"):
        model.learn(total_timesteps=4)
