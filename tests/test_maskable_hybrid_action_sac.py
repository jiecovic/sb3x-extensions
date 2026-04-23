"""Sanity checks for MaskableHybridActionSAC."""

from __future__ import annotations

from typing import ClassVar

import gymnasium as gym
import numpy as np
import pytest
import torch as th
from gymnasium import spaces

from sb3x import MaskableHybridActionSAC
from sb3x.common.hybrid_action import HybridAction
from sb3x.sac_mask_hybrid_action.buffers import MaskableHybridActionReplayBuffer

OBSERVATION_SHAPE = (3,)
VALID_MASK = np.array([False, True, True, True, False], dtype=np.float32)


def _hybrid_action_space() -> spaces.Dict:
    return spaces.Dict(
        {
            "continuous": spaces.Box(
                low=np.array([-1.0], dtype=np.float32),
                high=np.array([1.0], dtype=np.float32),
                dtype=np.float32,
            ),
            "discrete": spaces.MultiDiscrete([3, 2]),
        }
    )


class MaskedHybridBanditEnv(gym.Env[np.ndarray, HybridAction]):
    """One-step hybrid env that rejects masked-out discrete actions."""

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}

    def __init__(self) -> None:
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=OBSERVATION_SHAPE,
            dtype=np.float32,
        )
        self.action_space = _hybrid_action_space()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        super().reset(seed=seed)
        del options
        return np.zeros(OBSERVATION_SHAPE, dtype=np.float32), {}

    def step(
        self,
        action: HybridAction,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        assert self.action_space.contains(action)
        discrete = action["discrete"].reshape(-1)
        assert VALID_MASK[discrete[0]]
        assert VALID_MASK[3 + discrete[1]]

        target_continuous = np.isclose(action["continuous"][0], 0.25, atol=0.25)
        target_discrete = np.array_equal(discrete, np.array([1, 0]))
        reward = 1.0 if target_continuous and target_discrete else 0.0
        obs = np.zeros(OBSERVATION_SHAPE, dtype=np.float32)
        return obs, reward, True, False, {}

    def action_masks(self) -> np.ndarray:
        return VALID_MASK.copy()


class UnmaskedHybridBanditEnv(gym.Env[np.ndarray, HybridAction]):
    """Same action space without the mask API."""

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}

    def __init__(self) -> None:
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=OBSERVATION_SHAPE,
            dtype=np.float32,
        )
        self.action_space = _hybrid_action_space()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        super().reset(seed=seed)
        del options
        return np.zeros(OBSERVATION_SHAPE, dtype=np.float32), {}

    def step(
        self,
        action: HybridAction,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        assert self.action_space.contains(action)
        obs = np.zeros(OBSERVATION_SHAPE, dtype=np.float32)
        return obs, 0.0, True, False, {}


def _make_model(env: gym.Env) -> MaskableHybridActionSAC:
    return MaskableHybridActionSAC(
        "MlpPolicy",
        env,
        seed=123,
        device="cpu",
        learning_starts=0,
        buffer_size=32,
        batch_size=2,
        train_freq=1,
        gradient_steps=1,
        ent_coef=0.01,
        policy_kwargs={"net_arch": [8]},
    )


def test_maskable_hybrid_action_sac_predict_masks_discrete_branch() -> None:
    model = _make_model(MaskedHybridBanditEnv())
    with th.no_grad():
        model.actor.discrete_logits.weight.zero_()
        model.actor.discrete_logits.bias.copy_(th.tensor([9.0, 2.0, 1.0, 2.0, 9.0]))

    obs = np.zeros(OBSERVATION_SHAPE, dtype=np.float32)
    masked_action, _ = model.predict(
        obs,
        deterministic=True,
        action_masks=VALID_MASK,
    )
    unmasked_action, _ = model.predict(obs, deterministic=True)

    assert not isinstance(masked_action, list)
    assert not isinstance(unmasked_action, list)
    np.testing.assert_array_equal(masked_action["discrete"], np.array([1, 0]))
    np.testing.assert_array_equal(unmasked_action["discrete"], np.array([0, 1]))


def test_maskable_hybrid_action_sac_learns_with_masked_warmup_actions() -> None:
    model = MaskableHybridActionSAC(
        "MlpPolicy",
        MaskedHybridBanditEnv(),
        seed=123,
        device="cpu",
        learning_starts=10,
        buffer_size=32,
        batch_size=2,
        train_freq=1,
        gradient_steps=0,
        policy_kwargs={"net_arch": [8]},
    )

    model.learn(total_timesteps=4)

    assert isinstance(model.replay_buffer, MaskableHybridActionReplayBuffer)
    sample = model.maskable_replay_buffer.sample(1)
    np.testing.assert_array_equal(sample.action_masks.cpu().numpy(), VALID_MASK[None])
    np.testing.assert_array_equal(
        sample.next_action_masks.cpu().numpy(),
        VALID_MASK[None],
    )


def test_maskable_hybrid_action_sac_requires_masks_by_default() -> None:
    model = _make_model(UnmaskedHybridBanditEnv())

    with pytest.raises(ValueError, match="does not support action masking"):
        model.learn(total_timesteps=1)


def test_maskable_hybrid_action_sac_can_train_without_masks() -> None:
    model = _make_model(UnmaskedHybridBanditEnv())

    model.learn(total_timesteps=2, use_masking=False)

    sample = model.maskable_replay_buffer.sample(1)
    assert np.all(sample.action_masks.cpu().numpy() == 1.0)
    assert np.all(sample.next_action_masks.cpu().numpy() == 1.0)


def test_maskable_hybrid_action_sac_rejects_empty_discrete_branch_mask() -> None:
    model = _make_model(MaskedHybridBanditEnv())
    obs = np.zeros(OBSERVATION_SHAPE, dtype=np.float32)

    with pytest.raises(ValueError, match="valid action"):
        model.predict(
            obs,
            deterministic=True,
            action_masks=np.array([False, False, False, True, True]),
        )
