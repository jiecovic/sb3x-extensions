"""Sanity checks for MaskableHybridActionSAC."""

from __future__ import annotations

from typing import ClassVar

import gymnasium as gym
import numpy as np
import pytest
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
from stable_baselines3.common.vec_env import VecNormalize

from sb3x import MaskableHybridActionSAC
from sb3x.common.hybrid_action import HybridAction
from sb3x.sac_mask_hybrid_action.buffers import (
    MaskableHybridActionDictReplayBuffer,
    MaskableHybridActionReplayBuffer,
)

OBSERVATION_SHAPE = (3,)
VALID_MASK = np.array([False, True, True, True, False], dtype=np.float32)
DICT_IMAGE_SHAPE = (64, 64, 1)


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


class MaskedHybridDictBanditEnv(gym.Env[dict[str, np.ndarray], HybridAction]):
    """One-step dict-observation hybrid env with a discrete mask API."""

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}

    def __init__(self) -> None:
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0,
                    high=255,
                    shape=DICT_IMAGE_SHAPE,
                    dtype=np.uint8,
                ),
                "state": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(2,),
                    dtype=np.float32,
                ),
            }
        )
        self.action_space = _hybrid_action_space()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, object]]:
        super().reset(seed=seed)
        del options
        return {
            "image": np.zeros(DICT_IMAGE_SHAPE, dtype=np.uint8),
            "state": np.zeros((2,), dtype=np.float32),
        }, {}

    def step(
        self,
        action: HybridAction,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, object]]:
        assert self.action_space.contains(action)
        obs = {
            "image": np.ones(DICT_IMAGE_SHAPE, dtype=np.uint8),
            "state": np.array([0.0, 1.0], dtype=np.float32),
        }
        return obs, 0.0, True, False, {}

    def action_masks(self) -> np.ndarray:
        return VALID_MASK.copy()


class DelegatingMaskableReplayBuffer(ReplayBuffer):
    """Custom replay buffer that satisfies the maskable interface structurally."""

    def __init__(self, *args: object, mask_dims: int, **kwargs: object) -> None:
        self._delegate = MaskableHybridActionReplayBuffer(
            *args,
            mask_dims=mask_dims,
            **kwargs,
        )

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, object]],
        *,
        action_masks: np.ndarray | None = None,
        next_action_masks: np.ndarray | None = None,
    ) -> None:
        self._delegate.add(
            obs,
            next_obs,
            action,
            reward,
            done,
            infos,
            action_masks=action_masks,
            next_action_masks=next_action_masks,
        )

    def sample(
        self,
        batch_size: int,
        env: VecNormalize | None = None,
    ) -> object:
        return self._delegate.sample(batch_size, env=env)


class DelegatingMaskableDictReplayBuffer(DictReplayBuffer):
    """Dict-observation custom replay buffer with the maskable hybrid interface."""

    def __init__(self, *args: object, mask_dims: int, **kwargs: object) -> None:
        self._delegate = MaskableHybridActionDictReplayBuffer(
            *args,
            mask_dims=mask_dims,
            **kwargs,
        )

    def add(
        self,
        obs: dict[str, np.ndarray],
        next_obs: dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, object]],
        *,
        action_masks: np.ndarray | None = None,
        next_action_masks: np.ndarray | None = None,
    ) -> None:
        self._delegate.add(
            obs,
            next_obs,
            action,
            reward,
            done,
            infos,
            action_masks=action_masks,
            next_action_masks=next_action_masks,
        )

    def sample(
        self,
        batch_size: int,
        env: VecNormalize | None = None,
    ) -> object:
        return self._delegate.sample(batch_size, env=env)


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


def test_maskable_hybrid_action_sac_accepts_structural_custom_replay_buffer() -> None:
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
        replay_buffer_class=DelegatingMaskableReplayBuffer,
        policy_kwargs={"net_arch": [8]},
    )

    model.learn(total_timesteps=4)

    sample = model.maskable_replay_buffer.sample(1)
    np.testing.assert_array_equal(sample.action_masks.cpu().numpy(), VALID_MASK[None])
    np.testing.assert_array_equal(
        sample.next_action_masks.cpu().numpy(),
        VALID_MASK[None],
    )


def test_maskable_hybrid_action_sac_accepts_structural_custom_dict_replay_buffer() -> (
    None
):
    model = MaskableHybridActionSAC(
        "MultiInputPolicy",
        MaskedHybridDictBanditEnv(),
        seed=123,
        device="cpu",
        learning_starts=10,
        buffer_size=32,
        batch_size=2,
        train_freq=1,
        gradient_steps=0,
        replay_buffer_class=DelegatingMaskableDictReplayBuffer,
        policy_kwargs={"net_arch": [8]},
    )

    model.learn(total_timesteps=4)

    sample = model.maskable_replay_buffer.sample(1)
    assert isinstance(sample.observations, dict)
    assert set(sample.observations.keys()) == {"image", "state"}
    np.testing.assert_array_equal(sample.action_masks.cpu().numpy(), VALID_MASK[None])
    np.testing.assert_array_equal(
        sample.next_action_masks.cpu().numpy(),
        VALID_MASK[None],
    )
