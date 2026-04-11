"""Sanity checks for MaskableHybridRecurrentPPO."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch as th
from gymnasium import spaces
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.vec_env import DummyVecEnv

from sb3x import HybridRecurrentPPO, MaskableHybridRecurrentPPO
from sb3x.common.hybrid_action import (
    HybridAction,
    HybridActionNet,
    MaskableHybridActionDistribution,
)

OBSERVATION_SHAPE = (3,)
ACTION_MASK = np.array([False, True, True, True, False], dtype=bool)
ALL_VALID_ACTION_MASK = np.ones(5, dtype=bool)


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


class MaskedHybridMemoryEnv(gym.Env[np.ndarray, HybridAction]):
    """Tiny recurrent hybrid-action env exposing a discrete-branch mask."""

    metadata = {"render_modes": []}

    def __init__(self, action_mask: np.ndarray = ACTION_MASK) -> None:
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=OBSERVATION_SHAPE,
            dtype=np.float32,
        )
        self.action_space = _hybrid_action_space()
        self._action_mask = action_mask
        self._step = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        super().reset(seed=seed)
        del options
        self._step = 0
        return np.zeros(OBSERVATION_SHAPE, dtype=np.float32), {}

    def step(
        self,
        action: HybridAction,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        assert self.action_space.contains(action)
        self._step += 1

        target_continuous = np.isclose(action["continuous"][0], 0.25, atol=0.5)
        target_discrete = np.array_equal(action["discrete"], np.array([1, 0]))
        reward = 1.0 if target_continuous and target_discrete else 0.0
        obs = np.full(OBSERVATION_SHAPE, self._step / 4, dtype=np.float32)
        return obs, reward, self._step >= 3, False, {}

    def action_masks(self) -> np.ndarray:
        return self._action_mask.copy()


class UnmaskedHybridMemoryEnv(MaskedHybridMemoryEnv):
    """Same toy env without the action-mask API."""

    def __getattribute__(self, name: str) -> object:
        if name == "action_masks":
            raise AttributeError(name)
        return super().__getattribute__(name)


def _build_maskable_model(env: GymEnv) -> MaskableHybridRecurrentPPO:
    return MaskableHybridRecurrentPPO(
        "MlpLstmPolicy",
        env,
        seed=123,
        device="cpu",
        n_steps=4,
        batch_size=4,
        n_epochs=1,
        policy_kwargs={"net_arch": [8], "lstm_hidden_size": 8},
    )


def _force_invalid_discrete_preferences(model: MaskableHybridRecurrentPPO) -> None:
    action_net = model.policy.action_net
    if not isinstance(action_net, HybridActionNet):
        raise TypeError("Expected HybridActionNet")

    with th.no_grad():
        action_net.continuous_net.weight.fill_(0.0)
        action_net.discrete_net.weight.fill_(0.0)
        if action_net.continuous_net.bias is None:
            raise TypeError("Expected continuous action bias")
        if action_net.discrete_net.bias is None:
            raise TypeError("Expected discrete action bias")

        action_net.continuous_net.bias.copy_(th.tensor([0.25]))
        # First branch prefers invalid action 0; second prefers invalid action 1.
        action_net.discrete_net.bias.copy_(th.tensor([3.0, 2.0, 1.0, 0.0, 4.0]))


def test_predict_masks_only_the_discrete_branch_with_recurrent_state() -> None:
    """Masks should constrain categorical choices without touching LSTM state."""
    env = MaskedHybridMemoryEnv()
    model = _build_maskable_model(env)
    _force_invalid_discrete_preferences(model)
    obs, _ = env.reset()

    unmasked_action, unmasked_state = model.predict(
        obs,
        deterministic=True,
        action_masks=ALL_VALID_ACTION_MASK,
    )
    masked_action, masked_state = model.predict(
        obs,
        deterministic=True,
        action_masks=env.action_masks(),
    )

    assert not isinstance(unmasked_action, list)
    assert not isinstance(masked_action, list)
    np.testing.assert_allclose(masked_action["continuous"], np.array([0.25]))
    np.testing.assert_array_equal(unmasked_action["discrete"], np.array([0, 1]))
    np.testing.assert_array_equal(masked_action["discrete"], np.array([1, 0]))
    assert masked_state is not None
    assert unmasked_state is not None
    assert masked_state[0].shape == unmasked_state[0].shape == (1, 1, 8)


def test_all_valid_mask_matches_hybrid_recurrent_prediction() -> None:
    """All-valid masks should preserve the base hybrid recurrent path."""
    base_model = HybridRecurrentPPO(
        "MlpLstmPolicy",
        MaskedHybridMemoryEnv(ALL_VALID_ACTION_MASK),
        seed=123,
        device="cpu",
        n_steps=4,
        batch_size=4,
        n_epochs=1,
        policy_kwargs={"net_arch": [8], "lstm_hidden_size": 8},
    )
    maskable_model = _build_maskable_model(MaskedHybridMemoryEnv(ALL_VALID_ACTION_MASK))
    maskable_model.policy.load_state_dict(base_model.policy.state_dict())
    obs, _ = MaskedHybridMemoryEnv(ALL_VALID_ACTION_MASK).reset()

    base_action, base_state = base_model.predict(obs, deterministic=True)
    masked_action, masked_state = maskable_model.predict(
        obs,
        deterministic=True,
        action_masks=ALL_VALID_ACTION_MASK,
    )

    assert not isinstance(base_action, list)
    assert not isinstance(masked_action, list)
    np.testing.assert_allclose(masked_action["continuous"], base_action["continuous"])
    np.testing.assert_array_equal(masked_action["discrete"], base_action["discrete"])
    assert base_state is not None
    assert masked_state is not None
    np.testing.assert_allclose(masked_state[0], base_state[0])
    np.testing.assert_allclose(masked_state[1], base_state[1])


def test_learn_defaults_to_masking_and_stores_discrete_branch_masks() -> None:
    """Default learn() should collect env masks for recurrent PPO training."""
    model = _build_maskable_model(MaskedHybridMemoryEnv())

    model.learn(total_timesteps=4)

    action_masks = model.maskable_rollout_buffer.action_masks.reshape(-1, 5)
    np.testing.assert_array_equal(action_masks[0], ACTION_MASK.astype(np.float32))


def test_learn_requires_env_support_when_masking_is_left_on() -> None:
    """Default learn() should fail fast when the env does not expose masks."""
    model = _build_maskable_model(UnmaskedHybridMemoryEnv())

    with pytest.raises(ValueError, match="does not support action masking"):
        model.learn(total_timesteps=4)


def test_learn_can_run_unmasked_when_requested() -> None:
    """use_masking=False should keep the base recurrent hybrid path available."""
    model = _build_maskable_model(UnmaskedHybridMemoryEnv())

    model.learn(total_timesteps=4, use_masking=False)

    action_masks = model.maskable_rollout_buffer.action_masks.reshape(-1, 5)
    np.testing.assert_array_equal(
        action_masks[0],
        ALL_VALID_ACTION_MASK.astype(np.float32),
    )


def test_vectorized_predict_uses_one_mask_per_env() -> None:
    """Vectorized recurrent prediction should apply masks per environment."""
    env = DummyVecEnv([MaskedHybridMemoryEnv, MaskedHybridMemoryEnv])
    model = _build_maskable_model(env)
    _force_invalid_discrete_preferences(model)
    obs = env.reset()
    assert isinstance(obs, np.ndarray)

    actions, state = model.predict(
        obs,
        deterministic=True,
        action_masks=np.stack([ACTION_MASK, ACTION_MASK]),
    )

    assert isinstance(actions, list)
    assert len(actions) == 2
    assert all(env.action_space.contains(action) for action in actions)
    for action in actions:
        np.testing.assert_array_equal(action["discrete"], np.array([1, 0]))
    assert state is not None
    assert state[0].shape == (1, 2, 8)


def test_maskable_hybrid_recurrent_ppo_uses_maskable_distribution() -> None:
    """The recurrent policy should use the maskable hybrid distribution."""
    model = _build_maskable_model(MaskedHybridMemoryEnv())

    assert isinstance(model.policy.action_dist, MaskableHybridActionDistribution)


def test_maskable_hybrid_recurrent_ppo_load_without_env_predicts_public_actions(
    tmp_path: Path,
) -> None:
    """Saved recurrent hybrid models should retain hybrid action metadata."""
    env = MaskedHybridMemoryEnv(ALL_VALID_ACTION_MASK)
    model = _build_maskable_model(env)
    save_path = tmp_path / "maskable_hybrid_recurrent_ppo"

    model.save(save_path)
    loaded_model = MaskableHybridRecurrentPPO.load(save_path, device="cpu")
    obs, _ = env.reset()
    action, state = loaded_model.predict(
        obs,
        deterministic=True,
        action_masks=env.action_masks(),
    )

    assert not isinstance(action, list)
    assert env.action_space.contains(action)
    assert state is not None


def test_maskable_hybrid_recurrent_ppo_load_with_env_accepts_public_hybrid_space(
    tmp_path: Path,
) -> None:
    """SB3 loading with a new env should wrap the public hybrid action space."""
    model = _build_maskable_model(MaskedHybridMemoryEnv(ALL_VALID_ACTION_MASK))
    save_path = tmp_path / "maskable_hybrid_recurrent_ppo"
    env = MaskedHybridMemoryEnv(ALL_VALID_ACTION_MASK)

    model.save(save_path)
    loaded_model = MaskableHybridRecurrentPPO.load(save_path, env=env, device="cpu")
    obs, _ = env.reset()
    action, state = loaded_model.predict(
        obs,
        deterministic=True,
        action_masks=env.action_masks(),
    )

    assert not isinstance(action, list)
    assert env.action_space.contains(action)
    assert state is not None
