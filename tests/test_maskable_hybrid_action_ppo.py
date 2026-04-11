"""Sanity checks for MaskableHybridActionPPO."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch as th
from gymnasium import spaces

from sb3x import HybridActionPPO, MaskableHybridActionPPO
from sb3x.common.hybrid_action import HybridAction, HybridActionNet
from sb3x.common.maskable import (
    MaskableMultiCategoricalDistribution,
    make_masked_proba_distribution,
    mask_dims_for_action_space,
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


class MaskedHybridBanditEnv(gym.Env[np.ndarray, HybridAction]):
    """One-step hybrid env exposing a discrete-branch action mask."""

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
        target_discrete = np.array_equal(action["discrete"], np.array([1, 0]))
        reward = 1.0 if target_discrete else 0.0
        return np.zeros(OBSERVATION_SHAPE, dtype=np.float32), reward, True, False, {}

    def action_masks(self) -> np.ndarray:
        return self._action_mask.copy()


class MaskedMultidimHybridBanditEnv(MaskedHybridBanditEnv):
    """Hybrid env with a multidimensional MultiDiscrete branch."""

    def __init__(self) -> None:
        super().__init__(np.ones(14, dtype=bool))
        self.action_space = spaces.Dict(
            {
                "continuous": spaces.Box(
                    low=np.array([-1.0], dtype=np.float32),
                    high=np.array([1.0], dtype=np.float32),
                    dtype=np.float32,
                ),
                "discrete": spaces.MultiDiscrete(np.array([[2, 3], [4, 5]])),
            }
        )


class UnmaskedHybridBanditEnv(MaskedHybridBanditEnv):
    """Same toy env without the action-mask API."""

    def __getattribute__(self, name: str) -> object:
        if name == "action_masks":
            raise AttributeError(name)
        return super().__getattribute__(name)


def _build_maskable_model(
    env: gym.Env[np.ndarray, HybridAction],
) -> MaskableHybridActionPPO:
    return MaskableHybridActionPPO(
        "MlpPolicy",
        env,
        seed=123,
        device="cpu",
        n_steps=4,
        batch_size=4,
        n_epochs=1,
        policy_kwargs={"net_arch": [8]},
    )


def _force_invalid_discrete_preferences(model: MaskableHybridActionPPO) -> None:
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


def test_predict_masks_only_the_discrete_branch() -> None:
    """Masks should change categorical choices without touching continuous output."""
    env = MaskedHybridBanditEnv()
    model = _build_maskable_model(env)
    _force_invalid_discrete_preferences(model)
    obs, _ = env.reset()

    unmasked_action, _ = model.predict(
        obs,
        deterministic=True,
        action_masks=ALL_VALID_ACTION_MASK,
    )
    masked_action, _ = model.predict(
        obs,
        deterministic=True,
        action_masks=env.action_masks(),
    )

    assert not isinstance(unmasked_action, list)
    assert not isinstance(masked_action, list)
    np.testing.assert_allclose(masked_action["continuous"], np.array([0.25]))
    np.testing.assert_array_equal(unmasked_action["discrete"], np.array([0, 1]))
    np.testing.assert_array_equal(masked_action["discrete"], np.array([1, 0]))


def test_all_valid_mask_matches_hybrid_action_ppo_prediction() -> None:
    """All-valid masks should preserve the base hybrid PPO prediction path."""
    env = MaskedHybridBanditEnv(ALL_VALID_ACTION_MASK)
    base_model = HybridActionPPO(
        "MlpPolicy",
        MaskedHybridBanditEnv(ALL_VALID_ACTION_MASK),
        seed=123,
        device="cpu",
        n_steps=4,
        batch_size=4,
        n_epochs=1,
        policy_kwargs={"net_arch": [8]},
    )
    maskable_model = _build_maskable_model(env)
    maskable_model.policy.load_state_dict(base_model.policy.state_dict())
    obs, _ = env.reset()

    base_action, _ = base_model.predict(obs, deterministic=True)
    masked_action, _ = maskable_model.predict(
        obs,
        deterministic=True,
        action_masks=ALL_VALID_ACTION_MASK,
    )

    assert not isinstance(base_action, list)
    assert not isinstance(masked_action, list)
    np.testing.assert_allclose(masked_action["continuous"], base_action["continuous"])
    np.testing.assert_array_equal(masked_action["discrete"], base_action["discrete"])


def test_learn_defaults_to_masking_and_stores_discrete_branch_masks() -> None:
    """Default learn() should collect env masks for PPO training."""
    model = _build_maskable_model(MaskedHybridBanditEnv())

    model.learn(total_timesteps=4)

    action_masks = model.maskable_rollout_buffer.action_masks.reshape(-1, 5)
    np.testing.assert_array_equal(action_masks[0], ACTION_MASK.astype(np.float32))


def test_learn_requires_env_support_when_masking_is_left_on() -> None:
    """Default learn() should fail fast when the env does not expose masks."""
    model = _build_maskable_model(UnmaskedHybridBanditEnv())

    with pytest.raises(ValueError, match="does not support action masking"):
        model.learn(total_timesteps=4)


def test_learn_can_run_unmasked_when_requested() -> None:
    """use_masking=False should keep the base hybrid path available."""
    model = _build_maskable_model(UnmaskedHybridBanditEnv())

    model.learn(total_timesteps=4, use_masking=False)

    action_masks = model.maskable_rollout_buffer.action_masks.reshape(-1, 5)
    np.testing.assert_array_equal(
        action_masks[0],
        ALL_VALID_ACTION_MASK.astype(np.float32),
    )


def test_multidimensional_multidiscrete_mask_dims_are_flattened() -> None:
    """Mask width should follow flattened MultiDiscrete logits."""
    discrete_space = spaces.MultiDiscrete(np.array([[2, 3], [4, 5]]))
    env = MaskedMultidimHybridBanditEnv()

    distribution = make_masked_proba_distribution(discrete_space)

    assert isinstance(distribution, MaskableMultiCategoricalDistribution)
    assert mask_dims_for_action_space(discrete_space) == 14
    assert distribution.action_dims == [2, 3, 4, 5]
    model = _build_maskable_model(env)
    assert model.maskable_rollout_buffer.mask_dims == 14


def test_maskable_hybrid_action_ppo_load_without_env_predicts_public_actions(
    tmp_path: Path,
) -> None:
    """Saved maskable hybrid models should retain hybrid action metadata."""
    env = MaskedHybridBanditEnv(ALL_VALID_ACTION_MASK)
    model = _build_maskable_model(env)
    save_path = tmp_path / "maskable_hybrid_action_ppo"

    model.save(save_path)
    loaded_model = MaskableHybridActionPPO.load(save_path, device="cpu")
    obs, _ = env.reset()
    action, _ = loaded_model.predict(
        obs,
        deterministic=True,
        action_masks=env.action_masks(),
    )

    assert not isinstance(action, list)
    assert env.action_space.contains(action)


def test_maskable_hybrid_action_ppo_load_with_env_accepts_public_hybrid_space(
    tmp_path: Path,
) -> None:
    """SB3 loading with a new env should wrap the public hybrid action space."""
    model = _build_maskable_model(MaskedHybridBanditEnv(ALL_VALID_ACTION_MASK))
    save_path = tmp_path / "maskable_hybrid_action_ppo"
    env = MaskedHybridBanditEnv(ALL_VALID_ACTION_MASK)

    model.save(save_path)
    loaded_model = MaskableHybridActionPPO.load(save_path, env=env, device="cpu")
    obs, _ = env.reset()
    action, _ = loaded_model.predict(
        obs,
        deterministic=True,
        action_masks=env.action_masks(),
    )

    assert not isinstance(action, list)
    assert env.action_space.contains(action)
