"""Sanity checks for the first HybridActionSAC implementation."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import gymnasium as gym
import numpy as np
import pytest
import torch as th
from gymnasium import spaces

from sb3x import HybridActionSAC
from sb3x.common.hybrid_action import HybridAction, make_hybrid_action_spec
from sb3x.sac_hybrid_action.encoding import (
    encode_scaled_hybrid_actions_for_critic,
    scale_discrete_actions,
)

OBSERVATION_SHAPE = (3,)


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


class HybridBanditEnv(gym.Env[np.ndarray, HybridAction]):
    """One-step env that asserts it receives the public hybrid action shape."""

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
        continuous = action["continuous"]
        discrete = action["discrete"]
        assert continuous.shape == (1,)
        assert discrete.shape == (2,)
        assert self.action_space.contains(action)

        target_continuous = np.isclose(continuous[0], 0.25, atol=0.25)
        target_discrete = np.array_equal(discrete, np.array([1, 0]))
        reward = 1.0 if target_continuous and target_discrete else 0.0
        obs = np.zeros(OBSERVATION_SHAPE, dtype=np.float32)
        return obs, reward, True, False, {}


def test_hybrid_sac_discrete_encoding_uses_one_hot_categories() -> None:
    """The SAC critic should see categorical one-hot actions, not raw floats."""
    spec = make_hybrid_action_spec(_hybrid_action_space())
    discrete = th.tensor([[0, 0], [2, 1]], dtype=th.long)
    scaled_discrete = scale_discrete_actions(
        spec,
        discrete,
    )
    scaled_actions = th.cat(
        [
            th.zeros((2, spec.continuous_dim)),
            scaled_discrete,
        ],
        dim=1,
    )

    encoded = encode_scaled_hybrid_actions_for_critic(spec, scaled_actions)

    np.testing.assert_allclose(
        encoded.numpy(),
        np.array(
            [
                [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )


def test_hybrid_action_sac_learns_and_predicts_public_actions() -> None:
    """Tiny end-to-end check through SAC, the env wrapper, and predict()."""
    env = HybridBanditEnv()
    model = HybridActionSAC(
        "MlpPolicy",
        env,
        seed=123,
        device="cpu",
        learning_starts=0,
        buffer_size=32,
        batch_size=2,
        train_freq=1,
        gradient_steps=1,
        policy_kwargs={"net_arch": [8]},
    )

    model.learn(total_timesteps=4)
    obs, _ = env.reset()
    action, _ = model.predict(obs, deterministic=True)

    assert not isinstance(action, list)
    assert set(action.keys()) == {"continuous", "discrete"}
    assert action["continuous"].shape == (1,)
    assert action["discrete"].shape == (2,)
    assert env.action_space.contains(action)


def test_hybrid_action_sac_load_without_env_predicts_public_actions(
    tmp_path: Path,
) -> None:
    """Saved models should retain enough metadata for env-free prediction."""
    env = HybridBanditEnv()
    model = HybridActionSAC(
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
    save_path = tmp_path / "hybrid_action_sac"

    model.save(save_path)
    loaded_model = HybridActionSAC.load(save_path, device="cpu")
    obs, _ = env.reset()
    action, _ = loaded_model.predict(obs, deterministic=True)

    assert not isinstance(action, list)
    assert env.action_space.contains(action)


def test_hybrid_action_sac_load_with_env_accepts_public_hybrid_space(
    tmp_path: Path,
) -> None:
    """SB3 loading with a new env should wrap the raw hybrid action space."""
    model = HybridActionSAC(
        "MlpPolicy",
        HybridBanditEnv(),
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
    save_path = tmp_path / "hybrid_action_sac"
    env = HybridBanditEnv()

    model.save(save_path)
    loaded_model = HybridActionSAC.load(save_path, env=env, device="cpu")
    obs, _ = env.reset()
    action, _ = loaded_model.predict(obs, deterministic=True)

    assert not isinstance(action, list)
    assert env.action_space.contains(action)


def test_hybrid_action_sac_rejects_large_discrete_enumeration() -> None:
    """Exact discrete expectations should fail fast for huge branches."""
    with pytest.raises(ValueError, match="max_discrete_combinations"):
        HybridActionSAC(
            "MlpPolicy",
            HybridBanditEnv(),
            seed=123,
            device="cpu",
            policy_kwargs={
                "net_arch": [8],
                "max_discrete_combinations": 2,
            },
        )
