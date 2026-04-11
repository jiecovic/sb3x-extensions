"""Sanity checks for HybridRecurrentPPO."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv

from sb3x import HybridRecurrentPPO
from sb3x.common.hybrid_action import HybridAction, HybridActionDistribution

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


class HybridMemoryBanditEnv(gym.Env[np.ndarray, HybridAction]):
    """Tiny hybrid-action env for recurrent PPO smoke tests."""

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=OBSERVATION_SHAPE,
            dtype=np.float32,
        )
        self.action_space = _hybrid_action_space()
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


class DiscreteBanditEnv(HybridMemoryBanditEnv):
    """Wrong action-space shape for validation checks."""

    def __init__(self) -> None:
        super().__init__()
        self.action_space = spaces.Discrete(2)


def _build_model(env: gym.Env[np.ndarray, HybridAction]) -> HybridRecurrentPPO:
    return HybridRecurrentPPO(
        "MlpLstmPolicy",
        env,
        seed=123,
        device="cpu",
        n_steps=4,
        batch_size=4,
        n_epochs=1,
        policy_kwargs={"net_arch": [8], "lstm_hidden_size": 8},
    )


def test_hybrid_recurrent_ppo_learns_and_predicts_public_actions() -> None:
    """Tiny end-to-end check through recurrent PPO and public hybrid actions."""
    env = HybridMemoryBanditEnv()
    model = _build_model(env)

    model.learn(total_timesteps=4)
    obs, _ = env.reset()
    action, state = model.predict(obs, deterministic=True)

    assert not isinstance(action, list)
    assert env.action_space.contains(action)
    assert state is not None
    assert state[0].shape == (1, 1, 8)
    assert state[1].shape == (1, 1, 8)


def test_hybrid_recurrent_ppo_predict_accepts_recurrent_state() -> None:
    """predict() should roundtrip recurrent state like sb3-contrib RecurrentPPO."""
    env = HybridMemoryBanditEnv()
    model = _build_model(env)
    obs, _ = env.reset()

    action, state = model.predict(obs, deterministic=True)
    next_action, next_state = model.predict(
        obs,
        state=state,
        episode_start=np.array([False]),
        deterministic=True,
    )

    assert not isinstance(action, list)
    assert not isinstance(next_action, list)
    assert env.action_space.contains(action)
    assert env.action_space.contains(next_action)
    assert next_state is not None
    assert next_state[0].shape == (1, 1, 8)


def test_hybrid_recurrent_ppo_supports_vectorized_envs() -> None:
    """Vectorized envs should return one public hybrid action per env."""
    env = DummyVecEnv([HybridMemoryBanditEnv, HybridMemoryBanditEnv])
    model = HybridRecurrentPPO(
        "MlpLstmPolicy",
        env,
        seed=123,
        device="cpu",
        n_steps=4,
        batch_size=4,
        n_epochs=1,
        policy_kwargs={"net_arch": [8], "lstm_hidden_size": 8},
    )

    model.learn(total_timesteps=8)
    obs = env.reset()
    assert isinstance(obs, np.ndarray)
    actions, state = model.predict(obs, deterministic=True)

    assert isinstance(actions, list)
    assert len(actions) == 2
    assert all(env.action_space.contains(action) for action in actions)
    assert state is not None
    assert state[0].shape == (1, 2, 8)
    assert state[1].shape == (1, 2, 8)


def test_hybrid_recurrent_ppo_uses_hybrid_distribution() -> None:
    """The recurrent policy should still use the hybrid action distribution."""
    model = _build_model(HybridMemoryBanditEnv())

    assert isinstance(model.policy.action_dist, HybridActionDistribution)


def test_hybrid_recurrent_ppo_load_without_env_predicts_public_actions(
    tmp_path: Path,
) -> None:
    """Saved recurrent hybrid models should retain hybrid action metadata."""
    env = HybridMemoryBanditEnv()
    model = _build_model(env)
    save_path = tmp_path / "hybrid_recurrent_ppo"

    model.save(save_path)
    loaded_model = HybridRecurrentPPO.load(save_path, device="cpu")
    obs, _ = env.reset()
    action, state = loaded_model.predict(obs, deterministic=True)

    assert not isinstance(action, list)
    assert env.action_space.contains(action)
    assert state is not None


def test_hybrid_recurrent_ppo_load_with_env_accepts_public_hybrid_space(
    tmp_path: Path,
) -> None:
    """SB3 loading with a new env should wrap the public hybrid action space."""
    model = _build_model(HybridMemoryBanditEnv())
    save_path = tmp_path / "hybrid_recurrent_ppo"
    env = HybridMemoryBanditEnv()

    model.save(save_path)
    loaded_model = HybridRecurrentPPO.load(save_path, env=env, device="cpu")
    obs, _ = env.reset()
    action, state = loaded_model.predict(obs, deterministic=True)

    assert not isinstance(action, list)
    assert env.action_space.contains(action)
    assert state is not None


def test_hybrid_recurrent_ppo_rejects_non_hybrid_action_space() -> None:
    """Construction should fail before SB3 sees an unsupported action space."""
    try:
        HybridRecurrentPPO(
            "MlpLstmPolicy",
            DiscreteBanditEnv(),
            seed=123,
            device="cpu",
            n_steps=4,
            batch_size=4,
            n_epochs=1,
        )
    except TypeError as exc:
        assert "spaces.Dict" in str(exc)
    else:
        raise AssertionError("Expected HybridRecurrentPPO to reject Discrete envs")
