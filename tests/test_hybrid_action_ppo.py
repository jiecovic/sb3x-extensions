"""Sanity checks for the first HybridActionPPO implementation."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch as th
from gymnasium import spaces

from sb3x import HybridActionPPO
from sb3x.common.hybrid_action import HybridAction, make_hybrid_action_spec
from sb3x.common.hybrid_action.wrappers import HybridActionEnvWrapper
from sb3x.ppo_hybrid_action.distributions import HybridActionDistribution

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

    metadata = {"render_modes": []}

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


class DiscreteBanditEnv(HybridBanditEnv):
    """Wrong action-space shape for fast validation checks."""

    def __init__(self) -> None:
        super().__init__()
        self.action_space = spaces.Discrete(2)


def test_hybrid_action_spec_roundtrip() -> None:
    """The flat SB3 action representation should roundtrip to the public dict."""
    spec = make_hybrid_action_spec(_hybrid_action_space())
    hybrid_action = {
        "continuous": np.array([0.5], dtype=np.float32),
        "discrete": np.array([2, 1], dtype=np.int64),
    }

    flat_action = spec.flatten_action(hybrid_action)
    restored_action = spec.unflatten_action(flat_action)

    assert spec.flat_action_space.shape == (3,)
    np.testing.assert_allclose(flat_action, np.array([0.5, 2.0, 1.0], dtype=np.float32))
    np.testing.assert_allclose(
        restored_action["continuous"],
        hybrid_action["continuous"],
    )
    np.testing.assert_array_equal(
        restored_action["discrete"],
        hybrid_action["discrete"],
    )


def test_hybrid_action_spec_rejects_nonzero_multidiscrete_start() -> None:
    """SB3's MultiCategorical branch expects zero-based discrete actions."""
    action_space = spaces.Dict(
        {
            "continuous": spaces.Box(
                low=np.array([-1.0], dtype=np.float32),
                high=np.array([1.0], dtype=np.float32),
                dtype=np.float32,
            ),
            "discrete": spaces.MultiDiscrete([3, 2], start=[1, 0]),
        }
    )

    with pytest.raises(ValueError, match="zero-based"):
        make_hybrid_action_spec(action_space)


def test_hybrid_action_wrapper_normalizes_public_dict_actions() -> None:
    """Manual dict actions through the wrapper should be validated consistently."""
    wrapped_env = HybridActionEnvWrapper(HybridBanditEnv())
    action = {
        "continuous": np.array([0.5], dtype=np.float64),
        "discrete": np.array([2.0, 1.0], dtype=np.float32),
    }

    normalized = wrapped_env.action(action)

    assert normalized["continuous"].dtype == np.float32
    assert normalized["discrete"].dtype == np.int64
    assert wrapped_env.env.action_space.contains(normalized)


def test_hybrid_action_wrapper_rejects_bad_public_dict_actions() -> None:
    """Manual dict actions with wrong shapes should fail before env.step()."""
    wrapped_env = HybridActionEnvWrapper(HybridBanditEnv())

    with pytest.raises(ValueError, match="Discrete action"):
        wrapped_env.action(
            {
                "continuous": np.array([0.5], dtype=np.float32),
                "discrete": np.array([1], dtype=np.int64),
            }
        )


def test_hybrid_distribution_combines_component_log_prob_and_entropy() -> None:
    """The hybrid distribution should sum branch log-probs and entropies."""
    spec = make_hybrid_action_spec(_hybrid_action_space())
    distribution = HybridActionDistribution(spec)
    action_params = th.zeros(
        (2, spec.continuous_dim + spec.discrete_logits_dim),
        dtype=th.float32,
    )
    distribution.proba_distribution(
        action_params=action_params,
        log_std=th.zeros(spec.continuous_dim, dtype=th.float32),
    )
    actions = th.tensor([[0.0, 1.0, 0.0], [0.2, 2.0, 1.0]], dtype=th.float32)

    continuous_actions, discrete_actions = actions[:, :1], actions[:, 1:].long()
    expected_log_prob = distribution.continuous_dist.log_prob(
        continuous_actions
    ) + distribution.discrete_dist.log_prob(discrete_actions)
    continuous_entropy = distribution.continuous_dist.entropy()
    discrete_entropy = distribution.discrete_dist.entropy()
    assert continuous_entropy is not None
    expected_entropy = continuous_entropy + discrete_entropy

    th.testing.assert_close(distribution.log_prob(actions), expected_log_prob)
    th.testing.assert_close(distribution.entropy(), expected_entropy)


def test_hybrid_action_ppo_learns_and_predicts_public_actions() -> None:
    """Tiny end-to-end check through PPO, the env wrapper, and predict()."""
    env = HybridBanditEnv()
    model = HybridActionPPO(
        "MlpPolicy",
        env,
        seed=123,
        device="cpu",
        n_steps=4,
        batch_size=4,
        n_epochs=1,
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


def test_hybrid_action_ppo_load_without_env_predicts_public_actions(
    tmp_path: Path,
) -> None:
    """Saved models should retain enough metadata for env-free prediction."""
    env = HybridBanditEnv()
    model = HybridActionPPO(
        "MlpPolicy",
        env,
        seed=123,
        device="cpu",
        n_steps=4,
        batch_size=4,
        n_epochs=1,
        policy_kwargs={"net_arch": [8]},
    )
    save_path = tmp_path / "hybrid_action_ppo"

    model.save(save_path)
    loaded_model = HybridActionPPO.load(save_path, device="cpu")
    obs, _ = env.reset()
    action, _ = loaded_model.predict(obs, deterministic=True)

    assert not isinstance(action, list)
    assert env.action_space.contains(action)


def test_hybrid_action_ppo_load_with_env_accepts_public_hybrid_space(
    tmp_path: Path,
) -> None:
    """SB3 loading with a new env should wrap the raw hybrid action space."""
    model = HybridActionPPO(
        "MlpPolicy",
        HybridBanditEnv(),
        seed=123,
        device="cpu",
        n_steps=4,
        batch_size=4,
        n_epochs=1,
        policy_kwargs={"net_arch": [8]},
    )
    save_path = tmp_path / "hybrid_action_ppo"
    env = HybridBanditEnv()

    model.save(save_path)
    loaded_model = HybridActionPPO.load(save_path, env=env, device="cpu")
    obs, _ = env.reset()
    action, _ = loaded_model.predict(obs, deterministic=True)

    assert not isinstance(action, list)
    assert env.action_space.contains(action)


def test_hybrid_action_ppo_rejects_non_hybrid_action_space() -> None:
    """Construction should fail before SB3 sees an unsupported action space."""
    with pytest.raises(TypeError, match="spaces.Dict"):
        HybridActionPPO(
            "MlpPolicy",
            DiscreteBanditEnv(),
            seed=123,
            device="cpu",
            n_steps=4,
            batch_size=4,
            n_epochs=1,
        )
