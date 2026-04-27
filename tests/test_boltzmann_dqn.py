"""Sanity checks for BoltzmannDQN."""

from __future__ import annotations

import importlib
from typing import ClassVar

import gymnasium as gym
import numpy as np
import pytest
import torch as th
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv

from sb3x import BoltzmannDQN

OBSERVATION_SHAPE = (3,)


class DiscreteBanditEnv(gym.Env[np.ndarray, int]):
    """One-step discrete env for DQN action-selection checks."""

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}

    def __init__(self) -> None:
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=OBSERVATION_SHAPE,
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)

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
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        reward = 1.0 if int(action) == 2 else 0.0
        obs = np.zeros(OBSERVATION_SHAPE, dtype=np.float32)
        return obs, reward, True, False, {}


def _make_model(env: gym.Env | DummyVecEnv | None = None) -> BoltzmannDQN:
    return BoltzmannDQN(
        "MlpPolicy",
        DiscreteBanditEnv() if env is None else env,
        seed=123,
        device="cpu",
        learning_starts=0,
        buffer_size=32,
        batch_size=2,
        train_freq=1,
        gradient_steps=1,
        temperature_initial=1.0,
        temperature_final=1.0,
        temperature_fraction=1.0,
        policy_kwargs={"net_arch": [8]},
    )


def _set_constant_q_values(model: BoltzmannDQN, q_values: list[float]) -> None:
    """Make the online Q-network emit fixed values for every observation."""
    for parameter in model.q_net.parameters():
        parameter.data.zero_()

    linear_layers = list(
        module
        for module in model.q_net.q_net.modules()
        if isinstance(module, th.nn.Linear)
    )
    output_layer = linear_layers[-1]
    with th.no_grad():
        output_layer.bias.copy_(th.as_tensor(q_values, dtype=output_layer.bias.dtype))


def test_boltzmann_dqn_probabilities_match_q_softmax() -> None:
    model = _make_model()
    model.temperature = 2.0
    _set_constant_q_values(model, [0.0, 1.0, 2.0])
    obs = np.zeros(OBSERVATION_SHAPE, dtype=np.float32)

    action_probs, vectorized_env = model._boltzmann_action_probabilities(obs)

    expected = th.softmax(th.tensor([[0.0, 1.0, 2.0]]) / 2.0, dim=1)
    assert not vectorized_env
    th.testing.assert_close(action_probs.cpu(), expected)


def test_boltzmann_dqn_deterministic_predict_uses_argmax() -> None:
    model = _make_model()
    _set_constant_q_values(model, [0.0, 1.0, 2.0])
    obs = np.zeros(OBSERVATION_SHAPE, dtype=np.float32)

    action, _ = model.predict(obs, deterministic=True)

    assert int(action) == 2


def test_boltzmann_dqn_nondeterministic_predict_samples_softmax(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    boltzmann_module = importlib.import_module("sb3x.dqn_boltzmann.dqn_boltzmann")
    captured: dict[str, th.Tensor] = {}

    def fake_multinomial(
        input: th.Tensor,
        num_samples: int,
    ) -> th.Tensor:
        captured["input"] = input.detach().clone()
        assert num_samples == 1
        return th.tensor([[1]], device=input.device)

    model = _make_model()
    _set_constant_q_values(model, [0.0, 1.0, 2.0])
    obs = np.zeros(OBSERVATION_SHAPE, dtype=np.float32)
    monkeypatch.setattr(boltzmann_module.th, "multinomial", fake_multinomial)

    action, _ = model.predict(obs, deterministic=False)

    expected = th.softmax(th.tensor([[0.0, 1.0, 2.0]]), dim=1)
    assert int(action) == 1
    th.testing.assert_close(captured["input"].cpu(), expected)


def test_boltzmann_dqn_vectorized_predict_returns_action_batch() -> None:
    env = DummyVecEnv([DiscreteBanditEnv, DiscreteBanditEnv])
    model = _make_model(env)
    _set_constant_q_values(model, [0.0, 1.0, 2.0])
    obs = np.zeros((2, *OBSERVATION_SHAPE), dtype=np.float32)

    action, _ = model.predict(obs, deterministic=True)

    np.testing.assert_array_equal(action, np.array([2, 2]))


def test_boltzmann_dqn_learn_runs() -> None:
    model = _make_model()

    model.learn(total_timesteps=4)

    assert model.num_timesteps >= 4


def test_boltzmann_dqn_rejects_non_positive_temperatures() -> None:
    with pytest.raises(ValueError, match="temperature_initial"):
        BoltzmannDQN("MlpPolicy", DiscreteBanditEnv(), temperature_initial=0.0)
