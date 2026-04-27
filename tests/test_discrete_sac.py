"""Sanity checks for DiscreteSAC."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import gymnasium as gym
import numpy as np
import pytest
import torch as th
from gymnasium import spaces

from sb3x import DiscreteSAC

OBSERVATION_SHAPE = (3,)


class DiscreteBanditEnv(gym.Env[np.ndarray, int]):
    """One-step discrete env for SAC smoke tests."""

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


class ContinuousActionEnv(gym.Env[np.ndarray, np.ndarray]):
    """One-step env with an unsupported continuous action space."""

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}

    def __init__(self) -> None:
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=OBSERVATION_SHAPE,
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

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
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        assert self.action_space.contains(action)
        obs = np.zeros(OBSERVATION_SHAPE, dtype=np.float32)
        return obs, 0.0, True, False, {}


def _make_model(ent_coef: str | float = 0.1) -> DiscreteSAC:
    return DiscreteSAC(
        "MlpPolicy",
        DiscreteBanditEnv(),
        seed=123,
        device="cpu",
        learning_starts=0,
        buffer_size=32,
        batch_size=2,
        train_freq=1,
        gradient_steps=1,
        ent_coef=ent_coef,
        policy_kwargs={"net_arch": [8]},
    )


def _set_actor_logits(model: DiscreteSAC, logits: list[float]) -> None:
    for parameter in model.actor.parameters():
        parameter.data.zero_()

    linear_layers = list(
        module
        for module in model.actor.logits_net.modules()
        if isinstance(module, th.nn.Linear)
    )
    output_layer = linear_layers[-1]
    with th.no_grad():
        output_layer.bias.copy_(th.as_tensor(logits, dtype=output_layer.bias.dtype))


def _set_critic_q_values(model: DiscreteSAC, q_values: list[float]) -> None:
    for q_network in model.critic.q_networks:
        for parameter in q_network.parameters():
            parameter.data.zero_()
        linear_layers = list(
            module for module in q_network.modules() if isinstance(module, th.nn.Linear)
        )
        output_layer = linear_layers[-1]
        with th.no_grad():
            output_layer.bias.copy_(
                th.as_tensor(q_values, dtype=output_layer.bias.dtype)
            )


def test_discrete_sac_actor_probabilities_match_softmax() -> None:
    model = _make_model()
    _set_actor_logits(model, [0.0, 1.0, 2.0])
    obs = np.zeros(OBSERVATION_SHAPE, dtype=np.float32)
    obs_tensor, vectorized_env = model.policy.obs_to_tensor(obs)

    action_probs, log_probs = model.actor.action_probabilities(obs_tensor)

    expected_probs = th.softmax(th.tensor([[0.0, 1.0, 2.0]]), dim=1)
    assert not vectorized_env
    th.testing.assert_close(action_probs.cpu(), expected_probs)
    th.testing.assert_close(log_probs.exp().cpu(), expected_probs)


def test_discrete_sac_deterministic_predict_uses_actor_argmax() -> None:
    model = _make_model()
    _set_actor_logits(model, [0.0, 1.0, 2.0])
    obs = np.zeros(OBSERVATION_SHAPE, dtype=np.float32)

    action, _ = model.predict(obs, deterministic=True)

    assert int(action) == 2


def test_discrete_sac_expected_soft_value_uses_exact_action_sum() -> None:
    model = _make_model(ent_coef=0.5)
    _set_actor_logits(model, [0.0, 0.0, 0.0])
    _set_critic_q_values(model, [1.0, 2.0, 3.0])
    obs = np.zeros(OBSERVATION_SHAPE, dtype=np.float32)
    obs_tensor, _ = model.policy.obs_to_tensor(obs)

    policy_eval = model._evaluate_policy(obs_tensor)
    min_q = model._min_q_values(model.critic, obs_tensor)
    soft_value = model._expected_soft_value(
        policy_eval,
        min_q,
        th.tensor(0.5),
    )

    expected_probs = th.full((1, 3), 1.0 / 3.0)
    expected_log_probs = expected_probs.log()
    expected = (expected_probs * (min_q - 0.5 * expected_log_probs)).sum(dim=1)
    th.testing.assert_close(soft_value.cpu(), expected.cpu())


def test_discrete_sac_auto_target_entropy_uses_action_count() -> None:
    model = _make_model(ent_coef="auto")

    assert model.target_entropy == pytest.approx(-float(np.log(3)))


def test_discrete_sac_learn_runs() -> None:
    model = _make_model()

    model.learn(total_timesteps=4)

    assert model.num_timesteps >= 4


def test_discrete_sac_load_without_env_predicts(tmp_path: Path) -> None:
    model = _make_model()
    _set_actor_logits(model, [0.0, 1.0, 2.0])
    save_path = tmp_path / "discrete_sac"

    model.save(save_path)
    loaded_model = DiscreteSAC.load(save_path, device="cpu")
    obs = np.zeros(OBSERVATION_SHAPE, dtype=np.float32)
    action, _ = loaded_model.predict(obs, deterministic=True)

    assert int(action) == 2


def test_discrete_sac_rejects_continuous_action_space() -> None:
    with pytest.raises(AssertionError, match="Discrete"):
        DiscreteSAC("MlpPolicy", ContinuousActionEnv())
