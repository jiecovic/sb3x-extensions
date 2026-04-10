"""MiniGrid validation helpers for recurrent PPO parity checks."""

from __future__ import annotations

import random
import warnings
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from importlib import import_module
from typing import Literal

import gymnasium as gym
import numpy as np
import torch as th
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    VecMonitor,
    VecTransposeImage,
)

from sb3x import MaskableRecurrentPPO

from .models import MinigridFeaturesExtractor

MINIGRID_MEMORY_ENV_ID = "MiniGrid-MemoryS7-v0"
RecurrentAlgorithm = RecurrentPPO | MaskableRecurrentPPO
RecurrentAlgorithmClass = type[RecurrentPPO] | type[MaskableRecurrentPPO]
RecurrentState = tuple[np.ndarray, np.ndarray] | None
ObservationMode = Literal["flat", "image"]


@dataclass(frozen=True)
class PredictStep:
    """Single deterministic recurrent-policy step for parity comparisons."""

    step: int
    action: np.ndarray
    hidden_state: np.ndarray
    cell_state: np.ndarray
    reward: float
    done: bool


@dataclass(frozen=True)
class EvaluationSummary:
    """Compact deterministic evaluation summary."""

    episode_returns: list[float]
    episode_lengths: list[int]
    mean_return: float
    mean_length: float
    positive_return_rate: float


class IgnoreResetSeedWrapper(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """Drop explicit reset seeds when an inner reseed wrapper already owns them."""

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        del seed
        return self.env.reset(options=options)


def set_global_seeds(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible local checks."""
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)
        th.backends.cudnn.benchmark = False
        th.backends.cudnn.deterministic = True

    th.use_deterministic_algorithms(True, warn_only=True)
    th.set_num_threads(1)


def recurrent_policy_kwargs() -> dict[str, object]:
    """Return the small flat-observation architecture used for parity checks."""
    return {
        "lstm_hidden_size": 32,
        "n_lstm_layers": 1,
        "net_arch": [32],
    }


def benchmark_policy_kwargs(*, cnn_features_dim: int = 128) -> dict[str, object]:
    """Return the image-based recurrent policy setup used for benchmark runs."""
    return {
        "features_extractor_class": MinigridFeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": cnn_features_dim},
        "lstm_hidden_size": 128,
        "n_lstm_layers": 1,
        "net_arch": [64],
    }


def make_minigrid_memory_env(
    seed: int,
    *,
    episode_seed_count: int = 256,
    deterministic_resets: bool = True,
    observation_mode: ObservationMode = "flat",
    render_mode: str | None = None,
) -> gym.Env[np.ndarray, int]:
    """Create one deterministic MiniGrid Memory env for parity or benchmark runs."""
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="pkg_resources is deprecated as an API",
                category=UserWarning,
            )
            wrappers_module = import_module("minigrid.wrappers")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "MiniGrid is not installed. Install development dependencies with "
            '`pip install -e ".[dev]"` or use `uv run --extra dev ...`.'
        ) from exc

    flat_obs_wrapper = getattr(wrappers_module, "FlatObsWrapper")
    image_obs_wrapper = getattr(wrappers_module, "ImgObsWrapper")
    reseed_wrapper = getattr(wrappers_module, "ReseedWrapper")

    env = gym.make(MINIGRID_MEMORY_ENV_ID, render_mode=render_mode)
    if deterministic_resets:
        env = reseed_wrapper(
            env,
            seeds=[seed + offset for offset in range(episode_seed_count)],
        )
        env = IgnoreResetSeedWrapper(env)
    if observation_mode == "flat":
        env = flat_obs_wrapper(env)
    else:
        env = image_obs_wrapper(env)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def make_minigrid_memory_vec_env(
    seed: int,
    *,
    n_envs: int = 1,
    episode_seed_count: int = 256,
    deterministic_resets: bool = True,
    observation_mode: ObservationMode = "flat",
) -> VecEnv:
    """Create a vectorized deterministic MiniGrid Memory env."""
    env_fns: list[Callable[[], gym.Env[np.ndarray, int]]] = []
    for env_index in range(n_envs):
        env_seed = seed + env_index * 10_000

        def make_env(current_seed: int = env_seed) -> gym.Env[np.ndarray, int]:
            return make_minigrid_memory_env(
                current_seed,
                episode_seed_count=episode_seed_count,
                deterministic_resets=deterministic_resets,
                observation_mode=observation_mode,
            )

        env_fns.append(make_env)

    vec_env: VecEnv = VecMonitor(DummyVecEnv(env_fns))
    if observation_mode == "image":
        vec_env = VecTransposeImage(vec_env)
    return vec_env


def build_recurrent_model(
    algorithm_cls: RecurrentAlgorithmClass,
    env: gym.Env[np.ndarray, int] | VecEnv,
    seed: int,
    *,
    policy: str = "MlpLstmPolicy",
    n_steps: int = 16,
    batch_size: int = 16,
    n_epochs: int = 1,
    policy_kwargs: dict[str, object] | None = None,
    tensorboard_log: str | None = None,
    verbose: int = 0,
) -> RecurrentAlgorithm:
    """Build one recurrent PPO model for MiniGrid parity or benchmark runs."""
    return algorithm_cls(
        policy,
        env,
        seed=seed,
        device="cpu",
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        policy_kwargs=policy_kwargs or recurrent_policy_kwargs(),
        tensorboard_log=tensorboard_log,
        verbose=verbose,
    )


def build_benchmark_recurrent_model(
    algorithm_cls: RecurrentAlgorithmClass,
    env: gym.Env[np.ndarray, int] | VecEnv,
    seed: int,
    *,
    n_steps: int = 128,
    batch_size: int = 128,
    n_epochs: int = 4,
    policy_kwargs: dict[str, object] | None = None,
    tensorboard_log: str | None = None,
    verbose: int = 0,
) -> RecurrentAlgorithm:
    """Build the image-based recurrent PPO benchmark model."""
    return build_recurrent_model(
        algorithm_cls,
        env,
        seed,
        policy="CnnLstmPolicy",
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        policy_kwargs=policy_kwargs or benchmark_policy_kwargs(),
        tensorboard_log=tensorboard_log,
        verbose=verbose,
    )


def copy_policy_state(
    source: RecurrentAlgorithm,
    target: RecurrentAlgorithm,
) -> None:
    """Copy policy weights from one recurrent algorithm to another."""
    target.policy.load_state_dict(deepcopy(source.policy.state_dict()))


def _require_recurrent_state(
    state: tuple[np.ndarray, ...] | None,
) -> RecurrentState:
    """Narrow the recurrent predict() state to a hidden/cell pair."""
    if state is None:
        return None
    if len(state) != 2:
        raise TypeError(
            f"Expected a hidden/cell state pair, received {len(state)} arrays"
        )
    return (state[0], state[1])


def max_policy_parameter_diff(
    left: RecurrentAlgorithm,
    right: RecurrentAlgorithm,
) -> float:
    """Return the maximum absolute parameter difference between two policies."""
    left_items = left.policy.state_dict()
    right_items = right.policy.state_dict()
    diffs = []
    for key, left_value in left_items.items():
        right_value = right_items[key]
        diffs.append(th.max(th.abs(left_value - right_value)).item())

    return max(diffs, default=0.0)


def collect_deterministic_rollout(
    model: RecurrentAlgorithm,
    *,
    seed: int,
    steps: int,
) -> list[PredictStep]:
    """Collect a deterministic recurrent-policy rollout on MiniGrid Memory."""
    env = make_minigrid_memory_env(seed)
    try:
        obs, _ = env.reset()
        state: RecurrentState = None
        episode_start = np.array([True], dtype=bool)
        rollout: list[PredictStep] = []

        for step_index in range(steps):
            action, next_state = model.predict(
                obs,
                state=state,
                episode_start=episode_start,
                deterministic=True,
            )
            state = _require_recurrent_state(next_state)
            assert state is not None

            next_obs, reward, terminated, truncated, _ = env.step(
                int(np.asarray(action).item())
            )
            done = terminated or truncated
            hidden_state, cell_state = state
            rollout.append(
                PredictStep(
                    step=step_index,
                    action=np.asarray(action).copy(),
                    hidden_state=hidden_state.copy(),
                    cell_state=cell_state.copy(),
                    reward=float(reward),
                    done=done,
                )
            )

            if done:
                obs, _ = env.reset()
            else:
                obs = next_obs
            episode_start = np.array([done], dtype=bool)

        return rollout
    finally:
        env.close()


def evaluate_deterministic_policy(
    model: RecurrentAlgorithm,
    *,
    seed: int,
    episodes: int,
) -> EvaluationSummary:
    """Evaluate a recurrent policy deterministically on MiniGrid Memory."""
    env = make_minigrid_memory_env(seed, episode_seed_count=max(episodes * 4, 32))
    try:
        obs, _ = env.reset()
        state: RecurrentState = None
        episode_start = np.array([True], dtype=bool)

        episode_returns: list[float] = []
        episode_lengths: list[int] = []
        current_return = 0.0
        current_length = 0

        while len(episode_returns) < episodes:
            action, next_state = model.predict(
                obs,
                state=state,
                episode_start=episode_start,
                deterministic=True,
            )
            state = _require_recurrent_state(next_state)

            obs, reward, terminated, truncated, _ = env.step(
                int(np.asarray(action).item())
            )
            current_return += float(reward)
            current_length += 1
            done = terminated or truncated
            episode_start = np.array([done], dtype=bool)

            if done:
                episode_returns.append(current_return)
                episode_lengths.append(current_length)
                current_return = 0.0
                current_length = 0
                obs, _ = env.reset()
                state = None

        returns_array = np.asarray(episode_returns, dtype=np.float64)
        lengths_array = np.asarray(episode_lengths, dtype=np.int64)
        positive_return_rate = float(np.mean(returns_array > 0.0))
        return EvaluationSummary(
            episode_returns=episode_returns,
            episode_lengths=episode_lengths,
            mean_return=float(np.mean(returns_array)),
            mean_length=float(np.mean(lengths_array)),
            positive_return_rate=positive_return_rate,
        )
    finally:
        env.close()
