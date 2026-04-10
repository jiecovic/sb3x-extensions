"""Watch one saved MiniGrid Memory recurrent PPO run."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Protocol

import numpy as np

from .runs import (
    ArtifactName,
    algorithm_class_from_name,
    load_run_config,
    resolve_artifact_path,
)
from .support import RecurrentAlgorithm, make_minigrid_memory_env, set_global_seeds

WatchObservation = np.ndarray | dict[str, np.ndarray]


class WatchModel(Protocol):
    """Small protocol for the saved recurrent policy used by watch mode."""

    def predict(
        self,
        observation: WatchObservation,
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None]: ...


@dataclass(frozen=True)
class AlgorithmWatchModel:
    """Thin adapter that exposes only the watch-time predict surface."""

    algorithm: RecurrentAlgorithm

    def predict(
        self,
        observation: WatchObservation,
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None]:
        return self.algorithm.predict(
            observation,
            state=state,
            episode_start=episode_start,
            deterministic=deterministic,
        )


@dataclass(frozen=True)
class LoadedWatchModel:
    """Loaded watch artifact together with the file stamp used for hot reload."""

    model: WatchModel
    artifact_path: Path
    artifact_mtime_ns: int
    artifact_size: int


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for watching a saved MiniGrid run."""
    parser = argparse.ArgumentParser(
        description=(
            "Watch one saved upstream or local recurrent PPO policy on "
            "MiniGrid-MemoryS7-v0."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="Training run directory containing run_config.json and model artifacts.",
    )
    parser.add_argument(
        "--artifact",
        choices=("best", "final", "latest"),
        default="latest",
        help="Which saved artifact to follow from the run directory.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Optional number of episodes to watch before exiting.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=8.0,
        help="Playback frame rate. Use 0 to disable sleeping.",
    )
    parser.add_argument(
        "--render-mode",
        choices=("human", "rgb_array"),
        default="human",
        help="How to render the environment during watch mode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional override for the watch env seed.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic action sampling instead of deterministic predict().",
    )
    return parser.parse_args(argv)


def _episode_seed_count(episodes: int | None) -> int:
    """Return a practical deterministic reset pool size for watch mode."""
    if episodes is None:
        return 4_096
    return max(episodes * 8, 128)


def _window_close_requested(render_mode: str) -> bool:
    """Return whether the human-render window received a close request."""
    if render_mode != "human":
        return False

    try:
        pygame = import_module("pygame")
    except ModuleNotFoundError:
        return False

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
    return False


def _load_watch_model(
    load_model: Callable[[Path], WatchModel],
    artifact_path: Path,
) -> LoadedWatchModel:
    """Load one saved watch artifact and capture its on-disk fingerprint."""
    stat_result = artifact_path.stat()
    return LoadedWatchModel(
        model=load_model(artifact_path),
        artifact_path=artifact_path,
        artifact_mtime_ns=stat_result.st_mtime_ns,
        artifact_size=stat_result.st_size,
    )


def _refresh_watch_model(
    run_dir: Path,
    artifact: ArtifactName,
    load_model: Callable[[Path], WatchModel],
    loaded_model: LoadedWatchModel | None,
) -> tuple[LoadedWatchModel, bool]:
    """Reload the watched artifact when its path or file stamp changes."""
    artifact_path = resolve_artifact_path(run_dir, artifact)
    stat_result = artifact_path.stat()

    if (
        loaded_model is not None
        and loaded_model.artifact_path == artifact_path
        and loaded_model.artifact_mtime_ns == stat_result.st_mtime_ns
        and loaded_model.artifact_size == stat_result.st_size
    ):
        return loaded_model, False

    return _load_watch_model(load_model, artifact_path), True


def main(argv: Sequence[str] | None = None) -> None:
    """Load a saved policy artifact and step through watch episodes."""
    args = parse_args(argv)
    run_dir = args.run_dir.expanduser().resolve()
    run_config = load_run_config(run_dir)
    algorithm_cls = algorithm_class_from_name(run_config.algorithm)

    def load_model(artifact_path: Path) -> WatchModel:
        algorithm = algorithm_cls.load(
            str(artifact_path),
            device="cpu",
        )
        return AlgorithmWatchModel(algorithm)

    watch_seed = run_config.seed if args.seed is None else args.seed
    set_global_seeds(watch_seed)
    env = make_minigrid_memory_env(
        watch_seed,
        episode_seed_count=_episode_seed_count(args.episodes),
        deterministic_resets=True,
        observation_mode=run_config.observation_mode,
        render_mode=args.render_mode,
    )

    try:
        loaded_model: LoadedWatchModel | None = None
        state: tuple[np.ndarray, ...] | None = None
        episode_start = np.array([True], dtype=bool)
        episode_index = 0

        while args.episodes is None or episode_index < args.episodes:
            loaded_model, did_reload = _refresh_watch_model(
                run_dir,
                args.artifact,
                load_model,
                loaded_model,
            )
            if did_reload:
                action = "Loaded" if episode_index == 0 else "Reloaded"
                print(f"{action} artifact: {loaded_model.artifact_path}")
            obs, _ = env.reset()
            state = None
            episode_start = np.array([True], dtype=bool)
            done = False
            episode_return = 0.0
            episode_length = 0

            if args.render_mode == "rgb_array":
                env.render()
            elif _window_close_requested(args.render_mode):
                raise KeyboardInterrupt

            while not done:
                step_started_at = time.perf_counter()
                action, state = loaded_model.model.predict(
                    obs,
                    state=state,
                    episode_start=episode_start,
                    deterministic=not args.stochastic,
                )
                obs, reward, terminated, truncated, _ = env.step(
                    int(np.asarray(action).item())
                )
                if args.render_mode == "rgb_array":
                    env.render()
                elif _window_close_requested(args.render_mode):
                    raise KeyboardInterrupt

                done = terminated or truncated
                episode_return += float(reward)
                episode_length += 1
                episode_start = np.array([done], dtype=bool)

                if args.fps > 0:
                    elapsed = time.perf_counter() - step_started_at
                    time.sleep(max(0.0, (1.0 / args.fps) - elapsed))

            print(
                f"episode={episode_index + 1} "
                f"return={episode_return:.4f} "
                f"length={episode_length}"
            )
            episode_index += 1
    except KeyboardInterrupt:
        print("watch interrupted")
    finally:
        env.close()


if __name__ == "__main__":
    main()
