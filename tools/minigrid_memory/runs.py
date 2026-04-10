"""Run metadata and artifact helpers for MiniGrid Memory training."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from sb3_contrib import RecurrentPPO

from sb3x import MaskableRecurrentPPO

from .support import MINIGRID_MEMORY_ENV_ID, MaskMode, ObservationMode

AlgorithmName = Literal["upstream", "local"]
ArtifactName = Literal["best", "final", "latest"]

_CHECKPOINT_STEPS_PATTERN = re.compile(r"^checkpoint_(\d+)_steps\.zip$")


@dataclass(frozen=True)
class MiniGridMemoryRunConfig:
    """Serialized training configuration for one MiniGrid Memory run."""

    algorithm: AlgorithmName
    env_id: str
    policy: str
    observation_mode: ObservationMode
    mask_mode: MaskMode
    seed: int
    total_timesteps: int
    n_envs: int
    n_steps: int
    batch_size: int
    n_epochs: int
    eval_freq: int
    eval_episodes: int
    checkpoint_freq: int
    lstm_hidden_size: int
    mlp_hidden_size: int
    cnn_features_dim: int
    verbose: int
    progress_bar: bool
    tensorboard: bool
    created_at: str


@dataclass(frozen=True)
class MiniGridRunPaths:
    """Filesystem layout for one saved MiniGrid training run."""

    run_dir: Path
    checkpoints_dir: Path
    eval_dir: Path
    tensorboard_dir: Path
    config_path: Path
    final_model_path: Path
    best_model_path: Path


def build_run_paths(run_dir: Path) -> MiniGridRunPaths:
    """Build the expected artifact paths for one run directory."""
    resolved_run_dir = run_dir.expanduser().resolve()
    return MiniGridRunPaths(
        run_dir=resolved_run_dir,
        checkpoints_dir=resolved_run_dir / "checkpoints",
        eval_dir=resolved_run_dir / "eval",
        tensorboard_dir=resolved_run_dir / "tensorboard",
        config_path=resolved_run_dir / "run_config.json",
        final_model_path=resolved_run_dir / "final_model.zip",
        best_model_path=resolved_run_dir / "checkpoints" / "best_model.zip",
    )


def ensure_run_dirs(paths: MiniGridRunPaths) -> None:
    """Create the directories needed for one training run."""
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    paths.eval_dir.mkdir(parents=True, exist_ok=True)
    paths.tensorboard_dir.mkdir(parents=True, exist_ok=True)


def default_run_dir(
    runs_root: Path,
    *,
    algorithm: AlgorithmName,
    label: str | None = None,
) -> Path:
    """Return the next numbered run directory, similar to rl-fzerox."""
    resolved_runs_root = runs_root.expanduser().resolve()
    run_name = _sanitize_run_name(label) if label is not None else algorithm
    return _next_run_dir(resolved_runs_root, run_name)


def save_run_config(config: MiniGridMemoryRunConfig, path: Path) -> None:
    """Write one run config JSON file."""
    path.write_text(
        json.dumps(asdict(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_run_config(run_dir: Path) -> MiniGridMemoryRunConfig:
    """Load one run config JSON file from a run directory."""
    config_path = build_run_paths(run_dir).config_path
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if "observation_mode" not in raw:
        raw["observation_mode"] = (
            "image" if raw.get("policy") == "CnnLstmPolicy" else "flat"
        )
    if "mask_mode" not in raw:
        raw["mask_mode"] = "none"
    if "cnn_features_dim" not in raw:
        raw["cnn_features_dim"] = 0
    return MiniGridMemoryRunConfig(**raw)


def resolve_artifact_path(run_dir: Path, artifact: ArtifactName) -> Path:
    """Resolve one saved model artifact from a run directory."""
    paths = build_run_paths(run_dir)
    if artifact == "final":
        if not paths.final_model_path.exists():
            raise FileNotFoundError(
                f"Final model not found at {paths.final_model_path}"
            )
        return paths.final_model_path

    if artifact == "best":
        if not paths.best_model_path.exists():
            raise FileNotFoundError(f"Best model not found at {paths.best_model_path}")
        return paths.best_model_path

    latest_checkpoint = _latest_checkpoint(paths.checkpoints_dir)
    if latest_checkpoint is not None:
        return latest_checkpoint
    if paths.final_model_path.exists():
        return paths.final_model_path
    raise FileNotFoundError(f"No checkpoint or final model found under {paths.run_dir}")


def algorithm_class_from_name(
    algorithm: AlgorithmName,
) -> type[RecurrentPPO] | type[MaskableRecurrentPPO]:
    """Map one saved algorithm name to its concrete class."""
    if algorithm == "upstream":
        return RecurrentPPO
    return MaskableRecurrentPPO


def run_config_from_args(
    *,
    algorithm: AlgorithmName,
    policy: str,
    observation_mode: ObservationMode,
    mask_mode: MaskMode,
    seed: int,
    total_timesteps: int,
    n_envs: int,
    n_steps: int,
    batch_size: int,
    n_epochs: int,
    eval_freq: int,
    eval_episodes: int,
    checkpoint_freq: int,
    lstm_hidden_size: int,
    mlp_hidden_size: int,
    cnn_features_dim: int,
    verbose: int,
    progress_bar: bool,
    tensorboard: bool,
) -> MiniGridMemoryRunConfig:
    """Build one serializable run config from train CLI arguments."""
    return MiniGridMemoryRunConfig(
        algorithm=algorithm,
        env_id=MINIGRID_MEMORY_ENV_ID,
        policy=policy,
        observation_mode=observation_mode,
        mask_mode=mask_mode,
        seed=seed,
        total_timesteps=total_timesteps,
        n_envs=n_envs,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        eval_freq=eval_freq,
        eval_episodes=eval_episodes,
        checkpoint_freq=checkpoint_freq,
        lstm_hidden_size=lstm_hidden_size,
        mlp_hidden_size=mlp_hidden_size,
        cnn_features_dim=cnn_features_dim,
        verbose=verbose,
        progress_bar=progress_bar,
        tensorboard=tensorboard,
        created_at=datetime.now(tz=timezone.utc).isoformat(),
    )


def _latest_checkpoint(checkpoints_dir: Path) -> Path | None:
    """Return the numerically latest checkpoint file if any exist."""
    best_path: Path | None = None
    best_steps = -1
    for path in checkpoints_dir.glob("checkpoint_*_steps.zip"):
        match = _CHECKPOINT_STEPS_PATTERN.match(path.name)
        if match is None:
            continue
        steps = int(match.group(1))
        if steps > best_steps:
            best_steps = steps
            best_path = path

    return best_path


def _next_run_dir(runs_root: Path, run_name: str) -> Path:
    """Return the next numbered run directory for one run name."""
    runs_root.mkdir(parents=True, exist_ok=True)
    prefix = f"{run_name}_"
    next_index = 1

    for child in runs_root.iterdir():
        if not child.is_dir() or not child.name.startswith(prefix):
            continue
        suffix = child.name.removeprefix(prefix)
        if suffix.isdigit():
            next_index = max(next_index, int(suffix) + 1)

    return runs_root / f"{run_name}_{next_index:04d}"


def _sanitize_run_name(value: str) -> str:
    """Normalize one free-form run label into a short directory stem."""
    stripped = value.strip().lower()
    if not stripped:
        raise ValueError("Run label must contain at least one non-space character")
    sanitized = re.sub(r"[^a-z0-9]+", "_", stripped).strip("_")
    if not sanitized:
        raise ValueError("Run label must contain at least one alphanumeric character")
    return sanitized
