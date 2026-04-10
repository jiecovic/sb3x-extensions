"""Tests for MiniGrid run metadata and artifact resolution helpers."""

from pathlib import Path

from tools.minigrid_memory.runs import (
    build_run_paths,
    default_run_dir,
    load_run_config,
    resolve_artifact_path,
    run_config_from_args,
    save_run_config,
)


def test_run_config_round_trip(tmp_path: Path) -> None:
    """Saved run configs should round-trip through JSON cleanly."""
    run_dir = tmp_path / "run"
    paths = build_run_paths(run_dir)
    paths.run_dir.mkdir()
    config = run_config_from_args(
        algorithm="local",
        policy="CnnLstmPolicy",
        observation_mode="image",
        mask_mode="all-valid",
        seed=123,
        total_timesteps=1_000,
        n_envs=4,
        n_steps=32,
        batch_size=32,
        n_epochs=2,
        eval_freq=200,
        eval_episodes=5,
        checkpoint_freq=100,
        lstm_hidden_size=64,
        mlp_hidden_size=64,
        cnn_features_dim=128,
        verbose=0,
        progress_bar=True,
        tensorboard=True,
    )

    save_run_config(config, paths.config_path)

    loaded = load_run_config(run_dir)
    assert loaded == config


def test_resolve_latest_checkpoint_prefers_highest_step(tmp_path: Path) -> None:
    """Latest artifact resolution should choose the highest-step checkpoint."""
    run_dir = tmp_path / "run"
    paths = build_run_paths(run_dir)
    paths.checkpoints_dir.mkdir(parents=True)
    paths.final_model_path.write_text("final", encoding="utf-8")
    (paths.checkpoints_dir / "checkpoint_100_steps.zip").write_text(
        "100",
        encoding="utf-8",
    )
    latest_path = paths.checkpoints_dir / "checkpoint_250_steps.zip"
    latest_path.write_text("250", encoding="utf-8")

    resolved = resolve_artifact_path(run_dir, "latest")
    assert resolved == latest_path


def test_resolve_latest_falls_back_to_final(tmp_path: Path) -> None:
    """Latest artifact resolution should fall back to the final model."""
    run_dir = tmp_path / "run"
    paths = build_run_paths(run_dir)
    paths.run_dir.mkdir(parents=True)
    paths.final_model_path.write_text("final", encoding="utf-8")

    resolved = resolve_artifact_path(run_dir, "latest")
    assert resolved == paths.final_model_path


def test_default_run_dir_is_concise_without_label(tmp_path: Path) -> None:
    """Default run names should use short numbered directories."""
    run_dir = default_run_dir(tmp_path, algorithm="local")

    assert run_dir.parent == tmp_path.resolve()
    assert run_dir.name == "local_0001"


def test_default_run_dir_uses_label_when_provided(tmp_path: Path) -> None:
    """Explicit labels should override the default algorithm stem cleanly."""
    run_dir = default_run_dir(
        tmp_path,
        algorithm="upstream",
        label="baseline a",
    )

    assert run_dir.parent == tmp_path.resolve()
    assert run_dir.name == "baseline_a_0001"


def test_default_run_dir_increments_existing_numbered_runs(tmp_path: Path) -> None:
    """Subsequent runs should increment the numeric suffix for the same stem."""
    (tmp_path / "local_0001").mkdir()
    (tmp_path / "local_0002").mkdir()

    run_dir = default_run_dir(tmp_path, algorithm="local")

    assert run_dir.name == "local_0003"
