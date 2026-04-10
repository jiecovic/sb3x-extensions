"""Tests for MiniGrid watch-mode defaults and hot-reload helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from tools.minigrid_memory import watch


class _FakeWatchModel:
    """Minimal predictable model used for watch helper tests."""

    def __init__(self, source_path: Path) -> None:
        self.source_path = source_path

    def predict(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None]:
        del observation, episode_start, deterministic
        return np.asarray([0]), state


class _FakeAlgorithm:
    """Minimal algorithm loader that records loaded artifact paths."""

    loaded_paths: list[Path] = []

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> _FakeWatchModel:
        del device
        resolved = Path(path)
        cls.loaded_paths.append(resolved)
        return _FakeWatchModel(resolved)


def _fake_load_model(path: Path) -> _FakeWatchModel:
    """Adapt the fake algorithm class to the watch loader callable shape."""
    return _FakeAlgorithm.load(str(path))


def test_watch_cli_defaults_follow_latest_and_run_until_interrupted() -> None:
    """Watch should follow the latest artifact and stay open by default."""
    args = watch.parse_args(["--run-dir", "local/runs/local_0001"])

    assert args.artifact == "latest"
    assert args.episodes is None


def test_episode_seed_count_defaults_high_for_unbounded_watch() -> None:
    """Unlimited watch mode should still get a practical deterministic seed pool."""
    assert watch._episode_seed_count(None) == 4_096
    assert watch._episode_seed_count(3) == 128


def test_refresh_watch_model_reuses_loaded_model_when_artifact_is_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated checks should avoid reloading when the watched artifact is unchanged."""
    artifact_path = tmp_path / "checkpoint_10_steps.zip"
    artifact_path.write_bytes(b"checkpoint-10")
    _FakeAlgorithm.loaded_paths.clear()
    monkeypatch.setattr(
        watch,
        "resolve_artifact_path",
        lambda run_dir, artifact: artifact_path,
    )

    loaded_model, did_reload = watch._refresh_watch_model(
        tmp_path,
        "latest",
        _fake_load_model,
        None,
    )
    same_model, did_reload_again = watch._refresh_watch_model(
        tmp_path,
        "latest",
        _fake_load_model,
        loaded_model,
    )

    assert did_reload is True
    assert did_reload_again is False
    assert same_model is loaded_model
    assert _FakeAlgorithm.loaded_paths == [artifact_path]


def test_refresh_watch_model_reloads_when_new_checkpoint_appears(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Watching the latest artifact should reload when a newer checkpoint appears."""
    first_path = tmp_path / "checkpoint_10_steps.zip"
    second_path = tmp_path / "checkpoint_20_steps.zip"
    first_path.write_bytes(b"checkpoint-10")
    second_path.write_bytes(b"checkpoint-20")
    _FakeAlgorithm.loaded_paths.clear()

    resolved_paths = iter([first_path, second_path])
    monkeypatch.setattr(
        watch,
        "resolve_artifact_path",
        lambda run_dir, artifact: next(resolved_paths),
    )

    loaded_model, did_reload = watch._refresh_watch_model(
        tmp_path,
        "latest",
        _fake_load_model,
        None,
    )
    reloaded_model, did_reload_again = watch._refresh_watch_model(
        tmp_path,
        "latest",
        _fake_load_model,
        loaded_model,
    )

    assert did_reload is True
    assert did_reload_again is True
    assert reloaded_model.artifact_path == second_path
    assert _FakeAlgorithm.loaded_paths == [first_path, second_path]


def test_window_close_requested_ignores_non_human_render(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Headless watch modes should not depend on pygame close events."""
    del monkeypatch
    assert watch._window_close_requested("rgb_array") is False
