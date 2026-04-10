"""Train upstream or local recurrent PPO on MiniGrid Memory."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import asdict
from importlib.util import find_spec
from pathlib import Path

from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)

from sb3x import MaskableRecurrentPPO

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
except ModuleNotFoundError:  # pragma: no cover - repo-local fallback
    Console = None
    Panel = None
    Table = None

from .runs import (
    MiniGridMemoryRunConfig,
    MiniGridRunPaths,
    algorithm_class_from_name,
    build_run_paths,
    default_run_dir,
    ensure_run_dirs,
    run_config_from_args,
    save_run_config,
)
from .support import (
    MaskMode,
    ObservationMode,
    RecurrentAlgorithm,
    benchmark_policy_kwargs,
    build_benchmark_recurrent_model,
    build_recurrent_model,
    make_minigrid_memory_vec_env,
    recurrent_policy_kwargs,
    set_global_seeds,
)


def _tensorboard_log_dir_or_none(
    tensorboard_dir: Path,
    *,
    enabled: bool,
) -> str | None:
    """Return a tensorboard log dir only when the optional dependency exists."""
    if not enabled:
        return None
    if find_spec("tensorboard") is None:
        print("tensorboard is not installed; disabling tensorboard logging")
        return None
    return str(tensorboard_dir)


def _progress_bar_enabled(*, enabled: bool) -> bool:
    """Return whether the optional SB3 progress bar can be enabled."""
    if not enabled:
        return False
    if find_spec("rich") is None or find_spec("tqdm") is None:
        print("rich or tqdm is not installed; disabling progress bar")
        return False
    return True


def _policy_parameter_count(model: RecurrentAlgorithm) -> int:
    """Return the total parameter count for one SB3 policy module."""
    policy = getattr(model, "policy")
    return sum(parameter.numel() for parameter in policy.parameters())


def _print_run_summary(
    run_paths: MiniGridRunPaths,
    run_config: MiniGridMemoryRunConfig,
    *,
    policy_kwargs: dict[str, object],
    model: RecurrentAlgorithm,
) -> None:
    """Print one structured summary of the run configuration and policy."""
    if Console is None or Panel is None or Table is None:
        print("Run configuration:")
        for key, value in asdict(run_config).items():
            print(f"  {key}: {value}")
        print(f"  run_dir: {run_paths.run_dir}")
        print(f"  checkpoints_dir: {run_paths.checkpoints_dir}")
        print(f"  final_model_path: {run_paths.final_model_path}")
        print("Policy:")
        print(model.policy)
        return

    console = Console()

    config_table = Table(title="MiniGrid Memory Run", show_header=False)
    config_table.add_column("key", style="cyan")
    config_table.add_column("value", style="white")
    for key, value in asdict(run_config).items():
        config_table.add_row(key, str(value))
    config_table.add_row("run_dir", str(run_paths.run_dir))
    config_table.add_row("checkpoints_dir", str(run_paths.checkpoints_dir))
    config_table.add_row("final_model_path", str(run_paths.final_model_path))

    policy = model.policy
    policy_table = Table(title="Policy Summary", show_header=False)
    policy_table.add_column("key", style="magenta")
    policy_table.add_column("value", style="white")
    policy_table.add_row("policy_class", type(policy).__name__)
    policy_table.add_row("features_extractor", type(policy.features_extractor).__name__)
    policy_table.add_row("lstm_actor", type(policy.lstm_actor).__name__)
    policy_table.add_row("mlp_extractor", type(policy.mlp_extractor).__name__)
    policy_table.add_row("action_net", type(policy.action_net).__name__)
    policy_table.add_row("value_net", type(policy.value_net).__name__)
    policy_table.add_row("parameter_count", f"{_policy_parameter_count(model):,}")
    policy_table.add_row("policy_kwargs", str(policy_kwargs))

    console.print(config_table)
    console.print(policy_table)
    console.print(
        Panel.fit(
            str(policy),
            title="Torch Modules",
            border_style="blue",
        )
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for MiniGrid Memory training."""
    parser = argparse.ArgumentParser(
        description=(
            "Train upstream sb3-contrib RecurrentPPO or the local "
            "MaskableRecurrentPPO copy on MiniGrid-MemoryS7-v0."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--algo",
        choices=("upstream", "local"),
        default="local",
        help="Which recurrent PPO implementation to train.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100_000,
        help="Total environment timesteps to train.",
    )
    parser.add_argument("--seed", type=int, default=123, help="Base random seed.")
    parser.add_argument(
        "--n-envs", type=int, default=8, help="Number of training envs."
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=128,
        help="Rollout steps per environment before each update.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size used during optimization.",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=4,
        help="Number of epochs per PPO update.",
    )
    parser.add_argument(
        "--observation-mode",
        choices=("flat", "image"),
        default="image",
        help="Observation path: flat for parity checks, image for benchmark runs.",
    )
    parser.add_argument(
        "--mask-mode",
        choices=("none", "all-valid", "minigrid-basic"),
        default="none",
        help=(
            "Masking path for the local algorithm. 'all-valid' forces the "
            "masked code path without restricting actions; 'minigrid-basic' "
            "masks obvious MiniGrid no-op actions."
        ),
    )
    parser.add_argument(
        "--lstm-hidden-size",
        type=int,
        default=128,
        help="LSTM hidden size for the recurrent policy.",
    )
    parser.add_argument(
        "--mlp-hidden-size",
        type=int,
        default=64,
        help="Hidden size for the policy/value MLP heads.",
    )
    parser.add_argument(
        "--cnn-features-dim",
        type=int,
        default=128,
        help="Feature dimension emitted by the MiniGrid CNN extractor.",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=5_000,
        help="Timesteps between evaluation runs.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Number of deterministic episodes per evaluation run.",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=5_000,
        help="Timesteps between checkpoint saves.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("local/runs"),
        help="Root directory for auto-created training runs.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Explicit output directory for this training run.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional label used when auto-creating a run directory.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        choices=(0, 1, 2),
        default=0,
        help="Stable-Baselines3 verbosity level.",
    )
    parser.add_argument(
        "--progress-bar",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show the SB3 progress bar during training.",
    )
    parser.add_argument(
        "--tensorboard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable tensorboard logging for this run.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Train a recurrent PPO variant and save the resulting run artifacts."""
    args = parse_args(argv)
    algorithm = args.algo
    observation_mode: ObservationMode = args.observation_mode
    mask_mode: MaskMode = args.mask_mode
    algorithm_cls = algorithm_class_from_name(algorithm)

    if mask_mode != "none" and algorithm != "local":
        raise ValueError(
            "Mask modes other than 'none' are only supported for --algo local"
        )

    run_dir = (
        args.run_dir.expanduser().resolve()
        if args.run_dir is not None
        else default_run_dir(
            args.runs_root,
            algorithm=algorithm,
            label=args.name,
        )
    )
    run_paths = build_run_paths(run_dir)
    ensure_run_dirs(run_paths)

    policy = "CnnLstmPolicy" if observation_mode == "image" else "MlpLstmPolicy"
    run_config = run_config_from_args(
        algorithm=algorithm,
        policy=policy,
        observation_mode=observation_mode,
        mask_mode=mask_mode,
        seed=args.seed,
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        checkpoint_freq=args.checkpoint_freq,
        lstm_hidden_size=args.lstm_hidden_size,
        mlp_hidden_size=args.mlp_hidden_size,
        cnn_features_dim=args.cnn_features_dim if observation_mode == "image" else 0,
        verbose=args.verbose,
        progress_bar=args.progress_bar,
        tensorboard=args.tensorboard,
    )
    save_run_config(run_config, run_paths.config_path)

    set_global_seeds(args.seed)
    policy_kwargs = (
        benchmark_policy_kwargs(cnn_features_dim=args.cnn_features_dim)
        if observation_mode == "image"
        else recurrent_policy_kwargs()
    )
    policy_kwargs["lstm_hidden_size"] = args.lstm_hidden_size
    policy_kwargs["net_arch"] = [args.mlp_hidden_size]

    train_env = make_minigrid_memory_vec_env(
        args.seed,
        n_envs=args.n_envs,
        episode_seed_count=max(args.timesteps // 16, 4_096),
        deterministic_resets=True,
        observation_mode=observation_mode,
        mask_mode=mask_mode,
    )
    eval_env = make_minigrid_memory_vec_env(
        args.seed + 50_000,
        n_envs=1,
        episode_seed_count=max(args.eval_episodes * 8, 128),
        deterministic_resets=True,
        observation_mode=observation_mode,
        mask_mode=mask_mode,
    )

    try:
        model_builder = (
            build_benchmark_recurrent_model
            if observation_mode == "image"
            else build_recurrent_model
        )
        model = model_builder(
            algorithm_cls,
            train_env,
            args.seed,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=_tensorboard_log_dir_or_none(
                run_paths.tensorboard_dir,
                enabled=args.tensorboard,
            ),
            verbose=args.verbose,
        )
        _print_run_summary(
            run_paths,
            run_config,
            policy_kwargs=policy_kwargs,
            model=model,
        )

        callbacks = CallbackList(
            [
                EvalCallback(
                    eval_env,
                    best_model_save_path=str(run_paths.checkpoints_dir),
                    log_path=str(run_paths.eval_dir),
                    eval_freq=max(1, args.eval_freq // args.n_envs),
                    n_eval_episodes=args.eval_episodes,
                    deterministic=True,
                    verbose=args.verbose,
                ),
                CheckpointCallback(
                    save_freq=max(1, args.checkpoint_freq // args.n_envs),
                    save_path=str(run_paths.checkpoints_dir),
                    name_prefix="checkpoint",
                    verbose=args.verbose,
                ),
            ]
        )

        print(
            f"Training {algorithm} recurrent PPO"
            f" (mask_mode={mask_mode}) into {run_paths.run_dir}"
        )
        progress_bar = _progress_bar_enabled(enabled=args.progress_bar)
        if algorithm == "local":
            if not isinstance(model, MaskableRecurrentPPO):
                raise TypeError("Expected local algorithm to be MaskableRecurrentPPO")
            model.learn(
                total_timesteps=args.timesteps,
                callback=callbacks,
                use_masking=mask_mode != "none",
                progress_bar=progress_bar,
            )
        else:
            model.learn(
                total_timesteps=args.timesteps,
                callback=callbacks,
                progress_bar=progress_bar,
            )
        model.save(str(run_paths.final_model_path))
        print(f"Saved final model to {run_paths.final_model_path}")
    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
