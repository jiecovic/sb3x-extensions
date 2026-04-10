"""Compare upstream RecurrentPPO and local MaskableRecurrentPPO on MiniGrid."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass

import torch as th
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecEnv
from tools.minigrid_memory.support import (
    EvaluationSummary,
    build_recurrent_model,
    collect_deterministic_rollout,
    copy_policy_state,
    evaluate_deterministic_policy,
    make_minigrid_memory_vec_env,
    max_policy_parameter_diff,
    set_global_seeds,
)

from sb3x import MaskableRecurrentPPO


@dataclass(frozen=True)
class ParityResult:
    """Artifacts from the initial deterministic parity check."""

    mismatch_count: int
    rollout_steps: int
    initial_policy_diff: float
    initial_policy_state: dict[str, th.Tensor]


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the MiniGrid comparison script."""
    parser = argparse.ArgumentParser(
        description=(
            "Run a deterministic parity check and optional train/eval "
            "comparison between upstream RecurrentPPO and the local "
            "MaskableRecurrentPPO copy on MiniGrid-MemoryS7-v0."
        )
    )
    parser.add_argument("--seed", type=int, default=123, help="Base random seed.")
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=24,
        help="Number of deterministic predict() steps to compare.",
    )
    parser.add_argument(
        "--train-timesteps",
        type=int,
        default=256,
        help="Training timesteps for the optional train/eval comparison.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Number of deterministic evaluation episodes after training.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=16,
        help="Rollout length used for both algorithms.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size used for both algorithms.",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=1,
        help="Number of training epochs used for both algorithms.",
    )
    return parser.parse_args()


def _build_model_for_comparison(
    algorithm_cls: type[RecurrentPPO] | type[MaskableRecurrentPPO],
    *,
    seed: int,
    n_steps: int,
    batch_size: int,
    n_epochs: int,
) -> tuple[RecurrentPPO | MaskableRecurrentPPO, VecEnv]:
    """Build a recurrent PPO model and return it with its vectorized env."""
    set_global_seeds(seed)
    env = make_minigrid_memory_vec_env(seed)
    model = build_recurrent_model(
        algorithm_cls,
        env,
        seed,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
    )
    return model, env


def _print_rollout_parity(seed: int, rollout_steps: int) -> ParityResult:
    """Print a deterministic predict() parity trace before any training."""
    upstream, upstream_env = _build_model_for_comparison(
        RecurrentPPO,
        seed=seed,
        n_steps=16,
        batch_size=16,
        n_epochs=1,
    )
    try:
        local, local_env = _build_model_for_comparison(
            MaskableRecurrentPPO,
            seed=seed,
            n_steps=16,
            batch_size=16,
            n_epochs=1,
        )
        try:
            copy_policy_state(upstream, local)
            rollout_seed = seed + 1_000
            upstream_rollout = collect_deterministic_rollout(
                upstream,
                seed=rollout_seed,
                steps=rollout_steps,
            )
            local_rollout = collect_deterministic_rollout(
                local,
                seed=rollout_seed,
                steps=rollout_steps,
            )
        finally:
            local_env.close()
    finally:
        upstream_env.close()

    mismatch_count = 0
    print("Deterministic predict() parity")
    for expected, actual in zip(upstream_rollout, local_rollout):
        action_match = bool((expected.action == actual.action).all())
        hidden_diff = float(
            max(
                abs(expected.hidden_state - actual.hidden_state).max(),
                abs(expected.cell_state - actual.cell_state).max(),
            )
        )
        if not action_match or hidden_diff > 0.0 or expected.reward != actual.reward:
            mismatch_count += 1
        is_exact_match = (
            action_match and hidden_diff == 0.0 and expected.reward == actual.reward
        )

        print(
            f"step={expected.step:02d} "
            f"action={expected.action.tolist()} "
            f"reward={expected.reward:.3f} "
            f"done={expected.done} "
            f"hidden_max_diff={hidden_diff:.3e} "
            f"match={is_exact_match}"
        )

    return ParityResult(
        mismatch_count=mismatch_count,
        rollout_steps=len(upstream_rollout),
        initial_policy_diff=max_policy_parameter_diff(upstream, local),
        initial_policy_state=deepcopy(dict(upstream.policy.state_dict())),
    )


def _train_and_evaluate(
    algorithm_cls: type[RecurrentPPO] | type[MaskableRecurrentPPO],
    *,
    seed: int,
    train_timesteps: int,
    eval_episodes: int,
    n_steps: int,
    batch_size: int,
    n_epochs: int,
    initial_policy_state: dict[str, th.Tensor],
) -> tuple[RecurrentPPO | MaskableRecurrentPPO, EvaluationSummary]:
    """Train a recurrent PPO variant from a shared initial policy state."""
    model, env = _build_model_for_comparison(
        algorithm_cls,
        seed=seed,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
    )
    try:
        model.policy.load_state_dict(deepcopy(initial_policy_state))
        set_global_seeds(seed)
        model.learn(total_timesteps=train_timesteps)
        summary = evaluate_deterministic_policy(
            model,
            seed=seed + 2_000,
            episodes=eval_episodes,
        )
        return model, summary
    finally:
        env.close()


def main() -> None:
    """Run deterministic parity and optional train/eval comparison."""
    args = parse_args()
    parity_result = _print_rollout_parity(args.seed, args.rollout_steps)
    print(
        "Initial policy max parameter diff:",
        f"{parity_result.initial_policy_diff:.3e}",
    )
    print("Predict mismatches:", parity_result.mismatch_count)

    if args.train_timesteps <= 0:
        return

    upstream_model, upstream_summary = _train_and_evaluate(
        RecurrentPPO,
        seed=args.seed,
        train_timesteps=args.train_timesteps,
        eval_episodes=args.eval_episodes,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        initial_policy_state=parity_result.initial_policy_state,
    )
    local_model, local_summary = _train_and_evaluate(
        MaskableRecurrentPPO,
        seed=args.seed,
        train_timesteps=args.train_timesteps,
        eval_episodes=args.eval_episodes,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        initial_policy_state=parity_result.initial_policy_state,
    )

    print()
    print("Post-train evaluation")
    print(
        "upstream:",
        f"mean_return={upstream_summary.mean_return:.4f}",
        f"mean_length={upstream_summary.mean_length:.2f}",
        f"positive_return_rate={upstream_summary.positive_return_rate:.2%}",
    )
    print(
        "local:   ",
        f"mean_return={local_summary.mean_return:.4f}",
        f"mean_length={local_summary.mean_length:.2f}",
        f"positive_return_rate={local_summary.positive_return_rate:.2%}",
    )
    print(
        "post-train max policy parameter diff:",
        f"{max_policy_parameter_diff(upstream_model, local_model):.3e}",
    )


if __name__ == "__main__":
    main()
