"""MiniGrid checks for upstream and local recurrent PPO variants."""

import numpy as np
import pytest
from sb3_contrib import RecurrentPPO
from tools.minigrid_memory.support import (
    build_recurrent_model,
    collect_deterministic_rollout,
    copy_policy_state,
    make_minigrid_memory_vec_env,
    max_policy_parameter_diff,
    set_global_seeds,
)

from sb3x import MaskableRecurrentPPO

pytest.importorskip("minigrid")


@pytest.mark.parametrize(
    ("algorithm_cls", "seed"),
    [
        (RecurrentPPO, 123),
        (MaskableRecurrentPPO, 456),
    ],
    ids=["upstream", "local-copy"],
)
def test_minigrid_memory_smoke_learn(
    algorithm_cls: type[RecurrentPPO] | type[MaskableRecurrentPPO],
    seed: int,
) -> None:
    """Both recurrent PPO variants should run a small MiniGrid learn() call."""
    set_global_seeds(seed)
    env = make_minigrid_memory_vec_env(seed)
    try:
        model = build_recurrent_model(algorithm_cls, env, seed)
        if isinstance(model, MaskableRecurrentPPO):
            model.learn(total_timesteps=32, use_masking=False)
        else:
            model.learn(total_timesteps=32)
    finally:
        env.close()


def test_local_ppo_mask_recurrent_matches_upstream_deterministic_rollout() -> None:
    """The local copied algorithm should match upstream on deterministic steps."""
    seed = 789

    set_global_seeds(seed)
    upstream_env = make_minigrid_memory_vec_env(seed)
    set_global_seeds(seed)
    local_env = make_minigrid_memory_vec_env(seed)

    try:
        upstream = build_recurrent_model(RecurrentPPO, upstream_env, seed)
        local = build_recurrent_model(MaskableRecurrentPPO, local_env, seed)
        copy_policy_state(upstream, local)
    finally:
        upstream_env.close()
        local_env.close()

    assert max_policy_parameter_diff(upstream, local) == 0.0

    upstream_rollout = collect_deterministic_rollout(
        upstream,
        seed=seed + 1_000,
        steps=24,
    )
    local_rollout = collect_deterministic_rollout(
        local,
        seed=seed + 1_000,
        steps=24,
    )

    assert len(upstream_rollout) == len(local_rollout)
    for expected, actual in zip(upstream_rollout, local_rollout):
        assert expected.step == actual.step
        np.testing.assert_array_equal(expected.action, actual.action)
        np.testing.assert_allclose(expected.hidden_state, actual.hidden_state)
        np.testing.assert_allclose(expected.cell_state, actual.cell_state)
        assert expected.reward == actual.reward
        assert expected.done is actual.done
