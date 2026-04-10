"""Smoke checks for the image-based MiniGrid benchmark setup."""

import pytest
from sb3_contrib import RecurrentPPO
from tools.minigrid_memory.models import MinigridFeaturesExtractor
from tools.minigrid_memory.support import (
    benchmark_policy_kwargs,
    build_benchmark_recurrent_model,
    make_minigrid_memory_vec_env,
    set_global_seeds,
)

from sb3x import MaskableRecurrentPPO
from sb3x.common.maskable import get_action_masks, is_masking_supported

pytest.importorskip("minigrid")


@pytest.mark.parametrize(
    ("algorithm_cls", "seed"),
    [
        (RecurrentPPO, 321),
        (MaskableRecurrentPPO, 654),
    ],
    ids=["upstream-image", "local-image"],
)
def test_minigrid_image_benchmark_smoke_learn(
    algorithm_cls: type[RecurrentPPO] | type[MaskableRecurrentPPO],
    seed: int,
) -> None:
    """Both recurrent PPO variants should build and learn on image observations."""
    set_global_seeds(seed)
    env = make_minigrid_memory_vec_env(seed, observation_mode="image")
    try:
        model = build_benchmark_recurrent_model(
            algorithm_cls,
            env,
            seed,
            n_steps=16,
            batch_size=16,
            n_epochs=1,
        )
        assert isinstance(model.policy.features_extractor, MinigridFeaturesExtractor)
        if isinstance(model, MaskableRecurrentPPO):
            model.learn(total_timesteps=32, use_masking=False)
        else:
            model.learn(total_timesteps=32)
    finally:
        env.close()


def test_benchmark_policy_kwargs_selects_custom_cnn() -> None:
    """The benchmark path should use the repo-local MiniGrid CNN extractor."""
    policy_kwargs = benchmark_policy_kwargs(cnn_features_dim=192)

    assert policy_kwargs["features_extractor_class"] is MinigridFeaturesExtractor
    assert policy_kwargs["features_extractor_kwargs"] == {"features_dim": 192}


def test_local_image_benchmark_smoke_learn_with_all_valid_masks() -> None:
    """The local algorithm should train through the trivial masked path."""
    seed = 777
    set_global_seeds(seed)
    env = make_minigrid_memory_vec_env(
        seed,
        observation_mode="image",
        mask_mode="all-valid",
    )
    try:
        assert is_masking_supported(env)
        assert get_action_masks(env).shape == (1, 7)
        model = build_benchmark_recurrent_model(
            MaskableRecurrentPPO,
            env,
            seed,
            n_steps=16,
            batch_size=16,
            n_epochs=1,
        )
        assert isinstance(model, MaskableRecurrentPPO)
        model.learn(total_timesteps=32, use_masking=True)
    finally:
        env.close()


def test_local_image_benchmark_smoke_learn_with_minigrid_basic_masks() -> None:
    """The local algorithm should train with the MiniGrid basic mask wrapper."""
    seed = 778
    set_global_seeds(seed)
    env = make_minigrid_memory_vec_env(
        seed,
        observation_mode="image",
        mask_mode="minigrid-basic",
    )
    try:
        assert is_masking_supported(env)
        action_masks = get_action_masks(env)
        assert action_masks.shape == (1, 7)
        assert bool(action_masks[0, 6]) is False
        model = build_benchmark_recurrent_model(
            MaskableRecurrentPPO,
            env,
            seed,
            n_steps=16,
            batch_size=16,
            n_epochs=1,
        )
        assert isinstance(model, MaskableRecurrentPPO)
        model.learn(total_timesteps=32, use_masking=True)
    finally:
        env.close()
