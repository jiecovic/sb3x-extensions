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
        model.learn(total_timesteps=32)
    finally:
        env.close()


def test_benchmark_policy_kwargs_selects_custom_cnn() -> None:
    """The benchmark path should use the repo-local MiniGrid CNN extractor."""
    policy_kwargs = benchmark_policy_kwargs(cnn_features_dim=192)

    assert policy_kwargs["features_extractor_class"] is MinigridFeaturesExtractor
    assert policy_kwargs["features_extractor_kwargs"] == {"features_dim": 192}
