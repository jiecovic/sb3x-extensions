"""Smoke tests for the minimal package scaffold."""

from importlib.metadata import version

import sb3x
from sb3x import (
    HybridActionPPO,
    HybridRecurrentPPO,
    MaskableHybridActionPPO,
    MaskableRecurrentPPO,
)
from sb3x.ppo_hybrid_action import (
    CnnPolicy,
    MlpPolicy,
    MultiInputPolicy,
)
from sb3x.ppo_hybrid_recurrent import (
    CnnLstmPolicy as HybridRecurrentCnnLstmPolicy,
)
from sb3x.ppo_hybrid_recurrent import (
    MlpLstmPolicy as HybridRecurrentMlpLstmPolicy,
)
from sb3x.ppo_hybrid_recurrent import (
    MultiInputLstmPolicy as HybridRecurrentMultiInputLstmPolicy,
)
from sb3x.ppo_mask_hybrid_action import (
    CnnPolicy as MaskableHybridCnnPolicy,
)
from sb3x.ppo_mask_hybrid_action import (
    MlpPolicy as MaskableHybridMlpPolicy,
)
from sb3x.ppo_mask_hybrid_action import (
    MultiInputPolicy as MaskableHybridMultiInputPolicy,
)
from sb3x.ppo_mask_recurrent import (
    CnnLstmPolicy,
    MlpLstmPolicy,
    MultiInputLstmPolicy,
)


def test_import_smoke() -> None:
    """The top-level package exposes the current algorithm namespace cleanly."""
    assert sb3x.__version__ == version("sb3x")
    assert sb3x.HybridActionPPO is HybridActionPPO
    assert sb3x.HybridRecurrentPPO is HybridRecurrentPPO
    assert sb3x.MaskableHybridActionPPO is MaskableHybridActionPPO
    assert sb3x.MaskableRecurrentPPO is MaskableRecurrentPPO
    assert MlpPolicy.__name__ == "HybridActionActorCriticPolicy"
    assert CnnPolicy.__name__ == "HybridActionActorCriticCnnPolicy"
    assert MultiInputPolicy.__name__ == "HybridActionMultiInputActorCriticPolicy"
    assert MaskableHybridMlpPolicy.__name__ == "MaskableHybridActionActorCriticPolicy"
    assert (
        MaskableHybridCnnPolicy.__name__ == "MaskableHybridActionActorCriticCnnPolicy"
    )
    assert (
        MaskableHybridMultiInputPolicy.__name__
        == "MaskableHybridActionMultiInputActorCriticPolicy"
    )
    assert HybridRecurrentMlpLstmPolicy.__name__ == "HybridRecurrentActorCriticPolicy"
    assert (
        HybridRecurrentCnnLstmPolicy.__name__ == "HybridRecurrentActorCriticCnnPolicy"
    )
    assert (
        HybridRecurrentMultiInputLstmPolicy.__name__
        == "HybridRecurrentMultiInputActorCriticPolicy"
    )
    assert MlpLstmPolicy.__name__ == "MaskableRecurrentActorCriticPolicy"
    assert CnnLstmPolicy.__name__ == "MaskableRecurrentActorCriticCnnPolicy"
    assert (
        MultiInputLstmPolicy.__name__ == "MaskableRecurrentMultiInputActorCriticPolicy"
    )
