"""Smoke tests for the minimal package scaffold."""

from importlib.metadata import version

import sb3x
from sb3x import (
    BoltzmannDQN,
    DiscreteSAC,
    HybridActionSAC,
    MaskableHybridActionPPO,
    MaskableHybridActionSAC,
    MaskableHybridRecurrentPPO,
)
from sb3x.common.hybrid_action import MaskableHybridActionDistribution
from sb3x.ppo_hybrid_action import (
    CnnPolicy,
    HybridActionPPO,
    MlpPolicy,
    MultiInputPolicy,
)
from sb3x.ppo_hybrid_recurrent import (
    CnnLstmPolicy as HybridRecurrentCnnLstmPolicy,
)
from sb3x.ppo_hybrid_recurrent import HybridRecurrentPPO
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
from sb3x.ppo_mask_hybrid_action.distributions import (
    MaskableHybridActionDistribution as LegacyMaskableHybridActionDistribution,
)
from sb3x.ppo_mask_hybrid_recurrent import (
    CnnLstmPolicy as MaskableHybridRecurrentCnnLstmPolicy,
)
from sb3x.ppo_mask_hybrid_recurrent import (
    MlpLstmPolicy as MaskableHybridRecurrentMlpLstmPolicy,
)
from sb3x.ppo_mask_hybrid_recurrent import (
    MultiInputLstmPolicy as MaskableHybridRecurrentMultiInputLstmPolicy,
)


def test_import_smoke() -> None:
    """The top-level package exposes the current algorithm namespace cleanly."""
    assert sb3x.__version__ == version("sb3x")
    assert sb3x.BoltzmannDQN is BoltzmannDQN
    assert sb3x.DiscreteSAC is DiscreteSAC
    assert sb3x.HybridActionSAC is HybridActionSAC
    assert sb3x.MaskableHybridActionPPO is MaskableHybridActionPPO
    assert sb3x.MaskableHybridActionSAC is MaskableHybridActionSAC
    assert sb3x.MaskableHybridRecurrentPPO is MaskableHybridRecurrentPPO
    assert not hasattr(sb3x, "HybridActionPPO")
    assert not hasattr(sb3x, "HybridRecurrentPPO")
    assert not hasattr(sb3x, "MaskableRecurrentPPO")
    assert HybridActionPPO.__name__ == "HybridActionPPO"
    assert HybridRecurrentPPO.__name__ == "HybridRecurrentPPO"
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
    assert (
        MaskableHybridRecurrentMlpLstmPolicy.__name__
        == "MaskableHybridRecurrentActorCriticPolicy"
    )
    assert (
        MaskableHybridRecurrentCnnLstmPolicy.__name__
        == "MaskableHybridRecurrentActorCriticCnnPolicy"
    )
    assert (
        MaskableHybridRecurrentMultiInputLstmPolicy.__name__
        == "MaskableHybridRecurrentMultiInputActorCriticPolicy"
    )
    assert LegacyMaskableHybridActionDistribution is MaskableHybridActionDistribution
