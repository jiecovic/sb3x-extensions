"""Smoke tests for the minimal package scaffold."""

from importlib.metadata import version

import sb3x
from sb3x import MaskableRecurrentPPO
from sb3x.ppo_mask_recurrent import (
    CnnLstmPolicy,
    MlpLstmPolicy,
    MultiInputLstmPolicy,
)


def test_import_smoke() -> None:
    """The top-level package exposes the current algorithm namespace cleanly."""
    assert sb3x.__version__ == version("sb3x")
    assert sb3x.MaskableRecurrentPPO is MaskableRecurrentPPO
    assert MlpLstmPolicy.__name__ == "MaskableRecurrentActorCriticPolicy"
    assert CnnLstmPolicy.__name__ == "MaskableRecurrentActorCriticCnnPolicy"
    assert (
        MultiInputLstmPolicy.__name__ == "MaskableRecurrentMultiInputActorCriticPolicy"
    )
