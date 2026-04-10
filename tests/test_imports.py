"""Smoke tests for the minimal package scaffold."""

from importlib.metadata import version

import sb3x
from sb3x import MaskableRecurrentPPO
from sb3x.maskable_recurrent import (
    CnnLstmPolicy,
    MlpLstmPolicy,
    MultiInputLstmPolicy,
)


def test_import_smoke() -> None:
    """The top-level package exposes the current algorithm namespace cleanly."""
    assert sb3x.__version__ == version("sb3x")
    assert sb3x.MaskableRecurrentPPO is MaskableRecurrentPPO
    assert MlpLstmPolicy.__name__ == "RecurrentActorCriticPolicy"
    assert CnnLstmPolicy.__name__ == "RecurrentActorCriticCnnPolicy"
    assert MultiInputLstmPolicy.__name__ == "RecurrentMultiInputActorCriticPolicy"
