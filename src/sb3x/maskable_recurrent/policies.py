"""Policy aliases for ``MaskableRecurrentPPO``.

This stays intentionally close to ``sb3_contrib.ppo_recurrent.policies`` so
the first local algorithm copy remains easy to diff against upstream.
"""

from sb3_contrib.common.recurrent.policies import (
    RecurrentActorCriticCnnPolicy,
    RecurrentActorCriticPolicy,
    RecurrentMultiInputActorCriticPolicy,
)

MlpLstmPolicy = RecurrentActorCriticPolicy
CnnLstmPolicy = RecurrentActorCriticCnnPolicy
MultiInputLstmPolicy = RecurrentMultiInputActorCriticPolicy

__all__ = ["CnnLstmPolicy", "MlpLstmPolicy", "MultiInputLstmPolicy"]
