"""Top-level package for the sb3x extension namespace."""

from importlib.metadata import PackageNotFoundError, version

from .dqn_boltzmann import BoltzmannDQN
from .ppo_hybrid_action import HybridActionPPO
from .ppo_hybrid_recurrent import HybridRecurrentPPO
from .ppo_mask_hybrid_action import MaskableHybridActionPPO
from .ppo_mask_hybrid_recurrent import MaskableHybridRecurrentPPO
from .ppo_mask_recurrent import MaskableRecurrentPPO
from .sac_discrete import DiscreteSAC
from .sac_hybrid_action import HybridActionSAC
from .sac_mask_hybrid_action import MaskableHybridActionSAC

try:
    __version__ = version("sb3x")
except PackageNotFoundError:  # pragma: no cover - fallback for direct source use
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    "BoltzmannDQN",
    "DiscreteSAC",
    "HybridActionPPO",
    "HybridRecurrentPPO",
    "HybridActionSAC",
    "MaskableHybridActionPPO",
    "MaskableHybridActionSAC",
    "MaskableHybridRecurrentPPO",
    "MaskableRecurrentPPO",
]
