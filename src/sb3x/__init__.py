"""Top-level package for the sb3x extension namespace."""

from importlib.metadata import PackageNotFoundError, version

from .maskable_recurrent import MaskableRecurrentPPO

try:
    __version__ = version("sb3x")
except PackageNotFoundError:  # pragma: no cover - fallback for direct source use
    __version__ = "0.0.0"

__all__ = ["__version__", "MaskableRecurrentPPO"]
