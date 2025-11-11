"""
usv_rl
======

Reinforcement-learning toolkit for unmanned surface vehicle (USV) path
planning with sequential Soft Actor-Critic (SAC), attention-augmented LSTMs,
and auxiliary physics supervision.  All modules target GPU execution on CUDA
devices (tested for RTX3090, CUDA 12.4).
"""

from importlib.metadata import version, PackageNotFoundError

__all__ = [
    "version",
]

try:
    __version__ = version("usv_rl")
except PackageNotFoundError:  # pragma: no cover - local source tree
    __version__ = "0.0.0"

