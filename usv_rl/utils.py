"""
Utility helpers for USV reinforcement learning.

All functions operate on SI-units unless noted otherwise.  Distances are
expressed in metres (m), velocities in metres per second (m·s⁻¹), forces in
newtons (N), and torques in newton-metres (N·m).
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import numpy as np
import torch

try:  # YAML is used for config overrides.
    import yaml
except ImportError as exc:  # pragma: no cover - dependency guard.
    raise ImportError("PyYAML is required to load configuration files.") from exc


LOGGER = logging.getLogger(__name__)

OBS_TOKEN_DIM = 6


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and torch RNGs for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name: Optional[str] = None) -> torch.device:
    """Return a CUDA device when available, else CPU."""

    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_yaml_config(path: str | os.PathLike) -> Dict[str, Any]:
    """Load a YAML configuration file."""

    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data


def merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge dictionaries."""

    out = dict(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = merge_dict(out[key], val)
        else:
            out[key] = val
    return out


def build_obs_layout(top_n: int, include_wind: bool) -> Tuple[Dict[str, slice], int]:
    """
    Construct observation slice indices.

    Returns
    -------
    slices : dict
        Slice indices for each semantic block.
    total_dim : int
        Total observation dimension ``D`` for tensors shaped ``[B, L, D]``.
    """

    idx = 0
    slices: Dict[str, slice] = {}

    blocks = [
        ("self", 5),
        ("goal", 3),
        ("current", 2),
    ]
    if include_wind:
        blocks.append(("wind", 2))
    blocks.append(("obstacles", top_n * OBS_TOKEN_DIM))
    blocks.append(("sdf_stub", 0))

    for name, width in blocks:
        start = idx
        idx += width
        slices[name] = slice(start, idx)

    return slices, idx


class Normalizer:
    """
    Running feature normalizer with Welford-style updates.

    Parameters
    ----------
    shape : tuple
        Shape of the feature vector.
    eps : float
        Numerical stabilizer for statistics.
    device : torch.device
        Storage device.
    """

    def __init__(self, shape: Tuple[int, ...], eps: float = 1e-5, device: torch.device | str = "cpu") -> None:
        self.eps = eps
        self.device = torch.device(device)
        self.mean = torch.zeros(shape, device=self.device)
        self.var = torch.ones(shape, device=self.device)
        self.count = torch.tensor(eps, device=self.device)
        self._frozen = False

    @torch.no_grad()
    def update(self, batch: torch.Tensor, mask: Optional[torch.Tensor] = None) -> None:
        """Update running statistics with an optional mask."""

        if self._frozen:
            return
        data = batch.to(self.device)
        if mask is None:
            weight = torch.ones_like(data[..., :1])
        else:
            w = mask.to(data.dtype)
            while w.ndim < data.ndim:
                w = w.unsqueeze(-1)
            weight = w

        flat_data = data.reshape(-1, *self.mean.shape)
        flat_weight = weight.reshape(-1, *([1] * len(self.mean.shape)))
        batch_count = flat_weight.sum()
        if batch_count <= 0:
            return

        weighted_sum = (flat_data * flat_weight).sum(dim=0)
        batch_mean = weighted_sum / batch_count
        centered = (flat_data - batch_mean) * torch.sqrt(flat_weight)
        batch_var = (centered**2).sum(dim=0) / batch_count

        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        new_var = (m_a + m_b + delta**2 * self.count * batch_count / total) / total

        self.mean.copy_(new_mean)
        self.var.copy_(new_var.clamp_min(self.eps))
        self.count = total

    def normalize(self, batch: torch.Tensor) -> torch.Tensor:
        """Normalize a batch using current statistics."""

        std = torch.sqrt(self.var + self.eps)
        return (batch - self.mean) / std

    def denormalize(self, batch: torch.Tensor) -> torch.Tensor:
        """Invert normalization."""

        std = torch.sqrt(self.var + self.eps)
        return batch * std + self.mean

    def freeze(self) -> None:
        """Stop updating statistics."""

        self._frozen = True

    def unfreeze(self) -> None:
        """Resume updates."""

        self._frozen = False

    def to(self, device: torch.device | str) -> "Normalizer":
        """Move statistics to a new device."""

        dev = torch.device(device)
        self.device = dev
        self.mean = self.mean.to(dev)
        self.var = self.var.to(dev)
        self.count = self.count.to(dev)
        return self


def catch_nan(tensor: torch.Tensor, name: str) -> torch.Tensor:
    """Raise an error if ``tensor`` contains NaNs."""

    if torch.isnan(tensor).any():
        raise ValueError(f"Detected NaN in {name}")
    return tensor


@contextmanager
def oom_guard(tag: str):
    """
    Context manager that clears CUDA cache when an out-of-memory occurs.

    Parameters
    ----------
    tag : str
        Human-readable identifier for the guarded block.
    """

    try:
        yield
    except RuntimeError as exc:
        if "CUDA out of memory" in str(exc):
            LOGGER.warning("OOM detected during %s. Clearing cache.", tag)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        raise


def max_batch_finder(run_fn: Callable[[int], None], start: int, growth: float = 1.2) -> int:
    """
    Increase batch size until OOM, then reduce by 10 %.

    Parameters
    ----------
    run_fn : callable
        Function that executes a trial run for a proposed batch size.
    start : int
        Initial batch size candidate.
    growth : float
        Multiplicative growth per iteration.
    """

    batch = start
    best = start
    while True:
        try:
            with oom_guard("auto-batch"):
                run_fn(batch)
        except RuntimeError as exc:
            if "out of memory" in str(exc):
                best = max(1, int(batch * 0.8))
                LOGGER.info("Auto-batch OOM at %d, backing off to %d.", batch, best)
                return best
            raise
        best = batch
        batch = int(batch * growth) + 1
        capacity_cap = 10_000
        if batch > capacity_cap:
            LOGGER.warning("Auto-batch search exceeded sentinel cap; using best=%d.", best)
            return max(1, int(best * 0.9))


def save_checkpoint(
    path: str | os.PathLike,
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    target_critic: torch.nn.Module,
    optimizers: Dict[str, torch.optim.Optimizer],
    normalizers: Dict[str, Normalizer],
    global_step: int,
    config: Dict[str, Any],
) -> None:
    """Serialize training state."""

    payload = {
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "target_critic": target_critic.state_dict(),
        "optimizers": {k: opt.state_dict() for k, opt in optimizers.items()},
        "normalizers": {k: {"mean": norm.mean, "var": norm.var, "count": norm.count} for k, norm in normalizers.items()},
        "global_step": global_step,
        "config": config,
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str | os.PathLike,
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    target_critic: torch.nn.Module,
    optimizers: Dict[str, torch.optim.Optimizer],
    normalizers: Dict[str, Normalizer],
    map_location: Optional[str] = None,
) -> Dict[str, Any]:
    """Load checkpoint and restore state dictionaries."""

    payload = torch.load(path, map_location=map_location)
    actor.load_state_dict(payload["actor"])
    critic.load_state_dict(payload["critic"])
    target_critic.load_state_dict(payload["target_critic"])
    for name, opt in optimizers.items():
        if name in payload["optimizers"]:
            opt.load_state_dict(payload["optimizers"][name])
    for name, norm in normalizers.items():
        if name in payload["normalizers"]:
            stats = payload["normalizers"][name]
            norm.mean.copy_(stats["mean"])
            norm.var.copy_(stats["var"])
            norm.count = stats["count"]
    return payload


def ensure_dir(path: str | os.PathLike) -> Path:
    """Create directory if it does not exist."""

    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target
