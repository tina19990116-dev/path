"""
Sequential replay buffer with optional GPU-resident storage.

Each entry stores a padded sequence of length ``L`` (default 64) matching the
observation layout.  Units follow SI (m, s, N).
"""

from __future__ import annotations

from typing import Dict, Optional

import torch


class ReplayBufferSeq:
    """Circular buffer storing fixed-length sequences."""

    def __init__(
        self,
        capacity_steps: int,
        seq_len: int,
        device: torch.device,
        gpu_storage: bool = True,
    ) -> None:
        self.seq_len = seq_len
        self.capacity_steps = capacity_steps
        self.max_sequences = max(1, capacity_steps // seq_len)
        self.device = device
        self.gpu_storage = gpu_storage
        self.storage: Dict[str, torch.Tensor] = {}
        self.ptr = 0
        self.full = False

    def __len__(self) -> int:
        return self.max_sequences if self.full else self.ptr

    def _allocate(self, sample: Dict[str, torch.Tensor]) -> None:
        dst_device = self.device if self.gpu_storage else torch.device("cpu")
        for key, tensor in sample.items():
            shape = (self.max_sequences,) + tensor.shape
            dtype = tensor.dtype
            storage = torch.zeros(shape, dtype=dtype, device=dst_device)
            if not self.gpu_storage and dtype.is_floating_point:
                storage = storage.pin_memory()
            self.storage[key] = storage

    def add(self, sample: Dict[str, torch.Tensor]) -> None:
        """Insert a new sequence sample."""

        if not self.storage:
            self._allocate(sample)
        idx = self.ptr
        for key, tensor in sample.items():
            target = self.storage[key]
            target[idx].copy_(tensor.to(target.device, non_blocking=True))
        self.ptr = (self.ptr + 1) % self.max_sequences
        if self.ptr == 0:
            self.full = True

    def sample(self, batch_size: int, device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
        """Random batch of sequences moved to ``device`` (default CUDA)."""

        if len(self) == 0:
            raise RuntimeError("Replay buffer is empty.")
        device = device or self.device
        max_idx = len(self)
        idx = torch.randint(0, max_idx, (batch_size,), device=device)
        idx_cpu = idx.cpu()
        batch = {}
        for key, tensor in self.storage.items():
            if tensor.device.type == "cpu":
                data = tensor[idx_cpu]
            else:
                data = tensor[idx]
            data = data.to(device, non_blocking=True)
            batch[key] = data
        return batch
