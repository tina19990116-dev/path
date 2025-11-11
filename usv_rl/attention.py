"""
Attention layers tailored for obstacle tokens.

Shapes follow the convention:

* ``q`` in ``ℝ^{B×L×1×d_model}``
* ``kv`` in ``ℝ^{B×L×N×d_model}``
* ``mask`` in ``{0,1}^{B×L×N}``

All tensors operate in SI units (positions in metres, velocities in m·s⁻¹).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn


class MultiHeadAttnMasked(nn.Module):
    """Multi-head attention with per-token padding masks."""

    def __init__(self, d_in: int, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_in, d_model)
        self.k_proj = nn.Linear(d_in, d_model)
        self.v_proj = nn.Linear(d_in, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.register_buffer("last_weights", None, persistent=False)

    def _reshape(self, tensor: torch.Tensor) -> torch.Tensor:
        b, l, n, _ = tensor.shape
        tensor = tensor.view(b, l, n, self.n_heads, self.head_dim)
        return tensor.transpose(2, 3)  # B, L, heads, N, head_dim

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        q : torch.Tensor
            Query tensor ``[B, L, 1, d_in]``.
        kv : torch.Tensor
            Key/value tensor ``[B, L, N, d_in]``.
        mask : torch.Tensor, optional
            Binary mask where ``0`` denotes padding.
        """

        q_proj = self.q_proj(q).view(q.size(0), q.size(1), 1, self.n_heads, self.head_dim)
        k_proj = self._reshape(self.k_proj(kv))
        v_proj = self._reshape(self.v_proj(kv))

        q_proj = q_proj.transpose(2, 3)  # B, L, heads, 1, head_dim
        scores = torch.matmul(q_proj, k_proj.transpose(-2, -1)) / (self.head_dim**0.5)  # B,L,heads,1,N
        if mask is not None:
            mask_bool = mask.to(dtype=torch.bool)
            scores = scores.masked_fill(mask_bool[:, :, None, None, :] == 0, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(weights, v_proj)  # B,L,heads,1,dim
        context = context.transpose(2, 3).contiguous().view(q.size(0), q.size(1), 1, self.d_model)
        context = self.out_proj(context).squeeze(2)

        self.last_weights = weights.squeeze(3).detach()
        return context, self.last_weights
