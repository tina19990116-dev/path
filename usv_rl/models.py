"""
Policy and value networks for LSTM-SAC with obstacle attention.

States use SI units (metres, seconds, radians).  Actions are tanh-squashed
body-frame velocity references ``ν_d = [u_d, v_d, r_d]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.distributions import Normal

from .attention import MultiHeadAttnMasked
from .utils import OBS_TOKEN_DIM


def orthogonal_init(module: nn.Module, gain: float = 1.0) -> None:
    """Apply orthogonal initialization to Linear layers."""

    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


@dataclass
class LSTMState:
    """Light-weight container for LSTM hidden + cell."""

    h: torch.Tensor
    c: torch.Tensor


def masked_lstm_roll(
    lstm: nn.LSTM,
    inputs: torch.Tensor,
    mask: torch.Tensor,
    burn_in: int,
    state: Optional[LSTMState] = None,
) -> Tuple[torch.Tensor, LSTMState]:
    """
    Roll an LSTM over a masked sequence.

    Parameters
    ----------
    lstm : nn.LSTM
        Recurrent module.
    inputs : torch.Tensor
        Sequence ``[B, L, D]``.
    mask : torch.Tensor
        Valid-step mask ``[B, L]``.
    burn_in : int
        Number of steps used only for state warm-up.
    state : LSTMState, optional
        Initial recurrent state.
    """

    batch, seq, _ = inputs.shape
    h, c = (
        (state.h, state.c)
        if state is not None
        else (
            torch.zeros(lstm.num_layers, batch, lstm.hidden_size, device=inputs.device),
            torch.zeros(lstm.num_layers, batch, lstm.hidden_size, device=inputs.device),
        )
    )
    outputs = []
    mask = mask.to(inputs.device)

    prev_h, prev_c = h, c
    for t in range(seq):
        inp = inputs[:, t : t + 1]
        out, (h, c) = lstm(inp, (h, c))
        valid = mask[:, t].view(1, batch, 1).bool()
        h = torch.where(valid, h, prev_h)
        c = torch.where(valid, c, prev_c)
        prev_h, prev_c = h, c
        if t >= burn_in:
            outputs.append(out)

    if outputs:
        stacked = torch.cat(outputs, dim=1)
    else:
        stacked = torch.zeros(batch, 0, lstm.hidden_size, device=inputs.device)
    return stacked, LSTMState(h, c)


class ScalarEncoder(nn.Module):
    """Simple MLP for scalar observation blocks."""

    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.apply(orthogonal_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActorLSTM(nn.Module):
    """Attention-enhanced actor producing Gaussian actions."""

    def __init__(
        self,
        obs_dim: int,
        action_limits: torch.Tensor,
        hidden_dim: int = 256,
        top_n: int = 16,
        include_attention: bool = True,
        token_dim: int = OBS_TOKEN_DIM,
        attn_heads: int = 4,
        ff_only: bool = False,
    ) -> None:
        super().__init__()
        token_total = top_n * token_dim
        scalar_dim = obs_dim - token_total
        self.action_limits = action_limits
        self.top_n = top_n
        self.use_attention = include_attention
        self.token_dim = token_dim
        self.ff_only = ff_only

        self.scalar_enc = ScalarEncoder(scalar_dim, hidden_dim)
        if include_attention:
            self.token_proj = nn.Linear(token_dim, hidden_dim)
            self.attn = MultiHeadAttnMasked(hidden_dim, hidden_dim, attn_heads)
            attn_out = hidden_dim
        else:
            self.token_proj = nn.Linear(token_dim, hidden_dim)
            attn_out = hidden_dim

        self.feature_dim = hidden_dim + attn_out
        if not ff_only:
            self.lstm = nn.LSTM(self.feature_dim, hidden_dim, batch_first=True)
        else:
            self.ff_body = nn.Sequential(
                nn.Linear(self.feature_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
            )
        self.head = nn.Linear(hidden_dim, 2 * action_limits.numel())
        self.log_std_min = -5.0
        self.log_std_max = 2.0
        self.apply(orthogonal_init)

    def _encode_tokens(self, tokens: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        b, l, _ = tokens.shape
        tokens = tokens.view(b, l, self.top_n, self.token_dim)
        projected = torch.tanh(self.token_proj(tokens))
        if self.use_attention:
            # Query from scalar encoder will be fused later.
            return projected, None
        weight = mask.unsqueeze(-1).to(tokens.dtype)
        summed = (projected * weight).sum(dim=2) / (weight.sum(dim=2).clamp_min(1.0))
        return summed, None

    def forward(
        self,
        obs: torch.Tensor,
        obs_mask: torch.Tensor,
        attn_mask: torch.Tensor,
        burn_in: int,
        hidden: Optional[LSTMState] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, LSTMState, Optional[torch.Tensor]]:
        b, l, d = obs.shape
        token_slice = slice(d - self.top_n * self.token_dim, d)
        scalar = obs[:, :, : token_slice.start]
        tokens = obs[:, :, token_slice]

        scalar_feat = self.scalar_enc(scalar)
        token_feat, _ = self._encode_tokens(tokens, attn_mask)
        if self.use_attention:
            query = scalar_feat.unsqueeze(2)
            context, weights = self.attn(query, token_feat, attn_mask)
        else:
            context = token_feat
            weights = None

        fused = torch.cat([scalar_feat, context], dim=-1)
        if self.ff_only:
            masked = fused * obs_mask.unsqueeze(-1)
            ff_out = masked[:, burn_in:]
            logits = self.head(self.ff_body(ff_out))
            next_state = hidden
        else:
            lstm_out, next_state = masked_lstm_roll(self.lstm, fused, obs_mask, burn_in, hidden)
        logits = self.head(lstm_out)
        mu, log_std = torch.chunk(logits, 2, dim=-1)
        log_std = torch.clamp(torch.nan_to_num(log_std), self.log_std_min, self.log_std_max)
        mu = torch.nan_to_num(mu)

        if deterministic:
            pre_tanh = mu
        else:
            mu_fp32 = mu.to(torch.float32)
            log_std_fp32 = log_std.to(torch.float32)
            dist = Normal(mu_fp32, log_std_fp32.exp())
            pre_tanh = dist.rsample().to(mu.dtype)

        action = torch.tanh(pre_tanh)
        log_prob = None
        if not deterministic:
            log_prob = (
                dist.log_prob(pre_tanh.to(torch.float32))
                - torch.log1p(-action.to(torch.float32).pow(2) + 1e-6)
            ).sum(-1).to(mu.dtype)
        scaled = action * self.action_limits
        return scaled, log_prob, next_state, weights


class CriticWithAuxLSTM(nn.Module):
    """Twin-Q critic with auxiliary drift prediction."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        top_n: int = 16,
        include_attention: bool = True,
        token_dim: int = OBS_TOKEN_DIM,
        attn_heads: int = 4,
        ff_only: bool = False,
    ) -> None:
        super().__init__()
        token_total = top_n * token_dim
        scalar_dim = obs_dim - token_total
        self.top_n = top_n
        self.use_attention = include_attention
        self.token_dim = token_dim

        self.scalar_enc = ScalarEncoder(scalar_dim, hidden_dim)
        self.token_proj = nn.Linear(token_dim, hidden_dim)
        if include_attention:
            self.attn = MultiHeadAttnMasked(hidden_dim, hidden_dim, attn_heads)
            attn_out = hidden_dim
        else:
            attn_out = hidden_dim

        self.feature_dim = hidden_dim + attn_out
        self.ff_only = ff_only
        if not ff_only:
            self.lstm_q = nn.LSTM(self.feature_dim + action_dim, hidden_dim, batch_first=True)
            self.lstm_aux = nn.LSTM(self.feature_dim, hidden_dim, batch_first=True)
        else:
            self.ff_body = nn.Sequential(
                nn.Linear(self.feature_dim + action_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
            )
            self.ff_aux = nn.Sequential(
                nn.Linear(self.feature_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
            )
        self.q1 = nn.Linear(hidden_dim, 1)
        self.q2 = nn.Linear(hidden_dim, 1)
        self.aux_head = nn.Linear(hidden_dim, 2)  # ν_c^B / U_max (u_c, v_c)
        self.apply(orthogonal_init)

    def _encode(self, obs: torch.Tensor, obs_mask: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        b, l, d = obs.shape
        token_slice = slice(d - self.top_n * self.token_dim, d)
        scalar = obs[:, :, : token_slice.start]
        tokens = obs[:, :, token_slice].view(b, l, self.top_n, self.token_dim)
        scalar_feat = self.scalar_enc(scalar)
        token_feat = torch.tanh(self.token_proj(tokens))
        if self.use_attention:
            context, _ = self.attn(scalar_feat.unsqueeze(2), token_feat, attn_mask)
        else:
            weight = attn_mask.unsqueeze(-1).to(token_feat.dtype)
            context = (token_feat * weight).sum(dim=2) / weight.sum(dim=2).clamp_min(1.0)
        return torch.cat([scalar_feat, context], dim=-1)

    def q_forward(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        obs_mask: torch.Tensor,
        attn_mask: torch.Tensor,
        burn_in: int,
        state: Optional[LSTMState] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, LSTMState]:
        features = self._encode(obs, obs_mask, attn_mask)
        fused = torch.cat([features, act], dim=-1)
        if self.ff_only:
            masked = fused * obs_mask.unsqueeze(-1)
            ff_out = self.ff_body(masked[:, burn_in:])
            q1 = self.q1(ff_out)
            q2 = self.q2(ff_out)
            next_state = state
        else:
            lstm_out, next_state = masked_lstm_roll(self.lstm_q, fused, obs_mask, burn_in, state)
            q1 = self.q1(lstm_out)
            q2 = self.q2(lstm_out)
        return q1, q2, next_state

    def aux_forward(
        self,
        obs: torch.Tensor,
        obs_mask: torch.Tensor,
        attn_mask: torch.Tensor,
        burn_in: int,
        state: Optional[LSTMState] = None,
    ) -> Tuple[torch.Tensor, LSTMState]:
        features = self._encode(obs, obs_mask, attn_mask)
        if self.ff_only:
            masked = features * obs_mask.unsqueeze(-1)
            ff_out = self.ff_aux(masked[:, burn_in:])
            drift = torch.tanh(self.aux_head(ff_out))
            next_state = state
        else:
            lstm_out, next_state = masked_lstm_roll(self.lstm_aux, features, obs_mask, burn_in, state)
            drift = torch.tanh(self.aux_head(lstm_out))
        return drift, next_state
