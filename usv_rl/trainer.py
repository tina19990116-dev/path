"""
Training utilities for LSTM-SAC with auxiliary physics supervision.

Losses operate on masked sequences expressed in SI units.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn, amp

from .models import CriticWithAuxLSTM, LSTMState


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """Polyak average target parameters."""

    for target_param, src_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * src_param.data + (1.0 - tau) * target_param.data)


@dataclass
class TrainerConfig:
    batch_size: int
    gamma: float
    tau: float
    actor_lr: float
    critic_lr: float
    alpha_lr: float
    target_entropy: float
    burn_in: int
    use_amp: bool
    u_max: float
    aux_weight: float
    grad_clip: float = 1.0


class Trainer:
    """Implements SAC updates with masked recurrent sequences."""

    def __init__(
        self,
        actor: nn.Module,
        critic: CriticWithAuxLSTM,
        target_critic: CriticWithAuxLSTM,
        buffer,
        cfg: TrainerConfig,
        device: torch.device,
        writer=None,
    ) -> None:
        self.actor = actor
        self.critic = critic
        self.target_critic = target_critic
        self.buffer = buffer
        self.cfg = cfg
        self.device = device
        self.writer = writer

        self.actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)
        self.log_alpha = torch.nn.Parameter(torch.zeros(1, device=device))
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)
        self.amp_device = "cuda" if (cfg.use_amp and torch.cuda.is_available() and device.type == "cuda") else "cpu"
        # torch.amp GradScaler may be unavailable in older versions; fall back to cuda.amp.
        GradScaler = getattr(amp, "GradScaler", torch.cuda.amp.GradScaler)
        self.scaler = GradScaler(enabled=cfg.use_amp)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def _masked_mean(self, tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.ndim < tensor.ndim:
            expand = mask
            while expand.ndim < tensor.ndim:
                expand = expand.unsqueeze(-1)
        else:
            expand = mask
        num = (tensor * expand).sum()
        denom = expand.sum().clamp_min(1.0)
        return num / denom

    def train_step(self, global_step: int) -> Dict[str, float]:
        batch = self.buffer.sample(self.cfg.batch_size, self.device)
        obs = batch["obs"]
        act = batch["act"]
        rew = batch["rew"]
        done = batch["done"]
        mask = batch["obs_mask"]
        attn = batch["attn_mask"]
        nu_c = batch["nu_c_B"]
        rew_terms = batch.get("rew_terms")

        obs_curr = obs[:, :-1]
        obs_next = obs[:, 1:]
        act_curr = act[:, :-1]
        rew_curr = rew[:, :-1]
        done_curr = done[:, :-1]
        mask_curr = mask[:, :-1]
        mask_next = mask[:, 1:]
        attn_curr = attn[:, :-1]
        attn_next = attn[:, 1:]
        nu_c_curr = nu_c[:, :-1]

        burn_in = self.cfg.burn_in
        autocast_fn = amp.autocast if hasattr(amp, "autocast") else torch.cuda.amp.autocast
        with autocast_fn(device_type=self.amp_device, enabled=self.cfg.use_amp):
            pi_actions_full, pi_logp_full, _, _ = self.actor(obs, mask, attn, burn_in)
            total_steps = pi_actions_full.shape[1] - 1
            if total_steps <= 0:
                raise RuntimeError("Sequence length must satisfy L > burn_in + 1 for training.")
            horizon_candidates = [
                total_steps,
                act_curr.shape[1] - burn_in,
                mask_curr.shape[1] - burn_in,
                rew_curr.shape[1] - burn_in,
                done_curr.shape[1] - burn_in,
            ]
            base_T = int(max(1, min(horizon_candidates)))
            pi_curr = pi_actions_full[:, :base_T]
            pi_next = pi_actions_full[:, 1 : base_T + 1]
            logp_curr = pi_logp_full[:, :base_T]
            logp_next = pi_logp_full[:, 1 : base_T + 1]

            q1, q2, _ = self.critic.q_forward(obs_curr, act_curr, mask_curr, attn_curr, burn_in)
            q1 = q1.squeeze(-1)
            q2 = q2.squeeze(-1)
            q_horizon = q1.shape[1]
            T = int(max(1, min(base_T, q_horizon, logp_curr.shape[1])))
            pi_curr = pi_curr[:, :T]
            pi_next = pi_next[:, :T]
            logp_curr = logp_curr[:, :T]
            logp_next = logp_next[:, :T]
            pi_seq = torch.zeros_like(act_curr)
            pi_seq[:, burn_in : burn_in + T] = pi_curr
            pi_next_seq = torch.zeros_like(act_curr)
            pi_next_seq[:, burn_in : burn_in + T] = pi_next
            mask_train = mask_curr[:, burn_in : burn_in + T]
            if mask_train.sum() <= 0:
                raise RuntimeError("No valid steps available after burn-in; reduce burn-in or increase seq length.")
            rew_train = rew_curr[:, burn_in : burn_in + T]
            done_train = done_curr[:, burn_in : burn_in + T]
            q1 = q1[:, :T]
            q2 = q2[:, :T]

            with torch.no_grad():
                target_q1, target_q2, _ = self.target_critic.q_forward(obs_next, pi_next_seq, mask_next, attn_next, burn_in)
                target_q = torch.min(target_q1, target_q2).squeeze(-1)[:, :T]
                backup = rew_train + self.cfg.gamma * (1.0 - done_train) * (target_q - self.alpha.detach() * logp_next)

            td_error1 = q1[:, :T] - backup
            td_error2 = q2[:, :T] - backup
            l_q1 = self._masked_mean(td_error1.pow(2), mask_train)
            l_q2 = self._masked_mean(td_error2.pow(2), mask_train)
            l_q = 0.5 * (l_q1 + l_q2)

            aux_pred, _ = self.critic.aux_forward(obs_curr, mask_curr, attn_curr, burn_in)
            aux_pred = aux_pred[:, :T]
            nu_target = (nu_c_curr[:, burn_in : burn_in + T] / self.cfg.u_max).clamp(-1.0, 1.0)
            aux_loss = self._masked_mean((aux_pred - nu_target).pow(2), mask_train)
            critic_loss = l_q + self.cfg.aux_weight * aux_loss

        self.critic_opt.zero_grad()
        self.scaler.scale(critic_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.grad_clip)
        self.scaler.step(self.critic_opt)
        self.scaler.update()

        with autocast_fn(device_type=self.amp_device, enabled=self.cfg.use_amp):
            q_pi, _, _ = self.critic.q_forward(obs_curr, pi_seq, mask_curr, attn_curr, burn_in)
            q_pi = q_pi.squeeze(-1)[:, :T]
            actor_loss = self._masked_mean(self.alpha.detach() * logp_curr - q_pi, mask_train)

        self.actor_opt.zero_grad()
        self.scaler.scale(actor_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.grad_clip)
        self.scaler.step(self.actor_opt)
        self.scaler.update()

        alpha_loss = -(self.log_alpha * (logp_curr + self.cfg.target_entropy).detach())
        alpha_loss = self._masked_mean(alpha_loss, mask_train)
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        soft_update(self.target_critic, self.critic, self.cfg.tau)

        metrics = {
            "L_Q1": float(l_q1.item()),
            "L_Q2": float(l_q2.item()),
            "L_Q": float(l_q.item()),
            "L_pi": float(actor_loss.item()),
            "L_alpha": float(alpha_loss.item()),
            "alpha": float(self.alpha.item()),
            "L_aux": float(aux_loss.item()),
        }
        if rew_terms is not None:
            terms_slice = rew_terms[:, burn_in : burn_in + T]
            mask_sum = mask_train.sum().clamp_min(1.0)
            weighted = (terms_slice * mask_train.unsqueeze(-1)).sum(dim=(0, 1)) / mask_sum
            names = ["r_shape", "r_coll", "r_energy", "r_smooth", "r_time", "r_goal", "power"]
            for idx, name in enumerate(names):
                metrics[f"reward/{name}"] = float(weighted[idx].item())
        if self.writer:
            for key, val in metrics.items():
                self.writer.add_scalar(f"train/{key}", val, global_step)
        return metrics
