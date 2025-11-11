"""
Parallel rollout utilities coupling the marine simulator and obstacle world.

All kinematics use SI units: metres, seconds, radians.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch

from marine_environment_simulation import MarineEnvironmentSimulation
from usv_obstacles import ObstacleWorld, to_body, to_world

from .reward import compute_step_reward
from .utils import build_obs_layout


@dataclass
class EnvState:
    """Container for USV state."""

    position: np.ndarray
    heading: float
    velocity: np.ndarray
    time: float
    goal: np.ndarray
    prev_action: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    done: bool = False


class SurrogateLowLevel:
    """Converts desired body velocities into control torques."""

    def __init__(self, dt: float, cfg: Dict[str, float]) -> None:
        self.dt = dt
        self.cfg = cfg
        self.nu = np.zeros(3, dtype=float)

    def step(self, nu_cmd: np.ndarray, current_B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        kp = 1.5
        kd = 0.3
        error = nu_cmd - self.nu
        accel = kp * error - kd * self.nu
        self.nu += accel * self.dt
        tau = np.clip(
            np.array([accel[0] * self.cfg["T_max"], accel[1] * self.cfg["T_max"], accel[2] * self.cfg["Mz_max"]]),
            -np.array([self.cfg["T_max"], self.cfg["T_max"], self.cfg["Mz_max"]]),
            np.array([self.cfg["T_max"], self.cfg["T_max"], self.cfg["Mz_max"]]),
        )
        return self.nu.copy(), tau


class VectorizedCollector:
    """Collects sequences from multiple USV environments."""

    def __init__(
        self,
        cfg: Dict[str, float],
        top_n: int,
        include_wind: bool,
        action_limits: np.ndarray,
        device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.top_n = top_n
        self.include_wind = include_wind
        self.device = device
        self.sim = MarineEnvironmentSimulation(map_extent=cfg["L"])
        self.world = ObstacleWorld(L=cfg["L"])
        self.obs_slices, self.obs_dim = build_obs_layout(top_n, include_wind)
        self.dt = getattr(self.world, "dt_RL", 0.2)
        self.action_limits = action_limits.astype(np.float32)
        self.rng = np.random.default_rng(int(cfg.get("seed", 0)))
        self.domain = float(cfg["L"])

    def reset_env(self, seed: int | None = None) -> EnvState:
        start_rect = (0.0, 0.0, 0.1 * self.cfg["L"], 0.1 * self.cfg["L"])
        goal_rect = (0.9 * self.cfg["L"], 0.9 * self.cfg["L"], self.cfg["L"], self.cfg["L"])
        info = self.world.reset(start_rect, goal_rect)
        position = info["p0_E"].astype(float)
        heading = float(info["psi0"])
        goal = info["goal_E"].astype(float)
        return EnvState(position=position, heading=heading, velocity=np.zeros(3, dtype=float), time=0.0, goal=goal)

    def _build_observation(
        self,
        state: EnvState,
        current_B: np.ndarray,
        wind_B: np.ndarray,
        obs_feats: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        goal_vec = state.goal - state.position
        goal_B = to_body(goal_vec, state.heading)
        d_goal = np.linalg.norm(goal_vec)
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        idx = 0

        # Self state
        obs[idx : idx + 5] = np.array(
            [state.velocity[0], state.velocity[1], state.velocity[2], np.sin(state.heading), np.cos(state.heading)],
            dtype=np.float32,
        )
        idx += 5
        # Goal
        obs[idx : idx + 3] = np.array([goal_B[0], goal_B[1], d_goal / self.cfg["L_scale"]], dtype=np.float32)
        idx += 3
        # Current + wind
        obs[idx : idx + 2] = current_B
        idx += 2
        if self.include_wind:
            obs[idx : idx + 2] = wind_B
            idx += 2
        tokens = obs_feats.reshape(-1)
        obs[idx : idx + tokens.size] = tokens
        attn_mask = (np.sum(obs_feats != 0.0, axis=1) > 0).astype(np.float32)
        return obs, attn_mask

    def collect(
        self,
        actor,
        buffer,
        normalizers,
        batch_envs: int,
        store: bool = True,
        occlusion: Dict[str, float] | None = None,
    ):
        """Roll ``batch_envs`` trajectories and optionally skip storage (evaluation)."""

        seq_len = int(self.cfg["seq_len"])
        burn_in = int(self.cfg["burn_in"])
        device = self.device
        episode_infos = []
        for _ in range(batch_envs):
            state = self.reset_env()
            controller = SurrogateLowLevel(self.dt, self.cfg)
            obs_seq: List[np.ndarray] = []
            obs_mask_seq: List[float] = []
            attn_seq: List[np.ndarray] = []
            act_seq: List[np.ndarray] = []
            rew_seq: List[float] = []
            done_seq: List[float] = []
            tau_seq: List[np.ndarray] = []
            nu_seq: List[np.ndarray] = []
            nu_c_seq: List[np.ndarray] = []
            meta_seq: List[np.ndarray] = []
            rew_terms: List[np.ndarray] = []
            prev_action = np.zeros(3, dtype=float)
            prev_goal = np.linalg.norm(state.goal - state.position)
            min_distance = np.inf
            energy_accum = 0.0
            occ_timer = 0
            goal_reached = False
            collision_flag = False
            current_E_fn = lambda x, y, t: self.sim.get_current_E(float(x), float(y), float(t))
            for step in range(seq_len):
                forced_prev = False
                if np.any(state.position < 0.0) or np.any(state.position > self.domain):
                    state.position = np.clip(state.position, 0.0, self.domain)
                    forced_prev = True
                current_B = self.sim.get_current_B(float(state.position[0]), float(state.position[1]), float(state.heading), state.time)
                wind_B = (
                    self.sim.get_wind_B(float(state.position[0]), float(state.position[1]), float(state.heading), state.time)
                    if self.include_wind
                    else np.zeros(2, dtype=float)
                )
                query = self.world.query(state.position, state.heading, N=self.top_n, d_safe=self.cfg["d_safe"])
                obs_feats = query["obs_feats"].astype(np.float32)
                min_distance = float(min(min_distance, np.min(query["mpc_pack"]["d"])))
                if occlusion and step >= int(occlusion.get("t0", 0)):
                    if occ_timer <= 0 and self.rng.random() < float(occlusion.get("prob", 0.5)):
                        occ_timer = int(occlusion.get("len", 1))
                    if occ_timer > 0:
                        obs_feats = np.zeros_like(obs_feats)
                        occ_timer -= 1
                obs_vec, attn_mask = self._build_observation(state, current_B, wind_B, obs_feats)
                obs_seq.append(obs_vec)
                attn_seq.append(attn_mask)
                obs_mask_seq.append(1.0)

                obs_tensor = torch.from_numpy(obs_vec).to(device).view(1, 1, -1)
                mask_tensor = torch.tensor([[1.0]], device=device)
                attn_tensor = torch.from_numpy(attn_mask).to(device).view(1, 1, -1)
                with torch.no_grad():
                    action_scaled, _, _, _ = actor(
                        obs_tensor,
                        mask_tensor,
                        attn_tensor,
                        burn_in=0,
                        deterministic=False,
                    )
                action = action_scaled.squeeze(0).squeeze(0).cpu().numpy()
                act_seq.append(action)

                nu_cmd = action
                nu, tau = controller.step(nu_cmd, current_B)
                nu_seq.append(nu)
                tau_seq.append(tau)
                nu_c_seq.append(current_B[:2])

                vel_E = to_world(nu[:2], state.heading)
                state.position = state.position + vel_E * self.dt
                state.heading = float((state.heading + nu[2] * self.dt + np.pi) % (2 * np.pi) - np.pi)
                state.time += self.dt
                d_goal = np.linalg.norm(state.goal - state.position)

                forced_after = False
                if np.any(state.position < 0.0) or np.any(state.position > self.domain):
                    state.position = np.clip(state.position, 0.0, self.domain)
                    forced_after = True

                distances = query["mpc_pack"]["d"]
                goal_hit = d_goal <= self.cfg["R_goal"]
                collided = query["collision_static"] or query["collision_dyn"]
                out = query["out_of_bounds"] or forced_prev or forced_after
                done = goal_hit or collided or out
                reward, terms = compute_step_reward(
                    prev_goal_dist=prev_goal,
                    goal_dist=d_goal,
                    tau_ctrl=tau,
                    nu_body=nu,
                    nu_current_body=current_B[:2],
                    action_tanh=np.clip(action / self.action_limits, -1.0, 1.0),
                    prev_action_tanh=np.clip(prev_action / self.action_limits, -1.0, 1.0),
                    distances=distances,
                    done=done,
                    cfg=self.cfg,
                )
                if done:
                    terminal_done = True
                    goal_reached = goal_hit
                    collision_flag = collided or out
                prev_goal = d_goal
                prev_action = action
                rew_seq.append(reward)
                rew_terms.append(
                    np.array(
                        [
                            terms["r_shape"],
                            terms["r_coll"],
                            terms["r_energy"],
                            terms["r_smooth"],
                            terms["r_time"],
                            terms["r_goal"],
                            terms["power_tau_nu_rel"],
                        ],
                        dtype=np.float32,
                    )
                )
                energy_accum += abs(terms["power_tau_nu_rel"])
                done_seq.append(1.0 if done else 0.0)

                meta_seq.append(
                    np.array([state.position[0], state.position[1], state.heading, state.time], dtype=np.float32)
                )

                self.world.step(state.time, state.position, state.heading, nu, current_E_fn, self.dt)

                if done:
                    state.done = True
                    break

            # Pad to seq_len
            while len(obs_seq) < seq_len:
                obs_seq.append(np.zeros_like(obs_seq[0]))
                obs_mask_seq.append(0.0)
                attn_seq.append(np.zeros_like(attn_seq[0]))
                act_seq.append(np.zeros_like(act_seq[0]))
                rew_seq.append(0.0)
                rew_terms.append(np.zeros_like(rew_terms[0]))
                done_seq.append(1.0)
                tau_seq.append(np.zeros_like(tau_seq[0]))
                nu_seq.append(np.zeros_like(nu_seq[0]))
                nu_c_seq.append(np.zeros_like(nu_c_seq[0]))
                meta_seq.append(np.zeros_like(meta_seq[0]))

            sample = {
                "obs": torch.from_numpy(np.stack(obs_seq)).float(),
                "obs_mask": torch.from_numpy(np.array(obs_mask_seq)).float(),
                "attn_mask": torch.from_numpy(np.stack(attn_seq)).float(),
                "act": torch.from_numpy(np.stack(act_seq)).float(),
                "rew": torch.from_numpy(np.array(rew_seq)).float(),
                "done": torch.from_numpy(np.array(done_seq)).float(),
                "tau_ctrl": torch.from_numpy(np.stack(tau_seq)).float(),
                "nu": torch.from_numpy(np.stack(nu_seq)).float(),
                "nu_c_B": torch.from_numpy(np.stack(nu_c_seq)).float(),
                "meta": torch.from_numpy(np.stack(meta_seq)).float(),
                "rew_terms": torch.from_numpy(np.stack(rew_terms)).float(),
            }
            if normalizers and "obs" in normalizers:
                normalizers["obs"].update(
                    sample["obs"].unsqueeze(0).to(device),
                    mask=sample["obs_mask"].unsqueeze(0).to(device),
                )
            if store and buffer is not None:
                buffer.add(sample)
            episode_infos.append(
                {
                    "success": float(goal_reached),
                    "min_distance": float(min_distance if np.isfinite(min_distance) else self.cfg["L"]),
                    "collisions": float(collision_flag),
                    "energy": float(energy_accum),
                    "time": float(state.time),
                }
            )
        return episode_infos
