"""
Reward shaping utilities for USV SAC training.

Per-step reward:

.. math::

    r_t = w_{sh} r_{shape,t} + w_c r_{coll,t} + w_e r_{energy,t}
          + w_s r_{smooth,t} + w_t r_{time,t} + w_g r_{goal,t}

All quantities use SI units (metres, seconds, newtons).
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np


def _safe_array(x) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    return arr


def compute_step_reward(
    prev_goal_dist: float,
    goal_dist: float,
    tau_ctrl: Iterable[float],
    nu_body: Iterable[float],
    nu_current_body: Iterable[float],
    action_tanh: Iterable[float],
    prev_action_tanh: Iterable[float],
    distances: Iterable[float],
    done: bool,
    cfg: Dict[str, float],
) -> Tuple[float, Dict[str, float]]:
    """
    Compute a shaped reward for one environment transition.

    Parameters
    ----------
    prev_goal_dist : float
        Distance ``d_{g,t-1}`` [m].
    goal_dist : float
        Distance ``d_{g,t}`` [m].
    tau_ctrl : Iterable[float]
        Body-frame control inputs ``τ = [τ_u, τ_v, τ_r]`` with
        ``τ_r`` in N·m.
    nu_body : Iterable[float]
        Body-frame velocity ``ν = [u, v, r]`` (m·s⁻¹, rad·s⁻¹).
    nu_current_body : Iterable[float]
        Environmental drift ``ν_c^B`` (m·s⁻¹).
    action_tanh : Iterable[float]
        Squashed policy command ``a_t`` in [-1, 1].
    prev_action_tanh : Iterable[float]
        Previous squashed command ``a_{t-1}``.
    distances : Iterable[float]
        Distances to nearest obstacles ``d_i`` [m].
    done : bool
        Whether the episode terminates after this transition.
    cfg : dict
        Reward and physical scale configuration.
    """

    prev_goal = float(prev_goal_dist)
    goal = float(goal_dist)
    tau = _safe_array(tau_ctrl)
    nu = _safe_array(nu_body)
    nu_c = _safe_array(nu_current_body)
    act = _safe_array(action_tanh)
    prev_act = _safe_array(prev_action_tanh)
    dists = _safe_array(distances)

    l_scale = cfg["L_scale"]
    r_shape = (prev_goal - goal) / max(l_scale, 1e-6)

    slack = np.maximum(0.0, cfg["d_safe"] - dists) / max(cfg["d_safe"], 1e-6)
    r_coll = -cfg["rho"] * float(np.sum(slack**2))

    nu_rel = np.array([nu[0] - nu_c[0], nu[1] - nu_c[1], 0.0], dtype=np.float32)
    power = float(np.dot(tau, nu_rel))
    r_energy = -power / (cfg["T_max"] * cfg["U_max"])

    r_smooth = -float(np.sum((act - prev_act) ** 2))
    r_time = -cfg["C_step"]
    r_goal = cfg["R_success"] if (done and goal <= cfg["R_goal"]) else 0.0

    total = (
        cfg["w_sh"] * r_shape
        + cfg["w_c"] * r_coll
        + cfg["w_e"] * r_energy
        + cfg["w_s"] * r_smooth
        + cfg["w_t"] * r_time
        + cfg["w_g"] * r_goal
    )

    components = {
        "r_shape": r_shape,
        "r_coll": r_coll,
        "r_energy": r_energy,
        "r_smooth": r_smooth,
        "r_time": r_time,
        "r_goal": r_goal,
        "power_tau_nu_rel": power,
    }
    return float(total), components

