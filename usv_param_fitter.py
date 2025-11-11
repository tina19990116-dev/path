"""usv_param_fitter.py
=================================
Generates a self-consistent 3-DOF USV (surge, sway, yaw) parameter set using
simple naval-architecture priors, inverse fits against target specs, and basic
actuator assumptions.

Model summary (body-frame, SI units):
    (M_RB + M_A) * d(v)/dt + C(v_r) * v_r + D(v_r) = tau_ctrl + tau_env
    v_r = v - v_c
    D(v_r) = D1 * v_r + D2 * |v_r| ⊙ v_r

Only diagonal mass/damping terms are modeled explicitly; cross-flow terms
Y_r (sway force from yaw rate) and N_v (yaw moment from sway velocity) can be
optionally enabled with small magnitudes for domain randomization.

All outputs use SI units: speed m/s, force N, torque N·m, mass kg, inertia
kg·m², length m, rate rad/s. Body axes: x (surge), y (sway), yaw ψ.

Example YAML config (can be passed via --config):

```
U_max: 25.0
r_star: 0.5235987756
U_turn: 12.0
a_max: 1.2
length: 6.0
beam: 2.0
draft: 0.30
Cb: 0.35
rho: 1025
thrust_max: 7000
lever_arm: 0.8
U_cruise: 15.0
T_cruise: 3500
act_timeconst_rud: 0.25
act_timeconst_rpm: 1.0
out: params.yaml
print: true
```
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import os

try:  # Optional dependency per requirements
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - fallback handled at runtime
    yaml = None

DEBUG_MODE = bool(os.environ.get("USV_DEBUG_TRACE"))


def _debug_log(message: str) -> None:
    """Write internal debug traces when USV_DEBUG_TRACE is set."""
    if not DEBUG_MODE:
        return
    with open("usv_debug.log", "a", encoding="utf-8") as fp:
        fp.write(message + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments with optional YAML config overrides."""
    if argv is None:
        argv = sys.argv[1:]

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config", type=str, default=None, help="YAML config file with defaults."
    )
    config_ns, remaining = config_parser.parse_known_args(argv)
    config_defaults: Dict[str, Any] = {}
    if config_ns.config:
        config_defaults = load_config(config_ns.config)
        # Backward-compatibility: allow "print: true" in configs even though the CLI
        # flag stores to args.do_print.
        if "print" in config_defaults and "do_print" not in config_defaults:
            config_defaults["do_print"] = bool(config_defaults.pop("print"))

    parser = argparse.ArgumentParser(
        description=(
            "Fit nominal USV masses/damping/actuation parameters to achieve the "
            "requested speed/turn specs without measured data."
        )
    )

    # Performance targets
    parser.add_argument("--U_max", type=float, default=25.0)
    parser.add_argument("--r_star", type=float, default=0.5235987756)
    parser.add_argument("--U_turn", type=float, default=12.0)
    parser.add_argument("--a_max", type=float, default=1.2)

    # Geometry / hydro
    parser.add_argument("--length", type=float, default=6.0)
    parser.add_argument("--beam", type=float, default=2.0)
    parser.add_argument("--draft", type=float, default=0.30)
    parser.add_argument("--disp_mass", type=float, default=None)
    parser.add_argument("--rho", type=float, default=1025.0)
    parser.add_argument("--Cb", type=float, default=0.35)
    parser.add_argument("--gyration_kz_frac", type=float, default=0.34)

    # Added-mass overrides
    parser.add_argument("--X_udot", type=float, default=None)
    parser.add_argument("--Y_vdot", type=float, default=None)
    parser.add_argument("--N_rdot", type=float, default=None)

    # Priors / gains
    parser.add_argument("--kappa_x", type=float, default=0.8)
    parser.add_argument("--kappa_y", type=float, default=1.2)
    parser.add_argument("--kappa_y_quad", type=float, default=0.5)
    parser.add_argument("--kappa_n", type=float, default=0.8)

    # Propulsion / actuation capability
    parser.add_argument("--thrust_max", type=float, default=7000.0)
    parser.add_argument("--lever_arm", type=float, default=0.80)
    parser.add_argument("--act_timeconst_rud", type=float, default=0.25)
    parser.add_argument("--act_timeconst_rpm", type=float, default=1.0)
    parser.add_argument("--U_cruise", type=float, default=15.0)
    parser.add_argument("--T_cruise", type=float, default=3500.0)
    parser.add_argument("--eta_tau", type=float, default=0.8)
    parser.add_argument("--KT", type=float, default=0.18)

    # Randomization bounds
    parser.add_argument("--rand_mass_add", type=float, default=0.30)
    parser.add_argument("--rand_d1", type=float, default=0.30)
    parser.add_argument("--rand_d2", type=float, default=0.50)
    parser.add_argument("--rand_KT", type=float, default=0.20)
    parser.add_argument("--rand_Tact", type=float, default=0.20)
    parser.add_argument("--rand_cross", type=float, default=0.20)
    parser.add_argument("--enable_cross", action="store_true", default=False)

    # Output / misc
    parser.add_argument("--out", type=str, default="params.yaml")
    parser.add_argument("--print", action="store_true", dest="do_print", default=False)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--selftest", action="store_true", default=False)

    if config_defaults:
        parser.set_defaults(**config_defaults)
    args = parser.parse_args(remaining)
    args.config = config_ns.config
    return args


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML config into flat dict suitable for argparse defaults."""
    with open(path, "r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) if yaml else json.load(fp)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping.")
    return data


def estimate_M_RB(args: argparse.Namespace) -> Dict[str, float]:
    """Estimate rigid body mass/inertia."""
    if args.disp_mass and args.disp_mass > 0:
        m = args.disp_mass
    else:
        m = args.rho * args.length * args.beam * args.draft * args.Cb
    k_z = args.gyration_kz_frac * args.length
    Iz = m * k_z**2
    args._mass_hull = m
    args._Iz = Iz
    return {"m": m, "Iz": Iz}


def estimate_MA_priors(args: argparse.Namespace) -> Dict[str, float]:
    """Construct simple diagonal added-mass priors."""
    m = getattr(args, "_mass_hull", None)
    Iz = getattr(args, "_Iz", None)
    if m is None or Iz is None:
        raise RuntimeError("estimate_M_RB must be called before estimate_MA_priors.")

    x_udot = args.X_udot if args.X_udot is not None else -0.12 * m
    y_vdot = args.Y_vdot if args.Y_vdot is not None else -0.80 * m
    n_rdot = args.N_rdot if args.N_rdot is not None else -0.12 * Iz
    return {"X_udot": x_udot, "Y_vdot": y_vdot, "N_rdot": n_rdot}


def fit_surge_damping_from_spec(
    args: argparse.Namespace, M_RB: Dict[str, float], MA: Dict[str, float]
) -> Tuple[float, float]:
    """Fit linear and quadratic surge damping from thrust and speed specs."""
    m = M_RB["m"]
    base_abs_xu = args.kappa_x * m / max(args.length, 1e-6)
    abs_xu = base_abs_xu

    U_max = max(args.U_max, 1.0)
    thrust_max = max(args.thrust_max, 1.0)
    abs_xuu = max(0.0, (thrust_max - abs_xu * U_max) / (U_max**2))

    if args.U_cruise and args.T_cruise:
        U2 = float(args.U_cruise)
        T2 = float(args.T_cruise)
        if abs(U2 - U_max) > 1e-6 and U2 > 0:
            denom = U_max * U2 * (U2 - U_max)
            fit_xu = (thrust_max * (U2**2) - T2 * (U_max**2)) / denom
            fit_xuu = (T2 * U_max - thrust_max * U2) / denom
            if math.isfinite(fit_xu) and fit_xu > 0:
                abs_xu = fit_xu
            if math.isfinite(fit_xuu) and fit_xuu >= 0:
                abs_xuu = fit_xuu

    abs_xu = max(abs_xu, 0.1 * base_abs_xu)
    abs_xuu = max(abs_xuu, 1e-6)
    return -abs_xu, -abs_xuu


def fit_yaw_from_turn_spec(
    args: argparse.Namespace, Iz: float, tau_z_max: float
) -> Tuple[float, float]:
    """Fit yaw damping coefficients from target turn-rate spec."""
    abs_Nr = args.kappa_n * Iz / max(args.length**2, 1e-6)
    r_star = max(args.r_star, 1e-3)
    tau_star = args.eta_tau * tau_z_max

    if tau_star <= 0:
        raise ValueError("Available yaw moment must be positive.")

    if abs_Nr * r_star >= tau_star:
        scale = max(0.1, 0.98 * tau_star / (abs_Nr * r_star))
        print(
            (
                "INFO: Reducing |N_r| from {:.2f} to {:.2f} to satisfy yaw moment "
                "target."
            ).format(abs_Nr, abs_Nr * scale)
        )
        abs_Nr *= scale

    numerator = tau_star - abs_Nr * r_star
    if numerator <= 0:
        numerator = 0.05 * tau_star
    abs_Nrr = numerator / (r_star**2)
    abs_Nrr = max(abs_Nrr, 1e-6)
    return -abs_Nr, -abs_Nrr


def build_sway_damping(args: argparse.Namespace, m: float) -> Tuple[float, float]:
    """Estimate sway damping coefficients."""
    abs_Yv = args.kappa_y * m / max(args.length, 1e-6)
    vel_scale = max(args.U_turn, 1.0)
    abs_Yvv = args.kappa_y_quad * abs_Yv / vel_scale
    abs_Yvv = max(abs_Yvv, 1e-6)
    return -abs_Yv, -abs_Yvv


def actuation_params(args: argparse.Namespace) -> Dict[str, Any]:
    """Pack actuation first-order and allocation metadata."""
    return {
        "T_act": {"rudder": float(args.act_timeconst_rud), "rpm": float(args.act_timeconst_rpm)},
        "Gamma": {"lever_arm_b": float(args.lever_arm), "KT": float(args.KT)},
    }


def build_bounds(args: argparse.Namespace) -> Dict[str, Any]:
    """Assemble percent randomization bounds."""
    return {
        "M_A": {"pct": args.rand_mass_add},
        "D1": {"pct": args.rand_d1},
        "D2": {"pct": args.rand_d2},
        "T_act": {"pct": args.rand_Tact},
        "KT": {"pct": args.rand_KT},
    }


def quick_check_surge(
    args: argparse.Namespace, D1_x: float, D2_x: float
) -> Dict[str, Any]:
    """Run steady-state and rise-time surge checks."""
    def resist(u: float) -> float:
        return -D1_x * u - D2_x * abs(u) * u

    U_max = max(args.U_max, 1e-6)
    steady_drag = resist(U_max)
    err_pct = 100.0 * (steady_drag - args.thrust_max) / max(args.thrust_max, 1e-6)

    m_total = getattr(args, "_surge_mass", None)
    if m_total is None:
        raise RuntimeError("Surge mass not set before quick_check_surge.")

    target_speed = 0.9 * U_max
    dt = 0.05
    u = 0.0
    t = 0.0
    t_max = 120.0
    while u < target_speed and t < t_max:
        net = args.thrust_max - resist(u)
        acc = net / m_total
        u = max(u + acc * dt, 0.0)
        t += dt
        if acc <= 0 and net <= 0:
            break

    t_target = target_speed / max(args.a_max, 1e-6)
    rise_ok = t <= 1.15 * t_target
    return {
        "surge_Umax_error_pct": err_pct,
        "surge_risetime_s": t,
        "surge_risetime_target_s": t_target,
        "surge_risetime_ok": bool(rise_ok),
    }


def quick_check_turn(
    args: argparse.Namespace, Nr: float, Nrr: float, tau_z_max: float
) -> Dict[str, Any]:
    """Check yaw authority against available control moment."""
    r = max(args.r_star, 1e-6)
    tau_required = -(Nr * r + Nrr * abs(r) * r)
    tau_available = tau_z_max
    ok = tau_required <= tau_available + 1e-6
    return {
        "turn_tau_required_Nm": tau_required,
        "turn_tau_available_Nm": tau_available,
        "turn_ok": bool(ok),
    }


def save_params(path: str, data: Dict[str, Any]) -> None:
    """Persist parameters to YAML if available else JSON."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if yaml:
        with dest.open("w", encoding="utf-8") as fp:
            yaml.safe_dump(data, fp, sort_keys=False)
    else:
        with dest.open("w", encoding="utf-8") as fp:
            json.dump(data, fp, indent=2)


def _cross_terms(args: argparse.Namespace, Yv: float, Nr: float) -> Dict[str, float]:
    if not args.enable_cross:
        return {"Y_r": 0.0, "N_v": 0.0}
    rng = getattr(args, "_rng", np.random.default_rng())
    Y_r_mag = abs(Yv) * args.rand_cross
    N_v_mag = abs(Nr) * args.rand_cross * max(args.length, 1.0)
    return {
        "Y_r": rng.uniform(-Y_r_mag, Y_r_mag),
        "N_v": rng.uniform(-N_v_mag, N_v_mag),
    }


def _assemble_metadata(args: argparse.Namespace) -> Dict[str, Any]:
    timestamp = _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")
    return {
        "units": {"speed": "mps", "force": "N", "torque": "N*m"},
        "inputs": {
            "U_max": args.U_max,
            "r_star": args.r_star,
            "U_turn": args.U_turn,
            "a_max": args.a_max,
        },
        "version": "usv_param_fitter v1.0",
        "timestamp": timestamp,
    }


def _print_summary(params: Dict[str, Any]) -> None:
    model = params["model"]
    checks = params["checks"]
    act = params["actuation"]
    print("=== USV Parameter Summary ===")
    print(f"Mass (m): {model['M_RB']['m']:.1f} kg, Iz: {model['M_RB']['Iz']:.1f} kg*m^2")
    print(
        "Surge damping: X_u={:.3f}, X_uu={:.5f}".format(
            model["D1"]["X_u"], model["D2"]["X_uu"]
        )
    )
    print(
        "Yaw damping: N_r={:.3f}, N_rr={:.5f}".format(
            model["D1"]["N_r"], model["D2"]["N_rr"]
        )
    )
    print(
        "Actuation time constants (rud/rpm): "
        f"{act['T_act']['rudder']:.2f}s / {act['T_act']['rpm']:.2f}s, "
        f"lever arm: {act['Gamma']['lever_arm_b']:.2f} m"
    )
    print(
        "Checks: Umax err={:+.2f}% | rise={:.2f}s (target {:.2f}s, ok={}) | "
        "turn tau req={:.1f}Nm avail={:.1f}Nm (ok={})".format(
            checks["surge_Umax_error_pct"],
            checks["surge_risetime_s"],
            checks["surge_risetime_target_s"],
            checks["surge_risetime_ok"],
            checks["turn_tau_required_Nm"],
            checks["turn_tau_available_Nm"],
            checks["turn_ok"],
        )
    )


def _warn_if_needed(checks: Dict[str, Any], args: argparse.Namespace) -> None:
    warn_msgs = []
    if abs(checks["surge_Umax_error_pct"]) > 5.0:
        if checks["surge_Umax_error_pct"] > 0:
            warn_msgs.append(
                "WARN: Drag exceeds thrust at U_max. Consider raising --thrust_max or "
                "reducing --U_max."
            )
        else:
            warn_msgs.append(
                "WARN: Available thrust exceeds drag by a large margin; verify "
                "--X_u priors or reduce --thrust_max."
            )
    if not checks["surge_risetime_ok"]:
        warn_msgs.append(
            "WARN: Surge rise time too slow. Increase --thrust_max or reduce --a_max."
        )
    if not checks["turn_ok"]:
        warn_msgs.append(
            "WARN: Turn authority insufficient. Increase --lever_arm or "
            "--thrust_max, or lower --r_star."
        )
    for msg in warn_msgs:
        print(msg)


def _run_selftest(params: Dict[str, Any]) -> None:
    print("# Self-test sanity checks")
    assert params["model"]["M_RB"]["m"] > 0
    assert params["model"]["M_RB"]["Iz"] > 0
    print("# Example config snippet:")
    print(
        "U_max: 25.0\nr_star: 0.5235987756\nU_turn: 12.0\na_max: 1.2\n"
        "length: 6.0\nbeam: 2.0\ndraft: 0.3\nthrust_max: 7000\nlever_arm: 0.8"
    )
    print("# Self-test completed.")


def main(argv: list[str] | None = None) -> None:
    _debug_log("main: start")
    args = parse_args(argv)
    _debug_log("main: args parsed")
    rng = np.random.default_rng(args.seed)
    args._rng = rng
    _debug_log("main: rng ready")

    M_RB = estimate_M_RB(args)
    _debug_log(f"main: M_RB={M_RB}")
    MA = estimate_MA_priors(args)
    _debug_log(f"main: MA={MA}")
    total_surge_mass = M_RB["m"] - MA["X_udot"]
    args._surge_mass = total_surge_mass
    _debug_log(f"main: surge mass {total_surge_mass:.2f}")

    D1_x, D2_x = fit_surge_damping_from_spec(args, M_RB, MA)
    _debug_log(f"main: D1_x={D1_x}, D2_x={D2_x}")
    Yv, Yvv = build_sway_damping(args, M_RB["m"])
    _debug_log(f"main: Yv={Yv}, Yvv={Yvv}")
    Nr, Nrr = fit_yaw_from_turn_spec(args, M_RB["Iz"], args.thrust_max * args.lever_arm)
    _debug_log(f"main: Nr={Nr}, Nrr={Nrr}")
    cross = _cross_terms(args, Yv, Nr)
    _debug_log(f"main: cross={cross}")

    actuation = actuation_params(args)
    bounds = build_bounds(args)
    _debug_log("main: actuation/bounds ready")

    surge_checks = quick_check_surge(args, D1_x, D2_x)
    turn_checks = quick_check_turn(args, Nr, Nrr, args.thrust_max * args.lever_arm)
    _debug_log(f"main: checks computed {surge_checks} {turn_checks}")
    checks = {**surge_checks, **turn_checks}

    params = {
        "model": {
            "M_RB": M_RB,
            "M_A": MA,
            "D1": {"X_u": D1_x, "Y_v": Yv, "N_r": Nr},
            "D2": {"X_uu": D2_x, "Y_vv": Yvv, "N_rr": Nrr},
            "cross": cross,
        },
        "actuation": actuation,
        "bounds": bounds,
        "checks": checks,
        "metadata": _assemble_metadata(args),
    }
    _debug_log("main: params assembled")

    save_params(args.out, params)
    _debug_log(f"main: saved to {args.out}")
    if args.do_print:
        _print_summary(params)
    _warn_if_needed(checks, args)

    if args.selftest:
        _run_selftest(params)
    _debug_log("main: complete")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
