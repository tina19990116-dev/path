"""
Marine environment simulator aligned with a USV 3-DOF control stack.

Conventions
-----------
* Earth/world frame **E**: x-East, y-North. Grid coordinates follow this axes order.
* Body frame **B**: x-surge (forward), y-sway (port → starboard).
* Heading ψ rotates **B→E** with R(ψ)=[[cosψ,-sinψ],[sinψ,cosψ]]. Use R(ψ)^T for **E→B**.
* Velocities ν=[u, v, r]^T live in **B** (units: m/s, rad/s). Grid fields are 2-D horizontal vectors.
* Apparent wind uses the body-frame relative wind v_aw,B = w_B - ν_B^(xy), and currents enter via
  ν_r = ν - ν_c with ν_c=[u_c, v_c, 0]^T expressed in **B**.
* Densities/units: ρ_air = 1.225 kg/m^3, forces N, torques N·m, distances m, time s, headings rad.

The module keeps the original bilinear spatial interpolation but adds linear time interpolation, USV
body-frame transforms, simple aerodynamic loads, and gust/domain-randomization knobs needed by model
predictive or SAC-based controllers. Everything lives in this single module so the control stack can
```
import marine_environment_simulation as mes
sim = mes.MarineEnvironmentSimulation()
current_B = sim.get_current_B(5000.0, 5200.0, psi=0.25, t=37.5)
wind_B = sim.get_wind_B(5000.0, 5200.0, psi=0.25, t=37.5)
tau_env = sim.calc_env_forces_B(nu_B=np.array([1.2, 0.1, 0.02]), x=5000, y=5200, psi=0.25, t=37.5)
reward = sim.instant_energy_reward(
    tau_ctrl_B=np.array([150.0, 10.0, 2.0]),
    nu_B=np.array([1.2, 0.1, 0.02]),
    x=5000,
    y=5200,
    psi=0.25,
    t=37.5,
)
```

CLI Examples
------------
* List time labels: `python marine_environment_simulation.py --list-times`
* Export data bundle: `python marine_environment_simulation.py --export-data outputs/sim.npz`
* Sample physics-friendly query:
  `python marine_environment_simulation.py --sample 5000 5000 0 0 0 0 0 --sample-tau 0 0 0`
* Run sanity demos without plots: `python marine_environment_simulation.py --skip-plots --run-sanity`
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

try:  # Optional, only used when a config file is provided.
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - fallback when PyYAML is absent.
    yaml = None

try:
    from matplotlib import colormaps as _colormaps

    def _get_colormap(name: str):
        return _colormaps[name]

except ImportError:  # pragma: no cover - fallback for older matplotlib
    import matplotlib.cm as _cm

    def _get_colormap(name: str):  # type: ignore[override]
        return _cm.get_cmap(name)


MAP_EXTENT_METERS = 10_000.0
RHO_AIR = 1.225
WIND_REF_HEIGHT = 10.0
EPS = 1e-9


def create_regular_grid(length: float, samples_per_axis: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return 1D arrays of x and y coordinates spanning a square domain."""
    if samples_per_axis < 2:
        raise ValueError("samples_per_axis must be at least 2 for interpolation.")
    coords = np.linspace(0.0, length, samples_per_axis)
    return coords, coords.copy()


def sanitize_label(label: str) -> str:
    """Return a filesystem-safe representation of a time label."""
    safe = "".join(ch if ch.isalnum() else "_" for ch in label)
    return safe.strip("_") or "time"


def adjust_wind_to_height(z_usv: float, z0: float) -> float:
    """Return the logarithmic profile scale factor for winds measured at 10 m."""
    if z_usv <= 0 or z0 <= 0:
        raise ValueError("Heights must be positive for the log wind profile.")
    if z_usv <= z0:
        raise ValueError("z_usv must be greater than z0.")
    return math.log(z_usv / z0) / math.log(WIND_REF_HEIGHT / z0)


def _rotation_components(psi: float) -> Tuple[float, float]:
    return math.cos(psi), math.sin(psi)


def to_body(vec_E: np.ndarray, psi: float) -> np.ndarray:
    """Rotate a 2-D vector from Earth frame into the body frame."""
    c, s = _rotation_components(psi)
    rot = np.array([[c, s], [-s, c]])  # R^T
    return rot @ np.asarray(vec_E, dtype=float)


def to_world(vec_B: np.ndarray, psi: float) -> np.ndarray:
    """Rotate a 2-D vector from the body frame into Earth coordinates."""
    c, s = _rotation_components(psi)
    rot = np.array([[c, -s], [s, c]])
    return rot @ np.asarray(vec_B, dtype=float)


@dataclass
class GridVectorField:
    """Container for a time-indexed vector field defined on a regular grid."""

    name: str
    x_coords: np.ndarray  # shape (nx,)
    y_coords: np.ndarray  # shape (ny,)
    fields: Dict[str, np.ndarray]  # time label -> (ny, nx, 2)

    def available_times(self) -> Iterable[str]:
        return sorted(self.fields.keys())

    def _require_time(self, time_key: str) -> np.ndarray:
        if time_key not in self.fields:
            raise KeyError(f"Time '{time_key}' is not available for {self.name}.")
        return self.fields[time_key]

    def interpolate(self, x: float, y: float, time_key: str) -> np.ndarray:
        """Bilinearly interpolate the vector field at position (x, y)."""
        field = self._require_time(time_key)
        if not (self.x_coords[0] <= x <= self.x_coords[-1]):
            raise ValueError(
                f"x={x:.2f} lies outside the domain [{self.x_coords[0]}, {self.x_coords[-1]}]."
            )
        if not (self.y_coords[0] <= y <= self.y_coords[-1]):
            raise ValueError(
                f"y={y:.2f} lies outside the domain [{self.y_coords[0]}, {self.y_coords[-1]}]."
            )

        ix_upper = int(np.searchsorted(self.x_coords, x, side="left"))
        iy_upper = int(np.searchsorted(self.y_coords, y, side="left"))

        def _bound(idx_upper: int, max_index: int) -> Tuple[int, int]:
            if idx_upper <= 0:
                return 0, 1
            if idx_upper >= max_index:
                return max_index - 2, max_index - 1
            return idx_upper - 1, idx_upper

        ix0, ix1 = _bound(ix_upper, self.x_coords.size)
        iy0, iy1 = _bound(iy_upper, self.y_coords.size)
        x0, x1 = self.x_coords[ix0], self.x_coords[ix1]
        y0, y1 = self.y_coords[iy0], self.y_coords[iy1]

        dx = x1 - x0
        dy = y1 - y0
        wx = 0.0 if dx == 0 else (x - x0) / dx
        wy = 0.0 if dy == 0 else (y - y0) / dy

        v00 = field[iy0, ix0]
        v10 = field[iy0, ix1]
        v01 = field[iy1, ix0]
        v11 = field[iy1, ix1]

        v0 = (1 - wx) * v00 + wx * v10
        v1 = (1 - wx) * v01 + wx * v11
        return (1 - wy) * v0 + wy * v1

    def magnitude(self, time_key: str) -> np.ndarray:
        """Return the speed magnitude grid for a given time."""
        return np.linalg.norm(self._require_time(time_key), axis=-1)


def _load_config(path: Optional[str]) -> Dict[str, object]:
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file '{path}' not found.")
    with open(path, "r", encoding="utf-8") as handle:
        raw = handle.read()
    if yaml is not None:
        data = yaml.safe_load(raw)  # type: ignore[arg-type]
    else:  # Fallback: accept JSON-as-YAML subsets.
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:  # pragma: no cover - formatting errors
            raise RuntimeError(
                "PyYAML is not installed and the config file is not JSON-compatible."
            ) from exc
    if not isinstance(data, dict):
        raise ValueError("Top-level config must be a mapping.")
    return data


def _apply_overrides(args: argparse.Namespace, overrides: Dict[str, object]) -> argparse.Namespace:
    if not overrides:
        return args
    arg_dict = vars(args)

    def _assign(key: str, value: object) -> None:
        normalized = key.replace("-", "_")
        if normalized in arg_dict:
            arg_dict[normalized] = value

    for key, value in overrides.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                _assign(sub_key, sub_value)
        else:
            _assign(key, value)
    return args


def time_to_keypair(simulation: "MarineEnvironmentSimulation", t: float) -> Tuple[str, str, float]:
    """Expose the key-pair helper as a pure function."""
    return simulation.time_to_keypair(t)


def interpolate_time(
    simulation: "MarineEnvironmentSimulation", field_name: str, x: float, y: float, t: float
) -> np.ndarray:
    """Public helper to use the time interpolation without touching internals."""
    return simulation.interpolate_time(field_name, x, y, t)


def get_current_E(simulation: "MarineEnvironmentSimulation", x: float, y: float, t: float) -> np.ndarray:
    """2-D ocean current expressed in **E**."""
    return simulation._interpolate_time_field("ocean_currents", x, y, t)


def get_current_B(
    simulation: "MarineEnvironmentSimulation", x: float, y: float, psi: float, t: float
) -> np.ndarray:
    """Ocean current expressed in the body frame."""
    vec_E = get_current_E(simulation, x, y, t)
    return to_body(vec_E, psi)


def get_wind_E(
    simulation: "MarineEnvironmentSimulation",
    x: float,
    y: float,
    t: float,
    z_usv: Optional[float] = None,
    z0: Optional[float] = None,
) -> np.ndarray:
    """Wind vector in **E** at the requested height."""
    base = simulation._interpolate_time_field("wind_reference", x, y, t)
    scale = adjust_wind_to_height(z_usv or simulation.z_usv, z0 or simulation.z0)
    return base * scale * simulation.scale_wind


def get_wind_B(
    simulation: "MarineEnvironmentSimulation",
    x: float,
    y: float,
    psi: float,
    t: float,
    z_usv: Optional[float] = None,
    z0: Optional[float] = None,
) -> np.ndarray:
    """Wind vector in the body frame."""
    vec_E = get_wind_E(simulation, x, y, t, z_usv=z_usv, z0=z0)
    return to_body(vec_E, psi)


def calc_env_forces_B(
    nu_B: np.ndarray,
    wind_B: np.ndarray,
    params: Dict[str, float],
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Return tau_env_B = [F_x, F_y, M_z] from wind-induced drag.

    F_x = 0.5 ρ CdA_x |v_aw| v_aw,x, F_y analogous with CdA_y, M_z = F_y * ell_aw + wave bias.
    """
    v_aw = np.asarray(wind_B, dtype=float) - np.asarray(nu_B, dtype=float)[:2]
    speed = float(np.linalg.norm(v_aw))
    CdA_x = params.get("CdA_x", 2.0)
    CdA_y = params.get("CdA_y", 3.0)
    ell_aw = params.get("ell_aw", 0.5)
    sigma_wave = params.get("sigma_wave", 0.0)

    Fx = 0.5 * RHO_AIR * CdA_x * speed * (v_aw[0] if speed > 0.0 else 0.0)
    Fy = 0.5 * RHO_AIR * CdA_y * speed * (v_aw[1] if speed > 0.0 else 0.0)
    Mz = Fy * ell_aw
    if sigma_wave > 0.0:
        noise = (rng or np.random.default_rng()).normal() * sigma_wave
        Mz += noise
    return np.array([Fx, Fy, Mz], dtype=float)


def instant_energy_reward(
    tau_ctrl_B: np.ndarray,
    nu_B: np.ndarray,
    current_B: np.ndarray,
) -> float:
    """Energy penalty consistent with ν_r = ν - ν_c (yaw not penalized here)."""
    nu_rel = np.asarray(nu_B, dtype=float).copy()
    nu_rel[0] -= float(current_B[0])
    nu_rel[1] -= float(current_B[1])
    work = np.dot(np.asarray(tau_ctrl_B, dtype=float), np.array([nu_rel[0], nu_rel[1], 0.0]))
    return -float(work)


class MarineEnvironmentSimulation:
    """Builds synthetic ocean current and wind fields over the target domain."""

    def __init__(
        self,
        map_extent: float = MAP_EXTENT_METERS,
        grid_resolution: int = 101,
        z_usv: float = 2.0,
        z0: float = 0.0002,
        t0: str = "2024-05-01T00:00Z",
        duration: float = 86_400.0,
        dt_field: float = 300.0,
        scale_current: float = 1.0,
        scale_wind: float = 1.0,
        CdA_x: float = 2.0,
        CdA_y: float = 3.0,
        ell_aw: float = 0.5,
        gust_sigma: float = 0.0,
        gust_tau: float = 5.0,
        seed: Optional[int] = None,
    ) -> None:
        if dt_field <= 0:
            raise ValueError("dt_field must be positive.")
        if duration < dt_field:
            raise ValueError("duration must be >= dt_field to enable interpolation.")
        self.map_extent = map_extent
        self.grid_resolution = grid_resolution
        self.z_usv = z_usv
        self.z0 = z0
        self.duration = duration
        self.dt_field = dt_field
        self.scale_current = scale_current
        self.scale_wind = scale_wind
        self.CdA_x = CdA_x
        self.CdA_y = CdA_y
        self.ell_aw = ell_aw
        self.gust_sigma = gust_sigma
        self.gust_tau = gust_tau
        self.rng = np.random.default_rng(seed)
        self._gust_state = np.zeros(2, dtype=float)
        self._last_gust = np.zeros(2, dtype=float)

        self._time_origin_iso, self._iso_origin = self._parse_t0(t0)
        self.time_seconds = np.arange(0.0, duration + EPS, dt_field, dtype=float)
        self.time_labels = self._format_time_labels()

        self.x_coords, self.y_coords = create_regular_grid(map_extent, grid_resolution)
        self.ocean_currents = self._build_ocean_current_field()
        self.wind_reference_field = self._build_wind_field()
        self.wind_field = self._scaled_wind_field()
        self._field_registry = {
            "ocean_currents": self.ocean_currents,
            "wind_field": self.wind_field,
            "wind_reference": self.wind_reference_field,
        }

    @staticmethod
    def _parse_t0(t0: str) -> Tuple[Optional[datetime], str]:
        if t0.upper().startswith("T+"):
            return None, t0
        try:
            if t0.endswith("Z"):
                dt = datetime.fromisoformat(t0.replace("Z", "+00:00"))
            else:
                dt = datetime.fromisoformat(t0)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc), dt.isoformat().replace("+00:00", "Z")
        except ValueError as exc:
            raise ValueError(
                "t0 must be ISO8601 with 'Z' or in the form 'T+seconds'."
            ) from exc

    def _format_time_labels(self) -> List[str]:
        labels: List[str] = []
        for seconds in self.time_seconds:
            if self._time_origin_iso:
                dt_val = self._time_origin_iso + timedelta(seconds=float(seconds))
                labels.append(dt_val.isoformat().replace("+00:00", "Z"))
            else:
                labels.append(f"T+{int(round(seconds)):05d}s")
        return labels

    def _build_ocean_current_field(self) -> GridVectorField:
        fields: Dict[str, np.ndarray] = {}
        X, Y = np.meshgrid(self.x_coords, self.y_coords, indexing="xy")
        denom = max(self.duration, self.dt_field)
        for label, t_sec in zip(self.time_labels, self.time_seconds):
            phase = 2.0 * math.pi * (t_sec / denom)
            tidal_component = 0.6 * np.sin(2 * math.pi * X / self.map_extent + phase)
            coriolis_component = 0.4 * np.cos(
                2 * math.pi * Y / self.map_extent - 0.5 * phase
            )

            u = 0.8 + 0.3 * np.sin(X / 1500.0 + phase) + 0.2 * np.cos(Y / 2500.0)
            v = 0.5 * np.cos(X / 2000.0 - phase) + 0.3 * np.sin(Y / 1200.0)
            u += 0.2 * tidal_component
            v += 0.2 * coriolis_component

            vector_field = np.stack((u, v), axis=-1).astype(np.float32) * self.scale_current
            fields[label] = vector_field
        return GridVectorField("Ocean current", self.x_coords, self.y_coords, fields)

    def _build_wind_field(self) -> GridVectorField:
        fields: Dict[str, np.ndarray] = {}
        X, Y = np.meshgrid(self.x_coords, self.y_coords, indexing="xy")
        denom = max(self.duration, self.dt_field)
        for label, t_sec in zip(self.time_labels, self.time_seconds):
            phase = 2.0 * math.pi * (t_sec / denom)
            thermal_gradient = 1.2 * np.sin(2 * math.pi * Y / self.map_extent + 0.3 * phase)
            pressure_gradient = 1.0 * np.cos(2 * math.pi * X / self.map_extent - 0.1 * phase)

            u10 = 6.0 + 1.5 * np.cos(X / 1800.0 + phase) + 0.5 * thermal_gradient
            v10 = 3.5 + 1.0 * np.sin(Y / 2200.0 - 0.5 * phase) + 0.5 * pressure_gradient

            vector_field = np.stack((u10, v10), axis=-1).astype(np.float32)
            fields[label] = vector_field
        return GridVectorField("Wind (10 m)", self.x_coords, self.y_coords, fields)

    def _scaled_wind_field(self) -> GridVectorField:
        base_scale = adjust_wind_to_height(self.z_usv, self.z0) * self.scale_wind
        scaled_fields = {
            label: field * base_scale for label, field in self.wind_reference_field.fields.items()
        }
        return GridVectorField("Wind", self.x_coords, self.y_coords, scaled_fields)

    def validate_time(self, time_key: str) -> None:
        if time_key not in self.wind_field.fields:
            raise KeyError(f"Unknown time '{time_key}'. Available options: {', '.join(self.time_labels)}.")

    def time_to_keypair(self, t: float) -> Tuple[str, str, float]:
        if t <= self.time_seconds[0]:
            return self.time_labels[0], self.time_labels[0], 0.0
        if t >= self.time_seconds[-1]:
            return self.time_labels[-1], self.time_labels[-1], 0.0
        idx_hi = int(np.searchsorted(self.time_seconds, t, side="right"))
        idx_lo = idx_hi - 1
        lo_t = self.time_seconds[idx_lo]
        hi_t = self.time_seconds[idx_hi]
        alpha = 0.0 if hi_t == lo_t else (t - lo_t) / (hi_t - lo_t)
        return self.time_labels[idx_lo], self.time_labels[idx_hi], float(np.clip(alpha, 0.0, 1.0))

    def _interpolate_time_field(self, field_name: str, x: float, y: float, t: float) -> np.ndarray:
        if field_name not in self._field_registry:
            raise KeyError(f"Unknown field '{field_name}'.")
        field = self._field_registry[field_name]
        key_lo, key_hi, alpha = self.time_to_keypair(t)
        vec_lo = field.interpolate(x, y, key_lo)
        if key_lo == key_hi or alpha <= EPS:
            return vec_lo
        vec_hi = field.interpolate(x, y, key_hi)
        return (1.0 - alpha) * vec_lo + alpha * vec_hi

    def interpolate_time(self, field_name: str, x: float, y: float, t: float) -> np.ndarray:
        """Public helper requested by the control stack."""
        return self._interpolate_time_field(field_name, x, y, t)

    def get_current_E(self, x: float, y: float, t: float) -> np.ndarray:
        return get_current_E(self, x, y, t)

    def get_current_B(self, x: float, y: float, psi: float, t: float) -> np.ndarray:
        return get_current_B(self, x, y, psi, t)

    def get_wind_E(
        self,
        x: float,
        y: float,
        t: float,
        z_usv: Optional[float] = None,
        z0: Optional[float] = None,
    ) -> np.ndarray:
        return get_wind_E(self, x, y, t, z_usv=z_usv, z0=z0)

    def get_wind_B(
        self,
        x: float,
        y: float,
        psi: float,
        t: float,
        z_usv: Optional[float] = None,
        z0: Optional[float] = None,
    ) -> np.ndarray:
        return get_wind_B(self, x, y, psi, t, z_usv=z_usv, z0=z0)

    def sample_gust_B(self, dt_ctrl: float) -> np.ndarray:
        if dt_ctrl <= 0:
            raise ValueError("dt_ctrl must be positive for the gust process.")
        if self.gust_sigma <= 0:
            self._last_gust = np.zeros(2)
            return self._last_gust.copy()
        self._gust_state += -self._gust_state * (dt_ctrl / max(self.gust_tau, EPS))
        noise_scale = self.gust_sigma * math.sqrt(2.0 / max(self.gust_tau, EPS))
        self._gust_state += noise_scale * math.sqrt(dt_ctrl) * self.rng.standard_normal(2)
        self._last_gust = self._gust_state.copy()
        return self._last_gust.copy()

    def calc_env_forces_B(
        self,
        nu_B: np.ndarray,
        x: float,
        y: float,
        psi: float,
        t: float,
        use_gust: bool = False,
    ) -> np.ndarray:
        wind_B = self.get_wind_B(x, y, psi, t)
        if use_gust and self.gust_sigma > 0.0:
            wind_B = wind_B + self._last_gust
        params = {"CdA_x": self.CdA_x, "CdA_y": self.CdA_y, "ell_aw": self.ell_aw, "sigma_wave": 0.0}
        return calc_env_forces_B(nu_B, wind_B, params, rng=self.rng)

    def instant_energy_reward(
        self,
        tau_ctrl_B: np.ndarray,
        nu_B: np.ndarray,
        x: float,
        y: float,
        psi: float,
        t: float,
    ) -> float:
        current_B = self.get_current_B(x, y, psi, t)
        return instant_energy_reward(tau_ctrl_B, nu_B, current_B)


def plot_vector_field(
    field: GridVectorField,
    time_key: str,
    output_path: str,
    cmap: str = "viridis",
    max_quivers: int = 25,
    use_tight_layout: bool = False,
    image_format: str = "png",
) -> None:
    """Render a magnitude heatmap with direction arrows using Pillow."""
    data = field._require_time(time_key)
    speed = np.linalg.norm(data, axis=-1)
    if speed.size == 0:
        raise ValueError("Vector field contains no data.")

    cmap_obj = _get_colormap(cmap)
    vmin, vmax = float(speed.min()), float(speed.max())
    if vmin == vmax:
        vmax = vmin + 1.0
    normed = np.clip((speed - vmin) / (vmax - vmin), 0.0, 1.0)
    rgba = cmap_obj(normed)
    rgb = (rgba[..., :3] * 255).astype(np.uint8)
    rgb = rgb[::-1]  # Flip vertically to emulate origin="lower".
    image = Image.fromarray(rgb, mode="RGB")

    scale = max(6, 600 // max(data.shape[0], data.shape[1]))
    image = image.resize((image.width * scale, image.height * scale), resample=Image.BILINEAR)

    draw = ImageDraw.Draw(image)
    ny, nx = speed.shape
    stride_x = max(1, nx // max_quivers)
    stride_y = max(1, ny // max_quivers)
    max_mag = float(np.max(speed))
    arrow_scale = scale * 2.0 if max_mag > 1e-8 else 0.0

    arrow_color = (255, 255, 255)
    width = max(1, scale // 4)
    head_length = arrow_scale * 0.3
    head_angle = math.radians(20)

    for y_idx in range(0, ny, stride_y):
        for x_idx in range(0, nx, stride_x):
            vec = data[y_idx, x_idx]
            mag = math.hypot(vec[0], vec[1])
            if mag <= 1e-6 or max_mag <= 1e-6:
                continue
            base_x = (x_idx + 0.5) * scale
            base_y = (ny - y_idx - 0.5) * scale
            dx = (vec[0] / max_mag) * arrow_scale
            dy = (vec[1] / max_mag) * arrow_scale
            end_x = base_x + dx
            end_y = base_y - dy
            draw.line((base_x, base_y, end_x, end_y), fill=arrow_color, width=width)

            angle = math.atan2(base_y - end_y, end_x - base_x)
            left_x = end_x - head_length * math.cos(angle + head_angle)
            left_y = end_y - head_length * math.sin(angle + head_angle)
            right_x = end_x - head_length * math.cos(angle - head_angle)
            right_y = end_y - head_length * math.sin(angle - head_angle)
            draw.line((end_x, end_y, left_x, left_y), fill=arrow_color, width=width)
            draw.line((end_x, end_y, right_x, right_y), fill=arrow_color, width=width)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path, format=image_format.upper())


def plot_raw_heatmap(
    field: GridVectorField,
    time_key: str,
    output_path: str,
    cmap: str = "viridis",
    image_format: str = "png",
) -> None:
    """Render the discrete grid magnitude without interpolation or arrows."""
    data = field._require_time(time_key)
    speed = np.linalg.norm(data, axis=-1)
    cmap_obj = _get_colormap(cmap)
    vmin, vmax = float(speed.min()), float(speed.max())
    if vmin == vmax:
        vmax = vmin + 1.0
    normed = np.clip((speed - vmin) / (vmax - vmin), 0.0, 1.0)
    rgba = cmap_obj(normed)
    rgb = (rgba[..., :3] * 255).astype(np.uint8)
    rgb = rgb[::-1]
    image = Image.fromarray(rgb, mode="RGB")
    scale = max(6, 600 // max(data.shape[0], data.shape[1]))
    image = image.resize((image.width * scale, image.height * scale), resample=Image.NEAREST)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path, format=image_format.upper())


def export_dataset(simulation: MarineEnvironmentSimulation, output_path: str) -> None:
    """Persist the gridded datasets with metadata for downstream training."""
    time_labels = np.array(simulation.time_labels, dtype="U48")
    ocean_stack = np.stack(
        [simulation.ocean_currents.fields[label] for label in simulation.time_labels], axis=0
    )
    wind_stack = np.stack(
        [simulation.wind_field.fields[label] for label in simulation.time_labels], axis=0
    )
    metadata = {
        "t0": simulation._iso_origin,
        "duration": simulation.duration,
        "dt_field": simulation.dt_field,
        "frames": {"world": "E", "body": "B"},
        "units": {"speed": "m/s", "force": "N", "torque": "N*m"},
    }
    np.savez_compressed(
        output_path,
        time_labels=time_labels,
        time_seconds=simulation.time_seconds.astype(np.float64),
        x_coords=simulation.x_coords,
        y_coords=simulation.y_coords,
        ocean_currents=ocean_stack,
        wind_field=wind_stack,
        metadata=json.dumps(metadata),
    )


def _add_cli_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--time", help="Time label (ISO) or seconds for plotting.")
    parser.add_argument("--list-times", action="store_true", help="List available time labels and exit.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for generated figures.")
    parser.add_argument("--grid-resolution", type=int, default=101, help="Grid points per axis.")
    parser.add_argument("--z-usv", type=float, default=2.0, help="USV-equivalent height in meters.")
    parser.add_argument("--z0", type=float, default=0.0002, help="Surface roughness length in meters.")
    parser.add_argument("--map-extent", type=float, default=MAP_EXTENT_METERS, help="Square map extent [m].")
    parser.add_argument("--t0", default="2024-05-01T00:00Z", help="Time origin (ISO8601 or 'T+seconds').")
    parser.add_argument("--duration", type=float, default=86_400.0, help="Simulation duration in seconds.")
    parser.add_argument("--dt-field", type=float, default=300.0, help="Base grid time step in seconds.")
    parser.add_argument("--dt-plot", type=float, default=3600.0, help="Spacing between plotted slices (s).")
    parser.add_argument("--scale-current", type=float, default=1.0, help="Randomization scale for currents.")
    parser.add_argument("--scale-wind", type=float, default=1.0, help="Randomization scale for winds.")
    parser.add_argument("--CdA-x", type=float, default=2.0, help="Aero Cd*A in surge.")
    parser.add_argument("--CdA-y", type=float, default=3.0, help="Aero Cd*A in sway.")
    parser.add_argument("--ell-aw", type=float, default=0.5, help="Lever arm for apparent wind moment (m).")
    parser.add_argument("--gust-sigma", type=float, default=0.0, help="OU gust sigma [m/s].")
    parser.add_argument("--gust-tau", type=float, default=5.0, help="OU gust time constant [s].")
    parser.add_argument("--seed", type=int, help="Random seed for gusts/domain randomization.")
    parser.add_argument("--config", help="Optional YAML/JSON config for overrides.")
    parser.add_argument(
        "--sample",
        type=float,
        nargs=7,
        metavar=("X", "Y", "PSI_DEG", "T_SEC", "U", "V", "R"),
        help="Evaluate physics APIs at a single state.",
    )
    parser.add_argument(
        "--sample-tau",
        type=float,
        nargs=3,
        metavar=("TAU_X", "TAU_Y", "TAU_N"),
        default=(0.0, 0.0, 0.0),
        help="Control effort used for the energy reward in the --sample query.",
    )
    parser.add_argument(
        "--tau-ctrl",
        type=float,
        nargs=3,
        metavar=("Fx", "Fy", "Mz"),
        help="Optional body-frame control wrench [N, N, N*m] used to compute energy reward.",
    )
    parser.add_argument(
        "--export-data",
        help="Optional .npz path to store the synthetic datasets for all time slices.",
    )
    parser.add_argument("--skip-plots", action="store_true", help="Skip heatmap generation.")
    parser.add_argument("--skip-raw", action="store_true", help="Skip the additional raw heatmaps.")
    parser.add_argument("--image-format", default="png", help="Image format for saved plots.")
    parser.add_argument("--run-sanity", action="store_true", help="Print the quick sanity outputs.")
    parser.add_argument(
        "--sample-point",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        help="Backward-compatible spatial sample for raw interpolation.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Marine environment simulator for ocean current and wind fields.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_cli_arguments(parser)
    return parser.parse_args()


def _resolve_plot_labels(sim: MarineEnvironmentSimulation, args: argparse.Namespace) -> List[str]:
    if args.time:
        try:
            t_sec = float(args.time)
            key_lo, _, _ = sim.time_to_keypair(t_sec)
            return [key_lo]
        except ValueError:
            sim.validate_time(args.time)
            return [args.time]
    times = np.arange(0.0, sim.duration + EPS, args.dt_plot)
    labels = {sim.time_to_keypair(float(t))[0] for t in times}
    return sorted(labels)


def _print_sample(sim: MarineEnvironmentSimulation, args: argparse.Namespace) -> None:
    if not args.sample:
        return
    x, y, psi_deg, t_sec, u, v, r = args.sample
    psi = math.radians(psi_deg)
    nu_B = np.array([u, v, r], dtype=float)
    current_B = sim.get_current_B(x, y, psi, t_sec)
    wind_B = sim.get_wind_B(x, y, psi, t_sec)
    tau_env = sim.calc_env_forces_B(nu_B, x, y, psi, t_sec, use_gust=args.gust_sigma > 0)
    tau_ctrl_B = np.array(args.tau_ctrl, dtype=float) if args.tau_ctrl else np.zeros(3, dtype=float)
    r_energy = sim.instant_energy_reward(
        tau_ctrl_B,
        nu_B=nu_B,
        x=x,
        y=y,
        psi=psi,
        t=t_sec,
    )
    print("Sample query results:")
    print(f"  current_B = [{current_B[0]:.3f}, {current_B[1]:.3f}] m/s")
    print(f"  wind_B    = [{wind_B[0]:.3f}, {wind_B[1]:.3f}] m/s")
    print(f"  tau_env_B = [{tau_env[0]:.2f}, {tau_env[1]:.2f}, {tau_env[2]:.2f}] N/Nm")
    print(f"  r_energy  = {r_energy:.3f}")


def _print_backward_compat_sample(sim: MarineEnvironmentSimulation, args: argparse.Namespace, time_key: str) -> None:
    if not args.sample_point:
        return
    x_q, y_q = args.sample_point
    oc_vector = sim.ocean_currents.interpolate(x_q, y_q, time_key)
    wind_vector = sim.wind_field.interpolate(x_q, y_q, time_key)
    print(
        "Interpolated vectors "
        f"at ({x_q:.1f} m, {y_q:.1f} m):\n"
        f"  Ocean current: u={oc_vector[0]:.3f} m/s, v={oc_vector[1]:.3f} m/s\n"
        f"  Wind: u={wind_vector[0]:.3f} m/s, v={wind_vector[1]:.3f} m/s"
    )


def _run_sanity(sim: MarineEnvironmentSimulation) -> None:
    center = sim.map_extent / 2.0
    t_mid = sim.dt_field / 2.0
    psi0 = 0.0
    psi90 = math.pi / 2.0
    nu_zero = np.array([0.0, 0.0, 0.0])

    cur0 = sim.get_current_B(center, center, psi0, 0.0)
    wind0 = sim.get_wind_B(center, center, psi0, 0.0)
    print(
        "Sanity @ psi=0°: |current|={:.3f} m/s, |wind|={:.3f} m/s".format(
            np.linalg.norm(cur0), np.linalg.norm(wind0)
        )
    )

    wind90 = sim.get_wind_B(center, center, psi90, 0.0)
    print("Sanity heading rotation: wind_B(90°) = [{:.3f}, {:.3f}] m/s".format(wind90[0], wind90[1]))

    cur_lo = sim.get_current_E(center, center, 0.0)
    cur_hi = sim.get_current_E(center, center, sim.dt_field)
    cur_mid = sim.get_current_E(center, center, t_mid)
    delta = np.linalg.norm(cur_mid - cur_lo)
    print("Time interpolation delta (mid vs start) = {:.4f} m/s".format(delta))

    if sim.gust_sigma > 0:
        gust = sim.sample_gust_B(dt_ctrl=0.1)
        print("Gust sample (dt=0.1 s) = [{:.3f}, {:.3f}] m/s".format(gust[0], gust[1]))

    tau_env = sim.calc_env_forces_B(nu_zero, center, center, psi0, 0.0, use_gust=False)
    print("Wind-only tau_env_B @ rest = [{:.2f}, {:.2f}, {:.2f}]".format(tau_env[0], tau_env[1], tau_env[2]))


def main() -> None:
    args = parse_args()
    overrides = _load_config(args.config)
    args = _apply_overrides(args, overrides)

    sim = MarineEnvironmentSimulation(
        map_extent=args.map_extent,
        grid_resolution=args.grid_resolution,
        z_usv=args.z_usv,
        z0=args.z0,
        t0=args.t0,
        duration=args.duration,
        dt_field=args.dt_field,
        scale_current=args.scale_current,
        scale_wind=args.scale_wind,
        CdA_x=args.CdA_x,
        CdA_y=args.CdA_y,
        ell_aw=args.ell_aw,
        gust_sigma=args.gust_sigma,
        gust_tau=args.gust_tau,
        seed=args.seed,
    )

    if args.list_times:
        print("Available time slices:")
        for label in sim.time_labels:
            print(f"  {label}")
        return

    plot_labels = _resolve_plot_labels(sim, args)

    if not args.skip_plots:
        os.makedirs(args.output_dir, exist_ok=True)
        extension = args.image_format.lower()
        for label in plot_labels:
            ocean_output = os.path.join(
                args.output_dir, f"ocean_current_{sanitize_label(label)}.{extension}"
            )
            wind_output = os.path.join(
                args.output_dir, f"wind_field_{sanitize_label(label)}.{extension}"
            )
            if not args.skip_raw:
                ocean_raw_output = os.path.join(
                    args.output_dir,
                    f"ocean_current_{sanitize_label(label)}_raw.{extension}",
                )
                wind_raw_output = os.path.join(
                    args.output_dir,
                    f"wind_field_{sanitize_label(label)}_raw.{extension}",
                )
                plot_raw_heatmap(
                    sim.ocean_currents,
                    label,
                    ocean_raw_output,
                    cmap="Blues",
                    image_format=args.image_format,
                )
                plot_raw_heatmap(
                    sim.wind_field,
                    label,
                    wind_raw_output,
                    cmap="Oranges",
                    image_format=args.image_format,
                )
                print(f"Saved ocean current raw heatmap to '{ocean_raw_output}'.")
                print(f"Saved wind field raw heatmap to '{wind_raw_output}'.")

            plot_vector_field(
                sim.ocean_currents,
                label,
                ocean_output,
                cmap="Blues",
                use_tight_layout=False,
                image_format=args.image_format,
            )
            plot_vector_field(
                sim.wind_field,
                label,
                wind_output,
                cmap="Oranges",
                use_tight_layout=False,
                image_format=args.image_format,
            )

            print(f"Saved ocean current heatmap to '{ocean_output}'.")
            print(f"Saved wind field heatmap to '{wind_output}'.")

    if args.sample_point:
        _print_backward_compat_sample(sim, args, plot_labels[0])
    _print_sample(sim, args)

    if args.export_data:
        export_dataset(sim, args.export_data)
        print(f"Exported synthetic dataset to '{args.export_data}'.")

    if args.run_sanity:
        _run_sanity(sim)


if __name__ == "__main__":
    main()
