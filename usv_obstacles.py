
"""
Obstacle modeling utilities for USV RL/MPC stacks.

Frames/units:
  - World frame E: x axis points East, y axis North (meters).
  - Body frame B: x points forward (surge), y starboard (sway).
  - Heading ψ rotates B → E via R(ψ) = [[cosψ, -sinψ], [sinψ, cosψ]].
    Use R(ψ).T for E → B.
Distances use meters, time uses seconds, velocities meters/second. All
random sampling relies on numpy's Generator for deterministic behavior.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from PIL import Image, ImageDraw

    PIL_AVAILABLE = True
except Exception:  # pragma: no cover
    PIL_AVAILABLE = False


EPS = 1e-9


def rotation_matrix(psi: float) -> np.ndarray:
    c = math.cos(psi)
    s = math.sin(psi)
    return np.array([[c, -s], [s, c]], dtype=float)


def to_body(vec_E: np.ndarray, psi: float) -> np.ndarray:
    rot = rotation_matrix(psi)
    return rot.T @ vec_E


def to_world(vec_B: np.ndarray, psi: float) -> np.ndarray:
    rot = rotation_matrix(psi)
    return rot @ vec_B


def unit_vector(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < EPS:
        return np.array([1.0, 0.0], dtype=float)
    return vec / norm


def point_in_polygon(points: np.ndarray, poly: np.ndarray) -> np.ndarray:
    x = points[:, 0][:, None]
    y = points[:, 1][:, None]
    x1 = poly[:, 0][None, :]
    y1 = poly[:, 1][None, :]
    x2 = np.roll(poly[:, 0], -1)[None, :]
    y2 = np.roll(poly[:, 1], -1)[None, :]
    denom = (y2 - y1) + EPS
    cond = ((y1 > y) != (y2 > y)) & (
        x < (x2 - x1) * (y - y1) / denom + x1
    )
    return np.count_nonzero(cond, axis=1) % 2 == 1


def closest_point_on_segment(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    denom = np.dot(ab, ab)
    if denom < EPS:
        return a
    t = np.clip(np.dot(point - a, ab) / denom, 0.0, 1.0)
    return a + t * ab


def closest_point_on_polygon(point: np.ndarray, poly: np.ndarray) -> Tuple[np.ndarray, float]:
    best_pt = poly[0]
    best_dist = np.inf
    for i in range(len(poly)):
        a = poly[i]
        b = poly[(i + 1) % len(poly)]
        candidate = closest_point_on_segment(point, a, b)
        dist = np.linalg.norm(point - candidate)
        if dist < best_dist:
            best_dist = dist
            best_pt = candidate
    return best_pt, best_dist


def rect_intersects_rect(rect: Tuple[float, float, float, float], other: Tuple[float, float, float, float]) -> bool:
    x0, y0, x1, y1 = rect
    u0, v0, u1, v1 = other
    return not (x1 <= u0 or u1 <= x0 or y1 <= v0 or v1 <= y0)


def polygon_bbox(poly: np.ndarray) -> Tuple[float, float, float, float]:
    xmin = float(np.min(poly[:, 0]))
    xmax = float(np.max(poly[:, 0]))
    ymin = float(np.min(poly[:, 1]))
    ymax = float(np.max(poly[:, 1]))
    return xmin, ymin, xmax, ymax


def circle_bbox(cx: float, cy: float, r: float) -> Tuple[float, float, float, float]:
    return cx - r, cy - r, cx + r, cy + r


def _linspace_coords(L: float, n: int) -> np.ndarray:
    return np.linspace(0.0, L, n, dtype=float)


def _polygon_signed_distance(points: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Vectorized signed distance from points batch to polygon boundary."""
    a = poly
    b = np.roll(poly, -1, axis=0)
    seg = b - a
    pma = points[:, None, :] - a[None, :, :]
    seg_len2 = np.sum(seg * seg, axis=1) + EPS
    t = np.clip(np.sum(pma * seg[None, :, :], axis=2) / seg_len2[None, :], 0.0, 1.0)
    proj = a[None, :, :] + t[:, :, None] * seg[None, :, :]
    diff = points[:, None, :] - proj
    dists = np.linalg.norm(diff, axis=2)
    min_d = np.min(dists, axis=1)
    inside = point_in_polygon(points, poly)
    signed = min_d.copy()
    signed[inside] *= -1.0
    return signed


def _circle_signed_distance(points: np.ndarray, circle: Tuple[float, float, float]) -> np.ndarray:
    cx, cy, r = circle
    center = np.array([cx, cy], dtype=float)
    return np.linalg.norm(points - center, axis=1) - r


def _build_sdf(
    polys: List[np.ndarray],
    circles: List[Tuple[float, float, float]],
    x_coords: np.ndarray,
    y_coords: np.ndarray,
) -> np.ndarray:
    X, Y = np.meshgrid(x_coords, y_coords)
    points = np.stack([X.ravel(), Y.ravel()], axis=1)
    sdf_flat = np.full(points.shape[0], np.inf, dtype=float)
    chunk = 8192
    for start in range(0, points.shape[0], chunk):
        stop = min(points.shape[0], start + chunk)
        pts = points[start:stop]
        chunk_vals = np.full(pts.shape[0], np.inf, dtype=float)
        for poly in polys:
            signed = _polygon_signed_distance(pts, poly)
            mask = np.abs(signed) < np.abs(chunk_vals)
            chunk_vals[mask] = signed[mask]
        for circ in circles:
            signed = _circle_signed_distance(pts, circ)
            mask = np.abs(signed) < np.abs(chunk_vals)
            chunk_vals[mask] = signed[mask]
        sdf_flat[start:stop] = chunk_vals
    return sdf_flat.reshape(Y.shape).astype(np.float32)


def sample_static_map(
    L: float,
    nx: int,
    ny: int,
    seed: int,
    n_polys: int = 6,
    n_circles: int = 6,
    keepout_rects: Optional[List[Tuple[float, float, float, float]]] = None,
) -> dict:
    rng = np.random.default_rng(seed)
    keepouts = keepout_rects or []
    polys: List[np.ndarray] = []
    circles: List[Tuple[float, float, float]] = []
    r_min = 0.02 * L
    r_max = 0.08 * L

    def allowed(rect: Tuple[float, float, float, float]) -> bool:
        for ro in keepouts:
            rect_norm = (min(rect[0], rect[2]), min(rect[1], rect[3]), max(rect[0], rect[2]), max(rect[1], rect[3]))
            ro_norm = (min(ro[0], ro[2]), min(ro[1], ro[3]), max(ro[0], ro[2]), max(ro[1], ro[3]))
            if rect_intersects_rect(rect_norm, ro_norm):
                return False
        return True

    attempts = 0
    while len(polys) < n_polys and attempts < n_polys * 20:
        attempts += 1
        center = rng.uniform(r_max, L - r_max, size=2)
        local_margin = min(center[0], center[1], L - center[0], L - center[1])
        if local_margin < r_min:
            continue
        limit = min(r_max, local_margin * 0.9)
        if limit <= r_min:
            continue
        m = rng.integers(4, 9)
        angles = np.sort(rng.uniform(0.0, 2 * math.pi, size=m))
        radii = rng.uniform(r_min, limit, size=m)
        pts = np.stack([np.cos(angles), np.sin(angles)], axis=1) * radii[:, None] + center
        if not allowed(polygon_bbox(pts)):
            continue
        polys.append(pts.astype(float))

    attempts = 0
    while len(circles) < n_circles and attempts < n_circles * 30:
        attempts += 1
        center = rng.uniform(r_max, L - r_max, size=2)
        local_margin = min(center[0], center[1], L - center[0], L - center[1])
        if local_margin < r_min:
            continue
        radius = rng.uniform(r_min, min(r_max, local_margin * 0.9))
        if not allowed(circle_bbox(center[0], center[1], radius)):
            continue
        circles.append((float(center[0]), float(center[1]), float(radius)))

    x_coords = _linspace_coords(L, nx)
    y_coords = _linspace_coords(L, ny)
    sdf = _build_sdf(polys, circles, x_coords, y_coords)
    return {"polys": polys, "circles": circles, "sdf": sdf, "x_coords": x_coords, "y_coords": y_coords}

def _coord_factors(coords: np.ndarray, value: float) -> Tuple[int, int, float]:
    value = float(np.clip(value, coords[0], coords[-1]))
    hi = np.searchsorted(coords, value, side="right")
    hi = min(max(1, hi), len(coords) - 1)
    lo = hi - 1
    span = coords[hi] - coords[lo]
    if span <= 0:
        return lo, hi, 0.0
    t = (value - coords[lo]) / span
    return lo, hi, t


def bilinear_lookup(grid: np.ndarray, x_coords: np.ndarray, y_coords: np.ndarray, point: np.ndarray) -> float:
    x = float(point[0])
    y = float(point[1])
    ix0, ix1, tx = _coord_factors(x_coords, x)
    iy0, iy1, ty = _coord_factors(y_coords, y)
    v00 = grid[iy0, ix0]
    v01 = grid[iy1, ix0]
    v10 = grid[iy0, ix1]
    v11 = grid[iy1, ix1]
    v0 = v00 * (1 - ty) + v01 * ty
    v1 = v10 * (1 - ty) + v11 * ty
    return float(v0 * (1 - tx) + v1 * tx)


class StaticSDF:
    def __init__(
        self,
        sdf: np.ndarray,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        polys: List[np.ndarray],
        circles: List[Tuple[float, float, float]],
    ) -> None:
        self.sdf = sdf.astype(np.float32)
        self.x_coords = x_coords
        self.y_coords = y_coords
        self.polys = polys
        self.circles = circles
        self.dx = x_coords[1] - x_coords[0] if len(x_coords) > 1 else 1.0
        self.dy = y_coords[1] - y_coords[0] if len(y_coords) > 1 else 1.0

    def bilinear(self, p_E: np.ndarray) -> float:
        return bilinear_lookup(self.sdf, self.x_coords, self.y_coords, p_E)

    def grad(self, p_E: np.ndarray) -> np.ndarray:
        x, y = float(p_E[0]), float(p_E[1])
        dx = self.dx
        dy = self.dy
        if x - dx < self.x_coords[0]:
            fx0 = self.bilinear(np.array([x, y], dtype=float))
            fx1 = self.bilinear(np.array([x + dx, y], dtype=float))
            dfdx = (fx1 - fx0) / dx
        elif x + dx > self.x_coords[-1]:
            fx0 = self.bilinear(np.array([x - dx, y], dtype=float))
            fx1 = self.bilinear(np.array([x, y], dtype=float))
            dfdx = (fx1 - fx0) / dx
        else:
            fx0 = self.bilinear(np.array([x - dx, y], dtype=float))
            fx1 = self.bilinear(np.array([x + dx, y], dtype=float))
            dfdx = (fx1 - fx0) / (2 * dx)

        if y - dy < self.y_coords[0]:
            fy0 = self.bilinear(np.array([x, y], dtype=float))
            fy1 = self.bilinear(np.array([x, y + dy], dtype=float))
            dfdy = (fy1 - fy0) / dy
        elif y + dy > self.y_coords[-1]:
            fy0 = self.bilinear(np.array([x, y - dy], dtype=float))
            fy1 = self.bilinear(np.array([x, y], dtype=float))
            dfdy = (fy1 - fy0) / dy
        else:
            fy0 = self.bilinear(np.array([x, y - dy], dtype=float))
            fy1 = self.bilinear(np.array([x, y + dy], dtype=float))
            dfdy = (fy1 - fy0) / (2 * dy)
        return np.array([dfdx, dfdy], dtype=float)

    def _closest_geometry(self, p_E: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        best_dist = np.inf
        best_point = p_E.copy()
        best_grad = np.array([1.0, 0.0], dtype=float)
        for cx, cy, r in self.circles:
            center = np.array([cx, cy], dtype=float)
            diff = p_E - center
            norm = np.linalg.norm(diff)
            if norm < EPS:
                direction = np.array([1.0, 0.0], dtype=float)
                boundary = center + direction * r
                dist = -r
            else:
                direction = diff / norm
                boundary = center + direction * r
                dist = norm - r
            if abs(dist) < abs(best_dist):
                best_dist = dist
                best_point = boundary
                best_grad = direction
        for poly in self.polys:
            boundary, dist_abs = closest_point_on_polygon(p_E, poly)
            inside = point_in_polygon(p_E[None, :], poly)[0]
            dist = -dist_abs if inside else dist_abs
            direction = p_E - boundary
            if abs(dist) > EPS:
                direction = direction / dist
            direction = unit_vector(direction)
            if abs(dist) < abs(best_dist):
                best_dist = dist
                best_point = boundary
                best_grad = direction
        return float(best_dist), best_grad, best_point

    def distance(self, p_E: np.ndarray) -> Tuple[float, np.ndarray]:
        d = float(self.bilinear(p_E))
        grad = self.grad(p_E)
        if np.linalg.norm(grad) < 1e-6:
            d_geo, grad_geo, _ = self._closest_geometry(p_E)
            return d_geo, grad_geo
        return d, grad

    def closest_point(self, p_E: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        d = float(self.bilinear(p_E))
        grad = self.grad(p_E)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-6:
            return self._closest_geometry(p_E)
        grad_unit = grad / max(grad_norm, EPS)
        p_star = p_E - grad_unit * d
        return d, grad_unit, p_star

class DynamicCatalog:
    def __init__(self, N_max: int, L: float, seed: Optional[int] = None) -> None:
        self.N_max = N_max
        self.L = L
        self.p_E = np.zeros((N_max, 2), dtype=float)
        self.vel_E = np.zeros((N_max, 2), dtype=float)
        self.r = np.zeros(N_max, dtype=float)
        self.R_trig = np.zeros(N_max, dtype=float)
        self.ttl = np.zeros(N_max, dtype=float)
        self.active = np.zeros(N_max, dtype=bool)
        self.class_id = np.zeros(N_max, dtype=int)
        self.rng = np.random.default_rng(seed)

    def _first_free(self) -> Optional[int]:
        free = np.where(~self.active)[0]
        return int(free[0]) if free.size else None

    def spawn_mine(
        self,
        p_E: np.ndarray,
        r: float,
        R_trig: float,
        ttl: float,
        vel_E: np.ndarray,
        class_id: int = 1,
    ) -> Optional[int]:
        idx = self._first_free()
        if idx is None:
            return None
        self.p_E[idx] = p_E
        self.r[idx] = float(r)
        self.R_trig[idx] = float(R_trig)
        self.ttl[idx] = float(ttl)
        self.vel_E[idx] = vel_E
        self.active[idx] = True
        self.class_id[idx] = class_id
        return idx

    def spawn_minefield(
        self,
        center_E: np.ndarray,
        rad: float,
        k: int,
        r_child: float,
        R_trig_child: float,
        ttl_child: float,
        drift_vel_E: np.ndarray,
    ) -> List[int]:
        spawned: List[int] = []
        for _ in range(k):
            rho = rad * math.sqrt(self.rng.random())
            theta = self.rng.uniform(0.0, 2 * math.pi)
            offset = np.array([rho * math.cos(theta), rho * math.sin(theta)], dtype=float)
            idx = self.spawn_mine(center_E + offset, r_child, R_trig_child, ttl_child, drift_vel_E, class_id=2)
            if idx is not None:
                spawned.append(idx)
        return spawned

    def update(self, dt: float) -> None:
        mask = self.active
        if not np.any(mask):
            return
        self.p_E[mask] += self.vel_E[mask] * dt
        self.ttl[mask] -= dt
        self.p_E[mask] = np.clip(self.p_E[mask], 0.0, self.L)
        expired = (self.ttl <= 0.0) & mask
        self.active[expired] = False

    def nearest_distance(self, p_E_query: np.ndarray) -> Tuple[float, int]:
        mask = self.active
        if not np.any(mask):
            return float(np.inf), -1
        centers = self.p_E[mask]
        radii = self.r[mask]
        diffs = centers - p_E_query
        dists = np.linalg.norm(diffs, axis=1) - radii
        idx_local = int(np.argmin(dists))
        idx_global = np.where(mask)[0][idx_local]
        return float(dists[idx_local]), idx_global


def min_distance_to_dynamic(
    p_E: np.ndarray,
    dyn: DynamicCatalog,
    inflated: bool,
    v_rel_max: float,
    dt_RL: float,
) -> Tuple[float, int, np.ndarray]:
    mask = dyn.active
    if not np.any(mask):
        return float(np.inf), -1, p_E.copy()
    indices = np.where(mask)[0]
    centers = dyn.p_E[indices]
    radii = dyn.r[indices]
    trig = dyn.R_trig[indices]
    diff = centers - p_E
    ranges = np.linalg.norm(diff, axis=1)
    if inflated:
        R_eff = np.maximum(trig + v_rel_max * dt_RL, radii)
    else:
        R_eff = radii
    dists = ranges - R_eff
    best = int(np.argmin(dists))
    idx_global = int(indices[best])
    center = centers[best]
    geom_radius = radii[best]
    vec = p_E - center
    norm = np.linalg.norm(vec)
    direction = np.array([1.0, 0.0], dtype=float) if norm < EPS else vec / norm
    closest_point = center + direction * geom_radius
    return float(dists[best]), idx_global, closest_point


def min_distance_all(
    p_E: np.ndarray,
    static: StaticSDF,
    dyn: DynamicCatalog,
    inflated: bool,
    v_rel_max: float,
    dt_RL: float,
) -> dict:
    d_static, grad_static = static.distance(p_E)
    d_dyn, idx_dyn, p_dyn = min_distance_to_dynamic(p_E, dyn, inflated, v_rel_max, dt_RL)
    return {
        "d_static": d_static,
        "grad_static_E": grad_static,
        "d_dyn": d_dyn,
        "idx_dyn": idx_dyn,
        "p_dyn_closest_E": p_dyn,
        "d_all": min(d_static, d_dyn),
    }


def topN_features(
    p_E: np.ndarray,
    psi: float,
    static: StaticSDF,
    dyn: DynamicCatalog,
    N: int,
    inflated: bool,
    v_rel_max: float,
    dt_RL: float,
) -> Tuple[np.ndarray, np.ndarray]:
    feats: List[Tuple[float, np.ndarray]] = []
    d_static, grad_static, p_star = static.closest_point(p_E)
    rel_static = p_star - p_E
    rel_static_B = to_body(rel_static, psi)
    feats.append((abs(d_static), np.array([rel_static_B[0], rel_static_B[1], 0.0, 0.0, 0.0, 0.0])))
    mask = dyn.active
    indices = np.where(mask)[0]
    for idx in indices:
        center = dyn.p_E[idx]
        radius = dyn.r[idx]
        vec = p_E - center
        dist_center = np.linalg.norm(vec)
        direction = np.array([1.0, 0.0], dtype=float) if dist_center < EPS else vec / dist_center
        closest_point = center + direction * radius
        rel = closest_point - p_E
        rel_B = to_body(rel, psi)
        vel_B = to_body(dyn.vel_E[idx], psi)
        R_eff = dyn.R_trig[idx] + v_rel_max * dt_RL if inflated else dyn.r[idx]
        feats.append((abs(dist_center - radius), np.array([
            rel_B[0], rel_B[1], vel_B[0], vel_B[1], R_eff, float(dyn.class_id[idx])
        ])))
    feats.sort(key=lambda item: item[0])
    obs_feats = np.zeros((N, 6), dtype=float)
    mask_array = np.zeros(N, dtype=float)
    for i in range(min(N, len(feats))):
        obs_feats[i] = feats[i][1]
        mask_array[i] = 1.0
    return obs_feats, mask_array


def nearestK_for_mpc(
    p_E: np.ndarray,
    static: StaticSDF,
    dyn: DynamicCatalog,
    K: int,
    d_safe: float,
    inflated: bool,
    v_rel_max: float,
    dt_RL: float,
) -> dict:
    entries: List[Tuple[float, dict]] = []
    d_static, grad_static, p_star_static = static.closest_point(p_E)
    n_static = unit_vector(grad_static)
    entries.append((abs(d_static), {
        "d": d_static,
        "xi": max(0.0, d_safe - d_static),
        "n": n_static,
        "p_star": p_star_static,
        "type": 0,
        "idx": -1,
    }))
    mask = dyn.active
    idxs = np.where(mask)[0]
    for idx in idxs:
        center = dyn.p_E[idx]
        radius = dyn.r[idx]
        vec = p_E - center
        norm = np.linalg.norm(vec)
        if norm < EPS:
            direction = np.array([1.0, 0.0], dtype=float)
            closest_point = center + direction * radius
            dist = -radius
        else:
            direction = vec / norm
            closest_point = center + direction * radius
            dist = norm - (dyn.R_trig[idx] + v_rel_max * dt_RL if inflated else radius)
        n = unit_vector(direction)
        entries.append((abs(dist), {
            "d": dist,
            "xi": max(0.0, d_safe - dist),
            "n": n,
            "p_star": closest_point,
            "type": 1,
            "idx": int(idx),
        }))
    entries.sort(key=lambda e: e[0])
    d = np.full(K, np.inf, dtype=float)
    xi = np.zeros(K, dtype=float)
    n_E = np.tile(np.array([[1.0, 0.0]], dtype=float), (K, 1))
    p_star = np.tile(p_E[None, :], (K, 1))
    types = np.zeros(K, dtype=int)
    idx_arr = -np.ones(K, dtype=int)
    for i in range(min(K, len(entries))):
        entry = entries[i][1]
        d[i] = entry["d"]
        xi[i] = entry["xi"]
        n_E[i] = entry["n"]
        p_star[i] = entry["p_star"]
        types[i] = entry["type"]
        idx_arr[i] = entry["idx"]
    return {"d": d, "xi": xi, "n_E": n_E, "p_star_E": p_star, "types": types, "idx": idx_arr}

def sample_start_goal(
    L: float,
    rng: np.random.Generator,
    start_rect: Tuple[float, float, float, float],
    goal_rect: Tuple[float, float, float, float],
    heading_to_goal: bool = True,
) -> Tuple[np.ndarray, float, np.ndarray]:
    start = rng.uniform([start_rect[0], start_rect[1]], [start_rect[2], start_rect[3]])
    goal = rng.uniform([goal_rect[0], goal_rect[1]], [goal_rect[2], goal_rect[3]])
    if heading_to_goal:
        vec = goal - start
        if np.linalg.norm(vec) < EPS:
            psi0 = rng.uniform(-math.pi, math.pi)
        else:
            psi0 = math.atan2(vec[1], vec[0])
    else:
        psi0 = rng.uniform(-math.pi, math.pi)
    return start, psi0, goal


class ObstacleWorld:
    def __init__(
        self,
        L: float = 10_000.0,
        nx: int = 401,
        ny: int = 401,
        seed: int = 0,
        n_polys: int = 6,
        n_circles: int = 6,
        N_dyn_max: int = 64,
        lambda_spawn: float = 0.02,
        sector_deg: float = 45.0,
        d_spawn_min: float = 80.0,
        d_spawn_max: float = 400.0,
        k_drift: float = 1.0,
        dt_RL: float = 0.2,
        v_rel_max: float = 15.0,
        mine_ttl: float = 120.0,
        mine_r: float = 6.0,
        mine_Rtrig: float = 12.0,
        minefield_prob: float = 0.05,
        minefield_rad: float = 60.0,
        minefield_k: int = 6,
        mine_child_r: float = 4.0,
        mine_child_Rtrig: float = 10.0,
        mine_child_ttl: float = 90.0,
    ) -> None:
        self.L = L
        self.lambda_spawn = lambda_spawn
        self.sector_deg = sector_deg
        self.d_spawn_min = d_spawn_min
        self.d_spawn_max = d_spawn_max
        self.k_drift = k_drift
        self.dt_RL = dt_RL
        self.v_rel_max = v_rel_max
        self.mine_params = {"ttl": mine_ttl, "r": mine_r, "R_trig": mine_Rtrig}
        self.minefield_prob = minefield_prob
        self.minefield_specs = {
            "rad": minefield_rad,
            "k": minefield_k,
            "r_child": mine_child_r,
            "R_trig_child": mine_child_Rtrig,
            "ttl_child": mine_child_ttl,
        }
        self.rng = np.random.default_rng(seed)
        static_map = sample_static_map(L, nx, ny, seed, n_polys, n_circles, keepout_rects=[])
        self.static = StaticSDF(static_map["sdf"], static_map["x_coords"], static_map["y_coords"], static_map["polys"], static_map["circles"])
        dyn_seed = int(self.rng.integers(0, 2**32 - 1, dtype=np.uint32))
        self.dyn = DynamicCatalog(N_dyn_max, L, seed=dyn_seed)

    def reset(
        self,
        start_rect: Tuple[float, float, float, float],
        goal_rect: Tuple[float, float, float, float],
        rng: Optional[np.random.Generator] = None,
    ) -> dict:
        worker = rng or self.rng
        seed = worker.integers(0, 2**32 - 1, dtype=np.uint32)
        static_map = sample_static_map(
            self.L,
            self.static.sdf.shape[1],
            self.static.sdf.shape[0],
            int(seed),
            len(self.static.polys) or 6,
            len(self.static.circles) or 6,
        )
        self.static = StaticSDF(static_map["sdf"], static_map["x_coords"], static_map["y_coords"], static_map["polys"], static_map["circles"])
        dyn_seed = int(worker.integers(0, 2**32 - 1, dtype=np.uint32))
        self.dyn = DynamicCatalog(self.dyn.N_max, self.L, seed=dyn_seed)
        start, psi0, goal = sample_start_goal(self.L, worker, start_rect, goal_rect)
        return {"p0_E": start, "psi0": psi0, "goal_E": goal}

    def _current_or_zero(self, current_E_fn: Optional[Callable[[float, float, float], np.ndarray]]) -> Callable[[float, float, float], np.ndarray]:
        if current_E_fn is None:
            return lambda x, y, t: np.zeros(2, dtype=float)
        return current_E_fn

    def step(
        self,
        t: float,
        p_E: np.ndarray,
        psi: float,
        v_B: np.ndarray,
        current_E_fn: Optional[Callable[[float, float, float], np.ndarray]],
        dt: float,
    ) -> None:
        current_fn = self._current_or_zero(current_E_fn)
        spawn_count = self.rng.poisson(self.lambda_spawn * dt)
        for _ in range(spawn_count):
            heading_offset = math.radians(self.rng.uniform(-self.sector_deg, self.sector_deg))
            direction_B = np.array([math.cos(heading_offset), math.sin(heading_offset)], dtype=float)
            dist = self.rng.uniform(self.d_spawn_min, self.d_spawn_max)
            offset_E = to_world(direction_B * dist, psi)
            center = p_E + offset_E
            if not (0.0 <= center[0] <= self.L and 0.0 <= center[1] <= self.L):
                continue
            drift = self.k_drift * current_fn(center[0], center[1], t)
            if self.rng.random() < 0.5:
                self.dyn.spawn_mine(center, self.mine_params["r"], self.mine_params["R_trig"], self.mine_params["ttl"], drift, class_id=1)
            if self.rng.random() < self.minefield_prob:
                far_dist = self.rng.uniform(self.d_spawn_max, min(self.L, self.d_spawn_max * 1.5))
                far_center = p_E + to_world(direction_B * far_dist, psi)
                if 0.0 <= far_center[0] <= self.L and 0.0 <= far_center[1] <= self.L:
                    specs = self.minefield_specs
                    self.dyn.spawn_minefield(
                        far_center,
                        specs["rad"],
                        specs["k"],
                        specs["r_child"],
                        specs["R_trig_child"],
                        specs["ttl_child"],
                        drift,
                    )
        active_idx = np.where(self.dyn.active)[0]
        for idx in active_idx:
            pos = self.dyn.p_E[idx]
            self.dyn.vel_E[idx] = self.k_drift * current_fn(pos[0], pos[1], t)
        self.dyn.update(dt)

    def query(
        self,
        p_E: np.ndarray,
        psi: float,
        N: int = 8,
        K: int = 4,
        d_safe: float = 12.0,
        inflated: bool = True,
    ) -> dict:
        info = min_distance_all(p_E, self.static, self.dyn, inflated, self.v_rel_max, self.dt_RL)
        obs_feats, obs_mask = topN_features(p_E, psi, self.static, self.dyn, N, inflated, self.v_rel_max, self.dt_RL)
        mpc_pack = nearestK_for_mpc(p_E, self.static, self.dyn, K, d_safe, inflated, self.v_rel_max, self.dt_RL)
        collision_static = info["d_static"] < 0.0
        collision_dyn = inflated and info["d_dyn"] < 0.0
        out_of_bounds = not (0.0 <= p_E[0] <= self.L and 0.0 <= p_E[1] <= self.L)
        return {
            "d_static": info["d_static"],
            "d_dyn": info["d_dyn"],
            "d_all": info["d_all"],
            "obs_feats": obs_feats,
            "obs_mask": obs_mask,
            "mpc_pack": mpc_pack,
            "collision_static": collision_static,
            "collision_dyn": collision_dyn,
            "out_of_bounds": out_of_bounds,
        }


def draw_static_map(static: StaticSDF, path: Path, dyn: Optional[DynamicCatalog] = None, size: int = 1024) -> None:
    if not PIL_AVAILABLE:
        print("Pillow not available; skipping drawing.", file=sys.stderr)
        return
    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)
    L = float(static.x_coords[-1])
    scale = size / L

    def world_to_img(point: np.ndarray) -> Tuple[float, float]:
        x = point[0] * scale
        y = size - point[1] * scale
        return x, y

    for poly in static.polys:
        pts = [world_to_img(p) for p in poly]
        draw.polygon(pts, fill="#d0e1f9", outline="#6b8fb4")
    for cx, cy, r in static.circles:
        lo = world_to_img(np.array([cx - r, cy - r]))
        hi = world_to_img(np.array([cx + r, cy + r]))
        draw.ellipse([lo[0], hi[1], hi[0], lo[1]], fill="#fddfb1", outline="#f4a259")
    if dyn is not None and np.any(dyn.active):
        for idx in np.where(dyn.active)[0]:
            cx, cy = dyn.p_E[idx]
            r = dyn.r[idx]
            lo = world_to_img(np.array([cx - r, cy - r]))
            hi = world_to_img(np.array([cx + r, cy + r]))
            color = "#d62828" if dyn.class_id[idx] == 1 else "#f77f00"
            draw.ellipse([lo[0], hi[1], hi[0], lo[1]], outline=color, width=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="USV obstacle utilities")
    parser.add_argument("--L", type=float, default=10_000.0)
    parser.add_argument("--nx", type=int, default=401)
    parser.add_argument("--ny", type=int, default=401)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-polys", type=int, default=6)
    parser.add_argument("--n-circles", type=int, default=6)
    parser.add_argument("--N-dyn-max", type=int, default=64)
    parser.add_argument("--lambda-spawn", type=float, default=0.02)
    parser.add_argument("--sector-deg", type=float, default=45.0)
    parser.add_argument("--d-spawn-min", type=float, default=80.0)
    parser.add_argument("--d-spawn-max", type=float, default=400.0)
    parser.add_argument("--k-drift", type=float, default=1.0)
    parser.add_argument("--dt-RL", type=float, default=0.2)
    parser.add_argument("--v-rel-max", type=float, default=15.0)
    parser.add_argument("--start-rect", nargs=4, type=float, metavar=("x0", "y0", "x1", "y1"), default=(100.0, 100.0, 400.0, 400.0))
    parser.add_argument("--goal-rect", nargs=4, type=float, metavar=("x0", "y0", "x1", "y1"), default=(9_000.0, 9_000.0, 9_500.0, 9_500.0))
    parser.add_argument("--N", type=int, default=8)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--d-safe", type=float, default=12.0)
    parser.add_argument("--make-map", type=str, default=None)
    parser.add_argument("--sample", nargs=7, type=float, metavar=("x", "y", "psi_deg", "t", "u", "v", "r"), default=None)
    parser.add_argument("--inflated", action="store_true")
    parser.add_argument("--draw-snapshot", type=str, default=None)
    return parser.parse_args(argv)


def _demo_current(x: float, y: float, t: float) -> np.ndarray:
    angle = 0.0002 * (x + y) + 0.1 * t
    return np.array([0.5 * math.sin(angle), 0.5 * math.cos(angle)], dtype=float)


def simulate_roll(world: ObstacleWorld, steps: int = 50, current_fn: Optional[Callable[[float, float, float], np.ndarray]] = None) -> None:
    L = world.L
    p = np.array([0.5 * L, 0.5 * L], dtype=float)
    psi = 0.0
    v_B = np.zeros(2, dtype=float)
    dt = world.dt_RL
    current_fn = current_fn or _demo_current
    for k in range(steps):
        world.step(k * dt, p, psi, v_B, current_fn, dt)
        psi += 0.02


def print_sample(world: ObstacleWorld, args: argparse.Namespace) -> None:
    sample = args.sample
    if sample is None:
        return
    x, y, psi_deg, t, u, v, r = sample
    p = np.array([x, y], dtype=float)
    psi = math.radians(psi_deg)
    report = world.query(p, psi, args.N, args.K, args.d_safe, args.inflated)
    print(f"d_static={report['d_static']:.3f} d_dyn={report['d_dyn']:.3f} d_all={report['d_all']:.3f}")
    print(f"collisions static={report['collision_static']} dyn={report['collision_dyn']} oob={report['out_of_bounds']}")
    feats = report["obs_feats"]
    mask = report["obs_mask"]
    for i in range(args.N):
        row = feats[i]
        print(
            f"obs[{i}] mask={mask[i]:.0f} vec_B=({row[0]:.2f},{row[1]:.2f}) vel_B=({row[2]:.2f},{row[3]:.2f}) R_eff={row[4]:.2f} class={row[5]:.0f}"
        )
    pack = report["mpc_pack"]
    for i in range(args.K):
        d = pack["d"][i]
        xi = pack["xi"][i]
        n = pack["n_E"][i]
        print(f"mpc[{i}] d={d:.3f} xi={xi:.3f} n=({n[0]:.2f},{n[1]:.2f}) type={pack['types'][i]} idx={pack['idx'][i]}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    world = ObstacleWorld(
        L=args.L,
        nx=args.nx,
        ny=args.ny,
        seed=args.seed,
        n_polys=args.n_polys,
        n_circles=args.n_circles,
        N_dyn_max=args.N_dyn_max,
        lambda_spawn=args.lambda_spawn,
        sector_deg=args.sector_deg,
        d_spawn_min=args.d_spawn_min,
        d_spawn_max=args.d_spawn_max,
        k_drift=args.k_drift,
        dt_RL=args.dt_RL,
        v_rel_max=args.v_rel_max,
    )
    if args.make_map:
        draw_static_map(world.static, Path(args.make_map))
        return
    print_sample(world, args)
    simulate_roll(world, steps=50, current_fn=_demo_current)
    snapshot_path = args.draw_snapshot or "outputs/snapshot.png"
    draw_static_map(world.static, Path(snapshot_path), dyn=world.dyn)


if __name__ == "__main__":
    main()
