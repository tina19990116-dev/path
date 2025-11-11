"""
Visualization utilities for obstacles, training curves, and attention weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np

COLORBLIND = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9"]


def render_map_snapshot(save_path: str | Path, sim, world, L: float, dpi: int = 220, draw_wind: bool = True) -> None:
    """
    Render static map, dynamic obstacles, and coarse flow fields.

    Parameters
    ----------
    save_path : str or Path
        Output image path.
    sim : MarineEnvironmentSimulation
        Environment simulator for currents/wind.
    world : ObstacleWorld
        Obstacle catalog.
    L : float
        Map extent [m].
    dpi : int
        Figure dots-per-inch.
    """

    save_path = Path(save_path)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
    ax.set_title("USV Map Snapshot")
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect("equal")

    static = getattr(world, "static", None)
    if static is not None:
        for poly in getattr(static, "polys", []):
            ax.fill(poly[:, 0], poly[:, 1], alpha=0.2, color=COLORBLIND[0], linewidth=0.8)
        for cx, cy, r in getattr(static, "circles", []):
            circ = plt.Circle((cx, cy), r, color=COLORBLIND[1], alpha=0.2)
            ax.add_patch(circ)

    dyn = getattr(world, "dyn", None)
    if dyn is not None and np.any(dyn.active):
        centers = dyn.p_E[dyn.active]
        radii = dyn.r[dyn.active]
        ax.scatter(centers[:, 0], centers[:, 1], s=15, color=COLORBLIND[2], label="Dynamic")
        for c, r in zip(centers, radii):
            circ = plt.Circle((c[0], c[1]), r, color=COLORBLIND[2], alpha=0.1)
            ax.add_patch(circ)

    grid = np.linspace(0, L, 12)
    X, Y = np.meshgrid(grid, grid)
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    WX = np.zeros_like(X)
    WY = np.zeros_like(Y)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            vec_c = sim.get_current_B(float(X[i, j]), float(Y[i, j]), 0.0, 0.0)
            U[i, j], V[i, j] = vec_c[0], vec_c[1]
            if draw_wind:
                try:
                    vec_w = sim.get_wind_B(float(X[i, j]), float(Y[i, j]), 0.0, 0.0)
                except AttributeError:
                    vec_w = np.zeros(2, dtype=float)
                WX[i, j], WY[i, j] = vec_w[0], vec_w[1]
    ax.quiver(X, Y, U, V, color=COLORBLIND[3], alpha=0.6, scale=60, label="Current")
    if draw_wind:
        ax.quiver(X, Y, WX, WY, color=COLORBLIND[4], alpha=0.5, scale=60, label="Wind")

    ax.legend(loc="upper right")
    ax.set_xlabel("x_E [m]")
    ax.set_ylabel("y_E [m]")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def plot_training_curves(log_csv: str | Path, save_dir: str | Path) -> None:
    """
    Plot scalar curves from a CSV summary created during training.

    The CSV must contain a header with ``step`` and additional scalar columns.
    """

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    data = np.genfromtxt(log_csv, delimiter=",", names=True)
    steps = data["step"]
    for name in data.dtype.names:
        if name == "step":
            continue
        fig, ax = plt.subplots(figsize=(6, 3), dpi=220)
        ax.plot(steps, data[name], color=COLORBLIND[0], linewidth=1.4)
        ax.set_xlabel("Global step")
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_dir / f"{name}.png")
        plt.close(fig)


def draw_attention_overlay(
    trajectory: np.ndarray,
    attn_weights: np.ndarray,
    world,
    save_path: str | Path,
) -> None:
    """
    Overlay attention weights on obstacle map.

    Parameters
    ----------
    trajectory : ndarray
        Positions ``[T, 2]`` in metres.
    attn_weights : ndarray
        Attention weights ``[heads, N]`` normalized.
    world : ObstacleWorld
        For drawing context.
    save_path : str or Path
        PNG output.
    """

    save_path = Path(save_path)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=220)
    ax_map, ax_bar = axes
    static = getattr(world, "static", None)
    if static is not None:
        for poly in getattr(static, "polys", []):
            ax_map.fill(poly[:, 0], poly[:, 1], alpha=0.15, color=COLORBLIND[0])
        for cx, cy, r in getattr(static, "circles", []):
            ax_map.add_patch(plt.Circle((cx, cy), r, color=COLORBLIND[1], alpha=0.1))
    ax_map.plot(trajectory[:, 0], trajectory[:, 1], color=COLORBLIND[2], linewidth=2.0)
    ax_map.set_aspect("equal")
    ax_map.set_title("Attention Overlay")
    ax_map.set_xlabel("x_E [m]")
    ax_map.set_ylabel("y_E [m]")

    heads = attn_weights.shape[0]
    ind = np.arange(attn_weights.shape[1])
    bottom = np.zeros_like(ind, dtype=float)
    for h in range(heads):
        ax_bar.bar(ind, attn_weights[h], bottom=bottom, color=COLORBLIND[h % len(COLORBLIND)], alpha=0.7, label=f"H{h}")
        bottom += attn_weights[h]
    ax_bar.set_title("Per-token Attention")
    ax_bar.set_xlabel("Token index")
    ax_bar.set_ylabel("Weight")
    ax_bar.legend()
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
