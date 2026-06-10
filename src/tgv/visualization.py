"""
Visualization module for the 2D Convecting Taylor-Green Vortex.

Produces:
  1. Flow field snapshots  — u, v, vorticity, pressure, speed, error panels
  2. Energy decay curve    — E(t) vs exact formula
  3. Animated GIF          — time evolution of the vortex

All functions accept a `save_path` kwarg; if None they show interactively.
"""

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for script use
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from pathlib import Path

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tgv.analytical import (
    make_grid, velocity_exact, pressure_ic, kinetic_energy_exact,
    kinetic_energy_field, l2_error, velocity_ic,
)

# ── Shared style ──────────────────────────────────────────────────────────────

FIGSIZE_PANEL  = (18, 10)
FIGSIZE_ENERGY = (8, 5)
FIGSIZE_ANIM   = (7, 6)
DPI            = 150
CMAP_VEL       = "RdBu_r"
CMAP_VORT      = "PiYG"
CMAP_PRESS     = "coolwarm"
CMAP_SPEED     = "viridis"
CMAP_ERROR     = "hot_r"


def _vorticity(x, y, t, *, L, V0, Uc, Vc, nu):
    """Exact vorticity ω = ∂v/∂x − ∂u/∂y."""
    decay = math.exp(-2.0 * nu * t / L**2)
    xc = x - Uc * t
    yc = y - Vc * t
    return 2.0 * V0 / L * np.sin(xc / L) * np.sin(yc / L) * decay


def _speed(u, v, *, Uc=1.0, Vc=0.0):
    """Fluctuation speed (background-flow subtracted)."""
    return np.sqrt((u - Uc)**2 + (v - Vc)**2)


def _add_panel(ax, data, title, cmap, x, y, symmetric=False, label=""):
    vmax = np.abs(data).max()
    vmin = -vmax if symmetric else data.min()
    vmax_plot = vmax if symmetric else data.max()
    im = ax.pcolormesh(x, y, data.T, cmap=cmap, vmin=vmin, vmax=vmax_plot,
                       shading="auto", rasterized=True)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=label)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")


# ── 1. Flow field snapshot ────────────────────────────────────────────────────

def plot_flow_field(
    t: float,
    Re: float,
    N: int = 128,
    *,
    L: float = 2 * math.pi,
    V0: float = 1.0,
    Uc: float = 1.0,
    Vc: float = 0.0,
    rho: float = 1.0,
    p0: float = 0.0,
    save_path: str | None = None,
    title_prefix: str = "Exact solution",
    u_sim: np.ndarray | None = None,
    v_sim: np.ndarray | None = None,
    solver_label: str = "",
):
    """
    Six-panel flow field figure:
      Row 1: u velocity | v velocity | vorticity ω
      Row 2: pressure p | speed |u'| | L2 error (if u_sim provided)

    Parameters
    ----------
    u_sim, v_sim : optional simulated velocity fields for the error panel
    """
    nu = V0 * L / Re
    x, y = make_grid(N, L)

    u, v   = velocity_exact(x, y, t=t, L=L, V0=V0, Uc=Uc, Vc=Vc, nu=nu)
    p      = pressure_ic(x, y, L=L, V0=V0, rho=rho, p0=p0)   # t=0 reference
    omega  = _vorticity(x, y, t, L=L, V0=V0, Uc=Uc, Vc=Vc, nu=nu)
    speed  = _speed(u, v, Uc=Uc, Vc=Vc)

    fig = plt.figure(figsize=FIGSIZE_PANEL, dpi=DPI)
    fig.suptitle(
        f"{title_prefix}  —  Re = {Re:.0f},  t = {t:.2f}",
        fontsize=14, fontweight="bold", y=1.01,
    )
    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    _add_panel(fig.add_subplot(gs[0, 0]), u, "u  (x-velocity)", CMAP_VEL, x, y,
               symmetric=True, label="m/s")
    _add_panel(fig.add_subplot(gs[0, 1]), v, "v  (y-velocity)", CMAP_VEL, x, y,
               symmetric=True, label="m/s")
    _add_panel(fig.add_subplot(gs[0, 2]), omega, "ω  (vorticity = ∂v/∂x − ∂u/∂y)",
               CMAP_VORT, x, y, symmetric=True, label="1/s")
    _add_panel(fig.add_subplot(gs[1, 0]), p, "p  (pressure)", CMAP_PRESS, x, y,
               label="Pa")
    _add_panel(fig.add_subplot(gs[1, 1]), speed, "|u'|  (fluctuation speed)", CMAP_SPEED,
               x, y, label="m/s")

    ax_err = fig.add_subplot(gs[1, 2])
    if u_sim is not None and v_sim is not None:
        err = np.sqrt((u_sim - u)**2 + (v_sim - v)**2)
        _add_panel(ax_err, err,
                   f"L2 error  |{solver_label} − exact|",
                   CMAP_ERROR, x, y, label="m/s")
        ax_err.set_title(ax_err.get_title(), color="crimson")
    else:
        decay = math.exp(-2.0 * nu * t / L**2)
        ax_err.text(0.5, 0.5,
                    f"Decay factor\nexp(−2νt/L²)\n= {decay:.4f}",
                    ha="center", va="center", fontsize=14,
                    transform=ax_err.transAxes,
                    bbox=dict(boxstyle="round", facecolor="lightyellow",
                              edgecolor="orange", linewidth=2))
        ax_err.set_axis_off()

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=DPI, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ── 2. Energy decay curve ─────────────────────────────────────────────────────

def plot_energy_decay(
    Re_list: list[float],
    t_end: float = 2.0,
    N_t: int = 200,
    N: int = 64,
    *,
    L: float = 2 * math.pi,
    V0: float = 1.0,
    save_path: str | None = None,
):
    """Plot E(t) exact formula for multiple Re values."""
    ts = np.linspace(0.0, t_end, N_t)

    fig, ax = plt.subplots(figsize=FIGSIZE_ENERGY, dpi=DPI)

    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(Re_list)))
    for Re, color in zip(Re_list, colors):
        nu = V0 * L / Re
        E  = [kinetic_energy_exact(t, L=L, V0=V0, nu=nu) / L**2 for t in ts]
        ax.semilogy(ts, E, color=color, linewidth=2.0, label=f"Re = {Re:.0f}")

    ax.set_xlabel("t  (seconds)", fontsize=12)
    ax.set_ylabel("Mean kinetic energy  E(t) / L²  [m²/s²]", fontsize=12)
    ax.set_title("Exact kinetic energy decay — 2D Convecting TGV", fontsize=13,
                 fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, which="both", alpha=0.3)
    ax.annotate("E(t) = (V₀²/4) · exp(−4νt/L²)",
                xy=(0.6, 0.82), xycoords="axes fraction",
                fontsize=10, color="gray",
                bbox=dict(boxstyle="round", facecolor="white",
                          edgecolor="lightgray"))

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=DPI, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ── 3. Animated GIF — vortex evolution ───────────────────────────────────────

def animate_vortex(
    Re: float,
    t_end: float = 2.0,
    n_frames: int = 40,
    N: int = 96,
    *,
    L: float = 2 * math.pi,
    V0: float = 1.0,
    Uc: float = 1.0,
    Vc: float = 0.0,
    fps: int = 10,
    save_path: str | None = None,
):
    """
    Animated GIF: vorticity field evolving over time.
    Shows both the advection (translation) and viscous decay.
    """
    nu = V0 * L / Re
    ts = np.linspace(0.0, t_end, n_frames)
    x, y = make_grid(N, L)

    # Pre-compute all frames
    frames = [_vorticity(x, y, t, L=L, V0=V0, Uc=Uc, Vc=Vc, nu=nu) for t in ts]
    vmax = max(np.abs(f).max() for f in frames) * 1.05

    fig, ax = plt.subplots(figsize=FIGSIZE_ANIM, dpi=DPI)
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    mesh = ax.pcolormesh(x, y, frames[0].T, cmap="RdBu_r",
                         vmin=-vmax, vmax=vmax, shading="auto", rasterized=True)
    cbar = plt.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Vorticity ω  [1/s]", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_xlabel("x", color="white")
    ax.set_ylabel("y", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("white")

    title = ax.set_title("", color="white", fontsize=11, fontweight="bold")
    decay_text = ax.text(0.02, 0.97, "", transform=ax.transAxes,
                         color="white", fontsize=9, va="top",
                         bbox=dict(facecolor="#0d1117", alpha=0.7, edgecolor="none"))

    def update(frame_idx):
        t = ts[frame_idx]
        decay = math.exp(-2.0 * nu * t / L**2)
        mesh.set_array(frames[frame_idx].T.ravel())
        title.set_text(f"2D Convecting TGV — Vorticity Field\nRe = {Re:.0f},  t = {t:.2f} s")
        decay_text.set_text(f"Amplitude decay: {decay:.3f}\n(exp(-2νt/L²))")
        return mesh, title, decay_text

    ani = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=1000 // fps, blit=True
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        writer = animation.PillowWriter(fps=fps)
        ani.save(save_path, writer=writer)
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ── 4. Multi-time comparison strip ───────────────────────────────────────────

def plot_time_strip(
    Re: float,
    times: list[float] | None = None,
    N: int = 128,
    field: str = "vorticity",
    *,
    L: float = 2 * math.pi,
    V0: float = 1.0,
    Uc: float = 1.0,
    Vc: float = 0.0,
    save_path: str | None = None,
):
    """
    Horizontal strip showing one field at multiple time snapshots.
    `field` ∈ {'vorticity', 'u', 'v', 'speed'}
    """
    if times is None:
        times = [0.0, 0.25, 0.5, 1.0, 2.0]

    nu = V0 * L / Re
    x, y = make_grid(N, L)

    cmaps   = {"vorticity": "RdBu_r", "u": "RdBu_r", "v": "RdBu_r", "speed": "plasma"}
    titles_ = {"vorticity": "ω  (vorticity)", "u": "u  (x-velocity)",
                "v": "v  (y-velocity)", "speed": "|u'|  (speed)"}

    def _get(t):
        u_, v_ = velocity_exact(x, y, t=t, L=L, V0=V0, Uc=Uc, Vc=Vc, nu=nu)
        if field == "vorticity":
            return _vorticity(x, y, t, L=L, V0=V0, Uc=Uc, Vc=Vc, nu=nu)
        if field == "speed":
            return _speed(u_, v_, Uc=Uc, Vc=Vc)
        return u_ if field == "u" else v_

    data_list = [_get(t) for t in times]
    vmax = max(np.abs(d).max() for d in data_list)

    n = len(times)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.5), dpi=DPI)
    fig.suptitle(
        f"2D Convecting TGV — {titles_[field]}   (Re = {Re:.0f})",
        fontsize=13, fontweight="bold",
    )

    for ax, data, t in zip(axes, data_list, times):
        sym = field in ("vorticity", "u", "v")
        vmin_ = -vmax if sym else 0
        im = ax.pcolormesh(x, y, data.T, cmap=cmaps[field],
                           vmin=vmin_, vmax=vmax, shading="auto", rasterized=True)
        ax.set_title(f"t = {t:.2f} s", fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlabel("x", fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel("y", fontsize=8)
        else:
            ax.set_yticklabels([])
        decay = math.exp(-2.0 * nu * t / L**2)
        ax.text(0.02, 0.02, f"amp={decay:.3f}", transform=ax.transAxes,
                fontsize=7, color="white", va="bottom",
                bbox=dict(facecolor="black", alpha=0.5, edgecolor="none"))

    plt.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=DPI, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)
