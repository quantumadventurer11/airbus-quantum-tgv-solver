"""
Generate the three required challenge benchmark plots from a list of result dicts.
"""
from __future__ import annotations

import math
import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_V0 = 1.0
_L_PHYS = 2.0 * math.pi


def _ke_exact(t: float, Re: float) -> float:
    """E(t) = V0²/4 · exp(−4νt), ν = V0·2π/Re (LBM wavenumber convention)."""
    nu = _V0 * _L_PHYS / Re
    return (_V0**2 / 4.0) * math.exp(-4.0 * nu * t)

L_PHYS = 2.0 * math.pi
V0 = 1.0
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "results", "figures")


def _figures_dir(out_dir: Optional[str] = None) -> str:
    if out_dir is not None:
        return out_dir
    return os.path.normpath(FIGURES_DIR)


def _filter_valid(results: list[dict], key: str) -> list[dict]:
    """Return rows where key is finite (not NaN, not timed-out)."""
    return [r for r in results if not r.get("timed_out", False) and math.isfinite(r.get(key, float("nan")))]


def plot_time_to_solution(results: list[dict], out_dir: Optional[str] = None) -> str:
    fdir = _figures_dir(out_dir)
    os.makedirs(fdir, exist_ok=True)

    valid = _filter_valid(results, "wall_time_s")
    timed = [r for r in results if r.get("timed_out", False)]

    re_v = [r["Re"] for r in valid]
    t_v  = [r["wall_time_s"] for r in valid]

    fig, ax = plt.subplots(figsize=(6, 4))
    if re_v:
        ax.loglog(re_v, t_v, "o-", color="#2171b5", linewidth=2, markersize=7, label="Hybrid QLBM (statevector)")
        for r in timed:
            ax.axvline(r["Re"], color="red", linestyle="--", alpha=0.6, label=f"Re={r['Re']:.0f} timed out")
    ax.set_xlabel("Reynolds number Re")
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title("Time-to-Solution vs Re\nHybrid QLBM — 2D Convecting TGV")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    path = os.path.join(fdir, "time_to_solution_vs_re.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_memory_and_qubits(results: list[dict], out_dir: Optional[str] = None) -> str:
    fdir = _figures_dir(out_dir)
    os.makedirs(fdir, exist_ok=True)

    valid = _filter_valid(results, "n_qubits")
    re_v  = [r["Re"] for r in valid]
    nq_v  = [r["n_qubits"] for r in valid]
    mem_v = [r["memory_mb"] for r in valid]

    fig, ax1 = plt.subplots(figsize=(6, 4))
    color1 = "#2171b5"
    color2 = "#d94801"

    if re_v:
        ax1.semilogx(re_v, nq_v, "s-", color=color1, linewidth=2, markersize=7, label="Qubit count")
        ax1.set_ylabel("Qubit count", color=color1)
        ax1.tick_params(axis="y", labelcolor=color1)

        ax2 = ax1.twinx()
        ax2.semilogx(re_v, mem_v, "^--", color=color2, linewidth=2, markersize=7, label="Statevector memory (MB)")
        ax2.set_ylabel("Statevector memory (MB)", color=color2)
        ax2.tick_params(axis="y", labelcolor=color2)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

    ax1.set_xlabel("Reynolds number Re")
    ax1.set_title("Qubit Count & Statevector Memory vs Re\nHybrid QLBM — 2D Convecting TGV")
    ax1.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    path = os.path.join(fdir, "memory_or_qubits_vs_re.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_l2_error(results: list[dict], out_dir: Optional[str] = None) -> str:
    fdir = _figures_dir(out_dir)
    os.makedirs(fdir, exist_ok=True)

    valid = _filter_valid(results, "l2_error")
    re_v  = [r["Re"] for r in valid]
    err_v = [r["l2_error"] for r in valid]
    timed = [r for r in results if r.get("timed_out", False)]

    fig, ax = plt.subplots(figsize=(6, 4))
    if re_v:
        ax.loglog(re_v, err_v, "o-", color="#6a3d9a", linewidth=2, markersize=7, label="Hybrid QLBM L2 error")
        for r in timed:
            ax.axvline(r["Re"], color="red", linestyle="--", alpha=0.6, label=f"Re={r['Re']:.0f} timed out")
    ax.set_xlabel("Reynolds number Re")
    ax.set_ylabel("L2 velocity error")
    ax.set_title("L2 Velocity Error vs Re\nHybrid QLBM vs Exact TGV Solution")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    path = os.path.join(fdir, "l2_error_vs_re.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_kinetic_energy_decay(
    re_list: Optional[list[float]] = None,
    t_end: float = 1.0,
    n_points: int = 100,
    out_dir: Optional[str] = None,
) -> str:
    """Plot analytical kinetic energy decay curves for a set of Reynolds numbers."""
    if re_list is None:
        re_list = [10.0, 100.0, 500.0]

    fdir = _figures_dir(out_dir)
    os.makedirs(fdir, exist_ok=True)

    ts = np.linspace(0, t_end, n_points)
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(re_list)))

    fig, ax = plt.subplots(figsize=(6, 4))
    for Re, col in zip(re_list, colors):
        ke = [_ke_exact(t, Re) for t in ts]
        ax.semilogy(ts, ke, linewidth=2, color=col, label=f"Re={Re:.0f}")

    ax.set_xlabel("Physical time t (s)")
    ax.set_ylabel("Domain-averaged kinetic energy")
    ax.set_title("Analytical Kinetic Energy Decay\n2D Convecting TGV")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    path = os.path.join(fdir, "kinetic_energy_decay.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def generate_plots(
    results: list[dict],
    re_list: Optional[list[float]] = None,
    out_dir: Optional[str] = None,
) -> list[str]:
    """Generate all benchmark plots. Returns list of saved file paths."""
    paths = [
        plot_time_to_solution(results, out_dir),
        plot_memory_and_qubits(results, out_dir),
        plot_l2_error(results, out_dir),
        plot_kinetic_energy_decay(re_list=re_list, out_dir=out_dir),
    ]
    return paths
