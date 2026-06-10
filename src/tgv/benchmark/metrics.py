"""
Benchmark metrics collection for the hybrid QLBM TGV solver.

Each call to run_single() runs the hybrid QLBM solver for one (Re, N, t_end)
configuration and returns a dict of performance and accuracy metrics.

Domain convention note
----------------------
analytical.py uses L=2π as a *length scale*, giving a physical domain
[0, 2πL] = [0, 4π²].  The LBM operates on [0, 2π] (one lattice period).
These are different parameterisations of the same vortex flow.  For the
LBM comparison we use the lattice-native exact solution (same formula as
test_qlbm_full.py's exact_velocity), which has:

  wavenumber κ = 1 rad/m (sin(2πk/N) over domain [0, 2π))
  kinematic viscosity  ν = V0 · 2π / Re
  decay = exp(−2 ν t)   (for both velocity components)

The convection velocity Uc=1 is NOT included because lbm_ic initialises
the TGV perturbation only (no mean flow).
"""
from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np

from tgv.quantum.encoding import n_qubits
from tgv.quantum.solver import (
    QLBMSolver,
    lbm_ic,
    omega_lbm,
    n_steps_for_time,
    dt_physical,
)

# Physical domain constant (LBM domain size = one vortex period)
L_PHYS = 2.0 * math.pi
V0 = 1.0


def _ke_exact_lbm(t_phys: float, Re: float) -> float:
    """
    Analytical kinetic energy E(t) = V0²/4 · exp(−4νt) for the LBM domain.

    The LBM TGV has wavenumber κ=1 rad/m on domain [0,2π)², so
    E(t) = (V0²/4) exp(−4νt) where ν = V0·2π/Re.
    """
    nu = V0 * L_PHYS / Re
    return (V0**2 / 4.0) * math.exp(-4.0 * nu * t_phys)


def _exact_velocity_lbm(N: int, t_phys: float, Re: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Exact TGV velocity for comparison with the LBM solver.

    Uses lattice-coordinate indexing (k=0..N-1) and the viscous decay rate
    that matches the LBM wavenumber κ=1. This is the same formula used in
    test_qlbm_full.py::exact_velocity — confirmed against the passing test.
    """
    K = np.arange(N)
    K2d, L2d = np.meshgrid(K, K, indexing="ij")
    nu = V0 * L_PHYS / Re          # kinematic viscosity [m²/s]
    decay = np.exp(-2.0 * nu * t_phys)
    ux = V0 * np.sin(2*math.pi * K2d / N) * np.cos(2*math.pi * L2d / N) * decay
    uy = -V0 * np.cos(2*math.pi * K2d / N) * np.sin(2*math.pi * L2d / N) * decay
    return ux, uy


def _qlbm_grid_for_re(Re: float, ci_mode: bool) -> int:
    """Select QLBM grid size (power-of-2, statevector tractable).

    N=4 (8 qubits) is too coarse for meaningful L2 error (spatial aliasing).
    N=8 (10 qubits) gives <5% L2 error at Re=10 per test_qlbm_l2_vs_exact_below_5pct.
    """
    if Re <= 10:
        return 8
    if Re <= 100:
        return 16  # Re=100: finer grid (12 qubits) for better accuracy
    return 16  # Re=500: finer grid (12 qubits) for better accuracy


def run_single(
    Re: float,
    t_end: float = 0.5,
    N_qlbm: Optional[int] = None,
    u_lbm: float = 0.05,
    timeout_s: float = 180.0,
    ci_mode: bool = False,
) -> dict:
    """
    Run the hybrid QLBM solver for one (Re, t_end) configuration.

    Returns a dict with keys:
      Re, N_qlbm, n_qubits, n_steps, wall_time_s, timed_out,
      l2_error, kinetic_energy_qlbm, kinetic_energy_exact,
      memory_mb, solver_mode
    """
    if N_qlbm is None:
        N_qlbm = _qlbm_grid_for_re(Re, ci_mode)

    nq = n_qubits(N_qlbm)
    memory_mb = (2 ** nq) * 16 / 1e6  # complex128 statevector

    omega = omega_lbm(Re, N_qlbm, u_lbm)
    n_steps = n_steps_for_time(t_end, N_qlbm, u_lbm)
    dt_phys = dt_physical(N_qlbm, u_lbm, L_PHYS)
    t_actual = n_steps * dt_phys

    solver = QLBMSolver(N=N_qlbm, omega=omega, u_lbm=u_lbm, use_vqc=False)
    f0 = lbm_ic(N_qlbm, u_lbm)

    timed_out = False
    t_start = time.perf_counter()

    try:
        f = f0.copy()
        for step in range(n_steps):
            if time.perf_counter() - t_start > timeout_s:
                timed_out = True
                # Use whatever we have so far
                t_actual = step * dt_phys
                break
            f = solver.step(f)
    except Exception as exc:
        timed_out = True
        wall_time = time.perf_counter() - t_start
        return {
            "Re": Re,
            "N_qlbm": N_qlbm,
            "n_qubits": nq,
            "n_steps": n_steps,
            "wall_time_s": wall_time,
            "timed_out": True,
            "error_msg": str(exc),
            "l2_error": float("nan"),
            "kinetic_energy_qlbm": float("nan"),
            "kinetic_energy_exact": _ke_exact_lbm(t_end, Re),
            "memory_mb": memory_mb,
            "solver_mode": "hybrid",
        }

    wall_time = time.perf_counter() - t_start

    # Convert to physical velocities and compute L2 error
    ux_phys, uy_phys = solver.velocity_physical(f, V0=V0)
    u_ex, v_ex = _exact_velocity_lbm(N_qlbm, t_actual, Re)
    diff2 = (ux_phys - u_ex) ** 2 + (uy_phys - v_ex) ** 2
    err = float(np.sqrt(np.mean(diff2)))

    ke_qlbm = solver.kinetic_energy(f) * (V0 / u_lbm) ** 2
    ke_exact = _ke_exact_lbm(t_actual, Re)

    return {
        "Re": Re,
        "N_qlbm": N_qlbm,
        "n_qubits": nq,
        "n_steps": n_steps,
        "wall_time_s": wall_time,
        "timed_out": timed_out,
        "l2_error": err,
        "kinetic_energy_qlbm": ke_qlbm,
        "kinetic_energy_exact": ke_exact,
        "memory_mb": memory_mb,
        "solver_mode": "hybrid",
    }


def run_sweep(
    re_list: list[float],
    t_end: float = 0.5,
    ci_mode: bool = False,
    timeout_s: float = 180.0,
) -> list[dict]:
    """Run benchmark for each Re in re_list and return list of result dicts."""
    results = []
    for Re in re_list:
        print(f"  Re={Re:.0f} ...", end=" ", flush=True)
        row = run_single(Re, t_end=t_end, ci_mode=ci_mode, timeout_s=timeout_s)
        if row.get("timed_out"):
            print(f"TIMEOUT after {row['wall_time_s']:.1f}s  (L2={row['l2_error']})")
        else:
            print(f"done in {row['wall_time_s']:.2f}s  L2={row['l2_error']:.4e}")
        results.append(row)
    return results
