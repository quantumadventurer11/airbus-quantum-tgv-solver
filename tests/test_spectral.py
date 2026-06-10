"""
Layer 2 tests — pseudo-spectral FFT solver for the 2D Convecting TGV.

Exit criteria (PROJECT_PLAN.md Week 2):
  ✓ L2(u) < 1e-4  at Re=10,  t=0.5,  N=64
  ✓ L2(u) < 5e-3  at Re=100, t=0.5,  N=128
  ✓ Convergence order ≥ 2 (N=32→64→128)
  ✓ Energy decay within 0.1% of exact formula at Re=10
  ✓ IC matches Layer 1 within 1e-12 at t=0
"""

import math
import numpy as np
import pytest

from tgv.analytical import (
    make_grid,
    velocity_ic,
    velocity_exact,
    kinetic_energy_exact,
    l2_error,
)
from tgv.classical.spectral_solver import SpectralSolver, solve_tgv

# ── Shared parameters (§6) ────────────────────────────────────────────────────
L  = 2 * math.pi
V0 = 1.0
Uc = 1.0
Vc = 0.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run(N, Re, t_end):
    """Run solver and return (u_sim, v_sim, u_exact, v_exact)."""
    nu = V0 * L / Re
    x, y = make_grid(N, L)
    u0, v0 = velocity_ic(x, y, L=L, V0=V0, Uc=Uc, Vc=Vc)
    solver = SpectralSolver(N, L, nu, Uc=Uc, Vc=Vc)
    _, u, v = solver.run(u0, v0, t_end)
    u_ex, v_ex = velocity_exact(x, y, t_end, L=L, V0=V0, Uc=Uc, Vc=Vc, nu=nu)
    return u, v, u_ex, v_ex


# ── Exit criterion tests ──────────────────────────────────────────────────────

def test_l2_error_re10_n64():
    """PRIMARY EXIT CRITERION: L2(u) < 1e-4 at Re=10, t=0.5, N=64."""
    u, v, u_ex, v_ex = _run(N=64, Re=10.0, t_end=0.5)
    err = l2_error(u, v, u_ex, v_ex)
    assert err < 1e-4, f"L2 error {err:.3e} exceeds 1e-4"


def test_l2_error_re100_n128():
    """SECONDARY EXIT CRITERION: L2(u) < 5e-3 at Re=100, t=0.5, N=128."""
    u, v, u_ex, v_ex = _run(N=128, Re=100.0, t_end=0.5)
    err = l2_error(u, v, u_ex, v_ex)
    assert err < 5e-3, f"L2 error {err:.3e} exceeds 5e-3"


def test_ic_match():
    """At t=0, solver output should reproduce the IC within floating-point precision."""
    N = 64
    Re = 10.0
    nu = V0 * L / Re
    x, y = make_grid(N, L)
    u0, v0 = velocity_ic(x, y, L=L, V0=V0, Uc=Uc, Vc=Vc)

    solver = SpectralSolver(N, L, nu, Uc=Uc, Vc=Vc)
    _, u, v = solver.run(u0, v0, t_end=0.0)
    err = l2_error(u, v, u0, v0)
    assert err < 1e-12, f"IC roundtrip error {err:.3e} exceeds 1e-12"


# ── Convergence test ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("Re,t_end", [(10.0, 0.5), (100.0, 0.2)])
def test_convergence_order(Re, t_end):
    """
    Grid-refinement convergence: error should halve at least 4× per doubling of N
    (spectral convergence is super-algebraic; 2nd order is a very conservative floor).
    """
    grids = [32, 64, 128]
    errors = []
    for N in grids:
        u, v, u_ex, v_ex = _run(N=N, Re=Re, t_end=t_end)
        errors.append(l2_error(u, v, u_ex, v_ex))

    for i in range(len(errors) - 1):
        ratio = errors[i] / errors[i + 1]
        assert ratio > 4.0, (
            f"Re={Re}: N={grids[i]}→{grids[i+1]} gave ratio {ratio:.2f} "
            f"(errors: {errors[i]:.3e} → {errors[i+1]:.3e}). "
            "Expected >4× (≥2nd order convergence)."
        )


# ── Energy decay test ─────────────────────────────────────────────────────────

def test_energy_decay_re10():
    """
    Kinetic energy tracks exact exponential decay within 0.1% at Re=10.
    Measured at t = 0.1, 0.2, 0.5.
    """
    N, Re = 64, 10.0
    nu = V0 * L / Re
    x, y = make_grid(N, L)
    u0, v0 = velocity_ic(x, y, L=L, V0=V0, Uc=Uc, Vc=Vc)
    solver = SpectralSolver(N, L, nu, Uc=Uc, Vc=Vc)

    for t_chk in [0.1, 0.2, 0.5]:
        _, u, v = solver.run(u0.copy(), v0.copy(), t_chk)
        # Fluctuation KE (subtract background)
        u_fluct = u - Uc
        v_fluct = v - Vc
        E_sim = float(np.mean(0.5 * (u_fluct**2 + v_fluct**2)))
        E_exact = kinetic_energy_exact(t_chk, L=L, V0=V0, nu=nu) / L**2
        rel_err = abs(E_sim - E_exact) / E_exact
        assert rel_err < 1e-3, (
            f"t={t_chk}: energy relative error {rel_err:.3e} > 0.1% "
            f"(E_sim={E_sim:.6f}, E_exact={E_exact:.6f})"
        )


# ── Divergence-free check ─────────────────────────────────────────────────────

def test_divergence_free():
    """
    Simulated velocity field should remain divergence-free (incompressible).
    Checked with centred finite differences; tolerance 1e-10.
    """
    N, Re = 64, 10.0
    nu = V0 * L / Re
    x, y = make_grid(N, L)
    u0, v0 = velocity_ic(x, y, L=L, V0=V0, Uc=Uc, Vc=Vc)
    solver = SpectralSolver(N, L, nu, Uc=Uc, Vc=Vc)
    _, u, v = solver.run(u0, v0, 0.3)

    dx = 2.0 * math.pi * L / N
    dudx = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2 * dx)
    dvdy = (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)) / (2 * dx)
    div = dudx + dvdy
    max_div = float(np.abs(div).max())
    assert max_div < 1e-10, f"Max divergence {max_div:.3e} exceeds 1e-10"


# ── Convenience wrapper test ──────────────────────────────────────────────────

def test_solve_tgv_wrapper():
    """solve_tgv() convenience function returns correct shapes and reasonable error."""
    x, y, u, v = solve_tgv(N=64, Re=10.0, t_end=0.5)
    assert u.shape == (64, 64)
    assert v.shape == (64, 64)

    nu = V0 * L / 10.0
    u_ex, v_ex = velocity_exact(x, y, 0.5, L=L, V0=V0, Uc=Uc, Vc=Vc, nu=nu)
    err = l2_error(u, v, u_ex, v_ex)
    assert err < 1e-4


# ── Stability — Re=400 ────────────────────────────────────────────────────────

def test_stability_re400():
    """
    Solver should remain stable (no NaN/Inf) at Re=400 for 10 time steps.
    Accuracy is not asserted here — just numerical stability.
    """
    N, Re = 64, 400.0
    nu = V0 * L / Re
    x, y = make_grid(N, L)
    u0, v0 = velocity_ic(x, y, L=L, V0=V0, Uc=Uc, Vc=Vc)
    solver = SpectralSolver(N, L, nu, Uc=Uc, Vc=Vc)
    dt = solver.dt
    _, u, v = solver.run(u0, v0, 10 * dt)
    assert np.all(np.isfinite(u)), "NaN/Inf in u at Re=400"
    assert np.all(np.isfinite(v)), "NaN/Inf in v at Re=400"
