"""
Layer 1 validation tests — Analytical ground truth.

Exit criterion: all tests pass with tolerance ≤ 1e-12.
These tests confirm the exact-solution formulas are correctly implemented
before any solver is built on top of them.
"""

import math
import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tgv.analytical import (
    make_grid,
    velocity_ic,
    pressure_ic,
    velocity_exact,
    kinetic_energy_exact,
    kinetic_energy_field,
    l2_error,
)

# ── Shared parameters (challenge statement §6) ────────────────────────────────
L = 2 * math.pi
V0 = 1.0
Uc = 1.0
Vc = 0.0
rho = 1.0
p0 = 0.0
Re = 10.0
nu = V0 * L / Re   # ≈ 0.6283

RNG = np.random.default_rng(42)
N_GRID = 64


@pytest.fixture(scope="module")
def grid():
    return make_grid(N_GRID, L)


# ── Test 1: initial conditions match formula at random points ─────────────────

def test_velocity_ic_random_points():
    xs = RNG.uniform(0, L, 10)
    ys = RNG.uniform(0, L, 10)
    u, v = velocity_ic(xs, ys, L=L, V0=V0, Uc=Uc, Vc=Vc)
    u_ref = Uc + V0 * np.sin(xs / L) * np.cos(ys / L)
    v_ref = Vc - V0 * np.cos(xs / L) * np.sin(ys / L)
    np.testing.assert_allclose(u, u_ref, atol=1e-14)
    np.testing.assert_allclose(v, v_ref, atol=1e-14)


def test_pressure_ic_random_points():
    xs = RNG.uniform(0, L, 10)
    ys = RNG.uniform(0, L, 10)
    p = pressure_ic(xs, ys, L=L, V0=V0, rho=rho, p0=p0)
    p_ref = p0 + (rho * V0**2 / 4.0) * (np.cos(2 * xs / L) + np.cos(2 * ys / L))
    np.testing.assert_allclose(p, p_ref, atol=1e-14)


# ── Test 2: exact solution at t=0 matches initial conditions ──────────────────

def test_exact_solution_at_t0(grid):
    x, y = grid
    u_ic, v_ic = velocity_ic(x, y, L=L, V0=V0, Uc=Uc, Vc=Vc)
    u_ex, v_ex = velocity_exact(x, y, t=0.0, L=L, V0=V0, Uc=Uc, Vc=Vc, nu=nu)
    np.testing.assert_allclose(u_ex, u_ic, atol=1e-14)
    np.testing.assert_allclose(v_ex, v_ic, atol=1e-14)


# ── Test 3: exact solution at t>0 matches formula at random points ────────────

@pytest.mark.parametrize("t", [0.1, 0.5, 1.0])
def test_exact_solution_at_t_positive(t):
    xs = RNG.uniform(0, L, 10)
    ys = RNG.uniform(0, L, 10)
    u, v = velocity_exact(xs, ys, t=t, L=L, V0=V0, Uc=Uc, Vc=Vc, nu=nu)
    decay = math.exp(-2.0 * nu * t / L**2)
    u_ref = Uc + V0 * np.sin((xs - Uc * t) / L) * np.cos((ys - Vc * t) / L) * decay
    v_ref = Vc - V0 * np.cos((xs - Uc * t) / L) * np.sin((ys - Vc * t) / L) * decay
    np.testing.assert_allclose(u, u_ref, atol=1e-14)
    np.testing.assert_allclose(v, v_ref, atol=1e-14)


# ── Test 4: kinetic energy follows exact exponential decay ────────────────────

@pytest.mark.parametrize("t", [0.0, 0.1, 0.5, 1.0])
def test_energy_decay_formula(t):
    E = kinetic_energy_exact(t, L=L, V0=V0, nu=nu)
    E_ref = (V0**2 * L**2 / 4.0) * math.exp(-4.0 * nu * t / L**2)
    assert abs(E - E_ref) < 1e-14


def test_energy_is_positive():
    for t in [0.0, 0.5, 2.0]:
        assert kinetic_energy_exact(t, L=L, V0=V0, nu=nu) > 0.0


def test_energy_monotone_decreasing():
    ts = np.linspace(0.0, 2.0, 20)
    Es = [kinetic_energy_exact(t, L=L, V0=V0, nu=nu) for t in ts]
    assert all(Es[i] >= Es[i + 1] for i in range(len(Es) - 1))


# ── Test 5: divergence-free condition ∂u/∂x + ∂v/∂y = 0 ─────────────────────

@pytest.mark.parametrize("t", [0.0, 0.5])
def test_divergence_free(t):
    """
    Verifies ∇·u = 0 on the grid using 2nd-order centred finite differences.
    The exact solution is analytically divergence-free; numerical FD confirms
    implementation is consistent.
    """
    N = 128
    x, y = make_grid(N, L)
    dx = L / N
    u, v = velocity_exact(x, y, t=t, L=L, V0=V0, Uc=Uc, Vc=Vc, nu=nu)
    # Central differences, periodic BCs
    dudx = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2 * dx)
    dvdy = (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)) / (2 * dx)
    div = dudx + dvdy
    assert np.max(np.abs(div)) < 1e-10


# ── Test 6: L2 error is zero when compared against itself ────────────────────

def test_l2_error_self_is_zero(grid):
    x, y = grid
    u, v = velocity_exact(x, y, t=0.5, L=L, V0=V0, Uc=Uc, Vc=Vc, nu=nu)
    err = l2_error(u, v, u, v)
    assert err == 0.0


def test_l2_error_nonzero_on_perturbed(grid):
    x, y = grid
    u, v = velocity_exact(x, y, t=0.5, L=L, V0=V0, Uc=Uc, Vc=Vc, nu=nu)
    err = l2_error(u + 0.01, v, u, v)
    assert err > 0.005


# ── Test 7: kinetic_energy_field is consistent with kinetic_energy_exact ──────

def test_energy_field_vs_exact(grid):
    """
    The grid-averaged fluctuation KE should match kinetic_energy_exact / L².

    kinetic_energy_exact returns V0²L²/4 · exp(-4νt/L²).
    The domain is [0, 2πL]², so one full period is covered and the true
    mean fluctuation KE = V0²/4 · exp(-4νt/L²) = kinetic_energy_exact / L².
    At N=64 the discrete quadrature error is < 0.1%.
    """
    x, y = grid
    t = 0.5
    u, v = velocity_exact(x, y, t=t, L=L, V0=V0, Uc=Uc, Vc=Vc, nu=nu)
    u_fluct = u - Uc
    v_fluct = v - Vc
    E_field = kinetic_energy_field(u_fluct, v_fluct, L=L)   # mean(0.5*(u²+v²))
    E_exact = kinetic_energy_exact(t, L=L, V0=V0, nu=nu)    # V0²L²/4 · exp(...)
    # mean field KE ≈ E_exact / L²  (both equal V0²/4 · exp(-4νt/L²))
    assert abs(E_field - E_exact / L**2) / (E_exact / L**2) < 0.005


# ── Test 8: make_grid properties ─────────────────────────────────────────────

def test_make_grid_shape():
    x, y = make_grid(32, L)
    assert x.shape == (32, 32)
    assert y.shape == (32, 32)


def test_make_grid_range():
    """Domain is [0, 2πL), so max coordinate < 2πL."""
    x, y = make_grid(16, L)
    domain = 2 * math.pi * L
    assert float(x.min()) >= 0.0
    assert float(x.max()) < domain
    assert float(y.min()) >= 0.0
    assert float(y.max()) < domain


def test_make_grid_periodic_spacing():
    """Grid spacing is 2πL / N."""
    N = 8
    x, y = make_grid(N, L)
    dx = 2 * math.pi * L / N
    np.testing.assert_allclose(np.diff(x[:, 0]), dx, atol=1e-14)
