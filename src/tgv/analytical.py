"""
Layer 1 — Analytical ground truth for the 2D Convecting Taylor-Green Vortex.

This module provides the EXACT closed-form solution for the TGV benchmark.
Every other solver layer is validated against these functions.  Do NOT modify
the formulas without re-reading §5 of the challenge statement (PDF pages 6-7).

Physical interpretation (plain language for engineers)
------------------------------------------------------
The Taylor-Green Vortex is a smooth, sinusoidal vortex that sits on top of a
uniform background flow (U_c, V_c).  In 2D, viscosity causes the vortex to
decay exponentially in amplitude while being advected — swept bodily — by the
background flow.  Because the vortex intensity falls off as exp(−2νt/L²) and
the flow remains periodic at all times, there is a perfect closed-form solution
for every t ≥ 0, which we use as the error ground truth.

Reference
---------
Challenge statement §5.2–5.3 (Airbus-Challenge-Statement-vF.pdf).
"""

import numpy as np


# ── Physical parameters (challenge statement §6) ──────────────────────────────
# These defaults match the problem specification exactly.
# All caller code should read from YAML config and pass them explicitly;
# these defaults are here only so the module is self-contained for tests.

_DEFAULTS = dict(
    L=2 * np.pi,   # Domain length [m]
    V0=1.0,        # Vortex velocity intensity [m/s]
    Uc=1.0,        # Convection velocity x-component [m/s]
    Vc=0.0,        # Convection velocity y-component [m/s]
    rho=1.0,       # Fluid density [kg/m³]
    p0=0.0,        # Background reference pressure [Pa]
)


def make_grid(N: int, L: float = 2 * np.pi) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (x, y) meshgrid arrays for an N×N periodic domain [0, 2πL).

    The domain extent is 2πL (not L) because sin(x/L) completes exactly one
    full period when x sweeps from 0 to 2πL:  sin((x+2πL)/L) = sin(x/L+2π)
    = sin(x/L).  With L = 2π this gives a domain [0, 4π²] ≈ [0, 39.48].
    L is the vortex length scale (= the reference length in the challenge
    parameter table, §6), not the domain size.
    """
    x1d = np.linspace(0.0, 2.0 * np.pi * L, N, endpoint=False)
    return np.meshgrid(x1d, x1d, indexing="ij")   # shape (N, N) each


def velocity_ic(
    x: np.ndarray,
    y: np.ndarray,
    *,
    L: float = _DEFAULTS["L"],
    V0: float = _DEFAULTS["V0"],
    Uc: float = _DEFAULTS["Uc"],
    Vc: float = _DEFAULTS["Vc"],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Exact initial velocity field at t = 0.

    u(x,y,0) = U_c + V0 · sin(x/L) · cos(y/L)
    v(x,y,0) = V_c − V0 · cos(x/L) · sin(y/L)

    The sin/cos structure ensures ∇·u = 0 (incompressibility).
    """
    u = Uc + V0 * np.sin(x / L) * np.cos(y / L)
    v = Vc - V0 * np.cos(x / L) * np.sin(y / L)
    return u, v


def pressure_ic(
    x: np.ndarray,
    y: np.ndarray,
    *,
    L: float = _DEFAULTS["L"],
    V0: float = _DEFAULTS["V0"],
    rho: float = _DEFAULTS["rho"],
    p0: float = _DEFAULTS["p0"],
) -> np.ndarray:
    """
    Exact initial pressure field at t = 0.

    p(x,y,0) = p0 + (ρ V0²/4) · [cos(2x/L) + cos(2y/L)]

    Derived from the Poisson equation ∇²p = −ρ ∇·(u·∇u) applied to the IC.
    """
    return p0 + (rho * V0**2 / 4.0) * (np.cos(2 * x / L) + np.cos(2 * y / L))


def velocity_exact(
    x: np.ndarray,
    y: np.ndarray,
    t: float,
    *,
    L: float = _DEFAULTS["L"],
    V0: float = _DEFAULTS["V0"],
    Uc: float = _DEFAULTS["Uc"],
    Vc: float = _DEFAULTS["Vc"],
    nu: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Exact velocity field at time t > 0.

    u(x,y,t) = U_c + V0 · sin((x − U_c·t)/L) · cos((y − V_c·t)/L) · e^(−2νt/L²)
    v(x,y,t) = V_c − V0 · cos((x − U_c·t)/L) · sin((y − V_c·t)/L) · e^(−2νt/L²)

    Two effects combine:
    1. Phase shift (x − U_c·t, y − V_c·t): the entire vortex moves at the
       background convection velocity (U_c, V_c).  This is the "convecting" part.
    2. Amplitude decay exp(−2νt/L²): viscosity irreversibly damps the vortex.
       Higher ν (lower Re) → faster decay.

    Parameters
    ----------
    nu : float
        Kinematic viscosity ν = V0·L/Re (challenge statement §6).
    """
    decay = np.exp(-2.0 * nu * t / L**2)
    xc = x - Uc * t   # convected x coordinate
    yc = y - Vc * t   # convected y coordinate
    u = Uc + V0 * np.sin(xc / L) * np.cos(yc / L) * decay
    v = Vc - V0 * np.cos(xc / L) * np.sin(yc / L) * decay
    return u, v


def kinetic_energy_exact(
    t: float,
    *,
    L: float = _DEFAULTS["L"],
    V0: float = _DEFAULTS["V0"],
    nu: float,
) -> float:
    """
    Domain-averaged kinetic energy at time t.

    E(t) = (1/|Ω|) ∫∫ (u² + v²)/2 dx dy  =  (V0² L²/4) · exp(−4νt/L²)

    Note the decay rate is −4ν/L² (double that of the velocity amplitude),
    because kinetic energy goes as the square of velocity.

    The convection velocities U_c, V_c contribute a constant (U_c² + V_c²)/2
    that cancels in the fluctuation energy and is excluded here for consistency
    with the LBM energy definition used in the benchmark.
    """
    return (V0**2 * L**2 / 4.0) * np.exp(-4.0 * nu * t / L**2)


def l2_error(
    u_sim: np.ndarray,
    v_sim: np.ndarray,
    u_exact: np.ndarray,
    v_exact: np.ndarray,
) -> float:
    """
    Root-mean-square L2 velocity error over the full domain.

    L2 = sqrt( mean( (u_sim − u_exact)² + (v_sim − v_exact)² ) )

    This is the primary accuracy metric used in the benchmark plots
    (challenge statement §4.1, error scaling vs Re).
    """
    err2 = (u_sim - u_exact) ** 2 + (v_sim - v_exact) ** 2
    return float(np.sqrt(np.mean(err2)))


def kinetic_energy_field(
    u: np.ndarray,
    v: np.ndarray,
    *,
    L: float = _DEFAULTS["L"],
) -> float:
    """Domain-averaged kinetic energy from a discrete velocity field."""
    N = u.shape[0]
    dx = L / N
    return float(np.mean(0.5 * (u**2 + v**2)))
