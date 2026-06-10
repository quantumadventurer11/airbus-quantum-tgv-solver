"""
D2Q9 lattice constants and classical LBM helpers.

D2Q9 velocity set (2D, 9 directions):
  i=0: rest      (0, 0)   w=4/9
  i=1: East      (+1, 0)  w=1/9
  i=2: North     (0, +1)  w=1/9
  i=3: West      (-1, 0)  w=1/9
  i=4: South     (0, -1)  w=1/9
  i=5: NE        (+1,+1)  w=1/36
  i=6: NW        (-1,+1)  w=1/36
  i=7: SW        (-1,-1)  w=1/36
  i=8: SE        (+1,-1)  w=1/36

Chapman-Enskog expansion recovers 2D incompressible NS with ν = cs²(τ − 1/2)Δt,
cs² = 1/3 (lattice speed of sound squared).

Reference: Qian et al. 1992 (challenge statement ref [4]).
"""

import numpy as np

# ── Lattice constants ─────────────────────────────────────────────────────────

CX = np.array([0,  1,  0, -1,  0,  1, -1, -1,  1], dtype=int)
CY = np.array([0,  0,  1,  0, -1,  1,  1, -1, -1], dtype=int)
W  = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])

# Index of the opposite direction: OPP[i] = j such that (CX[j],CY[j]) = -(CX[i],CY[i])
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=int)

CS2 = 1.0 / 3.0    # lattice speed of sound squared
N_DIRS = 9
N_DIR_QUBITS = 4   # 4 qubits encode 16 directions (9 used, 7 padded)


# ── Classical LBM functions ───────────────────────────────────────────────────

def equilibrium(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    """
    BGK equilibrium distribution f_i^eq(ρ, u).

    f_i^eq = w_i · ρ · [1 + (e_i·u)/cs² + (e_i·u)²/(2cs⁴) − u²/(2cs²)]

    Parameters
    ----------
    rho : (N, N) — fluid density
    ux  : (N, N) — x-velocity
    uy  : (N, N) — y-velocity

    Returns
    -------
    feq : (9, N, N)
    """
    eu = CX[:, None, None] * ux[None] + CY[:, None, None] * uy[None]
    u2 = ux**2 + uy**2
    return W[:, None, None] * rho[None] * (1 + eu/CS2 + eu**2/(2*CS2**2) - u2[None]/(2*CS2))


def stream(f: np.ndarray) -> np.ndarray:
    """
    Classical D2Q9 streaming: f_i(x,y) → f_i(x+cx_i, y+cy_i) (periodic).

    Parameters
    ----------
    f : (9, N, N) — distribution functions

    Returns
    -------
    f_streamed : (9, N, N)
    """
    return np.stack([
        np.roll(f[i], (int(CX[i]), int(CY[i])), axis=(0, 1))
        for i in range(N_DIRS)
    ])


def moments(f: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute macroscopic moments: density ρ and velocity (ux, uy).

    ρ = Σ_i f_i,  ρ·u = Σ_i f_i·e_i

    Returns
    -------
    rho : (N, N)
    ux  : (N, N)
    uy  : (N, N)
    """
    rho = f.sum(axis=0)
    ux  = (CX[:, None, None] * f).sum(axis=0) / rho
    uy  = (CY[:, None, None] * f).sum(axis=0) / rho
    return rho, ux, uy


def bgk_collision(f: np.ndarray, omega: float) -> np.ndarray:
    """
    Single-relaxation-time (BGK) collision: f* = f + ω(f^eq − f).

    ω = 1/τ,  τ = 3ν + 0.5  (lattice units)

    Parameters
    ----------
    f     : (9, N, N)
    omega : float — relaxation rate ω ∈ (0, 2)

    Returns
    -------
    f_post : (9, N, N)
    """
    rho, ux, uy = moments(f)
    return f + omega * (equilibrium(rho, ux, uy) - f)
