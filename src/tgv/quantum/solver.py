"""
Full QLBM time-stepper: encode → stream → collide → decode → repeat.

Physical setup
--------------
The TGV is initialised in LBM lattice units on an N×N grid covering one
vortex period.  Key parameters:

  u_lbm   = peak LBM velocity (default 0.05 — small Mach number for accuracy)
  nu_lbm  = u_lbm * N / Re       (kinematic viscosity in lattice units)
  tau     = 3 * nu_lbm + 0.5     (relaxation time)
  omega   = 1 / tau              (relaxation rate)

One LBM time step corresponds to physical time:
  dt_phys = dt_lbm * (L_phys / N) * u_lbm     (L_phys = 2π, vortex scale)

Hybrid quantum-classical loop (default mode)
--------------------------------------------
Each time step performs:
  1. Encode   f → |ψ⟩  (quantum state, unit norm)
  2. Stream   |ψ⟩ → U_stream|ψ⟩   (exact unitary quantum gate)
  3. Decode   |ψ⟩ → f_stream       (classical readout)
  4. Collide  f_stream → f_coll = BGK(f_stream)  (classical BGK)

This "streaming-quantum / collision-classical" hybrid is the physically correct
NISQ-era QLBM: the streaming step IS unitary and CAN be executed on quantum
hardware; the BGK collision is approximated classically.  For fault-tolerant
quantum computing the collision would be replaced by a VQC gate, demonstrated
separately in `collision.py`.

VQC collision mode (optional, demonstration only)
-------------------------------------------------
When use_vqc=True, step 4 is replaced by a VQC gate applied to |ψ⟩.
The VQC approximates BGK with a single unitary rotation of the DIRECTION
register.  Because BGK needs spatially-varying rotations (feq depends on
local u) but the VQC applies the same rotation everywhere, this introduces
an approximation error proportional to the velocity variation across the grid.
This mode is suitable for demonstrating quantum ML training; the default hybrid
mode gives exact classical-LBM results and is used for the accuracy benchmark.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from qiskit import QuantumCircuit

from .d2q9 import equilibrium, moments, bgk_collision, stream as classical_stream
from .encoding import encode, decode
from .streaming import build_streaming_circuit
from .collision import (
    train_vqc,
    build_full_collision_circuit,
    build_analytic_collision_circuit,
)
from .backend import StatevectorBackend


# ── Utilities ──────────────────────────────────────────────────────────────────

def omega_lbm(Re: float, N: int, u_lbm: float = 0.05) -> float:
    """Compute LBM relaxation rate ω for given Reynolds number."""
    nu_lbm = u_lbm * N / Re
    tau    = 3.0 * nu_lbm + 0.5
    return 1.0 / tau


def lbm_ic(N: int, u_lbm: float = 0.05) -> np.ndarray:
    """
    TGV initial condition as LBM equilibrium distributions.

    u_x = u_lbm * sin(2π x/N) * cos(2π y/N)
    u_y = −u_lbm * cos(2π x/N) * sin(2π y/N)
    ρ   = 1  (uniform density)

    Returns f0 : (9, N, N) float array
    """
    xs  = np.arange(N)
    ys  = np.arange(N)
    X, Y = np.meshgrid(xs, ys, indexing="ij")

    phase_x = 2.0 * math.pi * X / N
    phase_y = 2.0 * math.pi * Y / N

    ux  = u_lbm * np.sin(phase_x) * np.cos(phase_y)
    uy  = -u_lbm * np.cos(phase_x) * np.sin(phase_y)
    rho = np.ones((N, N))

    return equilibrium(rho, ux, uy)


def dt_physical(N: int, u_lbm: float = 0.05, L_phys: float = 2.0 * math.pi) -> float:
    """Physical time elapsed per one LBM step."""
    return L_phys / N * u_lbm


def n_steps_for_time(t_phys: float, N: int, u_lbm: float = 0.05) -> int:
    """Number of LBM steps to reach physical time t_phys."""
    return max(1, round(t_phys / dt_physical(N, u_lbm)))


# ── QLBM Solver ───────────────────────────────────────────────────────────────

class QLBMSolver:
    """
    Full QLBM time-stepper using quantum streaming + BGK collision.

    Parameters
    ----------
    N        : grid side (power of 2; N=4 or N=8 for statevector simulation)
    omega    : BGK relaxation rate  ω = 1/τ
    n_layers : VQC depth (for VQC collision mode)
    u_lbm    : peak lattice velocity (for IC / time conversion)
    use_vqc  : if True, use VQC gate for collision (demonstration mode);
               if False (default), use classical BGK — gives exact results
    """

    def __init__(
        self,
        N: int,
        omega: float,
        n_layers: int = 4,
        u_lbm: float = 0.05,
        use_vqc: bool = False,
    ) -> None:
        self.N       = N
        self.omega   = omega
        self.n_layers = n_layers
        self.u_lbm   = u_lbm
        self.use_vqc = use_vqc

        self._backend        = StatevectorBackend()
        self._stream_circ    = build_streaming_circuit(N)
        self._collision_circ: Optional[QuantumCircuit] = None
        self._vqc_params: Optional[np.ndarray] = None

    # ── Collision circuit setup ───────────────────────────────────────────────

    def use_analytic_collision(self) -> None:
        """
        Set VQC collision to the Procrustes-optimal analytic circuit.
        No training required.  Enables use_vqc=True mode.
        """
        self.use_vqc = True
        self._collision_circ = build_analytic_collision_circuit(
            self.N, self.omega
        )

    def train(
        self,
        n_samples: int = 400,
        n_iter: int = 120,
        eps_max: float = 0.04,
        seed: int = 0,
    ) -> float:
        """
        Train the VQC collision circuit and switch to VQC mode.

        Returns validation infidelity (should be < 1e-3 for eps_max=0.04).
        """
        params, val_loss = train_vqc(
            self.omega,
            n_layers  = self.n_layers,
            n_samples = n_samples,
            n_iter    = n_iter,
            eps_max   = eps_max,
            seed      = seed,
        )
        self._vqc_params    = params
        self._collision_circ = build_full_collision_circuit(
            self.N, params, self.n_layers
        )
        self.use_vqc = True
        return val_loss

    # ── Time stepping ─────────────────────────────────────────────────────────

    def step(self, f: np.ndarray) -> np.ndarray:
        """
        One QLBM time step.

        In hybrid mode (use_vqc=False, default):
          Quantum streaming followed by classical BGK collision.
          Result is bit-for-bit identical to classical LBM.

        In VQC mode (use_vqc=True):
          Quantum streaming followed by quantum VQC collision gate.
          Introduces approximation error from the uniform-rotation constraint.

        Parameters
        ----------
        f : (9, N, N) current distributions

        Returns
        -------
        f_next : (9, N, N)
        """
        # 1. Encode
        sv, norm = encode(f)

        # 2. Quantum streaming (exact unitary)
        sv = self._backend.run(self._stream_circ, sv)

        # 3. Decode after streaming
        f_stream = decode(sv, self.N, norm)

        # 4. Collision
        if self.use_vqc and self._collision_circ is not None:
            # Quantum VQC collision (demonstration mode)
            # Track normalization classically (hybrid resource)
            rho, ux, uy = moments(f_stream)
            f_eq   = equilibrium(rho, ux, uy)
            f_coll = f_stream + self.omega * (f_eq - f_stream)
            norm_coll = float(np.linalg.norm(f_coll))
            sv = self._backend.run(self._collision_circ, sv)
            return decode(sv, self.N, norm_coll)
        else:
            # Classical BGK collision (hybrid default — gives exact LBM)
            return bgk_collision(f_stream, self.omega)

    def run(self, f0: np.ndarray, n_steps: int) -> np.ndarray:
        """
        Advance QLBM for n_steps time steps.

        Parameters
        ----------
        f0      : (9, N, N) initial distributions
        n_steps : number of QLBM time steps

        Returns f_final : (9, N, N)
        """
        f = f0.copy()
        for _ in range(n_steps):
            f = self.step(f)
        return f

    # ── Diagnostics ──────────────────────────────────────────────────────────

    @staticmethod
    def velocity(f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Extract velocity (ux, uy) in LBM lattice units from distributions."""
        _, ux, uy = moments(f)
        return ux, uy

    def velocity_physical(
        self, f: np.ndarray, V0: float = 1.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert LBM velocity to physical units (scale by V0/u_lbm)."""
        ux_lbm, uy_lbm = self.velocity(f)
        scale = V0 / self.u_lbm
        return ux_lbm * scale, uy_lbm * scale

    def kinetic_energy(self, f: np.ndarray) -> float:
        """Mean kinetic energy 0.5⟨|u|²⟩ in LBM units."""
        ux, uy = self.velocity(f)
        return 0.5 * float(np.mean(ux**2 + uy**2))
