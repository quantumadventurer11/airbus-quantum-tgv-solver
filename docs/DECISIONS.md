# Design Decisions Log

Running log of non-obvious choices.  Every entry has a rationale and the
date it was made.  Update this whenever a significant choice is changed.

---

## D-001 — Domain convention: [0, 2πL] not [0, L]  (2026-06-08)

**Decision:** `make_grid` generates coordinates on `[0, 2πL)`, not `[0, L)`.

**Reason:** The velocity formula `sin(x/L)` is periodic only when x sweeps a full
period of length `2πL` (since `sin((x+2πL)/L) = sin(x/L+2π) = sin(x/L)`).
With `L = 2π` this gives domain `[0, 4π²]`.  Using `[0, L] = [0, 2π]` instead
creates a non-periodic velocity field that violates the periodic-boundary
conditions required by the spectral solver.

**Implication:** All grid coordinates and spatial plots have physical extent
`2πL ≈ 39.5` (dimensionless).  The parameter L = 2π is the vortex length scale,
not the domain size.  The domain size is `2πL`.  The energy formula
`E = V0²L²/4 · exp(-4νt/L²)` is consistent: dividing by L² gives the
domain-averaged fluctuation KE = V0²/4 · exp(-4νt/L²).

---

## D-002 — Primary quantum method: QLBM + VQC collision  (2026-06-08)

**Decision:** Use Quantum Lattice Boltzmann Method (D2Q9) with a variational
quantum circuit (VQC) trained to approximate the BGK collision operator.

**Reason:**
- Directly endorsed by challenge bibliography refs [14] (Lacatus & Müller 2026)
  and [15] (Zamora et al. 2026).
- The streaming step is naturally unitary (permutation circuits).
- The VQC collision is trainable on NISQ hardware with SPSA.
- Memory advantage is demonstrable on Aer statevector simulation:
  2⌈log₂N⌉ + 4 qubits encodes N²×9 distribution functions.
- Qiskit-native implementation, no external LBM library needed.

**Fallback:** Carleman linearisation + QLSA (HHL) for FTQC resource estimates
at Re=10 only.

---

## D-003 — Classical baseline: pseudo-spectral FFT  (2026-06-08)

**Decision:** Use pseudo-spectral FFT + RK4 time-stepping as the classical
reference solver (Layer 2), not finite-volume or finite-difference.

**Reason:** The TGV IC is a smooth sinusoidal function that is exactly
representable in Fourier space.  A spectral method achieves exponential
convergence in space (spectral accuracy), meaning we need far fewer grid points
to reach high accuracy.  This gives the cleanest classical baseline to compare
against the quantum solver.  Finite-volume would require far larger grids for
the same accuracy, complicating the comparison.

---

## D-004 — Quantum grid capped at N=16 for statevector simulation  (2026-06-08)

**Decision:** The QLBM quantum solver will be validated on grids N ≤ 16
(12 qubits) when using Aer statevector simulation.

**Reason:** Statevector simulation requires 2^n complex amplitudes.  At N=16
we have 12 qubits → 2^12 = 4096 amplitudes → 65 kB memory per circuit
evaluation.  At N=32 (14 qubits) it's 16 MB, which is tractable.  At N=64
(16 qubits) it's 1 GB.  We cap at N=16 for routine tests and extend to N=32
only for the scaling study where we accept the longer runtime.

---
