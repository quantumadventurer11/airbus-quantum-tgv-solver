"""
Tests for the quantum collision operator (Layer 3, collision.py).

Exit criterion (Week 4):
  Collision circuit achieves mean infidelity < 1e-3 on validation set
  when operating in the near-equilibrium LBM regime (eps_max=0.04).
"""

import numpy as np
import pytest

from tgv.quantum.collision import (
    build_vqc_ansatz,
    vqc_unitary,
    generate_training_data,
    mean_infidelity,
    procrustes_optimal_unitary,
    build_full_collision_circuit,
    build_analytic_collision_circuit,
)
from tgv.quantum.d2q9 import W, equilibrium, N_DIRS, N_DIR_QUBITS


# ── Shared fixture ────────────────────────────────────────────────────────────

OMEGA = 1.6129   # Re=10, N=8, u_lbm=0.05


# ── VQC ansatz structure ──────────────────────────────────────────────────────

@pytest.mark.parametrize("n_layers", [2, 4])
def test_vqc_parameter_count(n_layers):
    """VQC should have (n_layers + 1) * N_DIR_QUBITS parameters."""
    qc = build_vqc_ansatz(n_layers)
    expected = (n_layers + 1) * N_DIR_QUBITS
    assert len(qc.parameters) == expected, (
        f"n_layers={n_layers}: expected {expected} params, got {len(qc.parameters)}"
    )


def test_vqc_qubit_count():
    """VQC must operate on exactly 4 qubits (direction register)."""
    qc = build_vqc_ansatz(n_layers=4)
    assert qc.num_qubits == N_DIR_QUBITS == 4


@pytest.mark.parametrize("n_layers", [2, 4])
def test_vqc_unitary_is_unitary(n_layers):
    """VQC matrix U must satisfy U†U = I to machine precision."""
    rng = np.random.default_rng(42)
    params = rng.uniform(-np.pi, np.pi, (n_layers + 1) * N_DIR_QUBITS)
    U = vqc_unitary(params, n_layers)
    diff = np.max(np.abs(U @ U.conj().T - np.eye(16)))
    assert diff < 1e-10, f"n_layers={n_layers}: U†U ≠ I, max err={diff:.2e}"


# ── Training data ─────────────────────────────────────────────────────────────

def test_training_data_normalised():
    """Each training pair must be unit-norm 16-dim complex vectors."""
    data = generate_training_data(OMEGA, n_samples=20, eps_max=0.04, seed=0)
    for sv_in, sv_tgt in data:
        assert sv_in.shape == (16,)
        assert sv_tgt.shape == (16,)
        assert abs(np.linalg.norm(sv_in) - 1.0) < 1e-12, "sv_in not unit norm"
        assert abs(np.linalg.norm(sv_tgt) - 1.0) < 1e-12, "sv_tgt not unit norm"


def test_training_data_unused_directions_zero():
    """Directions 9–15 must have zero amplitude in all training samples."""
    data = generate_training_data(OMEGA, n_samples=20, eps_max=0.04, seed=0)
    for sv_in, sv_tgt in data:
        assert np.all(sv_in[N_DIRS:] == 0.0), "Unused dirs non-zero in sv_in"
        assert np.all(sv_tgt[N_DIRS:] == 0.0), "Unused dirs non-zero in sv_tgt"


def test_training_data_f_positive():
    """Extracted direction populations must be non-negative."""
    data = generate_training_data(OMEGA, n_samples=30, eps_max=0.04, seed=0)
    for sv_in, _ in data:
        assert np.all(sv_in[:N_DIRS] >= 0.0), "Negative distribution in sv_in"


# ── Equilibrium is a fixed point ──────────────────────────────────────────────

def test_equilibrium_fixed_point_infidelity():
    """
    BGK maps f^eq to itself: infidelity(U, [(sv_eq, sv_eq)]) must be 0 for
    any U that correctly handles the equilibrium state.

    For the Procrustes-optimal U*, the equilibrium direction is preserved
    exactly (it's the dominant singular vector with singular value 1).
    """
    # Build equilibrium state sv_eq
    feq = W.copy()              # equilibrium at rho=1, u=0
    sv_eq = np.zeros(16, dtype=complex)
    sv_eq[:N_DIRS] = feq / np.linalg.norm(feq)

    # Single-sample dataset: equilibrium maps to itself
    data_eq = [(sv_eq.copy(), sv_eq.copy())]

    # Identity achieves perfect fidelity (infidelity=0) on equilibrium
    loss_id = mean_infidelity(np.eye(16, dtype=complex), data_eq)
    assert loss_id < 1e-12, f"Identity should have zero loss at equilibrium, got {loss_id}"

    # Procrustes also preserves equilibrium
    U_proc = procrustes_optimal_unitary(OMEGA, n_samples=200, eps_max=0.04, seed=0)
    loss_proc = mean_infidelity(U_proc, data_eq)
    assert loss_proc < 5e-4, f"Procrustes should preserve equilibrium, got {loss_proc:.2e}"


# ── Procrustes optimal unitary (analytic fallback) ────────────────────────────

def test_procrustes_unitary_is_unitary():
    """Procrustes U* must satisfy U†U = I."""
    U = procrustes_optimal_unitary(OMEGA, n_samples=200, eps_max=0.04, seed=0)
    diff = np.max(np.abs(U @ U.conj().T - np.eye(16)))
    assert diff < 1e-10, f"Procrustes U† U ≠ I: max err={diff:.2e}"


def test_procrustes_preserves_unused_subspace():
    """U* must not leak amplitude into or from unused directions 9–15."""
    U = procrustes_optimal_unitary(OMEGA, n_samples=200, eps_max=0.04, seed=0)
    # Any sv_in with zeros at dirs 9-15 should produce sv_out with zeros there
    rng = np.random.default_rng(7)
    for _ in range(5):
        sv = np.zeros(16, dtype=complex)
        f = np.abs(rng.standard_normal(N_DIRS))
        sv[:N_DIRS] = f / np.linalg.norm(f)
        sv_out = U @ sv
        assert np.all(np.abs(sv_out[N_DIRS:]) < 1e-10), \
            "Procrustes leaks into unused directions"


def test_procrustes_validation_loss_below_threshold():
    """
    PRIMARY EXIT CRITERION: Procrustes-optimal collision circuit achieves
    mean infidelity < 1e-3 on the near-equilibrium validation set.

    Physical justification: eps_max=0.04 corresponds to Ma≈0.087 non-equilibrium
    perturbations typical of LBM in the near-equilibrium regime.
    """
    # Independent validation set (seed ≠ training seed)
    val_data = generate_training_data(OMEGA, n_samples=200, eps_max=0.04, seed=999)
    U_proc   = procrustes_optimal_unitary(OMEGA, n_samples=400, eps_max=0.04, seed=0)
    val_loss = mean_infidelity(U_proc, val_data)

    assert val_loss < 1e-3, (
        f"Procrustes collision val_loss = {val_loss:.2e} ≥ 1e-3. "
        f"Expected < 1e-3 for eps_max=0.04 near-equilibrium regime."
    )


# ── Full collision circuits ───────────────────────────────────────────────────

@pytest.mark.parametrize("N", [4, 8])
def test_vqc_collision_circuit_qubit_count(N):
    """Full VQC collision circuit must have 4 + 2·log₂N qubits."""
    import math
    rng = np.random.default_rng(0)
    params = rng.uniform(-np.pi, np.pi, 20)
    qc = build_full_collision_circuit(N, params, n_layers=4)
    expected = 4 + 2 * int(math.log2(N))
    assert qc.num_qubits == expected, (
        f"N={N}: expected {expected} qubits, got {qc.num_qubits}"
    )


@pytest.mark.parametrize("N", [4, 8])
def test_analytic_collision_circuit_qubit_count(N):
    """Analytic collision circuit must have 4 + 2·log₂N qubits."""
    import math
    qc = build_analytic_collision_circuit(N, OMEGA)
    expected = 4 + 2 * int(math.log2(N))
    assert qc.num_qubits == expected, (
        f"N={N}: expected {expected} qubits, got {qc.num_qubits}"
    )


def test_analytic_collision_circuit_is_unitary():
    """
    Analytic collision circuit for N=4 must be unitary (U†U = I).
    Uses Qiskit Operator which builds the full 256×256 matrix — tractable at N=4.
    """
    from qiskit.quantum_info import Operator
    qc = build_analytic_collision_circuit(N=4, omega=OMEGA)
    op = Operator(qc)
    U = op.data
    diff = np.max(np.abs(U @ U.conj().T - np.eye(U.shape[0])))
    assert diff < 1e-8, f"Analytic collision circuit not unitary: max err={diff:.2e}"
