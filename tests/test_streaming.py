"""
Tests for quantum D2Q9 streaming circuit (Layer 3, streaming.py).

EXIT CRITERION (PROJECT_PLAN.md Week 3):
  QLBM streaming circuit applied to encoded 4×4 and 8×8 grids matches
  classical LBM streaming exactly (amplitude-by-amplitude, tol 1e-10).

All tests use statevector simulation (exact unitary evolution).
"""

import numpy as np
import pytest

from tgv.quantum.d2q9 import CX, CY, N_DIRS, stream as classical_stream
from tgv.quantum.encoding import encode, decode, n_qubits
from tgv.quantum.streaming import build_streaming_circuit, circuit_stats
from tgv.quantum.backend import StatevectorBackend

backend = StatevectorBackend()


# ── Helper ────────────────────────────────────────────────────────────────────

def quantum_stream(f: np.ndarray) -> np.ndarray:
    """Encode f, apply quantum streaming circuit, decode."""
    N = f.shape[1]
    qc = build_streaming_circuit(N)
    sv_init, norm = encode(f)
    sv_out = backend.run(qc, sv_init)
    return decode(sv_out, N, norm)


# ── Correctness — exit criterion ──────────────────────────────────────────────

@pytest.mark.parametrize("N", [4, 8])
def test_streaming_matches_classical(N):
    """
    PRIMARY EXIT CRITERION: quantum streaming == classical streaming on all
    (9, N, N) entries, tolerance 1e-10.
    """
    rng = np.random.default_rng(seed=42)
    f = rng.random((N_DIRS, N, N)) + 0.01    # positive, like real LBM distributions

    f_q = quantum_stream(f)
    f_c = classical_stream(f)

    max_diff = np.max(np.abs(f_q - f_c))
    assert max_diff < 1e-10, (
        f"N={N}: quantum vs classical max|diff| = {max_diff:.3e} > 1e-10"
    )


@pytest.mark.parametrize("direction", range(N_DIRS))
def test_each_direction_independently(direction):
    """
    Load only one direction register; verify only that direction's rows shift.
    All other directions' amplitudes must be zero before AND after (they stay 0).
    """
    N = 4
    rng = np.random.default_rng(direction)

    # Only direction `direction` has non-zero distributions
    f = np.zeros((N_DIRS, N, N))
    f[direction] = rng.random((N, N)) + 0.01

    f_q = quantum_stream(f)
    f_c = classical_stream(f)

    max_diff = np.max(np.abs(f_q - f_c))
    assert max_diff < 1e-10, (
        f"direction={direction} (cx={CX[direction]},cy={CY[direction]}): "
        f"max|diff| = {max_diff:.3e}"
    )


# ── Rest direction (no shift) ─────────────────────────────────────────────────

def test_rest_direction_unchanged():
    """Direction 0 (rest, cx=cy=0) must leave x,y addresses unmodified."""
    N = 4
    rng = np.random.default_rng(99)
    f = np.zeros((N_DIRS, N, N))
    f[0] = rng.random((N, N))    # only rest direction populated

    f_q = quantum_stream(f)
    np.testing.assert_allclose(f_q[0], f[0], atol=1e-10)


# ── Streaming is a permutation (norm preserving) ──────────────────────────────

@pytest.mark.parametrize("N", [4, 8])
def test_streaming_preserves_norm(N):
    """After quantum streaming, ‖f_out‖_F must equal ‖f_in‖_F."""
    rng = np.random.default_rng(13)
    f = rng.random((N_DIRS, N, N)) + 0.01
    f_q = quantum_stream(f)
    norm_before = np.linalg.norm(f)
    norm_after  = np.linalg.norm(f_q)
    assert abs(norm_after - norm_before) / norm_before < 1e-10


# ── Streaming twice == shift by 2 ─────────────────────────────────────────────

def test_two_streaming_steps():
    """Two successive quantum streaming steps must equal two classical steps."""
    N = 4
    rng = np.random.default_rng(77)
    f = rng.random((N_DIRS, N, N)) + 0.01

    f_q_2 = quantum_stream(quantum_stream(f))
    f_c_2 = classical_stream(classical_stream(f))

    max_diff = np.max(np.abs(f_q_2 - f_c_2))
    assert max_diff < 1e-10, f"Two-step max|diff| = {max_diff:.3e}"


# ── Circuit structure ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("N,expected_q", [(4, 8), (8, 10)])
def test_circuit_qubit_count(N, expected_q):
    """Streaming circuit must use exactly 4 + 2·log₂N qubits."""
    qc = build_streaming_circuit(N)
    assert qc.num_qubits == expected_q, (
        f"N={N}: expected {expected_q} qubits, got {qc.num_qubits}"
    )


def test_circuit_is_unitary():
    """
    Streaming circuit must be unitary (permutation matrix):
    U†U = I  within numerical precision.
    Uses N=4 (8 qubits = 256×256 matrix — tractable).
    """
    from qiskit.quantum_info import Operator
    qc = build_streaming_circuit(N=4)
    op = Operator(qc)
    U = op.data
    diff = np.max(np.abs(U @ U.conj().T - np.eye(U.shape[0])))
    assert diff < 1e-10, f"U†U ≠ I: max error = {diff:.3e}"


@pytest.mark.parametrize("N", [4, 8])
def test_circuit_stats_output(N):
    """circuit_stats() must return sensible counts."""
    stats = circuit_stats(N)
    assert stats["n_qubits"] == n_qubits(N)
    assert stats["n_qubits_formula"] == n_qubits(N)
    assert stats["depth"] > 0
