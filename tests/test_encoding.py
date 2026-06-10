"""
Tests for quantum amplitude encoding (Layer 3, encoding.py).

Validates that the encode/decode roundtrip is exact, state indices are
consistent with the qubit layout, and qubit counts match the formula
2·⌈log₂N⌉ + 4.
"""

import math
import numpy as np
import pytest

from tgv.quantum.encoding import (
    encode,
    decode,
    state_index,
    n_qubits,
    n_x_for,
)
from tgv.quantum.d2q9 import N_DIRS


# ── Index formula ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("N", [4, 8, 16])
def test_state_index_formula(N):
    """state_index(i, x, y) must equal i + 16*x + 16*N*y."""
    n_x = n_x_for(N)
    rng = np.random.default_rng(0)
    for _ in range(20):
        i = int(rng.integers(0, N_DIRS))
        x = int(rng.integers(0, N))
        y = int(rng.integers(0, N))
        expected = i + 16 * x + 16 * N * y
        assert state_index(i, x, y, n_x) == expected


@pytest.mark.parametrize("N,expected_q", [(4, 8), (8, 10), (16, 12)])
def test_qubit_count_formula(N, expected_q):
    """Total qubit count must be 4 + 2·log₂N (from DECISIONS.md D-002 / plan §3)."""
    assert n_qubits(N) == expected_q


# ── Statevector properties ────────────────────────────────────────────────────

@pytest.mark.parametrize("N", [4, 8])
def test_statevector_unit_norm(N):
    """Encoded statevector must be unit-norm."""
    rng = np.random.default_rng(42)
    f = rng.random((N_DIRS, N, N)) + 0.01   # positive distributions
    sv, _ = encode(f)
    assert abs(np.dot(sv.conj(), sv).real - 1.0) < 1e-12


@pytest.mark.parametrize("N", [4, 8])
def test_statevector_dimension(N):
    """Encoded statevector must have dimension 2^(4 + 2*log₂N)."""
    f = np.ones((N_DIRS, N, N))
    sv, _ = encode(f)
    assert sv.shape == (2 ** n_qubits(N),)


@pytest.mark.parametrize("N", [4, 8])
def test_unused_states_are_zero(N):
    """Direction indices 9–15 must have zero amplitude."""
    f = np.ones((N_DIRS, N, N))
    sv, _ = encode(f)
    n_x = n_x_for(N)
    for unused_dir in range(N_DIRS, 16):   # directions 9..15 are unused
        for x in range(N):
            for y in range(N):
                idx = state_index(unused_dir, x, y, n_x)
                assert sv[idx] == 0.0, f"Non-zero amplitude at unused dir={unused_dir}"


# ── Roundtrip ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("N", [4, 8])
def test_encode_decode_roundtrip(N):
    """decode(encode(f)) must recover f exactly (up to floating-point precision)."""
    rng = np.random.default_rng(7)
    f = rng.random((N_DIRS, N, N))
    sv, norm = encode(f)
    f_rec = decode(sv, N, norm)
    np.testing.assert_allclose(f_rec, f, atol=1e-12)


def test_encode_decode_zeros():
    """Encoding an all-zero f returns zero statevector with norm=0."""
    f = np.zeros((N_DIRS, 4, 4))
    sv, norm = encode(f)
    assert norm == 0.0
    assert np.all(sv == 0.0)


def test_encode_amplitude_values():
    """
    For a single non-zero entry f[i,x,y]=v, the statevector should have
    amplitude 1.0 at index s(i,x,y) and 0 elsewhere.
    """
    N = 4
    n_x = n_x_for(N)
    f = np.zeros((N_DIRS, N, N))
    f[2, 1, 3] = 5.0        # only one non-zero entry
    sv, norm = encode(f)

    assert abs(norm - 5.0) < 1e-12
    idx = state_index(2, 1, 3, n_x)
    assert abs(sv[idx] - 1.0) < 1e-12
    # All other entries must be zero
    sv_copy = sv.copy()
    sv_copy[idx] = 0.0
    assert np.all(np.abs(sv_copy) < 1e-12)
