"""
Quantum amplitude encoding for D2Q9 LBM distribution functions.

Qubit register layout (total: 4 + 2·n_x qubits):
  qubits 0–3          : direction register  |i⟩  (i = 0..8, padded to 0..15)
  qubits 4..4+n_x−1   : x-position register |x⟩  (x = 0..N−1,  N = 2^n_x)
  qubits 4+n_x..4+2n_x−1 : y-position register |y⟩

State index in the 2^(4+2n_x) dimensional statevector:
  s(i, x, y) = i  +  x·2^4  +  y·2^(4+n_x)

Amplitude encoding:
  α_{i,x,y} = f[i, x, y] / ‖f‖_F

This ensures |ψ⟩ is normalised (⟨ψ|ψ⟩ = 1) and streaming (a permutation)
preserves the norm, so decoding requires only one stored scalar Z = ‖f‖_F.

Memory advantage:
  Classical:  9·N² floats
  Quantum:    2·⌈log₂N⌉ + 4 qubits  →  O(log N) vs O(N²)
"""

from __future__ import annotations
import math
import numpy as np
from .d2q9 import N_DIRS, N_DIR_QUBITS


def n_x_for(N: int) -> int:
    """Number of qubits needed to address N grid points (N must be a power of 2)."""
    n = int(math.log2(N))
    if 2**n != N:
        raise ValueError(f"N={N} is not a power of 2")
    return n


def n_qubits(N: int) -> int:
    """Total qubit count for an N×N D2Q9 grid."""
    return N_DIR_QUBITS + 2 * n_x_for(N)


def state_index(direction: int, x: int, y: int, n_x: int) -> int:
    """
    Statevector index for basis state |direction, x, y⟩.

    In Qiskit's convention qubit k is bit k of the state index
    (qubit 0 = LSB of index).
    """
    return direction + (1 << N_DIR_QUBITS) * x + (1 << (N_DIR_QUBITS + n_x)) * y


def _index_array(N: int, n_x: int) -> np.ndarray:
    """
    Pre-compute statevector indices for all (direction, x, y) triples.

    Returns shape (9, N, N) integer array.
    """
    dirs = np.arange(N_DIRS)[:, None, None]
    xs   = np.arange(N)[None, :, None]
    ys   = np.arange(N)[None, None, :]
    return dirs + (1 << N_DIR_QUBITS) * xs + (1 << (N_DIR_QUBITS + n_x)) * ys


def encode(f: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Encode distribution function f into a normalised quantum statevector.

    Parameters
    ----------
    f : (9, N, N) real array — LBM distribution functions

    Returns
    -------
    sv   : complex array of shape (2^n_qubits,) — unit-norm statevector
    norm : float — Frobenius norm of f (needed for decoding)
    """
    _N = f.shape[1]
    if f.shape != (N_DIRS, _N, _N):
        raise ValueError(f"Expected shape (9,N,N), got {f.shape}")

    n_x  = n_x_for(_N)
    n    = N_DIR_QUBITS + 2 * n_x
    dim  = 1 << n
    sv   = np.zeros(dim, dtype=complex)

    norm = float(np.linalg.norm(f))
    if norm == 0.0:
        return sv, 0.0

    idx  = _index_array(_N, n_x)
    sv[idx.ravel()] = (f / norm).ravel().astype(complex)
    return sv, norm


def decode(sv: np.ndarray, N: int, norm: float) -> np.ndarray:
    """
    Decode a quantum statevector back to distribution functions.

    Parameters
    ----------
    sv   : complex statevector (output of a quantum circuit)
    N    : int — grid side length
    norm : float — normalization factor returned by encode()

    Returns
    -------
    f : (9, N, N) real array
    """
    n_x = n_x_for(N)
    idx = _index_array(N, n_x)
    return np.real(sv[idx]) * norm
