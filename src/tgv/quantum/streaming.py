"""
Quantum streaming circuit for D2Q9 QLBM.

Classical streaming: f_i(x, y) → f_i(x + cx_i mod N,  y + cy_i mod N)

This is a PERMUTATION of basis states — exactly unitary — so it maps
directly to quantum shift gates:
  |i⟩|x⟩|y⟩  →  |i⟩|x + cx_i mod N⟩|y + cy_i mod N⟩

Implementation: for each of the 8 non-rest directions, apply a
controlled increment or decrement on the x (and/or y) register,
controlled on the direction register being in state |i⟩.

Controlled increment (+1 mod 2^n) on register [q_0 (LSB) ... q_{n-1} (MSB)],
controlled on a set of "direction" qubits (all must be |1⟩ for the gate to fire):

  Apply from MSB to LSB  (so higher-order flips see original lower bits):
    for j = n-1 down to 0:
      MCX( dir_qubits + reg_qubits[:j],  reg_qubits[j] )

Controlled decrement = reverse of increment (all gates self-inverse):
    for j = 0 to n-1:
      MCX( dir_qubits + reg_qubits[:j],  reg_qubits[j] )

To control on direction i (may have 0-bits): prepend X gates on direction
qubits where bit k of i is 0, apply shift, then undo those X gates.

Qubit layout (same as encoding.py):
  q[0:4]                : direction register
  q[4 : 4+n_x]          : x register  (q[4]=LSB, q[4+n_x-1]=MSB)
  q[4+n_x : 4+2*n_x]    : y register  (q[4+n_x]=LSB, q[4+2*n_x-1]=MSB)
"""

from __future__ import annotations
import math

from qiskit import QuantumCircuit

from .d2q9 import CX, CY, N_DIRS, N_DIR_QUBITS
from .encoding import n_x_for, n_qubits


# ── Internal helpers ──────────────────────────────────────────────────────────

def _add_controlled_increment(
    qc: QuantumCircuit,
    dir_qubits: list[int],
    reg_qubits: list[int],
) -> None:
    """
    Append gates for +1 (mod 2^n) on reg_qubits, controlled by dir_qubits.

    All dir_qubits must be |1⟩ for any gate to fire.
    Applies from MSB (reg_qubits[-1]) to LSB (reg_qubits[0]).
    """
    n = len(reg_qubits)
    for j in range(n - 1, -1, -1):
        controls = dir_qubits + reg_qubits[:j]
        target   = reg_qubits[j]
        qc.mcx(controls, target)


def _add_controlled_decrement(
    qc: QuantumCircuit,
    dir_qubits: list[int],
    reg_qubits: list[int],
) -> None:
    """
    Append gates for −1 (mod 2^n) on reg_qubits, controlled by dir_qubits.

    Decrement = inverse of increment = gates in reversed order (all self-inverse).
    Applies from LSB (reg_qubits[0]) to MSB (reg_qubits[-1]).
    """
    n = len(reg_qubits)
    for j in range(n):
        controls = dir_qubits + reg_qubits[:j]
        target   = reg_qubits[j]
        qc.mcx(controls, target)


# ── Public API ────────────────────────────────────────────────────────────────

def build_streaming_circuit(N: int) -> QuantumCircuit:
    """
    Build the D2Q9 streaming circuit for an N×N periodic grid.

    The circuit acts on  4 + 2·⌈log₂N⌉  qubits and implements:
        |i⟩|x⟩|y⟩  →  |i⟩|x + cx_i mod N⟩|y + cy_i mod N⟩

    for all 9 D2Q9 directions simultaneously (direction 0 = rest, no shift).

    Parameters
    ----------
    N : int — grid side length, must be a power of 2

    Returns
    -------
    QuantumCircuit with name "D2Q9_stream_N{N}"
    """
    n_x    = n_x_for(N)
    n_tot  = N_DIR_QUBITS + 2 * n_x
    qc     = QuantumCircuit(n_tot, name=f"D2Q9_stream_N{N}")

    dir_qubits = list(range(N_DIR_QUBITS))
    x_qubits   = list(range(N_DIR_QUBITS, N_DIR_QUBITS + n_x))
    y_qubits   = list(range(N_DIR_QUBITS + n_x, N_DIR_QUBITS + 2 * n_x))

    for direction in range(N_DIRS):
        cx = int(CX[direction])
        cy = int(CY[direction])

        if cx == 0 and cy == 0:
            continue    # direction 0 (rest): no shift

        # ── Prep: flip direction qubits where bit k of direction is 0 ─────────
        # After this, the direction register equals |1111⟩  ⟺  original = |i⟩
        flip_bits = [k for k in range(N_DIR_QUBITS) if not (direction >> k) & 1]
        for k in flip_bits:
            qc.x(k)

        # ── Apply shift(s) controlled on |1111⟩ direction register ──────────
        if cx == +1:
            _add_controlled_increment(qc, dir_qubits, x_qubits)
        elif cx == -1:
            _add_controlled_decrement(qc, dir_qubits, x_qubits)

        if cy == +1:
            _add_controlled_increment(qc, dir_qubits, y_qubits)
        elif cy == -1:
            _add_controlled_decrement(qc, dir_qubits, y_qubits)

        # ── Undo prep ─────────────────────────────────────────────────────────
        for k in flip_bits:
            qc.x(k)

    return qc


def circuit_stats(N: int) -> dict:
    """Return qubit count and gate count for the streaming circuit."""
    qc = build_streaming_circuit(N)
    n_x = n_x_for(N)
    ops = qc.count_ops()
    return {
        "N":           N,
        "n_qubits":    qc.num_qubits,
        "n_qubits_formula": N_DIR_QUBITS + 2 * n_x,
        "gate_counts": dict(ops),
        "depth":       qc.depth(),
    }
