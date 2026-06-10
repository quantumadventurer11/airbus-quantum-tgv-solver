"""
Backend abstraction for QLBM circuit simulation.

Provides a thin layer over Qiskit Aer so that the rest of the codebase
can switch between:
  - statevector:  exact simulation (Aer statevector or qiskit.quantum_info)
  - shot-based:   Aer sampler with N shots (for NISQ noise emulation)
  - IBM Runtime:  cloud IBM Quantum hardware / FakeBackend

For Week 3 (streaming tests), only the statevector backend is needed.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


class StatevectorBackend:
    """
    Exact statevector simulation using qiskit.quantum_info.Statevector.

    No shots, no noise — bitwise-exact unitary evolution.
    Suitable for N ≤ 16 (≤ 12 qubits).
    """

    def run(
        self,
        circuit: QuantumCircuit,
        initial_sv: np.ndarray,
    ) -> np.ndarray:
        """
        Apply `circuit` to `initial_sv` and return the resulting statevector.

        Parameters
        ----------
        circuit    : QuantumCircuit with n qubits
        initial_sv : complex array of shape (2^n,) — unit-norm initial state

        Returns
        -------
        sv_out : complex array of shape (2^n,)
        """
        sv = Statevector(initial_sv)
        sv_out = sv.evolve(circuit)
        return np.array(sv_out)

    @staticmethod
    def max_n_qubits() -> int:
        """Practical cap for exact statevector simulation (memory limit ~8 GB)."""
        return 28   # 2^28 complex128 = 4 GB


def get_backend(name: str = "statevector") -> StatevectorBackend:
    """
    Factory: return a backend by name.

    Supported: "statevector" (default)
    Future:    "aer_statevector", "aer_shot", "ibm_runtime"
    """
    if name == "statevector":
        return StatevectorBackend()
    raise ValueError(f"Unknown backend '{name}'. Supported: 'statevector'")
