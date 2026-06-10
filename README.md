# airbus-quantum-tgv-solver

**Team:** Pantheon  
**Authors:** Benjamin Charles Brümm · Vidhi Jain  
**Affiliation:** Dalhousie University · SpacexAI Research Cohort  
**Challenge:** Airbus Global Quantum + AI Challenge 2026  
**Version:** v0.4 — benchmark scaffold and hybrid QLBM solver

---

## What this is

A prototype hybrid quantum-classical Quantum Lattice Boltzmann Method (QLBM)
solver for the **2D Convecting Taylor-Green Vortex** (TGV) benchmark, developed
for the Airbus Global Quantum + AI Challenge 2026.

This is a **versioned milestone**, not a finished product. It implements:

- An exact analytical ground-truth module validated against the challenge spec
- A high-order pseudo-spectral classical baseline
- A D2Q9 QLBM with exact quantum streaming circuit (Qiskit)
- A Variational Quantum Circuit (VQC) collision approximation
- A Reynolds-number scaling benchmark harness (Re = 10, 100, 500)
- Three required scaling plots: time-to-solution, qubit/memory, L2 error

---

## Challenge scope

This project addresses only the Airbus **2D Convecting Taylor-Green Vortex**
benchmark. It is not a general aircraft CFD solver, airfoil simulator, or
web dashboard.

Parameters: L = 2π m · V0 = 1.0 m/s · Uc = 1.0 m/s · ρ = 1.0 kg/m³ ·
Re ∈ {10, 100, 500}

---

## Repository structure

```
airbus-quantum-tgv-solver/
├── src/tgv/
│   ├── analytical.py              # Layer 1: exact TGV solution
│   ├── visualization.py           # Flow field and energy plots
│   ├── classical/
│   │   └── spectral_solver.py     # Layer 2: pseudo-spectral FFT solver
│   ├── quantum/
│   │   ├── d2q9.py                # D2Q9 lattice constants and classical LBM
│   │   ├── encoding.py            # Quantum amplitude encoding
│   │   ├── streaming.py           # Exact quantum streaming circuit
│   │   ├── collision.py           # VQC + Procrustes collision approximation
│   │   ├── solver.py              # Hybrid QLBM time-stepper
│   │   └── backend.py             # Qiskit statevector backend
│   └── benchmark/
│       ├── metrics.py             # Benchmark data collection
│       └── plots.py               # Required scaling plots
├── tests/                         # 98 unit and integration tests
├── scripts/
│   ├── sweep.py                   # Reynolds sweep benchmark
│   ├── generate_visuals.py        # TGV flow field snapshots
│   └── final_check.py             # Submission checklist
├── notebooks/
│   └── demo_taylor_green_vortex.ipynb
├── reports/
│   └── final_report.md
├── results/
│   ├── benchmarks/                # benchmark_results.csv / .json
│   └── figures/                   # PNG plots
├── configs/                       # Hydra YAML configs
├── experiments/aero_demo/         # Out-of-scope classical CFD demos
├── docs/                          # Architecture notes and decisions
├── pyproject.toml
├── LICENSE                        # MIT
└── .github/workflows/tests.yml    # CI
```

---

## Installation

**Windows PowerShell:**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

**Unix / macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

---

## Run tests

```bash
python -m pytest -v
```

All 98 tests should pass (Layer 1 analytical, Layer 2 classical, Layer 3
quantum encoding/streaming/collision/QLBM integration).

---

## Run benchmarks

```bash
python scripts/sweep.py --full     # Re=[10, 100, 500], t_end=0.5s
python scripts/sweep.py --ci       # Re=[10, 100], fast CI mode
```

Results are written to:
- `results/benchmarks/benchmark_results.csv`
- `results/benchmarks/benchmark_results.json`

Plots are saved to `results/figures/`.

---

## Regenerate plots

```bash
python scripts/sweep.py --full
```

Or individually via `src/tgv/benchmark/plots.py`.

---

## Open the notebook

```bash
pip install -e ".[dev]"
jupyter lab notebooks/demo_taylor_green_vortex.ipynb
```

---

## Implemented methods

| Module | Method | Status |
|--------|--------|--------|
| `analytical.py` | Exact TGV closed-form solution | Complete |
| `classical/spectral_solver.py` | Pseudo-spectral FFT + RK4 | Complete |
| `quantum/encoding.py` | Amplitude encoding (O(log N) qubits) | Complete |
| `quantum/streaming.py` | Exact unitary streaming circuit | Complete |
| `quantum/collision.py` | Procrustes-optimal + trained VQC | Complete |
| `quantum/solver.py` | Hybrid QLBM time-stepper | Complete |
| `benchmark/` | Reynolds sweep harness + plots | Complete |

---

## Current status

**v0.4 versioned milestone.** The hybrid QLBM solver is implemented and
validated. Tests pass. Benchmark harness produces the three required scaling
plots. The report is in `reports/final_report.md`.

Known limitations:
- Default accurate mode is hybrid (quantum streaming + classical BGK)
- VQC collision is a prototype approximation; spatially-uniform rotation
- Statevector simulation limits grids to N ≤ 32 on a laptop
- No proven quantum advantage yet (statevector emulation, not quantum hardware)
- Re=500 at N=8 approaches LBM stability limit (omega ≈ 1.99)

---

## License

MIT License — see `LICENSE`.

Copyright (c) 2026 Benjamin Charles Brümm and Vidhi Jain.

---

## Citation

If you use this code, please cite as:

> Brümm, B. C. and Jain, V. (2026). *airbus-quantum-tgv-solver v0.4:
> Hybrid quantum-classical QLBM solver for the 2D Convecting Taylor-Green
> Vortex.* Airbus Global Quantum + AI Challenge 2026, Team Pantheon.
> GitHub: https://github.com/BenjaminBrumm/airbus-quantum-tgv-solver
