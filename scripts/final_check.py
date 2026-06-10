#!/usr/bin/env python
"""
Final validation checklist for the airbus-quantum-tgv-solver v0.4 milestone.

Prints a PASS/FAIL line for each requirement. Run from the project root:
    python scripts/final_check.py
"""
from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

results: list[tuple[str, bool, str]] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    results.append((label, condition, detail))


def file_exists(rel_path: str) -> bool:
    return os.path.isfile(os.path.join(ROOT, rel_path))


def dir_exists(rel_path: str) -> bool:
    return os.path.isdir(os.path.join(ROOT, rel_path))


# ── File presence ─────────────────────────────────────────────────────────────
check("README.md exists",                       file_exists("README.md"))
check("LICENSE exists",                         file_exists("LICENSE"))
check(".gitignore exists",                      file_exists(".gitignore"))
check("reports/final_report.md exists",         file_exists("reports/final_report.md"))
check("notebooks/demo_taylor_green_vortex.ipynb exists",
      file_exists("notebooks/demo_taylor_green_vortex.ipynb"))
check(".github/workflows/tests.yml exists",     file_exists(".github/workflows/tests.yml"))
check("pyproject.toml exists",                  file_exists("pyproject.toml"))

# ── Benchmark outputs ─────────────────────────────────────────────────────────
check("results/benchmarks/benchmark_results.csv exists",
      file_exists("results/benchmarks/benchmark_results.csv"))
check("results/benchmarks/benchmark_results.json exists",
      file_exists("results/benchmarks/benchmark_results.json"))
check("results/figures/time_to_solution_vs_re.png exists",
      file_exists("results/figures/time_to_solution_vs_re.png"))
check("results/figures/memory_or_qubits_vs_re.png exists",
      file_exists("results/figures/memory_or_qubits_vs_re.png"))
check("results/figures/l2_error_vs_re.png exists",
      file_exists("results/figures/l2_error_vs_re.png"))
check("results/figures/kinetic_energy_decay.png exists",
      file_exists("results/figures/kinetic_energy_decay.png"))

# ── Aero demo isolation ───────────────────────────────────────────────────────
check("generate_aero_visuals.py moved out of scripts/",
      not file_exists("scripts/generate_aero_visuals.py"))
check("aero_*.png moved out of results/figures/",
      not any(
          f.startswith("aero_")
          for f in os.listdir(os.path.join(ROOT, "results", "figures"))
          if os.path.isfile(os.path.join(ROOT, "results", "figures", f))
      ))

# ── Source structure ──────────────────────────────────────────────────────────
check("src/tgv/benchmark/__init__.py exists",   file_exists("src/tgv/benchmark/__init__.py"))
check("src/tgv/benchmark/metrics.py exists",    file_exists("src/tgv/benchmark/metrics.py"))
check("src/tgv/benchmark/plots.py exists",      file_exists("src/tgv/benchmark/plots.py"))
check("scripts/sweep.py exists",                file_exists("scripts/sweep.py"))

# ── Python imports ────────────────────────────────────────────────────────────
try:
    from tgv.analytical import velocity_exact        # noqa: F401
    check("from tgv.analytical import velocity_exact", True)
except Exception as exc:
    check("from tgv.analytical import velocity_exact", False, str(exc))

try:
    from tgv.quantum.solver import QLBMSolver        # noqa: F401
    check("from tgv.quantum.solver import QLBMSolver", True)
except Exception as exc:
    check("from tgv.quantum.solver import QLBMSolver", False, str(exc))

try:
    from tgv.benchmark import run_single             # noqa: F401
    check("from tgv.benchmark import run_single", True)
except Exception as exc:
    check("from tgv.benchmark import run_single", False, str(exc))

# ── pytest collection ─────────────────────────────────────────────────────────
ret = subprocess.run(
    [sys.executable, "-m", "pytest", "--co", "-q"],
    capture_output=True, text=True,
    cwd=ROOT,
)
check("pytest --co exits 0 (all tests collectible)",
      ret.returncode == 0,
      ret.stderr.strip()[:120] if ret.returncode != 0 else "")

# ── Print results ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  airbus-quantum-tgv-solver  v0.4  final checklist")
print("=" * 60)
n_pass = sum(1 for _, ok, _ in results if ok)
n_fail = len(results) - n_pass

for label, ok, detail in results:
    status = "PASS" if ok else "FAIL"
    mark   = "+" if ok else "-"
    line   = f"  [{status}] {mark}  {label}"
    if not ok and detail:
        line += f"\n         >> {detail}"
    print(line)

print("=" * 60)
print(f"  {n_pass}/{len(results)} checks passed", end="")
if n_fail:
    print(f"  ({n_fail} FAILED)")
else:
    print("  - all clear")
print("=" * 60 + "\n")

sys.exit(0 if n_fail == 0 else 1)
