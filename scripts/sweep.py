#!/usr/bin/env python
"""
Reynolds number scaling sweep for the hybrid QLBM TGV solver.

Usage:
    python scripts/sweep.py --full    # Re=[10, 100, 500], t_end=0.5 (full benchmark)
    python scripts/sweep.py --ci      # Re=[10, 100], N=4, t_end=0.1  (fast CI check)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

# Allow running from project root without installing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tgv.benchmark.metrics import run_sweep
from tgv.benchmark.plots import generate_plots

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "benchmarks")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "figures")


def main() -> None:
    parser = argparse.ArgumentParser(description="QLBM Reynolds sweep benchmark")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="Full benchmark: Re=[10,100,500], t_end=0.5")
    mode.add_argument("--ci",   action="store_true", help="CI benchmark:  Re=[10,100],    t_end=0.1")
    args = parser.parse_args()

    if args.ci:
        re_list = [10.0, 100.0]
        t_end   = 0.1
        timeout = 60.0
        ci_mode = True
        print("=== CI benchmark mode (Re=[10,100], t_end=0.1) ===")
    else:
        re_list = [10.0, 100.0, 500.0]
        t_end   = 0.5
        timeout = 180.0
        ci_mode = False
        print("=== Full benchmark mode (Re=[10,100,500], t_end=0.5) ===")

    print(f"Reynolds numbers: {re_list}")
    print(f"t_end = {t_end}s,  timeout per run = {timeout}s\n")

    results = run_sweep(re_list, t_end=t_end, ci_mode=ci_mode, timeout_s=timeout)

    # Save CSV and JSON
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path  = os.path.join(RESULTS_DIR, "benchmark_results.csv")
    json_path = os.path.join(RESULTS_DIR, "benchmark_results.json")

    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as fh:
        json.dump(results, fh, indent=2)

    print(f"\nResults saved:")
    print(f"  {csv_path}")
    print(f"  {json_path}")

    # Generate plots
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print("\nGenerating plots ...")
    paths = generate_plots(results, re_list=re_list, out_dir=FIGURES_DIR)
    for p in paths:
        print(f"  {p}")

    # Summary table
    print("\n--- Benchmark summary ---")
    print(df[["Re", "N_qlbm", "n_qubits", "n_steps", "wall_time_s", "l2_error", "timed_out"]].to_string(index=False))

    timed_out = [r for r in results if r.get("timed_out")]
    if timed_out:
        print(f"\nNote: {len(timed_out)} run(s) timed out: Re={[r['Re'] for r in timed_out]}")
        print("Timed-out runs are recorded as NaN in results but plots use available data.")

    print("\nDone.")


if __name__ == "__main__":
    main()
