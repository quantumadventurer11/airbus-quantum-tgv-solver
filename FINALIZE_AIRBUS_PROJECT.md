# Finalization Instructions — Airbus Quantum Taylor-Green Vortex Solver

Project folder:

```text
C:\Windows\system32\airbus-quantum-challenge
```

Project title:

```text
airbus-quantum-tgv-solver
```

Team:

```text
Pantheon
```

Authors:

```text
Benjamin Charles Brümm
Vidhi Jain
```

Affiliation note:

```text
Benjamin Charles Brümm is affiliated with Dalhousie University and is a member of the SpacexAI Research Cohort.
```

Repository visibility:

```text
Public GitHub repository
```

License:

```text
MIT License
```

## Project positioning

This is a hybrid quantum-classical QLBM solver for the 2D Convecting Taylor-Green Vortex, developed for the Airbus Global Quantum + AI Challenge 2026.

## Controlling scope rule

Do **not** expand the final deliverable beyond the Airbus challenge statement.

The official submission must focus on:

- 2D Convecting Taylor-Green Vortex
- incompressible Navier-Stokes benchmark
- exact analytical solution
- hybrid quantum-classical QLBM solver
- Reynolds-number scaling
- time-to-solution
- memory requirement or qubit complexity
- L2 velocity error or kinetic-energy decay

Do **not** turn this into:

- a general aircraft CFD project
- an airfoil simulator
- a web dashboard
- a broad aerospace visualization platform
- a production-grade aerodynamic simulator

The Airbus problem statement asks participants to solve the 2D Convecting Taylor-Green Vortex and quantify scaling as Reynolds number increases, including:

- time-to-solution
- memory requirement or qubit complexity
- error scaling, especially L2 velocity error or kinetic-energy decay

Use this as the controlling scope for all finalization work.

## Current project status assumption

The repository already contains:

- analytical Taylor-Green Vortex solver
- classical spectral solver
- quantum D2Q9/QLBM scaffolding
- quantum encoding
- quantum streaming
- VQC collision approximation
- hybrid QLBM solver
- tests

Continue from the existing state. Do not rebuild from scratch.

## Main objective

Prepare the project for a public GitHub push as a credible versioned milestone, **not** a final finished product.

Use version-history language, not final-product language.

Recommended commit title:

```bash
git commit -m "v0.4: add hybrid QLBM solver, benchmark scaffolding, and challenge documentation"
```

Do **not** use:

```bash
git commit -m "Finalize Airbus quantum Taylor-Green vortex solver submission"
```

because the project is not finished yet.

---

# Tasks

## 1. Inspect the repository state

Run:

```bash
git status
```

List the project structure.

Identify:

- source files
- tests
- scripts
- reports
- notebooks
- results
- temporary/cache files
- out-of-scope aerodynamic demo files

Do not delete anything important without first moving it safely.

## 2. Verify tests

Run:

```bash
python -m pytest -v
```

If imports fail because Python cannot find `src`, fix `pyproject.toml` so pytest includes:

```toml
[tool.pytest.ini_options]
pythonpath = ["src"]
```

If dependency issues appear, update `pyproject.toml` properly.

All existing tests should pass before pushing.

## 3. Keep the project scoped to Taylor-Green Vortex

The main README, report, notebook, benchmark harness, and plots must focus on:

- 2D Convecting Taylor-Green Vortex
- incompressible Navier-Stokes benchmark
- exact analytical solution
- Reynolds numbers Re = 10, Re = 100, and Re = 500
- hybrid quantum-classical QLBM method
- classical comparison
- time-to-solution
- memory/qubit requirement
- L2 velocity error
- kinetic-energy decay if already supported

Do not present airfoil or cylinder simulations as part of the official Airbus challenge solution.

## 4. Move out-of-scope aerodynamic demo files

If files related to NACA airfoil, cylinder flow, Kármán vortex streets, or general aerodynamic demos exist, move them to:

```text
experiments/aero_demo/
```

Add a short README there:

```text
experiments/aero_demo/README.md
```

The README should explain:

```text
These files are exploratory classical CFD visualizations and are not part of the official Airbus Taylor-Green Vortex challenge submission. They are retained only as optional background experiments.
```

Do not include these files in the main report except as a clearly labelled out-of-scope experiment. Prefer omitting them entirely from the final report.

## 5. Complete or refine the benchmark harness

Create or update:

```text
src/tgv/benchmark/__init__.py
src/tgv/benchmark/metrics.py
src/tgv/benchmark/plots.py
scripts/sweep.py
```

Benchmark Reynolds numbers:

```text
Re = 10
Re = 100
Re = 500
```

If Re = 500 is too slow or unstable, implement a graceful fallback:

- record the failure or timeout
- explain it in the benchmark results
- still produce plots with the available data
- do not fake results

The benchmark should collect:

- Reynolds number
- grid size
- number of time steps
- wall-clock runtime
- time-to-solution
- memory estimate
- qubit count estimate
- L2 velocity error against exact analytical solution
- kinetic energy if available
- solver mode used

Output benchmark data to:

```text
results/benchmarks/benchmark_results.csv
results/benchmarks/benchmark_results.json
```

Generate plots:

```text
results/figures/time_to_solution_vs_re.png
results/figures/memory_or_qubits_vs_re.png
results/figures/l2_error_vs_re.png
```

Optionally generate, if already supported:

```text
results/figures/kinetic_energy_decay.png
```

## 6. Add a minimal Jupyter notebook viewer

Create:

```text
notebooks/demo_taylor_green_vortex.ipynb
```

The notebook should be lightweight and challenge-scoped.

It should show:

- the analytical Taylor-Green Vortex velocity field
- the hybrid QLBM solver output on a small case
- the velocity error field
- the benchmark plots
- optionally the quantum/VQC circuit diagram, only if it renders reliably

Do not create:

- Streamlit dashboard
- React dashboard
- Flask app
- full web portal
- aircraft CFD interface

A notebook is enough.

## 7. Create or update the final report

Create:

```text
reports/final_report.md
```

Also create:

```text
reports/final_report.pdf
```

only if PDF export works without adding fragile dependencies. If PDF export fails, leave the Markdown report and document how to export it later.

The report should be executive/judge-friendly with technical depth.

Use this structure:

```markdown
# Hybrid Quantum-Classical QLBM Solver for the 2D Convecting Taylor-Green Vortex

## 1. Executive Summary
Explain the challenge, the solver, and the current milestone. Use conservative wording. Do not claim proven quantum advantage.

## 2. Challenge Scope
State that this project addresses the Airbus 2D Convecting Taylor-Green Vortex benchmark.

## 3. Mathematical Problem
Define the 2D incompressible Navier-Stokes setting, Taylor-Green Vortex initial condition, analytical decay solution, and validation metrics.

Use the challenge parameters:
- L = 2π
- V0 = 1.0
- Uc = 1.0
- Vc = 0.0
- ρ = 1.0
- Re = 10, 100, 500
- ν = V0 L / Re
- p0 = 0.0

## 4. Solver Architecture
Explain:
- analytical reference solution
- classical spectral solver
- D2Q9 lattice representation
- quantum amplitude encoding
- exact quantum streaming circuit
- VQC collision approximation
- hybrid QLBM mode: quantum streaming + classical BGK collision

Be honest that the default accurate solver is hybrid, not a fully fault-tolerant end-to-end quantum CFD solver.

## 5. Treatment of Nonlinearity and Nonunitarity
Explain that fluid dynamics is nonlinear and dissipative, while quantum circuits are linear and unitary. State that the project handles this through a hybrid QLBM design and VQC collision approximation.

## 6. Benchmark Methodology
Describe:
- Reynolds sweep
- grid choice
- time horizon
- exact analytical comparison
- L2 velocity error
- time-to-solution
- memory/qubit estimate
- kinetic energy decay if included

## 7. Results
Insert or reference:
- time_to_solution_vs_re.png
- memory_or_qubits_vs_re.png
- l2_error_vs_re.png
- kinetic_energy_decay.png if available

## 8. Limitations
Include:
- no proven industrial quantum advantage yet
- VQC collision is a prototype approximation
- default accurate mode is hybrid quantum-classical
- statevector simulation limits grid sizes
- Re = 500 may require careful resolution/runtime tradeoffs
- FTQC relevance is forward-looking

## 9. Reproducibility
Include commands to install, test, run benchmarks, regenerate plots, and open the notebook.

## 10. Next Work
Include:
- larger Reynolds sweeps
- improved VQC collision locality
- hardware-aware circuit optimization
- noise modelling
- comparison with stronger classical baselines
- FTQC resource estimation
```

Tone:

Conservative, rigorous, and credible.

Avoid:

- “we prove quantum advantage”
- “industrial-ready CFD”
- “complete aircraft solver”
- “final Airbus solution”
- “production-grade aerodynamic simulator”

Use language such as:

- “prototype”
- “versioned milestone”
- “hybrid NISQ-era approach”
- “potential pathway”
- “benchmark scaffold”
- “validated against the exact Taylor-Green Vortex solution”

## 8. Update README.md

README.md should include:

```markdown
# airbus-quantum-tgv-solver

Team Pantheon

Authors:
- Benjamin Charles Brümm
- Vidhi Jain

Affiliation:
- Dalhousie University
- SpacexAI Research Cohort
```

Then include:

- short project summary
- challenge scope
- repository structure
- installation instructions
- how to run tests
- how to run benchmarks
- how to regenerate plots
- how to open the notebook
- summary of implemented methods
- current status
- limitations
- license
- citation note

Use clear commands.

Example Windows PowerShell commands:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
python -m pytest -v
python scripts/sweep.py --full
jupyter lab notebooks/demo_taylor_green_vortex.ipynb
```

Example Unix/macOS/Linux commands:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python -m pytest -v
python scripts/sweep.py --full
jupyter lab notebooks/demo_taylor_green_vortex.ipynb
```

## 9. Update pyproject.toml

Use `pyproject.toml` as the main dependency and configuration file.

Make sure it includes:

- project metadata
- Python version requirement
- runtime dependencies
- optional dev dependencies if appropriate
- pytest config
- package discovery from `src`

Suggested project metadata:

```toml
name = "airbus-quantum-tgv-solver"
description = "Hybrid quantum-classical QLBM solver for the 2D Convecting Taylor-Green Vortex benchmark."
authors = [
  {name = "Benjamin Charles Brümm"},
  {name = "Vidhi Jain"}
]
license = {text = "MIT"}
```

## 10. Add MIT License

Create:

```text
LICENSE
```

Use the standard MIT License.

Copyright holder:

```text
Benjamin Charles Brümm and Vidhi Jain
```

Year:

```text
2026
```

## 11. Add GitHub Actions CI

Create:

```text
.github/workflows/tests.yml
```

The workflow must:

- run on push and pull_request
- install Python
- install package dependencies from `pyproject.toml`
- run pytest
- run benchmark in CI mode

Because the full Re = 500 benchmark may be slow or unstable on GitHub Actions, support two benchmark modes:

Local full benchmark:

```bash
python scripts/sweep.py --full
```

CI benchmark:

```bash
python scripts/sweep.py --ci
```

The CI workflow should run:

```bash
python -m pytest -v
python scripts/sweep.py --ci
```

If `scripts/sweep.py` currently has no flags, add them.

## 12. Create a final validation script

Create:

```text
scripts/final_check.py
```

It should verify:

- README.md exists
- LICENSE exists
- reports/final_report.md exists
- notebooks/demo_taylor_green_vortex.ipynb exists
- benchmark CSV/JSON exists or can be generated
- required figures exist or can be generated
- imports work
- pytest can be invoked

It should print a clear pass/fail checklist.

## 13. Add .gitignore

Ensure `.gitignore` excludes:

```gitignore
.venv/
venv/
__pycache__/
*.pyc
.pytest_cache/
.mypy_cache/
.ruff_cache/
.ipynb_checkpoints/
.DS_Store
Thumbs.db
*.log
.env
.env.*
dist/
build/
*.egg-info/
htmlcov/
.coverage
```

Do not ignore:

```text
results/figures/*.png
results/benchmarks/*.csv
results/benchmarks/*.json
```

because generated benchmark plots and results should be committed.

## 14. Commit generated benchmark plots and results

Commit:

- benchmark CSV
- benchmark JSON
- small PNG plots
- final report Markdown
- final report PDF if successfully generated
- notebook
- source
- tests
- scripts
- README
- LICENSE
- GitHub Actions workflow

Do not commit:

- virtual environments
- caches
- temporary files
- huge GIFs unless clearly required
- local machine paths
- API keys
- secrets

## 15. GitHub setup and push

Target repository name:

```text
airbus-quantum-tgv-solver
```

Visibility:

```text
public
```

If the GitHub repo already exists, use its URL.

If not, create it manually on GitHub first, then run:

```bash
git remote add origin https://github.com/<USERNAME>/airbus-quantum-tgv-solver.git
```

Then:

```bash
git status
git add README.md LICENSE pyproject.toml src tests scripts notebooks reports results .github .gitignore experiments
git status
git commit -m "v0.4: add hybrid QLBM solver, benchmark scaffolding, and challenge documentation"
git branch -M main
git push -u origin main
```

If there are already commits, do not squash unless asked. Use several smaller commits if the changes are naturally separable.

Preferred multi-commit approach because the project is still evolving:

```bash
git add README.md LICENSE pyproject.toml .gitignore
git commit -m "v0.4.1: organize challenge scope and repository metadata"

git add src/tgv/benchmark scripts/sweep.py results/benchmarks results/figures
git commit -m "v0.4.2: add benchmark harness and Reynolds scaling outputs"

git add notebooks reports experiments
git commit -m "v0.4.3: add Taylor-Green demo notebook and report draft"

git add .github scripts/final_check.py
git commit -m "v0.4.4: add CI workflow and final validation checks"
```

Then push:

```bash
git branch -M main
git push -u origin main
```

## 16. Final response after completion

When finished, report:

- current branch
- commit hash or hashes
- whether tests pass
- whether benchmark ran successfully
- whether Re = 500 completed
- generated benchmark files
- generated plot files
- report path
- notebook path
- GitHub remote URL
- any known limitations
- next recommended development step

Do not claim the project is complete. Say it is a versioned milestone ready for review and continued development.
