"""
Streamlit interactive demo — Airbus Quantum TGV Solver
Team Pantheon: Benjamin Charles Brumm, Vidhi Jain
Airbus Global Quantum + AI Challenge 2026
"""
from __future__ import annotations

import math
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Allow import from src/ whether running locally or on Streamlit Cloud
ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from tgv.quantum.solver import (
    QLBMSolver,
    lbm_ic,
    omega_lbm,
    n_steps_for_time,
    dt_physical,
)
from tgv.quantum.streaming import build_streaming_circuit

# ── Physical constants ─────────────────────────────────────────────────────────
L_PHYS = 2.0 * math.pi
V0 = 1.0
RESULTS_CSV = os.path.join(ROOT, "results", "benchmarks", "benchmark_results.csv")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Airbus Quantum TGV Solver",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("Hybrid Quantum-Classical QLBM Solver")
st.caption(
    "2D Convecting Taylor-Green Vortex  |  "
    "Airbus Global Quantum + AI Challenge 2026  |  "
    "Team Pantheon"
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Analytical TGV",
    "Live QLBM Solver",
    "Benchmark Scaling",
    "Quantum Circuit",
])

# ══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Overview
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_desc, col_res = st.columns([2, 1])

    with col_desc:
        st.header("Project overview")
        st.markdown("""
**Team Pantheon** &nbsp;·&nbsp; Benjamin Charles Brumm &nbsp;·&nbsp; Vidhi Jain
Dalhousie University &nbsp;·&nbsp; SpacexAI Research Cohort

This demo presents a prototype hybrid quantum-classical **Quantum Lattice Boltzmann
Method (QLBM)** solver for the 2D Convecting Taylor-Green Vortex (TGV) benchmark,
developed for the Airbus Global Quantum + AI Challenge 2026.

**How it works:**
1. The TGV velocity field is amplitude-encoded into a quantum statevector
   using **4 + 2·log₂N qubits** for an N×N grid.
2. The streaming step (distribution function advection) is applied as an
   **exact unitary quantum circuit** — this step genuinely runs on quantum hardware.
3. The BGK collision step (viscous relaxation) is applied classically (hybrid mode).
4. The output is decoded and validated against the **exact analytical TGV solution**.

**Scope:** This is a versioned prototype milestone — a hybrid NISQ-era approach,
not a claim of proven quantum advantage over classical solvers.
        """)

    with col_res:
        st.header("Benchmark results")
        st.markdown("Validated against the exact TGV solution at *t* = 0.5 s:")
        st.dataframe(
            pd.DataFrame({
                "Re": [10, 100, 500],
                "Grid N": [8, 16, 16],
                "Qubits": [10, 12, 12],
                "L2 error": ["3.7 %", "2.6 %", "3.4 %"],
                "Wall time": ["11 s", "56 s", "57 s"],
            }),
            hide_index=True,
            use_container_width=True,
        )
        st.success("All three Re < 5 % L2 threshold.")

    st.divider()
    st.subheader("Repository and reports")
    st.markdown("""
- **GitHub:** [quantumadventurer11/airbus-quantum-tgv-solver](https://github.com/quantumadventurer11/airbus-quantum-tgv-solver)
- **Technical report:** `reports/final_report.md`
- **Notebook:** `notebooks/demo_taylor_green_vortex.ipynb`
- **Benchmark data:** `results/benchmarks/benchmark_results.csv`
    """)

# ══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Analytical TGV
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Analytical Taylor-Green Vortex")
    st.markdown(
        "The exact closed-form solution for the TGV vorticity pattern. "
        "Adjust Re and time — the vortex decays exponentially due to viscosity."
    )

    ctrl_col, plot_col = st.columns([1, 3])
    with ctrl_col:
        re_an = st.select_slider("Reynolds number", options=[10, 100, 500], value=10,
                                  key="re_analytical")
        t_an = st.slider("Time t (s)", 0.0, 1.0, 0.0, step=0.02, key="t_analytical")

    nu_an = V0 * L_PHYS / re_an
    decay_an = math.exp(-2.0 * nu_an * t_an)

    xs = np.linspace(0, 2 * math.pi, 64, endpoint=False)
    X64, Y64 = np.meshgrid(xs, xs, indexing="ij")
    u_an = V0 * np.sin(X64) * np.cos(Y64) * decay_an
    v_an = -V0 * np.cos(X64) * np.sin(Y64) * decay_an

    with plot_col:
        fig_an, (ax_u, ax_e) = plt.subplots(1, 2, figsize=(11, 4))

        im = ax_u.pcolormesh(X64, Y64, u_an, cmap="RdBu_r", shading="auto",
                              vmin=-V0, vmax=V0)
        plt.colorbar(im, ax=ax_u, label="u (m/s)")
        ax_u.set_title(f"u(x,y)  |  t={t_an:.2f}s  |  amplitude={decay_an:.3f}")
        ax_u.set_xlabel("x (rad)"); ax_u.set_ylabel("y (rad)")
        ax_u.set_aspect("equal")

        ts_plot = np.linspace(0, 1.0, 300)
        ke_plot = (V0**2 / 4.0) * np.exp(-4.0 * nu_an * ts_plot)
        ke_now  = (V0**2 / 4.0) * math.exp(-4.0 * nu_an * t_an)
        ax_e.semilogy(ts_plot, ke_plot, color="#2171b5", linewidth=2,
                      label=f"Re={re_an}")
        ax_e.axvline(t_an, color="#d94801", linestyle="--", alpha=0.8,
                     label=f"t = {t_an:.2f} s")
        ax_e.scatter([t_an], [ke_now], color="#d94801", zorder=5, s=60)
        ax_e.set_xlabel("t (s)"); ax_e.set_ylabel("E(t)")
        ax_e.set_title("Kinetic energy decay  E(t) = (V₀²/4) exp(−4νt)")
        ax_e.legend(fontsize=9); ax_e.grid(True, which="both", alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig_an, use_container_width=True)
        plt.close(fig_an)

    st.caption(
        f"nu = V0 * 2pi / Re = {nu_an:.4f} m²/s  |  "
        f"Amplitude decay at t={t_an:.2f}s: {decay_an*100:.1f}%  |  "
        f"Kinetic energy: {ke_now:.4f}"
    )

# ══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Live QLBM Solver
# ══════════════════════════════════════════════════════════════════════════════

def _exact_vel_lbm(N: int, t: float, Re: float):
    """Exact TGV velocity in lattice coordinates — matches the LBM convention."""
    K = np.arange(N)
    K2d, L2d = np.meshgrid(K, K, indexing="ij")
    nu = V0 * L_PHYS / Re
    d = math.exp(-2.0 * nu * t)
    ux = V0 * np.sin(2 * math.pi * K2d / N) * np.cos(2 * math.pi * L2d / N) * d
    uy = -V0 * np.cos(2 * math.pi * K2d / N) * np.sin(2 * math.pi * L2d / N) * d
    return ux, uy


@st.cache_data(show_spinner=False)
def _run_qlbm(Re: float, t_end: float, u_lbm: float = 0.05):
    N = 8
    omega  = omega_lbm(Re, N, u_lbm)
    n_steps = n_steps_for_time(t_end, N, u_lbm)
    dt_phys = dt_physical(N, u_lbm, L_PHYS)
    t_actual = n_steps * dt_phys

    solver = QLBMSolver(N=N, omega=omega, u_lbm=u_lbm, use_vqc=False)
    f0 = lbm_ic(N, u_lbm)

    t0 = time.perf_counter()
    f  = solver.run(f0, n_steps)
    elapsed = time.perf_counter() - t0

    ux, uy = solver.velocity_physical(f, V0=V0)
    ux_ex, uy_ex = _exact_vel_lbm(N, t_actual, Re)
    l2 = float(np.sqrt(np.mean((ux - ux_ex) ** 2 + (uy - uy_ex) ** 2)))

    return ux, uy, ux_ex, uy_ex, l2, t_actual, elapsed, N, n_steps


with tab3:
    st.header("Live QLBM Solver")
    st.markdown(
        "Run the hybrid QLBM on demand. The quantum streaming circuit executes "
        "on a statevector simulator (N=8, 10 qubits). Results are cached — "
        "repeat runs with the same settings are instant."
    )
    st.info(
        "Re=500 is unavailable for live runs: at N=8 the relaxation rate "
        "omega=1.98 leaves very little stability margin. "
        "Re=500 results at N=16 are available in the Benchmark Scaling tab."
    )

    ctrl_col2, res_col = st.columns([1, 3])
    with ctrl_col2:
        re_live = st.selectbox("Reynolds number", [10, 100], key="re_live")
        t_live  = st.slider("t_end (s)", 0.05, 0.25, 0.10, step=0.05, key="t_live")
        run_btn = st.button("Run Solver", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner(
            f"Running QLBM — Re={re_live}, t_end={t_live:.2f}s, N=8 (10 qubits)..."
        ):
            ux, uy, ux_ex, uy_ex, l2, t_actual, elapsed, N, n_steps = _run_qlbm(
                float(re_live), float(t_live)
            )

        with res_col:
            m1, m2, m3 = st.columns(3)
            m1.metric("L2 velocity error", f"{l2 / V0 * 100:.1f} %")
            m2.metric("Wall time", f"{elapsed:.2f} s")
            m3.metric("QLBM steps", n_steps)

            xs8 = np.arange(N) / N * 2 * math.pi
            X8, Y8 = np.meshgrid(xs8, xs8, indexing="ij")

            fig_live, axes_live = plt.subplots(1, 3, figsize=(13, 4))

            for ax, field, title in zip(
                axes_live,
                [ux, ux_ex, np.sqrt((ux - ux_ex) ** 2 + (uy - uy_ex) ** 2)],
                [f"QLBM  u  (t={t_actual:.3f}s)",
                 f"Exact  u  (t={t_actual:.3f}s)",
                 f"|error|   L2={l2/V0*100:.1f}%"],
            ):
                cmap = "RdBu_r" if "error" not in title else "hot_r"
                vmin = -V0 if "error" not in title else 0
                vmax = V0  if "error" not in title else None
                im = ax.pcolormesh(X8, Y8, field, cmap=cmap, shading="auto",
                                   vmin=vmin, vmax=vmax)
                plt.colorbar(im, ax=ax)
                ax.set_title(title)
                ax.set_xlabel("x (rad)"); ax.set_ylabel("y (rad)")
                ax.set_aspect("equal")

            plt.tight_layout()
            st.pyplot(fig_live, use_container_width=True)
            plt.close(fig_live)
    else:
        with res_col:
            st.markdown("*Select parameters and click **Run Solver** to execute the QLBM.*")

# ══════════════════════════════════════════════════════════════════════════════
# Tab 4 — Benchmark Scaling
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Benchmark Scaling")
    st.markdown(
        "Pre-computed results from `python scripts/sweep.py --full`. "
        "Hover over data points to see exact values."
    )

    if not os.path.isfile(RESULTS_CSV):
        st.warning(
            "benchmark_results.csv not found. "
            "Run `python scripts/sweep.py --full` to generate it."
        )
    else:
        df_bm = pd.read_csv(RESULTS_CSV)

        def _scatter(x, y, text, title, xlab, ylab, log_x=True, log_y=False,
                     color="#2171b5", hline=None):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x, y=y, mode="lines+markers+text",
                text=text, textposition="top center",
                marker=dict(size=11, color=color),
                line=dict(width=2.5, color=color),
                hovertemplate=f"{xlab}=%{{x}}<br>{ylab}=%{{y}}<extra></extra>",
            ))
            if hline is not None:
                fig.add_hline(y=hline, line_dash="dash", line_color="gray",
                              annotation_text=f"{hline}% threshold",
                              annotation_position="bottom right")
            fig.update_layout(
                title=title,
                xaxis_title=xlab, yaxis_title=ylab,
                xaxis_type="log" if log_x else "linear",
                yaxis_type="log" if log_y else "linear",
                template="plotly_white",
                margin=dict(t=50, b=40),
            )
            return fig

        c1, c2, c3 = st.columns(3)
        c1.plotly_chart(
            _scatter(
                df_bm["Re"], df_bm["wall_time_s"],
                [f"{v:.1f}s" for v in df_bm["wall_time_s"]],
                "Time-to-Solution vs Re",
                "Reynolds number Re", "Wall-clock time (s)",
                log_y=True, color="#2171b5",
            ),
            use_container_width=True,
        )
        c2.plotly_chart(
            _scatter(
                df_bm["Re"], df_bm["n_qubits"],
                [str(v) for v in df_bm["n_qubits"]],
                "Qubit Count vs Re",
                "Reynolds number Re", "Qubit count",
                color="#6a3d9a",
            ),
            use_container_width=True,
        )
        c3.plotly_chart(
            _scatter(
                df_bm["Re"], df_bm["l2_error"] * 100,
                [f"{v*100:.1f}%" for v in df_bm["l2_error"]],
                "L2 Velocity Error vs Re",
                "Reynolds number Re", "L2 error (%)",
                color="#d94801", hline=5.0,
            ),
            use_container_width=True,
        )

        st.subheader("Raw benchmark data")
        st.dataframe(
            df_bm[["Re", "N_qlbm", "n_qubits", "n_steps",
                   "wall_time_s", "l2_error", "memory_mb", "timed_out"]].rename(
                columns={"N_qlbm": "N", "wall_time_s": "wall time (s)",
                         "l2_error": "L2 error", "memory_mb": "memory (MB)"}
            ),
            hide_index=True,
            use_container_width=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# Tab 5 — Quantum Circuit
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("Quantum Streaming Circuit")
    st.markdown("""
The D2Q9 streaming step is implemented as an **exact unitary quantum circuit** —
the only step that genuinely runs on quantum hardware.

**Qubit layout** for an N×N grid (4 + 2·log₂N qubits total):

| Register | Qubits | Purpose |
|----------|--------|---------|
| Direction \\|i⟩ | 4 | D2Q9 velocity direction (i = 0..8) |
| x-position \\|x⟩ | log₂N | Lattice x-index (0..N-1) |
| y-position \\|y⟩ | log₂N | Lattice y-index (0..N-1) |

**Operation:** \\|i⟩\\|x⟩\\|y⟩ → \\|i⟩\\|x + cxᵢ mod N⟩\\|y + cyᵢ mod N⟩

Implemented with direction-controlled cyclic increments (multi-controlled NOT gates).

| N | Qubits | Memory (statevector) |
|---|--------|---------------------|
| 4 | 8 | 4 KB |
| 8 | 10 | 16 KB |
| 16 | 12 | 64 KB |
| 32 | 14 | 256 KB |
    """)

    with st.spinner("Building N=4 streaming circuit (8 qubits)..."):
        circ4 = build_streaming_circuit(4)

    c1, c2, c3 = st.columns(3)
    c1.metric("Qubits", circ4.num_qubits)
    c2.metric("Gate count", circ4.size())
    c3.metric("Circuit depth", circ4.depth())

    st.subheader("Circuit diagram (N=4, 8 qubits)")
    try:
        fig_circ = circ4.draw("mpl", fold=60, style="iqp")
        st.pyplot(fig_circ, use_container_width=True)
        plt.close(fig_circ)
    except Exception:
        st.code(str(circ4.draw("text")), language=None)
        st.caption("Install pylatexenc for the graphical circuit diagram: pip install pylatexenc")

    st.caption(
        "Circuit shown for N=4 (smallest tractable grid). "
        "Benchmark runs use N=8 (10 qubits) and N=16 (12 qubits)."
    )
