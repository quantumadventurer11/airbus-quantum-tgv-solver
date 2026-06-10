"""
Aerodynamic visualizations using classical D2Q9 Lattice Boltzmann simulation.

Simulates:
  1. Flow past a NACA 0012 airfoil at Re=300, AoA=8°  (turbulent wake, separation)
  2. Flow past a cylinder at Re=150  (Kármán vortex street)

These are exactly the kind of simulations the quantum QLBM solver targets.
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

# ── D2Q9 lattice constants ────────────────────────────────────────────────────
CX = np.array([0,  1,  0, -1,  0,  1, -1, -1,  1], dtype=float)
CY = np.array([0,  0,  1,  0, -1,  1,  1, -1, -1], dtype=float)
W  = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])   # opposite direction indices


def feq(rho, ux, uy):
    """BGK equilibrium. rho,ux,uy: any shape that broadcasts."""
    cu = CX[:, None, None] * ux[None] + CY[:, None, None] * uy[None]
    u2 = ux**2 + uy**2
    return W[:, None, None] * rho[None] * (1 + 3*cu + 4.5*cu**2 - 1.5*u2[None])


def stream(f):
    """Streaming: shift each population by its velocity direction."""
    return np.stack([
        np.roll(f[i], (int(CX[i]), int(CY[i])), axis=(0, 1))
        for i in range(9)
    ])


def moments(f):
    rho = f.sum(axis=0)
    ux  = (CX[:, None, None] * f).sum(axis=0) / rho
    uy  = (CY[:, None, None] * f).sum(axis=0) / rho
    return rho, ux, uy


def apply_solid_bc(f, solid):
    """Half-way bounce-back at solid nodes."""
    for i in range(9):
        f[i, solid] = f[OPP[i], solid]
    return f


# ── Geometry ──────────────────────────────────────────────────────────────────

def naca_mask(nx, ny, chord, x0, y0, aoa_deg, series="0012"):
    """Boolean mask (nx,ny) True inside NACA 4-digit airfoil."""
    m = int(series[0]) / 100.0
    p = int(series[1]) / 10.0
    t = int(series[2:]) / 100.0
    aoa = np.radians(aoa_deg)

    II, JJ = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    dx = II - x0
    dy = JJ - y0
    xr =  dx * np.cos(aoa) + dy * np.sin(aoa)
    yr = -dx * np.sin(aoa) + dy * np.cos(aoa)
    xn = xr / chord

    yt = 5*t*(0.2969*np.sqrt(np.clip(xn, 0, 1))
              - 0.1260*xn
              - 0.3516*xn**2
              + 0.2843*xn**3
              - 0.1015*xn**4)

    if p > 0:
        yc = np.where(xn < p,
                      m/p**2 * (2*p*xn - xn**2),
                      m/(1-p)**2 * (1 - 2*p + 2*p*xn - xn**2))
    else:
        yc = np.zeros_like(xn)

    return (xn >= 0) & (xn <= 1) & (yr >= (yc - yt)*chord) & (yr <= (yc + yt)*chord)


def cylinder_mask(nx, ny, cx, cy, r):
    II = np.arange(nx)[:, None]
    JJ = np.arange(ny)[None, :]
    return (II - cx)**2 + (JJ - cy)**2 <= r**2


# ── LBM solver ────────────────────────────────────────────────────────────────

def run_lbm(nx, ny, solid, u_inlet, omega, n_steps, save_every=500, label=""):
    """D2Q9 LBM. Returns list of (step, ux, uy, rho) snapshots."""
    rho = np.ones((nx, ny))
    ux  = np.full((nx, ny), u_inlet)
    uy  = np.zeros((nx, ny))
    f   = feq(rho, ux, uy)

    # Pre-compute equilibrium for the inlet strip (constant)
    rho_in = np.ones((1, ny))
    ux_in  = np.full((1, ny), u_inlet)
    uy_in  = np.zeros((1, ny))
    f_in   = feq(rho_in, ux_in, uy_in)[:, 0, :]   # shape (9, ny)

    snapshots = []
    t0 = time.time()

    for step in range(n_steps):
        # 1. Collision (BGK)
        rho, ux, uy = moments(f)
        f += omega * (feq(rho, ux, uy) - f)

        # 2. Solid bounce-back (pre-streaming)
        f = apply_solid_bc(f, solid)

        # 3. Streaming
        f = stream(f)

        # 4. Solid bounce-back (post-streaming)
        f = apply_solid_bc(f, solid)

        # 5. Inlet BC (left wall: uniform velocity)
        f[:, 0, :] = f_in

        # 6. Outlet BC (right wall: zero-gradient)
        f[:, -1, :] = f[:, -2, :]

        # 7. Top/bottom walls: no-slip bounce-back
        for i, j in [(2, 4), (5, 7), (6, 8)]:
            f[i, :, -1] = f[j, :, -1]
            f[j, :, 0]  = f[i, :, 0]

        if step % save_every == 0:
            rho_, ux_, uy_ = moments(f)
            ux_[solid] = 0;  uy_[solid] = 0
            snapshots.append((step, ux_.copy(), uy_.copy(), rho_.copy()))
            elapsed = time.time() - t0
            rate = (step + 1) / elapsed if elapsed > 0 else 0
            eta = (n_steps - step - 1) / rate if rate > 0 else 0
            speed = np.sqrt(ux_**2 + uy_**2)
            print(f"  {label} step {step:5d}/{n_steps}  "
                  f"max|u|={speed[~solid].max():.4f}  "
                  f"{rate:.0f} steps/s  ETA {eta:.0f}s")

    return snapshots


# ── Plotting ──────────────────────────────────────────────────────────────────

def compute_fields(ux, uy, rho, solid):
    """Vorticity, speed, pressure — NaN at solid."""
    dvdx = (np.roll(uy, -1, axis=0) - np.roll(uy,  1, axis=0)) / 2
    dudy = (np.roll(ux, -1, axis=1) - np.roll(ux,  1, axis=1)) / 2
    vort = dvdx - dudy
    speed = np.sqrt(ux**2 + uy**2)
    pres  = rho / 3.0
    for arr in (vort, speed, pres):
        arr[solid] = np.nan
    return vort, speed, pres


def plot_snapshot(ux, uy, rho, solid, title, save_path):
    """3-panel: vorticity | speed | pressure with streamlines on vorticity."""
    vort, speed, pres = compute_fields(ux.copy(), uy.copy(), rho.copy(), solid)
    nx, ny = ux.shape
    x = np.arange(nx)
    y = np.arange(ny)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150,
                              facecolor="#0d1117")
    fig.suptitle(title, fontsize=12, fontweight="bold", color="white")

    configs = [
        (axes[0], vort,  "Vorticity  ω = ∂v/∂x − ∂u/∂y", "RdBu_r",  True),
        (axes[1], speed, "Velocity magnitude  |u|",          "plasma",  False),
        (axes[2], pres,  "Pressure  p (lattice units)",       "coolwarm", True),
    ]

    for ax, data, label, cmap, symmetric in configs:
        ax.set_facecolor("#0d1117")
        vmax = np.nanpercentile(np.abs(data), 99)
        vmin = -vmax if symmetric else 0
        im = ax.imshow(data.T, origin="lower", cmap=cmap,
                       vmin=vmin, vmax=vmax,
                       extent=[0, nx, 0, ny], aspect="equal")
        cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.ax.tick_params(colors="white")
        ax.set_title(label, fontsize=9, color="white")
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#333")

        # Streamlines on vorticity panel
        if symmetric:
            ux_m = np.ma.array(ux, mask=solid)
            uy_m = np.ma.array(uy, mask=solid)
            try:
                ax.streamplot(x, y, ux_m.T, uy_m.T, density=1.0,
                              linewidth=0.5, color="white", arrowsize=0.6)
            except Exception:
                pass

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def make_gif(snapshots, solid, title, save_path, field="vorticity", fps=6):
    """Animated GIF of vorticity or speed over time."""
    fig, ax = plt.subplots(figsize=(10, 4), dpi=120, facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")

    def get_field(ux, uy, rho):
        if field == "vorticity":
            dvdx = (np.roll(uy, -1, axis=0) - np.roll(uy, 1, axis=0)) / 2
            dudy = (np.roll(ux, -1, axis=1) - np.roll(ux, 1, axis=1)) / 2
            d = dvdx - dudy
        else:
            d = np.sqrt(ux**2 + uy**2)
        d[solid] = np.nan
        return d

    frames = [get_field(ux, uy, rho) for _, ux, uy, rho in snapshots]
    vmax = np.nanpercentile(np.abs(np.concatenate([f.ravel() for f in frames])), 99)
    vmin = -vmax if field == "vorticity" else 0
    cmap = "RdBu_r" if field == "vorticity" else "plasma"

    nx, ny = frames[0].shape
    im = ax.imshow(frames[0].T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=[0, nx, 0, ny], aspect="equal", animated=True)
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(colors="white")
    cbar.set_label("ω [vorticity]" if field == "vorticity" else "|u|",
                   color="white")

    ttl = ax.set_title("", color="white", fontsize=10, fontweight="bold")

    def update(i):
        im.set_array(frames[i].T)
        ttl.set_text(f"{title}  —  step {snapshots[i][0]:,}")
        return im, ttl

    ani = animation.FuncAnimation(fig, update, frames=len(snapshots),
                                  interval=1000 // fps, blit=True)
    writer = animation.PillowWriter(fps=fps)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    ani.save(save_path, writer=writer)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_cp(ux, uy, rho, solid, chord, x0, y0, u_inf, save_path):
    """Pressure coefficient along airfoil chord (upper vs lower surface)."""
    p     = rho / 3.0
    p_inf = 1.0 / 3.0
    q_inf = 0.5 * u_inf**2

    n = 40
    xc = np.linspace(0, chord, n)
    xn = xc / chord
    t  = 0.12
    yt = 5*t*(0.2969*np.sqrt(xn) - 0.1260*xn - 0.3516*xn**2
              + 0.2843*xn**3 - 0.1015*xn**4)

    cp_u, cp_l = [], []
    for xi, yt_i in zip(xc, yt):
        xi_i = int(np.clip(x0 + xi, 0, rho.shape[0]-1))
        yu_i = int(np.clip(y0 + yt_i * chord + 2, 0, rho.shape[1]-1))
        yl_i = int(np.clip(y0 - yt_i * chord - 2, 0, rho.shape[1]-1))
        cp_u.append((p[xi_i, yu_i] - p_inf) / q_inf)
        cp_l.append((p[xi_i, yl_i] - p_inf) / q_inf)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    xc_n = np.linspace(0, 1, n)
    ax.plot(xc_n, cp_u, "b-o", ms=4, label="Upper surface")
    ax.plot(xc_n, cp_l, "r-o", ms=4, label="Lower surface")
    ax.fill_between(xc_n, cp_u, cp_l, alpha=0.08, color="navy")
    ax.invert_yaxis()
    ax.set_xlabel("x/c  (normalised chord)", fontsize=11)
    ax.set_ylabel("Cp  (pressure coefficient)", fontsize=11)
    ax.set_title("Pressure coefficient — NACA 0012  (Re=300, AoA=8°)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    FIG = "results/figures"
    os.makedirs(FIG, exist_ok=True)
    t_total = time.time()

    # ── Case 1: NACA 0012 airfoil, Re=300, AoA=8° ─────────────────────────
    print("\n=== Case 1: NACA 0012 airfoil, Re=300, AoA=8° ===")
    NX, NY = 200, 80
    CHORD  = 50
    X0     = 40          # leading edge x (grid cells)
    Y0     = NY // 2     # centred vertically
    AOA    = 8.0
    U_INF  = 0.08
    RE     = 300.0
    nu_lbm = U_INF * CHORD / RE
    tau    = 3.0 * nu_lbm + 0.5
    omega  = 1.0 / tau
    print(f"  Grid: {NX}×{NY}  chord={CHORD}  nu={nu_lbm:.5f}  "
          f"tau={tau:.4f}  omega={omega:.4f}")

    solid_foil = naca_mask(NX, NY, CHORD, X0, Y0, AOA, "0012")
    print(f"  Solid cells: {solid_foil.sum()}")

    snaps_foil = run_lbm(NX, NY, solid_foil, U_INF, omega,
                         n_steps=18_000, save_every=600, label="[NACA]")

    _, ux_f, uy_f, rho_f = snaps_foil[-1]
    plot_snapshot(ux_f, uy_f, rho_f, solid_foil,
                  "LBM — NACA 0012 Airfoil  (Re=300, AoA=8°, classical D2Q9)",
                  f"{FIG}/aero_naca0012_flow.png")

    make_gif(snaps_foil, solid_foil,
             "NACA 0012 — Vorticity (Re=300, AoA=8°)",
             f"{FIG}/aero_naca0012_vorticity.gif",
             field="vorticity", fps=5)

    make_gif(snaps_foil, solid_foil,
             "NACA 0012 — Speed (Re=300, AoA=8°)",
             f"{FIG}/aero_naca0012_speed.gif",
             field="speed", fps=5)

    plot_cp(ux_f, uy_f, rho_f, solid_foil, CHORD, X0, Y0, U_INF,
            f"{FIG}/aero_naca0012_cp.png")

    # ── Case 2: Kármán vortex street (cylinder), Re=150 ───────────────────
    print("\n=== Case 2: Flow past cylinder, Re=150 (Kármán vortex street) ===")
    NX2, NY2 = 250, 90
    CX2, CY2 = 60, NY2 // 2
    R2       = 12
    U2       = 0.06
    RE2      = 150.0
    nu2      = U2 * (2*R2) / RE2
    tau2     = 3.0 * nu2 + 0.5
    omega2   = 1.0 / tau2
    print(f"  Grid: {NX2}×{NY2}  D={2*R2}  nu={nu2:.5f}  "
          f"tau={tau2:.4f}  omega={omega2:.4f}")

    solid_cyl = cylinder_mask(NX2, NY2, CX2, CY2, R2)
    print(f"  Solid cells: {solid_cyl.sum()}")

    snaps_cyl = run_lbm(NX2, NY2, solid_cyl, U2, omega2,
                        n_steps=22_000, save_every=600, label="[CYL]")

    _, ux_c, uy_c, rho_c = snaps_cyl[-1]
    plot_snapshot(ux_c, uy_c, rho_c, solid_cyl,
                  "LBM — Kármán Vortex Street, Cylinder  (Re=150, classical D2Q9)",
                  f"{FIG}/aero_karman_vortex.png")

    make_gif(snaps_cyl, solid_cyl,
             "Kármán Vortex Street (Re=150)",
             f"{FIG}/aero_karman_animation.gif",
             field="vorticity", fps=6)

    print(f"\n=== Done in {time.time()-t_total:.0f}s  ===")
    print(f"Outputs saved to {FIG}/")
    print("  aero_naca0012_flow.png   — 3-panel vorticity/speed/pressure")
    print("  aero_naca0012_vorticity.gif — animated wake development")
    print("  aero_naca0012_speed.gif")
    print("  aero_naca0012_cp.png     — pressure coefficient distribution")
    print("  aero_karman_vortex.png   — Kármán vortex street snapshot")
    print("  aero_karman_animation.gif")
