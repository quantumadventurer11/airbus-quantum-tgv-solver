"""
Layer 2 — Pseudo-spectral FFT solver for the 2D Convecting Taylor-Green Vortex.

Method
------
Vorticity-streamfunction formulation:
  ∂ω/∂t + (U_c + u')·∂ω/∂x + (V_c + v')·∂ω/∂y = ν∇²ω

where u', v' are the periodic velocity fluctuations derived from the streamfunction:
  ψ̂(k) = -ω̂(k) / |k|²,  u' = ∂ψ/∂y,  v' = -∂ψ/∂x

In Fourier space:
  û'(k) = i·ky·ψ̂(k),  v̂'(k) = -i·kx·ψ̂(k)

Nonlinear term evaluated in physical space (pseudo-spectral); 2/3 dealiasing
removes aliased modes before and after each nonlinear product.  Viscous term
applied spectrally (exact within each RK4 stage).

Time integration: classical 4th-order Runge-Kutta (RK4).

Wavenumbers for the domain [0, 2πL)^2 with N points per side:
  kx = fftfreq(N) * N / L    (spacing 1/L)

References
----------
Canuto et al. 1988, Spectral Methods in Fluid Dynamics — dealiasing §3.2
Challenge statement §5 for exact solution used as validation target
"""

from __future__ import annotations

import numpy as np


class SpectralSolver:
    """
    Pseudo-spectral solver for 2D periodic incompressible NS.

    Parameters
    ----------
    N   : int   — grid points per side (power of 2 recommended)
    L   : float — vortex length scale (domain size = 2πL)
    nu  : float — kinematic viscosity ν = V0·L/Re
    Uc  : float — background convection velocity, x (default 1.0)
    Vc  : float — background convection velocity, y (default 0.0)
    """

    def __init__(
        self,
        N: int,
        L: float,
        nu: float,
        Uc: float = 1.0,
        Vc: float = 0.0,
    ) -> None:
        self.N = N
        self.L = L
        self.nu = nu
        self.Uc = Uc
        self.Vc = Vc

        # ── Wavenumber arrays for rfft2 output shape (N, N//2+1) ──────────────
        # Domain [0, 2πL): spacing dk = 1/L, so km = m/L
        kx_1d = np.fft.fftfreq(N) * N / L        # shape (N,)
        ky_1d = np.fft.rfftfreq(N) * N / L       # shape (N//2+1,)
        self.KX, self.KY = np.meshgrid(kx_1d, ky_1d, indexing="ij")
        self.K2 = self.KX**2 + self.KY**2        # |k|², shape (N, N//2+1)

        # Poisson: ψ̂ = -ω̂/|k|²; avoid division by zero at k=(0,0)
        self._K2_inv = np.where(self.K2 == 0, 0.0, 1.0 / np.where(self.K2 == 0, 1.0, self.K2))

        # ── 2/3 dealiasing mask ───────────────────────────────────────────────
        # Keep only modes |m_x| ≤ N//3 and |m_y| ≤ N//3
        # (fftfreq indices: mode index = freq * N)
        kx_idx = np.fft.fftfreq(N) * N     # integer mode indices [0,1,...,N/2,-N/2+1,...,-1]
        ky_idx = np.fft.rfftfreq(N) * N    # [0,1,...,N/2]
        KX_idx, KY_idx = np.meshgrid(kx_idx, ky_idx, indexing="ij")
        cutoff = N // 3
        self._dealias = (np.abs(KX_idx) <= cutoff) & (KY_idx <= cutoff)

        # ── Safe time step ────────────────────────────────────────────────────
        self.dt = self._safe_dt()

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        u0: np.ndarray,
        v0: np.ndarray,
        t_end: float,
        dt: float | None = None,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """
        Integrate from t = 0 to t = t_end starting from velocity fields u0, v0.

        Parameters
        ----------
        u0, v0 : N×N arrays — initial velocity fields (physical space)
        t_end  : float — target end time
        dt     : float | None — time step; None → use computed safe dt

        Returns
        -------
        (t, u, v) : final time and velocity fields on the N×N physical grid
        """
        if dt is None:
            dt = self.dt

        omega0 = self._vorticity(u0, v0)
        omega_hat = np.fft.rfft2(omega0)

        t = 0.0
        while t < t_end - 1e-12:
            dt_step = min(dt, t_end - t)
            omega_hat = self._rk4(omega_hat, dt_step)
            t += dt_step

        u, v = self._vel_from_omega_hat(omega_hat)
        return t, u, v

    def run_with_snapshots(
        self,
        u0: np.ndarray,
        v0: np.ndarray,
        t_end: float,
        n_snapshots: int = 10,
        dt: float | None = None,
    ) -> list[tuple[float, np.ndarray, np.ndarray]]:
        """
        Integrate and return (t, u, v) snapshots at evenly-spaced times.

        Includes t=0 and t=t_end.
        """
        if dt is None:
            dt = self.dt

        ts = np.linspace(0.0, t_end, n_snapshots)
        snapshots: list[tuple[float, np.ndarray, np.ndarray]] = []

        omega_hat = np.fft.rfft2(self._vorticity(u0, v0))
        t = 0.0
        snap_idx = 0

        while snap_idx < n_snapshots:
            # Record snapshot if we've just passed or reached the target time
            if t >= ts[snap_idx] - 1e-12:
                u, v = self._vel_from_omega_hat(omega_hat)
                snapshots.append((float(t), u, v))
                snap_idx += 1
                if snap_idx >= n_snapshots:
                    break

            t_next = ts[snap_idx] if snap_idx < n_snapshots else t_end
            dt_step = min(dt, t_next - t)
            if dt_step < 1e-15:
                snap_idx += 1
                continue
            omega_hat = self._rk4(omega_hat, dt_step)
            t += dt_step

        return snapshots

    # ── Internal methods ──────────────────────────────────────────────────────

    def _safe_dt(self, cfl: float = 0.4) -> float:
        """Compute a safe time step from CFL and viscous stability constraints."""
        dx = 2.0 * np.pi * self.L / self.N
        max_speed = abs(self.Uc) + abs(self.Vc) + 1.0   # +1 for fluctuation
        dt_cfl = cfl * dx / max_speed

        if self.nu > 0:
            # Highest retained wavenumber after 2/3 dealiasing
            k_max = (self.N // 3) / self.L
            k2_max = 2.0 * k_max**2    # diagonal: kx=ky=k_max
            # RK4 diffusion stability: dt ≤ 2.79 / (ν·k²_max)
            dt_visc = cfl * 2.79 / (self.nu * k2_max)
        else:
            dt_visc = dt_cfl

        return min(dt_cfl, dt_visc)

    def _rhs(self, omega_hat: np.ndarray) -> np.ndarray:
        """
        Right-hand side: d(ω̂)/dt = -N̂(ω) - ν|k|²·ω̂

        where N(ω) = (U_c + u')·∂ω/∂x + (V_c + v')·∂ω/∂y
        """
        N = self.N
        # Apply dealiasing
        oh = omega_hat * self._dealias

        # Streamfunction: -∇²ψ = ω  →  ψ̂ = ω̂ / |k|²
        psi_hat =  oh * self._K2_inv
        u_hat   =  1j * self.KY * psi_hat   # u' = ∂ψ/∂y
        v_hat   = -1j * self.KX * psi_hat   # v' = −∂ψ/∂x

        # Vorticity gradients (Fourier space)
        domega_dx_hat = 1j * self.KX * oh
        domega_dy_hat = 1j * self.KY * oh

        # Transform to physical space
        u_p = np.fft.irfft2(u_hat, s=(N, N))
        v_p = np.fft.irfft2(v_hat, s=(N, N))
        domega_dx = np.fft.irfft2(domega_dx_hat, s=(N, N))
        domega_dy = np.fft.irfft2(domega_dy_hat, s=(N, N))

        # Nonlinear advection in physical space
        nl = (self.Uc + u_p) * domega_dx + (self.Vc + v_p) * domega_dy

        # Back to Fourier; apply dealiasing to remove aliased modes
        nl_hat = np.fft.rfft2(nl) * self._dealias

        # Viscous dissipation (spectrally exact)
        visc_hat = -self.nu * self.K2 * omega_hat

        return -nl_hat + visc_hat

    def _rk4(self, omega_hat: np.ndarray, dt: float) -> np.ndarray:
        """Single 4th-order Runge-Kutta step."""
        k1 = self._rhs(omega_hat)
        k2 = self._rhs(omega_hat + 0.5 * dt * k1)
        k3 = self._rhs(omega_hat + 0.5 * dt * k2)
        k4 = self._rhs(omega_hat + dt * k3)
        return omega_hat + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _vorticity(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """ω = ∂v/∂x − ∂u/∂y via spectral differentiation."""
        u_hat = np.fft.rfft2(u)
        v_hat = np.fft.rfft2(v)
        omega_hat = 1j * self.KX * v_hat - 1j * self.KY * u_hat
        return np.fft.irfft2(omega_hat, s=(self.N, self.N))

    def _vel_from_omega_hat(
        self, omega_hat: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Recover (u, v) = (U_c + u', V_c + v') from ω̂."""
        N = self.N
        oh = omega_hat * self._dealias
        psi_hat = oh * self._K2_inv          # ψ̂ = ω̂ / |k|²
        u_phys = np.fft.irfft2( 1j * self.KY * psi_hat, s=(N, N)) + self.Uc
        v_phys = np.fft.irfft2(-1j * self.KX * psi_hat, s=(N, N)) + self.Vc
        return u_phys, v_phys


# ── Convenience function ──────────────────────────────────────────────────────

def solve_tgv(
    *,
    N: int,
    Re: float,
    t_end: float,
    L: float = 2.0 * np.pi,
    V0: float = 1.0,
    Uc: float = 1.0,
    Vc: float = 0.0,
    dt: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience wrapper: set up and run the spectral TGV solver.

    Returns
    -------
    (x, y, u, v) — grid coordinates and final velocity fields
    """
    from tgv.analytical import make_grid, velocity_ic

    nu = V0 * L / Re
    x, y = make_grid(N, L)
    u0, v0 = velocity_ic(x, y, L=L, V0=V0, Uc=Uc, Vc=Vc)

    solver = SpectralSolver(N, L, nu, Uc=Uc, Vc=Vc)
    _, u, v = solver.run(u0, v0, t_end, dt=dt)
    return x, y, u, v
