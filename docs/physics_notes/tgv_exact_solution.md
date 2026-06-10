# TGV Exact Solution — Plain-Language Notes

Reference: Airbus Challenge Statement §5.2–5.3.

---

## What Is the Taylor-Green Vortex?

Imagine a smooth, sinusoidal swirling pattern (a "vortex") sitting inside a
channel of fluid that flows in the x-direction at a constant speed Uc.  The
vortex is initialised at t=0 and then:

1. **Advected** — the entire vortex is swept bodily in the x-direction at
   speed Uc.  Think of ink dropped into a river: the ink pattern moves with
   the flow.

2. **Diffused** — viscosity (`ν`) slowly damps the vortex intensity.  The
   faster the fluid shears (higher amplitude), the faster viscosity drains
   energy.  At high Reynolds number (Re = V0·L/ν → small ν), diffusion is
   slow and the vortex persists a long time.  At low Re, it decays quickly.

Because the governing equations (2D incompressible Navier-Stokes) are LINEAR
around this particular initial condition, there is a perfect closed-form
solution that combines both effects into a simple formula.

---

## The Exact Solution (copied from §5.3)

```
u(x,y,t) = Uc + V0 · sin((x − Uc·t)/L) · cos((y − Vc·t)/L) · e^(−2νt/L²)
v(x,y,t) = Vc − V0 · cos((x − Uc·t)/L) · sin((y − Vc·t)/L) · e^(−2νt/L²)
```

**Breaking it down:**

| Term | Meaning |
|------|---------|
| `Uc` | The background flow speed. The velocity never drops below Uc. |
| `V0 · sin(...) · cos(...)` | The vortex pattern. At t=0 it fills the domain. |
| `(x − Uc·t)/L` | The x-coordinate in the frame MOVING WITH the vortex. As t increases, the argument shifts left, so the pattern translates. |
| `e^(−2νt/L²)` | The decay envelope. This factor starts at 1 (t=0) and decays toward 0. The higher ν (lower Re), the faster it decays. |

---

## Domain Convention

The formula `sin(x/L)` completes one full period when x sweeps from 0 to 2πL.
So the computational domain is **[0, 2πL] × [0, 2πL]** (not [0, L]).

With L = 2π: domain = [0, 4π²] ≈ [0, 39.5] in physical units.

See `DECISIONS.md` D-001 for the full justification.

---

## Kinetic Energy Decay

The spatially averaged fluctuation kinetic energy (subtracting the background
flow Uc, Vc) decays as:

```
E(t) = (V0² L²/4) · exp(−4νt/L²)
```

The decay rate is **−4ν/L²** (double the velocity amplitude decay rate of
−2ν/L²) because energy is proportional to velocity squared.

Note: `E` as defined above has dimensions of velocity² × length², which is the
domain-integrated energy normalised by (2π)². To get the true domain-averaged
energy density, divide by L².

---

## Reynolds Number and Viscosity

```
Re = V0 · L / ν  →  ν = V0 · L / Re
```

- Re = 10:  ν ≈ 0.628  (rapid decay, easy for any solver)
- Re = 100: ν ≈ 0.063  (slower decay, finer grid needed)
- Re = 400: ν ≈ 0.016  (very slow decay, fine vortex structures)

Grid resolution rule of thumb for 2D NS: `N ≥ C · Re^(3/4)` where C ≈ 4
ensures the Kolmogorov scale is resolved.

---

## Why Is This a Good Benchmark?

1. **Known exact solution** — the error is measured against a formula, not
   another simulation.  No reference-solution bias.

2. **Non-trivial physics** — the vortex decays AND advects.  A solver must get
   BOTH the advection speed AND the decay rate right.

3. **Tunable difficulty** — increasing Re increases the demand on spatial
   resolution and numerical stability, allowing a clear scaling study.

4. **Non-unitary** — viscous decay makes the evolution non-unitary
   (irreversible), which is the key challenge for quantum algorithms.
   Quantum mechanics is inherently unitary, so this benchmark tests whether
   you can "fake" dissipation on quantum hardware.
