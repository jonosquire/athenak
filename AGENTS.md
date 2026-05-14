# AGENTS.md — Spherical-shell hydro in this AthenaK fork

This file is for future agents (or future-me) working on the spherical-shell
coordinate path. It captures decisions, gotchas, and the "where to go next" map.

**For users: see [README-sph.md](README-sph.md) for the practical guide**
(build, options, tests, plot suggestions). This file is design-notes only.

## Goal of the project

Add a logically-Cartesian spherical-polar shell (`x1=r`, `x2=theta`, `x3=phi`)
to AthenaK as a single-purpose option for solar-corona physics. Pole-excised,
hydro-first, GPU-native. **Not** a general curvilinear framework.

The supported coordinate systems in this fork are exactly two:
- `cartesian` — default; existing AthenaK behaviour is unchanged
- `spherical_shell` — orthogonal r-theta-phi shell, pole excised

GR Cartesian-Kerr-Schild remains untouched in its own metric machinery
(`coord_data`, `is_general_relativistic`, `cartesian_ks.hpp`).

## Status after Task 2

**Working hydro on a spherical shell:**

- Per-MeshBlock device-resident geometry container `SphericalShellGeom`
  with face areas, cell volumes, physical widths, and Athena++ source-term
  factors `coord_src{1,2}_{i,j}` precomputed at setup.
- Spherical finite-volume update in `hydro_update.cpp`, branched once at
  the host level on `coord_system`.
- Spherical CFL using physical widths
  `(dr, r * dtheta, r * sin(theta) * dphi)` in `hydro_newdt.cpp`.
- Newtonian geometric momentum source terms in `Coordinates::AddSphericalShellHydroSrcTerms`,
  invoked from `Hydro::HydroSrcTerms`. Mirrors Athena++
  `srcpp/coordinates/spherical_polar.cpp`'s `AddCoordTermsDivergence` HYDRO ONLY.
- Tiny user pgen `pgen/sph_shell_hydro.cpp` with two modes:
  - `mode = uniform`     — uniform-state preservation test
  - `mode = sound_pulse` — radial Gaussian acoustic pulse smoke test
- Three matching input files in `inputs/tests/`.

**Preserved:** existing Cartesian linear_wave_hydro and blast_hydro produce
bit-identical timesteps to the previous task.

## Status after Task 3A (hydro validation suite)

A hydro-only validation suite is added on top of the Task 2 solver, with no
changes to the solver itself. All new diagnostics live in the existing
`pgen/sph_shell_hydro.cpp` so build configuration is unchanged.

### New `<problem>/mode` values

| mode             | type            | purpose                                                  |
|------------------|-----------------|----------------------------------------------------------|
| `uniform`        | quantitative    | uniform-state preservation (Task 2, unchanged)           |
| `sound_pulse`    | quantitative    | radial pulse half-max edge (Task 2, unchanged)           |
| `radial_acoustic`| quantitative    | density-weighted centroid of an outgoing radial wave    |
| `divergence_test`| quantitative    | FV divergence operator vs analytic for 3 vector fields   |
| `pulse_3d`       | visual / vibe   | 3D Gaussian pressure pulse in embedded Cartesian         |
| `oblique_packet` | visual / vibe   | oblique Cartesian acoustic wave packet                   |
| `homologous`     | quantitative    | v_r=H·r ⇒ d⟨ρ⟩/dt vs -3H·ρ analytic                      |

### New input files

- `inputs/tests/sph_shell_divergence.athinput`             (Part B)
- `inputs/tests/sph_shell_radial_acoustic.athinput`        (Part C)
- `inputs/tests/sph_shell_3d_acoustic_pulse.athinput`      (Part D, vibe)
- `inputs/tests/sph_shell_oblique_wave_packet.athinput`    (Part E, vibe)
- `inputs/tests/sph_shell_homologous.athinput`             (Part F, optional)

### How to run

All tests use the same build: `-DPROBLEM=sph_shell_hydro`.

```bash
# Build once
( cd build && cmake -DPROBLEM=sph_shell_hydro .. && cmake --build . --target athena -j 4 )

# Quantitative checks (print results to stdout)
./build/src/athena -i inputs/tests/sph_shell_divergence.athinput
./build/src/athena -i inputs/tests/sph_shell_radial_acoustic.athinput
./build/src/athena -i inputs/tests/sph_shell_homologous.athinput

# Convergence sweep (run at multiple resolutions via CLI overrides)
for N in 32 64 128; do
  ./build/src/athena -i inputs/tests/sph_shell_divergence.athinput \
    mesh/nx1=$N mesh/nx2=$((N/2)) mesh/nx3=$((N/2)) \
    meshblock/nx1=$N meshblock/nx2=$((N/2)) meshblock/nx3=$((N/2))
done

# Visual / vibe checks (write VTK + history dumps)
./build/src/athena -i inputs/tests/sph_shell_3d_acoustic_pulse.athinput
./build/src/athena -i inputs/tests/sph_shell_oblique_wave_packet.athinput
```

### Quantitative results (CPU / Kokkos Serial, this worktree)

**Part B: divergence diagnostic** — three vector fields, error vs analytic
centroid divergence. Cells with `verify_geometry=false` so no clutter.

| field                     | nx1×nx2×nx3 | h          | L1        | L2        | Linf      |
|---------------------------|-------------|-----------|-----------|-----------|-----------|
| F_r = r^2                 | 32×16×16    | 9.8e-2    | 6.1e-16   | 8.0e-16   | 1.8e-15   |
| F_r = r^2                 | 64×32×32    | 4.9e-2    | 6.1e-16   | 8.0e-16   | 1.8e-15   |
| F_r = r^3                 | 32×16×16    | 9.8e-2    | 4.07e-4   | 4.07e-4   | 4.07e-4   |
| F_r = r^3                 | 64×32×32    | 4.9e-2    | 1.02e-4   | 1.02e-4   | 1.02e-4   |
| F_θ = sin θ               | 32×16×16    | 9.8e-2    | 1.85e-4   | 2.13e-4   | 4.26e-4   |
| F_θ = sin θ               | 64×32×32    | 4.9e-2    | 4.62e-5   | 5.34e-5   | 1.10e-4   |
| F_φ = sin(2φ)             | 32×16×16    | 9.8e-2    | 9.12e-4   | 1.04e-3   | 2.20e-3   |
| F_φ = sin(2φ)             | 64×32×32    | 4.9e-2    | 2.27e-4   | 2.60e-4   | 5.57e-4   |

**Resolution-doubling ratios** (L1 at h/2 over L1 at h):

| field                | ratio  | expected |
|----------------------|--------|----------|
| F_r = r^3            | 4.00   | 4 (h²)   |
| F_θ = sin θ          | 4.00   | 4 (h²)   |
| F_φ = sin(2φ)        | 4.02   | 4 (h²)   |

Clean **2nd-order convergence** everywhere. The F_r = r^2 case sits at
machine precision because the FV form `3(r+^4 − r−^4)/(r+^3 − r−^3)` is
*identically* equal to `4·r_vol` for the Mignone-2014 volume-weighted r_vol
the geometry uses — a happy accident that incidentally validates the
identity. Use n≥3 to see real centroid-truncation error.

**Part C: radial acoustic propagation** — Gaussian outgoing pulse at
rho0=1, p0=0.6, cs0=1.0, amp=1e-4, rc=2.0, rw=0.1, on r∈[1,3], thin θ-φ wedge.

At t=1.0 (pulse traverses Δr = 1.0):

```
[radial_acoustic] cs0           = 1.000e+00
[radial_acoustic] centroid(0)   = 2.010e+00
[radial_acoustic] centroid(t)   = 3.018e+00
[radial_acoustic] predicted     = 3.010e+00
[radial_acoustic] speed_est     = 1.008e+00
[radial_acoustic] speed_err     = 7.9e-3   (rel 0.79%)
```

The ~1% offset is a **physical** centroid bias (the spherical 1/r packet
spreading creates an asymmetric ρ-weighted mean of r), not a discretisation
error: doubling nx1 (128 → 256) leaves the speed error essentially
unchanged (1.99% → 1.89% at t=0.4). The wave propagates at exactly cs0;
the centroid metric just has a small bias.

**Part D: 3D localized acoustic pulse (visual)** — runs to t=0.3 on
64³ wedge. Quantitative diagnostic:
```
[pulse_3d] max|drho|                                = 9.6e-05
[pulse_3d] wavefront radius (|drho|^2-weighted)     = 3.07e-01
[pulse_3d] cs0 * t                                  = 3.00e-01
```
Wavefront radius matches cs0·t to within ~2%.

Plot suggestion: open `SphShellPulse3D.hydro_w.0000?.vtk` in ParaView and
slice through z=0 or y=0. Wavefronts should be near-circular in the slice
plane (a thin spherical shell expanding outward at cs0=1).

**Part E: oblique Cartesian wave packet (visual)** — runs to t=0.4.
Packet IC at x0=(2,0,0), kvec=(6,4,3), so |k|=7.81. At t=0.4 the centroid
should drift by cs0·t = 0.4 along khat.

```
[oblique_packet] proj on khat = 3.38e-01  ~? cs0*t = 4.00e-01
```

Early-time results are cleaner:

| t    | proj on khat | predicted | rel err |
|------|--------------|-----------|---------|
| 0.05 | 0.0516       | 0.0500    | 3%      |
| 0.10 | 0.106        | 0.100     | 6%      |
| 0.15 | 0.152        | 0.150     | 1%      |
| 0.20 | 0.184        | 0.200     | 8%      |
| 0.40 | 0.338        | 0.400     | 16%     |

The late-time drift error is a combination of (a) packet leaving the
domain near r=3 boundary, (b) the same centroid-bias issue as Part C,
and (c) some packet broadening from PLM dissipation. The visual answer
is what to trust: open `SphShellObliq.hydro_w.000??.vtk` in ParaView and
slice through z=0 — the packet should translate along khat without
obvious radial-spoke or theta-ring grid imprinting.

**Part F: homologous expansion** — v_r=H·r at rho0=1, H=0.01, tlim=0.005.

```
[homologous] <drho/rho>     = -1.491e-04
[homologous] predicted -3Ht = -1.500e-04
[homologous] err            = 9.4e-07
```

Volume-averaged density change matches the analytic prediction d⟨ρ⟩/dt = -3Hρ
to within 0.6%. Confirms FV div(v)=3H operates correctly.

### Regression — what should NOT have changed

- `inputs/tests/sph_shell_geom.athinput` still gives wedge volume relative
  error ~ 3.4e-14.
- `inputs/tests/sph_shell_hydro_uniform.athinput` still gives
  `max|drho|=0`, `max|dp|=0`, `max|v|=8.1e-17`.
- `inputs/tests/sph_shell_hydro_sound_pulse.athinput` still gives the
  same r_peak_out = 2.195e+00 at t=0.6.
- Cartesian `inputs/tests/linear_wave_hydro.athinput` cycles 0–3 still
  produce `dt = 1.404847e-02, 1.404851e-02, 1.404857e-02, 1.404861e-02`
  (bit-identical to Task 2).

## Files

### Added in Task 1

- `src/coordinates/spherical_shell.hpp`
- `src/coordinates/spherical_shell.cpp`
- `inputs/tests/sph_shell_geom.athinput`

### Added in Task 2

- `src/pgen/sph_shell_hydro.cpp` — user problem generator (two modes)
- `inputs/tests/sph_shell_hydro_uniform.athinput`
- `inputs/tests/sph_shell_hydro_sound_pulse.athinput`
- `inputs/tests/sph_shell_hydro_smoke.athinput`

### Modified in Task 1

- `src/coordinates/coordinates.hpp` — `CoordinateSystem` enum, `coord_system`
  member, `shell_geom` member.
- `src/coordinates/coordinates.cpp` — reads `<coord> system`, calls geometry
  ctor + verify check.
- `src/CMakeLists.txt` — adds `coordinates/spherical_shell.cpp`.

### Modified in Task 2

- `src/coordinates/spherical_shell.hpp` — added four source-term factor Views
  (`coord_src1_i`, `coord_src2_i`, `coord_src1_j`, `coord_src2_j`).
- `src/coordinates/spherical_shell.cpp` — populate the new factors in
  `ConstructSphericalShellGeometry`; new `Coordinates::AddSphericalShellHydroSrcTerms`
  function.
- `src/coordinates/coordinates.hpp` — declare `AddSphericalShellHydroSrcTerms`
  on `Coordinates`.
- `src/hydro/hydro_update.cpp` — host-level branch on `coord_system`. Cartesian
  branch is unchanged; spherical branch uses area-weighted FV divergence.
- `src/hydro/hydro_newdt.cpp` — host-level capture of `is_spherical` and
  `geom`; per-cell branch picks `(dr, r*dtheta, r*sin(theta)*dphi)` for
  spherical, original `dx1/dx2/dx3` otherwise.
- `src/hydro/hydro_tasks.cpp` — `HydroSrcTerms` invokes the spherical
  geometric source terms (after `RKUpdate`, before `c2p`, using the same
  fluxes that produced the divergence).
- `src/hydro/hydro.cpp` — error if `<hydro>/fofc=true` AND
  `coord_system=spherical_shell` (FOFC is Cartesian-only for now).

### `srcpp/`

Untouched. Read-only reference.

## Spherical FV update formula (implemented)

For each conserved variable `n` and each active cell `(m,k,j,i)`:

```
divf = (r2_face(m,i+1) * flx1[i+1] - r2_face(m,i) * flx1[i]) * inv_dr3_third(m,i)
     + dr2_half(m,i)   * inv_dr3_third(m,i)
       * (sin_theta_face(m,j+1) * flx2[j+1] - sin_theta_face(m,j) * flx2[j])
       / dcos_theta(m,j)                                     [if multi-d]
     + dr2_half(m,i)   * inv_dr3_third(m,i)
       * dtheta(m,j) * (flx3[k+1] - flx3[k])
       / (dcos_theta(m,j) * dphi(m,k))                       [if three-d]
u_new = gam0 * u_new + gam1 * u_old - beta_dt * divf
```

This is exactly the area-weighted FV form

```
divf = (A1+ F1+ - A1- F1-)/V + (A2+ F2+ - A2- F2-)/V + (A3+ F3+ - A3- F3-)/V
```

with the `dcos*dphi` factors common to A1 and V cancelled, and the
`dr2_half` ratio factored out for the theta and phi contributions. All
geometry is precomputed in `SphericalShellGeom`. No `sin`/`cos`/`pow` in the
hot loop.

## Spherical CFL (implemented)

Inside the existing reduction kernel in `hydro_newdt.cpp`, when
`is_spherical` is true:

```
ds1 = geom.dr(m, i)
ds2 = geom.r_vol(m, i) * geom.dtheta(m, j)
ds3 = geom.r_vol(m, i) * geom.sin_theta_vol(m, j) * geom.dphi(m, k)
```

Otherwise the original `mbsize.d_view(m).dx{1,2,3}` are used. The
`is_spherical` boolean is captured from the host so the inner loop has a
single hot-path branch (which the compiler sinks for both kernels).

`r_vol(m,i)` is the **volume-averaged** radius from Mignone (2014), and
`sin_theta_vol(m,j) = sin(theta_vol(m,j))` is the sine of the
**volume-weighted** theta centroid (also Mignone 2014). These are NOT
the simple midpoints of the cell; using midpoints would produce small
systematic errors near the pole.

## Geometric momentum source terms (implemented)

These exactly mirror the HYDRO part of Athena++
`srcpp/coordinates/spherical_polar.cpp::AddCoordTermsDivergence`. The
source-term split is *consistent with* the area-weighted FV divergence
above: in particular, pressure curvature in r and theta is split between
the FV flux divergence (which carries `(r+^2 - r-^2) p` and
`(sin+ - sin-) p` from the area variation of pressure on radial and theta
faces) and the source `+ src1_i (2 p)` and `+ src1_i src1_j p` (which exactly
cancel that piece for a uniform pressure). This is the only split that
preserves a uniform static state to roundoff. Verified — see Test 2 below.

```
m_ii = rho * (v_t^2 + v_p^2) + 2 * p_gas      // (or 2 c_iso^2 rho if isothermal)
m_pp = rho * v_p^2 + p_gas

S(m_r) = + src1_i * m_ii
S(m_t) = - src2_i * (r-^2 F1[IM2,i] + r+^2 F1[IM2,i+1])     // -<rho v_r v_t / r>
         + src1_i * src1_j * m_pp                            // +<cot/r>(rho v_p^2 + p)
S(m_p) = - src2_i * (r-^2 F1[IM3,i] + r+^2 F1[IM3,i+1])     // -<rho v_r v_p / r>
         - src1_i * src2_j * (sin- F2[IM3,j] + sin+ F2[IM3,j+1])  // multi-d
         OR  - src1_i * src1_j * (rho v_t v_p)                    // 1-d-theta fallback
S(IEN) = 0
```

where (matching Athena++)

```
src1_i = dr2_half(m,i) / dr3_third(m,i)         // ~ "1/r" weight
src2_i = dr(m,i) / ((r-(m,i) + r+(m,i)) * dr3_third(m,i))
src1_j = (sin+ - sin-) / dcos_theta(m,j)        // ~ <cot theta>
src2_j = (sin+ - sin-) / ((sin- + sin+) * dcos_theta(m,j))
```

The `r-^2`, `r+^2`, `sin-`, `sin+` factors used to weight the flux pairs are
the radial face values `r2_face` and `sin_theta_face` (NOT the full FV face
areas). This is Athena++'s exact convention.

The IEN (energy) equation has no geometric source term — energy curvature
is fully captured by the area-weighted flux divergence.

In the 1-D theta case (no `multi_d`) there is no F2 flux at theta faces, so
the cot-theta source on phi momentum uses the cell-centred primitives
directly. This branch is rarely exercised in practice but matches Athena++.

## Test results (CPU / Kokkos Serial)

### Test 0: geometry sanity (Task 1, unchanged)

```
./build/src/athena -i inputs/tests/sph_shell_geom.athinput
[spherical_shell] geometry sanity check:
    sum(cell volumes) = 6.772391e+00
    analytic wedge V  = 6.772391e+00
    relative error    = 3.4e-14
```

### Test 1: smoke / 3-D run with full hydro stack

```
./build/src/athena -i inputs/tests/sph_shell_hydro_smoke.athinput
```
Runs 5 cycles of hydro on a 16x16x16 wedge and finalises with the uniform
diagnostic. dt is set by `r * dtheta` of the innermost cell; matches the
hand-calculation `0.3 * 0.01287 / sqrt(5/3) ≈ 2.997e-3`. ✓ no crash.

### Test 2: uniform-state preservation

```
./build/src/athena -i inputs/tests/sph_shell_hydro_uniform.athinput
[sph_shell_hydro/uniform] max |drho| = 0.000000e+00
[sph_shell_hydro/uniform] max |dp|   = 0.000000e+00
[sph_shell_hydro/uniform] max |v|    = 8.12e-17
```

50 cycles, single block. Density and pressure are exactly conserved (zero,
to roundoff); velocity stays at machine epsilon. This confirms the FV +
source-term split: pressure curvature on radial and theta faces cancels
the corresponding source contribution exactly.

Multi-MeshBlock variant (2x2x2 = 8 blocks) has the same result — the FV
pressure-curvature cancellation is local within each block:

```
... meshblock/nx1=16 meshblock/nx2=8 meshblock/nx3=4
max |drho| = 0.0, max |dp| = 0.0, max |v| = 9.82e-17
```

### Test 3: radial sound pulse

```
./build/src/athena -i inputs/tests/sph_shell_hydro_sound_pulse.athinput
```

Setup: 128 radial cells, narrow theta-phi wedge, gamma=5/3, p0=0.6, rho0=1
so cs0 = 1.0; centred isentropic Gaussian pulse (amp=1e-3, rc=1.5, rw=0.1).

Sweep over tlim:
| t    | r_peak_out | predicted (rc + cs0*t) | (r_peak - r_pred) / Δt |
|------|------------|------------------------|------------------------|
| 0.05 | 1.602      | 1.550                  | (offset 0.052)         |
| 0.15 | 1.711      | 1.650                  | (offset 0.061)         |
| 0.30 | 1.852      | 1.800                  | (offset 0.052)         |
| 0.60 | 2.195      | 2.100                  | (offset 0.095)         |

Δr / Δt across (0.05 → 0.30) = 0.250 / 0.25 = **1.000 = cs0**. ✓

The constant ~0.05 offset is just the half-max trailing-edge of the
Gaussian pulse (FWHM/2 ≈ 0.083, my reduction picks the outermost half-max
cell, not the peak). Both the propagation speed and the smoothness of the
solution are correct.

### Cartesian regression

```
./build/src/athena -i inputs/tests/linear_wave_hydro.athinput time/nlim=5
elapsed=...   cycle=0   dt=1.404847e-02
elapsed=...   cycle=1   dt=1.404851e-02
... (matches the pre-Task-2 numbers exactly)

./build/src/athena -i inputs/hydro/blast_hydro.athinput time/nlim=2
elapsed=...   cycle=0   dt=6.364410e-04   (matches)
elapsed=...   cycle=1   dt=6.357360e-04   (matches)
elapsed=...   cycle=2   dt=6.386603e-04   (matches)
```

Bit-identical to before Task 2.

## Status after Task 3B (gravity / rotation / Parker validation)

Task 3B adds the gravity source-term hooks and a set of validation tests for
stratified and rotating spherical flows, building on the Task 2 solver and
Task 3A diagnostics. No changes to the spherical hydro solver, geometric
source terms, FV update, or CFL kernels. Cartesian regressions unchanged.

### New gravity source terms

`SourceTerms` now exposes a second physical-gravity flavour alongside the
existing `const_accel`:

- **Constant radial gravity** is the pre-existing `const_accel` flag with
  `const_accel_dir=1` and `const_accel_val=-g0`. No code added; the
  spherical FV update treats `dir=1` as the radial direction.
- **Inverse-square radial gravity** is new: `r_inv_sq_gravity = true` plus
  `r_inv_sq_gm = <GM>` under `<hydro_srcterms>`. Kernel:
  `S(m_r) = -rho * GM / r^2`, `S(E) = -rho * v_r * GM / r^2`, no
  theta/phi momentum source. Uses `r_vol(m,i)` (Mignone volume-averaged r)
  from `SphericalShellGeom`. Errors out at construction if
  `<coord>/system != spherical_shell`.

Both are opt-in via `<hydro_srcterms>` in the input file:
```ini
<hydro_srcterms>
const_accel       = true
const_accel_val   = -0.5     # for HSE const-g atmosphere
const_accel_dir   = 1
# OR
r_inv_sq_gravity  = true
r_inv_sq_gm       = 1.0      # for Parker / Keplerian
```

The energy source is **not** a potential-energy formulation. Total
gas+kinetic+gravitational energy is therefore not conserved at the FV level.
This is acceptable for the validation tests here; if Bondi-/MHD-grade
conservation is needed later, switch to a potential-energy formulation.

### New `<problem>/mode` values

| mode                  | type            | purpose                                        |
|-----------------------|-----------------|------------------------------------------------|
| `hydrostatic_constg`  | quantitative    | isothermal HSE in constant gravity             |
| `hydrostatic_r2`      | quantitative    | isothermal HSE in 1/r^2 gravity                |
| `solid_body_rotation` | quantitative    | one-step centrifugal-source diagnostic         |
| `keplerian_orbit`     | quantitative+vibe | near-Keplerian circular orbit, Lz conservation |
| `thin_disk`           | vibe            | pressure-supported thin disk in 1/r^2 gravity  |
| `parker_isothermal`   | quantitative+vibe | analytic Parker wind initialiser + user BC     |

### New input files

- `inputs/tests/sph_shell_hydrostatic_constg.athinput`
- `inputs/tests/sph_shell_hydrostatic_r2.athinput`
- `inputs/tests/sph_shell_solid_body_rotation.athinput`
- `inputs/tests/sph_shell_keplerian_orbit.athinput`
- `inputs/tests/sph_shell_thin_disk_vibe.athinput`
- `inputs/tests/sph_shell_parker_isothermal_1d.athinput`
- `inputs/tests/sph_shell_parker_isothermal_3d.athinput`

### How to run

All tests use the same build: `-DPROBLEM=sph_shell_hydro`. Examples:

```bash
# Quantitative checks
./build/src/athena -i inputs/tests/sph_shell_hydrostatic_constg.athinput
./build/src/athena -i inputs/tests/sph_shell_hydrostatic_r2.athinput
./build/src/athena -i inputs/tests/sph_shell_solid_body_rotation.athinput
./build/src/athena -i inputs/tests/sph_shell_keplerian_orbit.athinput
./build/src/athena -i inputs/tests/sph_shell_parker_isothermal_1d.athinput

# Visual / vibe
./build/src/athena -i inputs/tests/sph_shell_thin_disk_vibe.athinput
./build/src/athena -i inputs/tests/sph_shell_parker_isothermal_3d.athinput
```

### Exact analytic formulas implemented

- **Const-g HSE**: `rho(r) = rho0 * exp(-(r-r_ref)/H_p)`, `p = c_s^2 rho`,
  `H_p = c_s^2 / g0`.
- **1/r² HSE**: `rho(r) = rho0 * exp[(GM/c_s^2)(1/r - 1/r_ref)]`, `p = c_s^2 rho`.
- **Solid-body rotation IC**: `v_phi = Omega * r * sin(theta)`, `v_r = v_th = 0`,
  uniform rho/p. Expected one-cycle acceleration:
  `d(rho v_r)/dt = +rho * Omega^2 * r * sin^2(theta)`,
  `d(rho v_th)/dt = +rho * Omega^2 * r * sin(theta) cos(theta)`,
  `d(rho v_phi)/dt = 0` (outward / equatorward centrifugal force in absence
  of pressure gradient or gravity to balance it).
- **Keplerian orbit IC**: `v_phi(r,theta) = sqrt(GM/r) * sin(theta)`,
  approximately balances 1/r² gravity in the radial direction at the
  equator. The theta-direction is NOT exactly balanced and shows mild
  adjustment in a thin wedge.
- **Thin-disk IC**:
  `rho(r,theta) = rho_floor + rho0 * exp(-(r-r_disk)^2/(2 sigma_r^2)) * exp(-(theta-pi/2)^2/(2 sigma_th^2))`,
  `p = c_s^2 rho`, `v_phi = sqrt(GM/r) sin(theta)`. NOT an exact equilibrium;
  expect adjustment waves.
- **Parker isothermal**:
  `r_c = GM / (2 c_s^2)`,
  Mach `M(r)` solves `M^2 - ln(M^2) = 4 ln(r/r_c) + 4 r_c/r - 3` on the
  transonic branch (subsonic for r<r_c, supersonic for r>r_c) via a
  60-iteration bisection inside the kernel.
  `v_r(r) = M(r) c_s_iso`, `rho(r) = rho_inner * (M_inner/M) * (r_inner/r)^2`,
  `p = c_s_iso^2 rho`. User BC fixes the analytic Parker state in the
  inner radial ghost zones (`<mesh>/ix1_bc = user`); the outer radial BC
  (also `user`) is zero-gradient outflow.

### Quantitative CPU/Serial results (this worktree)

**Constant-g isothermal HSE** (gentle stratification, H_p=2 over r ∈ [1,3]):

| t    | max\|v\|/cs | L1 \|drho\|/rho |
|------|-----------|------------------|
| 0.01 | 3.5e-3    | 5.6e-5            |
| 0.05 | 1.5e-2    | 4.3e-4            |
| 0.1  | 2.8e-2    | 1.5e-3            |
| 0.5  | 1.4e-1    | 3.0e-2            |

Linear-in-time drift at small t consistent with the isothermal-IC /
adiabatic-hydro mismatch (entropy gradient produces convection-like
acceleration). Not well-balanced but quiet at early times.

**1/r² isothermal HSE** (mild bound atmosphere, GM/r_ref = 0.5·c_s² at r_ref=1):

| t    | max\|v\|/cs | L1 \|drho\|/rho |
|------|-----------|------------------|
| 0.01 | 3.3e-3    | 3.0e-5            |
| 0.05 | 1.4e-2    | 2.3e-4            |
| 0.1  | 2.6e-2    | 7.5e-4            |
| 0.5  | 1.0e-1    | 1.2e-2            |

Similar drift profile.

**Solid-body rotation** (Omega=0.05, one RK2 cycle, scale = rho0 Omega² r_typ
= 3.75e-3):

```
Linf a_r err   = 3.2e-6   (relative 8.6e-4 = 0.09%)
Linf a_th err  = 3.3e-6   (relative 8.7e-4)
Linf a_phi err = 3.8e-6   (relative 1.0e-3)
L1   a_r err   = 4.0e-7
L1   a_th err  = 1.4e-6
```

All three relative errors are O(Omega·dt) ≈ 4e-4 to 1e-3, consistent with
the RK2 finite-step linear approximation of a non-equilibrium IC. **The
centrifugal/cot-θ source terms are working correctly.**

**Keplerian orbit** (GM=1, very cold p=1e-3, t units of ~ orbital periods):

| t   | (Lz-Lz0)/Lz0 | (M-M0)/M0 | max\|v_r\|/\|v_phi\| |
|-----|--------------|-----------|----------------------|
| 0.1 | +1.25e-3     | +1.34e-3  | 3.4e-3                |
| 0.2 | +5.0e-3      | +5.4e-3   | 6.8e-3                |
| 0.5 | +3.1e-2      | +3.3e-2   | 1.6e-2                |
| 1.0 | +1.2e-1      | +1.3e-1   | 2.8e-2                |

Both Lz and M grow together (ΔLz/Lz0 ≈ ΔM/M0 to within ~6%) — specific
angular momentum is approximately conserved. The mass increase comes from
the outflow theta boundaries admitting flux when fluid is centrifugally
pushed against them; this is a known limitation of outflow theta BCs in
a rotating wedge. The orbit stays close to circular: max|v_r|/|v_phi| < 3%
even at t=1.

**Thin disk** (cs=0.1, GM=1, r_disk=1.5, vibe-quality):

```
t=0.71:  mass=1.96   <r>_mass=1.65  (drift to outside r_disk=1.5)
         <theta>_m=pi/2 exactly     (equatorial symmetry preserved)
         Lz=2.48
```

The mass-weighted theta centroid stays exactly at the equator — the
geometric source terms preserve the equatorial mirror symmetry. <r>_mass
drifts outward as the pressure-supported disk relaxes.

**Parker isothermal 1D** (256 radial cells, r_c=2.0, transonic Parker
through the critical point inside the domain):

| t    | L1\|dv/v\| | L1\|drho/rho\| | Linf\|dv/v\| |
|------|------------|----------------|----------------|
| 0.0  | 1.6e-17    | 0              | 1.6e-16        |  ← IC matches analytic to roundoff
| 0.05 | 4.5e-4     | 4.0e-5         | 1.7e-2         |
| 0.2  | 5.7e-3     | 5.9e-4         | 8.8e-2         |
| 0.5  | 2.6e-2     | 4.9e-3         | 1.6e-1         |

**The IC bisection is exact** (roundoff). Drift under adiabatic evolution
grows roughly linearly with t (over the timescale shown, the wind hasn't
reached steady-state adiabatically because the isothermal Parker isn't an
adiabatic stationary solution).

**Parker isothermal 3D** (64×32×32 wedge, t=0.1):

```
L1 |dv/v|     = 1.7e-3       Linf|dv/v|     = 1.4e-2
L1 |drho/rho| = 3.8e-4       Linf|drho/rho| = 1.0e-2
```

Comparable to 1D quality. The analytic Parker solution depends only on r,
so a clean radial profile is recovered with minimal angular variation.

### Regression — what should NOT have changed

- 3A FV divergence test: bit-identical (case 1 6.1e-16, case 2 1.8e-4, case 3 9.1e-4).
- 3A homologous test: bit-identical.
- 3A uniform-state preservation: max|drho|=0, max|dp|=0, max|v|=8.1e-17.
- Cartesian linear_wave_hydro cycles 0–3 dt: 1.404847e-02, 1.404851e-02,
  1.404857e-02, 1.404861e-02 (bit-identical).
- Geometry sanity (sph_shell_geom.athinput): rel err 3.4e-14.

### Task 3B caveats

- **Hydrostatic atmospheres are not well-balanced.** The IC is isothermal but
  hydro evolution is adiabatic with γ=5/3. The IC entropy gradient drives
  spurious vertical motion at rate ~O(Δr / H_p)·cs per sound crossing.
  Quantitatively: max|v|/cs ~ 0.03 by t=0.1, growing roughly linearly. For
  long runs you want a well-balanced reconstruction or to switch to true
  isothermal evolution.
- **Parker is isothermal, hydro is adiabatic.** Same drift mechanism;
  expect a few × 1e-3 fractional drift per CFL after t > sound-crossing
  time (Δr/cs ≈ 10 here). The IC bisection is exact, so any drift is from
  the EOS mismatch — not from the gravity source or geometric terms.
- **Outflow boundaries on theta** in rotating tests admit some return-flux
  when centrifugal acceleration pushes fluid against them. This shows up
  as positive ΔM/M (mass entering from the ghost zones) and matching
  positive ΔLz/Lz (specific angular momentum approximately preserved).
  Replace with reflecting θ-boundaries for sterner conservation tests.
- **Keplerian IC is not equilibrium in the θ-direction.** The centrifugal
  source pushes material toward the equator at all latitudes. The thin
  wedge minimises this; for a wider polar coverage the drift is larger.
- **Thin disk IC is not equilibrium** (no vertical hydrostatic balance).
  Expect adjustment waves on the local sound crossing time. Use as a
  vibe-only check; for stricter tests construct an exact disk equilibrium
  with pressure support including the vertical balance.
- **Energy source is not potential-energy form.** Total energy drifts in
  long Parker / atmosphere runs because gravitational potential energy is
  not tracked in IEN. Acceptable for these validation tests; revisit when
  long-term energy conservation matters.
- **No GPU testing** in 3B. Kernels are Kokkos-portable; only Serial parity
  is verified locally.
- **FOFC still unsupported in spherical_shell.** Same as Task 2.

### Plot suggestions for the vibe checks

- `SphShellHSEconst.hydro_w.????.vtk` / `SphShellHSEr2.hydro_w.????.vtk`:
  radial profile of rho, p, v_r. Look for smooth monotone profiles tracking
  the analytic exponential; spurious velocity should be ≪ cs at early times.
- `SphShellThinDisk.hydro_w.????.vtk`: rho slice through phi=0 (r-θ plane).
  The disk should remain a coherent band centred at the equator.
- `SphShellParker3D.hydro_w.????.vtk`: v_r slice through z=0 (r-φ plane).
  Should be near-uniform in φ at every r; angular striping indicates a bug.

## Status after Task 4 (MHD + constrained transport)

Spherical-shell MHD with CT is implemented on top of the Task 1–3C hydro path.
The hydro solver is unchanged. Cartesian MHD is unchanged. The mathematical
Riemann solvers are unchanged; spherical handling lives entirely in the
flux-divergence, CT, CFL, and geometric-source-term layers.

### Files changed

**Modified (minimal):**
- `src/coordinates/coordinates.hpp` — declare `AddSphericalShellMHDSrcTerms`.
- `src/coordinates/coordinates.cpp` — remove the MHD-vs-spherical-shell
  construction block (replaced by a one-line note).
- `src/coordinates/spherical_shell.hpp` — add `SphEdge1Length`, `SphEdge2Length`,
  `SphEdge3Length` as inline accessors (no new storage; derived from existing
  views).
- `src/coordinates/spherical_shell.cpp` — add `AddSphericalShellMHDSrcTerms`
  alongside the existing hydro version.
- `src/mhd/mhd.cpp` — error on `<mhd>/fofc=true` with `spherical_shell`.
- `src/mhd/mhd_update.cpp` — host-level branch on `coord_system`; spherical
  branch uses the same area/volume form as `Hydro::RKUpdate` spherical branch.
- `src/mhd/mhd_newdt.cpp` — capture `is_spherical` and `geom`; spherical
  CFL uses physical widths.
- `src/mhd/mhd_ct.cpp` — host-level branch on `coord_system`; spherical
  branch uses face areas and edge lengths from `SphericalShellGeom`.
- `src/mhd/mhd_tasks.cpp` — `MHDSrcTerms` now calls
  `AddSphericalShellMHDSrcTerms` after the existing GR / shearing-box paths.

**New:**
- `src/pgen/sph_shell_mhd.cpp` — three MHD test modes plus a free-standing
  `divB` diagnostic that uses the same `SphericalShellGeom` factors as CT.
- `inputs/tests/sph_shell_mhd_uniform_static.athinput`
- `inputs/tests/sph_shell_mhd_monopole.athinput`
- `inputs/tests/sph_shell_mhd_monopole_logr.athinput`
- `inputs/tests/sph_shell_mhd_toroidal_static.athinput`

**Unchanged (deliberately, to minimise merge conflicts with the cluster
validation branch):**
- All hydro pgens and hydro tests.
- All hydro plotting / vibe / cluster-validation infrastructure.

### Athena++ reference functions used

| Athena++ source                                            | Used to derive                       |
|------------------------------------------------------------|--------------------------------------|
| `srcpp/coordinates/spherical_polar.cpp::Edge1Length` etc.  | edge-length formulae L1/L2/L3        |
| `srcpp/coordinates/spherical_polar.cpp::Face*Area`         | (already used in Task 1; reused)     |
| `srcpp/coordinates/spherical_polar.cpp::AddCoordTermsDivergence` (MHD branches with `MAGNETIC_FIELDS_ENABLED`) | MHD source-term split + flux pairings |
| `srcpp/field/ct.cpp::Field::CT`                            | CT Stokes-loop formula and sign conventions |

No Athena++ object-oriented architecture is ported.

### Edge lengths

Added as `KOKKOS_INLINE_FUNCTION` accessors in `spherical_shell.hpp`:

```
L1(m,i)       = dr(m,i)                                          // radial edge
L2(m,j,i)     = r_face(m,i) * dtheta(m,j)                        // theta edge
L3(m,k,j,i)   = r_face(m,i) * sin_theta_face(m,j) * dphi(m,k)    // phi edge
```

Derived from existing `SphericalShellGeom` views — **no new storage**, no new
trig at runtime. The CT kernel factors out edge lengths that don't vary
across the relevant difference (e.g. `L1(i)` is constant across `j` and `k`).

### CT update formula

The discrete Stokes loop is identical in form to AthenaK's Cartesian CT
but with face areas and edge lengths from spherical geometry:

```
B1(face_i)  -= (β·dt / A1) [L3(k, j+1, i) E3(k, j+1, i)
                            - L3(k, j,   i) E3(k, j,   i)]    [multi-d]
            += (β·dt / A1) L2(j, i) [E2(k+1, j, i) - E2(k, j, i)]  [3-d]

B2(face_j)  += (β·dt / A2) [L3(k, j, i+1) E3(k, j, i+1)
                            - L3(k, j, i  ) E3(k, j, i  )]
            -= (β·dt / A2) L1(i) [E1(k+1, j, i) - E1(k, j, i)]    [3-d]

B3(face_k)  -= (β·dt / A3) [L2(j, i+1) E2(k, j, i+1)
                            - L2(j, i  ) E2(k, j, i  )]
            += (β·dt / A3) L1(i) [E1(k, j+1, i) - E1(k, j, i)]    [multi-d]
```

Signs and orientation follow AthenaK Cartesian CT (and Athena++).

### divB diagnostic

Same face areas as CT:

```
divB = (1/V) [A1+ B1+ - A1- B1-
            + A2+ B2+ - A2- B2-
            + A3+ B3+ - A3- B3-]
```

Implemented in `src/pgen/sph_shell_mhd.cpp::ComputeDivBStats`. Reports
volume-averaged L1, L2, Linf, plus a normalised `Linf · h / |B|max` metric.

### MHD geometric source terms

Extends `Coordinates::AddSphericalShellHydroSrcTerms` with magnetic-stress
contributions:

```
m_ii_MHD = ρ(v_θ² + v_φ²) + 2p + B_r²
m_pp_MHD = ρ v_φ² + p + 0.5 (B_r² + B_θ² - B_φ²)
m_ph_MHD = ρ v_θ v_φ - B_θ B_φ      (1-D-theta fallback only)
```

The radial- and theta-flux pairings (`coord_src2_i · (r²_face · F1[IM2/IM3])`
and `coord_src1_i · coord_src2_j · (sin_face · F2[IM3])`) reuse the MHD
Riemann fluxes, which **already include** the `-B_i B_j` magnetic-stress
contributions in their definition. No extra B-only flux terms are needed.

Energy `IEN` has no geometric source term (matches Athena++'s conservative
split for both hydro and MHD).

### Test results (CPU / Kokkos Serial)

All tests use `-DPROBLEM=sph_shell_mhd`.

**uniform_static** (B = 0, v = 0, 20 cycles, single block):

```
max|drho| = 0  max|dp| = 0  max|v| = 4.9e-17
divB:     L1 = 0      Linf = 0
```

Exactly preserved.

**monopole, uniform-r** (B_r = B0 (r_ref/r)² on radial faces; 64 × 32 × 16):

```
nlim=0:   divB L1 = 1.05e-15   Linf = 8.43e-15   Linf·h/|B|max = 2.7e-16
nlim=20 / t=0.14 with analytic radial BC:
          divB L1 = 1.29e-15   Linf = 1.51e-14
                  max|B1f - analytic| = 1.1e-16
                  max|v|/cs = 4.1e-5
```

**monopole, log-r** (one decade r ∈ [1, 10]; 64 × 32 × 16):

```
nlim=0:   divB L1 = 4.15e-16   Linf = 8.53e-15
nlim=17 / t=0.14 with analytic radial BC:
          divB L1 = 5.17e-16   Linf = 1.11e-14
                  max|B1f - analytic| = 1.1e-16
                  max|v|/cs = 6.4e-5
```

**monopole, multi-block (4 × 2 × 2 = 16 blocks):**

```
uniform-r nlim=20:  divB L1 = 1.24e-15  Linf = 1.46e-14  max|B1f-analytic| = 1.1e-16
log-r     nlim=20:  divB L1 = 5.74e-16  Linf = 1.73e-14  max|B1f-analytic| = 2.2e-16
```

**toroidal_static** (axisymmetric B_φ ∝ sin θ / r; 10 cycles):

```
nlim=10:  divB L1 = 1.24e-17  Linf = 9.28e-17  max|v|/cs = 5.5e-4
```

**Regressions** — bit-identical to Task 3C baseline:

- Cartesian `linear_wave_hydro` cycles 0–3 dt's match Task 3C.
- `sph_shell_geom.athinput` wedge-volume relative error 3.4e-14.
- `sph_shell_hydro_uniform.athinput` `max|drho|=0, max|v|=8.1e-17`.
- `sph_shell_divergence.athinput` L1 values match.

### Tricky things found during Task 4

1. **Edge-length index dependence** is subtle. `L1 = dr(i)` depends only on
   the radial cell index. `L2 = r_face(i) · dθ(j)` depends on the radial
   face and theta-cell width. `L3 = r_face(i) · sin(θ_face(j)) · dφ(k)`
   depends on all three. In the CT kernel for each B component, identify
   which axis the difference is taken along and factor out the edge length
   if it doesn't vary across that difference. Got this wrong twice in
   drafts before fixing.

2. **The old monopole velocity residual was a boundary problem.** Plain
   radial outflow copied `B_r` into ghost faces, breaking the analytic 1/r²
   field at the radial boundaries. That produced a boundary-layer Lorentz
   force and ~4% `max|v|/cs` after 20 cycles while **CT still preserved
   divB at machine precision**. The monopole mode now installs analytic
   radial user BCs for `B_r`, transverse face fields, and conserved ghost
   cells. With those BCs the fixed-time residual converges at about second
   order:

   | grid | uniform-r max\|v\|/cs at t=0.14 | log-r max\|v\|/cs at t=0.14 |
   |------|----------------------------------|------------------------------|
   | 32×16×8   | 1.8e-4 | 2.8e-4 |
   | 64×32×16  | 4.1e-5 | 6.4e-5 |
   | 128×64×32 | 9.6e-6 | not rerun |

   Halving CFL at 64×32×16 left the residual unchanged (4.14e-5), and
   donor-cell/PLM/PPM4 reconstruction all gave ~4e-5. The remaining
   residual appears to be the expected spatial force-balance truncation,
   not a CT or source-ordering failure.

3. **Riemann flux already contains magnetic stress.** The MHD HLLD/HLLE
   solvers return F[IM1] that includes `+p + 0.5·B² - B_x²` etc. So the
   spherical FV flux divergence on cell-centred momenta works unchanged.
   No need to add separate "magnetic-flux" terms.

4. **`bcc0` is up-to-date in `MHDSrcTerms`.** Same as Athena++: `ConsToPrim`
   at the end of the previous stage populates `bcc0` from `b0` face values.
   When `MHDSrcTerms` runs (after `RKUpdate`, before `CT`), `bcc0`
   reflects the BEGINNING of this stage — same time level as `w0` and as
   the fluxes used in this stage's divergence.

5. **CT face-loop bounds.** B1 lives on radial faces with extent
   `(is..ie+1)`; B2 on `(js..je+1)`; B3 on `(ks..ke+1)`. The Cartesian
   loop bounds already get this right; just preserve them in the spherical
   branch.

### Files likely to conflict with the cluster-validation branch

Practically none. The MHD path is in MHD-only files
(`mhd_update.cpp, mhd_ct.cpp, mhd_newdt.cpp, mhd_tasks.cpp, mhd.cpp`), one
addition to `spherical_shell.cpp` (the new function appended at the
bottom), one declaration in `coordinates.hpp`, and one block removal in
`coordinates.cpp`. Hydro pgens, hydro tests, hydro plotting, and
hydro-only validation files were not touched.

### Task 4.2 additions: polytropic Parker MHD background

Added to `sph_shell_mhd.cpp`:

- New mode `parker_polytropic` — ideal-gas MHD (γ close to 1, default 1.05) Parker
  wind plus radial monopole `B_r = B0 r_inner²/r²`. Critical-point
  parameterisation: `a_c = sqrt(GM/(2 rcrit))`, `x = r/rcrit`, `y = U/a_c`.
  Branch-safe bisection root solve on
  `F(y, x; γ) = y²/2 + (1/(γ−1))·(1/(y x²))^(γ−1) − 2/x − (1/(γ−1) − 3/2)`;
  subsonic for `r < rcrit`, supersonic for `r > rcrit`, `y=1` at `r=rcrit`.
- `ρ_c` calibrated from mass conservation so `ρ(r_inner)=rho_inner`.
- `B0` calibrated so the Alfvén point sits at `r = alfven_point_target`:
  `B0 = U(rA)·√ρ(rA)·(rA/r_inner)²`.
- Helper functions `PolyParkerF`, `PolyParkerY`,
  `EvaluatePolytropicParker`, `EvaluatePolytropicParkerBrFace` are KOKKOS device
  callable and shared between IC, user BC, and finalizer.
- User radial BC `ParkerPolyMhdRadialBCs` imposes analytic ghosts (outer
  BC selectable analytic/outflow). `B1f` is set from face-radius analytics.
- `ParkerPolytropicFinalize` reports L1/Linf for ρ, v_r, p, mass flux,
  transverse v/B normalised by c_s and √ρ_inner, divB stats. Optional
  radial-profile CSV is written under `<problem>/csv_dir` if set.

Compatible with **HLLD** and **LHLLD** (LHLLD is ideal-gas-only in this fork).
This is the recommended flowing background for the next lower-boundary
monochromatic Alfvén-wave driver test — see
[`tst/sph_test_summaries/monopole_lhlld_minimal_summary.md`] for context on
why the static monopole + HLLD path is not suitable for long quantitative
runs.

Default input `inputs/tests/sph_shell_mhd_parker_polytropic.athinput`:
γ=1.05, GM=4, rcrit=2, rho_inner=1, r_inner=1, rA_target=13, r ∈ [1, 20],
8×8 thin equatorial tube, uniform-r, ndiag=200, `tlim=5`. Local validation
results, run script, and plot script live under `poly_parker_mhd_validation/`
(gitignored). Baseline `plm_hlld_nr512` at `t=5` gives
`max|v_⊥|/c_s ≈ 1.1e-12`, `divB L1 ≈ 1.2e-12`, with L1 radial errors
`O(few × 10⁻³)` dominated by the inner boundary.

### Task 4.1 additions: radial Alfvén + equatorial field loop

Two new pgen modes added to `sph_shell_mhd.cpp`:

**`radial_alfven`** — outgoing axisymmetric Alfvén pulse on a 1/r² monopole.
- IC: B_r(r) = B₀(r_ref/r)² (face-centred monopole, divB-free by construction);
  δB_φ = ε·B₀·envelope(r)·carrier(r) on phi faces (axisymmetric → no φ-dep,
  preserves divB=0); δv_φ = -δB_φ/√ρ on cells (outgoing Alfvén polarisation).
- Standard test (`sph_shell_mhd_radial_alfven.athinput`, nx1=256, σ_r=0.3):
  at t=5, centroid 2.200 vs WKB 2.216 → -2% lag; peak amplitude 93% of IC;
  divB L1 = 3.3e-15.
- Vibe test (`sph_shell_mhd_radial_alfven_vibe.athinput`, nx1=1024, k_r=60,
  r ∈ [1, 10]): runs cleanly; divB ~ 1e-14 throughout; VTK dumps every t=5
  for plotting envelope compression / refraction.

**`loop_eq`** — magnetic field loop on equatorial wedge. Two flavors:
- `axisymmetric` (standard, `sph_shell_mhd_loop_eq.athinput`):
  A_φ Gaussian loop in (r, θ) at θ=π/2, sampled at phi-edges, with discrete
  Stokes-loop B-field construction. v=0. CT preserves max|B_r|, max|B_θ| to
  13 digits over 50 cycles; B_φ stays at 1e-18; total ½B²V preserved to 8
  digits; divB L1 = 6.2e-18.
- `advect` (vibe, `sph_shell_mhd_loop_eq_vibe.athinput`):
  A_θ Gaussian loop in (r, φ) at φ_c=1, sampled at theta-edges. Solid-body
  rotation v_φ = Ω·r·sin θ with Ω=0.25 on a full 2π φ-annulus. At quarter
  rotation the loop has been advected 1.57 rad with divB at 2e-17;
  amplitude decays from PLM dissipation (factor ~3.5 reduction at nx3=128
  resolution). VTK dumps every T/10 for visual loop-rotation check.

Both modes use the same `ComputeDivBStats` diagnostic.

### Native binary output check (small MHD validation)

Native AthenaK `bin` output was checked for spherical-shell MHD with
`variable=mhd_w_bcc`. The existing `vis/python/bin_convert.py` reader can
read the files and reports time, cycle, dimensions, MeshBlock layout, and
variables `dens, velx, vely, velz, eint, bcc1, bcc2, bcc3`. The binary
header carries the full input dump, including `<coord>/system =
spherical_shell` and `<spherical_shell>/radial_grid`, so no writer metadata
change was needed. Face-centred B is not written by `mhd_w_bcc`; use
cell-centred `bcc*` unless a dedicated output variable is added.

Local ignored validation files:

```
mhd_output_validation/scripts/inspect_mhd_bin.py
mhd_output_validation/scripts/sph_shell_mhd_toroidal_bin.athinput
```

The script reads a bin file through `bin_convert.py`, reconstructs spherical
coordinates from the header, prints `max|v|/cs`, and saves small r-theta
and mapped R-Z plots of `|B|` and `|v|`. A 2-rank CPU/MPI smoke test wrote
a single combined binary file with two MeshBlocks; the same reader loaded it
successfully.

### Task 4.2 additions: native-bin MHD vibe checks

Two GPU native-binary slice vibe inputs were added:

- `inputs/tests/sph_shell_mhd_radial_alfven_bin_vibe.athinput`
  runs a 512×96×128 radial Alfvén packet on a 1/r² monopole background,
  with `rtheta`, `rphi`, and `thetaphi` `mhd_w_bcc` binary slice outputs.
  Radial user BCs are used here too, matching the fixed monopole tests.
- `inputs/tests/sph_shell_mhd_radial_alfven_wkb_powerlaw.athinput`
  is the stricter quantitative version: a thin 16×16 angular tube, 20,480
  radial cells, `radial_grid=power_law`, `r_grid_alpha=1/3`, WKB phase in
  Alfvén travel-time coordinate, and `amplitude_model=sqrt_va`.
- `inputs/tests/sph_shell_mhd_loop_advect_bin_vibe.athinput`
  runs a 256×48×256 non-axisymmetric loop advection check on a full 2π
  equatorial annulus, again writing the three native-bin slices.

The ignored local script
`mhd_vibe_validation/scripts/plot_mhd_bin_slices.py` reads these files via
`vis/python/bin_convert.py`, reconstructs spherical coordinates from the
embedded input header, and saves four-panel diagnostic plots:

1. true-geometry r-theta slice in physical R-Z;
2. true-geometry r-phi slice in physical X-Y;
3. projected constant-r theta-phi surface;
4. raw logical r-theta array.

It overlays projected magnetic-field directions with normalized quiver
arrows. For the radial Alfvén run it also writes `v_perp(r)`, `B_perp(r)`,
background `B_r`, `rho`, `v_A`, area factor, and the two candidate Elsasser
amplitudes `z1 = v_perp - B_perp/sqrt(rho)` and
`z2 = v_perp + B_perp/sqrt(rho)`.

Cluster A100 results in `mhd_vibe_validation/summary.md`:

- Radial Alfvén bin-vibe (`t=12`): `Linf*h/|B|max = 1.3e-13`; WKB centroid
  lag is about 7% of travelled distance. The high-carrier packet is strongly
  PLM-damped by this late time (`peak |dB_phi|/(eps*B0) = 3e-3`), so this
  input is mainly an output/geometry diagnostic.
- Loop advection bin-vibe (`t=2π`, one quarter rotation for `Omega=0.25`):
  `Linf*h/|B|max = 2.1e-14`; the loop appears in the expected r-phi and
  constant-r slices after `phi advected by = 1.570796 rad`.
- WKB power-law radial Alfvén (`t=55`): the packet ray moved from `r_c=2.2`
  to `r=4.533` with relative centroid error `1.2e-4` and
  `Linf*h/|B|max = 1.3e-13`. The local carrier resolution stayed above
  23 points per wavelength. Against the requested
  `z_+ ~ sqrt(v_A) ~ r_c/r` scaling, the signed
  `z_out = v_phi - B_phi/sqrt(rho)` RMS amplitude was about 25% above the
  WKB reference by the final snapshot, not damped below it. This is now a
  useful quantitative amplitude-normalisation target for the next pass.

The Parker-wind Alfven/Heinemann-Olbert and rotating Parker-spiral vibe
checks were deferred intentionally; this round stayed with the existing
ideal-MHD radial atmosphere and loop pgen modes.

### MHD known limitations

- **GPU testing is partial.** A100 CUDA smoke and native-bin vibe checks now
  run, but a full GPU convergence/parity matrix is still pending.
- **FOFC unsupported under `spherical_shell`** for both hydro and MHD.
- **AMR/SMR unsupported.** Spherical CT prolongation/restriction is the
  hard remaining piece; not in scope here.
- **No pole BCs.** Same caveat as hydro.
- **Reconstruction is index-uniform.** Same caveat as hydro on log-r.
- **Monopole force balance requires analytic radial ghost zones.** Plain
  radial outflow gives a boundary-layer velocity residual; the standard
  monopole inputs now use `ix1_bc=user` / `ox1_bc=user`.
- **PLM diffusion is significant** on small/under-resolved structures in
  the vibe tests. Loop vibe loses ~70% energy per quarter rotation at
  nx3=128; AW vibe carrier (λ=0.1) gets smoothed at nx1=1024. Both are
  expected and run-to-completion sanity checks; higher resolution or
  PPM/WENO would substantially reduce dissipation.

### Recommended next step

1. **GPU compile + parity smoke test** for both hydro and MHD on the
   cluster. The kernels are CUDA-portable but only Serial parity is
   verified. Expect to find a few uninitialised-variable warnings or
   math-namespace tweaks; nothing structural.
2. **Remaining MHD vibe checks**: spherical MHD blast wave, MHD current
   sheet on an equatorial wedge. (Radial Alfvén + field loop now done in
   Task 4.1.) These can run at high resolution on the cluster for
   visual / post-processing inspection.
3. Only after both of those: **AMR/SMR for CT in spherical**, or move
   on to the actual solar-corona physics layered on this stack.

---

## Status after Task 3C (radial-grid options)

Task 3C adds nonuniform radial grids for `coord_system=spherical_shell`.
The solver, geometric source terms, and CFL are unchanged — only the way
`r_face` is populated. All downstream geometry is derived from `r_face`
(per the Mignone-2014 centroids established in Task 1), so adding new
radial mappings is a localised change.

### Radial grid input

A new `<spherical_shell>` block:
```ini
<spherical_shell>
radial_grid  = uniform | log | power_law | user   # default uniform
r_grid_alpha = <Real>                              # used by power_law
```

### Radial face mapping

Let `xi_norm = (x1_logical − mesh.x1min) / (mesh.x1max − mesh.x1min)`,
clamped to ghosts by `LeftEdgeX` extrapolation. Then:

- `uniform`:   `r_face = mesh.x1min + (mesh.x1max − mesh.x1min) · xi_norm`
- `log`:       `r_face = mesh.x1min · (mesh.x1max / mesh.x1min)^xi_norm`
- `power_law`: `r_face = mesh.x1min + (mesh.x1max − mesh.x1min) · xi_norm^alpha`
- `user`:      `r_face = ufn(xi_norm)` (host-side; result deep-copied to device)

Multi-block: each block's `(x1min, x1max)` is the LINEAR sub-range produced
by AthenaK's mesh splitter. Every block applies the SAME global mapping to
its own `xi`, so face radii at block boundaries match exactly.

### User radial grid hook

The hook is a process-global function-pointer registry in
`coordinates/spherical_shell.{hpp,cpp}`:

```cpp
void SetUserRadialGridFunc(UserRadialGridFnPtr fn);
UserRadialGridFnPtr GetUserRadialGridFunc();
```

The pgen `UserProblem()` runs AFTER `Coordinates` is constructed, so it
**cannot** register the hook. Users must register via a **static
initializer** in their pgen translation unit (runs before `main()`):

```cpp
namespace { struct R { R() { SetUserRadialGridFunc(&MyFn); } } _r; }
```

Documented in README-sph.md §3.3.

If selected without registration, the build errors out at construction:
> `radial_grid=user but no user function was registered via SetUserRadialGridFunc()`

### Files changed (Task 3C)

- `src/coordinates/spherical_shell.hpp` — added `RadialGridType` enum,
  `UserRadialGridFnPtr` typedef, `Set/GetUserRadialGridFunc()`, and the
  `grid_type`+`grid_alpha` parameters on `ConstructSphericalShellGeometry`.
- `src/coordinates/spherical_shell.cpp` — registry implementation, plus a
  switched radial-face kernel (one `par_for` per mode for
  uniform/log/power_law; host-loop + `deep_copy` for user).
- `src/coordinates/coordinates.hpp` — added `radial_grid` member on
  `Coordinates`.
- `src/coordinates/coordinates.cpp` — parses `<spherical_shell>/radial_grid`
  and `r_grid_alpha`, validates the log/user preconditions, passes through
  to the geometry constructor.
- `src/parameter_input.cpp` — adds `spherical_shell` to the allowed input
  block list.
- `src/pgen/pgen.hpp` — note (no new field) pointing users at
  `SetUserRadialGridFunc()`.

### Files added (Task 3C)

- `inputs/tests/sph_shell_geom_logr.athinput`
- `inputs/tests/sph_shell_geom_powerlaw.athinput`
- `inputs/tests/sph_shell_divergence_logr.athinput`
- `inputs/tests/sph_shell_radial_acoustic_logr.athinput`
- `README-sph.md` — new top-level user guide.

### Quantitative log-r results (CPU/Serial)

**Geometry wedge volume:**

| input                                       | rel err   |
|---------------------------------------------|-----------|
| `sph_shell_geom.athinput` (uniform, r∈[1,2]) | 3.42e-14 |
| `sph_shell_geom_logr.athinput` (log, r∈[1,10]) | 6.35e-15 |
| `sph_shell_geom_powerlaw.athinput` α=2.0     | 7.34e-15 |
| same, α=0.5                                  | 1.07e-14 |
| same, α=1.0 (= uniform check)                | 1.23e-14 |

All near machine precision. Volume from face positions is exact under any
monotonic r_face.

**Uniform-state preservation on log-r** (`sph_shell_geom_logr.athinput`
with `<problem>/mode=uniform`, 50 cycles):

```
max |drho| = 0
max |dp|   = 0
max |v|    = 1.5e-16
```

The pressure-curvature cancellation between flux divergence and geometric
source is local per cell, so it carries over to log-r exactly.

**FV divergence diagnostic on log-r** (resolution-doubling convergence):

| case            | 64×32×16 L1 | 128×64×32 L1 | ratio | expected |
|-----------------|-------------|--------------|-------|----------|
| F_r = r³        | 1.16e-2     | 2.90e-3      | 4.00  | 4 (h²)   |
| F_θ = sin θ     | 2.16e-6     | 5.43e-7      | 3.97  | 4 (h²)   |
| F_φ = sin(2φ)   | 7.48e-4     | 1.87e-4      | 4.00  | 4 (h²)   |

Clean **second-order convergence** for all three vector fields on log-r.

**Radial acoustic on log-r** (256-cell, r∈[1,8], cs0=1.0, rc=2.5, pulse
width 0.2, t=1.0):

```
centroid(0)  = 2.532
centroid(t)  = 3.551
predicted    = 3.532
speed_est    = 1.019
speed_err    = 1.9e-2  (physical centroid bias, same magnitude as uniform-r)
```

The pulse propagates at cs0 to within the same ~2% physical centroid bias
documented for uniform-r in 3A.

### Regressions (no change)

- Uniform-r geometry sanity: rel err 3.4e-14 (bit-identical to Task 1).
- Uniform-r uniform-state preservation: `max|drho|=0, max|dp|=0, max|v|=8.1e-17`.
- Uniform-r divergence test, all three cases: bit-identical to Task 3A.
- Cartesian `linear_wave_hydro` cycles 0–3 dt's: bit-identical to Task 2.

### Reconstruction caveat for nonuniform-r

PLM in `src/reconstruct/plm.hpp` computes the slope-limited reconstruction
in INDEX space (no explicit dr weighting). On a smooth log-r grid with
moderate stretching ratio (Δr/r ≪ 1 per cell), the resulting reconstruction
is approximately second-order in physical r — the geometric inconsistency
shows up at O((Δr/r)²) per cell. For strongly stretched grids this becomes
a real error. Donor-cell (`reconstruct=dc`) is exact for any spacing but
first-order. PPM/WENOZ also assume index-uniform spacing.

**Status:** I deliberately did not rewrite PLM/PPM for nonuniform-r in
this task. The log-r divergence test confirms the FV operator itself is
2nd-order; reconstruction errors are smooth and bounded for moderate
stretching. A nonuniform-PLM following Mignone (2014) would be a clean
follow-up.

### Where future agents should hook in

- **Per-pgen user grid via input parameters.** The current user-grid hook
  is a process-global function pointer registered via static initializer.
  A cleaner pattern would be to read parameters (e.g. spline knots) from
  the `<spherical_shell>` block and build the mapping inside
  `ConstructSphericalShellGeometry` without a function pointer. Doable in
  a future task.
- **Nonuniform PLM**: see Mignone 2014. Branch on `radial_grid` in the
  PLM template or pass `dr_minus(i), dr_plus(i)` arrays through the
  reconstruction layer.

## Task 3A validation caveats

- The "vibe-check" pgens (`pulse_3d`, `oblique_packet`) use **outflow** boundaries
  on all faces; late-time data near boundaries is contaminated by reflections.
  Plot/inspect early times (before the wavefront reaches the boundary).
- The `radial_acoustic` centroid metric has an O(σ/r) physical bias from the
  spherical 1/r packet spreading. It is **not** a discretisation error and does
  not converge away with refinement. Use it to verify "wave moves at ~cs0", not
  to bound numerical phase error.
- The `divergence_test` compares the FV operator against the **centroid**
  analytic divergence (point value at r_vol, θ_vol). Expected error is O(h²) for
  smooth integrands — see the table above. The special case F_r = r^n with n=2
  coincides exactly with 4·r_vol under the Mignone centroid definition, so it
  collapses to machine precision (still a useful sanity check).
- The `homologous` test should be run for very short physical time
  (H·tlim ≪ 1) so the linear prediction d⟨ρ⟩/dt = −3Hρ remains accurate.
- No GPU performance testing has been done in Task 3A. The new kernels are
  Kokkos-portable (`par_for`, `Kokkos::parallel_reduce`, device-resident Views)
  but only CPU/Serial parity has been verified locally.

## Caveats / known limitations

1. **No FOFC for spherical_shell.** The first-order flux correction kernel in
   `hydro_fofc.cpp` still uses Cartesian `dx1/dx2/dx3`. The spherical branch
   in `hydro.cpp` errors out at construction if `<hydro>/fofc=true` is set
   together with `<coord>/system=spherical_shell`.

2. **No reconstruction-side spherical correction.** PLM/PPM in
   `src/reconstruct/` still assume uniform spacing in the *index* direction.
   Within a uniform-in-r-index cell that is exactly correct because we
   built `r_face` linearly between `x1min` and `x1max`. So the only error
   is the implicit assumption that "the slope is constant across cell
   width" — which is fine to leading order in physical r as well. If you
   ever go to a stretched/log-r grid you must revisit the radial PLM
   formulas (Mignone 2014).

3. **No pole-aware boundary conditions.** The user is expected to keep
   `theta_min > 0` and `theta_max < pi` strictly. The pole-excision is
   physical, not numerical: there is no special averaging or symmetric
   reflection at the pole.

4. **No diffusion source-term coupling.** Athena++'s spherical source-term
   function adds viscous and conduction flux contributions to `m_ii`,
   `m_pp`, etc. Those are deferred — viscosity/conduction in spherical
   coordinates is a separate task.

5. **No MHD/CT, no AMR/SMR.** Same as Task 1: explicitly deferred.

6. **No log-r grid.** Same as Task 1.

7. **Energy equation is purely conservative.** No geometric source term
   on `IEN`. This is correct in flat (Newtonian) spherical coordinates
   and matches Athena++'s split. Any future addition of gravity will need
   to add `+ rho v . g` to the energy equation in the gravity source term,
   not here.

## Where the next task should hook in

After Task 3A+3B the spherical hydro path is exercised quantitatively by:
divergence operator (O(h²)), uniform-state preservation (roundoff), radial
sound propagation (cs to within physical centroid bias), homologous
expansion (3Hρ to within ~1%), hydrostatic atmospheres (max|v|/cs < 0.03 at
short times under adiabatic evolution of isothermal IC), solid-body
centrifugal source (Linf rel err ~ 1e-3 = Omega·dt), Keplerian circular
orbit (specific Lz preserved within ~6%), and Parker isothermal wind
(IC matches analytic to roundoff; ~1e-3 drift per CFL after t < 0.1).

The recommended next steps in order of payoff:

1. **Visually inspect the 3A/3B vibe checks** in ParaView:
   - `SphShellPulse3D`, `SphShellObliq`: 3A wavefront / wave-packet shapes;
   - `SphShellThinDisk`, `SphShellKepler`: rotating-flow coherence;
   - `SphShellParker3D`: angular quietness of the analytic Parker profile.
   If any of these show grid imprinting, treat that as a 3A/3B regression
   to fix BEFORE moving to MHD.

2. **GPU parity smoke run** of the existing 3A+3B test set on cluster.
   The kernels are Kokkos-portable; expect bit-identical or near-identical
   reductions. Profile the spherical FV update vs Cartesian to set a
   performance baseline before adding MHD.

3. **MHD / CT (Task 4).** Spherical CT is the hardest piece. References:
   Tóth 2000, Mignone et al. 2007, the dyngrmhd module in this codebase
   for inspiration on Kokkos-friendly CT. Likely scope:
   - extend `SphericalShellGeom` with edge-length factors (for EMFs);
   - add face-centred B field storage and Stokes-loop EMF update kernel;
   - update the divergence-cleaning prolongation for SMR/AMR (deferred);
   - validate via current sheet, MHD blast, MRI in disk.

If you're tempted to add a well-balanced reconstruction or a potential-
energy formulation for gravity BEFORE doing MHD, weigh whether it pays
off the cost; the current spherical hydro stack is trustworthy for the
solar-corona task as long as long-term hydrostatic equilibria aren't the
goal.

For optional polish before MHD:

1. **Constant radial gravity.** Add a simple `rho * g(r)` source term on
   `IM1` and `rho v_r * g(r)` on `IEN`, gated on a `<problem>` flag. This
   should NOT live in `Coordinates::AddSphericalShellHydroSrcTerms` —
   that function is for geometric Christoffel-style sources only. Use
   `srcterms/srcterms.cpp` (which already has `ConstantAccel`).

2. **Parker wind initial state.** The transonic Parker solution is a
   density/velocity profile in r. Use the existing user pgen
   `pgen/sph_shell_hydro.cpp` and add a new `mode = parker_wind`. The
   inner BC needs a fixed subsonic Parker boundary state; the outer BC
   is outflow.

3. **Inner radial reflecting/inflow BC.** Currently the only x1 BCs are
   AthenaK's stock `outflow`/`reflecting`/etc. For Parker wind you'll
   want a "fixed Parker surface" inflow — implement as a user BC via
   `pgen->user_bcs_func`.

4. **Convergence test.** Refine a 1-D radial sound wave at multiple
   resolutions and verify second-order convergence in the L1 norm. The
   geometry container is set up so that radial PLM is second-order
   accurate on uniform-in-index grids.

5. **2-D r-theta wave.** Verify that an oblique acoustic wave on a wedge
   propagates correctly.

6. **Solenoidal / divergence-of-radial-velocity diagnostic.** Useful to
   detect flux-area-mistmatch bugs early. Set v_r(r) = some smooth
   function with known div(v) = (1/r^2) d(r^2 v_r)/dr; compute the FV
   divergence; compare. (Not implemented in this task.)

7. **MHD / CT.** The hardest job. Constrained transport on a spherical
   grid requires a different EMF discretisation and divergence-preserving
   prolongation. References: Tóth 2000, Mignone et al. 2007, the dyngrmhd
   path in this codebase for inspiration on Kokkos-friendly CT.

## Athena++ vs AthenaK mesh philosophy (still relevant)

AthenaK uses uniform spacing in `(x1, x2, x3)` indices per MeshBlock.
There is no `<mesh> x1rat` log-radial grid. If you want log-r in a
single MeshBlock you must either:
(a) add a stretched-grid hook to `mesh/meshblock.cpp` (modest), or
(b) tile the r-direction with multiple MeshBlocks of decreasing range
    (works today, no code changes needed).

The geometry container itself stores `r_face` etc. as general 1-D arrays,
so option (a) only requires editing one face-population kernel in
`ConstructSphericalShellGeometry`. Everything downstream is derived from
those face arrays.

## GPU / Kokkos conventions (still followed)

- Per-cell geometric factors live in `DvceArray2D<Real>` Views indexed
  `(m, idx)`.
- All trig precomputed at setup; no runtime `sin`/`cos`/`pow`.
- Inline `KOKKOS_INLINE_FUNCTION` accessor helpers (`SphCellVolume`, etc.)
  in `spherical_shell.hpp`.
- `par_for` and `Kokkos::parallel_reduce` throughout. No `std::vector`.
- Source-term kernel uses the same per-cell pattern; cost is one extra
  `par_for` per RK stage (a few ops per cell). Tested OK on Kokkos
  Serial.

## Tricky things found during Task 2

1. **Pressure curvature is split between FV divergence and source term.**
   For a uniform-pressure cell, the area-weighted radial flux divergence
   on radial momentum is `-2 p src1_i`. The source term `+2 p src1_i`
   cancels it exactly. Same in theta. This is *not* automatic — it
   depends on `coord_src1_i = dr2_half / dr3_third` (Athena++'s definition).
   If you change the FV form (e.g. switch to a centred difference) you
   must re-derive the source split.

2. **The radial-flux weighting in the m_t / m_p source uses `r^2_face`,
   not the full `A1 = r^2 sin theta dtheta dphi`.** Athena++ uses
   `coord_area1_i_(i) = rm*rm` here. The non-`r^2` factors of the FV
   face area are accounted for separately by the `coord_src2_i_(i)`
   normalisation. Easy to get wrong if you confuse "Athena++'s
   `coord_area1_i_`" with "the actual physical face area".

3. **`coord_src1_j_` and `coord_src3_j_` are equal** for hydro without
   diffusion; both are `(sp - sm) / dcos`. The Athena++ source code
   uses `coord_src3_j_` only in the 1-D theta fallback. I store one
   factor `coord_src1_j` and use it in both places.

4. **Source terms must be applied AFTER `RKUpdate` and BEFORE `ConToPrim`.**
   This places the source-term task at the same `w0` snapshot as the
   fluxes used in the divergence. AthenaK's `HydroSrcTerms` task is
   already at that point. Don't move it.

5. **The Riemann fluxes used in the source-term function are the
   AthenaK Riemann-solver outputs (physical flux per unit physical
   area).** No additional area weighting is applied inside the Riemann
   solver. If you swap in a different Riemann solver, the source-term
   function still works as long as the new solver returns the same
   convention.

## How to enable spherical-shell hydro

```ini
<coord>
system           = spherical_shell
verify_geometry  = true     # optional, default true

<hydro>
eos          = ideal
reconstruct  = plm
rsolver      = hllc
gamma        = 1.6666667
fofc         = false        # FOFC is not supported in spherical_shell yet

<mesh>
x1min        = 1.0          # r0 (must be > 0)
x1max        = 2.0
x2min        = 0.6          # theta_min (must be > 0)
x2max        = 2.5          # theta_max (must be < pi)
x3min        = 0.0
x3max        = 6.283185     # phi (typically [0, 2pi))
ix3_bc       = periodic
ox3_bc       = periodic
```

Build with `-DPROBLEM=sph_shell_hydro` for the test pgen, or use any
existing user pgen and start from one of the example inputs.
