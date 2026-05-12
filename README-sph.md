# README-sph: Spherical-shell hydro fork of AthenaK

Practical guide for running and extending the spherical-shell HYDRO solver
in this AthenaK fork. For low-level design notes, gotchas, and historical
task summaries, see `AGENTS.md` in the repo root.

---

## 1. Overview

**Purpose.** This fork adds a specialised logically-Cartesian spherical-polar
shell (`x1=r`, `x2=θ`, `x3=φ`, pole excised) to AthenaK for solar-corona
research. Hydro **and MHD** with constrained transport. GPU-native via
Kokkos. Cartesian behaviour is unchanged.

**What has been added (cumulative, Tasks 1 → 4):**

| Area                          | Status                                                    |
|-------------------------------|-----------------------------------------------------------|
| Spherical-shell geometry      | volumes, face areas, widths, Mignone centroids, edge lengths |
| Hydro FV update               | area-weighted flux divergence using `SphericalShellGeom`  |
| Hydro CFL                     | physical widths `(dr, r·dθ, r·sinθ·dφ)`                  |
| Geometric momentum sources    | Athena++ `spherical_polar.cpp::AddCoordTermsDivergence`   |
| Gravity                       | constant-g (existing AthenaK) + 1/r² (new `RInvSqGravity`) |
| Radial grid options           | uniform / log / power_law / user (Task 3C)                |
| MHD FV update                 | same area/volume weighting as hydro (Task 4)              |
| MHD CFL                       | spherical widths + fast magnetosonic speed (Task 4)       |
| MHD geometric source terms    | Athena++ MHD path with B² stress (Task 4)                 |
| Constrained transport (CT)    | discrete Stokes loop with edge lengths + face areas (Task 4) |
| divB diagnostic               | face-area-weighted FV form (Task 4)                       |
| Validation suites             | 3A/3B/3C (hydro) + 4 (MHD: monopole, uniform, toroidal)   |

**What is deliberately unsupported.**
- AMR / SMR (CT prolongation/restriction on a spherical grid not implemented)
- FOFC with `spherical_shell` (errored out at construction for both hydro and MHD)
- Pole-aware boundary conditions (user must keep `θ_min > 0`, `θ_max < π`)
- Nonuniform θ or φ grids (only radial nonuniformity supported)
- GPU performance testing (kernels are Kokkos-portable but only CPU/Serial
  parity has been verified)

---

## 2. Build

### Requirements

- CMake ≥ 3.16
- C++17 compiler
- Kokkos (vendored as a git submodule under `kokkos/`)

### Initialize submodules

```bash
git submodule update --init kokkos
```

### Configure and build for CPU / Kokkos Serial

```bash
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DPROBLEM=sph_shell_hydro ..
cmake --build . --target athena -j 4
```

The `-DPROBLEM=sph_shell_hydro` flag selects [src/pgen/sph_shell_hydro.cpp](src/pgen/sph_shell_hydro.cpp)
as the user problem generator. This pgen ships several modes; pick the one
you want via `<problem>/mode` in the input file (see §5).

For Cartesian tests, rebuild with `-DPROBLEM=built_in_pgens` or a specific
test pgen (e.g. `-DPROBLEM=blast`).

### GPU note

The new kernels use only Kokkos-portable patterns (`par_for`,
`Kokkos::parallel_reduce`, device-resident `DvceArray*`). A CUDA/HIP build
should compile, but GPU parity has not been validated in this task and is
the recommended next step.

---

## 3. Coordinate and radial-grid options

### Selecting spherical-shell coordinates

```ini
<coord>
system          = spherical_shell      # required
verify_geometry = true                 # optional; prints wedge-volume sanity at startup

<mesh>
x1min = 1.0                            # inner radius r_min  (must be > 0)
x1max = 10.0                           # outer radius r_max
x2min = 0.3927                         # theta_min  (must be > 0)
x2max = 2.7489                         # theta_max  (must be < pi)
x3min = 0.0                            # phi_min
x3max = 6.2832                         # phi_max    (typically [0, 2pi))
ix3_bc = periodic
ox3_bc = periodic
```

Hard rules:

- **Pole excised.** Pick `x2min > 0` and `x2max < π`. There is no pole BC.
- **r_min > 0.** Required for any spherical-polar grid; required for log-r.
- **No SMR/AMR.** Mesh refinement is not supported.
- **No FOFC** under `spherical_shell` — the build errors out at startup if
  `<hydro>/fofc=true` is combined with this coordinate.

### Radial grid options

Selected in the `<spherical_shell>` input block (a new block added by this
fork):

```ini
<spherical_shell>
radial_grid = uniform        # default; r_face linear in logical x1
# radial_grid = log          # uniform in ln(r)
# radial_grid = power_law    # r_face = r_min + (r_max-r_min) * xi_norm^alpha
# r_grid_alpha = 2.0         # alpha for power_law (default 1.0 = uniform)
# radial_grid = user         # see §3.3 below
```

Formulas for each mode (xi_norm = (x1_logical − mesh.x1min) /
(mesh.x1max − mesh.x1min), in [0, 1] over the active mesh; ghost zones
extrapolate):

- **uniform**: `r_face = mesh.x1min + (mesh.x1max − mesh.x1min) · xi_norm`
- **log**:     `r_face = mesh.x1min · (mesh.x1max / mesh.x1min)^xi_norm`
- **power_law**: `r_face = mesh.x1min + (mesh.x1max − mesh.x1min) · xi_norm^alpha`
  - `alpha = 1.0` recovers uniform.
  - `alpha > 1` packs cells near `r_min`.
  - `alpha < 1` packs cells near `r_max`.

For all four modes:
- Theta and phi are always uniform in their logical coordinates.
- Multi-MeshBlock decomposition is consistent: each block's `(x1min, x1max)`
  is the linear sub-range, and the **same global mapping** is applied to
  every face, so r at block edges matches across blocks.
- Cell-volume, face-area, and source-factor formulas are derived **from the
  r_face array only** (Mignone 2014 centroids), so the spherical FV update,
  CFL, and source-term split work for any monotonic r_face.

### 3.3 User-defined radial grid

If `power_law` and `log` don't fit your problem, register a host-side
mapping function. **Important ordering note**: AthenaK constructs
`Coordinates` *before* the `ProblemGenerator`, so you cannot register the
hook inside `UserProblem()`. You must use a **static initializer** that
runs before `main()`. In your pgen translation unit:

```cpp
#include "coordinates/spherical_shell.hpp"

namespace {
Real MyRadialGrid(Real xi_norm) {
  // xi_norm in [0, 1]; return a strictly increasing physical radius.
  // For ghosts (xi_norm outside [0,1]) the framework extrapolates linearly;
  // you don't need to handle that case.
  // ... your mapping here ...
}

struct RegisterMyGrid {
  RegisterMyGrid() { SetUserRadialGridFunc(&MyRadialGrid); }
};
static RegisterMyGrid g_register_my_grid;
}
```

Then in the input file:

```ini
<spherical_shell>
radial_grid = user
```

The framework calls the function once per face during geometry construction,
then deep-copies the result to device. Ghosts outside `[0, 1]` are linearly
extrapolated automatically. Only one user function may be registered per
process. The build errors out at coordinate construction if `radial_grid=user`
is selected but no function has been registered.

**Status of this hook:** implemented and tested via the negative path
(missing-function fatal error). A positive demo registration is not shipped
to keep the user pgen file uncluttered; users plug it in when needed.

---

## 4. Implemented numerics

### 4.1 Geometry container (`SphericalShellGeom`)

Per-MeshBlock 1-D `DvceArray2D<Real>` views, captured into hot-path kernels:

- `r_face(m, i)`, `r2_face`, `dr`, `dr2_half`, `dr3_third`, `inv_dr3_third`
- `theta_face`, `theta_vol`, `dtheta`, `sin_theta_face`, `sin_theta_vol`, `dcos_theta`
- `phi_face`, `phi_center`, `dphi`
- Source-term factors `coord_src{1,2}_{i,j}` (Athena++ conventions)

All trig is precomputed once at setup. The radial section is reconstructed
from `r_face` only; switching radial grid modes only changes how `r_face`
is filled.

Device-callable accessors (in `spherical_shell.hpp`):

- `SphCellVolume(g, m, k, j, i)`
- `SphFace1Area / SphFace2Area / SphFace3Area`
- `SphPhysWidth1 / SphPhysWidth2 / SphPhysWidth3`

### 4.2 Hydro FV update

In [src/hydro/hydro_update.cpp](src/hydro/hydro_update.cpp), the host branches
once on `coord_system`. The spherical branch computes:

```
divf = (r2_face(i+1)·F1(i+1) − r2_face(i)·F1(i)) / dr3_third
     + dr2_half · (sin_theta_face(j+1)·F2(j+1) − sin_theta_face(j)·F2(j))
                                                 / (dr3_third · dcos_theta)
     + dr2_half · dtheta · (F3(k+1) − F3(k))
                                                 / (dr3_third · dcos_theta · dphi)
```

No coordinate-system branching in the inner kernels.

### 4.3 CFL

In [src/hydro/hydro_newdt.cpp](src/hydro/hydro_newdt.cpp), the per-cell physical
widths are
```
ds1 = dr(m, i)
ds2 = r_vol(m, i) · dtheta(m, j)
ds3 = r_vol(m, i) · sin_theta_vol(m, j) · dphi(m, k)
```
where `r_vol` and `sin_theta_vol` are Mignone volume/sin-weighted centroids.
On log-r these still produce a valid CFL; on power-law they automatically
track the local cell size.

### 4.4 Geometric momentum source terms

`Coordinates::AddSphericalShellHydroSrcTerms()` in
[src/coordinates/spherical_shell.cpp](src/coordinates/spherical_shell.cpp).
Mirrors the HYDRO-only path of
`srcpp/coordinates/spherical_polar.cpp::AddCoordTermsDivergence`. Uses the
precomputed `coord_src{1,2}_{i,j}` factors, the radial Riemann fluxes
`flx1[IM2,IM3]`, and the theta Riemann fluxes `flx2[IM3]` to assemble
flux-weighted Christoffel-style sources. No energy source.

The uniform-state preservation property holds for **any** monotonic radial
grid because the pressure-curvature cancellation between flux divergence
and source terms is local to each cell.

### 4.5 Gravity

Both options live in [src/srcterms/srcterms.cpp](src/srcterms/srcterms.cpp),
toggled in `<hydro_srcterms>`:

- **Constant radial gravity** (existing AthenaK):
  ```ini
  <hydro_srcterms>
  const_accel       = true
  const_accel_val   = -0.5      # g_r = -g0 (negative = inward)
  const_accel_dir   = 1         # x1 = r under spherical_shell
  ```

- **Inverse-square radial gravity** (new):
  ```ini
  <hydro_srcterms>
  r_inv_sq_gravity  = true
  r_inv_sq_gm       = 1.0       # GM value
  ```
  `S(m_r) = −ρ·GM/r²`, `S(E) = −ρ·v_r·GM/r²`, no θ/φ source. Requires
  `<coord>/system=spherical_shell`. Uses `geom.r_vol(m, i)` so it
  automatically tracks the chosen radial grid.

Both gravities use the simple non-potential energy source. Total
gas+kinetic+gravitational energy is therefore not FV-conserved. Acceptable
for the validation tests; a potential-energy formulation may be needed
later for Bondi-/MHD-grade conservation.

### 4.6 Caveats

- **Reconstruction.** PLM in [src/reconstruct/plm.hpp](src/reconstruct/plm.hpp)
  uses cell-centred values with a logical (index-uniform) slope. On a
  smooth log-r grid with moderate stretching (Δr/r ≪ 1 per cell), this is
  approximately second-order in physical r; on a strongly stretched grid
  the reconstruction degrades to first-order in the dr-asymmetry. Donor-
  cell (`reconstruct=dc`) is exact for any spacing. Higher-order PPM/WENOZ
  also assume index-uniform spacing; treat as approximate.
- **No FOFC** for spherical_shell.
- **No pole BC.** Theta boundaries are user-handled.
- **Adiabatic vs isothermal IC drift.** Hydrostatic / Parker IC profiles
  are isothermal but hydro evolves adiabatically (γ=5/3); the IC drifts on
  ~sound-crossing time.

---

## 5. Tests and pgens

All tests live under `inputs/tests/` and use
`-DPROBLEM=sph_shell_hydro`. The pgen modes are selected via
`<problem>/mode`. See AGENTS.md for full quantitative tables.

| Input file                                       | mode               | type           | what it tests                                       |
|--------------------------------------------------|--------------------|----------------|-----------------------------------------------------|
| `sph_shell_geom.athinput`                        | uniform            | quantitative   | wedge volume on uniform-r                            |
| `sph_shell_geom_logr.athinput`                   | uniform            | quantitative   | wedge volume on log-r                                |
| `sph_shell_geom_powerlaw.athinput`               | uniform            | quantitative   | wedge volume on power-law-r                          |
| `sph_shell_hydro_uniform.athinput`               | uniform            | quantitative   | uniform-state preservation (uniform-r)               |
| `sph_shell_hydro_smoke.athinput`                 | uniform            | smoke          | spherical hydro stack stability                      |
| `sph_shell_hydro_sound_pulse.athinput`           | sound_pulse        | quantitative   | radial pulse half-max edge                           |
| `sph_shell_radial_acoustic.athinput`             | radial_acoustic    | quantitative   | radial pulse centroid speed (uniform-r)              |
| `sph_shell_radial_acoustic_logr.athinput`        | radial_acoustic    | quantitative   | radial pulse centroid speed (log-r)                  |
| `sph_shell_divergence.athinput`                  | divergence_test    | quantitative   | FV operator vs analytic for 3 fields (uniform-r)     |
| `sph_shell_divergence_logr.athinput`             | divergence_test    | quantitative   | same, log-r (2nd-order convergence)                  |
| `sph_shell_3d_acoustic_pulse.athinput`           | pulse_3d           | visual / vibe  | 3D wavefront sphericity                              |
| `sph_shell_oblique_wave_packet.athinput`         | oblique_packet     | visual / vibe  | oblique Cartesian wave packet                         |
| `sph_shell_homologous.athinput`                  | homologous         | quantitative   | div(v) = 3H verification                              |
| `sph_shell_hydrostatic_constg.athinput`          | hydrostatic_constg | quantitative   | isothermal HSE in constant g                          |
| `sph_shell_hydrostatic_r2.athinput`              | hydrostatic_r2     | quantitative   | isothermal HSE in 1/r² gravity                        |
| `sph_shell_solid_body_rotation.athinput`         | solid_body_rotation| quantitative   | one-step centrifugal source verification              |
| `sph_shell_keplerian_orbit.athinput`             | keplerian_orbit    | quantitative   | Lz conservation in 1/r² gravity                       |
| `sph_shell_thin_disk_vibe.athinput`              | thin_disk          | visual / vibe  | pressure-supported disk coherence                     |
| `sph_shell_parker_isothermal_1d.athinput`        | parker_isothermal  | quantitative   | Parker wind initialiser / drift                       |
| `sph_shell_parker_isothermal_3d.athinput`        | parker_isothermal  | visual / vibe  | Parker 3D angular quietness                           |

Each finalizer prints L1/L2/Linf or task-specific metrics; see AGENTS.md
for typical CPU/Serial numbers.

---

## 6. Suggested local validation workflow

Before any GPU work, run this sequence on a workstation:

```bash
# 1. build
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DPROBLEM=sph_shell_hydro ..
cmake --build . --target athena -j 4
cd ..

# 2. geometry consistency
./build/src/athena -i inputs/tests/sph_shell_geom.athinput          # uniform-r
./build/src/athena -i inputs/tests/sph_shell_geom_logr.athinput     # log-r
./build/src/athena -i inputs/tests/sph_shell_geom_powerlaw.athinput # power-law

# 3. exact preservation under both grids
./build/src/athena -i inputs/tests/sph_shell_hydro_uniform.athinput
./build/src/athena -i inputs/tests/sph_shell_geom_logr.athinput time/nlim=50

# 4. FV operator convergence (run at two resolutions and check ratio)
./build/src/athena -i inputs/tests/sph_shell_divergence.athinput
./build/src/athena -i inputs/tests/sph_shell_divergence.athinput \
    mesh/nx1=64 mesh/nx2=32 mesh/nx3=32 \
    meshblock/nx1=64 meshblock/nx2=32 meshblock/nx3=32
./build/src/athena -i inputs/tests/sph_shell_divergence_logr.athinput
./build/src/athena -i inputs/tests/sph_shell_divergence_logr.athinput \
    mesh/nx1=128 mesh/nx2=64 mesh/nx3=32 \
    meshblock/nx1=128 meshblock/nx2=64 meshblock/nx3=32

# 5. radial acoustic propagation
./build/src/athena -i inputs/tests/sph_shell_radial_acoustic.athinput
./build/src/athena -i inputs/tests/sph_shell_radial_acoustic_logr.athinput

# 6. gravity / rotation / Parker (3B suite)
./build/src/athena -i inputs/tests/sph_shell_hydrostatic_constg.athinput
./build/src/athena -i inputs/tests/sph_shell_hydrostatic_r2.athinput
./build/src/athena -i inputs/tests/sph_shell_solid_body_rotation.athinput
./build/src/athena -i inputs/tests/sph_shell_keplerian_orbit.athinput
./build/src/athena -i inputs/tests/sph_shell_parker_isothermal_1d.athinput

# 7. vibe checks (open VTK in ParaView; see AGENTS.md for plot tips)
./build/src/athena -i inputs/tests/sph_shell_3d_acoustic_pulse.athinput
./build/src/athena -i inputs/tests/sph_shell_oblique_wave_packet.athinput
./build/src/athena -i inputs/tests/sph_shell_thin_disk_vibe.athinput
./build/src/athena -i inputs/tests/sph_shell_parker_isothermal_3d.athinput
```

Expected outcomes:
- All geometry tests print `relative error ~ 1e-14` (machine precision).
- Uniform-state preservation: `max|drho|=0, max|dp|=0, max|v|~1e-17`.
- Log-r divergence diagnostic: L1 should fall by factor ~4 when h is halved.
- Cartesian regression (`-DPROBLEM=blast` or `built_in_pgens`): cycle-by-cycle dt's
  should be bit-identical to baseline.

---

## 6b. MHD / constrained transport (Task 4)

Spherical-shell MHD with constrained transport is implemented in:

- `src/mhd/mhd_update.cpp`, `mhd_ct.cpp`, `mhd_newdt.cpp` — host-level branch on
  `coord_system`, identical Cartesian path preserved; spherical branch uses
  `SphericalShellGeom` face areas, cell volumes, and (for CT) edge lengths.
- `src/coordinates/spherical_shell.cpp::Coordinates::AddSphericalShellMHDSrcTerms` —
  geometric MHD source terms (Athena++
  `srcpp/coordinates/spherical_polar.cpp::AddCoordTermsDivergence` MHD path).
- `src/pgen/sph_shell_mhd.cpp` — MHD pgen with `monopole` / `uniform_static` /
  `toroidal_static` modes and a face-area-weighted `divB` diagnostic that uses
  the SAME `SphericalShellGeom` factors as CT.

### What is implemented

- Spherical FV update for cell-centred conserved variables (rho, momenta,
  energy, scalars). Same area/volume weighting as hydro.
- CT update for face-centred B fields via the discrete Stokes loop:
  ```
  dB1/dt = -(1/A1) [L3+ E3+ - L3- E3-] + (1/A1) [L2+ E2+ - L2- E2-]
  dB2/dt = +(1/A2) [L3+ E3+ - L3- E3-] - (1/A2) [L1+ E1+ - L1- E1-]
  dB3/dt = -(1/A3) [L2+ E2+ - L2- E2-] + (1/A3) [L1+ E1+ - L1- E1-]
  ```
  with face areas A_i and edge lengths L_i from `SphericalShellGeom`.
- MHD CFL using spherical physical widths (`dr, r·dθ, r·sin θ·dφ`) and the
  existing MHD fast-magnetosonic speed.
- Geometric source terms extending the hydro form with magnetic-stress pieces:
  ```
  m_ii_MHD = ρ(v_θ² + v_φ²) + 2p + B_r²
  m_pp_MHD = ρ·v_φ² + p + ½(B_r² + B_θ² - B_φ²)
  m_ph_MHD = ρ·v_θ·v_φ - B_θ·B_φ          (1-D-theta fallback only)
  ```
  Radial- and theta-flux pairings use the MHD Riemann fluxes (which already
  include `-B_i B_j` magnetic stress). No `IEN` geometric source (matches
  Athena++ split).
- `divB` diagnostic in the MHD pgen using the same face-area FV form:
  ```
  divB = (1/V) [A1+ B1+ - A1- B1- + A2+ B2+ - A2- B2- + A3+ B3+ - A3- B3-]
  ```

### Edge-length / face-area conventions

Match Athena++ `srcpp/coordinates/spherical_polar.cpp` exactly:

| Quantity              | Formula                                         |
|-----------------------|-------------------------------------------------|
| A1 (radial face area) | `r_face^2 * (cos θ_- - cos θ_+) * dφ`           |
| A2 (theta face area)  | `½(r_+² - r_-²) * sin(θ_face) * dφ`             |
| A3 (phi face area)    | `½(r_+² - r_-²) * dθ`                            |
| L1 (radial edge)      | `dr`                                             |
| L2 (theta edge)       | `r_face * dθ`                                    |
| L3 (phi edge)         | `r_face * sin(θ_face) * dφ`                      |

`L1` depends only on `i`; `L2` on `(i, j)`; `L3` on `(i, j, k)`. The CT kernel
factors out edge lengths that don't vary across the relevant difference.

### MHD tests

| Input file                                       | mode / flavor      | what it tests                                      |
|--------------------------------------------------|--------------------|----------------------------------------------------|
| `sph_shell_mhd_uniform_static.athinput`          | uniform_static     | MHD code path with B=0: should be exact            |
| `sph_shell_mhd_monopole.athinput`                | monopole           | divB preservation for radial 1/r² field (uniform-r) |
| `sph_shell_mhd_monopole_logr.athinput`           | monopole           | same, log-r grid                                    |
| `sph_shell_mhd_toroidal_static.athinput`         | toroidal_static    | divB preservation for axisymmetric B_φ              |
| `sph_shell_mhd_radial_alfven.athinput`           | radial_alfven      | outgoing axisym Alfvén pulse on 1/r² monopole; WKB centroid + divB |
| `sph_shell_mhd_radial_alfven_vibe.athinput`      | radial_alfven      | high-res WKB vibe: nx1=1024, carrier k_r=60, r∈[1,10], VTK dumps |
| `sph_shell_mhd_loop_eq.athinput`                 | loop_eq / axisym   | axisymmetric A_φ Gaussian loop in (r,θ), v=0; CT field preservation |
| `sph_shell_mhd_loop_eq_vibe.athinput`            | loop_eq / advect   | non-axisym A_θ loop in (r,φ), solid-body Ω=0.25; VTK rotation vibe |

Run with `-DPROBLEM=sph_shell_mhd`. Each test reports:

- `divB` L1, L2, Linf and a normalised `Linf · h / |B|max` metric.
- Max `|v|/cs_iso` (should remain near roundoff for symmetric ICs; small
  residual on the monopole due to the discrete force-free split).
- Max `|B1f − analytic|` for the monopole, verifying CT preserves the field.

### Quantitative CPU/Serial results

Single block:

```
monopole (uniform-r) nlim=20:  divB L1=1.2e-15, Linf=1.7e-14, max|B1f-analytic|=1.1e-16
monopole (log-r)     nlim=20:  divB L1=5.9e-16, Linf=1.2e-14, max|B1f-analytic|=1.1e-16
toroidal_static      nlim=10:  divB L1=1.2e-17, Linf=9.3e-17
uniform_static       nlim=20:  divB=0, max|v|=4.9e-17
```

Multi-block (4×2×2 = 16 blocks):

```
monopole (uniform-r) nlim=20:  divB L1=1.2e-15, Linf=1.5e-14, max|B1f-analytic|=1.1e-16
monopole (log-r)     nlim=20:  divB L1=5.7e-16, Linf=1.7e-14, max|B1f-analytic|=2.2e-16
```

CT preserves divB at machine precision across MeshBlock boundaries on both
uniform-r and log-r grids.

The ~4% residual `max|v|/cs` on the radial monopole is **not** a CT bug. It
comes from the discrete spherical FV / source-term split: cell-centred B²
in the curvature source does not exactly cancel face-centred B² in the
flux divergence for a 1/r² monopole. The same residual is present in
Athena++. `divB` itself remains at machine precision throughout.

**Radial Alfvén (`radial_alfven`)**, standard test (`tlim=5`, σ_r=0.3, nx1=256, r ∈ [1, 8]):
```
<r>_|dB|^2 = 2.200    r_WKB = 2.216   centroid err = -1.5e-2 (-2% of distance)
peak |dB_phi|/(eps*B0) = 0.93     peak |dv_phi| = 4.4e-4
divB L1 = 3.3e-15
```
Pulse propagates outward following the WKB ray to ~2%; divB at machine
precision. Amplitude drop ~7% is PLM dissipation. The vibe variant
(`*_vibe.athinput`) runs the same setup on r ∈ [1, 10] with nx1=1024 and
a carrier wavelength λ ~ 0.1 for high-resolution geometric-optics
inspection via VTK output.

**Field loop (`loop_eq`)**, axisymmetric standard test (`v=0`, 50 cycles,
nx1=nx2=64):
```
max|Br|      = 4.02e-3  (preserved to 13+ digits; max|Bphi| ~ 1e-18)
max|Btheta|  = 4.31e-3
0.5*B^2*V    = 6.26e-7   (preserved to 8 digits)
divB L1      = 6.2e-18   Linf = 4.6e-16
```
CT keeps the closed poloidal loop static to machine precision under v=0.

Vibe variant (`*_vibe.athinput`) uses a non-axisymmetric A_θ loop in (r, φ)
on a full 2π annulus with solid-body rotation Ω=0.25. At quarter rotation
the loop has translated 1.57 rad in φ as predicted; magnetic energy
decays from numerical PLM dissipation (factor ~3.5 at quarter period at
nx3=128 resolution -- ramps up with resolution). divB stays at 2e-17.
Plot `bcc1` and `bcc3` in r-φ slices from VTK to see the loop rotate.

### MHD known limitations

- **No GPU validation.** Kernels are Kokkos-portable but only CPU/Serial
  parity is verified here.
- **FOFC unsupported.** `<mhd>/fofc=true` errors out in spherical_shell.
- **AMR/SMR unsupported.** Prolongation/restriction of face-centred B with
  spherical CT geometry is non-trivial; deferred.
- **No pole boundary handling.** User must keep θ strictly in (0, π).
- **PLM/PPM reconstruction is index-uniform.** Same caveat as for hydro
  on log-r.
- **Monopole residual velocity ~4% over many cycles** is a known property
  of the discrete force-free split, not a CT problem. Same in Athena++.

---

## 7. Known limitations / next steps

In rough order of urgency:

1. **No GPU validation yet.** Kernels are Kokkos-portable (both hydro and
   MHD/CT); needs a CUDA/HIP smoke run to confirm parity and set a
   performance baseline.
2. **Reconstruction is index-uniform** for nonuniform-r. Acceptable for
   moderate log-r stretching; document for strong stretching. A nonuniform-
   PLM (Mignone 2014) would be a clean follow-up.
3. **User radial grid hook** is implemented as a static-initializer
   function-pointer registry; no example registration is shipped. Users
   plug in their own. A future cleanup could add per-pgen-block grid
   parameters so the hook can be configured in the input file.
4. **Pole boundary conditions.** Not implemented. User must keep
   θ_min > 0, θ_max < π. Adding pole-aware BCs is required for full-sphere
   simulations.
5. **FOFC in spherical_shell.** Errored out for both hydro and MHD; the
   FOFC kernel still uses Cartesian widths. Easy to port if needed.
6. **MHD AMR/SMR.** Constrained transport with prolongation/restriction of
   face-centred B on a spherical grid is non-trivial; deferred.
7. **Larger MHD vibe tests.** Radial Alfvén wave, MHD blast in spherical,
   MHD current sheet on a wedge -- next round of validation after GPU parity.
6. **AMR/SMR.** Not implemented.
7. **MHD / CT.** The big one. Spherical CT requires face-centred B field
   storage and a Stokes-loop EMF update. References:
   - Tóth 2000
   - Mignone et al. 2007
   - the dyngrmhd module in this codebase for Kokkos-friendly CT patterns
8. **Energy source for gravity** is non-potential. A potential-energy
   formulation would conserve total energy at the FV level.

The recommended workflow before MHD:

1. **Visually inspect** the vibe-check VTK outputs from §6 — confirm there
   are no spurious grid imprintings.
2. **GPU parity smoke run** on cluster (CUDA or HIP). Bit-identical or
   near-identical reductions on a small wedge.
3. **Start MHD/CT** (Task 4).

For the latest design notes and per-task results, see `AGENTS.md`.
