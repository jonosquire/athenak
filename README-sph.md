# README-sph: Spherical-shell hydro fork of AthenaK

Practical guide for running and extending the spherical-shell HYDRO solver
in this AthenaK fork. For low-level design notes, gotchas, and historical
task summaries, see `AGENTS.md` in the repo root.

---

## 1. Overview

**Purpose.** This fork adds a specialised logically-Cartesian spherical-polar
shell (`x1=r`, `x2=θ`, `x3=φ`, pole excised) to AthenaK for solar-corona
research. Hydro only. GPU-native via Kokkos. Cartesian behaviour is unchanged.

**What has been added (cumulative, Tasks 1 → 3C):**

| Area                          | Status                                                    |
|-------------------------------|-----------------------------------------------------------|
| Spherical-shell geometry      | volumes, face areas, widths, Mignone centroids            |
| Hydro FV update               | area-weighted flux divergence using `SphericalShellGeom`  |
| Hydro CFL                     | physical widths `(dr, r·dθ, r·sinθ·dφ)`                  |
| Geometric momentum sources    | Athena++ `spherical_polar.cpp::AddCoordTermsDivergence`   |
| Gravity                       | constant-g (existing AthenaK) + 1/r² (new `RInvSqGravity`) |
| Radial grid options           | uniform / log / power_law / user (Task 3C)                |
| Validation suites             | 3A (FV+wave), 3B (gravity+rotation), 3C (log-r)           |

**What is deliberately unsupported.**
- MHD / CT (face-centred B field on a spherical grid)
- AMR / SMR
- FOFC with `spherical_shell` (errored out at construction)
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

## 7. Known limitations / next steps

In rough order of urgency:

1. **No GPU validation yet.** Kernels are Kokkos-portable; needs a CUDA/HIP
   smoke run to confirm parity and set a performance baseline before MHD.
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
5. **FOFC in spherical_shell.** Errored out; the FOFC kernel still uses
   Cartesian widths. Easy to port if needed.
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
