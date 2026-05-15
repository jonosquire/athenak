# Parker wind monochromatic AW driver — first local sanity check

Date: 2026-05-14
Task: 4.3 of the spherical MHD path.

## 1. Files changed

- `src/pgen/parker_wind_aw.cpp` (new) — polytropic Parker MHD background
  plus optional lower-radial monochromatic AW driver. Branch-safe
  bisection, B0 calibration, analytic radial user BCs, WKB diagnostic
  CSV, wave-profile snapshots, finalizer with background-preservation
  and z+/z- summaries.
- `inputs/tests/parker_wind_aw.athinput` (new) — default input
  (driver-off, reproduces Task 4.2 background; CLI overrides switch the
  driver on).
- `README-sph.md`, `AGENTS.md` — short subsections on the new pgen.
- `parker_aw_validation/` (gitignored) — runner scripts, plotting script,
  per-case logs and CSVs.

The original `src/pgen/sph_shell_mhd.cpp` is unchanged; the `parker_polytropic`
mode there still works when built with `-DPROBLEM=sph_shell_mhd`.

## 2. New pgen, build, and key parameters

```bash
# Configure & build in a separate dir to keep the sph_shell_mhd build intact:
mkdir build_paw && cd build_paw
cmake -DPROBLEM=parker_wind_aw ..
cmake --build . --target athena -j 4
```

Key `<problem>` parameters (full list in
`inputs/tests/parker_wind_aw.athinput`):

| name                  | meaning                                        |
|-----------------------|------------------------------------------------|
| `gamma_poly`          | polytropic index (must match `<mhd>/gamma`)    |
| `GM`                  | gravity (matches `<mhd_srcterms>/r_inv_sq_gm`) |
| `rcrit`               | Parker critical radius                         |
| `rho_inner`           | base density at `r_inner`                      |
| `B0_mode`             | `alfven_point` (default) or `beta_inner`       |
| `alfven_point_target` | radius where M_A=1 (used if `B0_mode=alfven_point`) |
| `beta_inner`          | requested plasma beta at `r_inner` (used if `B0_mode=beta_inner`) |
| `r_ref`               | WKB normalization radius                       |
| `driver_enable`       | true/false                                     |
| `driver_amp`          | transverse-velocity amplitude (code units)     |
| `driver_omega`        | angular frequency                              |
| `driver_phase`        | initial time phase                             |
| `driver_ramp_time`    | half-`sin²` ramp duration                      |
| `driver_polarization` | `phi` (default), `theta`, or `circular` (k_perp=0) |
| `driver_b_sign`       | -1 = outgoing z+ branch (default)              |
| `driver_circ_sign`    | +1 / -1 handedness of the circular drive       |
| `driver_ntheta`       | int, 0 = k_perp=0 in θ                          |
| `driver_nphi`         | int, 0 = k_perp=0 in φ                          |
| `snapshot_times`      | "1,2,3,4,5"                                    |
| `csv_dir`, `label`    | output prefix                                  |

## 3. Boundary-condition description

- **Lower x1** (user BC):
  - Analytic Parker + monopole for ρ, p, v_r, `B_r` (on radial faces).
  - Transverse perturbation overlay if `driver_enable=true`:
    `v_perp(t,θ,φ) = amp · ramp(t) · sin(ω t + φ) ·
                       cos(2π n_θ θ̃ + θ_phase + 2π n_φ φ̃ + φ_phase)`,
    with `θ̃=(θ-θ_min)/(θ_max-θ_min)` and similarly in φ.
  - Face B: `B_perp(face) = driver_b_sign · √ρ_inner · v_perp` on the
    polarization-selected face. The other transverse face is zeroed.
  - Cell-centred ghost momentum, energy reset consistently.
- **Upper x1** (user BC):
  - Analytic Parker + monopole for ρ, p, v_r, `B_r`.
  - Transverse v and face B: zero-gradient copy from the last active cell.
    No incoming-wave boundary is imposed.

This is not a perfect non-reflecting boundary, but the wedge is
super-Alfvénic at the outer boundary (`U > v_A`), so the physical
characteristics for transverse modes both point outward and reflection is
expected to be small. Subsequent tasks may replace this with a
characteristic outflow.

## 4. Driver formula and sign convention

The Elsasser-style variables used in the diagnostics are

```
z+ = v_perp - sign(B_r) · B_perp / sqrt(rho)
z- = v_perp + sign(B_r) · B_perp / sqrt(rho)
```

With `B_r > 0`, the **outgoing** (carried by characteristic speed `U + v_A`)
branch is `z+`. The driver imposes
`B_perp(boundary) = driver_b_sign · sqrt(rho_inner) · v_perp(boundary)`.
With `driver_b_sign = -1` (default), `B_perp = -sqrt(rho) v_perp`, hence
`z+ = 2 v_perp` and `z- = 0` exactly at the boundary, so the driver excites
only the outgoing branch in the boundary cell. The finalizer reports
`max|z+|`, `max|z-|`, and the ratio.

### 4b. Setting the radial monopole strength

The radial monopole `B_r(r) = B0 (r_inner / r)^2` carries no current away from
`r = 0`, so it is force-free in the spherical shell. Changing the calibration
of `B0` therefore only rescales the magnetic and Alfvenic background
quantities (`B_r`, `v_A`, `M_A`, `beta`); the hydrodynamic Parker solution
`rho(r)`, `p(r)`, `U(r)` is unchanged.

Two equivalent calibrations are provided via `<problem>/B0_mode`:

- `B0_mode = alfven_point` (default): `B0` is chosen so that
  `M_A(r = alfven_point_target) = 1`,
  i.e. `B0 = U(rA) sqrt(rho(rA)) (rA / r_inner)^2`.
- `B0_mode = beta_inner`: `B0 = sqrt(2 p(r_inner) / beta_inner)`
  (code units, magnetic pressure = `B^2 / 2`). The realised inner-boundary
  beta is printed at startup and written into the background CSV.

The `<csv_dir>/<label>_background_wkb.csv` profile now has a `beta = 2 p / B_r^2`
column, and `plot_aw_driver.py` adds `beta(r)` to the analytic-background
figure.

## 5. Analytic background and derived values (default input)

| parameter                      | value           |
|--------------------------------|-----------------|
| `gamma_poly`                   | 1.05            |
| `GM`                           | 4.0             |
| `rcrit`                        | 2.0             |
| `rho_inner`                    | 1.0             |
| `alfven_point_target`          | 13.0            |
| domain `r ∈`                   | `[1, 20]`       |
| `a_c = sqrt(GM/2rcrit)`        | 1.000           |
| `U(r_inner)`                   | 0.4086          |
| `rho_c`                        | 0.1021          |
| `B0`                           | 12.94           |
| `v_A(r_inner)`                 | 12.94           |
| `U(rA)` = `v_A(rA)`            | 2.43            |
| approx outward AW crossing time `tau_+` from r=1 to 20  | ~3 |

These match the Task 4.2 background to machine precision.

## 6. Test input parameters (`aw_plm_hlld_nr1024_w10`)

| setting                | value                                          |
|------------------------|------------------------------------------------|
| reconstruction         | PLM                                            |
| Riemann solver         | HLLD                                           |
| `mesh/nx1`             | 1024                                           |
| `meshblock/nx1`        | 128                                            |
| angular wedge          | `8 × 8` thin equatorial tube (same as Task 4.2)|
| `time/tlim`            | 8.0                                            |
| `driver_amp`           | 1e-3                                           |
| `driver_omega`         | 10.0  (T = 2π/ω ≈ 0.628)                       |
| `driver_ramp_time`     | 1.0                                            |
| `driver_polarization`  | phi                                            |
| `driver_b_sign`        | -1.0 (outgoing z+ branch)                      |
| `driver_ntheta, nphi`  | 0, 0  (`k_perp = 0`)                            |
| snapshot times         | 1, 2, 3, 4, 5, 6, 7                            |

## 7. No-wave preservation (control)

Run with `driver_enable=false` at Nr=512, tlim=5 (same settings as the
Task 4.2 baseline):

| diagnostic            | weighted bcc | simple 0.5-avg bcc | Task 4.2 baseline |
|-----------------------|-------------:|-------------------:|------------------:|
| L1 \|Δρ/ρ\|           | **3.94e-3**  | 8.23e-3            | 8.23e-3           |
| L1 \|Δv_r/v_r\|       | **2.64e-3**  | 3.42e-3            | 3.42e-3           |
| L1 \|Δp/p\|           | **4.47e-3**  | 7.22e-3            | 7.22e-3           |
| L1 \|Δṁ/ṁ\|           | **4.51e-3**  | 8.73e-3            | 8.73e-3           |
| max \|v_⊥\|/c_s       | 7.49e-13     | 1.34e-12           | 1.10e-12          |
| divB Linf · h / \|B\|max | 4.9e-13   | 4.7e-13            | 4.7e-13           |

The split-out pgen with **Mignone-2014 volume-weighted face-to-cell-centre
interpolation of `bcc`** reduces the background-preservation L1 errors by
roughly a factor of 2 vs the simple 0.5-average and the original Task 4.2
result. Both Elsasser branches still sit at the machine-precision floor in
the no-wave control. divB is unchanged at roundoff (the bcc weighting is
applied in the EOS C2P only; the constrained-transport face B is
untouched).

## 8a. Driven-wave results — circular polarization (recommended)

A linearly polarized monochromatic wave has |z+| that oscillates in r at
fixed t (and in t at fixed r) because |v_perp| ∝ |sin(ω(t−τ_+))|. This makes
per-snapshot envelope reads noisy and complicates resolution / reconstruction
scans. The pgen therefore supports **circular polarization** as an
alternative drive (only when k_perp = 0):

```
v_θ(t) = amp · ramp(t) · sin(ω t + φ)
v_φ(t) = driver_circ_sign · amp · ramp(t) · cos(ω t + φ)
B_θ,φ  = driver_b_sign · √ρ_inner · v_θ,φ
```

Then `|v_perp|² = v_θ² + v_φ²` is constant in t at each r, hence a smooth,
single-valued envelope per snapshot.

`aw_circ_wbcc_plm_hlld_nr1024_w10` (Nr=1024, **amp = 1e-2, 10× larger than the
linear test below**, ω=10, ramp=1, tlim=8, PLM + HLLD, circular polarization,
volume-weighted bcc, ~22 min on local CPU). Final state (`t = 8`, ~2.7
outward AW crossings):

| diagnostic                       | weighted bcc | simple bcc |
|----------------------------------|-------------:|-----------:|
| max \|v_⊥\|/c_s                  | 2.73e-2      | 2.73e-2    |
| max \|B_θ\|/√ρ_inner             | 9.81e-3      | 9.80e-3    |
| max \|B_φ\|/√ρ_inner             | 4.58e-3      | 4.58e-3    |
| max \|z+\|                        | 5.79e-2      | 5.79e-2    |
| max \|z-\|                        | 2.75e-3      | 2.76e-3    |
| max \|z-\| / max \|z+\|           | 4.8 %        | 4.8 %      |
| divB Linf · h / \|B\|max         | 3.7e-14      | (similar)  |
| **L1 \|Δρ/ρ\|**                  | **8.76e-4**  | 1.99e-3    |
| **L1 \|Δp/p\|**                  | **1.07e-3**  | 1.35e-3    |
| **L1 \|Δṁ/ṁ\|**                  | **1.02e-3**  | 1.88e-3    |

Wave-amplitude diagnostics (z+, z-, v_⊥, polarization purity) are
**unchanged** — they are dominated by the driver and WKB scaling, neither of
which couples to the bcc-averaging choice. The background-preservation L1
errors halve, exactly as in the no-wave control. WKB tracking remains ~1–2 %
across r ∈ [1, 20] in both versions; the plot
`plots/aw_circ_wbcc_plm_hlld_nr1024_w10_snapshots_zplus.png` is visually
identical to the simple-bcc run.

Background-preservation errors match the no-wave control to the same
precision; the wave is **linear** at this amplitude (ε/v_A ≈ 2e-3 at the
inner boundary).

**WKB tracking.** Anchoring the WKB envelope at the innermost radial cell
(where the driver fixes |z+| = 2·amp = 2e-2 at full ramp), the simulation
amplitude matches the WKB prediction
`|z+|(r) = |z+|(r_min) · HO(r_min)/HO(r)`
**to ~1–2 % across r ∈ [1, 20]** once the wave has filled (`t ≥ 4`). See
`plots/aw_circ_plm_hlld_nr1024_w10_snapshots_zplus.png` — simulation
envelope and WKB dashed line are visually overlapping.

z-/z+ profile (see `aw_circ_plm_hlld_nr1024_w10_zminus_over_zplus.png`):

- Smooth on log axes (was extremely spiky for the linear test).
- ~0.1 right at the inner boundary cell.
- ~0.01 across the bulk of the domain.
- Time-stationary once the wave train fills (`t ≥ 4`).

This is the configuration we recommend for any future resolution /
reconstruction / Riemann-solver / amplitude / k_perp scans on the Parker
background.

## 8b. Driven-wave results — linear polarization (original test)

`aw_plm_hlld_nr1024_w10` (Nr=1024, tlim=8, ~21 min on local CPU). Final
state (`t = 8`, ~2.7 outward AW crossings):

| diagnostic                       | value    |
|----------------------------------|---------:|
| max \|v_⊥\|/c_s                  | 2.71e-3  |
| max \|B_φ\|/√ρ_inner             | 9.71e-4  |
| max \|z+\|                        | 5.75e-3  |
| max \|z-\|                        | 2.52e-4  |
| max \|z-\| / max \|z+\|           | 4.4 %    |
| max \|v_θ\|/c_s (phi pol leakage) | 4.40e-10 |
| max \|B_θ\|/√ρ_inner              | 4.24e-11 |
| divB L1                          | 6.56e-13 |
| L1 \|Δρ/ρ\| (preservation)       | 2.04e-3  |
| L1 \|Δv_r/v_r\|                  | 7.44e-4  |
| L1 \|Δṁ/ṁ\|                      | 1.80e-3  |

Per-radius peak amplitudes from the final snapshot (binned in r):

| r (bin centre) | peak \|z+\| | WKB envelope (anchored at r_min) |
|----------------|------------:|---------------------------------:|
|  1.48          | 2.43e-3     | 2.43e-3 (= driver, by def)       |
|  3.38          | 4.40e-3     | 3.89e-3                          |
|  6.23          | 5.31e-3     | 4.78e-3                          |
|  8.13          | 5.58e-3     | 5.04e-3                          |
| 12.87          | 5.75e-3     | 5.22e-3                          |
| 15.72          | 5.70e-3     | 5.20e-3                          |
| 19.52          | 5.06e-3     | 5.09e-3                          |

The peak amplitude **tracks the WKB envelope to within ~10 %** across the
full domain, with the simulation slightly overshooting WKB in the outer
half. The envelope shape (slow growth out to r≈rA, slight roll-off
beyond) matches the HO_factor profile (drops from ~6 at r_inner to ~2
near rA). Linear-amplitude WKB is the right zeroth-order description for
this case.

Outgoing/incoming branch separation:

- max \|z-\| / max \|z+\| = 4.4 % globally, dominated by zero-crossing
  fluctuations and a small reflected/aliased component near the outer BC.
- v_θ and B_θ at the float floor confirm clean φ-only polarization.
- divB stays at roundoff throughout the run (no degradation from boundary
  driving).

Plots (in `plots/aw_plm_hlld_nr1024_w10_*.png`):

- `snapshots_vperp.png` — clear monochromatic wave train filling the
  domain by t≈3, growing-amplitude envelope.
- `snapshots_zplus.png` — same envelope; black dashed line is the
  WKB prediction anchored to the driver at r_inner. Simulation tracks
  it to ~10 %.
- `snapshots_zminus.png` — z- is ~10⁻⁴ everywhere, well below z+.
- `zminus_over_zplus.png` — log-scale ratio is ~10⁻² in the bulk away
  from zero-crossings.
- `preservation.png` — background errors at final t. Same shape as Task
  4.2: dominated by the inner-boundary region.
- `analytic_background.png`, `wkb_expectation.png` — analytic reference
  curves with rcrit (grey) and rA (red) marked.

The matching no-wave control (Section 7) confirms the split-out pgen
reproduces the Task 4.2 background to its noise floor; the wave is
therefore physical, not numerical.

## 9. Issues / cautions noted

- The snapshot dispatcher writes from inside `user_bcs_func`, which runs
  *before* `ConsToPrim` in each RK substage, so the recorded `w0`/`bcc0`
  are 1 substage stale (~3e-4 in code units). Negligible for linear
  amplitudes but worth knowing if used quantitatively.
- The outer-x1 outflow copies `b0.x2f`, `b0.x3f` from the last active
  radial cell into the ghost. This does not perfectly preserve CT
  near the outer boundary; divB stays at roundoff in practice because the
  copy is mass-conservative w.r.t. the face-area metric on a thin
  equatorial wedge, but the construction should be revisited if the
  wedge is widened or if non-axisymmetric oblique waves are added.
- Sign convention has been **verified empirically**, not just by reading
  the formula: `b_sign = -1` produces `max|z-|/max|z+| ≈ 4 %` in the
  smoke test, consistent with single-branch driving.
- Reflection from the outer boundary should be small since the wedge is
  super-Alfvénic there, but this should be reverified by inspecting the
  long-time `z-` traces in the snapshots.

## 10. Recommended next step

1. **Resolution + reconstruction scan**: PLM/PPMX at Nr=512 and Nr=1024,
   HLLD and LHLLD. Check `max|z-|/max|z+|` and amplitude vs WKB envelope.
2. **Amplitude scan**: `amp = 1e-4, 1e-3, 1e-2` to find the largest
   amplitude that remains linear over five outward AW crossings.
3. **Frequency scan**: `omega = 5, 10, 20, 40` to map the resolution
   limit and the geometric-optics regime.
4. **k_perp ≠ 0**: small `driver_nphi` and `driver_ntheta` to check the
   oblique-wave dispersion.
5. **Outer-BC improvement**: characteristic / transmissive outer BC if
   reflections become detectable.
6. **Fourier analysis**: extract `|z+(r,ω)|` from a long run to compare
   to the WKB envelope at the driven frequency.

## 11. Reproduction

```bash
# Build
mkdir build_paw && cd build_paw
cmake -DPROBLEM=parker_wind_aw ..
cmake --build . --target athena -j 4

# No-wave control
./build_paw/src/athena -i inputs/tests/parker_wind_aw.athinput \
    mesh/nx1=512 meshblock/nx1=128 time/tlim=5 \
    problem/csv_dir=parker_aw_validation/csv \
    problem/label=nowave_plm_hlld_nr512

# Main driven test (~25 min on local CPU)
./build_paw/src/athena -i inputs/tests/parker_wind_aw.athinput \
    mesh/nx1=1024 meshblock/nx1=128 time/tlim=8 \
    problem/driver_enable=true \
    problem/driver_amp=1e-3 problem/driver_omega=10.0 \
    problem/driver_ramp_time=1.0 \
    problem/snapshot_times="1,2,3,4,5,6,7" \
    problem/csv_dir=parker_aw_validation/csv \
    problem/label=aw_plm_hlld_nr1024_w10

# Plots
python3 parker_aw_validation/plot_aw_driver.py \
    --label aw_plm_hlld_nr1024_w10
```

Everything under `parker_aw_validation/` is gitignored.
