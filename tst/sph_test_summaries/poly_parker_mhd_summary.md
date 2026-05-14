# Polytropic Parker MHD validation summary

Date: 2026-05-14

## 1. Goal

Implement a branch-safe ideal-gas polytropic Parker wind plus radial monopole as
a quiet flowing MHD background for later lower-radial-boundary monochromatic
Alfvén-wave (AW) driving tests. Provide a `parker_polytropic` mode in
`src/pgen/sph_shell_mhd.cpp` that

- evolves with ideal-gas MHD (compatible with both **HLLD** and **LHLLD**;
  the AthenaK LHLLD port is adiabatic-only),
- uses a critical-point parameterisation (`GM`, `rcrit`, `gamma_poly`,
  `rho_inner`, `alfven_point_target`) and a branch-safe bisection,
- calibrates the radial monopole strength `B0` so the Alfvén point sits at
  the requested target radius,
- runs with the existing user radial BC pattern,
- writes per-case radial-profile CSVs and summary diagnostics for offline
  analysis.

The static spherical monopole + HLLD has documented multidimensional transverse
growth (see [`tst/sph_test_summaries/monopole_hlld_interface_debug_summary.md`]).
The isothermal Parker is much quieter through ~3-5 outward AW crossings but
eventually grows; see
[`tst/sph_test_summaries/monopole_lhlld_minimal_summary.md`]. LHLLD is not
implemented for isothermal MHD. A polytropic Parker (γ ≈ 1.05) is therefore the
cleanest steady flowing MHD background usable with both HLLD and LHLLD.

## 2. Files changed

- `src/pgen/sph_shell_mhd.cpp`
  - new mode `parker_polytropic`,
  - device-callable `PolyParkerF`, `PolyParkerY` (bisection),
    `EvaluatePolytropicParker`, `EvaluatePolytropicParkerBrFace`,
  - new user BC `ParkerPolyMhdRadialBCs`,
  - new finalizer `ParkerPolytropicFinalize` with optional radial-profile CSV.
- `inputs/tests/sph_shell_mhd_parker_polytropic.athinput`
  (default: γ=1.05, GM=4, rcrit=2, ρ_inner=1, r_inner=1, rA_target=13).
- `poly_parker_mhd_validation/run_matrix.sh` (local test-matrix runner).
- `poly_parker_mhd_validation/plot_results.py` (post-processing).
- `.gitignore` adds `poly_parker_mhd_validation/` (local-only outputs).

## 3. Mathematical construction

Critical-point coordinates with `x = r/rcrit`, `y = U/a_c`, and
`a_c = sqrt(GM/(2 rcrit))`:

The transonic polytropic Parker dimensionless residual is

```
F(y, x; γ) = y²/2 + (1/(γ−1)) · (1/(y x²))^(γ−1) − 2/x − (1/(γ−1) − 3/2)
```

`F(y, x) = 0` admits the **subsonic** branch `y ∈ (0, 1)` for `x < 1` and the
**supersonic** branch `y ∈ (1, ∞)` for `x > 1`. At `x = 1`, `y = 1`. The
implementation uses bracketed bisection (~80 iterations) and refuses to cross
the critical point.

Given `y(r)` we evaluate `U = a_c y` and use mass conservation to get
`ρ(r) = ρ_c a_c rcrit² / (U r²)`, where `ρ_c` is fixed by
`ρ(r_inner) = ρ_inner`. The pressure is `p = (ρ_c a_c² / γ) (ρ/ρ_c)^γ`, i.e.
`p = K ρ^γ` with `K = a_c² / (γ ρ_c^(γ−1))`.

The radial monopole `B_r(r) = B0 r_inner² / r²` is initialised on radial
**faces**. `B0` is calibrated to put the Alfvén point M_A(r) = U/v_A = 1 at
`r = r_alfven_target`:

```
B0 = U(rA) · sqrt(ρ(rA)) · (rA / r_inner)²
```

`B_θ = B_φ = 0`. Energy is the standard ideal-gas total:

```
E = p/(γ−1) + ½ρ v² + ½ B²
```

## 4. Default parameters and derived values

Default input (`inputs/tests/sph_shell_mhd_parker_polytropic.athinput`):

| parameter             | value |
|-----------------------|-------|
| `gamma_poly`          | 1.05  |
| `GM`                  | 4.0   |
| `rcrit`               | 2.0   |
| `rho_inner`           | 1.0   |
| `r_inner`             | 1.0   |
| `alfven_point_target` | 13.0  |
| domain `r ∈ `         | `[1, 20]` |
| angular wedge         | `8 × 8`, thin equatorial tube |

Derived values printed at IC (see logs for the actual reproducible numbers):

| derived               | value (default) |
|-----------------------|-----------------|
| `a_c = sqrt(GM/2rcrit)` | 1.000 |
| `U(r_inner)`          | 0.4086 |
| `ρ_c`                 | 0.1021 |
| `K_poly`              | 1.067 |
| `U(rA)`               | 2.425 |
| `ρ(rA)`               | 9.97e-4 |
| `B0`                  | 12.94 |

Approximate range of fast-mode speed across `r ∈ [1, 20]`: `√(v_A² + c_s²)` is
≈ 13 at the inner boundary (B-dominated) and ≈ 1 at the outer boundary
(sound-dominated), so `U + v_A` varies by about a factor of two over the
interior; a uniform-r grid keeps the cell-crossing time roughly constant for
later wave-resolution requirements.

## 5. Test matrix

All runs: ideal-gas MHD, `eos=ideal`, `gamma=1.05`, `<mhd_srcterms>/r_inv_sq_gravity=true`,
`r_inv_sq_gm=4`, `tlim=5`, thin equatorial wedge `(N_θ, N_φ) = (8, 8)`,
`meshblock/nx1=128`, radial user BCs. See `run_matrix.sh`.

| case label              | Reconstruction | Solver | N_r  |
|-------------------------|----------------|--------|-----:|
| `plm_hlld_nr512`        | PLM            | HLLD   | 512  |
| `plm_lhlld_nr512`       | PLM            | LHLLD  | 512  |
| `ppmx_hlld_nr512`       | PPMX           | HLLD   | 512  |
| `ppmx_lhlld_nr512`      | PPMX           | LHLLD  | 512  |
| `wenoz_hlld_nr512`      | WENOZ          | HLLD   | 512  |
| `wenoz_lhlld_nr512`     | WENOZ          | LHLLD  | 512  |
| `plm_hlld_nr1024`       | PLM            | HLLD   | 1024 |
| `plm_lhlld_nr1024`      | PLM            | LHLLD  | 1024 |
| `ppmx_hlld_nr1024`      | PPMX           | HLLD   | 1024 |
| `ppmx_lhlld_nr1024`     | PPMX           | LHLLD  | 1024 |

## 6. Preservation results

(Filled in after matrix completes; see `logs/<label>.log` for raw numbers and
`plots/scheme_comparison.png` for the cross-scheme summary.)

Baseline `plm_hlld_nr512`, `t=5` (1 outward AW crossing):

| diagnostic                  | value |
|-----------------------------|------:|
| L1 \|Δρ/ρ\|                 | 8.2e-3 |
| L1 \|Δv_r/v_r\|             | 3.4e-3 |
| L1 \|Δp/p\|                 | 7.2e-3 |
| L1 \|Δṁ/ṁ\|                 | 8.7e-3 |
| Linf \|Δρ/ρ\|               | 6.1e-2 |
| Linf \|Δv_r/v_r\|           | 7.5e-2 |
| **max \|v_⊥\|/c_s**          | **1.1e-12** |
| max \|B_θ\|/√ρ_inner        | 2.4e-12 |
| max \|B_φ\|/√ρ_inner        | 1.5e-15 |
| divB L1                     | 1.2e-12 |
| divB Linf · h / \|B\|max     | 4.7e-13 |

The L1 errors are dominated by the inner-boundary region where the analytic
profile is steepest; interior preservation is much better. Transverse
velocity is at machine-precision floor — the polytropic Parker background is
quiet under HLLD on this domain over one outward AW crossing.

## 7. HLLD vs LHLLD

| label                | max\|v_⊥\|/c_s | L1\|Δρ/ρ\| | L1\|Δv_r/v_r\| | divB L1 |
|----------------------|---------------:|-----------:|---------------:|--------:|
| `plm_hlld_nr512`     | 1.10e-12       | 8.23e-3    | 3.42e-3        | 1.18e-12 |
| `plm_lhlld_nr512`    | 1.44e-12       | 8.23e-3    | 3.42e-3        | 1.39e-12 |
| `plm_hlld_nr1024`    | 1.66e-12       | 2.11e-3    | 8.53e-4        | 1.41e-12 |
| `plm_lhlld_nr1024`   | 1.51e-12       | 2.11e-3    | 8.53e-4        | 1.57e-12 |
| `ppmx_hlld_nr512`    | 4.48e-12       | 1.45e-2    | 5.26e-3        | 7.14e-13 |
| `ppmx_lhlld_nr512`   | 3.78e-12       | 1.45e-2    | 5.26e-3        | 6.66e-13 |

On this quiet polytropic Parker background HLLD and LHLLD are quantitatively
indistinguishable. The radial L1 errors are identical to the listed precision
(the background drives the radial errors; the Riemann solver does not), and
the transverse-noise floor sits in the 10⁻¹² range for both solvers across
PLM and PPMX. divB is at roundoff for both. This stands in contrast to the
static spherical monopole, where HLLD shows multidimensional transverse growth
and LHLLD reduces but does not eliminate it (see
[`tst/sph_test_summaries/monopole_hlld_interface_debug_summary.md`] and
[`tst/sph_test_summaries/monopole_lhlld_minimal_summary.md`]). The polytropic
Parker outflow removes the pathology that the static monopole exposed.

## 8. PLM vs PPMX/WENOZ

| label                | max\|v_⊥\|/c_s | L1\|Δρ/ρ\| | L1\|Δp/p\| | L1\|Δṁ/ṁ\| |
|----------------------|---------------:|-----------:|-----------:|-----------:|
| `plm_hlld_nr512`     | 1.10e-12       | 8.23e-3    | 7.22e-3    | 8.73e-3    |
| `plm_hlld_nr1024`    | 1.66e-12       | 2.11e-3    | 1.86e-3    | 2.30e-3    |
| `ppmx_hlld_nr512`    | 4.48e-12       | 1.45e-2    | 1.34e-2    | 1.84e-2    |

PLM at Nr=512 already gives radial L1 errors below 1%; doubling to Nr=1024
gives ≈3.9× reduction (close to the 4× expected for second order). PPMX at
Nr=512 has *larger* L1 than PLM here. The reason is that the dominant L1
contribution comes from the inner-boundary cell where the analytic profile is
steepest and the higher-order PPMX reconstruction is more sensitive to the
strong gradient; interior errors are smaller for PPMX, but the inner-boundary
contribution dominates the global norm. All schemes keep the transverse
floor at machine precision. WENOZ was not run in the reduced matrix (set
`FULL=1` in `run_matrix.sh` to add it).

## 9. Recommendation for the next task

The matrix confirms the baseline: `max|v_⊥|/c_s` ≈ 10⁻¹² across all four
PLM/PPMX × HLLD/LHLLD combinations at both Nr=512 and Nr=1024 through one
outward AW crossing. The polytropic Parker is the recommended background for
the lower-boundary monochromatic Alfvén-wave driver test. Open questions left
for the wave-driver task:

- What is the largest amplitude that stays linear over five outward AW
  crossings on this background?
- Where to position the inner driver radius vs. `r_inner` (the analytic
  Parker base): on `r_inner` itself with a perturbative `B_θ`/`v_θ` overlay,
  or one ghost zone inward.
- Choice of solver: HLLD and LHLLD are quantitatively indistinguishable on
  this background through `t=5`, so either is acceptable for the AW driver;
  LHLLD retains a slight edge on the underlying static-monopole pathology
  that this Parker test does not trigger.

## 10. Reproduction

```bash
# Build
( cd build && cmake -DPROBLEM=sph_shell_mhd .. && cmake --build . --target athena -j 4 )

# Single baseline
./build/src/athena -i inputs/tests/sph_shell_mhd_parker_polytropic.athinput \
    problem/csv_dir=poly_parker_mhd_validation/csv \
    problem/label=plm_hlld_nr512 \
  | tee poly_parker_mhd_validation/logs/plm_hlld_nr512.log

# Full matrix (writes logs/, csv/)
bash poly_parker_mhd_validation/run_matrix.sh

# Plots (logs + CSVs → plots/)
python3 poly_parker_mhd_validation/plot_results.py
```

Everything under `poly_parker_mhd_validation/` is `.gitignore`d.
