# Monopole HLLD Interface Debug Summary

Date: 2026-05-13

Build used:
- Executable: `build-wave-action-mhd-gpu-gcc/src/athena`
- GPU run environment: `module load cuda/12.5.1-b6iqzzi`
- Problem: `sph_shell_mhd`
- Input template: `monopole_hlld_interface_debug/inputs/sph_shell_mhd_monopole_interface_debug.athinput`

## Code Instrumentation

Added a guarded monopole-only finalizer diagnostic:

```ini
<problem>
debug_internal_interfaces = true
debug_output_dir = /abs/path/to/monopole_hlld_interface_debug/csv
debug_label = case_name
```

It writes:

- `csv/global_component_diagnostics.csv`
- `csv/interface_state_jumps.csv`
- `csv/radial_riemann_interface_fluxes.csv`
- `csv/emf_interface_diagnostics.csv`
- `csv/processed_case_summary.csv`

The diagnostic compares internal radial MeshBlock interfaces only. Physical radial boundaries are excluded.

Plots were generated with:

```bash
/home/squjo23p/conda-envs/curmpy/bin/python monopole_hlld_interface_debug/scripts/analyze_interface_debug.py
```

Plot outputs:

- `plots/ref_hlld_growth_vs_time.png`
- `plots/solver_reconstruction_comparison_t10.png`
- `plots/block_interface_dependence_t10.png`
- `plots/interface_copy_jumps_t10.png`
- `plots/interface_flux_emf_t10.png`

## Reproduction Matrix

All main runs used a thin equatorial tube, `B0=0.5`, `rho0=p0=1`, `r in [1,8]`, `theta in [1.5108,1.6308]`, and `N_theta x N_phi = 8 x 8` unless noted.

| Case | Result |
|---|---:|
| `N_r=1024`, `nx1_mb=128`, PLM+HLLD | bad: final max `v_perp/c_s = 9.56e-3` |
| same, PLM+HLLE | quiet: final max `v_perp/c_s = 5.96e-14` |
| same, PLM+LLF | quiet: final max `v_perp/c_s = 3.60e-14` |
| same, DC+HLLD | worse: final max `v_perp/c_s = 3.08e-2` |
| same, PPMX+HLLD, `nghost=3` | still grows: final max `v_perp/c_s = 4.92e-3` |
| 1D radial, PLM+HLLD | quiet: final max `v_perp = 0` |
| `N_r=128`, one radial block, PLM+HLLD | quiet: final max `v_perp/c_s = 1.41e-14` |
| `N_r=256`, 2 radial blocks, PLM+HLLD | quiet: final max `v_perp/c_s = 9.78e-14` |
| `N_r=512`, 4 radial blocks, PLM+HLLD | tiny: final max `v_perp/c_s = 4.09e-10` |
| `N_r=960`, `nx1_mb=120`, PLM+HLLD | grows: final max `v_perp/c_s = 1.86e-3` |
| `N_r=960`, `nx1_mb=192`, PLM+HLLD | grows less: final max `v_perp/c_s = 1.58e-3` |

Attempts with `nx1_mb >= 256` on this GPU build failed at launch with:

```text
Kokkos::Impl::ParallelFor< Cuda > insufficient shared memory
```

so the exact `N_r=1024`, `nx1_mb=256` quiet comparison could not be reproduced on this executable.

## Time Behavior

For the reference PLM+HLLD case:

| Time | max `v_perp/c_s` | max interface `E2_x1` | max interface `E3_x1` |
|---:|---:|---:|---:|
| 2.5 | `2.81e-14` | `3.87e-17` | `2.47e-16` |
| 5.0 | `2.78e-10` | `5.39e-13` | `2.14e-12` |
| 7.5 | `3.45e-6` | `4.71e-9` | `1.16e-8` |
| 10.0 | `9.56e-3` | `2.55e-5` | `1.82e-5` |

Conclusion: this is not a slow linear drift. It is quiet to near roundoff, then grows rapidly after a small seed reaches roughly `1e-10` to `1e-6`.

## Interface State Checks

For the bad PLM+HLLD run at `t=10`:

- left-ghost vs right-active jumps in `rho`, `p`, `v_r`, `v_theta`, `v_phi`: exactly `0` in the CSV
- right-ghost vs left-active jumps in the same quantities: exactly `0`
- ghost-active jumps in `Bcc_r`, `Bcc_theta`, `Bcc_phi`: exactly `0`
- shared radial face `B1f` jump across duplicate MeshBlocks: exactly `0`

This rules out a simple cell-centered ghost exchange mismatch and a shared radial face `B1f` mismatch as the immediate cause.

## Radial Interface Flux Checks

For the bad PLM+HLLD run at `t=10`:

- duplicate left/right radial-interface fluxes are identical: `max_flux_mtheta_jump = 0`
- but the actual HLLD transverse fluxes/EMFs are nonzero:
  - max `F(m_theta)` on x1 interfaces: `1.86e-5`
  - max `E2_x1`: `2.55e-5`
  - max `E3_x1`: `1.82e-5`

For PLM+HLLE and PLM+LLF the same quantities remain at roundoff.

Conclusion: the evidence does not point to inconsistent duplicate fluxes across the same interface. The HLLD flux is consistently producing/amplifying transverse dynamics once transverse perturbations exist.

## EMF / CT

The corner EMFs grow with the same timing as `v_perp` and transverse `B`.

At `t=10` in PLM+HLLD:

- global max corner `E2 = 1.66e-4`
- global max corner `E3 = 4.94e-5`
- internal-interface `E2_x1 = 2.55e-5`
- internal-interface `E3_x1 = 1.82e-5`

HLLE/LLF and 1D HLLD remain at roundoff. This suggests CT is responding to the transverse mode rather than obviously injecting it through a mismatched interface sync.

## Localization

The maximum transverse velocity in the bad `1024 x 8 x 8`, `nx1_mb=128` run is at:

- `r = 6.096`
- close to, but not exactly on, the radial block interface at `r = 6.25`
- about 20-25 active cells inward from that interface

The same physical radius appears in the `nx1_mb=64` case. Lower total radial resolution is much quieter, so the pathology depends on multidimensional HLLD plus sufficient radial resolution, with internal radial interfaces likely acting as a seed/modulator rather than showing an explicit copied-state mismatch.

## Angular-Domain and Radial-Grid Dependence

A quick production-relevance matrix was run with PLM+HLLD, `N_r=1024`,
`nx1_mb=128`, `B0=0.5`, `rho0=p0=1`, and `tlim=10`.

Outputs:

- `csv/angular_radial_grid_matrix.csv`
- `plots/angular_radial_grid_summary.png`
- raw per-run diagnostics under `csv/angular_radial_raw/`

The threshold crossing times in the CSV are sampled upper bounds from the
available `t=2.5, 5.0, 7.5, 10.0` outputs, not exact crossing times.

| Case | final max `v_perp/c_s` | boundary-excluded | Notes |
|---|---:|---:|---|
| thin `8x8`, uniform r | `9.56e-3` | `6.48e-3` | reference bad case |
| same extent `16x16`, uniform r | `4.64e-3` | `4.50e-3` | growth persists |
| same extent `32x16`, uniform r | `1.71e-3` | `1.70e-3` | smaller but not gone |
| theta extent `2x`, `16x8`, uniform r | `2.58e-3` | `2.31e-3` | smaller than reference |
| theta extent `4x`, `32x8`, uniform r | `5.48e-3` | `5.48e-3` | interior maximum; not only boundary-localized |
| phi extent `2x`, `8x16`, uniform r | `9.56e-3` | `6.48e-3` | essentially identical to reference |
| phi extent `4x`, `8x32`, uniform r | `9.56e-3` | `6.48e-3` | essentially identical to reference |
| `8x8`, log r | `4.23e-9` | `4.03e-9` | dramatically quieter through t=10 |
| `8x8`, power_law `alpha=1/3` | `9.88e-1` | `9.88e-1` | unusably unstable |
| `8x8`, r_cubed | `1.06e0` | `4.20e-1` | unusably unstable |

Interpretation:

- The thin angular tube is not the sole cause. Increasing angular resolution
  and/or widening theta can reduce the growth, but does not eliminate it.
- Widening periodic phi has no measurable effect, consistent with a
  `k_perp=0` or nearly axisymmetric transverse mode.
- The radial grid matters strongly. Log-r is much quieter for this reference
  setup, while the current power-law and r-cubed grids trigger much larger
  transverse growth by `t=10`.
- The issue should be treated as production-relevant for uniform-r HLLD runs
  with many radial MeshBlocks. It is not just an 8x8 plotting/diagnostic artefact.

## HLLD Solver-Path Comparison

AthenaK and Athena++ HLLD sources were compared directly.

Notes:

- `hlld_comparison_notes.md`
- `csv/hlld_unit_flux_states.csv`
- `csv/hlld_unit_flux_results.csv`

Files inspected:

- AthenaK `src/mhd/rsolvers/hlld_mhd.hpp`
- AthenaK `src/mhd/rsolvers/hlle_mhd.hpp`
- Athena++ `srcpp/hydro/rsolvers/mhd/hlld.cpp`
- Athena++ `srcpp/hydro/rsolvers/mhd/hlld_iso.cpp`
- Athena++ `srcpp/hydro/rsolvers/mhd/lhlld.cpp`
- Athena++ `srcpp/hydro/rsolvers/mhd/hlle_mhd.cpp`
- Athena++ `srcpp/pgen/quirk.cpp`

The AthenaK adiabatic HLLD path matches the Athena++ adiabatic HLLD formulae:
same degeneracy constants, same star-state pressure formula, same transverse
star-state handling, and same transverse magnetic-flux / EMF sign convention.

A standalone unit-style flux probe tested exact and nearly degenerate radial
monopole states. Results:

- exact identical `v_t=B_t=0` states produce zero transverse fluxes in HLLD;
- tiny pure density or pressure jumps do not generate transverse fluxes;
- tiny transverse inputs produce fluxes proportional to the input amplitude;
- HLLD and HLLE both behave sensibly for these single-face unit states.

The reference Athena++ tree contains `lhlld.cpp`, and `srcpp/pgen/quirk.cpp`
notes that Roe/HLLC/HLLD-type solvers can amplify odd-even perturbations in the
Quirk/carbuncle problem, with LHLLC/LHLLD as the intended mitigation. That is a
better analogue for this observed transverse-mode amplification than a simple
wrong-sign or zero-state HLLD flux bug.

## HLLD Guard Decision

No HLLD degeneracy guard was implemented in this pass.

Reason:

- the standalone exact-state HLLD probe does not generate spurious transverse
  flux from a degenerate monopole state;
- pressure/density-only roundoff perturbations also do not generate transverse
  flux;
- the previous duplicate-interface diagnostics showed consistent left/right
  fluxes and no shared `B1f` or ghost-copy mismatch;
- a strict transverse guard would not address the likely coupled
  multidimensional amplification mechanism, while a looser guard risks
  suppressing real small-amplitude Alfvén waves.

If a defensive HLLD modification is pursued later, the evidence points more
toward an LHLLD-style multidimensional anti-carbuncle regularization or a
targeted local fallback triggered by a measured odd-even/interface instability,
not a generic zero-transverse clamp for roundoff states.

## Current Conclusion

The drift is best characterized as a multidimensional HLLD transverse-mode instability/amplification in the spherical monopole background. The debug data rule out the most obvious internal radial MeshBlock interface bugs:

- no active/ghost state mismatch;
- no shared `B1f` mismatch;
- no duplicate radial flux mismatch;
- divB stays small;
- HLLE/LLF suppress the mode;
- 1D radial HLLD is quiet.

No targeted solver fix was made in this pass. A defensive HLLD zero-transverse guard would be risky without more evidence because the duplicated HLLD fluxes are internally consistent; it could hide rather than fix a real multidimensional force/Alfven-mode imbalance.

## Recommendation for Driven AW Work

Do not use HLLD for the first quantitative driven-AW damping/action test yet.

Safe options:

1. Use HLLE or LLF for the first driven-AW diagnostic.
2. Keep any HLLD test in the early-time window, before `t ~ 5`, where the reference case is still below `v_perp/c_s ~ 3e-10`.
3. If HLLD is required, first add a more invasive debug hook that records pre-flux reconstructed left/right states at every x1 face, not only final conserved/primitive states and final fluxes.

Next focused debug steps:

- add a controlled tiny transverse seed and measure HLLD/HLLE growth rates;
- add a debug option to zero transverse fields after each RK stage to confirm the mode needs accumulated transverse `v/B`;
- compare radial HLLD reconstructed states before the Riemann solve at all x1 faces, not only MeshBlock interfaces;
- test a CPU/Kokkos build with larger radial MeshBlocks (`nx1_mb=256+`) because the current GPU build cannot launch those block sizes.
