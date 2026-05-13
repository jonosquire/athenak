# Monopole Baseline Drift Debug

This pass pauses the driven AW work and tests the existing `monopole` mode using native-bin r-theta slices. Diagnostics exclude ghost zones; the `interior` metric excludes 8 radial cells and one theta cell where available.

## Main conclusion

The transverse monopole drift is not a one-step force-balance error and not a slow linear physical drift. In the problematic case it stays at roundoff through early times, then grows rapidly after about `t=5`.

The strongest trigger found is **HLLD in multidimensional runs with many radial MeshBlocks**. HLLE/LLF remain at roundoff, 1D radial HLLD remains exactly quiet, and HLLD becomes quiet again when the same 1024-cell case is run with larger radial MeshBlocks.

Practical implication: do not use small-amplitude AW diagnostics with `mhd/rsolver=hlld` and many radial MeshBlocks until the block-boundary/HLLD issue is fixed or better understood. Use HLLE/LLF, or use larger radial MeshBlocks with a verified no-wave baseline.

## Key evidence

Reference case:

- `N_r x N_theta x N_phi = 1024 x 8 x 8`
- `meshblock/nx1 = 128`, so 8 radial MeshBlocks
- `reconstruct=plm`, `rsolver=hlld`
- uniform-r, thin equatorial tube, `B0=0.5`, `p0=rho0=1`

Time behavior:

| time range | behavior |
|---|---|
| `t <= 2` | transverse components remain near roundoff |
| `t ~ 5` | transverse mode becomes visible at `~3e-10 c_s` |
| `t = 8` | `max |v_perp|/c_s ~ 2e-5` |
| `t = 10` | `max |v_perp|/c_s = 9.56e-3` |

An exponential fit over `t=5..10` gives growth rate `~3.5`, or an e-fold time `~0.28`. This is an instability-like growth from a tiny seed, not a small secular drift.

The maximum is interior, near `r ~= 6.1`; for `N_r=1024, meshblock/nx1=128`, this is about 20 cells inside the radial MeshBlock interface at `i=768`.

## Final case summary

| case | final max `|v_perp|/c_s` | interpretation |
|---|---:|---|
| `ref_plm_hlld_nr1024_th8_ph8` | `9.56e-3` | problematic reference |
| `hi_plm_hlle_nr1024_th8_ph8` | `5.96e-14` | HLLE suppresses/removes it |
| `hi_plm_llf_nr1024_th8_ph8` | `3.60e-14` | LLF suppresses/removes it |
| `hi_dc_hlld_nr1024_th8_ph8` | `3.08e-2` | donor + HLLD is worse |
| `hi_ppmx_hlld_nr1024_th8_ph8` | `6.72e-3` | PPMX + HLLD still grows, somewhat less than PLM |
| `hi_plm_hlld_cfl015_nr1024_th8_ph8` | `3.90e-3` | smaller CFL reduces but does not remove growth |
| `hi_plm_hlld_nr1024_th16_ph8` | `4.64e-3` | more theta cells reduces but does not remove growth |
| `hi_rtheta2d_plm_hlld_nr1024` | `4.29e-3` | r-theta 2D is enough to trigger growth |
| `hi_radial1d_plm_hlld_nr1024` | `0.0` | pure radial HLLD is quiet |
| `mb_plm_hlld_nr1024_mb64` | `9.56e-3` | many radial MeshBlocks grows |
| `mb_plm_hlld_nr1024_mb128` / reference | `9.56e-3` | many radial MeshBlocks grows |
| `mb_plm_hlld_nr1024_mb256` | `0.0` | 4 radial MeshBlocks is quiet to `t=10` |
| `mb_plm_hlld_nr1024_mb1024` | `0.0` | single radial MeshBlock is quiet to `t=10` |
| `radial1d_plm_hlld_nr512` | `0.0` | pure radial case is quiet |

At `N_r=512` with `meshblock/nx1=128` there are only 4 radial MeshBlocks, and HLLD cases stay small through `t=10` (`~1e-10` or below). This matches the 1024-cell `meshblock/nx1=256` result.

The attempted `mb256` extension to `t=20` aborted immediately with `Kokkos::Impl::ParallelFor< Cuda > insufficient shared memory`; it is not used as physics evidence.

## What this rules out

- **Diagnostic-only error:** the pgen final `max|v|/c_s` and bin-slice component diagnostics agree. The transverse components in the bin data are real.
- **Immediate source-term imbalance:** the reference run is at roundoff for transverse components at early times. There is no visible one-step or early-time transverse force.
- **Radial boundary condition as the primary cause:** pure radial 1D HLLD is exactly quiet, and the maxima are interior, not at the radial domain boundary.
- **Theta boundary alone:** the r-theta 2D case grows, but the maximum is not consistently a domain theta-boundary layer. Increasing theta resolution reduces the amplitude but does not remove it.
- **divB failure:** divB remains near roundoff in all cases (`Linf*h/|B|max ~ 1e-13` for the multidimensional cases).

## Likely origin

The most likely origin is a tiny transverse seed generated or amplified by the HLLD path at internal radial MeshBlock boundaries in multidimensional spherical-shell MHD/CT. The exact low-level source still needs a code audit. The next places to inspect are:

- radial MeshBlock ghost exchange for face-centered magnetic fields and `bcc`;
- CT EMF synchronization at radial MeshBlock interfaces;
- HLLD degeneracy handling when the normal field is nonzero and transverse fields/velocities are analytically zero;
- whether radial block interface states are bitwise identical for `B_r`, `B_theta`, `B_phi`, and transverse momenta after boundary exchange;
- whether the issue appears under MPI, where pack boundaries and rank boundaries may differ.

## Recommended workaround for driven AW

Before continuing the AW driver:

- Use `mhd/rsolver=hlle` or `llf` for the first quantitative driver test if very small amplitudes are required.
- If testing HLLD, first run a no-wave baseline with the exact same `N_r`, angular grid, CFL, and MeshBlock layout.
- Keep radial MeshBlocks large enough that the no-wave baseline is quiet. In these tests, `N_r=1024` with `meshblock/nx1=256` or `1024` was quiet through `t=10`; `meshblock/nx1=128` and `64` were not.
- Do not interpret AW reflection/damping below the measured no-wave transverse floor.

## Outputs

- `csv/component_timeseries.csv`
- `csv/final_radial_profiles.csv`
- `csv/final_theta_profiles.csv`
- `plots/reference_component_timeseries.png`
- `plots/reference_vperp_r_time.png`
- `plots/case_final_vperp_comparison.png`
- representative final radial-profile plots in `plots/`

## Next code-debug step

Add a guarded internal-interface diagnostic for the monopole state:

1. After boundary exchange and before fluxes, compare left/right states adjacent to radial MeshBlock interfaces for transverse velocity, transverse `bcc`, and face-centered `B_r`.
2. Record the HLLD radial-interface transverse fluxes at those interfaces for the static monopole.
3. Compare the same interfaces under HLLE and HLLD.
4. Repeat with `meshblock/nx1=128` and `256` at `N_r=1024`.

That should distinguish a ghost/exchange seed from an HLLD amplification of roundoff.
