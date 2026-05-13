# LHLLD Minimal Port and Parker HLLD Follow-up

Date: 2026-05-14

## Environment and Build

- GPU node: NVIDIA H200 NVL, 143771 MiB
- Git hash: `ec9aff518ce5813bdcb663bae871667ed2a1540c`
- Executable: `build-wave-action-mhd-gpu-gcc/src/athena`
- Build cache: `PROBLEM=sph_shell_mhd`, `CMAKE_BUILD_TYPE=Release`, `Kokkos_ENABLE_CUDA=ON`, `Athena_ENABLE_MPI=OFF`
- Compiler path: `kokkos/bin/nvcc_wrapper`; Kokkos CUDA compiler version in cache: `12.5.82`
- Shell module state during this continuation: no modules loaded

## Files Changed for LHLLD

- Added `src/mhd/rsolvers/lhlld_mhd.hpp`
- Modified `src/mhd/mhd.hpp`
- Modified `src/mhd/mhd.cpp`
- Modified `src/mhd/mhd_fluxes.cpp`
- Modified `src/mhd/mhd_tasks.cpp`

The new solver is selected with:

```ini
<mhd>
rsolver = lhlld
```

The port follows Athena++ `srcpp/hydro/rsolvers/mhd/lhlld.cpp` closely, using
AthenaK's Kokkos device style and the existing `mhd_fluxes.cpp` dispatch
structure. Existing `rsolver=hlld` is unchanged.

Important limitation: this LHLLD port is ideal-gas/adiabatic only. The parser
intentionally rejects `rsolver=lhlld` for `eos=isothermal`; no new isothermal
LHLLD path was invented.

The rejection check is preserved in:

- `logs/parker_iso_lhlld_rejected.log`

It fails with:

```text
<mhd>/rsolver = lhlld is implemented only for <mhd>/eos = ideal. Use hlld/hlle/llf for isothermal MHD.
```

## Monopole LHLLD Result

Reference layout:

- `N_r x N_theta x N_phi = 1024 x 8 x 8`
- `meshblock/nx1 = 128`
- thin equatorial tube
- `B0=0.5`, `rho0=p0=1`
- `tlim=10`

| Case | max `v_perp/c_s` at `t=10` | divB Linf | Comment |
|---|---:|---:|---|
| PLM + HLLD | `9.56e-3` | roundoff | prior bad reference |
| PLM + HLLE | `5.96e-14` | roundoff | prior quiet reference |
| PLM + LHLLD | `6.70e-3` | `2.74e-11` | modestly better, not fixed |
| PPMX + HLLD | `4.92e-3` | roundoff | prior comparison |
| PPMX + LHLLD | `5.31e-3` | `1.18e-11` | not better than PPMX+HLLD |

Logs:

- `logs/lhlld_plm_ref_t10_abs.log`
- `logs/lhlld_ppmx_ref_t10.log`

Conclusion: LHLLD reduces the PLM bad-reference monopole transverse growth by
only about 30 percent. It does not remove the multidimensional transverse-mode
pathology. With PPMX, LHLLD is not clearly better than HLLD. This is not enough
to resume static-monopole driven-wave diagnostics with LHLLD as a cure.

## Parker Input Used

Local ignored input:

- `inputs/sph_shell_mhd_parker_isothermal_debug.athinput`

This is an isothermal MHD Parker wind with a radial monopole field:

- `mode=parker_isothermal`
- `eos=isothermal`, `iso_sound_speed=1`
- `rsolver=hlld`
- `B0=0.5`
- `GM=4`
- `<mhd_srcterms>/r_inv_sq_gravity=true`
- radial user BCs with analytic Parker/monopole state

An earlier no-gravity run was discarded for radial preservation conclusions;
it was still transversely quiet, but the radial Parker profile drifted for the
obvious reason that gravity was absent. All results below use the corrected
gravity input.

## Parker HLLD Grid and Domain Checks

CSV and plot:

- `csv/parker_hlld_summary.csv`
- `plots/parker_hlld_grid_domain_summary.png`
- `plots/parker_hlld_long_baseline.png`

All runs used PLM + HLLD, isothermal MHD, `tlim=10`, `meshblock/nx1=128`.

| Case | `N_r` | Grid/domain | L1 `|dv_r/v_r|` | L1 `|drho/rho|` | max `|v_theta|/cs` | max `|v_phi|/cs` | divB Linf |
|---|---:|---|---:|---:|---:|---:|---:|
| uniform thin | 1024 | `8x8`, theta width 0.12, phi width pi/2 | `1.21e-6` | `8.49e-5` | `6.41e-11` | `7.49e-16` | `3.15e-11` |
| log thin | 1024 | `8x8`, theta width 0.12, phi width pi/2 | `8.56e-7` | `6.50e-6` | `7.83e-10` | `2.05e-15` | `7.51e-11` |
| theta 4x | 1024 | `32x8`, theta width 0.48 | `1.21e-6` | `8.49e-5` | `5.83e-10` | `1.33e-15` | `5.49e-11` |
| phi 4x | 1024 | `8x32`, phi width 2pi | `1.21e-6` | `8.49e-5` | `6.95e-11` | `6.04e-11` | `7.68e-11` |
| uniform thin high-res | 2048 | `8x8`, theta width 0.12, phi width pi/2 | `3.08e-7` | `2.11e-5` | `4.42e-11` | `9.03e-16` | `1.16e-10` |
| log thin high-res | 2048 | `8x8`, theta width 0.12, phi width pi/2 | `2.14e-7` | `1.62e-6` | `2.01e-9` | `8.17e-16` | `4.39e-10` |

Raw logs:

- `logs/parker_iso_hlld_grav_uniform_nr1024_t10.log`
- `logs/parker_iso_hlld_grav_log_nr1024_t10.log`
- `logs/parker_iso_hlld_grav_theta4x_nr1024_t10.log`
- `logs/parker_iso_hlld_grav_phi4x_nr1024_t10.log`
- `logs/parker_iso_hlld_grav_uniform_nr2048_t10.log`
- `logs/parker_iso_hlld_grav_log_nr2048_t10.log`

## Parker Interpretation

The corrected isothermal Parker wind is very quiet transversely with HLLD.
Across radial grids, wider theta, full-2pi phi, and `N_r=2048`, the largest
transverse component is at most `~2e-9 cs`. This is many orders of magnitude
below the static-monopole HLLD pathology (`~1e-2 cs` by `t=10` in the bad
uniform-grid reference case).

The Parker radial preservation diagnostics also look healthy:

- uniform-grid L1 density/mass-flux errors drop by about 4x from `N_r=1024` to
  `N_r=2048`, consistent with second-order behavior for PLM;
- log-r gives smaller density and mass-flux errors for this Parker domain at
  both resolutions;
- divB remains small in all cases. The absolute divB Linf rises with log/high
  resolution, but the scaled `Linf*h/|B|max` printed in the logs remains
  `O(1e-12)` or below.

The 4x wider theta and phi-domain runs are important: the Parker quietness is
not just an artefact of the original thin `8x8` tube. The full-phi case does
show a larger `v_phi` floor (`6e-11 cs`) than the thin run, but this is still
tiny and does not resemble the static-monopole runaway.

## Long Parker Baseline for AW Tests

The analytic Parker+monopole background over `r=[1,6]` has an approximate
outward Alfvén-wave crossing time

```text
tau_AW = integral dr / (U + v_A) = 2.99
```

for `GM=4`, `c_s=1`, `B0=0.5`, and `rho_inner=1`. Thus:

- `t=10` is already about 3.3 outward AW crossings;
- `t=15` is about 5 crossings;
- `t=30` is about 10 crossings.

Additional long runs were made to check whether the quiet `t=10` result
survives over a driven-wave style baseline.

| Case | `tlim` | Approx AW crossings | L1 `|dv_r/v_r|` | L1 `|drho/rho|` | max `|v_theta|/cs` | Comment |
|---|---:|---:|---:|---:|---:|---|
| uniform `Nr=1024`, PLM | 10 | 3.3 | `1.21e-6` | `8.49e-5` | `6.41e-11` | clean |
| uniform `Nr=1024`, PLM | 15 | 5.0 | `1.63e-6` | `8.57e-5` | `2.37e-8` | still clean enough |
| uniform `Nr=1024`, PLM | 30 | 10.0 | `1.72e-2` | `3.91e-2` | `4.55e-2` | unacceptable late growth |
| log `Nr=1024`, PLM | 10 | 3.3 | `8.56e-7` | `6.50e-6` | `7.83e-10` | clean |
| log `Nr=1024`, PLM | 15 | 5.0 | `8.92e-7` | `6.53e-6` | `5.49e-8` | still clean enough |
| log `Nr=1024`, PLM | 30 | 10.0 | `4.65e-2` | `3.15e-2` | `6.39e-2` | unacceptable late growth |
| log `Nr=2048`, PLM | 10 | 3.3 | `2.14e-7` | `1.62e-6` | `2.01e-9` | clean |
| log `Nr=2048`, PLM | 30 | 10.0 | `6.16e-2` | `4.97e-2` | `6.25e-2` | not improved by radial resolution |
| log `Nr=1024`, PPMX | 30 | 10.0 | `3.16e-2` | `1.08e-2` | `7.04e-2` | PPMX does not cure it |

The long-time behavior is qualitatively different from the `t<=15` baselines:
the transverse velocity grows from `~1e-10` to `~1e-8` by five crossings, then
to `O(5e-2)` by ten crossings. Radial preservation also collapses at `t=30`,
with several-percent density/mass-flux errors and `v_r` errors of order
`1e-2` to `1e-1`.

This looks like a delayed growing mode rather than a steady small drift. The
available data are sparse in time, so the growth rate is not precisely fitted,
but the jump between `t=15` and `t=30` is consistent with a rapid/exponential
late-time instability or contamination after a small seed reaches nonlinear
amplitude. divB remains controlled, so this is not a simple divergence failure.

## Current Recommendation

- LHLLD should not be considered a fix for the static spherical monopole
  HLLD transverse growth.
- For static-monopole Alfvén-wave damping/action tests, avoid HLLD/LHLLD unless
  the run uses a known-quiet configuration such as HLLE/LLF, log-r with verified
  early-time quietness, or larger radial MeshBlocks on a build that can launch
  them.
- For short isothermal Parker-wind MHD tests, HLLD is acceptable through about
  3-5 outward AW crossings in these runs. This is enough for a single packet
  transit or a short propagation/vibe check if the wave amplitude is safely
  above the `1e-8 cs` transverse floor at `t=15`.
- For a quantitative driven monochromatic AW damping/action suite, HLLD Parker
  is not yet acceptable over a ten-crossing baseline. The no-wave background
  itself develops `O(0.05 cs)` transverse velocity and percent-level radial
  errors by `t=30`.
- Because LHLLD is not implemented for isothermal MHD, it cannot currently be
  used as the Parker workaround.
- Before using Parker+HLLD for production damping measurements, either shorten
  the analysis to a few crossings, switch to a quieter solver for the baseline
  comparison, or diagnose the late Parker HLLD growth in the same way the
  static monopole case was diagnosed.
- If a future production target is static monopole plus HLLD, the next solver
  step should not be another local zero-transverse guard. The evidence points to
  a coupled multidimensional amplification issue, not a single-face HLLD algebra
  bug.
