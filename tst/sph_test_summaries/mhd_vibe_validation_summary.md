# Spherical-Shell MHD Vibe Validation

Date: 2026-05-12

## Environment and Build

- Repo: `/weka/scratch/squjo23p/athenak-tests/athenak`
- Git commit: `ec9aff51` with local MHD validation edits
- GPU seen by this session: NVIDIA A100 80GB PCIe, driver 555.42.06
- Modules: none loaded
- CUDA: `/opt/cuda/12.9.1`, nvcc 12.9.86
- Host compiler: GCC 11.3.1
- CMake: 3.26.5
- Build directory: `build-mhd-vibe-gpu-gcc`
- Build: Release, `PROBLEM=sph_shell_mhd`, `Kokkos_ENABLE_CUDA=ON`,
  `Kokkos_ARCH_AMPERE80=ON`, `Athena_ENABLE_MPI=OFF`

CMake command used:

```bash
env -i PATH=/opt/cuda/12.9.1/bin:/usr/bin:/bin \
  LD_LIBRARY_PATH=/opt/cuda/12.9.1/lib64:/usr/lib64 \
  HOME=/tmp USER=squjo23p NVCC_WRAPPER_DEFAULT_COMPILER=/usr/bin/g++ \
  cmake -S . -B build-mhd-vibe-gpu-gcc \
  -DCMAKE_BUILD_TYPE=Release -DPROBLEM=sph_shell_mhd \
  -DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ARCH_AMPERE80=ON -DAthena_ENABLE_MPI=OFF \
  -DCMAKE_CXX_COMPILER=/weka/scratch/squjo23p/athenak-tests/athenak/kokkos/bin/nvcc_wrapper \
  -DCMAKE_C_COMPILER=/usr/bin/gcc
```

## Inputs Added

- `inputs/tests/sph_shell_mhd_radial_alfven_bin_vibe.athinput`
  - 512 x 96 x 128 cells, meshblocks 32 x 16 x 16
  - Native `bin` slices: r-theta, r-phi, theta-phi
  - `mhd_w_bcc`, output every `dt=3`, `tlim=12`
- `inputs/tests/sph_shell_mhd_loop_advect_bin_vibe.athinput`
  - 256 x 48 x 256 cells, meshblocks 32 x 16 x 32
  - Native `bin` slices: r-theta, r-phi, theta-phi
  - `mhd_w_bcc`, output every `dt=pi/2`, `tlim=2*pi`

The radial Alfven mode now uses the same analytic radial monopole user BCs
as the monopole preservation test, avoiding the known outflow-ghost boundary
force residual.

## GPU Smoke Tests

| input | status | key diagnostics |
|---|---:|---|
| `sph_shell_mhd_monopole.athinput` | pass | `max|v|/cs=4.22e-5`, `max|B1f-analytic|=1.1e-16`, `Linf*h/|B|max=4.9e-16` |
| `sph_shell_mhd_monopole_logr.athinput` | pass | `max|v|/cs=7.15e-5`, `max|B1f-analytic|=1.1e-16`, `Linf*h/|B|max=1.8e-15` |
| `sph_shell_mhd_radial_alfven.athinput` | pass | centroid lag `-2.1%` of distance, peak amplitude `0.926` of IC, `Linf*h/|B|max=3.7e-15` |
| `sph_shell_mhd_loop_eq.athinput` | pass | static loop preserved, `Linf*h/|B|max=3.4e-15` |

The first radial Alfven smoke attempt with a 256 x 8 x 8 meshblock exceeded
CUDA shared memory. Rerunning with 32 x 8 x 8 meshblocks passed.

## Native Bin Slice Status

Both vibe runs wrote three native binary slice streams under each run's
`bin/` directory. The reader loaded one combined file per slice and time
number and exposed:

`dens, velx, vely, velz, eint, bcc1, bcc2, bcc3`

The binary header includes the input dump, so the plot script recovered
`coord/system=spherical_shell`, mesh extents, slice orientation, time, cycle,
and dimensions without writer changes.

Generated binary files:

- AW atmosphere: 15 files, 20 MB
- Loop advection: 15 files, 14 MB

## Plot Script

Script:

- `mhd_vibe_validation/scripts/plot_mhd_bin_slices.py`

Outputs:

- `mhd_vibe_validation/plots/aw_atmosphere_fourpanel_00000.png`
- `mhd_vibe_validation/plots/aw_atmosphere_fourpanel_00001.png`
- `mhd_vibe_validation/plots/aw_atmosphere_fourpanel_00002.png`
- `mhd_vibe_validation/plots/aw_atmosphere_fourpanel_00003.png`
- `mhd_vibe_validation/plots/aw_atmosphere_fourpanel_00004.png`
- `mhd_vibe_validation/plots/aw_atmosphere_vperp_vs_r.png`
- `mhd_vibe_validation/plots/aw_atmosphere_bperp_vs_r.png`
- `mhd_vibe_validation/plots/aw_atmosphere_background.png`
- `mhd_vibe_validation/plots/aw_atmosphere_elsasser_vs_r.png`
- `mhd_vibe_validation/plots/loop_advect_fourpanel_00000.png`
- `mhd_vibe_validation/plots/loop_advect_fourpanel_00001.png`
- `mhd_vibe_validation/plots/loop_advect_fourpanel_00002.png`
- `mhd_vibe_validation/plots/loop_advect_fourpanel_00003.png`
- `mhd_vibe_validation/plots/loop_advect_fourpanel_00004.png`

Each four-panel figure shows true physical r-theta R-Z, true physical r-phi
X-Y, projected constant-r theta-phi, and raw logical r-theta. Magnetic-field
directions are overlaid with normalized projected quiver arrows.

## Radial Alfven Atmosphere

Run command:

```bash
env -i PATH=/opt/cuda/12.9.1/bin:/usr/bin:/bin \
  LD_LIBRARY_PATH=/opt/cuda/12.9.1/lib64:/usr/lib64 HOME=/tmp USER=squjo23p \
  build-mhd-vibe-gpu-gcc/src/athena \
  -i inputs/tests/sph_shell_mhd_radial_alfven_bin_vibe.athinput \
  -d mhd_vibe_validation/data/aw_bin_vibe
```

Final diagnostics at `t=12`, cycle 2387:

- `<r>_|dB|^2 = 2.685043`
- WKB ray radius `r_WKB = 2.775250`
- centroid lag `-9.02e-2`, about `-7.1%` of travelled distance
- peak `|dB_phi|/(eps*B0) = 3.04e-3`
- peak `|dv_phi| = 1.49e-6`
- `divB`: L1 `1.09e-13`, Linf `4.68e-12`, normalized `1.30e-13`
- Runtime: 226.5 s, throughput `6.63e7` zone-cycles/s

Visual result: the packet is cleanly radial in true spherical geometry and
the slice orientations are correct. The late-time amplitude is strongly
damped by PLM for this carrier (`k_r=35`, `sigma_r=0.22`), so the current
bin-vibe input is useful as an output/geometry diagnostic but too dissipative
for a quantitative WKB amplitude-conservation benchmark. A lower carrier,
higher radial resolution, or higher-order reconstruction should be used for
that stricter test.

The profile plots show:

- `B_r ~ 1/r^2`
- constant `rho`
- `v_A ~ 1/r^2`
- area expansion proportional to `r^2`
- both candidate Elsasser amplitudes `z1`, `z2` for sign-convention checks

## Loop Advection

Run command:

```bash
env -i PATH=/opt/cuda/12.9.1/bin:/usr/bin:/bin \
  LD_LIBRARY_PATH=/opt/cuda/12.9.1/lib64:/usr/lib64 HOME=/tmp USER=squjo23p \
  build-mhd-vibe-gpu-gcc/src/athena \
  -i inputs/tests/sph_shell_mhd_loop_advect_bin_vibe.athinput \
  -d mhd_vibe_validation/data/loop_bin_vibe
```

Final diagnostics at `t=2*pi`, cycle 3504:

- `max|Br| = 9.27e-4`
- `max|Btheta| = 3.99e-5`
- `max|Bphi| = 1.21e-3`
- magnetic energy `sum 0.5 B^2 V = 7.41e-7`
- `Omega = 0.25`, `phi advected by = 1.570796 rad`
- `divB`: L1 `6.36e-17`, Linf `3.41e-15`, normalized `2.09e-14`
- Runtime: 148.6 s, throughput `7.42e7` zone-cycles/s

Visual result: the loop is visible in the r-phi and constant-r projected
surface slices at the expected advected phase. The chosen fixed-phi
r-theta slice only intersects the loop at some times, which is expected for
this non-axisymmetric check. CT keeps `divB` small.

## Deferred Items

- Alfven wave on isothermal Parker wind was not implemented in this round.
  This avoids evolving an isothermal Parker background under the current
  ideal-gas MHD build and overinterpreting it as a Parker preservation test.
- Heinemann-Olbert wave-action diagnostics were therefore not run. The plot
  script already computes `z1` and `z2` for the static radial atmosphere.
- Rotating Parker / Parker-spiral was also deferred.
- No MHD performance benchmarking was attempted.

## Recommendations

1. For quantitative radial Alfven amplitude work, rerun with a less
   dissipative carrier/reconstruction choice and compare WKB energy flux.
2. Add an isothermal MHD build or a clearly supported isothermal MHD EOS path
   before implementing Parker-wind Alfven preservation tests.
3. If exact field-line appearance matters, add a small 2D streamfunction or
   field-line integrator for the slices; the current overlays are projected
   direction quivers only.
4. Run one MPI GPU/native-bin slice smoke when multiple GPUs are available.

## Round 2 Radial Alfven WKB Power-Law Check

Input:

- `inputs/tests/sph_shell_mhd_radial_alfven_wkb_powerlaw.athinput`

Purpose:

- thin theta-phi tube: `theta = pi/2 +/- 0.08`, `phi in [0, 0.16]`
- power-law radial grid: `radial_grid=power_law`, `r_grid_alpha=1/3`
- 20,480 radial cells, 16 x 16 angular cells, meshblocks 64 x 8 x 8
- WKB phase in Alfven travel time `tau = int dr/v_A`
- carrier frequency `omega=6`, `sigma_tau=3`
- initial amplitude model `sqrt_va`, so expected `z_+ ~ sqrt(v_A) ~ r_c/r`
- low gas pressure `p0=1e-5` so the CFL follows the Alfven speed rather than
  a hot sound speed in the tiny outer radial cells

The first attempt used `p0=0.1`; it had `dt ~ 8.4e-5` and was stopped because
it would have taken many hours. The retuned low-pressure run completed in
734 s, close to the requested 15-minute target.

Run command:

```bash
env -i PATH=/opt/cuda/12.9.1/bin:/usr/bin:/bin \
  LD_LIBRARY_PATH=/opt/cuda/12.9.1/lib64:/usr/lib64 HOME=/tmp USER=squjo23p \
  build-mhd-vibe-gpu-gcc/src/athena \
  -i inputs/tests/sph_shell_mhd_radial_alfven_wkb_powerlaw.athinput \
  -d mhd_vibe_validation/data/aw_wkb_powerlaw_lowcs
```

Resolution check for the WKB carrier:

| radius | local dr | local wavelength | points/wavelength |
|---:|---:|---:|---:|
| 2.20 | 3.87e-3 | 1.08e-1 | 27.9 |
| 3.01 | 1.39e-3 | 5.80e-2 | 41.8 |
| 3.52 | 8.79e-4 | 4.22e-2 | 48.1 |
| 3.92 | 6.56e-4 | 3.41e-2 | 52.0 |
| 4.25 | 5.29e-4 | 2.90e-2 | 54.8 |
| 4.53 | 4.47e-4 | 2.55e-2 | 57.0 |

The minimum over `r in [2, 4.8]` is about 23.6 points per wavelength.

Final diagnostics at `t=55`, cycle 12678:

- WKB ray: `r_WKB = 4.533057`
- measured `|dB|^2` centroid: `4.532773`
- centroid error: `-2.84e-4`, relative `-1.22e-4`
- expected WKB amplitude factor: `r_c/r = 0.4853`
- peak `|dB_phi|/(eps*B0) = 0.5570`
- peak `|dB_phi| / WKB expected = 1.148`
- `divB`: L1 `6.80e-12`, Linf `1.51e-10`, normalized `1.26e-13`
- throughput: `9.06e7` zone-cycles/s

Post-processing:

- `mhd_vibe_validation/plots/aw_wkb_powerlaw_wkb_metrics.txt`
- `mhd_vibe_validation/plots/aw_wkb_powerlaw_wkb_dissipation.png`
- `mhd_vibe_validation/plots/aw_wkb_powerlaw_wkb_z_profiles.png`
- `mhd_vibe_validation/plots/aw_wkb_powerlaw_fourpanel_00000.png` ...
  `aw_wkb_powerlaw_fourpanel_00005.png`

WKB amplitude comparison from the signed outgoing variable
`z_out = v_phi - B_phi/sqrt(rho)`:

| time | r_WKB | measured centroid | coherent projection | RMS amplitude / expected |
|---:|---:|---:|---:|---:|
| 0.0 | 2.200 | 2.177 | 1.000 | 1.000 |
| 11.0 | 3.006 | 2.999 | 1.132 | 1.147 |
| 22.0 | 3.521 | 3.518 | 1.184 | 1.207 |
| 33.0 | 3.918 | 3.917 | 1.204 | 1.235 |
| 44.0 | 4.248 | 4.247 | 1.206 | 1.246 |
| 55.0 | 4.533 | 4.533 | 1.197 | 1.246 |

Interpreting the user's requested `sqrt(v_A)` scaling literally, this run
does not show net numerical damping of the WKB-scaled amplitude. It shows the
opposite: by the final time the RMS signed amplitude is about 25% above the
`sqrt(v_A)` expectation, while the peak amplitude is about 15% above it.
The packet position is excellent, so the main discrepancy is amplitude
normalisation/profile evolution, not group speed. The relative L2 residual
against the full expected signed profile grows to about 0.40 by `t=55`,
indicating measurable profile/phase distortion even though the integrated
amplitude is not damped below the `sqrt(v_A)` reference.
