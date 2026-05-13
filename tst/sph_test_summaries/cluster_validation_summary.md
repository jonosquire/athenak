# AthenaK spherical-shell cluster validation summary

Date: 2026-05-11
Worktree: `/weka/scratch/squjo23p/athenak-tests/athenak`
Git commit: `6940a5e0a4ad3bacce1321920e96c5cce440917f`

## Environment

- Cluster allocation: Slurm GPU node, partition reported as `aoraki_gpu_H100`
- GPU: NVIDIA H100 80GB HBM3, driver `550.54.15`
- Modules used:
  - `cuda/12.9.1`
  - `openmpi/4.1.8`
- CUDA compiler: `nvcc` 12.9.86
- Host compiler: `/usr/bin/gcc`, `/usr/bin/g++`, GCC 11.4.1
- CMake: 3.26.5
- Python plotting: `/home/squjo23p/conda-envs/curmpy/bin/python`, run with
  `LD_LIBRARY_PATH=/home/squjo23p/conda-envs/curmpy/lib:$LD_LIBRARY_PATH`

Notes:
- The default `cc`/`g++` path went through ccache, which could not write to
  `/home/squjo23p/.cache/ccache` in this environment. Successful builds used
  explicit `/usr/bin/gcc` and `/usr/bin/g++`.
- MPI-enabled executables inherited Slurm PMI variables that caused direct
  `MPI_Init` failure. Runtime commands used a clean `env -i ...` environment.

## Builds

### Spherical CPU/MPI

Build directory: `build-cluster-sph-cpu-gcc`

```bash
module purge
module load openmpi/4.1.8
cmake -S . -B build-cluster-sph-cpu-gcc \
  -DCMAKE_BUILD_TYPE=Release \
  -DPROBLEM=sph_shell_hydro \
  -DAthena_ENABLE_MPI=ON \
  -DCMAKE_C_COMPILER=/usr/bin/gcc \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++
cmake --build build-cluster-sph-cpu-gcc --target athena -j 8
```

### Spherical GPU/MPI

Build directory: `build-cluster-sph-gpu-gcc`

```bash
module purge
module load cuda/12.9.1 openmpi/4.1.8
NVCC_WRAPPER_DEFAULT_COMPILER=/usr/bin/g++ cmake -S . -B build-cluster-sph-gpu-gcc \
  -DCMAKE_BUILD_TYPE=Release \
  -DPROBLEM=sph_shell_hydro \
  -DAthena_ENABLE_MPI=ON \
  -DKokkos_ENABLE_CUDA=ON \
  -DKokkos_ARCH_HOPPER90=ON \
  -DCMAKE_CXX_COMPILER=$PWD/kokkos/bin/nvcc_wrapper \
  -DCMAKE_C_COMPILER=/usr/bin/gcc
cmake --build build-cluster-sph-gpu-gcc --target athena -j 8
```

### Cartesian CPU/GPU

- CPU/MPI build: `build-cluster-cart-cpu-gcc`, `-DPROBLEM=built_in_pgens`
- GPU/MPI build: `build-cluster-cart-gpu-gcc`, `-DPROBLEM=built_in_pgens`,
  `Kokkos_ENABLE_CUDA=ON`, `Kokkos_ARCH_HOPPER90=ON`

## Standard Check

Cartesian linear-wave hydro ran with the Cartesian CPU build:

- Log: `cluster_validation/logs/cpu_cart_linear_wave_hydro.log`
- Input: `inputs/tests/linear_wave_hydro.athinput`, `time/nlim=5`
- First dt values: `1.404847e-02`, `1.404851e-02`, `1.404857e-02`,
  `1.404861e-02`
- Status: pass; matches the documented regression values.

## MPI Check

Two-rank CPU MPI geometry run:

- Log: `cluster_validation/logs/mpi_cpu_sph_shell_geom_np2_split.log`
- Input: `inputs/tests/sph_shell_geom.athinput`
- Overrides: `meshblock/nx1=32 meshblock/nx2=32 meshblock/nx3=16`
- Result: 2 MeshBlocks on 2 ranks, geometry relative error `3.409821e-15`
- Status: pass.

An earlier unsplit two-rank attempt is preserved in
`cluster_validation/logs/mpi_cpu_sph_shell_geom_np2.log`; it failed correctly
because the input had only one MeshBlock for two ranks.

## Formal GPU Tests

All formal tests below ran with `build-cluster-sph-gpu-gcc/src/athena`.
Logs are under `cluster_validation/logs/`, outputs under
`cluster_validation/data/formal_gpu/`.

| Test | Log | Key result | Status |
| --- | --- | --- | --- |
| Spherical smoke | `gpu_standard_sph_smoke.log` | uniform drift zero, `max|v|=4.36e-18` | pass |
| Geometry uniform | `gpu_geom_uniform.log` | volume relative error `0.0` | pass |
| Geometry log-r | `gpu_geom_logr.log` | volume relative error `7.06e-16` | pass |
| Geometry power-law | `gpu_geom_powerlaw.log` | volume relative error `1.18e-16` | pass |
| Uniform | `gpu_uniform.log` | `max|drho|=0`, `max|dp|=0`, `max|v|=9.86e-17` | pass |
| Uniform log-r | `gpu_uniform_logr.log` | `max|drho|=0`, `max|dp|=0`, `max|v|=1.31e-16` | pass |
| Divergence | `gpu_divergence.log` | L1: `2.61e-15`, `1.85e-4`, `9.12e-4` | pass |
| Divergence log-r | `gpu_divergence_logr.log` | L1: `1.16e-2`, `2.16e-6`, `7.48e-4` | pass |
| Radial acoustic | `gpu_radial_acoustic.log` | speed error `7.87e-3` | pass |
| Radial acoustic log-r | `gpu_radial_acoustic_logr.log` | speed error `1.88e-2` | pass |
| Homologous | `gpu_homologous.log` | density error `9.41e-7` | pass |
| Hydrostatic const-g | `gpu_hydrostatic_constg.log` | `max|v|/cs=1.36e-1`, expected drift | pass |
| Hydrostatic 1/r2 | `gpu_hydrostatic_r2.log` | `max|v|/cs=1.02e-1`, expected drift | pass |
| Solid-body rotation | `gpu_solid_body_rotation.log` | relative source errors `8.6e-4`, `8.7e-4`, `1.0e-3` | pass |
| Keplerian orbit, t=1 | `gpu_keplerian_orbit_t1.log` | `dLz/Lz=1.236e-1`, `dM/M=1.328e-1`, `max|vr|/|vphi|=2.76e-2` | pass |
| Parker 1D | `gpu_parker_isothermal_1d.log` | L1 `|dv/v|=2.64e-2`, L1 `|drho/rho|=4.87e-3` | pass |

Issue preserved:
- Default `inputs/tests/sph_shell_keplerian_orbit.athinput` runs to `tlim=5.0`
  and produced `NaN` in mass and angular momentum diagnostics on GPU:
  `cluster_validation/logs/gpu_keplerian_orbit.log`.
- The documented short-run diagnostic at `time/tlim=1.0` remains finite and
  matches the expected Task 3B behaviour. Treat the default `tlim=5.0` as too
  long for the current outflow-boundary rotating wedge.

## Vibe Simulations

Run script: `cluster_validation/scripts/run_vibe_gpu.sh`

| Case | Input | Data dir | Log | Notes |
| --- | --- | --- | --- | --- |
| 3D acoustic pulse | `sph_shell_3d_acoustic_pulse.athinput` | `cluster_validation/data/pulse_3d` | `gpu_vibe_pulse_3d.log` | wavefront radius `0.3069` vs `cs*t=0.3000` |
| Oblique packet | `sph_shell_oblique_wave_packet.athinput` | `cluster_validation/data/oblique_packet` | `gpu_vibe_oblique_packet.log` | projected drift `0.338` vs `0.400`, documented late-time bias |
| Thin disk | `sph_shell_thin_disk_vibe.athinput`, `time/tlim=0.7` | `cluster_validation/data/thin_disk` | `gpu_vibe_thin_disk.log` | equatorial centroid preserved at `pi/2`; radial adjustment expected |
| Parker 3D | `sph_shell_parker_isothermal_3d.athinput` | `cluster_validation/data/parker` | `gpu_vibe_parker.log` | L1 `|dv/v|=2.77e-2`, L1 `|drho/rho|=5.29e-3` |

## Plots

Plot script: `cluster_validation/scripts/plot_vibe_checks.py`

Each four-panel figure contains:
1. r-theta slice,
2. r-phi slice,
3. theta-phi slice,
4. mapped r-theta slice using `R=r sin(theta)`, `Z=r cos(theta)`.

Saved plots:

- Pulse: `pulse_3d_pulse_3d_00000.png`, `pulse_3d_pulse_3d_00003.png`,
  `pulse_3d_pulse_3d_00006.png`
- Oblique packet: `oblique_packet_oblique_packet_00000.png`,
  `oblique_packet_oblique_packet_00004.png`,
  `oblique_packet_oblique_packet_00008.png`
- Thin disk density: `thin_disk_thin_disk_rho_00000.png`,
  `thin_disk_thin_disk_rho_00004.png`, `thin_disk_thin_disk_rho_00007.png`
- Thin disk v_phi: `thin_disk_thin_disk_vphi_00000.png`,
  `thin_disk_thin_disk_vphi_00004.png`, `thin_disk_thin_disk_vphi_00007.png`
- Parker density: `parker_parker_rho_00000.png`,
  `parker_parker_rho_00003.png`, `parker_parker_rho_00005.png`
- Parker v_r: `parker_parker_vr_00000.png`, `parker_parker_vr_00003.png`,
  `parker_parker_vr_00005.png`
- Parker profiles: `parker_vr_profiles.png`, `parker_rho_profiles.png`,
  `parker_massflux_profiles.png`, `parker_angular_residuals.png`

Visual inspection notes:
- Pulse mapped slice shows a round wavefront in physical meridional
  coordinates; no obvious block seam or spoke artefact.
- Oblique packet remains coherent; the late-time centroid shortfall matches
  the documented boundary/centroid bias.
- Thin disk remains centered on the equator over the shortened run and shows
  expected adjustment waves, not a collapse.
- Parker is angularly quiet; theta-phi slices are nearly uniform at fixed r.

## Performance Sanity

Performed after formal and vibe validation.

Both runs used GPU/MPI builds, one rank, 64x32x32 cells, 100 cycles, HLLC, PLM,
and output intervals set to `1000`.

| Case | Log | Throughput |
| --- | --- | --- |
| Spherical shell, pulse pgen | `perf_gpu_spherical_64x32x32.log` | `5.672544e+07` zone-cycles/cpu_second |
| Cartesian, linear-wave pgen | `perf_gpu_cartesian_64x32x32.log` | `6.474589e+07` zone-cycles/cpu_second |

Ratio: spherical / Cartesian = `0.876`. This is within the expected sanity
range; no catastrophic GPU bottleneck is evident.

## Issues and Recommendations

- Fix or shorten the default Keplerian input before treating it as a formal
  long-run pass. At `t=5.0` it reaches NaN in the current GPU run; at `t=1.0`
  it matches the documented finite short-run behaviour.
- Build warnings are from existing broad AthenaK files (`bvals_part`,
  `track_prtcl`, `athena_tensor`, unused variables in pgen/srcterms). No
  CUDA runtime or Kokkos runtime errors appeared in the validation logs.
- The log-r divergence test was run at the shipped resolution only. For a
  formal convergence table, rerun the documented doubled-resolution log-r
  sweep.
- Keep using a clean runtime environment for MPI builds on this allocation, or
  build with an MPI stack that matches the Slurm PMI/PMIx setup.
