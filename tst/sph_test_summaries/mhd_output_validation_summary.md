# Spherical-Shell MHD Monopole and Binary Output Check

Date: 2026-05-12

## Environment and Build

- Git commit at test start: `ec9aff518ce5813bdcb663bae871667ed2a1540c`
- Modules loaded: none
- Compiler: GCC 11.3.1
- MPI: OpenMPI 4.1.8 from `/opt/openmpi/4.1.8`
- CPU/Serial build: `build-mhd-output-cpu-gcc2`
- CPU/MPI build: `build-mhd-output-cpu-mpi-gcc`
- Serial CMake: `cmake -S . -B build-mhd-output-cpu-gcc2 -DCMAKE_BUILD_TYPE=Release -DPROBLEM=sph_shell_mhd -DKokkos_ENABLE_SERIAL=ON -DCMAKE_CXX_COMPILER=/usr/bin/g++ -DCMAKE_C_COMPILER=/usr/bin/gcc`
- MPI CMake: `cmake -S . -B build-mhd-output-cpu-mpi-gcc -DCMAKE_BUILD_TYPE=Release -DPROBLEM=sph_shell_mhd -DKokkos_ENABLE_SERIAL=ON -DAthena_ENABLE_MPI=ON -DCMAKE_CXX_COMPILER=/opt/openmpi/4.1.8/bin/mpicxx -DCMAKE_C_COMPILER=/usr/bin/gcc`

## Monopole Residual Investigation

The original `sph_shell_mhd_monopole.athinput` used radial outflow
boundaries. That copied `B_r` into radial ghost zones, breaking the analytic
`B_r proportional to 1/r^2` profile at the radial boundaries. The resulting
boundary-layer Lorentz force produced the reported percent-level velocity
while the face-area divB diagnostic stayed at roundoff.

Pre-fix reference runs:

| case | max\|v\|/cs | note |
|------|-------------|------|
| 64x32x16, nlim=1 | 1.918e-3 | residual appears immediately |
| 64x32x16, t=0.14 | 3.942e-2 | boundary-localized at inner radial edge |
| 128x64x32, t=0.14 | 4.071e-2 | no fixed-time convergence |

The maximum velocity was at the inner radial boundary. Excluding radial
boundary layers in the 64x32x16, t=0.14 binary output reduced max\|v\|/cs
from 3.94e-2 to 1.21e-2 after 4 cells, 1.59e-3 after 6 cells, and 1.93e-5
after 8 cells.

## Fix

`src/pgen/sph_shell_mhd.cpp` now installs a monopole-only radial user BC.
It fills radial ghost faces with analytic `B_r = B0 r_ref^2 / r^2`, zeros
transverse face fields in those radial ghosts, and fills conserved ghost
cells with the static rho/p/v state plus magnetic energy from the local
face-averaged radial field. The standard uniform-r and log-r monopole input
files now use `ix1_bc=user` and `ox1_bc=user`.

## Post-Fix Results

All post-fix runs kept `max|B1f - analytic|` at roundoff and divB at machine
precision.

| grid | radial grid | t/cycles | max\|v\|/cs | divB L1 | divB Linf |
|------|-------------|----------|-------------|---------|-----------|
| 64x32x16 | uniform | nlim=1 | 4.533e-6 | 1.006e-15 | 1.011e-14 |
| 32x16x8 | uniform | t=0.14 | 1.811e-4 | 5.511e-16 | 5.774e-15 |
| 64x32x16 | uniform | t=0.14 | 4.139e-5 | 1.288e-15 | 1.513e-14 |
| 128x64x32 | uniform | t=0.14 | 9.570e-6 | 3.073e-15 | 7.507e-14 |
| 32x16x8 | log | t=0.14 | 2.790e-4 | 2.401e-16 | 3.779e-15 |
| 64x32x16 | log | t=0.14 | 6.373e-5 | 5.168e-16 | 1.113e-14 |

Uniform-r convergence in max velocity is about second order: the 32->64 and
64->128 ratios are 4.38 and 4.32. Log-r shows the same 32->64 ratio, with a
slightly larger residual at the same nominal resolution.

Sensitivity checks at 64x32x16, t=0.14:

| variant | max\|v\|/cs |
|---------|-------------|
| PLM, CFL 0.3 | 4.139e-5 |
| PLM, CFL 0.15 | 4.140e-5 |
| donor-cell | 4.010e-5 |
| PPM4 | 4.162e-5 |

The residual is not driven by timestep/source ordering and has little
reconstruction dependence. It now looks like a normal spatial truncation
residual in the discrete force balance.

## Binary Output Inspection

Native AthenaK `bin` output works for spherical-shell MHD with
`variable=mhd_w_bcc`. The existing `vis/python/bin_convert.py` reader exposes:

- time and cycle;
- root grid dimensions and MeshBlock layout;
- variables `dens, velx, vely, velz, eint, bcc1, bcc2, bcc3`;
- the full input header, including `<coord>/system=spherical_shell` and
  `<spherical_shell>/radial_grid` when present.

No binary writer change was needed. Face-centred B is not written by
`mhd_w_bcc`; the simple plots use cell-centred `bcc*`.

Validation script:

`mhd_output_validation/scripts/inspect_mhd_bin.py`

Generated plots:

- `mhd_output_validation/plots/monopole_fix_t014_bmag_rtheta.png`
- `mhd_output_validation/plots/monopole_fix_t014_vmag_rtheta.png`
- `mhd_output_validation/plots/monopole_fix_t014_bmag_rz.png`
- `mhd_output_validation/plots/toroidal_bin_n1_bmag_rtheta.png`
- `mhd_output_validation/plots/toroidal_bin_n1_vmag_rtheta.png`
- `mhd_output_validation/plots/toroidal_bin_n1_bmag_rz.png`
- `mhd_output_validation/plots/toroidal_bin_mpi_np2_bmag_rtheta.png`
- `mhd_output_validation/plots/toroidal_bin_mpi_np2_vmag_rtheta.png`
- `mhd_output_validation/plots/toroidal_bin_mpi_np2_bmag_rz.png`

## MPI Output Smoke

A 2-rank CPU/MPI toroidal-static binary-output smoke completed after running
outside the sandboxed MPI interface restrictions:

`mpirun -np 2 build-mhd-output-cpu-mpi-gcc/src/athena -i mhd_output_validation/scripts/sph_shell_mhd_toroidal_bin.athinput -d mhd_output_validation/data/toroidal_bin_mpi meshblock/nx1=16`

The run wrote a single readable binary file containing two MeshBlocks. The
same reader script loaded it and reported:

- time = 1.44831125e-02, cycle = 1
- dimensions = 32x16x8, n_mbs = 2
- max\|v\|/cs0 = 1.1465e-4
- coordinate_system = spherical_shell

## Remaining Concerns

- The monopole is now clean enough for the next MHD validation stage, but
  the residual should be rechecked on GPU after the GPU MHD build exists.
- Native `mhd_w_bcc` bin output does not expose face-centred magnetic fields;
  use the pgen diagnostics for exact `B1f` preservation or add a dedicated
  output variable later if face-field plots are needed.
- No high-resolution Alfvén-wave or field-loop vibe runs were started in
  this task.

## Recommended Next Prompt

Proceed to high-resolution GPU MHD vibe tests with native bin slices:
radial Alfvén pulse and magnetic loop advection, with field-line overlays
from `bcc*` and pgen divB diagnostics. Keep MHD performance comparisons as a
separate later task.
