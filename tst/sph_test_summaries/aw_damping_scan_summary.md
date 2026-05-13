# Alfven Damping Scan Summary

## Environment

- Executable: `/weka/scratch/squjo23p/athenak-tests/athenak/build-aw-scan-mhd-gpu-gcc/src/athena`
- GPU: NVIDIA H200 NVL, 143771 MiB (from nvidia-smi)
- Build: `build-aw-scan-mhd-gpu-gcc`, Release, Kokkos CUDA, `Kokkos_ARCH_HOPPER90=ON`, MPI disabled
- Problem generator: `sph_shell_mhd`

## Run Matrix

| case | status | cpu seconds | zone-cycles/s |
|---|---|---:|---:|
| monopole_uniform_plm_nr256 | ok | 0.11 | 1.148e+07 |
| monopole_uniform_plm_nr512 | ok | 0.22 | 2.273e+07 |
| monopole_uniform_plm_nr1024 | ok | 0.49 | 4.150e+07 |
| monopole_uniform_plm_nr2048 | ok | 1.34 | 6.048e+07 |
| monopole_uniform_ppmx_nr256 | ok | 0.14 | 9.377e+06 |
| monopole_uniform_ppmx_nr512 | ok | 0.28 | 1.812e+07 |
| monopole_uniform_ppmx_nr1024 | ok | 0.61 | 3.342e+07 |
| monopole_uniform_ppmx_nr2048 | ok | 1.76 | 4.588e+07 |
| monopole_uniform_dc_nr512 | ok | 0.21 | 2.419e+07 |
| monopole_uniform_ppm4_nr512 | ok | 0.24 | 2.133e+07 |
| monopole_uniform_wenoz_nr512 | ok | 0.27 | 1.853e+07 |
| parker_uniform_plm_nr256 | ok | 0.54 | 9.823e+06 |
| parker_uniform_plm_nr512 | ok | 0.61 | 1.926e+07 |
| parker_uniform_plm_nr1024 | ok | 1.28 | 3.658e+07 |
| parker_log_plm_nr512 | ok | 1.27 | 1.983e+07 |
| parker_log_plm_nr1024 | ok | 2.73 | 3.696e+07 |
| parker_uniform_ppmx_nr256 | ok | 0.67 | 7.923e+06 |
| parker_uniform_ppmx_nr512 | ok | 0.75 | 1.549e+07 |
| parker_uniform_ppmx_nr1024 | ok | 1.60 | 2.928e+07 |
| parker_log_ppmx_nr512 | ok | 1.59 | 1.584e+07 |
| parker_log_ppmx_nr1024 | ok | 3.43 | 2.933e+07 |
| aw_powerlaw_plm_nr4096 | ok | 111.71 | 8.767e+07 |
| aw_powerlaw_plm_nr8192 | ok | 59.95 | 1.098e+08 |
| aw_powerlaw_plm_nr16384 | ok | 96.41 | 1.221e+08 |
| aw_powerlaw_ppmx_nr4096 | ok | 37.45 | 6.284e+07 |
| aw_powerlaw_ppmx_nr8192 | ok | 73.23 | 7.553e+07 |
| aw_powerlaw_ppmx_nr16384 | ok | 136.03 | 8.079e+07 |

## Static Monopole

Uniform-r static monopole remained quiet and second-order convergent in the residual velocity. PLM and PPMX were effectively identical.

| case | max |v|/cs | divB Linf*h/|B|max |
|---|---:|---:|
| monopole_uniform_plm_nr256 | 6.007996e-07 | 1.754650e-15 |
| monopole_uniform_plm_nr512 | 1.507526e-07 | 9.652661e-16 |
| monopole_uniform_plm_nr1024 | 3.773780e-08 | 8.877510e-16 |
| monopole_uniform_plm_nr2048 | 9.441006e-09 | 5.883275e-16 |
| monopole_uniform_ppmx_nr256 | 6.002026e-07 | 2.809577e-15 |
| monopole_uniform_ppmx_nr512 | 1.506138e-07 | 1.249341e-15 |
| monopole_uniform_ppmx_nr1024 | 3.773227e-08 | 9.258629e-16 |
| monopole_uniform_ppmx_nr2048 | 9.439573e-09 | 4.175902e-15 |

At N_r=512, DC/PLM/PPM4/PPMX/WENOZ all landed near max |v|/cs = 1.5e-7; reconstruction choice did not dominate the monopole residual.

## Parker Preservation

The Parker cases use `<mhd>/eos=isothermal` with analytic radial user BCs and a passive radial monopole field. Log-r is much cleaner than uniform-r for the same N_r, and PPMX reduces the mass-flux error relative to PLM.

| case | L1 mdot error | L1 rho error | L1 vr error | divB Linf*h/|B|max |
|---|---:|---:|---:|---:|
| parker_uniform_plm_nr1024 | 1.591002e-04 | 9.667218e-05 | 6.309391e-05 | 1.990325e-15 |
| parker_uniform_ppmx_nr1024 | 5.556764e-05 | 3.513036e-05 | 2.044449e-05 | 7.901360e-15 |
| parker_log_plm_nr1024 | 6.915878e-06 | 4.279711e-06 | 3.075073e-06 | 1.223214e-14 |
| parker_log_ppmx_nr1024 | 1.557231e-06 | 1.218070e-06 | 5.177180e-07 | 2.783847e-14 |

## AW Packet

The radial Alfven packet uses a thin theta-phi tube, `radial_grid=power_law`, `r_grid_alpha=1/3`, and a travel-time carrier with `omega=6`. The 8192 and 16384 cases correspond to about 25 and 50 points per wavelength in Alfven travel-time coordinate.

| case | ppw | centroid | peak/expected WKB | action frac | sqrt(A-/A+) | divB Linf*h/|B|max |
|---|---:|---:|---:|---:|---:|---:|
| aw_powerlaw_plm_nr4096 | 12.62 | 4.384166 | 3.026681e-01 | -9.372423e-01 | 6.564659e-02 | 7.104825e-13 |
| aw_powerlaw_plm_nr8192 | 25.22 | 4.425125 | 4.228148e-02 | -9.986937e-01 | 2.860966e-03 | 1.381816e-13 |
| aw_powerlaw_plm_nr16384 | 50.42 | 4.539304 | 7.748942e-01 | -2.893854e-01 | 1.527879e-03 | 9.416661e-14 |
| aw_powerlaw_ppmx_nr4096 | 12.62 | 4.394116 | 3.612707e-01 | -9.232015e-01 | 2.630596e-02 | 5.746349e-14 |
| aw_powerlaw_ppmx_nr8192 | 25.22 | 4.541588 | 3.566167e-01 | -8.449676e-01 | 3.806709e-02 | 9.129181e-14 |
| aw_powerlaw_ppmx_nr16384 | 50.42 | 4.526798 | 1.041338e+00 | 4.121756e-01 | 2.459257e-02 | 7.337611e-14 |

The expected WKB amplitude scaling is `z+ proportional to sqrt(v_A)`, equivalently `sqrt(B_r)` here. The `peak/expected WKB` column is the measured peak signed Elsasser amplitude divided by `2 eps B0 sqrt(v_A(r_peak)/v_A(r_c))`.

The PLM N_r=8192 damping anomaly is reproducible: it reran with the same severe damping. Treat it as a real numerical-method/input interaction until a narrower follow-up isolates whether this is PLM phase error, packet/grid commensurability, or the current action-window definition. The 16384-cell PLM and PPMX cases are much cleaner by the peak-amplitude metric.

## Reconstruction Cost

This is a rough read from existing logs, not a careful performance scan. The cleanest all-method comparison is the static monopole N_r=512 case, where every method used the same 512x8x8 grid and 64x8x8 MeshBlocks.

| method | case | zone-cycles/s | throughput / PLM | cost vs PLM |
|---|---|---:|---:|---:|
| DC | monopole_uniform_dc_nr512 | 2.419e+07 | 1.064 | 0.940x |
| PLM | monopole_uniform_plm_nr512 | 2.273e+07 | 1.000 | 1.000x |
| PPM4 | monopole_uniform_ppm4_nr512 | 2.133e+07 | 0.939 | 1.065x |
| PPMX | monopole_uniform_ppmx_nr512 | 1.812e+07 | 0.797 | 1.254x |
| WENOZ | monopole_uniform_wenoz_nr512 | 1.853e+07 | 0.815 | 1.227x |

Matched PLM/PPMX pairs in the Parker and AW scans give similar or larger PPMX costs:

| pair | PLM zc/s | PPMX zc/s | PPMX cost vs PLM |
|---|---:|---:|---:|
| Parker uniform N_r=1024 | 3.658e+07 | 2.928e+07 | 1.249x |
| Parker log N_r=1024 | 3.696e+07 | 2.933e+07 | 1.260x |
| AW power-law N_r=16384 | 1.221e+08 | 8.079e+07 | 1.511x |

Takeaway: PPM4 is only about 7 percent slower than PLM in the quick all-method case; PPMX and WENOZ are about 23-25 percent slower there. In the longer AW packet run, PPMX costs about 1.5x PLM, but gives a much better WKB peak-amplitude result at 50 ppw.

## Diagnostics Table

| case | test | recon | grid | N_r | final metric |
|---|---|---|---|---:|---:|
| monopole_uniform_plm_nr256 | monopole | plm | uniform | 256 | 6.007996e-07 |
| monopole_uniform_plm_nr512 | monopole | plm | uniform | 512 | 1.507526e-07 |
| monopole_uniform_plm_nr1024 | monopole | plm | uniform | 1024 | 3.773780e-08 |
| monopole_uniform_plm_nr2048 | monopole | plm | uniform | 2048 | 9.441006e-09 |
| monopole_uniform_ppmx_nr256 | monopole | ppmx | uniform | 256 | 6.002026e-07 |
| monopole_uniform_ppmx_nr512 | monopole | ppmx | uniform | 512 | 1.506138e-07 |
| monopole_uniform_ppmx_nr1024 | monopole | ppmx | uniform | 1024 | 3.773227e-08 |
| monopole_uniform_ppmx_nr2048 | monopole | ppmx | uniform | 2048 | 9.439573e-09 |
| monopole_uniform_dc_nr512 | monopole | dc | uniform | 512 | 1.502998e-07 |
| monopole_uniform_ppm4_nr512 | monopole | ppm4 | uniform | 512 | 1.507320e-07 |
| monopole_uniform_wenoz_nr512 | monopole | wenoz | uniform | 512 | 1.511591e-07 |
| parker_uniform_plm_nr256 | parker | plm | uniform | 256 | 2.816680e-03 |
| parker_uniform_plm_nr512 | parker | plm | uniform | 512 | 6.589653e-04 |
| parker_uniform_plm_nr1024 | parker | plm | uniform | 1024 | 1.591002e-04 |
| parker_log_plm_nr512 | parker | plm | log | 512 | 2.776935e-05 |
| parker_log_plm_nr1024 | parker | plm | log | 1024 | 6.915878e-06 |
| parker_uniform_ppmx_nr256 | parker | ppmx | uniform | 256 | 1.275728e-03 |
| parker_uniform_ppmx_nr512 | parker | ppmx | uniform | 512 | 2.423507e-04 |
| parker_uniform_ppmx_nr1024 | parker | ppmx | uniform | 1024 | 5.556764e-05 |
| parker_log_ppmx_nr512 | parker | ppmx | log | 512 | 6.273258e-06 |
| parker_log_ppmx_nr1024 | parker | ppmx | log | 1024 | 1.557231e-06 |
| aw_powerlaw_plm_nr4096 | aw_packet | plm | power_law | 4096 | -9.372423e-01 |
| aw_powerlaw_plm_nr8192 | aw_packet | plm | power_law | 8192 | -9.986937e-01 |
| aw_powerlaw_plm_nr16384 | aw_packet | plm | power_law | 16384 | -2.893854e-01 |
| aw_powerlaw_ppmx_nr4096 | aw_packet | ppmx | power_law | 4096 | -9.232015e-01 |
| aw_powerlaw_ppmx_nr8192 | aw_packet | ppmx | power_law | 8192 | -8.449676e-01 |
| aw_powerlaw_ppmx_nr16384 | aw_packet | ppmx | power_law | 16384 | 4.121756e-01 |

## Plots

- `plots/monopole_spurious_velocity_vs_time.png`
- `plots/monopole_convergence.png`
- `plots/parker_mdot_error_vs_resolution.png`
- `plots/aw_action_fraction_vs_time.png`
- `plots/aw_zminus_ratio_vs_time.png`
- `plots/aw_dissipation_vs_ppw.png`
- `plots/aw_peak_wkb_ratio_vs_ppw.png`
- `plots/aw_action_retention_vs_ppw.png`
- `plots/monopole_final_radial_velocity_profile.png`
- `plots/parker_final_profiles_nr1024.png`
- `plots/aw_final_profiles_nr16384.png`

## Recommendations

- Use PPMX for Parker-like steady backgrounds; it is consistently lower-error than PLM in this scan.
- For the AW packet, 50 ppw in the travel-time coordinate is the first clearly acceptable point here. PPMX at N_r=16384 gives peak/expected WKB amplitude near unity; PLM is still about 22 percent low.
- The integrated action proxy is useful, but the nonmonotonic PLM result means the next scan should add a fixed window in travel-time coordinate and a driven monochromatic lower-boundary test.
- The optional driven lower-boundary monochromatic AW test was deferred in this first scan.
