# aw_damping_scan replot_v2 report

## Data Audit

- CSV files audited: 57
- Log files audited: 27
- Logs not cleanly completed or marked fatal: 0
- CSVs with NaN/Inf in numeric key columns: 0
- Native bin r-theta slices available: 141

Schemes present by test:
- aw_packet: plm, ppmx
- monopole: dc, plm, ppm4, ppmx, wenoz
- parker: plm, ppmx

The original CSVs contain enough time snapshots for monopole, Parker, and AW packet re-analysis. AW final-profile CSVs alone are insufficient for propagation diagnostics, so this pass recomputed AW profiles directly from the native bin slices.

## Parker

The Parker preservation diagnostics look robust. The cleaned plots separate uniform-r and log-r and recompute boundary-excluded errors from the radial profiles.

| case | L1 mdot | boundary-excluded L1 mdot | L1 rho | L1 v_r |
|---|---:|---:|---:|---:|
| parker_uniform_plm_nr1024 | 1.591002e-04 | 1.124205e-04 | 9.667218e-05 | 6.309391e-05 |
| parker_uniform_ppmx_nr1024 | 5.556764e-05 | 3.863787e-05 | 3.513036e-05 | 2.044449e-05 |
| parker_log_plm_nr1024 | 6.915878e-06 | 7.086086e-06 | 4.279711e-06 | 3.075073e-06 |
| parker_log_ppmx_nr1024 | 1.557231e-06 | 1.617947e-06 | 1.218070e-06 | 5.177180e-07 |

Log-r is much better for this Parker domain. At N_r=1024, log PPMX has L1 mass-flux error of order 1.6e-6, while uniform PLM is about 1.6e-4. PPMX improves both grids relative to PLM.

## Monopole

The previous monopole scheme plot hid the one-resolution schemes. The cleaned N_r=512 time plot now includes dc, plm, ppm4, ppmx, and wenoz. The convergence plot shows all available points and fitted slopes where a scheme has at least three resolutions.

| case | final max |v|/cs | boundary-excluded max | divB norm |
|---|---:|---:|---:|
| monopole_uniform_plm_nr2048 | 9.441006e-09 | 9.441006e-09 | 5.883275e-16 |
| monopole_uniform_ppmx_nr2048 | 9.439573e-09 | 9.439573e-09 | 4.175902e-15 |
| monopole_uniform_dc_nr512 | 1.502998e-07 | 1.502998e-07 | 5.604343e-16 |
| monopole_uniform_ppm4_nr512 | 1.507320e-07 | 1.507320e-07 | 2.406851e-15 |
| monopole_uniform_wenoz_nr512 | 1.511591e-07 | 1.511591e-07 | 1.782795e-15 |

The static monopole residual remains a clean second-order spatial imbalance for PLM/PPMX, with divB near roundoff. The single-resolution dc/ppm4/wenoz points are nearly coincident with PLM/PPMX at N_r=512, so reconstruction choice does not dominate this static residual.

## AW Packet

The previous integrated wave-action plot should be treated as inconclusive. In this pass, the Elsasser branch is identified from the t=0 native-bin data: z1 = v_perp - B_perp/sqrt(rho) is the initialized outward branch for all AW packet cases. The reflected branch is z2.

This pass computes several quantities instead of presenting one as definitive:

- Eplus_full = integral rho |z_out|^2 dV over the full slice domain.
- Eplus_window = the same quantity in a moving travel-time window, with a 1e-3 peak-amplitude mask.
- Aplus_window = integral rho |z_out|^2 / v_A dV in that same window.
- peak z_out / WKB expected, where WKB expected uses z_out proportional to sqrt(v_A).
- sqrt(Eminus/Eplus) for local reflected/noise production.

| case | peak/WKB | Eplus_window/f0 | Aplus_window/f0 | sqrt(Eminus/Eplus) window | centroid | width |
|---|---:|---:|---:|---:|---:|---:|
| aw_powerlaw_plm_nr4096 | 3.026681e-01 | 1.459968e-02 | 6.200939e-02 | 1.214507e-02 | 4.392951 | 0.019716 |
| aw_powerlaw_plm_nr8192 | 4.228148e-02 | 3.053673e-04 | 1.306081e-03 | 2.399297e-03 | 4.421840 | 0.085392 |
| aw_powerlaw_plm_nr16384 | 7.748942e-01 | 1.576123e-01 | 7.106024e-01 | 8.709394e-04 | 4.538200 | 0.050483 |
| aw_powerlaw_ppmx_nr4096 | 3.612707e-01 | 1.789444e-02 | 7.614514e-02 | 1.798115e-02 | 4.397048 | 0.019970 |
| aw_powerlaw_ppmx_nr8192 | 3.566167e-01 | 3.438631e-02 | 1.550305e-01 | 3.788502e-02 | 4.540445 | 0.051157 |
| aw_powerlaw_ppmx_nr16384 | 1.041338e+00 | 3.149784e-01 | 1.412204e+00 | 2.452123e-02 | 4.525536 | 0.053622 |

The radial time-evolution plots show that the packet compresses strongly as it propagates into lower v_A. That makes peak amplitude and integrated quantities sensitive to windowing and to whether the diagnostic is in r or travel-time coordinates. PPMX at N_r=16384 is close to the WKB peak amplitude, but the integrated windowed quantities still do not form a clean conservation statement. The PPMX growth seen previously is therefore diagnostic-dependent, not a trustworthy anti-damping conclusion.

z_ref/z_out profiles are masked wherever z_out is below 1e-3 of its instantaneous peak, which prevents noise outside the packet from dominating the ratio. z-minus/reflected production is small but not monotonic with scheme/resolution in this packet setup, so it should be used as a qualitative warning only.

## Plots Produced

- `plots/aw_integrated_quantities_v2.png`
- `plots/aw_packet_action_density_profiles_plm_nr16384.png`
- `plots/aw_packet_action_density_profiles_plm_nr8192.png`
- `plots/aw_packet_action_density_profiles_ppmx_nr16384.png`
- `plots/aw_packet_action_density_profiles_ppmx_nr8192.png`
- `plots/aw_packet_centroid_vs_time_v2.png`
- `plots/aw_packet_profiles_plm_nr16384.png`
- `plots/aw_packet_profiles_plm_nr8192.png`
- `plots/aw_packet_profiles_ppmx_nr16384.png`
- `plots/aw_packet_profiles_ppmx_nr8192.png`
- `plots/aw_packet_width_vs_time_v2.png`
- `plots/aw_packet_zminus_ratio_profiles_plm_nr16384.png`
- `plots/aw_packet_zminus_ratio_profiles_plm_nr8192.png`
- `plots/aw_packet_zminus_ratio_profiles_ppmx_nr16384.png`
- `plots/aw_packet_zminus_ratio_profiles_ppmx_nr8192.png`
- `plots/monopole_divb_vs_resolution.png`
- `plots/monopole_final_convergence_all_schemes.png`
- `plots/monopole_final_profiles_all_schemes_nr512.png`
- `plots/monopole_time_all_schemes_nr512.png`
- `plots/parker_boundary_excluded_convergence.png`
- `plots/parker_convergence_clean.png`
- `plots/parker_final_profiles_clean_nr1024.png`
- `plots/parker_massflux_time_evolution.png`

## Diagnostic Issues Fixed

- Scheme parsing now keeps dc, plm, ppm4, ppmx, and wenoz visible where they exist.
- Monopole one-resolution schemes are shown as points rather than silently omitted from convergence fits.
- Parker plots separate uniform and log grids and include boundary-excluded errors.
- AW z-out/z-ref sign convention is checked from t=0 rather than assumed.
- AW z_ref/z_out ratios are masked below a z_out amplitude threshold.
- AW integrated quantities are recomputed from native bin variables and written to `csv/aw_integrals_v2.csv`.

## Recommendations

The Parker and static monopole conclusions are reliable from this scan. The AW packet is useful as a propagation/visual diagnostic, but not yet a clean damping measurement. Move to a lower-boundary forced monochromatic AW test for quantitative damping/reflection: drive a small sinusoidal transverse velocity at the inner radial boundary, impose the corresponding outgoing magnetic perturbation, output radial line diagnostics at high cadence, and Fourier-fit z_out/z_ref amplitude and phase versus travel-time coordinate. That will avoid Gaussian packet compression and window sensitivity.
