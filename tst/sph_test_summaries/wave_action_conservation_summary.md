# wave_action_conservation summary

## Scope

Pilot lower-boundary driven, k_perp=0, phi-polarized Alfven-wave test on a spherical-shell monopole background. The outer radial boundary uses simple zero-gradient transverse perturbations; the analysis window is tau in [8, 22], well inside the domain and before outer-boundary reflection can return.

## Runs

| case | kind | recon | N_r | profile times | ppw_tau | final divB norm | max |v|/cs |
|---|---|---|---:|---:|---:|---:|---:|
| baseline_plm_nr1024 | baseline | plm | 1024 | 5 | 58.57 | 4.795e-13 | 4.062e-03 |

## Driven Fourier Diagnostics

| case | branch | L1 dev z/sqrt(vA) | median z_ref/z_out | max z_ref/z_out | phase RMS | phase slope |
|---|---|---:|---:|---:|---:|---:|

## Plots

- `plots/baseline_residuals.png`
- `plots/driven_aw_time_radius.png` variants for the N_r=4096 driven cases
- `plots/driven_aw_fourier_profiles.png`
- `plots/driven_aw_invariant_profiles.png`
- `plots/driven_aw_convergence_indicators.png`

## Interpretation Notes

- The signed branch is identified by the fitted amplitude near the inner boundary. For the default B_phi = -sqrt(rho) v_phi driver, the expected outward branch is z1.
- The power-law alpha=1/3 grid is only an approximate travel-time grid. The diagnostics use the actual tau(r) computed from B_r and rho.
- The first production scan should keep this simple outflow outer boundary but use an analysis window that ends before any returning reflected wave can enter it.
