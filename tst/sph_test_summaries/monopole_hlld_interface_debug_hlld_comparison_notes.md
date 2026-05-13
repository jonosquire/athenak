# HLLD Comparison Notes

Date: 2026-05-13

## Files Inspected

AthenaK:

- `src/mhd/rsolvers/hlld_mhd.hpp`
- `src/mhd/rsolvers/hlle_mhd.hpp`

Athena++ reference:

- `srcpp/hydro/rsolvers/mhd/hlld.cpp`
- `srcpp/hydro/rsolvers/mhd/hlld_iso.cpp`
- `srcpp/hydro/rsolvers/mhd/lhlld.cpp`
- `srcpp/hydro/rsolvers/mhd/hlle_mhd.cpp`
- `srcpp/pgen/quirk.cpp`

## Implementation Comparison

The AthenaK adiabatic HLLD branch in `src/mhd/rsolvers/hlld_mhd.hpp` is a direct
Kokkos-port style implementation of the Athena++ adiabatic HLLD solver in
`srcpp/hydro/rsolvers/mhd/hlld.cpp`.

The key degenerate-state handling matches:

- `HLLD_SMALL_NUMBER = 1.0e-4`, matching Athena++ `SMALL_NUMBER`.
- The left/right star-state denominators use the same guard:
  `abs(rho * Sd * Sdm - Bn^2) < SMALL * ptst`.
- The `Bn -> 0` double-star fallback uses the same test:
  `0.5 * Bn^2 < SMALL * ptst`.
- The total-pressure star state uses the same average of the left and right
  estimates, `ptst = 0.5 * (ptstl + ptstr)`.
- The transverse double-star states and final flux-region selection follow
  the same Miyoshi & Kusano formulae.
- The radial x1 transverse magnetic flux / EMF sign convention matches the
  Athena++ end-of-solver convention: `E_y = -F(B_y)` and `E_z = F(B_z)`.

The reference tree also contains Athena++ `lhlld.cpp`, a low-dissipation HLLD
variant from Minoshima et al. 2021. `srcpp/pgen/quirk.cpp` explicitly notes that
Roe/HLLC/HLLD-type solvers can amplify odd-even perturbations in the Quirk
problem and that LHLLC/LHLLD suppress that carbuncle-like behavior. AthenaK does
not currently have an LHLLD path in this fork.

## Standalone Degenerate-State Flux Probe

A small standalone Python probe was added:

- `monopole_hlld_interface_debug/scripts/hlld_unit_flux_probe.py`

It writes:

- `monopole_hlld_interface_debug/csv/hlld_unit_flux_states.csv`
- `monopole_hlld_interface_debug/csv/hlld_unit_flux_results.csv`

The probe implements the AthenaK/Athena++ adiabatic HLLD formulae for 1D radial
states and compares them with HLLE for simple monopole-interface states.

Tested states:

1. Exact identical monopole state:
   `rho_L = rho_R`, `p_L = p_R`, `v = 0`, `B_t = 0`, `B_n = 0.25`.
2. Tiny transverse Alfvén-like perturbations with epsilon
   `1e-14`, `1e-12`, and `1e-10`, both on both sides and on the left side only.
3. Tiny pure pressure jumps with epsilon `1e-14`, `1e-12`, and `1e-10`.
4. Tiny pure density jumps with epsilon `1e-14`, `1e-12`, and `1e-10`.

Results:

- Exact identical zero-transverse states give zero transverse momentum flux and
  zero transverse magnetic flux in both HLLD and HLLE.
- Tiny pure pressure or density jumps do not generate transverse fluxes.
- Tiny transverse perturbations produce transverse fluxes proportional to the
  input epsilon. For example, the symmetric transverse case gives
  `F(m_phi) = Bn * Bphi = 2.5e-15`, `2.5e-13`, and `2.5e-11` for epsilons
  `1e-14`, `1e-12`, and `1e-10`.
- HLLD does not show a single-call roundoff degeneracy that creates transverse
  flux from an exactly degenerate monopole state.

## Interpretation

The current evidence does not justify a roundoff-level HLLD transverse guard.
The bad monopole run appears to be a multidimensional growth/amplification of a
small transverse seed in the coupled reconstruction/Riemann/CT/source update,
not a local HLLD formula that creates transverse flux from an exact state.

A guard that zeros HLLD transverse fluxes for roundoff-small states would likely
hide this specific monopole symptom, but it would also risk masking legitimate
small-amplitude Alfvén waves unless the trigger is extremely strict. Since the
unit-state probe is clean, no such guard was implemented in this pass.

The more relevant reference direction is likely LHLLD-style anti-carbuncle
regularization or deeper instrumentation of the reconstructed face states and CT
edge EMFs over the full domain, not a special monopole-only flux clamp.
