//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file sph_shell_mhd.cpp
//! \brief User problem generator for MHD validation on a spherical shell.
//!
//! Modes (selected via <problem>/mode):
//!   "uniform_static"   : rho, p constant; v=0; B=0. Tests MHD code path with no field.
//!   "monopole"         : radial 1/r^2 field with divB=0 by construction.
//!   "toroidal_static"  : axisymmetric B_phi = B0 sin(theta)/r, divB=0 by axisymmetry.
//!   "radial_alfven"    : outgoing axisymmetric Alfven pulse on a 1/r^2 monopole
//!                        background. WKB test on a large radial domain.
//!   "driven_alfven"    : lower-radial-boundary monochromatic k_perp=0 Alfven wave
//!                        on the same monopole background.
//!   "parker_isothermal": analytic isothermal Parker wind plus radial monopole B_r.
//!   "loop_eq"          : magnetic field loop on the equatorial wedge.
//!                        flavor=axisymmetric:    A_phi loop in (r,theta), v=0 (standard)
//!                        flavor=advect:          A_theta loop in (r,phi), v_phi=Omega*r*sin(theta)
//!                                                solid-body rotation (vibe)
//!
//! Finalize prints divB diagnostics (L1, L2, Linf, normalised by |B|/V^(1/3)) plus
//! mode-specific metrics (WKB centroid speed for AW; field-amplitude preservation
//! for the loop).

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <string>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/cell_locations.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/spherical_shell.hpp"
#include "eos/eos.hpp"
#include "mhd/mhd.hpp"
#include "pgen/pgen.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace {
  enum class Mode { kNone, kUniformStatic, kMonopole, kToroidalStatic,
                    kRadialAlfven, kDrivenAlfven, kParkerIsothermal,
                    kLoopEqAxi, kLoopEqAdvect };
  static Mode g_mode = Mode::kNone;
  static Real g_rho0 = 1.0;
  static Real g_p0   = 1.0;
  static Real g_B0   = 0.0;
  static Real g_r_ref = 1.0;
  static Real g_gam  = 5.0/3.0;

  // Radial Alfven wave state
  static Real g_aw_rc      = 0.0;   // pulse centre at t=0
  static Real g_aw_sigma_r = 0.0;
  static Real g_aw_eps     = 0.0;
  static Real g_aw_kr      = 0.0;
  static Real g_aw_vA0     = 0.0;   // background Alfven speed at r_ref
  static Real g_aw_omega   = 0.0;   // carrier angular frequency for WKB phase
  static Real g_aw_sigma_tau = 0.0; // Gaussian width in Alfven travel time
  static bool g_aw_use_tau_profile = false;
  static bool g_aw_amp_sqrt_va = false;

  // Driven monochromatic Alfven wave state
  static Real g_drv_amp = 0.0;
  static Real g_drv_omega = 0.0;
  static Real g_drv_phase = 0.0;
  static Real g_drv_start = 0.0;
  static Real g_drv_ramp = 0.0;
  static Real g_drv_b_sign = -1.0;  // B_phi = sign * sqrt(rho) * v_phi; -1 drives z1.

  // Isothermal Parker wind state
  static Real g_parker_gm = 1.0;
  static Real g_parker_csiso = 1.0;
  static Real g_parker_rc = 0.5;
  static Real g_parker_r_inner = 1.0;
  static Real g_parker_rho_inner = 1.0;
  static Real g_parker_M_inner = 1.0;
  static bool g_parker_outer_analytic = true;

  // Loop-test state
  static Real g_loop_A0    = 0.0;
  static Real g_loop_rc    = 0.0;
  static Real g_loop_sigma = 0.0;
  static Real g_loop_thc   = 0.0;
  static Real g_loop_phc   = 0.0;
  static Real g_loop_Omega = 0.0;

  void UniformStaticFinalize(ParameterInput *pin, Mesh *pm);
  void MonopoleFinalize(ParameterInput *pin, Mesh *pm);
  void WriteMonopoleInterfaceDebug(ParameterInput *pin, Mesh *pm);
  void ToroidalStaticFinalize(ParameterInput *pin, Mesh *pm);
  void RadialAlfvenFinalize(ParameterInput *pin, Mesh *pm);
  void DrivenAlfvenFinalize(ParameterInput *pin, Mesh *pm);
  void ParkerIsothermalFinalize(ParameterInput *pin, Mesh *pm);
  void LoopEqFinalize(ParameterInput *pin, Mesh *pm);
  void MonopoleRadialBCs(Mesh *pm);
  void DrivenAlfvenRadialBCs(Mesh *pm);
  void ParkerMhdRadialBCs(Mesh *pm);

  KOKKOS_INLINE_FUNCTION
  Real ParkerMachAtR(Real r, Real r_c) {
    Real rhs = 4.0*Kokkos::log(r/r_c) + 4.0*r_c/r - 3.0;
    Real Mlo, Mhi;
    if (r < r_c) {
      Mlo = 1.0e-8;
      Mhi = 1.0 - 1.0e-10;
    } else {
      Mlo = 1.0 + 1.0e-10;
      Mhi = 50.0;
    }
    auto f = [rhs] (Real M) {
      return M*M - Kokkos::log(M*M) - rhs;
    };
    Real flo = f(Mlo);
    Real fhi = f(Mhi);
    if (flo * fhi > 0.0) {
      return 0.5*(Mlo + Mhi);
    }
    for (int it = 0; it < 60; ++it) {
      Real Mmid = 0.5*(Mlo + Mhi);
      Real fmid = f(Mmid);
      if (flo * fmid <= 0.0) {
        Mhi = Mmid;
        fhi = fmid;
      } else {
        Mlo = Mmid;
        flo = fmid;
      }
    }
    return 0.5*(Mlo + Mhi);
  }

  KOKKOS_INLINE_FUNCTION
  Real ParkerRhoAtR(Real r, Real r_c,
                    Real r_ref, Real rho_ref, Real M_ref) {
    Real M = ParkerMachAtR(r, r_c);
    return rho_ref * (M_ref / M) * (r_ref / r) * (r_ref / r);
  }

  //! Volume-averaged divB on active cells (face-area-weighted FV form, matching
  //! the SphericalShellGeom factors used in CT). Returns L1, L2, Linf, mean |B|, and
  //! a normalised Linf metric Linf * h_typ / |B|_max where h_typ is a typical
  //! geometric scale.
  struct DivBStats {
    Real l1     = 0.0;
    Real l2     = 0.0;
    Real linf   = 0.0;
    Real bmax   = 0.0;
    Real h_typ  = 0.0;
    int  Ncell  = 0;
  };
  DivBStats ComputeDivBStats(MeshBlockPack *pmbp);
  void PrintDivBStats(const std::string &label, const DivBStats &s);
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem
//! \brief IC dispatch for spherical-shell MHD test modes.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  if (restart) return;
  if (pmy_mesh_->pmb_pack->pmhd == nullptr) {
    std::cout << "### FATAL ERROR: sph_shell_mhd pgen requires <mhd> block"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pmy_mesh_->pmb_pack->pcoord->coord_system !=
      CoordinateSystem::spherical_shell) {
    std::cout << "### FATAL ERROR: sph_shell_mhd pgen requires"
              << " <coord>/system = spherical_shell" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  const std::string mode_str = pin->GetOrAddString("problem", "mode", "monopole");
  const Real rho0 = pin->GetOrAddReal("problem", "rho0", 1.0);
  const Real p0   = pin->GetOrAddReal("problem", "p0", 1.0);
  const Real B0   = pin->GetOrAddReal("problem", "B0", 1.0);
  const Real r_ref = pin->GetOrAddReal("problem", "r_ref",
                                       pmy_mesh_->mesh_size.x1min);
  g_rho0 = rho0; g_p0 = p0; g_B0 = B0; g_r_ref = r_ref;

  auto *pmbp = pmy_mesh_->pmb_pack;
  auto &indcs = pmy_mesh_->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pmbp->nmb_thispack - 1;
  auto &u0 = pmbp->pmhd->u0;
  auto &b0 = pmbp->pmhd->b0;
  auto &eos = pmbp->pmhd->peos->eos_data;
  auto geom = pmbp->pcoord->shell_geom;
  const Real gam = eos.gamma;
  g_gam = gam;

  // ------------------------------------------------------------------
  // Always: zero all face B components first; modes that set them will
  // overwrite. We need an over-the-extra-face write at block edges.
  par_for("sph_mhd_zero_bx1f", DevExeSpace(),
          0, nmb1, ks, ke, js, je, is, ie+1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    b0.x1f(m, k, j, i) = 0.0;
  });
  par_for("sph_mhd_zero_bx2f", DevExeSpace(),
          0, nmb1, ks, ke, js, je+1, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    b0.x2f(m, k, j, i) = 0.0;
  });
  par_for("sph_mhd_zero_bx3f", DevExeSpace(),
          0, nmb1, ks, ke+1, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    b0.x3f(m, k, j, i) = 0.0;
  });

  // ------------------------------------------------------------------
  if (mode_str == "uniform_static") {
    g_mode = Mode::kUniformStatic;
    pgen_final_func = UniformStaticFinalize;
    par_for("sph_mhd_uniform_ic", DevExeSpace(),
            0, nmb1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      u0(m, IDN, k, j, i) = rho0;
      u0(m, IM1, k, j, i) = 0.0;
      u0(m, IM2, k, j, i) = 0.0;
      u0(m, IM3, k, j, i) = 0.0;
      // Energy: gas pressure / (gamma - 1). No magnetic contribution since B=0.
      u0(m, IEN, k, j, i) = p0 / (gam - 1.0);
    });

  } else if (mode_str == "monopole") {
    g_mode = Mode::kMonopole;
    pgen_final_func = MonopoleFinalize;
    user_bcs_func = MonopoleRadialBCs;
    // Set face-centred radial field so that A1 * B1 is identical across radial
    // faces in (m, j, k). Choose B1(face_i) = B0 * (r_ref / r_face(i))^2.
    // Then on the radial face A1 = r_face^2 * dcos * dphi, so
    //   A1 * B1 = B0 * r_ref^2 * dcos(j) * dphi(k)
    // which is independent of i (the radial face index). Discrete divB then has
    // exact cancellation in the radial direction.
    const Real Bref = B0;
    const Real rref2 = r_ref * r_ref;
    par_for("sph_mhd_monopole_bx1f", DevExeSpace(),
            0, nmb1, ks, ke, js, je, is, ie+1,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      Real rf = geom.r_face(m, i);
      b0.x1f(m, k, j, i) = Bref * rref2 / (rf * rf);
    });
    // b0.x2f and b0.x3f already zeroed.
    // Energy: include magnetic contribution using a face-averaged B^2.
    par_for("sph_mhd_monopole_uc", DevExeSpace(),
            0, nmb1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      // Cell-centred B_r ~ 0.5 * (B1f(i) + B1f(i+1))  for energy IC; the
      // ConsToPrim step at start of run will redo this consistently from b0.
      Real br_cc = 0.5 * (b0.x1f(m,k,j,i) + b0.x1f(m,k,j,i+1));
      Real eb = 0.5 * (br_cc * br_cc);
      u0(m, IDN, k, j, i) = rho0;
      u0(m, IM1, k, j, i) = 0.0;
      u0(m, IM2, k, j, i) = 0.0;
      u0(m, IM3, k, j, i) = 0.0;
      u0(m, IEN, k, j, i) = p0 / (gam - 1.0) + eb;
    });

  } else if (mode_str == "toroidal_static") {
    g_mode = Mode::kToroidalStatic;
    pgen_final_func = ToroidalStaticFinalize;
    // B_phi(r, theta) = B0 * sin(theta) / r. Set on phi faces. Divergence-free
    // for any axisymmetric B_phi(r, theta). Sample at cell-centred (r, theta).
    par_for("sph_mhd_toroidal_bx3f", DevExeSpace(),
            0, nmb1, ks, ke+1, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      Real r  = geom.r_vol(m, i);
      Real th = geom.theta_vol(m, j);
      b0.x3f(m, k, j, i) = B0 * Kokkos::sin(th) / r;
    });
    par_for("sph_mhd_toroidal_uc", DevExeSpace(),
            0, nmb1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      Real r  = geom.r_vol(m, i);
      Real th = geom.theta_vol(m, j);
      Real bp = B0 * Kokkos::sin(th) / r;
      Real eb = 0.5 * bp * bp;
      u0(m, IDN, k, j, i) = rho0;
      u0(m, IM1, k, j, i) = 0.0;
      u0(m, IM2, k, j, i) = 0.0;
      u0(m, IM3, k, j, i) = 0.0;
      u0(m, IEN, k, j, i) = p0 / (gam - 1.0) + eb;
    });

  } else if (mode_str == "radial_alfven") {
    // Outgoing axisymmetric Alfven pulse on a 1/r^2 radial monopole background.
    //
    // Background:  rho=rho0, p=p0, v=0, B_r(r) = B0 (r_ref/r)^2 on radial faces.
    // Perturbation (axisymmetric, divergence-free since dB_phi/dphi = 0):
    //   delta B_phi(r) = eps * B0 * env(r) * carrier(r)   on phi-faces
    //   delta v_phi    = -delta B_phi / sqrt(rho)         on cell centres  (outgoing wave)
    // Defaults:
    //   env(r)     = exp(-(r - r_c)^2 / (2 sigma_r^2))
    //   carrier(r) = cos(k_r * (r - r_c))    (=1 if k_r = 0, pure pulse)
    // Optional WKB packet:
    //   phase_model=travel_time uses tau(r)=int_{r_c}^r dr/v_A
    //   env(tau)=exp(-tau^2/(2 sigma_tau^2)), carrier=cos(omega*tau)
    //   amplitude_model=sqrt_va multiplies by sqrt(v_A(r)/v_A(r_c))=r_c/r.
    //
    // Alfven speed: v_A(r) = B(r) / sqrt(rho) = (B0/sqrt(rho)) * (r_ref/r)^2.
    // WKB ray: r(t) = (r_c^3 + 3 * r_ref^2 * v_A0 * t)^(1/3),  v_A0 = B0/sqrt(rho0).
    // Energy-flux conservation with v_A ~ 1/r^2, rho=const, area ~ r^2 gives
    // |dv_phi| and |dB_phi| constant along the ray (no amplitude growth/decay).
    // The pulse compresses radially as v_A drops outward (trailing edge catches up).
    g_mode = Mode::kRadialAlfven;
    pgen_final_func = RadialAlfvenFinalize;
    user_bcs_func = MonopoleRadialBCs;
    const Real r_c    = pin->GetOrAddReal("problem", "r_c", 2.0);
    const Real sigma_r= pin->GetOrAddReal("problem", "sigma_r", 0.3);
    const Real eps    = pin->GetOrAddReal("problem", "eps", 1.0e-3);
    const Real k_r    = pin->GetOrAddReal("problem", "k_r", 0.0);
    const std::string phase_model =
        pin->GetOrAddString("problem", "phase_model", "radial");
    const std::string amplitude_model =
        pin->GetOrAddString("problem", "amplitude_model", "constant");
    const Real Bref = B0;
    const Real rref2 = r_ref * r_ref;
    const Real vA_ref = Bref / std::sqrt(rho0);
    const Real vA_c = vA_ref * rref2 / (r_c * r_c);
    const bool use_tau_profile =
        (phase_model == "travel_time" || phase_model == "wkb" || phase_model == "wkb_tau");
    const bool amp_sqrt_va =
        (amplitude_model == "sqrt_va" || amplitude_model == "sqrt_B");
    if (phase_model != "radial" && !use_tau_profile) {
      std::cout << "### FATAL ERROR: radial_alfven phase_model must be radial, "
                << "travel_time, wkb, or wkb_tau" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (amplitude_model != "constant" && !amp_sqrt_va) {
      std::cout << "### FATAL ERROR: radial_alfven amplitude_model must be constant, "
                << "sqrt_va, or sqrt_B" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    const Real omega = pin->GetOrAddReal("problem", "omega", k_r * vA_c);
    const Real sigma_tau =
        pin->GetOrAddReal("problem", "sigma_tau", sigma_r / (vA_c + 1.0e-30));
    g_aw_rc      = r_c;
    g_aw_sigma_r = sigma_r;
    g_aw_eps     = eps;
    g_aw_kr      = k_r;
    g_aw_vA0     = vA_ref;  // v_A at r = r_ref
    g_aw_omega   = omega;
    g_aw_sigma_tau = sigma_tau;
    g_aw_use_tau_profile = use_tau_profile;
    g_aw_amp_sqrt_va = amp_sqrt_va;

    // (1) Background monopole on radial faces (A1 * B1 invariant in i).
    par_for("sph_mhd_aw_bx1f", DevExeSpace(),
            0, nmb1, ks, ke, js, je, is, ie+1,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      Real rf = geom.r_face(m, i);
      b0.x1f(m, k, j, i) = Bref * rref2 / (rf * rf);
    });

    // (2) Axisymmetric perturbation on phi-faces. Sample at cell-centred (r, theta).
    // For axisymmetric (no phi dependence) this exactly satisfies dB_phi/dphi = 0
    // so divB is preserved by construction.
    par_for("sph_mhd_aw_bx3f", DevExeSpace(),
            0, nmb1, ks, ke+1, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      Real r = geom.r_vol(m, i);
      Real tau = (r*r*r - r_c*r_c*r_c) / (3.0 * vA_ref * rref2);
      Real arg = use_tau_profile ? (tau / sigma_tau) : ((r - r_c) / sigma_r);
      Real env = Kokkos::exp(-0.5 * arg * arg);
      Real phase = use_tau_profile ? (omega * tau) : (k_r * (r - r_c));
      Real carrier = (phase != 0.0) ? Kokkos::cos(phase) : 1.0;
      Real amp = amp_sqrt_va ? (r_c / r) : 1.0;
      b0.x3f(m, k, j, i) = eps * Bref * amp * env * carrier;
    });

    // (3) Cell-centred state. delta v_phi from Alfven relation for outgoing wave on
    // B_r > 0:  delta v_phi = -delta B_phi / sqrt(rho).
	    par_for("sph_mhd_aw_uc", DevExeSpace(),
	            0, nmb1, ks, ke, js, je, is, ie,
	    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      Real r = geom.r_vol(m, i);
      Real tau = (r*r*r - r_c*r_c*r_c) / (3.0 * vA_ref * rref2);
      Real arg = use_tau_profile ? (tau / sigma_tau) : ((r - r_c) / sigma_r);
      Real env = Kokkos::exp(-0.5 * arg * arg);
      Real phase = use_tau_profile ? (omega * tau) : (k_r * (r - r_c));
      Real carrier = (phase != 0.0) ? Kokkos::cos(phase) : 1.0;
      Real amp = amp_sqrt_va ? (r_c / r) : 1.0;
      Real dbphi_cc = eps * Bref * amp * env * carrier;
      Real dvphi    = -dbphi_cc / Kokkos::sqrt(rho0);
      Real br_cc    = 0.5 * (b0.x1f(m, k, j, i) + b0.x1f(m, k, j, i+1));
      Real eb = 0.5 * (br_cc * br_cc + dbphi_cc * dbphi_cc);
      u0(m, IDN, k, j, i) = rho0;
      u0(m, IM1, k, j, i) = 0.0;
      u0(m, IM2, k, j, i) = 0.0;
      u0(m, IM3, k, j, i) = rho0 * dvphi;
	      u0(m, IEN, k, j, i) = p0 / (gam - 1.0) + 0.5 * rho0 * dvphi * dvphi + eb;
	    });

  } else if (mode_str == "driven_alfven") {
    // Static monopole background with a sinusoidal lower-radial-boundary driver.
    // The active domain starts from the unperturbed state; the user radial BC
    // injects a k_perp=0 phi-polarized Alfven wave through the inner ghost zones.
    //
    // For B_r > 0, B_phi = -sqrt(rho) v_phi makes
    //   z1 = v_phi - B_phi/sqrt(rho) = 2 v_phi, z2 = 0,
    // matching the outward branch used by the radial_alfven packet diagnostics.
    // The opposite sign can be selected with outgoing_branch=z2 for sign tests.
    g_mode = Mode::kDrivenAlfven;
    pgen_final_func = DrivenAlfvenFinalize;
    user_bcs_func = DrivenAlfvenRadialBCs;

    const Real r_inner = pmy_mesh_->mesh_size.x1min;
    const Real vA_inner = (B0 * r_ref * r_ref / (r_inner * r_inner)) / std::sqrt(rho0);
    g_drv_amp = pin->GetOrAddReal("problem", "driver_amp", 1.0e-4 * vA_inner);
    g_drv_omega = pin->GetOrAddReal("problem", "driver_omega", 6.0);
    g_drv_phase = pin->GetOrAddReal("problem", "driver_phase", 0.0);
    g_drv_start = pin->GetOrAddReal("problem", "driver_start_time", 0.0);
    g_drv_ramp = pin->GetOrAddReal("problem", "driver_ramp_time", 0.0);
    const std::string pol = pin->GetOrAddString("problem", "driver_polarization", "phi");
    const std::string branch = pin->GetOrAddString("problem", "outgoing_branch", "z1");
    if (pol != "phi") {
      std::cout << "### FATAL ERROR: driven_alfven currently supports only "
                << "driver_polarization=phi. Phi polarization is axisymmetric "
                << "and keeps divB clean for k_perp=0." << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (branch == "z1") {
      g_drv_b_sign = -1.0;
    } else if (branch == "z2") {
      g_drv_b_sign = 1.0;
    } else {
      std::cout << "### FATAL ERROR: driven_alfven outgoing_branch must be z1 or z2"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }

    const Real Bref = B0;
    const Real rref2 = r_ref * r_ref;
    par_for("sph_mhd_driven_aw_bx1f", DevExeSpace(),
            0, nmb1, ks, ke, js, je, is, ie+1,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      Real rf = geom.r_face(m, i);
      b0.x1f(m, k, j, i) = Bref * rref2 / (rf * rf);
    });
    par_for("sph_mhd_driven_aw_uc", DevExeSpace(),
            0, nmb1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      Real br_cc = 0.5 * (b0.x1f(m, k, j, i) + b0.x1f(m, k, j, i+1));
      u0(m, IDN, k, j, i) = rho0;
      u0(m, IM1, k, j, i) = 0.0;
      u0(m, IM2, k, j, i) = 0.0;
      u0(m, IM3, k, j, i) = 0.0;
      u0(m, IEN, k, j, i) = p0 / (gam - 1.0) + 0.5 * br_cc * br_cc;
    });
  } else if (mode_str == "parker_isothermal") {
	    // Analytic isothermal Parker wind plus a radial monopole B field. This is
	    // intentionally restricted to isothermal MHD; evolving the same profile with
	    // ideal-gas energy is a relaxation test, not a Parker preservation test.
	    if (eos.is_ideal) {
	      std::cout << "### FATAL ERROR: sph_shell_mhd parker_isothermal requires "
	                << "<mhd>/eos = isothermal and <mhd>/iso_sound_speed matching "
	                << "<problem>/cs_iso" << std::endl;
	      std::exit(EXIT_FAILURE);
	    }
	    g_mode = Mode::kParkerIsothermal;
	    pgen_final_func = ParkerIsothermalFinalize;
	    user_bcs_func = ParkerMhdRadialBCs;

	    const Real gm = pin->GetReal("problem", "gm");
	    const Real cs = pin->GetReal("problem", "cs_iso");
	    const Real rho_inner = pin->GetOrAddReal("problem", "rho_inner", rho0);
	    const std::string outer_bc = pin->GetOrAddString("problem", "outer_bc", "analytic");
	    if (outer_bc == "analytic") {
	      g_parker_outer_analytic = true;
	    } else if (outer_bc == "outflow") {
	      g_parker_outer_analytic = false;
	    } else {
	      std::cout << "### FATAL ERROR: parker_isothermal <problem>/outer_bc must be "
	                << "'analytic' or 'outflow', got '" << outer_bc << "'"
	                << std::endl;
	      std::exit(EXIT_FAILURE);
	    }
	    const Real rel_cs = std::abs(eos.iso_cs - cs) / std::max(std::abs(cs), 1.0e-30);
	    if (rel_cs > 1.0e-12) {
	      std::cout << "### FATAL ERROR: parker_isothermal requires "
	                << "<mhd>/iso_sound_speed to match <problem>/cs_iso. "
	                << "Got iso_sound_speed=" << eos.iso_cs
	                << " and cs_iso=" << cs << std::endl;
	      std::exit(EXIT_FAILURE);
	    }

	    g_parker_gm = gm;
	    g_parker_csiso = cs;
	    g_parker_rho_inner = rho_inner;
	    g_parker_r_inner = pmy_mesh_->mesh_size.x1min;
	    g_parker_rc = gm / (2.0 * cs * cs);
	    g_parker_M_inner = ParkerMachAtR(g_parker_r_inner, g_parker_rc);
	    g_p0 = cs * cs * rho_inner;

	    const Real rc = g_parker_rc;
	    const Real r_inner = g_parker_r_inner;
	    const Real M_inner = g_parker_M_inner;
	    const Real Bref = B0;
	    const Real rref2 = r_ref * r_ref;

	    par_for("sph_mhd_parker_bx1f", DevExeSpace(),
	            0, nmb1, ks, ke, js, je, is, ie+1,
	    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
	      Real rf = geom.r_face(m, i);
	      b0.x1f(m, k, j, i) = Bref * rref2 / (rf * rf);
	    });

	    par_for("sph_mhd_parker_uc", DevExeSpace(),
	            0, nmb1, ks, ke, js, je, is, ie,
	    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
	      Real r = geom.r_vol(m, i);
	      Real M = ParkerMachAtR(r, rc);
	      Real rho = ParkerRhoAtR(r, rc, r_inner, rho_inner, M_inner);
	      Real vr = M * cs;
	      u0(m, IDN, k, j, i) = rho;
	      u0(m, IM1, k, j, i) = rho * vr;
	      u0(m, IM2, k, j, i) = 0.0;
	      u0(m, IM3, k, j, i) = 0.0;
	    });

	  } else if (mode_str == "loop_eq") {
    // Magnetic field loop on an equatorial wedge.
    //
    //   flavor=axisymmetric:
    //     A_phi(r, theta) Gaussian-loop in (r, theta) at theta_c = pi/2.
    //     Discrete Stokes-loop curl populates B1f, B2f. B3f = 0.
    //     v = 0. Tests CT preservation of a closed poloidal loop.
    //
    //   flavor=advect:
    //     A_theta(r, phi) Gaussian-loop in (r, phi) at phi_c.
    //     Discrete Stokes-loop curl populates B1f, B3f. B2f = 0.
    //     v_phi = Omega * r * sin(theta) -- solid-body rotation around z.
    //     Loop rotates rigidly. Tests CT preservation under advection.
    //
    // Discrete curl from analytic vector potential is computed at edges and then
    // taken around each face's closed boundary, so divB = 0 exactly by Stokes.
    const std::string flav = pin->GetOrAddString("problem", "flavor", "axisymmetric");
    pgen_final_func = LoopEqFinalize;
    const Real A0     = pin->GetOrAddReal("problem", "A0", 1.0e-3);
    const Real rc     = pin->GetOrAddReal("problem", "rc",
                                          0.5*(pmy_mesh_->mesh_size.x1min
                                              +pmy_mesh_->mesh_size.x1max));
    const Real sigma  = pin->GetOrAddReal("problem", "sigma", 0.15);
    const Real thc    = pin->GetOrAddReal("problem", "theta_c", M_PI*0.5);
    const Real phc    = pin->GetOrAddReal("problem", "phi_c",
                                          0.5*(pmy_mesh_->mesh_size.x3min
                                              +pmy_mesh_->mesh_size.x3max));
    g_loop_A0    = A0;
    g_loop_rc    = rc;
    g_loop_sigma = sigma;
    g_loop_thc   = thc;
    g_loop_phc   = phc;

    if (flav == "axisymmetric") {
      g_mode = Mode::kLoopEqAxi;
      // A_phi(r, theta) Gaussian centred at (rc, thc). Sampled at phi-edge centres
      // (r_face, theta_face). Discrete Stokes loop gives:
      //   B1f(i_face, j_cell, k_cell)
      //     = [ sin(theta_face_{j+1}) A_phi(j+1, i) - sin(theta_face_j) A_phi(j, i) ]
      //       / [ r_face_i * (cos theta_face_j - cos theta_face_{j+1}) ]
      //   B2f(i_cell, j_face, k_cell)
      //     = -[ r_face_{i+1} A_phi(j, i+1) - r_face_i A_phi(j, i) ] / dr2_half(i)
      par_for("sph_mhd_loop_axi_b1", DevExeSpace(),
              0, nmb1, ks, ke, js, je, is, ie+1,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        Real rf  = geom.r_face(m, i);
        Real tlo = geom.theta_face(m, j);
        Real thi = geom.theta_face(m, j+1);
        Real Aplo = A0 * Kokkos::exp(-0.5 * ((rf - rc)*(rf - rc)
                                            + rc*rc*(tlo - thc)*(tlo - thc))
                                          / (sigma * sigma));
        Real Aphi = A0 * Kokkos::exp(-0.5 * ((rf - rc)*(rf - rc)
                                            + rc*rc*(thi - thc)*(thi - thc))
                                          / (sigma * sigma));
        Real dcos = geom.dcos_theta(m, j);
        b0.x1f(m, k, j, i) = (Kokkos::sin(thi) * Aphi - Kokkos::sin(tlo) * Aplo)
                             / (rf * dcos);
      });
      par_for("sph_mhd_loop_axi_b2", DevExeSpace(),
              0, nmb1, ks, ke, js, je+1, is, ie,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        Real rlo = geom.r_face(m, i);
        Real rhi = geom.r_face(m, i+1);
        Real tj  = geom.theta_face(m, j);
        Real Aplo = A0 * Kokkos::exp(-0.5 * ((rlo - rc)*(rlo - rc)
                                            + rc*rc*(tj - thc)*(tj - thc))
                                          / (sigma * sigma));
        Real Aphi = A0 * Kokkos::exp(-0.5 * ((rhi - rc)*(rhi - rc)
                                            + rc*rc*(tj - thc)*(tj - thc))
                                          / (sigma * sigma));
        Real dr2h = geom.dr2_half(m, i);
        b0.x2f(m, k, j, i) = -(rhi * Aphi - rlo * Aplo) / dr2h;
      });
      // b0.x3f already zero.
      par_for("sph_mhd_loop_axi_uc", DevExeSpace(),
              0, nmb1, ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        Real br_cc = 0.5 * (b0.x1f(m, k, j, i) + b0.x1f(m, k, j, i+1));
        Real bt_cc = 0.5 * (b0.x2f(m, k, j, i) + b0.x2f(m, k, j+1, i));
        Real eb = 0.5 * (br_cc * br_cc + bt_cc * bt_cc);
        u0(m, IDN, k, j, i) = rho0;
        u0(m, IM1, k, j, i) = 0.0;
        u0(m, IM2, k, j, i) = 0.0;
        u0(m, IM3, k, j, i) = 0.0;
        u0(m, IEN, k, j, i) = p0 / (gam - 1.0) + eb;
      });
    } else if (flav == "advect") {
      g_mode = Mode::kLoopEqAdvect;
      const Real Om = pin->GetReal("problem", "Omega");
      g_loop_Omega = Om;
      // A_theta(r, phi) Gaussian centred at (rc, phc). Sampled at theta-edge centres
      // (r_face, phi_face). Discrete Stokes loop gives:
      //   B1f(i_face, j_cell, k_cell)
      //     = L2(j_cell, i_face) * [A_theta(k, i) - A_theta(k+1, i)] / A1
      //   B3f(i_cell, j_cell, k_face)
      //     = [r_face_{i+1} A_theta(k_face, i+1) - r_face_i A_theta(k_face, i)]
      //       / dr2_half(i)
      par_for("sph_mhd_loop_advect_b1", DevExeSpace(),
              0, nmb1, ks, ke, js, je, is, ie+1,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        Real rf  = geom.r_face(m, i);
        Real plo = geom.phi_face(m, k);
        Real phi_hi = geom.phi_face(m, k+1);
        Real At_lo = A0 * Kokkos::exp(-0.5 * ((rf - rc)*(rf - rc)
                                              + rc*rc*(plo - phc)*(plo - phc))
                                            / (sigma * sigma));
        Real At_hi = A0 * Kokkos::exp(-0.5 * ((rf - rc)*(rf - rc)
                                              + rc*rc*(phi_hi - phc)*(phi_hi - phc))
                                            / (sigma * sigma));
        // L2 = r_face * dtheta(j), A1 = r_face^2 * dcos * dphi
        Real A1 = rf * rf * geom.dcos_theta(m, j) * geom.dphi(m, k);
        Real L2 = rf * geom.dtheta(m, j);
        b0.x1f(m, k, j, i) = L2 * (At_lo - At_hi) / A1;
      });
      par_for("sph_mhd_loop_advect_b3", DevExeSpace(),
              0, nmb1, ks, ke+1, js, je, is, ie,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        Real rlo = geom.r_face(m, i);
        Real rhi = geom.r_face(m, i+1);
        Real pk  = geom.phi_face(m, k);
        Real At_lo = A0 * Kokkos::exp(-0.5 * ((rlo - rc)*(rlo - rc)
                                              + rc*rc*(pk - phc)*(pk - phc))
                                            / (sigma * sigma));
        Real At_hi = A0 * Kokkos::exp(-0.5 * ((rhi - rc)*(rhi - rc)
                                              + rc*rc*(pk - phc)*(pk - phc))
                                            / (sigma * sigma));
        Real dr2h = geom.dr2_half(m, i);
        b0.x3f(m, k, j, i) = (rhi * At_hi - rlo * At_lo) / dr2h;
      });
      // b0.x2f already zero.
      par_for("sph_mhd_loop_advect_uc", DevExeSpace(),
              0, nmb1, ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        Real r  = geom.r_vol(m, i);
        Real th = geom.theta_vol(m, j);
        Real vphi = Om * r * Kokkos::sin(th);
        Real br_cc = 0.5 * (b0.x1f(m, k, j, i) + b0.x1f(m, k, j, i+1));
        Real bp_cc = 0.5 * (b0.x3f(m, k, j, i) + b0.x3f(m, k+1, j, i));
        Real eb = 0.5 * (br_cc * br_cc + bp_cc * bp_cc);
        u0(m, IDN, k, j, i) = rho0;
        u0(m, IM1, k, j, i) = 0.0;
        u0(m, IM2, k, j, i) = 0.0;
        u0(m, IM3, k, j, i) = rho0 * vphi;
        u0(m, IEN, k, j, i) = p0 / (gam - 1.0) + 0.5 * rho0 * vphi * vphi + eb;
      });
    } else {
      std::cout << "### FATAL ERROR: sph_shell_mhd loop_eq flavor='" << flav
                << "' unknown. Use 'axisymmetric' or 'advect'." << std::endl;
      std::exit(EXIT_FAILURE);
    }

	  } else {
	    std::cout << "### FATAL ERROR: sph_shell_mhd mode='" << mode_str
	              << "' not recognised. Use: uniform_static, monopole, toroidal_static,"
	              << " radial_alfven, driven_alfven, parker_isothermal, loop_eq."
                << std::endl;
	    std::exit(EXIT_FAILURE);
	  }
}

//----------------------------------------------------------------------------------------
// Finalizers + divB diagnostic
//----------------------------------------------------------------------------------------
namespace {

DivBStats ComputeDivBStats(MeshBlockPack *pmbp) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pmbp->nmb_thispack - 1;
  auto geom = pmbp->pcoord->shell_geom;
  auto &b0 = pmbp->pmhd->b0;
  const int N = (nmb1+1)*(ke-ks+1)*(je-js+1)*(ie-is+1);

  Real local_l1 = 0.0, local_l2 = 0.0, local_linf = 0.0, local_bmax = 0.0;
  Kokkos::parallel_reduce("divB_diag",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, N),
  KOKKOS_LAMBDA(const int idx,
                Real &s1, Real &s2, Real &smax, Real &smaxb) {
    int nx1 = ie - is + 1, nx2 = je - js + 1, nx3 = ke - ks + 1;
    int nji = nx2 * nx1, nkji = nx3 * nx2 * nx1;
    int m = idx / nkji;
    int k = (idx - m*nkji) / nji;
    int j = (idx - m*nkji - k*nji) / nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks; j += js;
    // Face areas at the six faces of this cell. Same factors as CT.
    Real A1m = SphFace1Area(geom, m, k, j, i  );
    Real A1p = SphFace1Area(geom, m, k, j, i+1);
    Real A2m = SphFace2Area(geom, m, k, j,   i);
    Real A2p = SphFace2Area(geom, m, k, j+1, i);
    Real A3m = SphFace3Area(geom, m, k,   j, i);
    Real A3p = SphFace3Area(geom, m, k+1, j, i);
    Real V   = SphCellVolume(geom, m, k, j, i);
    Real B1m = b0.x1f(m, k, j, i  );
    Real B1p = b0.x1f(m, k, j, i+1);
    Real B2m = b0.x2f(m, k, j,   i);
    Real B2p = b0.x2f(m, k, j+1, i);
    Real B3m = b0.x3f(m, k,   j, i);
    Real B3p = b0.x3f(m, k+1, j, i);
    Real divB = (A1p*B1p - A1m*B1m
               + A2p*B2p - A2m*B2m
               + A3p*B3p - A3m*B3m) / V;
    Real a = Kokkos::fabs(divB);
    // Cell-centred |B| approximation using face-average.
    Real Br = 0.5*(B1m + B1p);
    Real Bt = 0.5*(B2m + B2p);
    Real Bp = 0.5*(B3m + B3p);
    Real bmag = Kokkos::sqrt(Br*Br + Bt*Bt + Bp*Bp);
    s1   += a;
    s2   += a*a;
    if (a    > smax)  smax = a;
    if (bmag > smaxb) smaxb = bmag;
  }, local_l1, local_l2,
     Kokkos::Max<Real>(local_linf), Kokkos::Max<Real>(local_bmax));

  Real total_l1 = local_l1, total_l2 = local_l2;
  Real total_linf = local_linf, total_bmax = local_bmax;
  int  total_N = N;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(&local_l1, &total_l1, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&local_l2, &total_l2, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&local_linf, &total_linf, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&local_bmax, &total_bmax, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
  int N_local = N;
  MPI_Allreduce(&N_local, &total_N, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif
  // Typical geometric scale: take the radial extent / nx1 as a reasonable proxy.
  Real h_typ = (pmbp->pmesh->mesh_size.x1max - pmbp->pmesh->mesh_size.x1min) /
               static_cast<Real>(pmbp->pmesh->mesh_indcs.nx1);

  DivBStats s;
  s.l1     = total_l1 / static_cast<Real>(total_N);
  s.l2     = std::sqrt(total_l2 / static_cast<Real>(total_N));
  s.linf   = total_linf;
  s.bmax   = total_bmax;
  s.h_typ  = h_typ;
  s.Ncell  = total_N;
  return s;
}

void PrintDivBStats(const std::string &label, const DivBStats &s) {
  if (global_variable::my_rank != 0) return;
  Real norm = (s.bmax > 0.0) ? (s.linf * s.h_typ / s.bmax) : 0.0;
  std::cout.precision(6);
  std::cout << std::scientific
            << "[" << label << "] divB diagnostic:\n"
            << "    L1   = " << s.l1   << "\n"
            << "    L2   = " << s.l2   << "\n"
            << "    Linf = " << s.linf << "\n"
            << "    |B|max = " << s.bmax
            << "    h_typ  = " << s.h_typ << "\n"
            << "    Linf*h/|B|max = " << norm
            << std::endl;
}

//----------------------------------------------------------------------------------------
// uniform_static: should preserve exactly (B=0, v=0).

void UniformStaticFinalize(ParameterInput *pin, Mesh *pm) {
  if (g_mode != Mode::kUniformStatic) return;
  auto *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pmbp->nmb_thispack - 1;
  auto u0 = pmbp->pmhd->u0;
  Real lmax_v = 0.0, lmax_drho = 0.0, lmax_dp = 0.0;
  const Real rho0 = g_rho0, p0 = g_p0, gam = g_gam;
  Kokkos::parallel_reduce("uniform_static_diag",
    Kokkos::RangePolicy<>(DevExeSpace(),
                          0, (nmb1+1)*(ke-ks+1)*(je-js+1)*(ie-is+1)),
  KOKKOS_LAMBDA(const int idx, Real &mv, Real &mdrho, Real &mdp) {
    int nx1 = ie - is + 1, nx2 = je - js + 1, nx3 = ke - ks + 1;
    int nji = nx2 * nx1, nkji = nx3 * nx2 * nx1;
    int m = idx / nkji;
    int k = (idx - m*nkji) / nji;
    int j = (idx - m*nkji - k*nji) / nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks; j += js;
    Real rho = u0(m, IDN, k, j, i);
    Real ek = 0.5 * (u0(m,IM1,k,j,i)*u0(m,IM1,k,j,i)
                    +u0(m,IM2,k,j,i)*u0(m,IM2,k,j,i)
                    +u0(m,IM3,k,j,i)*u0(m,IM3,k,j,i)) / rho;
    Real p = (gam - 1.0) * (u0(m,IEN,k,j,i) - ek);
    Real vmag = Kokkos::sqrt(2.0*ek/rho);
    Real adrho = Kokkos::fabs(rho - rho0);
    Real adp   = Kokkos::fabs(p - p0);
    if (vmag  > mv)   mv   = vmag;
    if (adrho > mdrho) mdrho = adrho;
    if (adp   > mdp)   mdp   = adp;
  }, Kokkos::Max<Real>(lmax_v), Kokkos::Max<Real>(lmax_drho),
     Kokkos::Max<Real>(lmax_dp));
  DivBStats s = ComputeDivBStats(pmbp);
  if (global_variable::my_rank == 0) {
    std::cout.precision(6);
    std::cout << std::scientific
              << "[sph_shell_mhd/uniform_static] max|drho|=" << lmax_drho
              << " max|dp|=" << lmax_dp << " max|v|=" << lmax_v << std::endl;
  }
  PrintDivBStats("sph_shell_mhd/uniform_static", s);
}

//----------------------------------------------------------------------------------------
// monopole: report divB stats + max deviation of B1f from the analytic 1/r^2 profile
// + max |v|/cs.

void MonopoleFinalize(ParameterInput *pin, Mesh *pm) {
  if (g_mode != Mode::kMonopole) return;
  auto *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pmbp->nmb_thispack - 1;
  auto geom = pmbp->pcoord->shell_geom;
  auto u0 = pmbp->pmhd->u0;
  auto &b0 = pmbp->pmhd->b0;
  const Real rho0 = g_rho0, gam = g_gam, p0 = g_p0;
  const Real Bref = g_B0, rref2 = g_r_ref * g_r_ref;
  const Real cs = std::sqrt(gam * p0 / rho0);

  Real lmax_v = 0.0, lmax_db1 = 0.0;
  Kokkos::parallel_reduce("monopole_diag",
    Kokkos::RangePolicy<>(DevExeSpace(),
                          0, (nmb1+1)*(ke-ks+1)*(je-js+1)*(ie-is+1)),
  KOKKOS_LAMBDA(const int idx, Real &mv, Real &mdb) {
    int nx1 = ie - is + 1, nx2 = je - js + 1, nx3 = ke - ks + 1;
    int nji = nx2 * nx1, nkji = nx3 * nx2 * nx1;
    int m = idx / nkji;
    int k = (idx - m*nkji) / nji;
    int j = (idx - m*nkji - k*nji) / nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks; j += js;
    Real rho = u0(m, IDN, k, j, i);
    Real vmag = Kokkos::sqrt(u0(m,IM1,k,j,i)*u0(m,IM1,k,j,i)
                            +u0(m,IM2,k,j,i)*u0(m,IM2,k,j,i)
                            +u0(m,IM3,k,j,i)*u0(m,IM3,k,j,i)) / rho;
    Real rf = geom.r_face(m, i);
    Real b1_an = Bref * rref2 / (rf * rf);
    Real db = Kokkos::fabs(b0.x1f(m, k, j, i) - b1_an);
    if (vmag > mv) mv = vmag;
    if (db   > mdb) mdb = db;
  }, Kokkos::Max<Real>(lmax_v), Kokkos::Max<Real>(lmax_db1));

  DivBStats s = ComputeDivBStats(pmbp);
  if (global_variable::my_rank == 0) {
    std::cout.precision(6);
    std::cout << std::scientific
              << "[sph_shell_mhd/monopole] cs0    = " << cs << "\n"
              << "[sph_shell_mhd/monopole] B0     = " << Bref << "\n"
              << "[sph_shell_mhd/monopole] r_ref  = " << g_r_ref << "\n"
              << "[sph_shell_mhd/monopole] max|v|/cs        = " << (lmax_v/cs) << "\n"
              << "[sph_shell_mhd/monopole] max|B1f-analytic|= " << lmax_db1
              << std::endl;
  }
  PrintDivBStats("sph_shell_mhd/monopole", s);
  if (pin->GetOrAddBoolean("problem", "debug_internal_interfaces", false)) {
    WriteMonopoleInterfaceDebug(pin, pm);
  }
}

//----------------------------------------------------------------------------------------
// Debug-only host diagnostic for the long-time HLLD transverse monopole drift.  It compares
// duplicate ghost/active states and duplicate radial fluxes on internal radial MeshBlock
// interfaces.  This is intentionally final-time only; run the same input with several tlim
// values to build a time history without adding a permanent task-list hook.

void WriteMonopoleInterfaceDebug(ParameterInput *pin, Mesh *pm) {
  if (global_variable::my_rank != 0) return;
  auto *pmbp = pm->pmb_pack;
  auto *pmhd = pmbp->pmhd;
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb = pmbp->nmb_thispack;
  const Real rref2 = g_r_ref * g_r_ref;
  const Real Bref = g_B0;

  const std::string out_dir =
      pin->GetOrAddString("problem", "debug_output_dir",
                          "monopole_hlld_interface_debug/csv");
  const std::string label =
      pin->GetOrAddString("problem", "debug_label", "monopole");
  const std::string state_path = out_dir + "/interface_state_jumps.csv";
  const std::string flux_path = out_dir + "/radial_riemann_interface_fluxes.csv";
  const std::string emf_path = out_dir + "/emf_interface_diagnostics.csv";
  const std::string global_path = out_dir + "/global_component_diagnostics.csv";

  auto h_w0 = Kokkos::create_mirror_view(pmhd->w0);
  auto h_bcc = Kokkos::create_mirror_view(pmhd->bcc0);
  auto h_b1 = Kokkos::create_mirror_view(pmhd->b0.x1f);
  auto h_b2 = Kokkos::create_mirror_view(pmhd->b0.x2f);
  auto h_b3 = Kokkos::create_mirror_view(pmhd->b0.x3f);
  auto h_flx1 = Kokkos::create_mirror_view(pmhd->uflx.x1f);
  auto h_e2x1 = Kokkos::create_mirror_view(pmhd->e2x1);
  auto h_e3x1 = Kokkos::create_mirror_view(pmhd->e3x1);
  auto h_e1 = Kokkos::create_mirror_view(pmhd->efld.x1e);
  auto h_e2 = Kokkos::create_mirror_view(pmhd->efld.x2e);
  auto h_e3 = Kokkos::create_mirror_view(pmhd->efld.x3e);
  auto h_rvol = Kokkos::create_mirror_view(pmbp->pcoord->shell_geom.r_vol);
  auto h_rface = Kokkos::create_mirror_view(pmbp->pcoord->shell_geom.r_face);
  Kokkos::deep_copy(h_w0, pmhd->w0);
  Kokkos::deep_copy(h_bcc, pmhd->bcc0);
  Kokkos::deep_copy(h_b1, pmhd->b0.x1f);
  Kokkos::deep_copy(h_b2, pmhd->b0.x2f);
  Kokkos::deep_copy(h_b3, pmhd->b0.x3f);
  Kokkos::deep_copy(h_flx1, pmhd->uflx.x1f);
  Kokkos::deep_copy(h_e2x1, pmhd->e2x1);
  Kokkos::deep_copy(h_e3x1, pmhd->e3x1);
  Kokkos::deep_copy(h_e1, pmhd->efld.x1e);
  Kokkos::deep_copy(h_e2, pmhd->efld.x2e);
  Kokkos::deep_copy(h_e3, pmhd->efld.x3e);
  Kokkos::deep_copy(h_rvol, pmbp->pcoord->shell_geom.r_vol);
  Kokkos::deep_copy(h_rface, pmbp->pcoord->shell_geom.r_face);
  auto &sizes = pmbp->pmb->mb_size.h_view;

  Real max_vr = 0.0, max_vth = 0.0, max_vph = 0.0, max_vperp = 0.0;
  Real max_vperp_excl = 0.0;
  Real max_bth = 0.0, max_bph = 0.0, max_bt = 0.0;
  int loc_vperp_m = -1, loc_vperp_k = -1, loc_vperp_j = -1, loc_vperp_i = -1;
  Real loc_vperp_r = 0.0, loc_vperp_theta = 0.0, loc_vperp_phi = 0.0;
  Real loc_dist_radial_interface = 0.0;
  const int nexcl = 2;
  for (int m = 0; m < nmb; ++m) {
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie; ++i) {
          const Real vr = std::fabs(h_w0(m,IVX,k,j,i));
          const Real vt = std::fabs(h_w0(m,IVY,k,j,i));
          const Real vp = std::fabs(h_w0(m,IVZ,k,j,i));
          const Real vp2 = std::hypot(h_w0(m,IVY,k,j,i), h_w0(m,IVZ,k,j,i));
          const Real bt2 = std::hypot(h_bcc(m,IBY,k,j,i), h_bcc(m,IBZ,k,j,i));
          max_vr = std::max(max_vr, vr);
          max_vth = std::max(max_vth, vt);
          max_vph = std::max(max_vph, vp);
          if (i >= is+nexcl && i <= ie-nexcl &&
              j >= js+nexcl && j <= je-nexcl &&
              k >= ks+nexcl && k <= ke-nexcl) {
            max_vperp_excl = std::max(max_vperp_excl, vp2);
          }
          max_bth = std::max(max_bth, std::fabs(h_bcc(m,IBY,k,j,i)));
          max_bph = std::max(max_bph, std::fabs(h_bcc(m,IBZ,k,j,i)));
          max_bt = std::max(max_bt, bt2);
          if (vp2 > max_vperp) {
            max_vperp = vp2;
            loc_vperp_m = m;
            loc_vperp_k = k;
            loc_vperp_j = j;
            loc_vperp_i = i;
            loc_vperp_r = h_rvol(m, i);
            loc_vperp_theta =
                sizes(m).x2min + (static_cast<Real>(j - js) + 0.5)*sizes(m).dx2;
            loc_vperp_phi =
                sizes(m).x3min + (static_cast<Real>(k - ks) + 0.5)*sizes(m).dx3;
            loc_dist_radial_interface =
                std::min(std::fabs(loc_vperp_r - h_rface(m, is)),
                         std::fabs(loc_vperp_r - h_rface(m, ie+1)));
          }
        }
      }
    }
  }

  const bool global_exists = static_cast<bool>(std::ifstream(global_path));
  std::ofstream global_out(global_path, std::ios::app);
  if (!global_out) {
    std::cout << "[monopole_interface_debug] ERROR opening " << global_path << std::endl;
    return;
  }
  global_out << std::setprecision(16);
  if (!global_exists) {
    global_out
      << "label,time,cycle,max_vr,max_vtheta,max_vphi,max_vperp,"
      << "max_vperp_excluding_2cells,max_Btheta,max_Bphi,max_Btrans,"
      << "loc_vperp_m,loc_vperp_k,loc_vperp_j,loc_vperp_i,"
      << "loc_vperp_r,loc_vperp_theta,loc_vperp_phi,"
      << "loc_dist_radial_interface\n";
  }
  global_out << label << ',' << pm->time << ',' << pm->ncycle << ','
             << max_vr << ',' << max_vth << ',' << max_vph << ',' << max_vperp
             << ',' << max_vperp_excl << ',' << max_bth << ',' << max_bph << ',' << max_bt << ','
             << loc_vperp_m << ',' << loc_vperp_k << ',' << loc_vperp_j << ','
             << loc_vperp_i << ',' << loc_vperp_r << ',' << loc_vperp_theta << ','
             << loc_vperp_phi << ',' << loc_dist_radial_interface << '\n';

  struct RadialPair {
    int left, right;
  };
  std::vector<RadialPair> pairs;
  constexpr Real tol = 1.0e-10;
  for (int ml = 0; ml < nmb; ++ml) {
    for (int mr = 0; mr < nmb; ++mr) {
      if (ml == mr) continue;
      const bool radial_touch =
          std::fabs(sizes(ml).x1max - sizes(mr).x1min) <=
          tol * std::max({1.0, std::fabs(sizes(ml).x1max), std::fabs(sizes(mr).x1min)});
      const bool same_x2 =
          std::fabs(sizes(ml).x2min - sizes(mr).x2min) <= tol &&
          std::fabs(sizes(ml).x2max - sizes(mr).x2max) <= tol;
      const bool same_x3 =
          std::fabs(sizes(ml).x3min - sizes(mr).x3min) <= tol &&
          std::fabs(sizes(ml).x3max - sizes(mr).x3max) <= tol;
      if (radial_touch && same_x2 && same_x3) pairs.push_back({ml, mr});
    }
  }

  const bool state_exists = static_cast<bool>(std::ifstream(state_path));
  std::ofstream state_out(state_path, std::ios::app);
  if (!state_out) {
    std::cout << "[monopole_interface_debug] ERROR opening " << state_path << std::endl;
    return;
  }
  state_out << std::setprecision(16);
  if (!state_exists) {
    state_out
      << "label,time,cycle,left_gid,right_gid,left_m,right_m,r_face,"
      << "max_lghost_ractive_rho,max_lghost_ractive_p,"
      << "max_lghost_ractive_vr,max_lghost_ractive_vtheta,max_lghost_ractive_vphi,"
      << "max_lghost_ractive_bcc_r,max_lghost_ractive_bcc_theta,max_lghost_ractive_bcc_phi,"
      << "max_rghost_lactive_rho,max_rghost_lactive_p,"
      << "max_rghost_lactive_vr,max_rghost_lactive_vtheta,max_rghost_lactive_vphi,"
      << "max_rghost_lactive_bcc_r,max_rghost_lactive_bcc_theta,max_rghost_lactive_bcc_phi,"
      << "max_shared_b1f_jump,max_shared_b1f_analytic_err_left,"
      << "max_shared_b1f_analytic_err_right,max_lactive_vperp,max_ractive_vperp,"
      << "max_lactive_bt,max_ractive_bt,max_lghost_transverse_face_b,"
      << "max_rghost_transverse_face_b\n";
  }

  const bool flux_exists = static_cast<bool>(std::ifstream(flux_path));
  std::ofstream flux_out(flux_path, std::ios::app);
  if (!flux_out) {
    std::cout << "[monopole_interface_debug] ERROR opening " << flux_path << std::endl;
    return;
  }
  flux_out << std::setprecision(16);
  if (!flux_exists) {
    flux_out
      << "label,time,cycle,left_gid,right_gid,left_m,right_m,r_face,"
      << "max_flux_rho_left,max_flux_rho_right,max_flux_rho_jump,"
      << "max_flux_mr_left,max_flux_mr_right,max_flux_mr_jump,"
      << "max_flux_mtheta_left,max_flux_mtheta_right,max_flux_mtheta_jump,"
      << "max_flux_mphi_left,max_flux_mphi_right,max_flux_mphi_jump,"
      << "max_flux_energy_left,max_flux_energy_right,max_flux_energy_jump,"
      << "max_e2x1_left,max_e2x1_right,max_e2x1_jump,"
      << "max_e3x1_left,max_e3x1_right,max_e3x1_jump\n";
  }

  auto max_abs = [](Real a, Real b) { return std::max(std::fabs(a), std::fabs(b)); };

  for (const auto &pr : pairs) {
    const int ml = pr.left;
    const int mr = pr.right;
    const int gl = pmbp->pmb->mb_gid.h_view(ml);
    const int gr = pmbp->pmb->mb_gid.h_view(mr);
    const Real rf = sizes(ml).x1max;
    const Real b_an = Bref * rref2 / (rf * rf);

    Real lg_ra[8] = {0.0};
    Real rg_la[8] = {0.0};
    Real b1_jump = 0.0, b1_err_l = 0.0, b1_err_r = 0.0;
    Real vperp_l = 0.0, vperp_r = 0.0, bt_l = 0.0, bt_r = 0.0;
    Real lghost_btf = 0.0, rghost_btf = 0.0;

    Real flux_l[5] = {0.0}, flux_r[5] = {0.0}, flux_j[5] = {0.0};
    Real e2_l = 0.0, e2_r = 0.0, e2_j = 0.0;
    Real e3_l = 0.0, e3_r = 0.0, e3_j = 0.0;

    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
        const int il_g = ie + 1;
        const int ir_g = is - 1;
        lg_ra[0] = std::max(lg_ra[0], std::fabs(h_w0(ml,IDN,k,j,il_g) - h_w0(mr,IDN,k,j,is)));
        lg_ra[1] = std::max(lg_ra[1], std::fabs(h_w0(ml,IPR,k,j,il_g) - h_w0(mr,IPR,k,j,is)));
        lg_ra[2] = std::max(lg_ra[2], std::fabs(h_w0(ml,IVX,k,j,il_g) - h_w0(mr,IVX,k,j,is)));
        lg_ra[3] = std::max(lg_ra[3], std::fabs(h_w0(ml,IVY,k,j,il_g) - h_w0(mr,IVY,k,j,is)));
        lg_ra[4] = std::max(lg_ra[4], std::fabs(h_w0(ml,IVZ,k,j,il_g) - h_w0(mr,IVZ,k,j,is)));
        lg_ra[5] = std::max(lg_ra[5], std::fabs(h_bcc(ml,IBX,k,j,il_g) - h_bcc(mr,IBX,k,j,is)));
        lg_ra[6] = std::max(lg_ra[6], std::fabs(h_bcc(ml,IBY,k,j,il_g) - h_bcc(mr,IBY,k,j,is)));
        lg_ra[7] = std::max(lg_ra[7], std::fabs(h_bcc(ml,IBZ,k,j,il_g) - h_bcc(mr,IBZ,k,j,is)));

        rg_la[0] = std::max(rg_la[0], std::fabs(h_w0(mr,IDN,k,j,ir_g) - h_w0(ml,IDN,k,j,ie)));
        rg_la[1] = std::max(rg_la[1], std::fabs(h_w0(mr,IPR,k,j,ir_g) - h_w0(ml,IPR,k,j,ie)));
        rg_la[2] = std::max(rg_la[2], std::fabs(h_w0(mr,IVX,k,j,ir_g) - h_w0(ml,IVX,k,j,ie)));
        rg_la[3] = std::max(rg_la[3], std::fabs(h_w0(mr,IVY,k,j,ir_g) - h_w0(ml,IVY,k,j,ie)));
        rg_la[4] = std::max(rg_la[4], std::fabs(h_w0(mr,IVZ,k,j,ir_g) - h_w0(ml,IVZ,k,j,ie)));
        rg_la[5] = std::max(rg_la[5], std::fabs(h_bcc(mr,IBX,k,j,ir_g) - h_bcc(ml,IBX,k,j,ie)));
        rg_la[6] = std::max(rg_la[6], std::fabs(h_bcc(mr,IBY,k,j,ir_g) - h_bcc(ml,IBY,k,j,ie)));
        rg_la[7] = std::max(rg_la[7], std::fabs(h_bcc(mr,IBZ,k,j,ir_g) - h_bcc(ml,IBZ,k,j,ie)));

        b1_jump = std::max(b1_jump, std::fabs(h_b1(ml,k,j,ie+1) - h_b1(mr,k,j,is)));
        b1_err_l = std::max(b1_err_l, std::fabs(h_b1(ml,k,j,ie+1) - b_an));
        b1_err_r = std::max(b1_err_r, std::fabs(h_b1(mr,k,j,is) - b_an));
        vperp_l = std::max(vperp_l, std::hypot(h_w0(ml,IVY,k,j,ie), h_w0(ml,IVZ,k,j,ie)));
        vperp_r = std::max(vperp_r, std::hypot(h_w0(mr,IVY,k,j,is), h_w0(mr,IVZ,k,j,is)));
        bt_l = std::max(bt_l, std::hypot(h_bcc(ml,IBY,k,j,ie), h_bcc(ml,IBZ,k,j,ie)));
        bt_r = std::max(bt_r, std::hypot(h_bcc(mr,IBY,k,j,is), h_bcc(mr,IBZ,k,j,is)));
        lghost_btf = std::max(lghost_btf, max_abs(h_b2(ml,k,j,ie+1), h_b3(ml,k,j,ie+1)));
        rghost_btf = std::max(rghost_btf, max_abs(h_b2(mr,k,j,ir_g), h_b3(mr,k,j,ir_g)));

        const int vars[5] = {IDN, IM1, IM2, IM3, IEN};
        for (int q = 0; q < 5; ++q) {
          const int n = vars[q];
          flux_l[q] = std::max(flux_l[q], std::fabs(h_flx1(ml,n,k,j,ie+1)));
          flux_r[q] = std::max(flux_r[q], std::fabs(h_flx1(mr,n,k,j,is)));
          flux_j[q] = std::max(flux_j[q], std::fabs(h_flx1(ml,n,k,j,ie+1) -
                                                     h_flx1(mr,n,k,j,is)));
        }
        e2_l = std::max(e2_l, std::fabs(h_e2x1(ml,k,j,ie+1)));
        e2_r = std::max(e2_r, std::fabs(h_e2x1(mr,k,j,is)));
        e2_j = std::max(e2_j, std::fabs(h_e2x1(ml,k,j,ie+1) - h_e2x1(mr,k,j,is)));
        e3_l = std::max(e3_l, std::fabs(h_e3x1(ml,k,j,ie+1)));
        e3_r = std::max(e3_r, std::fabs(h_e3x1(mr,k,j,is)));
        e3_j = std::max(e3_j, std::fabs(h_e3x1(ml,k,j,ie+1) - h_e3x1(mr,k,j,is)));
      }
    }

    state_out << label << ',' << pm->time << ',' << pm->ncycle << ','
              << gl << ',' << gr << ',' << ml << ',' << mr << ',' << rf;
    for (Real x : lg_ra) state_out << ',' << x;
    for (Real x : rg_la) state_out << ',' << x;
    state_out << ',' << b1_jump << ',' << b1_err_l << ',' << b1_err_r
              << ',' << vperp_l << ',' << vperp_r << ',' << bt_l << ',' << bt_r
              << ',' << lghost_btf << ',' << rghost_btf << '\n';

    flux_out << label << ',' << pm->time << ',' << pm->ncycle << ','
             << gl << ',' << gr << ',' << ml << ',' << mr << ',' << rf;
    for (int q = 0; q < 5; ++q) {
      flux_out << ',' << flux_l[q] << ',' << flux_r[q] << ',' << flux_j[q];
    }
    flux_out << ',' << e2_l << ',' << e2_r << ',' << e2_j
             << ',' << e3_l << ',' << e3_r << ',' << e3_j << '\n';
  }

  Real max_e1 = 0.0, max_e2 = 0.0, max_e3 = 0.0;
  for (int m = 0; m < nmb; ++m) {
    for (int k = ks; k <= ke+1; ++k) {
      for (int j = js; j <= je+1; ++j) {
        for (int i = is; i <= ie; ++i) max_e1 = std::max(max_e1, std::fabs(h_e1(m,k,j,i)));
      }
    }
    for (int k = ks; k <= ke+1; ++k) {
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie+1; ++i) max_e2 = std::max(max_e2, std::fabs(h_e2(m,k,j,i)));
      }
    }
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je+1; ++j) {
        for (int i = is; i <= ie+1; ++i) max_e3 = std::max(max_e3, std::fabs(h_e3(m,k,j,i)));
      }
    }
  }
  Real max_if_e2x1 = 0.0, max_if_e3x1 = 0.0;
  for (const auto &pr : pairs) {
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
        max_if_e2x1 = std::max(max_if_e2x1, std::fabs(h_e2x1(pr.left,k,j,ie+1)));
        max_if_e2x1 = std::max(max_if_e2x1, std::fabs(h_e2x1(pr.right,k,j,is)));
        max_if_e3x1 = std::max(max_if_e3x1, std::fabs(h_e3x1(pr.left,k,j,ie+1)));
        max_if_e3x1 = std::max(max_if_e3x1, std::fabs(h_e3x1(pr.right,k,j,is)));
      }
    }
  }

  const bool emf_exists = static_cast<bool>(std::ifstream(emf_path));
  std::ofstream emf_out(emf_path, std::ios::app);
  if (!emf_out) {
    std::cout << "[monopole_interface_debug] ERROR opening " << emf_path << std::endl;
    return;
  }
  emf_out << std::setprecision(16);
  if (!emf_exists) {
    emf_out << "label,time,cycle,n_internal_radial_interfaces,"
            << "max_corner_e1,max_corner_e2,max_corner_e3,"
            << "max_interface_e2x1,max_interface_e3x1\n";
  }
  emf_out << label << ',' << pm->time << ',' << pm->ncycle << ','
          << pairs.size() << ',' << max_e1 << ',' << max_e2 << ',' << max_e3
          << ',' << max_if_e2x1 << ',' << max_if_e3x1 << '\n';
  std::cout << "[monopole_interface_debug] wrote " << pairs.size()
            << " internal radial interface rows to " << out_dir << std::endl;
}

//----------------------------------------------------------------------------------------
// Analytic radial ghost zones for tests with a radial 1/r^2 monopole background.
// Plain outflow copies B_r into radial ghosts, which breaks the 1/r^2 profile and
// launches a boundary-layer Lorentz-force residual even though divB remains at roundoff.

void MonopoleRadialBCs(Mesh *pm) {
  if (g_mode != Mode::kMonopole && g_mode != Mode::kRadialAlfven) return;
  auto *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int ng = indcs.ng;
  const int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  const int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
  const int nmb1 = pmbp->nmb_thispack - 1;
  auto geom = pmbp->pcoord->shell_geom;
  auto u0 = pmbp->pmhd->u0;
  auto &b0 = pmbp->pmhd->b0;
  auto &eos = pmbp->pmhd->peos->eos_data;
  auto &mb_bcs = pmbp->pmb->mb_bcs;
  const Real rho0 = g_rho0;
  const Real p0 = g_p0;
  const Real gam = eos.gamma;
  const Real Bref = g_B0;
  const Real rref2 = g_r_ref * g_r_ref;

  par_for("sph_mhd_monopole_radial_b", DevExeSpace(),
          0, nmb1, 0, n3-1, 0, n2-1, 0, ng-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int g) {
    if (mb_bcs.d_view(m, BoundaryFace::inner_x1) == BoundaryFlag::user) {
      const int i = is - g - 1;
      const Real rf = geom.r_face(m, i);
      b0.x1f(m, k, j, i) = Bref * rref2 / (rf * rf);
      b0.x2f(m, k, j, i) = 0.0;
      if (j == n2-1) b0.x2f(m, k, j+1, i) = 0.0;
      b0.x3f(m, k, j, i) = 0.0;
      if (k == n3-1) b0.x3f(m, k+1, j, i) = 0.0;
    }
    if (mb_bcs.d_view(m, BoundaryFace::outer_x1) == BoundaryFlag::user) {
      const int i = ie + g + 2;
      const Real rf = geom.r_face(m, i);
      b0.x1f(m, k, j, i) = Bref * rref2 / (rf * rf);
      b0.x2f(m, k, j, i-1) = 0.0;
      if (j == n2-1) b0.x2f(m, k, j+1, i-1) = 0.0;
      b0.x3f(m, k, j, i-1) = 0.0;
      if (k == n3-1) b0.x3f(m, k+1, j, i-1) = 0.0;
    }
  });

  par_for("sph_mhd_monopole_radial_u", DevExeSpace(),
          0, nmb1, 0, n3-1, 0, n2-1, 0, ng-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int g) {
    if (mb_bcs.d_view(m, BoundaryFace::inner_x1) == BoundaryFlag::user) {
      const int i = is - g - 1;
      const Real br_cc = 0.5 * (b0.x1f(m, k, j, i) + b0.x1f(m, k, j, i+1));
      u0(m, IDN, k, j, i) = rho0;
      u0(m, IM1, k, j, i) = 0.0;
      u0(m, IM2, k, j, i) = 0.0;
      u0(m, IM3, k, j, i) = 0.0;
      u0(m, IEN, k, j, i) = p0 / (gam - 1.0) + 0.5 * br_cc * br_cc;
    }
    if (mb_bcs.d_view(m, BoundaryFace::outer_x1) == BoundaryFlag::user) {
      const int i = ie + g + 1;
      const Real br_cc = 0.5 * (b0.x1f(m, k, j, i) + b0.x1f(m, k, j, i+1));
      u0(m, IDN, k, j, i) = rho0;
      u0(m, IM1, k, j, i) = 0.0;
      u0(m, IM2, k, j, i) = 0.0;
      u0(m, IM3, k, j, i) = 0.0;
      u0(m, IEN, k, j, i) = p0 / (gam - 1.0) + 0.5 * br_cc * br_cc;
    }
	  });
	}

void DrivenAlfvenRadialBCs(Mesh *pm) {
  if (g_mode != Mode::kDrivenAlfven) return;
  auto *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int ng = indcs.ng;
  const int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  const int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
  const int nmb1 = pmbp->nmb_thispack - 1;
  auto geom = pmbp->pcoord->shell_geom;
  auto u0 = pmbp->pmhd->u0;
  auto &b0 = pmbp->pmhd->b0;
  auto &eos = pmbp->pmhd->peos->eos_data;
  auto &mb_bcs = pmbp->pmb->mb_bcs;
  const Real rho0 = g_rho0;
  const Real p0 = g_p0;
  const Real gam = eos.gamma;
  const Real Bref = g_B0;
  const Real rref2 = g_r_ref * g_r_ref;
  const Real tdrive = pm->time - g_drv_start;
  Real ramp = 0.0;
  if (tdrive > 0.0) {
    if (g_drv_ramp > 0.0 && tdrive < g_drv_ramp) {
      const Real arg = 1.57079632679489661923 * tdrive / g_drv_ramp;
      ramp = std::sin(arg);
      ramp *= ramp;
    } else {
      ramp = 1.0;
    }
  }
  const Real vdrv = g_drv_amp * ramp * std::sin(g_drv_omega * tdrive + g_drv_phase);
  const Real bdrv = g_drv_b_sign * std::sqrt(rho0) * vdrv;

  par_for("sph_mhd_driven_aw_radial_b", DevExeSpace(),
          0, nmb1, 0, n3-1, 0, n2-1, 0, ng-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int g) {
    if (mb_bcs.d_view(m, BoundaryFace::inner_x1) == BoundaryFlag::user) {
      const int i = is - g - 1;
      const Real rf = geom.r_face(m, i);
      b0.x1f(m, k, j, i) = Bref * rref2 / (rf * rf);
      b0.x2f(m, k, j, i) = 0.0;
      if (j == n2-1) b0.x2f(m, k, j+1, i) = 0.0;
      b0.x3f(m, k, j, i) = bdrv;
      if (k == n3-1) b0.x3f(m, k+1, j, i) = bdrv;
    }
    if (mb_bcs.d_view(m, BoundaryFace::outer_x1) == BoundaryFlag::user) {
      const int i_face = ie + g + 2;
      const int i_cell = ie + g + 1;
      const Real rf = geom.r_face(m, i_face);
      b0.x1f(m, k, j, i_face) = Bref * rref2 / (rf * rf);
      b0.x2f(m, k, j, i_cell) = 0.0;
      if (j == n2-1) b0.x2f(m, k, j+1, i_cell) = 0.0;
      b0.x3f(m, k, j, i_cell) = b0.x3f(m, k, j, ie);
      if (k == n3-1) b0.x3f(m, k+1, j, i_cell) = b0.x3f(m, k+1, j, ie);
    }
  });

  par_for("sph_mhd_driven_aw_radial_u", DevExeSpace(),
          0, nmb1, 0, n3-1, 0, n2-1, 0, ng-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int g) {
    if (mb_bcs.d_view(m, BoundaryFace::inner_x1) == BoundaryFlag::user) {
      const int i = is - g - 1;
      const Real br_cc = 0.5 * (b0.x1f(m, k, j, i) + b0.x1f(m, k, j, i+1));
      const Real eb = 0.5 * (br_cc * br_cc + bdrv * bdrv);
      u0(m, IDN, k, j, i) = rho0;
      u0(m, IM1, k, j, i) = 0.0;
      u0(m, IM2, k, j, i) = 0.0;
      u0(m, IM3, k, j, i) = rho0 * vdrv;
      u0(m, IEN, k, j, i) = p0 / (gam - 1.0) + 0.5 * rho0 * vdrv * vdrv + eb;
    }
    if (mb_bcs.d_view(m, BoundaryFace::outer_x1) == BoundaryFlag::user) {
      const int i = ie + g + 1;
      const Real br_cc = 0.5 * (b0.x1f(m, k, j, i) + b0.x1f(m, k, j, i+1));
      const Real vphi = u0(m, IM3, k, j, ie) / u0(m, IDN, k, j, ie);
      const Real bphi = b0.x3f(m, k, j, i);
      const Real eb = 0.5 * (br_cc * br_cc + bphi * bphi);
      u0(m, IDN, k, j, i) = rho0;
      u0(m, IM1, k, j, i) = 0.0;
      u0(m, IM2, k, j, i) = 0.0;
      u0(m, IM3, k, j, i) = rho0 * vphi;
      u0(m, IEN, k, j, i) = p0 / (gam - 1.0) + 0.5 * rho0 * vphi * vphi + eb;
    }
  });
}

void ParkerMhdRadialBCs(Mesh *pm) {
  if (g_mode != Mode::kParkerIsothermal) return;
  auto *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int ng = indcs.ng;
  const int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  const int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
  const int nmb1 = pmbp->nmb_thispack - 1;
  auto geom = pmbp->pcoord->shell_geom;
  auto u0 = pmbp->pmhd->u0;
  auto &b0 = pmbp->pmhd->b0;
  auto &mb_bcs = pmbp->pmb->mb_bcs;
  const bool outer_analytic = g_parker_outer_analytic;
  const Real cs = g_parker_csiso;
  const Real rc = g_parker_rc;
  const Real r_inner = g_parker_r_inner;
  const Real rho_inner = g_parker_rho_inner;
  const Real M_inner = g_parker_M_inner;
  const Real Bref = g_B0;
  const Real rref2 = g_r_ref * g_r_ref;

  par_for("sph_mhd_parker_radial_b", DevExeSpace(),
          0, nmb1, 0, n3-1, 0, n2-1, 0, ng-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int g) {
    if (mb_bcs.d_view(m, BoundaryFace::inner_x1) == BoundaryFlag::user) {
      const int i = is - g - 1;
      const Real rf = geom.r_face(m, i);
      b0.x1f(m, k, j, i) = Bref * rref2 / (rf * rf);
      b0.x2f(m, k, j, i) = 0.0;
      if (j == n2-1) b0.x2f(m, k, j+1, i) = 0.0;
      b0.x3f(m, k, j, i) = 0.0;
      if (k == n3-1) b0.x3f(m, k+1, j, i) = 0.0;
    }
    if (mb_bcs.d_view(m, BoundaryFace::outer_x1) == BoundaryFlag::user) {
      const int i = ie + g + 2;
      const Real rf = geom.r_face(m, i);
      b0.x1f(m, k, j, i) = Bref * rref2 / (rf * rf);
      b0.x2f(m, k, j, i-1) = 0.0;
      if (j == n2-1) b0.x2f(m, k, j+1, i-1) = 0.0;
      b0.x3f(m, k, j, i-1) = 0.0;
      if (k == n3-1) b0.x3f(m, k+1, j, i-1) = 0.0;
    }
  });

  par_for("sph_mhd_parker_radial_u", DevExeSpace(),
          0, nmb1, 0, n3-1, 0, n2-1, 0, ng-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int g) {
    if (mb_bcs.d_view(m, BoundaryFace::inner_x1) == BoundaryFlag::user) {
      const int i = is - g - 1;
      const Real r = geom.r_vol(m, i);
      const Real M = ParkerMachAtR(r, rc);
      const Real rho = ParkerRhoAtR(r, rc, r_inner, rho_inner, M_inner);
      const Real vr = M * cs;
      u0(m, IDN, k, j, i) = rho;
      u0(m, IM1, k, j, i) = rho * vr;
      u0(m, IM2, k, j, i) = 0.0;
      u0(m, IM3, k, j, i) = 0.0;
    }
    if (mb_bcs.d_view(m, BoundaryFace::outer_x1) == BoundaryFlag::user) {
      const int i = ie + g + 1;
      if (outer_analytic) {
        const Real r = geom.r_vol(m, i);
        const Real M = ParkerMachAtR(r, rc);
        const Real rho = ParkerRhoAtR(r, rc, r_inner, rho_inner, M_inner);
        const Real vr = M * cs;
        u0(m, IDN, k, j, i) = rho;
        u0(m, IM1, k, j, i) = rho * vr;
        u0(m, IM2, k, j, i) = 0.0;
        u0(m, IM3, k, j, i) = 0.0;
      } else {
        u0(m, IDN, k, j, i) = u0(m, IDN, k, j, ie);
        u0(m, IM1, k, j, i) = u0(m, IM1, k, j, ie);
        u0(m, IM2, k, j, i) = u0(m, IM2, k, j, ie);
        u0(m, IM3, k, j, i) = u0(m, IM3, k, j, ie);
      }
    }
  });
}

//----------------------------------------------------------------------------------------
// toroidal_static: report divB stats + max |v|/cs. Field will accelerate (not in
// pressure balance) but divB should remain near roundoff.

void ToroidalStaticFinalize(ParameterInput *pin, Mesh *pm) {
  if (g_mode != Mode::kToroidalStatic) return;
  auto *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pmbp->nmb_thispack - 1;
  auto u0 = pmbp->pmhd->u0;
  const Real rho0 = g_rho0, gam = g_gam, p0 = g_p0;
  const Real cs = std::sqrt(gam * p0 / rho0);
  Real lmax_v = 0.0;
  Kokkos::parallel_reduce("toroidal_diag",
    Kokkos::RangePolicy<>(DevExeSpace(),
                          0, (nmb1+1)*(ke-ks+1)*(je-js+1)*(ie-is+1)),
  KOKKOS_LAMBDA(const int idx, Real &mv) {
    int nx1 = ie - is + 1, nx2 = je - js + 1, nx3 = ke - ks + 1;
    int nji = nx2 * nx1, nkji = nx3 * nx2 * nx1;
    int m = idx / nkji;
    int k = (idx - m*nkji) / nji;
    int j = (idx - m*nkji - k*nji) / nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks; j += js;
    Real rho = u0(m, IDN, k, j, i);
    Real vmag = Kokkos::sqrt(u0(m,IM1,k,j,i)*u0(m,IM1,k,j,i)
                            +u0(m,IM2,k,j,i)*u0(m,IM2,k,j,i)
                            +u0(m,IM3,k,j,i)*u0(m,IM3,k,j,i)) / rho;
    if (vmag > mv) mv = vmag;
  }, Kokkos::Max<Real>(lmax_v));
  DivBStats s = ComputeDivBStats(pmbp);
  if (global_variable::my_rank == 0) {
    std::cout.precision(6);
    std::cout << std::scientific
              << "[sph_shell_mhd/toroidal_static] cs0=" << cs
              << " max|v|/cs=" << (lmax_v/cs) << std::endl;
  }
  PrintDivBStats("sph_shell_mhd/toroidal_static", s);
}

//----------------------------------------------------------------------------------------
// parker_isothermal: compare MHD state against analytic isothermal Parker wind plus
// radial monopole B. This mode requires isothermal MHD, so pressure is p=cs^2 rho.

void ParkerIsothermalFinalize(ParameterInput *pin, Mesh *pm) {
  if (g_mode != Mode::kParkerIsothermal) return;
  auto *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pmbp->nmb_thispack - 1;
  auto geom = pmbp->pcoord->shell_geom;
  auto u0 = pmbp->pmhd->u0;
  const Real cs = g_parker_csiso;
  const Real rc = g_parker_rc;
  const Real r_inner = g_parker_r_inner;
  const Real rho_inner = g_parker_rho_inner;
  const Real M_inner = g_parker_M_inner;
  const Real mdot_ref = rho_inner * (M_inner * cs) * r_inner * r_inner;
  const int Ncell = (nmb1+1)*(ke-ks+1)*(je-js+1)*(ie-is+1);

  Real linf_v = 0.0, linf_rho = 0.0, linf_mdot = 0.0;
  Real l1_v = 0.0, l2_v = 0.0;
  Real l1_rho = 0.0, l2_rho = 0.0;
  Real l1_mdot = 0.0;
  Real max_vtheta = 0.0, max_vphi = 0.0;
  Kokkos::parallel_reduce("parker_mhd_diag",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, Ncell),
  KOKKOS_LAMBDA(const int idx,
                Real &sv, Real &sr, Real &sm,
                Real &s2v, Real &s2r,
                Real &mxv, Real &mxr, Real &mxm,
                Real &mvt, Real &mvp) {
    int nx1 = ie - is + 1, nx2 = je - js + 1, nx3 = ke - ks + 1;
    int nji = nx2 * nx1, nkji = nx3 * nx2 * nx1;
    int m = idx / nkji;
    int k = (idx - m*nkji) / nji;
    int j = (idx - m*nkji - k*nji) / nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks; j += js;
    Real r = geom.r_vol(m, i);
    Real M = ParkerMachAtR(r, rc);
    Real vr_an = M * cs;
    Real rho_an = ParkerRhoAtR(r, rc, r_inner, rho_inner, M_inner);
    Real rho = u0(m, IDN, k, j, i);
    Real vr = u0(m, IM1, k, j, i) / rho;
    Real vtheta = u0(m, IM2, k, j, i) / rho;
    Real vphi = u0(m, IM3, k, j, i) / rho;
    Real mdot = rho * vr * r * r;
    Real ev = Kokkos::fabs((vr - vr_an) / vr_an);
    Real er = Kokkos::fabs((rho - rho_an) / rho_an);
    Real em = Kokkos::fabs((mdot - mdot_ref) / mdot_ref);
    sv += ev;
    sr += er;
    sm += em;
    s2v += ev * ev;
    s2r += er * er;
    if (ev > mxv) mxv = ev;
    if (er > mxr) mxr = er;
    if (em > mxm) mxm = em;
    Real avt = Kokkos::fabs(vtheta);
    Real avp = Kokkos::fabs(vphi);
    if (avt > mvt) mvt = avt;
    if (avp > mvp) mvp = avp;
  }, l1_v, l1_rho, l1_mdot, l2_v, l2_rho,
     Kokkos::Max<Real>(linf_v), Kokkos::Max<Real>(linf_rho),
     Kokkos::Max<Real>(linf_mdot), Kokkos::Max<Real>(max_vtheta),
     Kokkos::Max<Real>(max_vphi));

  l1_v /= static_cast<Real>(Ncell);
  l1_rho /= static_cast<Real>(Ncell);
  l1_mdot /= static_cast<Real>(Ncell);
  l2_v = std::sqrt(l2_v / static_cast<Real>(Ncell));
  l2_rho = std::sqrt(l2_rho / static_cast<Real>(Ncell));

  DivBStats s = ComputeDivBStats(pmbp);
  if (global_variable::my_rank == 0) {
    std::cout.precision(6);
    std::cout << std::scientific
              << "[sph_shell_mhd/parker_isothermal] t              = " << pm->time << "\n"
              << "[sph_shell_mhd/parker_isothermal] cs_iso         = " << cs << "\n"
              << "[sph_shell_mhd/parker_isothermal] gm             = " << g_parker_gm << "\n"
              << "[sph_shell_mhd/parker_isothermal] r_c            = " << rc << "\n"
              << "[sph_shell_mhd/parker_isothermal] M(r_inner)     = " << M_inner << "\n"
              << "[sph_shell_mhd/parker_isothermal] mdot_ref       = " << mdot_ref << "\n"
              << "[sph_shell_mhd/parker_isothermal] L1 |dv_r/v_r|  = " << l1_v << "\n"
              << "[sph_shell_mhd/parker_isothermal] L2 |dv_r/v_r|  = " << l2_v << "\n"
              << "[sph_shell_mhd/parker_isothermal] Linf|dv_r/v_r| = " << linf_v << "\n"
              << "[sph_shell_mhd/parker_isothermal] L1 |drho/rho|  = " << l1_rho << "\n"
              << "[sph_shell_mhd/parker_isothermal] L2 |drho/rho|  = " << l2_rho << "\n"
              << "[sph_shell_mhd/parker_isothermal] Linf|drho/rho| = " << linf_rho << "\n"
              << "[sph_shell_mhd/parker_isothermal] L1 |dmdot/mdot|= " << l1_mdot << "\n"
              << "[sph_shell_mhd/parker_isothermal] Linf|dmdot/mdot|= " << linf_mdot << "\n"
              << "[sph_shell_mhd/parker_isothermal] max|v_theta|/cs= "
              << (max_vtheta / cs) << "\n"
              << "[sph_shell_mhd/parker_isothermal] max|v_phi|/cs  = "
              << (max_vphi / cs) << std::endl;
  }
  PrintDivBStats("sph_shell_mhd/parker_isothermal", s);
}

//----------------------------------------------------------------------------------------
// radial_alfven: report pulse centroid in r vs WKB prediction, peak amplitude, divB.
//
// WKB ray:  r(t) = (r_c^3 + 3 r_ref^2 v_A0 t)^(1/3)
// The optional WKB profile measures amplitudes against sqrt(v_A(r)/v_A(r_c)).

void RadialAlfvenFinalize(ParameterInput *pin, Mesh *pm) {
  if (g_mode != Mode::kRadialAlfven) return;
  auto *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pmbp->nmb_thispack - 1;
  auto geom = pmbp->pcoord->shell_geom;
  auto u0   = pmbp->pmhd->u0;
  auto bcc  = pmbp->pmhd->bcc0;
  const Real rho0  = g_rho0;
  const Real rref  = g_r_ref;
  const Real rc0   = g_aw_rc;
  const Real vA0   = g_aw_vA0;
  const Real eps   = g_aw_eps;
  const Real B0    = g_B0;
  const Real t     = pm->time;

  // |delta B_phi|^2-weighted radial centroid + peak amplitude over active cells.
  Real num = 0.0, den = 0.0, peak_dB = 0.0, peak_dv = 0.0;
  Kokkos::parallel_reduce("aw_centroid",
    Kokkos::RangePolicy<>(DevExeSpace(),
                          0, (nmb1+1)*(ke-ks+1)*(je-js+1)*(ie-is+1)),
  KOKKOS_LAMBDA(const int idx, Real &n_, Real &d_, Real &pB, Real &pv) {
    int nx1 = ie - is + 1, nx2 = je - js + 1, nx3 = ke - ks + 1;
    int nji = nx2 * nx1, nkji = nx3 * nx2 * nx1;
    int m = idx / nkji;
    int k = (idx - m*nkji) / nji;
    int j = (idx - m*nkji - k*nji) / nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks; j += js;
    Real r = geom.r_vol(m, i);
    Real V = SphCellVolume(geom, m, k, j, i);
    Real dB = bcc(m, IBZ, k, j, i);  // B_phi cell-centred
    Real rho = u0(m, IDN, k, j, i);
    Real dv = u0(m, IM3, k, j, i) / rho;  // v_phi
    Real w = dB * dB;
    n_ += w * r * V;
    d_ += w * V;
    Real adB = Kokkos::fabs(dB);
    Real adv = Kokkos::fabs(dv);
    if (adB > pB) pB = adB;
    if (adv > pv) pv = adv;
  }, num, den, Kokkos::Max<Real>(peak_dB), Kokkos::Max<Real>(peak_dv));

#if MPI_PARALLEL_ENABLED
  Real n_g = num, d_g = den, pB_g = peak_dB, pv_g = peak_dv;
  MPI_Allreduce(&num, &n_g, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&den, &d_g, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&peak_dB, &pB_g, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&peak_dv, &pv_g, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
  num = n_g; den = d_g; peak_dB = pB_g; peak_dv = pv_g;
#endif
  Real rbar = (den > 0.0) ? (num / den) : 0.0;
  Real rWKB = std::pow(rc0*rc0*rc0 + 3.0 * rref * rref * vA0 * t, 1.0/3.0);
  Real wkb_amp = g_aw_amp_sqrt_va ? (rc0 / rWKB) : 1.0;

  DivBStats s = ComputeDivBStats(pmbp);
  if (global_variable::my_rank == 0) {
    std::cout.precision(6);
    std::cout << std::scientific
              << "[sph_shell_mhd/radial_alfven] t            = " << t << "\n"
              << "[sph_shell_mhd/radial_alfven] B0           = " << B0 << "\n"
              << "[sph_shell_mhd/radial_alfven] v_A0 (=B0/sqrt(rho)) = " << vA0 << "\n"
              << "[sph_shell_mhd/radial_alfven] r_c(0)       = " << rc0 << "\n"
              << "[sph_shell_mhd/radial_alfven] <r>_|dB|^2   = " << rbar << "\n"
              << "[sph_shell_mhd/radial_alfven] r_WKB(t)     = " << rWKB << "\n"
              << "[sph_shell_mhd/radial_alfven] centroid err = "
              << (rbar - rWKB) << "  (rel " << (rbar - rWKB)/(rWKB - rc0 + 1e-30) << ")\n"
              << "[sph_shell_mhd/radial_alfven] WKB amp factor = " << wkb_amp << "\n"
              << "[sph_shell_mhd/radial_alfven] peak |dB_phi|        = " << peak_dB << "\n"
              << "[sph_shell_mhd/radial_alfven] peak |dB_phi| / eps*B0 = "
              << (peak_dB / (eps * B0 + 1e-30)) << "\n"
              << "[sph_shell_mhd/radial_alfven] peak |dB_phi| / WKB expected = "
              << (peak_dB / (eps * B0 * wkb_amp + 1e-30)) << "\n"
              << "[sph_shell_mhd/radial_alfven] peak |dv_phi|        = " << peak_dv
              << std::endl;
  }
  PrintDivBStats("sph_shell_mhd/radial_alfven", s);
}

//----------------------------------------------------------------------------------------
// driven_alfven: report branch amplitudes and divB. The detailed Fourier diagnostics are
// computed in post-processing from native-bin radial slices.

void DrivenAlfvenFinalize(ParameterInput *pin, Mesh *pm) {
  if (g_mode != Mode::kDrivenAlfven) return;
  auto *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pmbp->nmb_thispack - 1;
  auto u0   = pmbp->pmhd->u0;
  auto bcc  = pmbp->pmhd->bcc0;
  auto &eos = pmbp->pmhd->peos->eos_data;
  const Real rho0 = g_rho0;
  const Real cs = eos.is_ideal ? std::sqrt(g_gam * g_p0 / rho0) : eos.iso_cs;

  Real max_v = 0.0, max_z1 = 0.0, max_z2 = 0.0;
  Kokkos::parallel_reduce("driven_aw_diag",
    Kokkos::RangePolicy<>(DevExeSpace(),
                          0, (nmb1+1)*(ke-ks+1)*(je-js+1)*(ie-is+1)),
  KOKKOS_LAMBDA(const int idx, Real &mv, Real &mz1, Real &mz2) {
    int nx1 = ie - is + 1, nx2 = je - js + 1, nx3 = ke - ks + 1;
    int nji = nx2 * nx1, nkji = nx3 * nx2 * nx1;
    int m = idx / nkji;
    int k = (idx - m*nkji) / nji;
    int j = (idx - m*nkji - k*nji) / nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks; j += js;
    Real rho = u0(m, IDN, k, j, i);
    Real vr = u0(m, IM1, k, j, i) / rho;
    Real vt = u0(m, IM2, k, j, i) / rho;
    Real vp = u0(m, IM3, k, j, i) / rho;
    Real bp = bcc(m, IBZ, k, j, i);
    Real vmag = Kokkos::sqrt(vr*vr + vt*vt + vp*vp);
    Real z1 = vp - bp / Kokkos::sqrt(rho);
    Real z2 = vp + bp / Kokkos::sqrt(rho);
    if (Kokkos::fabs(vmag) > mv) mv = Kokkos::fabs(vmag);
    if (Kokkos::fabs(z1) > mz1) mz1 = Kokkos::fabs(z1);
    if (Kokkos::fabs(z2) > mz2) mz2 = Kokkos::fabs(z2);
  }, Kokkos::Max<Real>(max_v), Kokkos::Max<Real>(max_z1),
     Kokkos::Max<Real>(max_z2));

#if MPI_PARALLEL_ENABLED
  Real max_v_g = max_v, max_z1_g = max_z1, max_z2_g = max_z2;
  MPI_Allreduce(&max_v, &max_v_g, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&max_z1, &max_z1_g, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&max_z2, &max_z2_g, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
  max_v = max_v_g; max_z1 = max_z1_g; max_z2 = max_z2_g;
#endif

  DivBStats s = ComputeDivBStats(pmbp);
  if (global_variable::my_rank == 0) {
    std::cout.precision(6);
    std::cout << std::scientific
              << "[sph_shell_mhd/driven_alfven] t                 = " << pm->time << "\n"
              << "[sph_shell_mhd/driven_alfven] driver_amp        = " << g_drv_amp << "\n"
              << "[sph_shell_mhd/driven_alfven] driver_omega      = " << g_drv_omega << "\n"
              << "[sph_shell_mhd/driven_alfven] B_phi sign        = " << g_drv_b_sign
              << "  (-1 drives z1 for B_r>0)\n"
              << "[sph_shell_mhd/driven_alfven] max|v|/cs         = " << max_v / cs << "\n"
              << "[sph_shell_mhd/driven_alfven] max|z1|           = " << max_z1 << "\n"
              << "[sph_shell_mhd/driven_alfven] max|z2|           = " << max_z2
              << std::endl;
  }
  PrintDivBStats("sph_shell_mhd/driven_alfven", s);
}

//----------------------------------------------------------------------------------------
// loop_eq: reports divB stats plus mode-specific shape/preservation metrics.
//
// axisymmetric flavor (v=0): track max |B1f|, max |B2f| -- should stay near initial.
//                            Track total magnetic energy in active cells.
// advect flavor (v_phi = Omega*r*sin theta): same metrics; the loop rotates rigidly.
//                            Total magnetic energy and field amplitudes should be
//                            approximately conserved over a small fraction of a
//                            rotation period.

void LoopEqFinalize(ParameterInput *pin, Mesh *pm) {
  if (g_mode != Mode::kLoopEqAxi && g_mode != Mode::kLoopEqAdvect) return;
  auto *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pmbp->nmb_thispack - 1;
  auto geom = pmbp->pcoord->shell_geom;
  auto &b0  = pmbp->pmhd->b0;
  auto bcc  = pmbp->pmhd->bcc0;

  Real max_b1 = 0.0, max_b2 = 0.0, max_b3 = 0.0, total_Emag = 0.0;
  Kokkos::parallel_reduce("loop_diag",
    Kokkos::RangePolicy<>(DevExeSpace(),
                          0, (nmb1+1)*(ke-ks+1)*(je-js+1)*(ie-is+1)),
  KOKKOS_LAMBDA(const int idx, Real &m1, Real &m2, Real &m3, Real &E_) {
    int nx1 = ie - is + 1, nx2 = je - js + 1, nx3 = ke - ks + 1;
    int nji = nx2 * nx1, nkji = nx3 * nx2 * nx1;
    int m = idx / nkji;
    int k = (idx - m*nkji) / nji;
    int j = (idx - m*nkji - k*nji) / nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks; j += js;
    Real V = SphCellVolume(geom, m, k, j, i);
    Real Br = bcc(m, IBX, k, j, i);
    Real Bt = bcc(m, IBY, k, j, i);
    Real Bp = bcc(m, IBZ, k, j, i);
    Real B2 = Br*Br + Bt*Bt + Bp*Bp;
    E_ += 0.5 * B2 * V;
    Real ab1 = Kokkos::fabs(Br);
    Real ab2 = Kokkos::fabs(Bt);
    Real ab3 = Kokkos::fabs(Bp);
    if (ab1 > m1) m1 = ab1;
    if (ab2 > m2) m2 = ab2;
    if (ab3 > m3) m3 = ab3;
  }, Kokkos::Max<Real>(max_b1), Kokkos::Max<Real>(max_b2),
     Kokkos::Max<Real>(max_b3), total_Emag);

#if MPI_PARALLEL_ENABLED
  Real m1g = max_b1, m2g = max_b2, m3g = max_b3, Eg = total_Emag;
  MPI_Allreduce(&max_b1, &m1g, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&max_b2, &m2g, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&max_b3, &m3g, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&total_Emag, &Eg, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  max_b1 = m1g; max_b2 = m2g; max_b3 = m3g; total_Emag = Eg;
#endif

  DivBStats s = ComputeDivBStats(pmbp);
  const char *label = (g_mode == Mode::kLoopEqAxi)
                        ? "sph_shell_mhd/loop_eq (axisymmetric)"
                        : "sph_shell_mhd/loop_eq (advect)";
  if (global_variable::my_rank == 0) {
    std::cout.precision(6);
    std::cout << std::scientific
              << "[" << label << "] t          = " << pm->time << "\n"
              << "[" << label << "] max|Br|    = " << max_b1 << "\n"
              << "[" << label << "] max|Btheta|= " << max_b2 << "\n"
              << "[" << label << "] max|Bphi|  = " << max_b3 << "\n"
              << "[" << label << "] sum 0.5 B^2 V = " << total_Emag
              << std::endl;
    if (g_mode == Mode::kLoopEqAdvect) {
      std::cout << "[" << label << "] Omega           = " << g_loop_Omega << "\n"
                << "[" << label << "] phi advected by = " << (g_loop_Omega * pm->time)
                << " rad\n";
    }
  }
  PrintDivBStats(label, s);
}

}  // namespace
