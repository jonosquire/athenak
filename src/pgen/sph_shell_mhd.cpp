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
//!   "loop_eq"          : magnetic field loop on the equatorial wedge.
//!                        flavor=axisymmetric:    A_phi loop in (r,theta), v=0 (standard)
//!                        flavor=advect:          A_theta loop in (r,phi), v_phi=Omega*r*sin(theta)
//!                                                solid-body rotation (vibe)
//!
//! Finalize prints divB diagnostics (L1, L2, Linf, normalised by |B|/V^(1/3)) plus
//! mode-specific metrics (WKB centroid speed for AW; field-amplitude preservation
//! for the loop).

#include <cmath>
#include <iostream>
#include <iomanip>
#include <limits>
#include <string>

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
                    kRadialAlfven, kLoopEqAxi, kLoopEqAdvect };
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

  // Loop-test state
  static Real g_loop_A0    = 0.0;
  static Real g_loop_rc    = 0.0;
  static Real g_loop_sigma = 0.0;
  static Real g_loop_thc   = 0.0;
  static Real g_loop_phc   = 0.0;
  static Real g_loop_Omega = 0.0;

  void UniformStaticFinalize(ParameterInput *pin, Mesh *pm);
  void MonopoleFinalize(ParameterInput *pin, Mesh *pm);
  void ToroidalStaticFinalize(ParameterInput *pin, Mesh *pm);
  void RadialAlfvenFinalize(ParameterInput *pin, Mesh *pm);
  void LoopEqFinalize(ParameterInput *pin, Mesh *pm);

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
    //   env(r)     = exp(-(r - r_c)^2 / (2 sigma_r^2))
    //   carrier(r) = cos(k_r * (r - r_c))    (=1 if k_r = 0, pure pulse)
    //
    // Alfven speed: v_A(r) = B(r) / sqrt(rho) = (B0/sqrt(rho)) * (r_ref/r)^2.
    // WKB ray: r(t) = (r_c^3 + 3 * r_ref^2 * v_A0 * t)^(1/3),  v_A0 = B0/sqrt(rho0).
    // Energy-flux conservation with v_A ~ 1/r^2, rho=const, area ~ r^2 gives
    // |dv_phi| and |dB_phi| constant along the ray (no amplitude growth/decay).
    // The pulse compresses radially as v_A drops outward (trailing edge catches up).
    g_mode = Mode::kRadialAlfven;
    pgen_final_func = RadialAlfvenFinalize;
    const Real r_c    = pin->GetOrAddReal("problem", "r_c", 2.0);
    const Real sigma_r= pin->GetOrAddReal("problem", "sigma_r", 0.3);
    const Real eps    = pin->GetOrAddReal("problem", "eps", 1.0e-3);
    const Real k_r    = pin->GetOrAddReal("problem", "k_r", 0.0);
    const Real Bref = B0;
    const Real rref2 = r_ref * r_ref;
    g_aw_rc      = r_c;
    g_aw_sigma_r = sigma_r;
    g_aw_eps     = eps;
    g_aw_kr      = k_r;
    g_aw_vA0     = Bref / std::sqrt(rho0);  // v_A at r = r_ref

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
      Real arg = (r - r_c) / sigma_r;
      Real env = Kokkos::exp(-0.5 * arg * arg);
      Real carrier = (k_r != 0.0) ? Kokkos::cos(k_r * (r - r_c)) : 1.0;
      b0.x3f(m, k, j, i) = eps * Bref * env * carrier;
    });

    // (3) Cell-centred state. delta v_phi from Alfven relation for outgoing wave on
    // B_r > 0:  delta v_phi = -delta B_phi / sqrt(rho).
    par_for("sph_mhd_aw_uc", DevExeSpace(),
            0, nmb1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      Real r = geom.r_vol(m, i);
      Real arg = (r - r_c) / sigma_r;
      Real env = Kokkos::exp(-0.5 * arg * arg);
      Real carrier = (k_r != 0.0) ? Kokkos::cos(k_r * (r - r_c)) : 1.0;
      Real dbphi_cc = eps * Bref * env * carrier;
      Real dvphi    = -dbphi_cc / Kokkos::sqrt(rho0);
      Real br_cc    = 0.5 * (b0.x1f(m, k, j, i) + b0.x1f(m, k, j, i+1));
      Real eb = 0.5 * (br_cc * br_cc + dbphi_cc * dbphi_cc);
      u0(m, IDN, k, j, i) = rho0;
      u0(m, IM1, k, j, i) = 0.0;
      u0(m, IM2, k, j, i) = 0.0;
      u0(m, IM3, k, j, i) = rho0 * dvphi;
      u0(m, IEN, k, j, i) = p0 / (gam - 1.0) + 0.5 * rho0 * dvphi * dvphi + eb;
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
              << " radial_alfven, loop_eq." << std::endl;
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
// radial_alfven: report pulse centroid in r vs WKB prediction, peak amplitude, divB.
//
// WKB ray:  r(t) = (r_c^3 + 3 r_ref^2 v_A0 t)^(1/3)
// Energy-flux conservation on a 1/r^2 radial monopole with uniform rho keeps the
// transverse amplitude approximately constant along the ray.

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
              << "[sph_shell_mhd/radial_alfven] peak |dB_phi|        = " << peak_dB << "\n"
              << "[sph_shell_mhd/radial_alfven] peak |dB_phi| / eps*B0 = "
              << (peak_dB / (eps * B0 + 1e-30)) << "\n"
              << "[sph_shell_mhd/radial_alfven] peak |dv_phi|        = " << peak_dv
              << std::endl;
  }
  PrintDivBStats("sph_shell_mhd/radial_alfven", s);
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
