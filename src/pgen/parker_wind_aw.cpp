//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file parker_wind_aw.cpp
//! \brief Polytropic Parker MHD background plus optional lower-radial-boundary
//!        monochromatic Alfven-wave forcing.
//!
//! Split-out version of the parker_polytropic mode from sph_shell_mhd.cpp,
//! extended with a transverse boundary driver suitable for monochromatic
//! Alfven-wave propagation tests on the Parker wind.
//!
//! Background (parameters via <problem>):
//!   gamma_poly, GM, rcrit, rho_inner, r_inner, alfven_point_target
//!   - Branch-safe bisection for the polytropic Parker U(r), rho(r), p(r).
//!   - Radial monopole B_r = B0 r_inner^2/r^2 calibrated to put the Alfven
//!     point at r = alfven_point_target.
//!
//! Driver (also via <problem>):
//!   driver_enable        true/false
//!   driver_amp           transverse-velocity amplitude
//!   driver_omega         angular frequency
//!   driver_phase         time phase
//!   driver_ramp_time     half-sin^2 ramp time (0 = step on)
//!   driver_polarization  "phi" (default), "theta", or "circular".
//!                        "circular" requires ntheta = nphi = 0 (parallel only).
//!                        Drives v_theta = amp ramp sin(omega t+phi),
//!                               v_phi   = circ_sign * amp ramp cos(omega t+phi),
//!                        so |v_perp| (and hence |z+|) is constant in time and
//!                        the radial envelope is smooth -- useful for WKB
//!                        envelope measurement.
//!   driver_circ_sign     +-1 (default +1) handedness of the circular drive.
//!   driver_b_sign        +-1 (default -1 = drives the outgoing z+ branch
//!                              where z+ = v_perp - sign(B_r) B_perp/sqrt(rho))
//!   driver_ntheta        integer mode in theta (default 0 -> k_perp=0)
//!   driver_nphi          integer mode in phi   (default 0 -> k_perp=0)
//!   driver_theta_phase   extra theta-direction phase
//!   driver_phi_phase     extra phi-direction phase
//!
//! Radial boundaries:
//!   Lower x1: analytic Parker + monopole for rho,p,U,B_r;
//!             transverse perturbation overlay v_perp, B_perp = b_sign * sqrt(rho) v_perp.
//!   Upper x1: analytic Parker + monopole for rho,p,U,B_r;
//!             outflow / zero-gradient copy of transverse v and B faces.
//!
//! Diagnostics:
//!   At init:  writes <csv_dir>/<label>_background_wkb.csv (analytic profile +
//!             HO_factor, z_wkb_rel, action_proxy_rel, energy_flux_proxy_rel).
//!             Writes <csv_dir>/<label>_wave_profile_t0000.csv (initial state).
//!   In run :  optional list of snapshot times (problem/snapshot_times)
//!             writes <csv_dir>/<label>_wave_profile_t<idx>.csv at each.
//!   Finalize: background preservation diagnostics + wave amplitude diagnostics,
//!             divB stats, and final-time wave profile CSV.
//!
//! Build:  -DPROBLEM=parker_wind_aw

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

// ---------- Background parameters --------------------------------------------
static Real g_gm           = 4.0;
static Real g_gamma        = 1.05;
static Real g_rcrit        = 2.0;
static Real g_ac           = 1.0;       // critical sound speed sqrt(GM/2rcrit)
static Real g_rho_inner    = 1.0;
static Real g_r_inner      = 1.0;
static Real g_rho_c        = 1.0;       // density normalization
static Real g_K_poly       = 1.0;       // p = K rho^gamma
static Real g_rA_target    = 13.0;
static Real g_B0           = 0.5;       // radial monopole strength at r_inner
static Real g_r_ref        = 2.0;       // WKB reference radius

// ---------- Driver parameters ------------------------------------------------
static bool g_drv_enable      = false;
static Real g_drv_amp         = 0.0;
static Real g_drv_omega       = 0.0;
static Real g_drv_phase       = 0.0;
static Real g_drv_ramp_time   = 0.0;
static Real g_drv_start       = 0.0;
static int  g_drv_pol         = 1;      // 1 = phi, 0 = theta, 2 = circular
static Real g_drv_b_sign      = -1.0;   // -1 = outgoing z+ branch
static Real g_drv_circ_sign   = 1.0;    // handedness for circular polarization
static int  g_drv_ntheta      = 0;
static int  g_drv_nphi        = 0;
static Real g_drv_theta_phase = 0.0;
static Real g_drv_phi_phase   = 0.0;

// ---------- Domain extents (for angular-mode normalization) ------------------
static Real g_theta_min = 0.0, g_theta_max = 1.0;
static Real g_phi_min   = 0.0, g_phi_max   = 1.0;

// ---------- Output -----------------------------------------------------------
static std::string g_csv_dir;
static std::string g_label = "default";

// ---------- Snapshot scheduler (host-only) -----------------------------------
static std::vector<Real> g_snap_times;
static int g_snap_idx = 0;

// ---------- Forward decls ----------------------------------------------------
void ParkerAwRadialBCs(Mesh *pm);
void ParkerAwFinalize(ParameterInput *pin, Mesh *pm);
void WriteWaveProfileCsv(Mesh *pm, const std::string &tag);
void WriteBackgroundWkbCsv(Mesh *pm);

// ---------- Device-callable Parker helpers -----------------------------------
// Dimensionless transonic residual F(y, x; g)=0; subsonic branch y<1 for x<1,
// supersonic y>1 for x>1.
KOKKOS_INLINE_FUNCTION
Real PolyParkerF(Real y, Real x, Real g) {
  Real inv = 1.0 / (y * x * x);
  Real pow_term = Kokkos::pow(inv, g - 1.0);
  Real C = 1.0/(g - 1.0) - 1.5;
  return 0.5*y*y + (1.0/(g - 1.0)) * pow_term - 2.0/x - C;
}

KOKKOS_INLINE_FUNCTION
Real PolyParkerY(Real r, Real rcrit, Real g) {
  Real x = r / rcrit;
  if (Kokkos::fabs(x - 1.0) < 1.0e-10) return 1.0;
  Real ylo, yhi;
  if (x < 1.0) {
    ylo = 1.0e-12;
    yhi = 1.0 - 1.0e-10;
  } else {
    ylo = 1.0 + 1.0e-10;
    yhi = 2.0;
    Real flo_e = PolyParkerF(ylo, x, g);
    for (int e = 0; e < 30; ++e) {
      Real fhi_e = PolyParkerF(yhi, x, g);
      if (flo_e * fhi_e <= 0.0) break;
      yhi *= 2.0;
      if (yhi > 1.0e6) break;
    }
  }
  Real flo = PolyParkerF(ylo, x, g);
  Real fhi = PolyParkerF(yhi, x, g);
  if (flo * fhi > 0.0) return 0.5*(ylo + yhi);
  for (int it = 0; it < 80; ++it) {
    Real ym = 0.5*(ylo + yhi);
    Real fm = PolyParkerF(ym, x, g);
    if (flo * fm <= 0.0) { yhi = ym; fhi = fm; }
    else                 { ylo = ym; flo = fm; }
  }
  return 0.5*(ylo + yhi);
}

KOKKOS_INLINE_FUNCTION
void EvaluatePolytropicParker(Real r, Real rcrit, Real ac, Real g, Real rho_c,
                              Real *Uout, Real *rhout, Real *pout) {
  Real y = PolyParkerY(r, rcrit, g);
  Real U = ac * y;
  Real rho = rho_c * ac * rcrit * rcrit / (U * r * r);
  Real p = rho_c * ac * ac / g * Kokkos::pow(rho / rho_c, g);
  *Uout = U; *rhout = rho; *pout = p;
}

KOKKOS_INLINE_FUNCTION
Real EvaluatePolytropicParkerBrFace(Real r_face, Real B0, Real r_inner) {
  return B0 * (r_inner * r_inner) / (r_face * r_face);
}

// ---------- divB diagnostics -------------------------------------------------
// Volume-averaged divB on active cells, matching the SphericalShellGeom face-area
// conventions used in CT. Identical to the helper used in sph_shell_mhd.cpp.
struct DivBStats {
  Real l1    = 0.0;
  Real l2    = 0.0;
  Real linf  = 0.0;
  Real bmax  = 0.0;
  Real h_typ = 0.0;
  int  Ncell = 0;
};

DivBStats ComputeDivBStats(MeshBlockPack *pmbp) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pmbp->nmb_thispack - 1;
  auto &b0 = pmbp->pmhd->b0;
  auto geom = pmbp->pcoord->shell_geom;
  const int Ncell = (nmb1+1)*(ke-ks+1)*(je-js+1)*(ie-is+1);

  Real l1 = 0.0, l2 = 0.0, linf = 0.0, bmax = 0.0;
  Kokkos::parallel_reduce("paw_divB",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, Ncell),
  KOKKOS_LAMBDA(const int idx, Real &s1, Real &s2,
                Real &mx, Real &bmx) {
    int nx1 = ie - is + 1, nx2 = je - js + 1, nx3 = ke - ks + 1;
    int nji = nx2 * nx1, nkji = nx3 * nx2 * nx1;
    int m = idx / nkji;
    int k = (idx - m*nkji) / nji;
    int j = (idx - m*nkji - k*nji) / nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks; j += js;
    Real A1m = geom.r2_face(m, i  ) * geom.dcos_theta(m, j) * geom.dphi(m, k);
    Real A1p = geom.r2_face(m, i+1) * geom.dcos_theta(m, j) * geom.dphi(m, k);
    Real A2m = geom.dr2_half(m, i) * geom.sin_theta_face(m, j  ) * geom.dphi(m, k);
    Real A2p = geom.dr2_half(m, i) * geom.sin_theta_face(m, j+1) * geom.dphi(m, k);
    Real A3m = geom.dr2_half(m, i) * geom.dcos_theta(m, j);
    Real A3p = A3m;
    Real V = geom.dr3_third(m, i) * geom.dcos_theta(m, j) * geom.dphi(m, k);
    Real flux = A1p * b0.x1f(m, k, j, i+1) - A1m * b0.x1f(m, k, j, i)
              + A2p * b0.x2f(m, k, j+1, i) - A2m * b0.x2f(m, k, j, i)
              + A3p * b0.x3f(m, k+1, j, i) - A3m * b0.x3f(m, k, j, i);
    Real d = flux / V;
    Real ad = Kokkos::fabs(d);
    s1 += ad;
    s2 += ad*ad;
    if (ad > mx) mx = ad;
    Real br = 0.5 * (b0.x1f(m, k, j, i) + b0.x1f(m, k, j, i+1));
    Real bt = 0.5 * (b0.x2f(m, k, j, i) + b0.x2f(m, k, j+1, i));
    Real bp = 0.5 * (b0.x3f(m, k, j, i) + b0.x3f(m, k+1, j, i));
    Real bmag = Kokkos::sqrt(br*br + bt*bt + bp*bp);
    if (bmag > bmx) bmx = bmag;
  }, l1, l2, Kokkos::Max<Real>(linf), Kokkos::Max<Real>(bmax));

  l1 /= static_cast<Real>(Ncell);
  l2 = std::sqrt(l2 / static_cast<Real>(Ncell));
  Real h_typ = (pmbp->pmesh->mesh_size.x1max - pmbp->pmesh->mesh_size.x1min) /
               static_cast<Real>(pmbp->pmesh->mesh_indcs.nx1);

  DivBStats s;
  s.l1 = l1; s.l2 = l2; s.linf = linf;
  s.bmax = bmax; s.h_typ = h_typ; s.Ncell = Ncell;
  return s;
}

void PrintDivBStats(const std::string &label, const DivBStats &s) {
  if (global_variable::my_rank != 0) return;
  Real norm = (s.bmax > 0.0) ? (s.linf * s.h_typ / s.bmax) : 0.0;
  std::cout.precision(6);
  std::cout << std::scientific
            << "[" << label << "] divB diagnostic:\n"
            << "    L1   = " << s.l1 << "\n"
            << "    Linf = " << s.linf << "\n"
            << "    |B|max = " << s.bmax << "    h_typ  = " << s.h_typ << "\n"
            << "    Linf*h/|B|max = " << norm << std::endl;
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem
//! \brief IC: polytropic Parker + monopole; register Finalize, BCs, snapshots.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  if (restart) return;
  if (pmy_mesh_->pmb_pack->pmhd == nullptr) {
    std::cout << "### FATAL ERROR: parker_wind_aw pgen requires <mhd> block"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pmy_mesh_->pmb_pack->pcoord->coord_system !=
      CoordinateSystem::spherical_shell) {
    std::cout << "### FATAL ERROR: parker_wind_aw pgen requires"
              << " <coord>/system = spherical_shell" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  auto *pmbp = pmy_mesh_->pmb_pack;
  auto &eos = pmbp->pmhd->peos->eos_data;
  if (!eos.is_ideal) {
    std::cout << "### FATAL ERROR: parker_wind_aw pgen requires <mhd>/eos=ideal"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // ---------- read background parameters --------------------------------------
  g_gm        = pin->GetReal("problem", "GM");
  g_gamma     = pin->GetOrAddReal("problem", "gamma_poly", 1.05);
  g_rcrit     = pin->GetReal("problem", "rcrit");
  g_rho_inner = pin->GetOrAddReal("problem", "rho_inner", 1.0);
  g_r_inner   = pin->GetOrAddReal("problem", "r_inner",
                                  pmy_mesh_->mesh_size.x1min);
  g_rA_target = pin->GetOrAddReal("problem", "alfven_point_target", 13.0);
  g_r_ref     = pin->GetOrAddReal("problem", "r_ref",
                                  std::max(2.0, g_r_inner + 1.0));
  if (std::fabs(g_gamma - eos.gamma) > 1.0e-12) {
    std::cout << "### FATAL ERROR: parker_wind_aw requires <problem>/gamma_poly"
              << " == <mhd>/gamma. Got " << g_gamma << " vs " << eos.gamma
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (g_gamma <= 1.0) {
    std::cout << "### FATAL ERROR: gamma_poly must be > 1, got " << g_gamma
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // ---------- read driver parameters ------------------------------------------
  g_drv_enable      = pin->GetOrAddBoolean("problem", "driver_enable", false);
  g_drv_amp         = pin->GetOrAddReal   ("problem", "driver_amp", 0.0);
  g_drv_omega       = pin->GetOrAddReal   ("problem", "driver_omega", 0.0);
  g_drv_phase       = pin->GetOrAddReal   ("problem", "driver_phase", 0.0);
  g_drv_ramp_time   = pin->GetOrAddReal   ("problem", "driver_ramp_time", 0.0);
  g_drv_start       = pin->GetOrAddReal   ("problem", "driver_start", 0.0);
  g_drv_b_sign      = pin->GetOrAddReal   ("problem", "driver_b_sign", -1.0);
  g_drv_circ_sign   = pin->GetOrAddReal   ("problem", "driver_circ_sign", 1.0);
  g_drv_ntheta      = pin->GetOrAddInteger("problem", "driver_ntheta", 0);
  g_drv_nphi        = pin->GetOrAddInteger("problem", "driver_nphi", 0);
  g_drv_theta_phase = pin->GetOrAddReal   ("problem", "driver_theta_phase", 0.0);
  g_drv_phi_phase   = pin->GetOrAddReal   ("problem", "driver_phi_phase", 0.0);
  const std::string pol = pin->GetOrAddString("problem", "driver_polarization",
                                              "phi");
  if (pol == "phi") {
    g_drv_pol = 1;
  } else if (pol == "theta") {
    g_drv_pol = 0;
  } else if (pol == "circular") {
    g_drv_pol = 2;
    if (g_drv_enable && (g_drv_ntheta != 0 || g_drv_nphi != 0)) {
      std::cout << "### FATAL ERROR: driver_polarization=circular requires "
                << "driver_ntheta = driver_nphi = 0 (parallel only). Got "
                << "ntheta=" << g_drv_ntheta << ", nphi=" << g_drv_nphi
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
  } else {
    std::cout << "### FATAL ERROR: driver_polarization must be "
              << "'phi', 'theta', or 'circular', got '" << pol << "'"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // ---------- output paths ----------------------------------------------------
  g_csv_dir = pin->GetOrAddString("problem", "csv_dir", "");
  g_label   = pin->GetOrAddString("problem", "label", "default");

  // ---------- snapshot times (comma-separated list) ---------------------------
  // problem/snapshot_times = "0.5, 1.0, 1.5, 2.5"  ; t=0 and t=final are auto.
  g_snap_times.clear();
  g_snap_idx = 0;
  {
    std::string s = pin->GetOrAddString("problem", "snapshot_times", "");
    std::string token;
    for (size_t p = 0; p <= s.size(); ++p) {
      char c = (p < s.size()) ? s[p] : ',';
      if (c == ',' || c == ';' || c == ' ') {
        if (!token.empty()) {
          try { g_snap_times.push_back(std::stod(token)); }
          catch (...) { /* skip junk */ }
          token.clear();
        }
      } else {
        token.push_back(c);
      }
    }
    std::sort(g_snap_times.begin(), g_snap_times.end());
  }

  // ---------- record domain extents (host) for angular-mode normalization -----
  g_theta_min = pmy_mesh_->mesh_size.x2min;
  g_theta_max = pmy_mesh_->mesh_size.x2max;
  g_phi_min   = pmy_mesh_->mesh_size.x3min;
  g_phi_max   = pmy_mesh_->mesh_size.x3max;

  // ---------- derived background values ---------------------------------------
  g_ac = std::sqrt(g_gm / (2.0 * g_rcrit));
  Real y_in = PolyParkerY(g_r_inner, g_rcrit, g_gamma);
  Real U_inner = g_ac * y_in;
  g_rho_c = g_rho_inner * U_inner * g_r_inner * g_r_inner
          / (g_ac * g_rcrit * g_rcrit);
  g_K_poly = g_ac * g_ac
           / (g_gamma * std::pow(g_rho_c, g_gamma - 1.0));
  Real U_rA, rho_rA, p_rA;
  EvaluatePolytropicParker(g_rA_target, g_rcrit, g_ac, g_gamma, g_rho_c,
                           &U_rA, &rho_rA, &p_rA);
  g_B0 = U_rA * std::sqrt(rho_rA)
       * (g_rA_target / g_r_inner) * (g_rA_target / g_r_inner);

  // ---------- log derived values ----------------------------------------------
  if (global_variable::my_rank == 0) {
    std::cout.precision(6);
    std::cout << std::scientific
              << "[parker_wind_aw] gamma     = " << g_gamma << "\n"
              << "[parker_wind_aw] GM        = " << g_gm << "\n"
              << "[parker_wind_aw] rcrit     = " << g_rcrit << "\n"
              << "[parker_wind_aw] a_c       = " << g_ac << "\n"
              << "[parker_wind_aw] r_inner   = " << g_r_inner << "\n"
              << "[parker_wind_aw] U_inner   = " << U_inner << "\n"
              << "[parker_wind_aw] rho_c     = " << g_rho_c << "\n"
              << "[parker_wind_aw] K_poly    = " << g_K_poly << "\n"
              << "[parker_wind_aw] rA_target = " << g_rA_target << "\n"
              << "[parker_wind_aw] U(rA)     = " << U_rA << "\n"
              << "[parker_wind_aw] rho(rA)   = " << rho_rA << "\n"
              << "[parker_wind_aw] B0        = " << g_B0 << "\n"
              << "[parker_wind_aw] r_ref     = " << g_r_ref << "\n"
              << "[parker_wind_aw] driver    = "
              << (g_drv_enable ? "ON" : "OFF") << "\n";
    if (g_drv_enable) {
      std::cout << "[parker_wind_aw] driver_amp     = " << g_drv_amp << "\n"
                << "[parker_wind_aw] driver_omega   = " << g_drv_omega << "\n"
                << "[parker_wind_aw] driver_ramp_t  = " << g_drv_ramp_time << "\n"
                << "[parker_wind_aw] driver_polariz = " << pol << "\n"
                << "[parker_wind_aw] driver_b_sign  = " << g_drv_b_sign << "\n";
      if (g_drv_pol == 2) {
        std::cout << "[parker_wind_aw] driver_circ_sign = "
                  << g_drv_circ_sign << "\n";
      }
      std::cout << "[parker_wind_aw] driver_ntheta  = " << g_drv_ntheta << "\n"
                << "[parker_wind_aw] driver_nphi    = " << g_drv_nphi << "\n";
    }
    std::cout << std::flush;
  }

  // ---------- IC: face B then cell-centred conserved state --------------------
  auto &indcs = pmy_mesh_->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pmbp->nmb_thispack - 1;
  auto &u0 = pmbp->pmhd->u0;
  auto &b0 = pmbp->pmhd->b0;
  auto geom = pmbp->pcoord->shell_geom;
  const Real ac      = g_ac;
  const Real rcrit   = g_rcrit;
  const Real gpoly   = g_gamma;
  const Real rhoc    = g_rho_c;
  const Real B0_v    = g_B0;
  const Real r_inn   = g_r_inner;

  par_for("paw_zero_bx2f", DevExeSpace(),
          0, nmb1, ks, ke, js, je+1, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    b0.x2f(m, k, j, i) = 0.0;
  });
  par_for("paw_zero_bx3f", DevExeSpace(),
          0, nmb1, ks, ke+1, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    b0.x3f(m, k, j, i) = 0.0;
  });

  par_for("paw_bx1f", DevExeSpace(),
          0, nmb1, ks, ke, js, je, is, ie+1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real rf = geom.r_face(m, i);
    b0.x1f(m, k, j, i) = EvaluatePolytropicParkerBrFace(rf, B0_v, r_inn);
  });

  par_for("paw_uc", DevExeSpace(),
          0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real r = geom.r_vol(m, i);
    Real U, rho, p;
    EvaluatePolytropicParker(r, rcrit, ac, gpoly, rhoc, &U, &rho, &p);
    Real br_cc = 0.5 * (b0.x1f(m, k, j, i) + b0.x1f(m, k, j, i+1));
    Real eb = 0.5 * br_cc * br_cc;
    u0(m, IDN, k, j, i) = rho;
    u0(m, IM1, k, j, i) = rho * U;
    u0(m, IM2, k, j, i) = 0.0;
    u0(m, IM3, k, j, i) = 0.0;
    u0(m, IEN, k, j, i) = p / (gpoly - 1.0) + 0.5 * rho * U * U + eb;
  });

  // ---------- register hooks --------------------------------------------------
  pgen_final_func = ParkerAwFinalize;
  user_bcs_func   = ParkerAwRadialBCs;

  // ---------- background+WKB CSV + initial wave profile CSV -------------------
  if (!g_csv_dir.empty()) {
    WriteBackgroundWkbCsv(pmy_mesh_);
    // Need primitives for the wave-profile dump; AthenaK normally fills bcc/w0
    // via the ConservedToPrimitive task each cycle. At t=0 those may not be
    // up to date yet, so we conservatively recompute primitives from u0/b0
    // here for the initial snapshot.
    {
      auto &w0  = pmbp->pmhd->w0;
      auto &bcc = pmbp->pmhd->bcc0;
      par_for("paw_init_w0", DevExeSpace(),
              0, nmb1, ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        Real rho = u0(m, IDN, k, j, i);
        Real vr  = u0(m, IM1, k, j, i) / rho;
        Real vt  = u0(m, IM2, k, j, i) / rho;
        Real vp  = u0(m, IM3, k, j, i) / rho;
        Real br_cc = 0.5 * (b0.x1f(m, k, j, i) + b0.x1f(m, k, j, i+1));
        Real bt_cc = 0.5 * (b0.x2f(m, k, j, i) + b0.x2f(m, k, j+1, i));
        Real bp_cc = 0.5 * (b0.x3f(m, k, j, i) + b0.x3f(m, k+1, j, i));
        Real ek = 0.5 * rho * (vr*vr + vt*vt + vp*vp);
        Real eb = 0.5 * (br_cc*br_cc + bt_cc*bt_cc + bp_cc*bp_cc);
        Real eint = u0(m, IEN, k, j, i) - ek - eb;
        w0(m, IDN, k, j, i) = rho;
        w0(m, IVX, k, j, i) = vr;
        w0(m, IVY, k, j, i) = vt;
        w0(m, IVZ, k, j, i) = vp;
        w0(m, IEN, k, j, i) = eint;
        bcc(m, IBX, k, j, i) = br_cc;
        bcc(m, IBY, k, j, i) = bt_cc;
        bcc(m, IBZ, k, j, i) = bp_cc;
      });
    }
    WriteWaveProfileCsv(pmy_mesh_, "t0000");
  }
}

namespace {

//----------------------------------------------------------------------------------------
// Radial user BC: lower-x1 imposes analytic Parker + optional AW driver,
// upper-x1 imposes analytic Parker for radial/background quantities and
// zero-gradient outflow for transverse v/B (so outgoing waves leave with at
// most weak reflection from the simple BC).

void ParkerAwRadialBCs(Mesh *pm) {
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

  const Real ac     = g_ac;
  const Real rcrit  = g_rcrit;
  const Real gpoly  = g_gamma;
  const Real rhoc   = g_rho_c;
  const Real B0_v   = g_B0;
  const Real r_inn  = g_r_inner;

  // Driver state (host-side ramped amplitude). 0 unless enabled.
  // vdrv0   = amp * ramp * sin(arg)  -> theta component (or phi for "phi" pol)
  // vdrv_c0 = circ_sign * amp * ramp * cos(arg)  -> phi component, circular only.
  Real vdrv0   = 0.0;
  Real vdrv_c0 = 0.0;
  if (g_drv_enable) {
    const Real tdrive = pm->time - g_drv_start;
    Real ramp = 0.0;
    if (tdrive > 0.0) {
      if (g_drv_ramp_time > 0.0 && tdrive < g_drv_ramp_time) {
        const Real arg = 1.57079632679489661923 * tdrive / g_drv_ramp_time;
        ramp = std::sin(arg); ramp *= ramp;
      } else {
        ramp = 1.0;
      }
    }
    const Real arg_t = g_drv_omega * tdrive + g_drv_phase;
    vdrv0   = g_drv_amp * ramp * std::sin(arg_t);
    vdrv_c0 = g_drv_amp * ramp * std::cos(arg_t) * g_drv_circ_sign;
  }
  const Real bdrv0   = g_drv_b_sign * std::sqrt(g_rho_inner) * vdrv0;
  const Real bdrv_c0 = g_drv_b_sign * std::sqrt(g_rho_inner) * vdrv_c0;
  const int  pol     = g_drv_pol;
  const int  ntheta  = g_drv_ntheta;
  const int  nphi    = g_drv_nphi;
  const Real th_min  = g_theta_min;
  const Real th_max  = g_theta_max;
  const Real ph_min  = g_phi_min;
  const Real ph_max  = g_phi_max;
  const Real th_ph   = g_drv_theta_phase;
  const Real ph_ph   = g_drv_phi_phase;
  const Real two_pi  = 6.28318530717958647692;
  const Real inv_dth = (th_max > th_min) ? 1.0/(th_max - th_min) : 0.0;
  const Real inv_dph = (ph_max > ph_min) ? 1.0/(ph_max - ph_min) : 0.0;

  // ---- Face B in radial ghost zones -----------------------------------------
  par_for("paw_bc_b", DevExeSpace(),
          0, nmb1, 0, n3-1, 0, n2-1, 0, ng-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int gh) {
    if (mb_bcs.d_view(m, BoundaryFace::inner_x1) == BoundaryFlag::user) {
      const int i = is - gh - 1;
      const Real rf = geom.r_face(m, i);
      b0.x1f(m, k, j, i) = EvaluatePolytropicParkerBrFace(rf, B0_v, r_inn);
      // angular factor at this (j,k) (uses theta_vol and phi_vol-centre)
      Real theta = geom.theta_vol(m, j);
      Real phi   = 0.5 * (geom.phi_face(m, k) + geom.phi_face(m, k+1));
      Real ang = Kokkos::cos(two_pi*ntheta*(theta - th_min)*inv_dth + th_ph
                           + two_pi*nphi  *(phi   - ph_min)*inv_dph + ph_ph);
      Real bdrv   = bdrv0 * ang;
      Real bdrv_c = bdrv_c0 * ang;   // = bdrv_c0 (ang=1 when ntheta=nphi=0)
      if (pol == 2) {           // circular: drive both B_theta (sin) and B_phi (cos)
        b0.x2f(m, k, j, i) = bdrv;
        if (j == n2-1) b0.x2f(m, k, j+1, i) = bdrv;
        b0.x3f(m, k, j, i) = bdrv_c;
        if (k == n3-1) b0.x3f(m, k+1, j, i) = bdrv_c;
      } else if (pol == 1) {    // phi polarization: drive B_phi only
        b0.x2f(m, k, j, i) = 0.0;
        if (j == n2-1) b0.x2f(m, k, j+1, i) = 0.0;
        b0.x3f(m, k, j, i) = bdrv;
        if (k == n3-1) b0.x3f(m, k+1, j, i) = bdrv;
      } else {                  // theta polarization: drive B_theta only
        b0.x2f(m, k, j, i) = bdrv;
        if (j == n2-1) b0.x2f(m, k, j+1, i) = bdrv;
        b0.x3f(m, k, j, i) = 0.0;
        if (k == n3-1) b0.x3f(m, k+1, j, i) = 0.0;
      }
    }
    if (mb_bcs.d_view(m, BoundaryFace::outer_x1) == BoundaryFlag::user) {
      const int i_face = ie + gh + 2;
      const int i_cell = ie + gh + 1;
      const Real rf = geom.r_face(m, i_face);
      b0.x1f(m, k, j, i_face) = EvaluatePolytropicParkerBrFace(rf, B0_v, r_inn);
      // outflow / zero-gradient transverse face B
      b0.x2f(m, k, j,   i_cell) = b0.x2f(m, k, j,   ie);
      if (j == n2-1)
        b0.x2f(m, k, j+1, i_cell) = b0.x2f(m, k, j+1, ie);
      b0.x3f(m, k,   j, i_cell) = b0.x3f(m, k,   j, ie);
      if (k == n3-1)
        b0.x3f(m, k+1, j, i_cell) = b0.x3f(m, k+1, j, ie);
    }
  });

  // ---- Cell-centred u in radial ghost zones ---------------------------------
  par_for("paw_bc_u", DevExeSpace(),
          0, nmb1, 0, n3-1, 0, n2-1, 0, ng-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int gh) {
    if (mb_bcs.d_view(m, BoundaryFace::inner_x1) == BoundaryFlag::user) {
      const int i = is - gh - 1;
      const Real r = geom.r_vol(m, i);
      Real U, rho, p;
      EvaluatePolytropicParker(r, rcrit, ac, gpoly, rhoc, &U, &rho, &p);
      Real theta = geom.theta_vol(m, j);
      Real phi   = 0.5 * (geom.phi_face(m, k) + geom.phi_face(m, k+1));
      Real ang = Kokkos::cos(two_pi*ntheta*(theta - th_min)*inv_dth + th_ph
                           + two_pi*nphi  *(phi   - ph_min)*inv_dph + ph_ph);
      Real vdrv   = vdrv0   * ang;
      Real bdrv   = bdrv0   * ang;
      Real vdrv_c = vdrv_c0 * ang;   // ang=1 for ntheta=nphi=0
      Real bdrv_c = bdrv_c0 * ang;
      Real vth = 0.0, vph = 0.0, bth = 0.0, bph = 0.0;
      if (pol == 2)      { vth = vdrv;   vph = vdrv_c;
                           bth = bdrv;   bph = bdrv_c; }
      else if (pol == 1) { vph = vdrv;   bph = bdrv; }
      else               { vth = vdrv;   bth = bdrv; }
      Real br_cc = 0.5 * (b0.x1f(m, k, j, i) + b0.x1f(m, k, j, i+1));
      Real eb = 0.5 * (br_cc*br_cc + bth*bth + bph*bph);
      u0(m, IDN, k, j, i) = rho;
      u0(m, IM1, k, j, i) = rho * U;
      u0(m, IM2, k, j, i) = rho * vth;
      u0(m, IM3, k, j, i) = rho * vph;
      u0(m, IEN, k, j, i) = p / (gpoly - 1.0)
                          + 0.5 * rho * (U*U + vth*vth + vph*vph)
                          + eb;
    }
    if (mb_bcs.d_view(m, BoundaryFace::outer_x1) == BoundaryFlag::user) {
      const int i = ie + gh + 1;
      const Real r = geom.r_vol(m, i);
      Real U, rho, p;
      EvaluatePolytropicParker(r, rcrit, ac, gpoly, rhoc, &U, &rho, &p);
      // Outflow on transverse velocity: copy from last active cell.
      Real rho_e = u0(m, IDN, k, j, ie);
      Real vth_e = u0(m, IM2, k, j, ie) / rho_e;
      Real vph_e = u0(m, IM3, k, j, ie) / rho_e;
      Real br_cc = 0.5 * (b0.x1f(m, k, j, i) + b0.x1f(m, k, j, i+1));
      Real bth_cc = 0.5 * (b0.x2f(m, k, j, i) + b0.x2f(m, k, j+1, i));
      Real bph_cc = 0.5 * (b0.x3f(m, k, j, i) + b0.x3f(m, k+1, j, i));
      Real eb = 0.5 * (br_cc*br_cc + bth_cc*bth_cc + bph_cc*bph_cc);
      u0(m, IDN, k, j, i) = rho;
      u0(m, IM1, k, j, i) = rho * U;
      u0(m, IM2, k, j, i) = rho * vth_e;
      u0(m, IM3, k, j, i) = rho * vph_e;
      u0(m, IEN, k, j, i) = p / (gpoly - 1.0)
                          + 0.5 * rho * (U*U + vth_e*vth_e + vph_e*vph_e)
                          + eb;
    }
  });

  // ---- Host-side snapshot dispatcher ----------------------------------------
  // user_bcs_func may be called multiple times per cycle (RK substages) but
  // pm->time only advances between cycles, so gating by g_snap_idx is safe.
  if (!g_csv_dir.empty() && g_snap_idx < static_cast<int>(g_snap_times.size())) {
    while (g_snap_idx < static_cast<int>(g_snap_times.size())
           && pm->time >= g_snap_times[g_snap_idx]) {
      char buf[32];
      std::snprintf(buf, sizeof(buf), "t%04d", g_snap_idx + 1);
      WriteWaveProfileCsv(pm, buf);
      ++g_snap_idx;
    }
  }
}

//----------------------------------------------------------------------------------------
// Host-side: write the analytic background + WKB scaling reference profile on a
// dense uniform radial grid covering the mesh domain. Columns documented in the
// CSV header line.

void WriteBackgroundWkbCsv(Mesh *pm) {
  if (g_csv_dir.empty() || global_variable::my_rank != 0) return;
  const std::string path = g_csv_dir + "/" + g_label + "_background_wkb.csv";
  std::ofstream out(path);
  if (!out) {
    std::cout << "WARNING: parker_wind_aw cannot open " << path << std::endl;
    return;
  }
  out.precision(10);
  out << std::scientific;
  out << "r,rho,p,U,a,Br,vA,MA,UplusvA,tau_plus,"
      << "HO_factor,z_wkb_rel,du_wkb_rel,dB_over_sqrtrho_wkb_rel,dB_wkb_rel,"
      << "area,energy_flux_proxy_rel,action_proxy_rel\n";

  const int N = 4096;
  const Real r_lo = pm->mesh_size.x1min;
  const Real r_hi = pm->mesh_size.x1max;
  std::vector<Real> r(N), rho(N), p(N), U(N), a_s(N), Br(N), vA(N), MA(N),
                    Upv(N), tau(N), HO(N), rho_ref_v(N);
  for (int i = 0; i < N; ++i) {
    Real ri = r_lo + (r_hi - r_lo) * (static_cast<Real>(i)
                                      / static_cast<Real>(N - 1));
    r[i] = ri;
    Real Ui, rhoi, pi_;
    EvaluatePolytropicParker(ri, g_rcrit, g_ac, g_gamma, g_rho_c,
                             &Ui, &rhoi, &pi_);
    U[i]   = Ui;
    rho[i] = rhoi;
    p[i]   = pi_;
    a_s[i] = std::sqrt(g_gamma * pi_ / rhoi);
    Br[i]  = g_B0 * (g_r_inner * g_r_inner) / (ri * ri);
    vA[i]  = std::fabs(Br[i]) / std::sqrt(rhoi);
    MA[i]  = Ui / vA[i];
    Upv[i] = Ui + vA[i];
    HO[i]  = std::sqrt(MA[i]) + 1.0 / std::sqrt(MA[i]);
  }
  // tau_plus(r) = integral_{r_inner}^r dr' / (U+vA)(r')
  // Trapezoidal rule on the dense grid; clamp r<r_inner to 0.
  tau[0] = 0.0;
  for (int i = 1; i < N; ++i) {
    Real dr = r[i] - r[i-1];
    tau[i] = tau[i-1] + 0.5 * dr * (1.0/Upv[i-1] + 1.0/Upv[i]);
  }
  // Reference state at r_ref
  Real HO_ref, rho_ref, Br_ref, vA_ref, U_ref;
  {
    Real Ui, rhoi, pi_;
    EvaluatePolytropicParker(g_r_ref, g_rcrit, g_ac, g_gamma, g_rho_c,
                             &Ui, &rhoi, &pi_);
    U_ref   = Ui;
    rho_ref = rhoi;
    Br_ref  = g_B0 * (g_r_inner * g_r_inner) / (g_r_ref * g_r_ref);
    vA_ref  = std::fabs(Br_ref) / std::sqrt(rhoi);
    Real MA_ref = U_ref / vA_ref;
    HO_ref  = std::sqrt(MA_ref) + 1.0/std::sqrt(MA_ref);
  }
  const Real area_ref = g_r_ref * g_r_ref;
  const Real flux_ref = area_ref * rho_ref * (U_ref + vA_ref);
  const Real act_ref  = area_ref * rho_ref;
  for (int i = 0; i < N; ++i) {
    Real z_rel  = HO_ref / HO[i];
    Real du_rel = z_rel;
    Real dB_over_sqrtrho_rel = z_rel;
    Real dB_rel = std::sqrt(rho[i] / rho_ref) * z_rel;
    Real area   = r[i] * r[i];
    Real flux_rel = (area * rho[i] * (U[i] + vA[i]) * z_rel * z_rel) / flux_ref;
    Real act_rel  = (area * rho[i] * z_rel * z_rel) / act_ref;
    out << r[i] << "," << rho[i] << "," << p[i] << "," << U[i] << ","
        << a_s[i] << "," << Br[i] << "," << vA[i] << "," << MA[i] << ","
        << Upv[i] << "," << tau[i] << "," << HO[i] << "," << z_rel << ","
        << du_rel << "," << dB_over_sqrtrho_rel << "," << dB_rel << ","
        << area << "," << flux_rel << "," << act_rel << "\n";
  }
  out.close();
  std::cout << "[parker_wind_aw] wrote " << path << std::endl;
}

//----------------------------------------------------------------------------------------
// Host-side: write a radial profile of the current simulation state taken on
// the central (j,k) cell of the angular wedge, plus angular RMS scatter at each
// r. Outputs primitives, Bcc components, transverse Elsasser amplitudes, and
// the WKB expectation evaluated locally.

void WriteWaveProfileCsv(Mesh *pm, const std::string &tag) {
  if (g_csv_dir.empty() || global_variable::my_rank != 0) return;
  auto *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pmbp->nmb_thispack - 1;
  auto geom = pmbp->pcoord->shell_geom;

  const int j_mid = js + (je - js) / 2;
  const int k_mid = ks + (ke - ks) / 2;

  auto h_w0  = Kokkos::create_mirror_view(pmbp->pmhd->w0);
  auto h_bcc = Kokkos::create_mirror_view(pmbp->pmhd->bcc0);
  auto h_rv  = Kokkos::create_mirror_view(geom.r_vol);
  Kokkos::deep_copy(h_w0,  pmbp->pmhd->w0);
  Kokkos::deep_copy(h_bcc, pmbp->pmhd->bcc0);
  Kokkos::deep_copy(h_rv,  geom.r_vol);

  // WKB reference at r_ref using analytic background (host).
  Real U_ref, rho_ref, p_ref;
  EvaluatePolytropicParker(g_r_ref, g_rcrit, g_ac, g_gamma, g_rho_c,
                           &U_ref, &rho_ref, &p_ref);
  Real Br_ref = g_B0 * (g_r_inner*g_r_inner) / (g_r_ref*g_r_ref);
  Real vA_ref = std::fabs(Br_ref) / std::sqrt(rho_ref);
  Real MA_ref = U_ref / vA_ref;
  Real HO_ref = std::sqrt(MA_ref) + 1.0/std::sqrt(MA_ref);

  const std::string path =
      g_csv_dir + "/" + g_label + "_wave_profile_" + tag + ".csv";
  std::ofstream out(path);
  if (!out) {
    std::cout << "WARNING: parker_wind_aw cannot open " << path << std::endl;
    return;
  }
  out.precision(10);
  out << std::scientific;
  out << "time,r,tau_plus,rho_an,p_an,U_an,Br_an,vA_an,MA_an,HO_factor,"
      << "z_wkb_rel,rho,p,vr,vtheta,vphi,vperp,"
      << "Br,Btheta,Bphi,Bperp,Bperp_over_sqrtrho,"
      << "z_plus_theta,z_plus_phi,z_minus_theta,z_minus_phi,"
      << "z_plus_amp,z_minus_amp,"
      << "std_vperp,std_Bperp\n";

  // Build sorted-by-r list of (m,i) cells so the CSV is monotonic in r.
  std::vector<std::pair<Real,std::pair<int,int>>> cells;
  cells.reserve(static_cast<std::size_t>(nmb1+1) * (ie-is+1));
  for (int m = 0; m <= nmb1; ++m)
    for (int i = is; i <= ie; ++i)
      cells.push_back({h_rv(m,i), {m,i}});
  std::sort(cells.begin(), cells.end(),
            [](const auto &a, const auto &b){ return a.first < b.first; });

  // tau_plus: integrate analytic 1/(U+vA) from r_inner along the sorted radii.
  std::vector<Real> tau(cells.size(), 0.0);
  Real prev_r   = g_r_inner;
  Real prev_inv = 0.0;
  {
    Real U0, rho0, p0;
    EvaluatePolytropicParker(g_r_inner, g_rcrit, g_ac, g_gamma, g_rho_c,
                             &U0, &rho0, &p0);
    Real Br0 = g_B0;
    Real vA0 = std::fabs(Br0)/std::sqrt(rho0);
    prev_inv = 1.0 / (U0 + vA0);
  }
  Real tau_acc = 0.0;
  for (std::size_t c = 0; c < cells.size(); ++c) {
    Real r = cells[c].first;
    Real Uc, rhoc_a, pc;
    EvaluatePolytropicParker(r, g_rcrit, g_ac, g_gamma, g_rho_c,
                             &Uc, &rhoc_a, &pc);
    Real Brc = g_B0 * (g_r_inner*g_r_inner) / (r*r);
    Real vAc = std::fabs(Brc)/std::sqrt(rhoc_a);
    Real inv = 1.0 / (Uc + vAc);
    tau_acc += 0.5 * (r - prev_r) * (prev_inv + inv);
    tau[c] = tau_acc;
    prev_r = r; prev_inv = inv;
  }

  // For each r, also compute angular RMS scatter of v_perp, B_perp.
  // (For k_perp=0 this should be 0; nonzero ntheta/nphi gives nonzero RMS.)
  for (std::size_t c = 0; c < cells.size(); ++c) {
    int m = cells[c].second.first;
    int i = cells[c].second.second;
    Real r = cells[c].first;

    Real U_a, rho_a, p_a;
    EvaluatePolytropicParker(r, g_rcrit, g_ac, g_gamma, g_rho_c,
                             &U_a, &rho_a, &p_a);
    Real Br_a = g_B0 * (g_r_inner*g_r_inner) / (r*r);
    Real vA_a = std::fabs(Br_a) / std::sqrt(rho_a);
    Real MA_a = U_a / vA_a;
    Real HO_a = std::sqrt(MA_a) + 1.0/std::sqrt(MA_a);
    Real z_rel = HO_ref / HO_a;

    // Mid-cell sample
    Real rho = h_w0(m, IDN, k_mid, j_mid, i);
    Real vr  = h_w0(m, IVX, k_mid, j_mid, i);
    Real vt  = h_w0(m, IVY, k_mid, j_mid, i);
    Real vp  = h_w0(m, IVZ, k_mid, j_mid, i);
    Real p_num = (g_gamma - 1.0) * h_w0(m, IEN, k_mid, j_mid, i);
    Real Br = h_bcc(m, IBX, k_mid, j_mid, i);
    Real Bt = h_bcc(m, IBY, k_mid, j_mid, i);
    Real Bp = h_bcc(m, IBZ, k_mid, j_mid, i);
    Real vperp = std::sqrt(vt*vt + vp*vp);
    Real Bperp = std::sqrt(Bt*Bt + Bp*Bp);
    Real Bperp_norm = Bperp / std::sqrt(rho_a);
    Real sgn_Br = (Br >= 0.0) ? 1.0 : -1.0;
    // z+ = v_perp - sign(Br)*B_perp/sqrt(rho)   (outgoing for Br>0)
    // z- = v_perp + sign(Br)*B_perp/sqrt(rho)   (incoming for Br>0)
    Real zp_t = vt - sgn_Br * Bt / std::sqrt(rho_a);
    Real zp_p = vp - sgn_Br * Bp / std::sqrt(rho_a);
    Real zm_t = vt + sgn_Br * Bt / std::sqrt(rho_a);
    Real zm_p = vp + sgn_Br * Bp / std::sqrt(rho_a);
    Real zp_a = std::sqrt(zp_t*zp_t + zp_p*zp_p);
    Real zm_a = std::sqrt(zm_t*zm_t + zm_p*zm_p);

    // Angular RMS scatter of v_perp and B_perp at this radial index.
    Real sum_vp2 = 0.0, sum_bp2 = 0.0;
    int nang = 0;
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
        Real vti = h_w0(m, IVY, k, j, i);
        Real vpi = h_w0(m, IVZ, k, j, i);
        Real bti = h_bcc(m, IBY, k, j, i);
        Real bpi = h_bcc(m, IBZ, k, j, i);
        sum_vp2 += vti*vti + vpi*vpi;
        sum_bp2 += bti*bti + bpi*bpi;
        ++nang;
      }
    }
    Real mean_vp2 = (nang > 0) ? sum_vp2 / nang : 0.0;
    Real mean_bp2 = (nang > 0) ? sum_bp2 / nang : 0.0;
    Real std_vp = std::sqrt(std::max(0.0, mean_vp2 - vperp*vperp));
    Real std_bp = std::sqrt(std::max(0.0, mean_bp2 - Bperp*Bperp));

    out << pm->time << "," << r << "," << tau[c] << ","
        << rho_a << "," << p_a << "," << U_a << ","
        << Br_a  << "," << vA_a << "," << MA_a << "," << HO_a << ","
        << z_rel << ","
        << rho   << "," << p_num << "," << vr  << ","
        << vt    << "," << vp    << "," << vperp << ","
        << Br    << "," << Bt    << "," << Bp    << ","
        << Bperp << "," << Bperp_norm << ","
        << zp_t  << "," << zp_p  << "," << zm_t  << "," << zm_p << ","
        << zp_a  << "," << zm_a  << ","
        << std_vp << "," << std_bp << "\n";
  }
  out.close();
  std::cout << "[parker_wind_aw] wrote " << path << std::endl;
}

//----------------------------------------------------------------------------------------
// End-of-run diagnostics: background preservation + transverse wave summaries
// + divB stats + final-time wave profile CSV.

void ParkerAwFinalize(ParameterInput *pin, Mesh *pm) {
  auto *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pmbp->nmb_thispack - 1;
  auto geom = pmbp->pcoord->shell_geom;
  auto u0 = pmbp->pmhd->u0;
  auto bcc = pmbp->pmhd->bcc0;
  const Real gpoly = g_gamma;
  const Real ac    = g_ac;
  const Real rcrit = g_rcrit;
  const Real rhoc  = g_rho_c;
  const Real B0_v  = g_B0;
  const Real r_inn = g_r_inner;
  const Real rho_in_an = g_rho_inner;

  // reference values at r_inner
  Real U_in, rho_in, p_in;
  {
    Real y_in = PolyParkerY(r_inn, rcrit, gpoly);
    U_in   = ac * y_in;
    rho_in = rho_in_an;
    p_in   = rhoc * ac*ac / gpoly * std::pow(rho_in / rhoc, gpoly);
  }
  const Real mdot_ref = rho_in * U_in * r_inn * r_inn;
  const Real cs_in = std::sqrt(gpoly * p_in / rho_in);
  const Real vA_in = std::fabs(B0_v) / std::sqrt(rho_in);
  const int Ncell = (nmb1+1)*(ke-ks+1)*(je-js+1)*(ie-is+1);

  Real l1_v = 0.0, l1_rho = 0.0, l1_p = 0.0, l1_m = 0.0;
  Real linf_v = 0.0, linf_rho = 0.0, linf_p = 0.0, linf_m = 0.0;
  Real max_vt = 0.0, max_vp = 0.0, max_bt = 0.0, max_bp = 0.0;
  Real max_vperp = 0.0;
  Real max_zplus = 0.0, max_zminus = 0.0;

  Kokkos::parallel_reduce("paw_diag",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, Ncell),
  KOKKOS_LAMBDA(const int idx,
                Real &sv, Real &sr, Real &sp, Real &sm,
                Real &mxv, Real &mxr, Real &mxp, Real &mxm,
                Real &mvt, Real &mvp, Real &mbt, Real &mbp,
                Real &mvperp, Real &mzp, Real &mzm) {
    int nx1 = ie - is + 1, nx2 = je - js + 1, nx3 = ke - ks + 1;
    int nji = nx2 * nx1, nkji = nx3 * nx2 * nx1;
    int m = idx / nkji;
    int k = (idx - m*nkji) / nji;
    int j = (idx - m*nkji - k*nji) / nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks; j += js;
    Real r = geom.r_vol(m, i);
    Real U_a, rho_a, p_a;
    EvaluatePolytropicParker(r, rcrit, ac, gpoly, rhoc, &U_a, &rho_a, &p_a);
    Real rho = u0(m, IDN, k, j, i);
    Real vr = u0(m, IM1, k, j, i) / rho;
    Real vt = u0(m, IM2, k, j, i) / rho;
    Real vp = u0(m, IM3, k, j, i) / rho;
    Real br = bcc(m, IBX, k, j, i);
    Real bt = bcc(m, IBY, k, j, i);
    Real bp = bcc(m, IBZ, k, j, i);
    Real ek = 0.5 * rho * (vr*vr + vt*vt + vp*vp);
    Real eb = 0.5 * (br*br + bt*bt + bp*bp);
    Real p  = (gpoly - 1.0) * (u0(m, IEN, k, j, i) - ek - eb);
    Real mdot = rho * vr * r * r;
    Real ev = Kokkos::fabs((vr - U_a)/U_a);
    Real er = Kokkos::fabs((rho - rho_a)/rho_a);
    Real ep = Kokkos::fabs((p - p_a)/p_a);
    Real em = Kokkos::fabs((mdot - mdot_ref)/mdot_ref);
    sv += ev; sr += er; sp += ep; sm += em;
    if (ev > mxv) mxv = ev;
    if (er > mxr) mxr = er;
    if (ep > mxp) mxp = ep;
    if (em > mxm) mxm = em;
    Real avt = Kokkos::fabs(vt), avp = Kokkos::fabs(vp);
    Real abt = Kokkos::fabs(bt), abp = Kokkos::fabs(bp);
    if (avt > mvt) mvt = avt;
    if (avp > mvp) mvp = avp;
    if (abt > mbt) mbt = abt;
    if (abp > mbp) mbp = abp;
    Real vperp = Kokkos::sqrt(vt*vt + vp*vp);
    if (vperp > mvperp) mvperp = vperp;
    Real sgn = (br >= 0.0) ? 1.0 : -1.0;
    Real zpt = vt - sgn*bt/Kokkos::sqrt(rho_a);
    Real zpp = vp - sgn*bp/Kokkos::sqrt(rho_a);
    Real zmt = vt + sgn*bt/Kokkos::sqrt(rho_a);
    Real zmp = vp + sgn*bp/Kokkos::sqrt(rho_a);
    Real zpa = Kokkos::sqrt(zpt*zpt + zpp*zpp);
    Real zma = Kokkos::sqrt(zmt*zmt + zmp*zmp);
    if (zpa > mzp) mzp = zpa;
    if (zma > mzm) mzm = zma;
  }, l1_v, l1_rho, l1_p, l1_m,
     Kokkos::Max<Real>(linf_v),  Kokkos::Max<Real>(linf_rho),
     Kokkos::Max<Real>(linf_p),  Kokkos::Max<Real>(linf_m),
     Kokkos::Max<Real>(max_vt),  Kokkos::Max<Real>(max_vp),
     Kokkos::Max<Real>(max_bt),  Kokkos::Max<Real>(max_bp),
     Kokkos::Max<Real>(max_vperp),
     Kokkos::Max<Real>(max_zplus),
     Kokkos::Max<Real>(max_zminus));

  l1_v   /= static_cast<Real>(Ncell);
  l1_rho /= static_cast<Real>(Ncell);
  l1_p   /= static_cast<Real>(Ncell);
  l1_m   /= static_cast<Real>(Ncell);

  DivBStats s = ComputeDivBStats(pmbp);

  if (global_variable::my_rank == 0) {
    std::cout.precision(6);
    std::cout << std::scientific
              << "[parker_wind_aw] t              = " << pm->time << "\n"
              << "[parker_wind_aw] cs(r_inner)    = " << cs_in << "\n"
              << "[parker_wind_aw] vA(r_inner)    = " << vA_in << "\n"
              << "[parker_wind_aw] mdot_ref       = " << mdot_ref << "\n"
              << "[parker_wind_aw] L1 |dv_r/v_r|  = " << l1_v << "\n"
              << "[parker_wind_aw] Linf|dv_r/v_r| = " << linf_v << "\n"
              << "[parker_wind_aw] L1 |drho/rho|  = " << l1_rho << "\n"
              << "[parker_wind_aw] Linf|drho/rho| = " << linf_rho << "\n"
              << "[parker_wind_aw] L1 |dp/p|      = " << l1_p << "\n"
              << "[parker_wind_aw] Linf|dp/p|     = " << linf_p << "\n"
              << "[parker_wind_aw] L1 |dmdot|     = " << l1_m << "\n"
              << "[parker_wind_aw] Linf|dmdot|    = " << linf_m << "\n"
              << "[parker_wind_aw] max|v_theta|/cs= " << (max_vt/cs_in) << "\n"
              << "[parker_wind_aw] max|v_phi|/cs  = " << (max_vp/cs_in) << "\n"
              << "[parker_wind_aw] max|v_perp|/cs = " << (max_vperp/cs_in) << "\n"
              << "[parker_wind_aw] max|B_theta|/sqrt(rho_inner) = "
              << (max_bt/std::sqrt(rho_in_an)) << "\n"
              << "[parker_wind_aw] max|B_phi|/sqrt(rho_inner)   = "
              << (max_bp/std::sqrt(rho_in_an)) << "\n"
              << "[parker_wind_aw] max|z+|        = " << max_zplus << "\n"
              << "[parker_wind_aw] max|z-|        = " << max_zminus << "\n"
              << "[parker_wind_aw] max|z-|/max|z+|= "
              << ((max_zplus > 0.0) ? max_zminus/max_zplus : 0.0)
              << std::endl;
  }
  PrintDivBStats("parker_wind_aw", s);

  // Final-time wave profile CSV
  if (!g_csv_dir.empty()) WriteWaveProfileCsv(pm, "final");
}

}  // namespace
