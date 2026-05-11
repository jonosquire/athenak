//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file sph_shell_hydro.cpp
//! \brief User problem generator for hydro tests on a spherical shell.
//!
//! Modes (selected via <problem>/mode):
//!   "uniform"               : constant rho, p, v=0 -- uniform-state preservation.
//!   "sound_pulse"           : original radial Gaussian sound pulse (Task 2).
//!   "radial_acoustic"       : radial acoustic test with centroid-speed reporter.
//!   "divergence_test"       : FV divergence operator vs analytic for 3 fields.
//!   "pulse_3d"              : 3D Gaussian pressure pulse (visual / vibe).
//!   "oblique_packet"        : oblique Cartesian acoustic wave packet (visual).
//!   "homologous"            : v_r=H*r ; check d<rho>/dt vs -3H.
//!   "hydrostatic_constg"    : isothermal HSE in constant radial g (Task 3B).
//!   "hydrostatic_r2"        : isothermal HSE in 1/r^2 gravity (Task 3B).
//!   "solid_body_rotation"   : one-step centrifugal-source diagnostic (Task 3B).
//!   "keplerian_orbit"       : near-equatorial circular orbit in 1/r^2 gravity.
//!   "thin_disk"             : pressure-supported thin disk in 1/r^2 gravity.
//!   "parker_isothermal"     : analytic isothermal Parker wind initialiser.
//!
//! This generator is intentionally narrow and only used for hydro-only
//! spherical_shell tests in this fork.

#include <cmath>
#include <iostream>
#include <iomanip>
#include <limits>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/cell_locations.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/spherical_shell.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "pgen/pgen.hpp"

namespace {
  // ---- Shared diagnostic state for end-of-run reporters --------------------
  enum class Mode { kNone, kUniform, kSoundPulse, kRadialAcoustic,
                    kDivergenceTest, kPulse3D, kObliquePacket, kHomologous,
                    kHydrostaticConstG, kHydrostaticR2,
                    kSolidBodyRotation, kKeplerianOrbit, kThinDisk,
                    kParkerIsothermal };
  static Mode g_mode = Mode::kNone;

  // Background state captured at IC setup, used by finalize routines.
  static Real g_rho0 = 1.0;
  static Real g_p0   = 1.0;
  static Real g_cs0  = 1.0;
  static Real g_gam  = 5.0/3.0;

  // Radial-acoustic centroid tracking
  static Real g_rc           = 0.0;
  static Real g_initial_centroid = 0.0;

  // 3D pulse / oblique packet IC parameters captured for finalize
  static Real g_x0 = 0.0, g_y0 = 0.0, g_z0 = 0.0;
  static Real g_kx = 0.0, g_ky = 0.0, g_kz = 0.0;

  // Homologous-expansion rate
  static Real g_H = 0.0;

  // Task 3B globals: gravity / Parker / rotation parameters used by IC,
  // diagnostics, and user BCs.
  static Real g_g0     = 0.0;  // constant gravity magnitude (positive)
  static Real g_r0     = 1.0;  // reference radius (inner boundary, IC anchor)
  static Real g_gm     = 0.0;  // GM for 1/r^2 gravity
  static Real g_Omega  = 0.0;  // rotation rate (solid_body_rotation)
  static Real g_L_init = 0.0;  // initial total Lz (Keplerian)
  static Real g_M_init = 0.0;  // initial total mass (Keplerian)
  static Real g_disk_rdisk = 1.5;
  static Real g_disk_thick = 0.1;
  static Real g_parker_rc  = 1.0;
  static Real g_parker_csiso = 1.0;
  static Real g_parker_rho_inner = 1.0;
  static Real g_parker_M_inner = 0.0;
  static Real g_parker_r_inner = 1.0;

  // Forward declarations
  void HydrostaticConstGFinalize(ParameterInput *pin, Mesh *pm);
  void HydrostaticR2Finalize(ParameterInput *pin, Mesh *pm);
  void SolidBodyRotationFinalize(ParameterInput *pin, Mesh *pm);
  void KeplerianOrbitFinalize(ParameterInput *pin, Mesh *pm);
  void ThinDiskFinalize(ParameterInput *pin, Mesh *pm);
  void ParkerIsothermalFinalize(ParameterInput *pin, Mesh *pm);

  // ParkerMachAtR: solve the transonic Parker equation for Mach M(r).
  //   M^2 - ln(M^2) - 4 ln(r/r_c) - 4 r_c/r + 3 = 0
  // For r < r_c return the subsonic branch (M in (0,1)); for r > r_c return
  // the supersonic branch (M in (1, inf)). 60 bisection iterations is plenty.
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
    // If bracket doesn't straddle zero (shouldn't happen for well-posed r),
    // return the nearer-end midpoint as a fallback.
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

  // Analytic Parker density at radius r, given rho at the inner reference
  // radius r_ref where the Mach number is M_ref. Uses mass continuity
  // rho * v_r * r^2 = const.
  KOKKOS_INLINE_FUNCTION
  Real ParkerRhoAtR(Real r, Real r_c,
                    Real r_ref, Real rho_ref, Real M_ref) {
    Real M = ParkerMachAtR(r, r_c);
    return rho_ref * (M_ref / M) * (r_ref / r) * (r_ref / r);
  }

  // User radial BC for the Parker pgen: fix ghost zones to the analytic Parker
  // solution at the inner radial boundary; outer boundary is left to outflow
  // by zero-gradient extension.
  void ParkerRadialBCs(Mesh *pm);

  // Pre-declarations
  void UniformFinalize(ParameterInput *pin, Mesh *pm);
  void SoundPulseFinalize(ParameterInput *pin, Mesh *pm);
  void RadialAcousticFinalize(ParameterInput *pin, Mesh *pm);
  void DivergenceFinalize(ParameterInput *pin, Mesh *pm);
  void Pulse3DFinalize(ParameterInput *pin, Mesh *pm);
  void ObliquePacketFinalize(ParameterInput *pin, Mesh *pm);
  void HomologousFinalize(ParameterInput *pin, Mesh *pm);
}

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem
//! \brief Initial conditions dispatch for spherical-shell hydro test modes.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  if (restart) return;
  if (pmy_mesh_->pmb_pack->phydro == nullptr) {
    std::cout << "### FATAL ERROR: sph_shell_hydro pgen requires <hydro> block"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pmy_mesh_->pmb_pack->pcoord->coord_system !=
      CoordinateSystem::spherical_shell) {
    std::cout << "### FATAL ERROR: sph_shell_hydro pgen requires"
              << " <coord>/system = spherical_shell" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  std::string mode_str = pin->GetOrAddString("problem", "mode", "uniform");

  Real rho0 = pin->GetOrAddReal("problem", "rho0", 1.0);
  Real p0   = pin->GetOrAddReal("problem", "p0", 1.0);
  Real amp  = pin->GetOrAddReal("problem", "amp", 0.0);
  Real rc   = pin->GetOrAddReal("problem", "rc", 0.0);
  Real rw   = pin->GetOrAddReal("problem", "rw", 0.05);
  bool entropy_pulse = pin->GetOrAddBoolean("problem", "entropy_pulse", false);

  auto *pmbp = pmy_mesh_->pmb_pack;
  auto &indcs = pmy_mesh_->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pmbp->nmb_thispack - 1;
  auto &u0 = pmbp->phydro->u0;
  auto &eos = pmbp->phydro->peos->eos_data;
  auto geom = pmbp->pcoord->shell_geom;
  const Real gam = eos.gamma;

  g_rho0 = rho0;
  g_p0   = p0;
  g_cs0  = std::sqrt(gam * p0 / rho0);
  g_gam  = gam;

  if (mode_str == "uniform") {
    g_mode = Mode::kUniform;
    pgen_final_func = UniformFinalize;
    par_for("sph_uniform_ic", DevExeSpace(),
            0, nmb1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      u0(m, IDN, k, j, i) = rho0;
      u0(m, IM1, k, j, i) = 0.0;
      u0(m, IM2, k, j, i) = 0.0;
      u0(m, IM3, k, j, i) = 0.0;
      u0(m, IEN, k, j, i) = p0 / (gam - 1.0);
    });

  } else if (mode_str == "sound_pulse") {
    g_mode = Mode::kSoundPulse;
    g_rc  = rc;
    pgen_final_func = SoundPulseFinalize;
    par_for("sph_pulse_ic", DevExeSpace(),
            0, nmb1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      Real r = geom.r_vol(m, i);
      Real arg = (r - rc) / rw;
      Real bump = amp * std::exp(-arg*arg);
      Real rho = rho0 * (1.0 + bump);
      Real p   = p0   * (1.0 + (entropy_pulse ? 0.0 : gam) * bump);
      u0(m, IDN, k, j, i) = rho;
      u0(m, IM1, k, j, i) = 0.0;
      u0(m, IM2, k, j, i) = 0.0;
      u0(m, IM3, k, j, i) = 0.0;
      u0(m, IEN, k, j, i) = p / (gam - 1.0);
    });

  } else if (mode_str == "radial_acoustic") {
    // Initial-value linear acoustic pulse, intended to launch one outgoing
    // and one ingoing wave. Use small amplitude to stay in the linear regime.
    // The IC is delta_rho = A exp[-(r-r0)^2/(2*sigma^2)], delta_p = cs^2 * delta_rho,
    // v_r = + delta_p/(rho0*cs). At leading order this is a half-amplitude
    // outgoing wave (amplitude A/2) plus a half-amplitude ingoing wave; the
    // OUTGOING piece is what we track.
    g_mode = Mode::kRadialAcoustic;
    g_rc = rc;
    pgen_final_func = RadialAcousticFinalize;
    const Real cs = g_cs0;
    const Real sigma = rw;
    par_for("sph_radac_ic", DevExeSpace(),
            0, nmb1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      Real r = geom.r_vol(m, i);
      Real arg = (r - rc) / sigma;
      Real bump = amp * std::exp(-0.5 * arg * arg);
      Real drho = rho0 * bump;
      Real dp   = cs * cs * drho;
      Real vr   = dp / (rho0 * cs);    // sign: outgoing pulse positive
      Real rho = rho0 + drho;
      Real p   = p0   + dp;
      u0(m, IDN, k, j, i) = rho;
      u0(m, IM1, k, j, i) = rho * vr;
      u0(m, IM2, k, j, i) = 0.0;
      u0(m, IM3, k, j, i) = 0.0;
      u0(m, IEN, k, j, i) = p/(gam - 1.0) + 0.5 * rho * vr * vr;
    });

    // Compute initial outgoing-pulse density-weighted centroid for later use.
    {
      Real local_num = 0.0;
      Real local_den = 0.0;
      auto u0c = u0;
      auto geomc = geom;
      const Real rho0c = rho0;
      Kokkos::parallel_reduce("radac_init_centroid",
        Kokkos::RangePolicy<>(DevExeSpace(),
                              0, (nmb1+1)*(ke-ks+1)*(je-js+1)*(ie-is+1)),
      KOKKOS_LAMBDA(const int idx, Real &n_, Real &d_) {
        int nx1 = ie - is + 1;
        int nx2 = je - js + 1;
        int nx3 = ke - ks + 1;
        int nji = nx2 * nx1;
        int nkji = nx3 * nx2 * nx1;
        int m = idx / nkji;
        int k = (idx - m*nkji) / nji;
        int j = (idx - m*nkji - k*nji) / nx1;
        int i = (idx - m*nkji - k*nji - j*nx1) + is;
        k += ks; j += js;
        Real drho = u0c(m, IDN, k, j, i) - rho0c;
        Real V = geomc.dr3_third(m,i) * geomc.dcos_theta(m,j) * geomc.dphi(m,k);
        Real r = geomc.r_vol(m, i);
        n_ += drho * r * V;
        d_ += drho * V;
      }, local_num, local_den);
      g_initial_centroid = (std::abs(local_den) > 0.0) ?
                           (local_num / local_den) : rc;
    }

  } else if (mode_str == "divergence_test") {
    // Trivial uniform IC; the real work happens in the Finalize routine.
    g_mode = Mode::kDivergenceTest;
    pgen_final_func = DivergenceFinalize;
    par_for("sph_divtest_ic", DevExeSpace(),
            0, nmb1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      u0(m, IDN, k, j, i) = rho0;
      u0(m, IM1, k, j, i) = 0.0;
      u0(m, IM2, k, j, i) = 0.0;
      u0(m, IM3, k, j, i) = 0.0;
      u0(m, IEN, k, j, i) = p0 / (gam - 1.0);
    });

  } else if (mode_str == "pulse_3d") {
    // Localized 3D Gaussian pressure pulse centred at (x0,y0,z0) in embedded
    // Cartesian space. delta_p = A exp(-d^2/(2*sigma^2)), delta_rho = delta_p/cs^2,
    // v = 0. Visual / vibe check: in embedded space the wavefront should be
    // a nearly spherical shell moving at cs.
    g_mode = Mode::kPulse3D;
    pgen_final_func = Pulse3DFinalize;
    const Real x0 = pin->GetOrAddReal("problem", "x0", 0.0);
    const Real y0 = pin->GetOrAddReal("problem", "y0", 0.0);
    const Real z0 = pin->GetOrAddReal("problem", "z0", 1.5);
    const Real sigma = pin->GetOrAddReal("problem", "sigma", 0.1);
    g_x0 = x0; g_y0 = y0; g_z0 = z0;
    const Real cs = g_cs0;
    par_for("sph_pulse3d_ic", DevExeSpace(),
            0, nmb1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      Real r  = geom.r_vol(m, i);
      Real th = geom.theta_vol(m, j);
      Real ph = geom.phi_center(m, k);
      Real sint = std::sin(th), cost = std::cos(th);
      Real sinp = std::sin(ph), cosp = std::cos(ph);
      Real x = r * sint * cosp;
      Real y = r * sint * sinp;
      Real z = r * cost;
      Real dx = x - x0, dy = y - y0, dz = z - z0;
      Real d2 = dx*dx + dy*dy + dz*dz;
      Real bump = amp * std::exp(-0.5 * d2 / (sigma * sigma));
      Real dp   = p0 * bump;
      Real drho = dp / (cs * cs);
      Real rho = rho0 + drho;
      Real p   = p0   + dp;
      u0(m, IDN, k, j, i) = rho;
      u0(m, IM1, k, j, i) = 0.0;
      u0(m, IM2, k, j, i) = 0.0;
      u0(m, IM3, k, j, i) = 0.0;
      u0(m, IEN, k, j, i) = p / (gam - 1.0);
    });

  } else if (mode_str == "oblique_packet") {
    // Oblique acoustic wave packet defined in embedded Cartesian.
    //  delta_rho = A * envelope(|x-x0|) * sin(kvec . (x-x0))
    //  delta_p   = cs^2 * delta_rho
    //  delta_v_cart = (cs * delta_rho / rho0) * khat
    // Project delta_v_cart onto spherical orthonormal (e_r, e_theta, e_phi).
    g_mode = Mode::kObliquePacket;
    pgen_final_func = ObliquePacketFinalize;
    const Real x0 = pin->GetOrAddReal("problem", "x0", 0.0);
    const Real y0 = pin->GetOrAddReal("problem", "y0", 0.0);
    const Real z0 = pin->GetOrAddReal("problem", "z0", 1.6);
    const Real kx = pin->GetOrAddReal("problem", "kx",  6.0);
    const Real ky = pin->GetOrAddReal("problem", "ky",  4.0);
    const Real kz = pin->GetOrAddReal("problem", "kz",  3.0);
    const Real sigma = pin->GetOrAddReal("problem", "sigma", 0.15);
    g_x0 = x0; g_y0 = y0; g_z0 = z0;
    g_kx = kx; g_ky = ky; g_kz = kz;
    const Real kmag = std::sqrt(kx*kx + ky*ky + kz*kz);
    const Real khx = kx / kmag, khy = ky / kmag, khz = kz / kmag;
    const Real cs = g_cs0;
    par_for("sph_obliq_ic", DevExeSpace(),
            0, nmb1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      Real r  = geom.r_vol(m, i);
      Real th = geom.theta_vol(m, j);
      Real ph = geom.phi_center(m, k);
      Real sint = std::sin(th), cost = std::cos(th);
      Real sinp = std::sin(ph), cosp = std::cos(ph);
      Real x = r * sint * cosp;
      Real y = r * sint * sinp;
      Real z = r * cost;
      Real dx = x - x0, dy = y - y0, dz = z - z0;
      Real d2 = dx*dx + dy*dy + dz*dz;
      Real env = std::exp(-0.5 * d2 / (sigma * sigma));
      Real phase = kx * dx + ky * dy + kz * dz;
      Real wave  = std::sin(phase);
      Real drho  = amp * rho0 * env * wave;
      Real dp    = cs * cs * drho;
      Real vmag  = cs * drho / rho0;
      // Cartesian velocity perturbation along khat.
      Real vxc = vmag * khx, vyc = vmag * khy, vzc = vmag * khz;
      // Project to spherical orthonormal basis.
      Real er_x =  sint*cosp, er_y =  sint*sinp, er_z =  cost;
      Real et_x =  cost*cosp, et_y =  cost*sinp, et_z = -sint;
      Real ep_x = -sinp,      ep_y =  cosp,      ep_z =  0.0;
      Real vr = vxc*er_x + vyc*er_y + vzc*er_z;
      Real vt = vxc*et_x + vyc*et_y + vzc*et_z;
      Real vp = vxc*ep_x + vyc*ep_y + vzc*ep_z;
      Real rho = rho0 + drho;
      Real p   = p0   + dp;
      u0(m, IDN, k, j, i) = rho;
      u0(m, IM1, k, j, i) = rho * vr;
      u0(m, IM2, k, j, i) = rho * vt;
      u0(m, IM3, k, j, i) = rho * vp;
      u0(m, IEN, k, j, i) = p / (gam - 1.0)
                          + 0.5 * rho * (vr*vr + vt*vt + vp*vp);
    });

  } else if (mode_str == "homologous") {
    // Uniform density with v_r = H * r. div(v) = 3 H so density should fall as
    // d rho / dt = -3 H rho at t=0+. We check d rho / dt against analytic at
    // first cycle in the finalize routine.
    g_mode = Mode::kHomologous;
    pgen_final_func = HomologousFinalize;
    const Real H = pin->GetOrAddReal("problem", "H", 0.0);
    g_H = H;
    par_for("sph_homol_ic", DevExeSpace(),
            0, nmb1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      Real r = geom.r_vol(m, i);
      Real vr = H * r;
      u0(m, IDN, k, j, i) = rho0;
      u0(m, IM1, k, j, i) = rho0 * vr;
      u0(m, IM2, k, j, i) = 0.0;
      u0(m, IM3, k, j, i) = 0.0;
      u0(m, IEN, k, j, i) = p0/(gam - 1.0) + 0.5 * rho0 * vr * vr;
    });

  } else if (mode_str == "hydrostatic_constg") {
    // Isothermal hydrostatic equilibrium in constant radial gravity g_r = -g0:
    //   rho(r) = rho0 * exp(-(r - r0) / H_p),   p = rho * c_s^2,
    //   H_p    = c_s^2 / g0
    // The hydro evolution is adiabatic with gamma=5/3 (and a uniform initial
    // temperature equal to p0/rho0/(gamma-1) per cell from the IC). To make
    // the IC pressure self-consistent with the isothermal HSE we set
    // p(r) = c_s^2 * rho(r), so the IC has a position-dependent temperature
    // but a globally uniform "isothermal" c_s. After one or two sound-crossing
    // times the adiabatic evolution will drift; documented in AGENTS.md.
    g_mode = Mode::kHydrostaticConstG;
    pgen_final_func = HydrostaticConstGFinalize;
    g_g0 = pin->GetReal("problem", "g0");
    g_r0 = pin->GetOrAddReal("problem", "r_ref", pmy_mesh_->mesh_size.x1min);
    const Real g0   = g_g0;
    const Real rref = g_r0;
    const Real cs   = std::sqrt(p0 / rho0);     // isothermal sound speed
    const Real Hp   = (cs * cs) / g0;
    par_for("sph_hse_constg_ic", DevExeSpace(),
            0, nmb1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      Real r = geom.r_vol(m, i);
      Real rho = rho0 * Kokkos::exp(-(r - rref) / Hp);
      Real p   = cs * cs * rho;
      u0(m, IDN, k, j, i) = rho;
      u0(m, IM1, k, j, i) = 0.0;
      u0(m, IM2, k, j, i) = 0.0;
      u0(m, IM3, k, j, i) = 0.0;
      u0(m, IEN, k, j, i) = p / (gam - 1.0);
    });

  } else if (mode_str == "hydrostatic_r2") {
    // Isothermal hydrostatic equilibrium in 1/r^2 gravity (g_r = -GM/r^2):
    //   rho(r) = rho0 * exp[ (GM / c_s^2) * (1/r - 1/r0) ],
    //   p(r)   = c_s^2 rho(r).
    // Same caveat about isothermal IC under adiabatic evolution applies.
    g_mode = Mode::kHydrostaticR2;
    pgen_final_func = HydrostaticR2Finalize;
    g_gm = pin->GetReal("problem", "gm");
    g_r0 = pin->GetOrAddReal("problem", "r_ref", pmy_mesh_->mesh_size.x1min);
    const Real gm_   = g_gm;
    const Real rref = g_r0;
    const Real cs   = std::sqrt(p0 / rho0);
    par_for("sph_hse_r2_ic", DevExeSpace(),
            0, nmb1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      Real r = geom.r_vol(m, i);
      Real rho = rho0 * Kokkos::exp((gm_ / (cs * cs)) * (1.0/r - 1.0/rref));
      Real p   = cs * cs * rho;
      u0(m, IDN, k, j, i) = rho;
      u0(m, IM1, k, j, i) = 0.0;
      u0(m, IM2, k, j, i) = 0.0;
      u0(m, IM3, k, j, i) = 0.0;
      u0(m, IEN, k, j, i) = p / (gam - 1.0);
    });

  } else if (mode_str == "solid_body_rotation") {
    // One-step centrifugal-source diagnostic.
    // IC: uniform rho, uniform p, v_r=v_theta=0, v_phi = Omega * r * sin(theta).
    // No gravity needed. After one tiny RK step we measure
    //   d(rho v_r)/dt    ~ rho * Omega^2 * r * sin^2(theta)    (positive, outward)
    //   d(rho v_th)/dt   ~ rho * Omega^2 * r * sin*cos         (positive, toward equator)
    //   d(rho v_phi)/dt  ~ 0
    // These come directly out of the implemented spherical FV + Athena++-style
    // geometric source terms; deviations expose wrong centrifugal/cot-theta sources.
    g_mode = Mode::kSolidBodyRotation;
    pgen_final_func = SolidBodyRotationFinalize;
    g_Omega = pin->GetReal("problem", "Omega");
    const Real Om = g_Omega;
    par_for("sph_sbr_ic", DevExeSpace(),
            0, nmb1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      Real r = geom.r_vol(m, i);
      Real th = geom.theta_vol(m, j);
      Real sint = Kokkos::sin(th);
      Real vp = Om * r * sint;
      u0(m, IDN, k, j, i) = rho0;
      u0(m, IM1, k, j, i) = 0.0;
      u0(m, IM2, k, j, i) = 0.0;
      u0(m, IM3, k, j, i) = rho0 * vp;
      u0(m, IEN, k, j, i) = p0/(gam - 1.0) + 0.5 * rho0 * vp * vp;
    });

  } else if (mode_str == "keplerian_orbit") {
    // Near-Keplerian circular orbit in 1/r^2 gravity.
    // For a thin equatorial wedge,
    //   v_phi(r, theta) = sqrt(GM/r) * sin(theta)
    // approximately balances the centrifugal "force" by gravity in the radial
    // direction; the theta-direction is NOT exactly balanced and shows mild
    // adjustment. Periodic phi is recommended.
    g_mode = Mode::kKeplerianOrbit;
    pgen_final_func = KeplerianOrbitFinalize;
    g_gm = pin->GetReal("problem", "gm");
    const Real gm_ = g_gm;
    par_for("sph_kep_ic", DevExeSpace(),
            0, nmb1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      Real r = geom.r_vol(m, i);
      Real th = geom.theta_vol(m, j);
      Real sint = Kokkos::sin(th);
      Real vphi = Kokkos::sqrt(gm_ / r) * sint;
      u0(m, IDN, k, j, i) = rho0;
      u0(m, IM1, k, j, i) = 0.0;
      u0(m, IM2, k, j, i) = 0.0;
      u0(m, IM3, k, j, i) = rho0 * vphi;
      u0(m, IEN, k, j, i) = p0/(gam - 1.0) + 0.5 * rho0 * vphi * vphi;
    });

    // Capture initial total Lz and total mass for the conservation diagnostic.
    {
      Real lLz = 0.0, lM = 0.0;
      auto u0c = u0;
      auto geomc = geom;
      Kokkos::parallel_reduce("kep_init_Lz",
        Kokkos::RangePolicy<>(DevExeSpace(),
                              0, (nmb1+1)*(ke-ks+1)*(je-js+1)*(ie-is+1)),
      KOKKOS_LAMBDA(const int idx, Real &Lz_, Real &M_) {
        int nx1 = ie - is + 1, nx2 = je - js + 1, nx3 = ke - ks + 1;
        int nji = nx2 * nx1, nkji = nx3 * nx2 * nx1;
        int m = idx / nkji;
        int k = (idx - m*nkji) / nji;
        int j = (idx - m*nkji - k*nji) / nx1;
        int i = (idx - m*nkji - k*nji - j*nx1) + is;
        k += ks; j += js;
        Real r = geomc.r_vol(m, i);
        Real th = geomc.theta_vol(m, j);
        Real sint = Kokkos::sin(th);
        Real V = geomc.dr3_third(m,i)*geomc.dcos_theta(m,j)*geomc.dphi(m,k);
        Real rho = u0c(m, IDN, k, j, i);
        Real vphi = u0c(m, IM3, k, j, i) / rho;
        Lz_ += rho * (r * sint) * vphi * V;
        M_  += rho * V;
      }, lLz, lM);
      g_L_init = lLz;
      g_M_init = lM;
    }

  } else if (mode_str == "thin_disk") {
    // Pressure-supported thin disk near the equator in 1/r^2 gravity.
    //   rho(r, theta) = rho_floor + rho0 * exp(-(r - r_disk)^2 / (2 sigma_r^2))
    //                              * exp(-(theta - pi/2)^2 / (2 sigma_th^2))
    //   p             = rho * c_s^2
    //   v_phi         = sqrt(GM/r) * sin(theta)   (approximate Keplerian)
    // Pure vibe check; not an exact equilibrium.
    g_mode = Mode::kThinDisk;
    pgen_final_func = ThinDiskFinalize;
    g_gm = pin->GetReal("problem", "gm");
    g_disk_rdisk = pin->GetOrAddReal("problem", "r_disk", 1.5);
    g_disk_thick = pin->GetOrAddReal("problem", "sigma_th", 0.05);
    const Real gm_      = g_gm;
    const Real r_disk   = g_disk_rdisk;
    const Real sigma_r  = pin->GetOrAddReal("problem", "sigma_r", 0.2);
    const Real sigma_th = g_disk_thick;
    const Real rho_floor = pin->GetOrAddReal("problem", "rho_floor", 1.0e-3);
    const Real cs       = std::sqrt(p0 / rho0);
    par_for("sph_disk_ic", DevExeSpace(),
            0, nmb1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      Real r = geom.r_vol(m, i);
      Real th = geom.theta_vol(m, j);
      Real arg_r  = (r - r_disk) / sigma_r;
      Real arg_th = (th - 0.5*M_PI) / sigma_th;
      Real rho = rho_floor + rho0 *
                 Kokkos::exp(-0.5*arg_r*arg_r) *
                 Kokkos::exp(-0.5*arg_th*arg_th);
      Real p   = rho * cs * cs;
      Real vphi = Kokkos::sqrt(gm_ / r) * Kokkos::sin(th);
      u0(m, IDN, k, j, i) = rho;
      u0(m, IM1, k, j, i) = 0.0;
      u0(m, IM2, k, j, i) = 0.0;
      u0(m, IM3, k, j, i) = rho * vphi;
      u0(m, IEN, k, j, i) = p / (gam - 1.0) + 0.5 * rho * vphi * vphi;
    });

  } else if (mode_str == "parker_isothermal") {
    // Analytic isothermal Parker wind initialiser.
    // r_c = GM / (2 c_s^2), Mach M(r) from the transonic Parker equation
    // (bisection at each cell), rho(r) from mass continuity rho v_r r^2 = const
    // anchored at rho(r_inner) = rho_inner.
    // Note: hydro EOS is ideal gas (gamma in input); the isothermal Parker
    // solution will not be an exact steady state under adiabatic evolution.
    // Use the user_bcs_func to hold the analytic inner BC; outer BC outflow.
    g_mode = Mode::kParkerIsothermal;
    pgen_final_func = ParkerIsothermalFinalize;
    user_bcs_func = ParkerRadialBCs;
    g_gm        = pin->GetReal("problem", "gm");
    g_parker_csiso = pin->GetReal("problem", "cs_iso");
    g_parker_rho_inner = pin->GetOrAddReal("problem", "rho_inner", 1.0);
    g_parker_r_inner   = pmy_mesh_->mesh_size.x1min;
    g_parker_rc        = g_gm / (2.0 * g_parker_csiso * g_parker_csiso);
    // Pre-compute Mach number at the inner reference radius.
    g_parker_M_inner   = ParkerMachAtR(g_parker_r_inner, g_parker_rc);
    const Real gm_    = g_gm;
    const Real cs     = g_parker_csiso;
    const Real rc     = g_parker_rc;
    const Real r_ref  = g_parker_r_inner;
    const Real rho_ref= g_parker_rho_inner;
    const Real M_ref  = g_parker_M_inner;
    par_for("sph_parker_ic", DevExeSpace(),
            0, nmb1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      Real r = geom.r_vol(m, i);
      Real M = ParkerMachAtR(r, rc);
      Real vr = M * cs;
      Real rho = rho_ref * (M_ref / M) * (r_ref / r) * (r_ref / r);
      Real p   = cs * cs * rho;
      u0(m, IDN, k, j, i) = rho;
      u0(m, IM1, k, j, i) = rho * vr;
      u0(m, IM2, k, j, i) = 0.0;
      u0(m, IM3, k, j, i) = 0.0;
      u0(m, IEN, k, j, i) = p / (gam - 1.0) + 0.5 * rho * vr * vr;
    });

  } else {
    std::cout << "### FATAL ERROR: sph_shell_hydro mode='" << mode_str
              << "' not recognised." << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

//----------------------------------------------------------------------------------------
// Finalize routines (run at end of Driver, after all output dumps).

namespace {

//----------------------------------------------------------------------------------------
// "uniform": print max deviation of (rho, p, v) from the analytic constant state.

void UniformFinalize(ParameterInput *pin, Mesh *pm) {
  if (g_mode != Mode::kUniform) return;
  auto *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pmbp->nmb_thispack - 1;

  auto u0 = pmbp->phydro->u0;
  Real local_max_drho = 0.0, local_max_dp = 0.0, local_max_v = 0.0;
  const Real rho0 = g_rho0;
  const Real p0   = g_p0;
  const Real gam  = pmbp->phydro->peos->eos_data.gamma;
  Kokkos::parallel_reduce("uniform_diag",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, (nmb1+1)*(ke-ks+1)*(je-js+1)*(ie-is+1)),
  KOKKOS_LAMBDA(const int idx, Real &mdrho, Real &mdp, Real &mv) {
    int nx1 = ie - is + 1;
    int nx2 = je - js + 1;
    int nx3 = ke - ks + 1;
    int nji = nx2 * nx1;
    int nkji = nx3 * nx2 * nx1;
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
    Real vmag = std::sqrt(u0(m,IM1,k,j,i)*u0(m,IM1,k,j,i)
                         +u0(m,IM2,k,j,i)*u0(m,IM2,k,j,i)
                         +u0(m,IM3,k,j,i)*u0(m,IM3,k,j,i)) / rho;
    Real adrho = std::abs(rho - rho0);
    Real adp   = std::abs(p   - p0);
    if (adrho > mdrho) mdrho = adrho;
    if (adp   > mdp)   mdp   = adp;
    if (vmag  > mv)    mv    = vmag;
  }, Kokkos::Max<Real>(local_max_drho),
     Kokkos::Max<Real>(local_max_dp),
     Kokkos::Max<Real>(local_max_v));

  std::cout.precision(6);
  std::cout << std::scientific
            << "[sph_shell_hydro/uniform] max |drho| = " << local_max_drho
            << " (relative " << local_max_drho/rho0 << ")\n"
            << "[sph_shell_hydro/uniform] max |dp|   = " << local_max_dp
            << " (relative " << local_max_dp/p0 << ")\n"
            << "[sph_shell_hydro/uniform] max |v|    = " << local_max_v
            << std::endl;
}

//----------------------------------------------------------------------------------------
// "sound_pulse": legacy half-max-trailing-edge reporter (Task 2 compatible).

void SoundPulseFinalize(ParameterInput *pin, Mesh *pm) {
  if (g_mode != Mode::kSoundPulse) return;
  auto *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pmbp->nmb_thispack - 1;
  const Real rho0 = g_rho0, rc = g_rc, cs0 = g_cs0;
  auto u0 = pmbp->phydro->u0;
  auto geom = pmbp->pcoord->shell_geom;

  Real local_max = 0.0;
  Kokkos::parallel_reduce("pulse_max_drho",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, (nmb1+1)*(ke-ks+1)*(je-js+1)*(ie-is+1)),
  KOKKOS_LAMBDA(const int idx, Real &mx) {
    int nx1 = ie - is + 1, nx2 = je - js + 1, nx3 = ke - ks + 1;
    int nji = nx2 * nx1, nkji = nx3 * nx2 * nx1;
    int m = idx / nkji;
    int k = (idx - m*nkji) / nji;
    int j = (idx - m*nkji - k*nji) / nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks; j += js;
    Real ad = std::abs(u0(m, IDN, k, j, i) - rho0);
    if (ad > mx) mx = ad;
  }, Kokkos::Max<Real>(local_max));

  Real r_at_max = 0.0;
  const Real tol = local_max * 0.5;
  Kokkos::parallel_reduce("pulse_r_at_max",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, (nmb1+1)*(ke-ks+1)*(je-js+1)*(ie-is+1)),
  KOKKOS_LAMBDA(const int idx, Real &r_out) {
    int nx1 = ie - is + 1, nx2 = je - js + 1, nx3 = ke - ks + 1;
    int nji = nx2 * nx1, nkji = nx3 * nx2 * nx1;
    int m = idx / nkji;
    int k = (idx - m*nkji) / nji;
    int j = (idx - m*nkji - k*nji) / nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks; j += js;
    Real ad = std::abs(u0(m, IDN, k, j, i) - rho0);
    Real r = geom.r_vol(m, i);
    if (ad >= tol && r > rc && r > r_out) r_out = r;
  }, Kokkos::Max<Real>(r_at_max));
  if (r_at_max < 0.0) r_at_max = 0.0;

  Real t = pm->time;
  Real r_pred = rc + cs0 * t;
  std::cout.precision(6);
  std::cout << std::scientific
            << "[sph_shell_hydro/sound_pulse] t          = " << t      << "\n"
            << "[sph_shell_hydro/sound_pulse] max|drho|  = " << local_max << "\n"
            << "[sph_shell_hydro/sound_pulse] r_peak_out = " << r_at_max
            << "  predicted ~= " << r_pred
            << "  err = "       << (r_at_max - r_pred)
            << std::endl;
}

//----------------------------------------------------------------------------------------
// "radial_acoustic": density-weighted centroid of the outgoing wave, expected to
// follow <r>_outgoing(t) = <r>_outgoing(0) + cs * t for small amplitudes and
// far from boundaries. We compute the centroid only over cells with r > rc
// (outgoing half).

void RadialAcousticFinalize(ParameterInput *pin, Mesh *pm) {
  if (g_mode != Mode::kRadialAcoustic) return;
  auto *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pmbp->nmb_thispack - 1;
  const Real rho0 = g_rho0, rc = g_rc, cs0 = g_cs0;
  auto u0 = pmbp->phydro->u0;
  auto geom = pmbp->pcoord->shell_geom;

  Real local_num = 0.0, local_den = 0.0, local_max = 0.0;
  Kokkos::parallel_reduce("radac_centroid",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, (nmb1+1)*(ke-ks+1)*(je-js+1)*(ie-is+1)),
  KOKKOS_LAMBDA(const int idx, Real &n_, Real &d_, Real &mx) {
    int nx1 = ie - is + 1, nx2 = je - js + 1, nx3 = ke - ks + 1;
    int nji = nx2 * nx1, nkji = nx3 * nx2 * nx1;
    int m = idx / nkji;
    int k = (idx - m*nkji) / nji;
    int j = (idx - m*nkji - k*nji) / nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks; j += js;
    Real drho = u0(m, IDN, k, j, i) - rho0;
    Real V = geom.dr3_third(m,i) * geom.dcos_theta(m,j) * geom.dphi(m,k);
    Real r = geom.r_vol(m, i);
    // Only positive perturbations on the outgoing side -- this assumes the
    // outgoing wave is the dominant feature for r > rc.
    if (r > rc && drho > 0.0) {
      n_ += drho * r * V;
      d_ += drho * V;
      if (drho > mx) mx = drho;
    }
  }, local_num, local_den, Kokkos::Max<Real>(local_max));

  Real centroid_now = (std::abs(local_den) > 0.0) ?
                      (local_num / local_den) : rc;
  Real t = pm->time;
  Real predicted = g_initial_centroid + cs0 * t;
  Real speed_est = (t > 0.0) ?
                   ((centroid_now - g_initial_centroid) / t) : 0.0;
  Real speed_err = speed_est - cs0;
  std::cout.precision(6);
  std::cout << std::scientific
            << "[sph_shell_hydro/radial_acoustic] t           = " << t << "\n"
            << "[sph_shell_hydro/radial_acoustic] cs0         = " << cs0 << "\n"
            << "[sph_shell_hydro/radial_acoustic] max(drho+)  = "
            << local_max << "\n"
            << "[sph_shell_hydro/radial_acoustic] centroid(0) = "
            << g_initial_centroid << "\n"
            << "[sph_shell_hydro/radial_acoustic] centroid(t) = "
            << centroid_now << "\n"
            << "[sph_shell_hydro/radial_acoustic] predicted   = "
            << predicted << "\n"
            << "[sph_shell_hydro/radial_acoustic] speed_est   = "
            << speed_est << "\n"
            << "[sph_shell_hydro/radial_acoustic] speed_err   = "
            << speed_err
            << " (rel " << (speed_err / cs0) << ")"
            << std::endl;
}

//----------------------------------------------------------------------------------------
// "divergence_test": evaluate the spherical FV divergence operator on three
// analytic vector fields and compare against analytic div(F) at the cell-volume
// centroid. The FV operator uses the SAME geometric factors as the hydro
// update: A_face F_face and V from SphericalShellGeom.
//
// Three cases (matches Athena++-style validation):
//   (1)  F_r = r^n,             div F = (n+2) r^(n-1)
//   (2)  F_th = sin(theta),      div F = 2 cos(theta) / r
//   (3)  F_ph = sin(m*phi),      div F = m cos(m*phi) / (r * sin theta)
//
// Errors compared against centroid analytic; expected to be O(h^2) for smooth
// integrands due to the centroid-vs-volume-average truncation. Case 1 with
// n=1 yields div F = 3 (constant), for which the FV and centroid values
// agree exactly at every cell -- a useful exact-roundoff sanity check.

void RunDivCase(MeshBlockPack *pmbp,
                int case_id, int n_exp, int m_phi,
                Real &l1, Real &l2, Real &linf, Real &linf_an, int &ncell) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pmbp->nmb_thispack - 1;
  auto geom = pmbp->pcoord->shell_geom;

  const int Ncell = (nmb1+1)*(ke-ks+1)*(je-js+1)*(ie-is+1);
  ncell = Ncell;

  Real l1_sum = 0.0, l2_sum = 0.0, linf_loc = 0.0, linf_an_loc = 0.0;
  Kokkos::parallel_reduce("div_test_case",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, Ncell),
  KOKKOS_LAMBDA(const int idx, Real &s1, Real &s2, Real &smax, Real &smax_an) {
    int nx1 = ie - is + 1, nx2 = je - js + 1, nx3 = ke - ks + 1;
    int nji = nx2 * nx1, nkji = nx3 * nx2 * nx1;
    int m = idx / nkji;
    int k = (idx - m*nkji) / nji;
    int j = (idx - m*nkji - k*nji) / nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks; j += js;

    Real divFV = 0.0;
    Real divAN = 0.0;

    Real rv = geom.r_vol(m, i);
    Real tv = geom.theta_vol(m, j);
    Real pv = geom.phi_center(m, k);
    Real rm = geom.r_face(m, i);
    Real rp = geom.r_face(m, i+1);
    Real r2m = geom.r2_face(m, i);
    Real r2p = geom.r2_face(m, i+1);
    Real inv_v_i = geom.inv_dr3_third(m, i);
    Real dr2h = geom.dr2_half(m, i);
    Real dcos_j = geom.dcos_theta(m, j);
    Real sm_face = geom.sin_theta_face(m, j);
    Real sp_face = geom.sin_theta_face(m, j+1);
    Real dphi_k = geom.dphi(m, k);
    Real dth_j = geom.dtheta(m, j);
    Real pm_face = geom.phi_face(m, k);
    Real pp_face = geom.phi_face(m, k+1);

    if (case_id == 1) {
      // F_r = r^n; F_th = F_ph = 0
      // FV divergence reduces to (r+^2 F+ - r-^2 F-) / dr3_third.
      Real F1m = std::pow(rm, (Real)n_exp);
      Real F1p = std::pow(rp, (Real)n_exp);
      divFV = (r2p * F1p - r2m * F1m) * inv_v_i;
      // Analytic centroid divergence.
      if (n_exp == 1) {
        divAN = (n_exp + 2.0);                       // (n+2) r^0 = 3
      } else {
        divAN = (n_exp + 2.0) * std::pow(rv, (Real)(n_exp - 1));
      }
    } else if (case_id == 2) {
      // F_th = sin(theta); others zero.
      //  divFV = (dr2_half / dr3_third) * (sin+ * sin+ - sin- * sin-) / dcos
      Real F2m = sm_face, F2p = sp_face;
      divFV = (dr2h * inv_v_i) * (sp_face * F2p - sm_face * F2m) / dcos_j;
      // Analytic centroid: 2 cos(theta) / r
      divAN = 2.0 * std::cos(tv) / rv;
    } else {
      // case_id == 3 : F_ph = sin(m phi); others zero.
      Real F3m = std::sin((Real)m_phi * pm_face);
      Real F3p = std::sin((Real)m_phi * pp_face);
      divFV = (dr2h * inv_v_i) * dth_j * (F3p - F3m) / (dcos_j * dphi_k);
      // Analytic centroid: m cos(m phi) / (r sin theta)
      divAN = (Real)m_phi * std::cos((Real)m_phi * pv) / (rv * std::sin(tv));
    }

    Real err = std::abs(divFV - divAN);
    s1   += err;
    s2   += err * err;
    if (err     > smax)    smax    = err;
    if (std::abs(divAN) > smax_an) smax_an = std::abs(divAN);
  }, l1_sum, l2_sum,
     Kokkos::Max<Real>(linf_loc),
     Kokkos::Max<Real>(linf_an_loc));

  l1  = l1_sum / static_cast<Real>(Ncell);
  l2  = std::sqrt(l2_sum / static_cast<Real>(Ncell));
  linf = linf_loc;
  linf_an = linf_an_loc;
}

void DivergenceFinalize(ParameterInput *pin, Mesh *pm) {
  if (g_mode != Mode::kDivergenceTest) return;
  auto *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const Real dr_typ = (pm->mesh_size.x1max - pm->mesh_size.x1min) / nx1;
  const Real dth_typ = (pm->mesh_size.x2max - pm->mesh_size.x2min) / nx2;
  const Real dph_typ = (pm->mesh_size.x3max - pm->mesh_size.x3min) / nx3;
  const Real h_typ = std::max({dr_typ, dth_typ, dph_typ});

  const int n_exp = pin->GetOrAddInteger("problem", "div_n_exp", 2);
  const int m_phi = pin->GetOrAddInteger("problem", "div_m_phi", 2);

  std::cout << "[sph_shell_hydro/divergence_test] grid: "
            << nx1 << "x" << nx2 << "x" << nx3
            << "  (dr,dth,dphi) ~ ("
            << dr_typ << "," << dth_typ << "," << dph_typ << ")\n";

  for (int c = 1; c <= 3; ++c) {
    Real l1, l2, linf, linf_an;
    int N;
    RunDivCase(pmbp, c, n_exp, m_phi, l1, l2, linf, linf_an, N);
    std::cout.precision(6);
    std::cout << std::scientific
              << "[divergence_test] case " << c;
    if (c == 1)      std::cout << " (F_r = r^"   << n_exp << ")";
    else if (c == 2) std::cout << " (F_th = sin theta)";
    else             std::cout << " (F_ph = sin(" << m_phi << " phi))";
    std::cout << "  L1="    << l1
              << "  L2="    << l2
              << "  Linf="  << linf
              << "  |an|max=" << linf_an
              << "  (h~" << h_typ << ")"
              << std::endl;
  }
}

//----------------------------------------------------------------------------------------
// "pulse_3d": brief reporter -- peak perturbation amplitude and a rough estimate
// of the wavefront radius around (x0,y0,z0). Detailed visual interpretation
// requires the standard VTK / tab outputs.

void Pulse3DFinalize(ParameterInput *pin, Mesh *pm) {
  if (g_mode != Mode::kPulse3D) return;
  auto *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pmbp->nmb_thispack - 1;
  const Real rho0 = g_rho0, cs0 = g_cs0;
  const Real x0 = g_x0, y0 = g_y0, z0 = g_z0;
  auto u0 = pmbp->phydro->u0;
  auto geom = pmbp->pcoord->shell_geom;

  Real local_num = 0.0, local_den = 0.0, local_max = 0.0;
  Kokkos::parallel_reduce("pulse3d_diag",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, (nmb1+1)*(ke-ks+1)*(je-js+1)*(ie-is+1)),
  KOKKOS_LAMBDA(const int idx, Real &n_, Real &d_, Real &mx) {
    int nx1 = ie - is + 1, nx2 = je - js + 1, nx3 = ke - ks + 1;
    int nji = nx2 * nx1, nkji = nx3 * nx2 * nx1;
    int m = idx / nkji;
    int k = (idx - m*nkji) / nji;
    int j = (idx - m*nkji - k*nji) / nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks; j += js;
    Real r = geom.r_vol(m,i);
    Real th = geom.theta_vol(m,j);
    Real ph = geom.phi_center(m,k);
    Real x = r*std::sin(th)*std::cos(ph);
    Real y = r*std::sin(th)*std::sin(ph);
    Real z = r*std::cos(th);
    Real dx = x - x0, dy = y - y0, dz = z - z0;
    Real dist = std::sqrt(dx*dx + dy*dy + dz*dz);
    Real V = geom.dr3_third(m,i)*geom.dcos_theta(m,j)*geom.dphi(m,k);
    Real drho_pos = u0(m,IDN,k,j,i) - rho0;
    Real w = drho_pos * drho_pos;
    n_ += w * dist * V;
    d_ += w * V;
    if (std::abs(drho_pos) > mx) mx = std::abs(drho_pos);
  }, local_num, local_den, Kokkos::Max<Real>(local_max));

  Real centroid_dist = (std::abs(local_den) > 0.0) ?
                       (local_num / local_den) : 0.0;
  Real t = pm->time;
  std::cout.precision(6);
  std::cout << std::scientific
            << "[sph_shell_hydro/pulse_3d] t                = " << t << "\n"
            << "[sph_shell_hydro/pulse_3d] cs0              = " << cs0 << "\n"
            << "[sph_shell_hydro/pulse_3d] max|drho|        = " << local_max
            << "\n"
            << "[sph_shell_hydro/pulse_3d] wavefront radius "
            << "(|drho|^2-weighted) = " << centroid_dist
            << "  ~? cs0*t = " << (cs0 * t)
            << std::endl;
}

//----------------------------------------------------------------------------------------
// "oblique_packet": density-weighted centroid in embedded Cartesian; print
// projection along khat. For visualisation just dump the standard VTK output.

void ObliquePacketFinalize(ParameterInput *pin, Mesh *pm) {
  if (g_mode != Mode::kObliquePacket) return;
  auto *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pmbp->nmb_thispack - 1;
  const Real rho0 = g_rho0, cs0 = g_cs0;
  const Real x0 = g_x0, y0 = g_y0, z0 = g_z0;
  const Real kx = g_kx, ky = g_ky, kz = g_kz;
  const Real kmag = std::sqrt(kx*kx + ky*ky + kz*kz);
  const Real khx = kx / kmag, khy = ky / kmag, khz = kz / kmag;
  auto u0 = pmbp->phydro->u0;
  auto geom = pmbp->pcoord->shell_geom;

  Real cx_num = 0.0, cy_num = 0.0, cz_num = 0.0;
  Real den = 0.0, local_max = 0.0;
  Kokkos::parallel_reduce("obliq_diag",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, (nmb1+1)*(ke-ks+1)*(je-js+1)*(ie-is+1)),
  KOKKOS_LAMBDA(const int idx,
                Real &nx, Real &ny, Real &nz, Real &dd, Real &mx) {
    int nx1 = ie - is + 1, nx2 = je - js + 1, nx3 = ke - ks + 1;
    int nji = nx2 * nx1, nkji = nx3 * nx2 * nx1;
    int m = idx / nkji;
    int k = (idx - m*nkji) / nji;
    int j = (idx - m*nkji - k*nji) / nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks; j += js;
    Real r = geom.r_vol(m,i);
    Real th = geom.theta_vol(m,j);
    Real ph = geom.phi_center(m,k);
    Real x = r*std::sin(th)*std::cos(ph);
    Real y = r*std::sin(th)*std::sin(ph);
    Real z = r*std::cos(th);
    Real V = geom.dr3_third(m,i)*geom.dcos_theta(m,j)*geom.dphi(m,k);
    Real drho = u0(m,IDN,k,j,i) - rho0;
    Real w = drho * drho;
    nx += w * x * V;
    ny += w * y * V;
    nz += w * z * V;
    dd += w * V;
    if (std::abs(drho) > mx) mx = std::abs(drho);
  }, cx_num, cy_num, cz_num, den, Kokkos::Max<Real>(local_max));

  Real cx = (std::abs(den) > 0.0) ? cx_num / den : x0;
  Real cy = (std::abs(den) > 0.0) ? cy_num / den : y0;
  Real cz = (std::abs(den) > 0.0) ? cz_num / den : z0;
  Real proj = (cx - x0) * khx + (cy - y0) * khy + (cz - z0) * khz;
  Real t = pm->time;
  std::cout.precision(6);
  std::cout << std::scientific
            << "[sph_shell_hydro/oblique_packet] t           = " << t << "\n"
            << "[sph_shell_hydro/oblique_packet] cs0         = " << cs0 << "\n"
            << "[sph_shell_hydro/oblique_packet] max|drho|   = " << local_max
            << "\n"
            << "[sph_shell_hydro/oblique_packet] centroid    = ("
            << cx << "," << cy << "," << cz << ")\n"
            << "[sph_shell_hydro/oblique_packet] proj on khat= "
            << proj << "  ~? cs0*t = " << (cs0*t)
            << std::endl;
}

//----------------------------------------------------------------------------------------
// "homologous": evaluate d(rho)/dt at the centre of the time-step using a finite
// difference of u0(IDN), and compare to the analytic -3 H rho0 prediction. This
// is intended to be run with a single short cycle.

void HomologousFinalize(ParameterInput *pin, Mesh *pm) {
  if (g_mode != Mode::kHomologous) return;
  auto *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pmbp->nmb_thispack - 1;
  const Real rho0 = g_rho0, H = g_H;
  auto u0 = pmbp->phydro->u0;
  auto geom = pmbp->pcoord->shell_geom;

  Real local_sum = 0.0, local_den = 0.0;
  Kokkos::parallel_reduce("homol_diag",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, (nmb1+1)*(ke-ks+1)*(je-js+1)*(ie-is+1)),
  KOKKOS_LAMBDA(const int idx, Real &s_, Real &d_) {
    int nx1 = ie - is + 1, nx2 = je - js + 1, nx3 = ke - ks + 1;
    int nji = nx2 * nx1, nkji = nx3 * nx2 * nx1;
    int m = idx / nkji;
    int k = (idx - m*nkji) / nji;
    int j = (idx - m*nkji - k*nji) / nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks; j += js;
    Real V = geom.dr3_third(m,i)*geom.dcos_theta(m,j)*geom.dphi(m,k);
    Real drho = u0(m, IDN, k, j, i) - rho0;
    s_ += drho * V;
    d_ += rho0 * V;
  }, local_sum, local_den);

  Real avg_drho_over_rho = (std::abs(local_den) > 0.0) ?
                           (local_sum / local_den) : 0.0;
  Real t = pm->time;
  Real predicted_avg_drho_rho = -3.0 * H * t;
  std::cout.precision(6);
  std::cout << std::scientific
            << "[sph_shell_hydro/homologous] t              = " << t << "\n"
            << "[sph_shell_hydro/homologous] H              = " << H << "\n"
            << "[sph_shell_hydro/homologous] <drho/rho>     = "
            << avg_drho_over_rho << "\n"
            << "[sph_shell_hydro/homologous] predicted (-3Ht)= "
            << predicted_avg_drho_rho
            << "  err = " << (avg_drho_over_rho - predicted_avg_drho_rho)
            << std::endl;
}

//----------------------------------------------------------------------------------------
// "hydrostatic_constg": Print max relative drift of (rho, p) from the analytic
// isothermal HSE in constant gravity, and max |v|/cs.

void HydrostaticConstGFinalize(ParameterInput *pin, Mesh *pm) {
  if (g_mode != Mode::kHydrostaticConstG) return;
  auto *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pmbp->nmb_thispack - 1;
  const Real rho0 = g_rho0, p0 = g_p0, gam = g_gam;
  const Real cs = std::sqrt(p0 / rho0);
  const Real g0 = g_g0;
  const Real rref = g_r0;
  const Real Hp = (cs * cs) / g0;
  auto u0 = pmbp->phydro->u0;
  auto geom = pmbp->pcoord->shell_geom;

  Real linf_drho = 0.0, linf_dp = 0.0, linf_vrel = 0.0;
  Real l1_drho = 0.0, l1_dp = 0.0;
  int Ncell = (nmb1+1)*(ke-ks+1)*(je-js+1)*(ie-is+1);
  Kokkos::parallel_reduce("hse_constg_diag",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, Ncell),
  KOKKOS_LAMBDA(const int idx, Real &mdrho, Real &mdp, Real &mvrel,
                              Real &s_drho, Real &s_dp) {
    int nx1 = ie - is + 1, nx2 = je - js + 1, nx3 = ke - ks + 1;
    int nji = nx2 * nx1, nkji = nx3 * nx2 * nx1;
    int m = idx / nkji;
    int k = (idx - m*nkji) / nji;
    int j = (idx - m*nkji - k*nji) / nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks; j += js;
    Real r = geom.r_vol(m, i);
    Real rho_an = rho0 * Kokkos::exp(-(r - rref) / Hp);
    Real p_an   = cs * cs * rho_an;
    Real rho = u0(m, IDN, k, j, i);
    Real ek = 0.5 * (u0(m,IM1,k,j,i)*u0(m,IM1,k,j,i)
                    +u0(m,IM2,k,j,i)*u0(m,IM2,k,j,i)
                    +u0(m,IM3,k,j,i)*u0(m,IM3,k,j,i)) / rho;
    Real p = (gam - 1.0) * (u0(m,IEN,k,j,i) - ek);
    Real vmag = Kokkos::sqrt(2.0 * ek / rho);
    Real drho_rel = Kokkos::fabs((rho - rho_an) / rho_an);
    Real dp_rel   = Kokkos::fabs((p - p_an) / p_an);
    Real vrel     = vmag / cs;
    if (drho_rel > mdrho) mdrho = drho_rel;
    if (dp_rel   > mdp)   mdp   = dp_rel;
    if (vrel     > mvrel) mvrel = vrel;
    s_drho += drho_rel;
    s_dp   += dp_rel;
  }, Kokkos::Max<Real>(linf_drho), Kokkos::Max<Real>(linf_dp),
     Kokkos::Max<Real>(linf_vrel), l1_drho, l1_dp);
  l1_drho /= static_cast<Real>(Ncell);
  l1_dp   /= static_cast<Real>(Ncell);

  std::cout.precision(6);
  std::cout << std::scientific
            << "[hydrostatic_constg] t          = " << pm->time << "\n"
            << "[hydrostatic_constg] cs_iso     = " << cs       << "\n"
            << "[hydrostatic_constg] H_p        = " << Hp       << "\n"
            << "[hydrostatic_constg] max|drho|/rho = " << linf_drho << "\n"
            << "[hydrostatic_constg] max|dp|/p     = " << linf_dp   << "\n"
            << "[hydrostatic_constg] max|v|/cs     = " << linf_vrel << "\n"
            << "[hydrostatic_constg] L1 |drho|/rho = " << l1_drho   << "\n"
            << "[hydrostatic_constg] L1 |dp|/p     = " << l1_dp
            << std::endl;
}

//----------------------------------------------------------------------------------------
// "hydrostatic_r2": same as constant-g but for 1/r^2 gravity.

void HydrostaticR2Finalize(ParameterInput *pin, Mesh *pm) {
  if (g_mode != Mode::kHydrostaticR2) return;
  auto *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pmbp->nmb_thispack - 1;
  const Real rho0 = g_rho0, p0 = g_p0, gam = g_gam;
  const Real cs = std::sqrt(p0 / rho0);
  const Real gm_ = g_gm, rref = g_r0;
  auto u0 = pmbp->phydro->u0;
  auto geom = pmbp->pcoord->shell_geom;

  Real linf_drho = 0.0, linf_dp = 0.0, linf_vrel = 0.0;
  Real l1_drho = 0.0, l1_dp = 0.0;
  int Ncell = (nmb1+1)*(ke-ks+1)*(je-js+1)*(ie-is+1);
  Kokkos::parallel_reduce("hse_r2_diag",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, Ncell),
  KOKKOS_LAMBDA(const int idx, Real &mdrho, Real &mdp, Real &mvrel,
                              Real &s_drho, Real &s_dp) {
    int nx1 = ie - is + 1, nx2 = je - js + 1, nx3 = ke - ks + 1;
    int nji = nx2 * nx1, nkji = nx3 * nx2 * nx1;
    int m = idx / nkji;
    int k = (idx - m*nkji) / nji;
    int j = (idx - m*nkji - k*nji) / nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks; j += js;
    Real r = geom.r_vol(m, i);
    Real rho_an = rho0 * Kokkos::exp((gm_/(cs*cs)) * (1.0/r - 1.0/rref));
    Real p_an = cs * cs * rho_an;
    Real rho = u0(m, IDN, k, j, i);
    Real ek = 0.5 * (u0(m,IM1,k,j,i)*u0(m,IM1,k,j,i)
                    +u0(m,IM2,k,j,i)*u0(m,IM2,k,j,i)
                    +u0(m,IM3,k,j,i)*u0(m,IM3,k,j,i)) / rho;
    Real p = (gam - 1.0) * (u0(m,IEN,k,j,i) - ek);
    Real vmag = Kokkos::sqrt(2.0 * ek / rho);
    Real drho_rel = Kokkos::fabs((rho - rho_an) / rho_an);
    Real dp_rel   = Kokkos::fabs((p - p_an) / p_an);
    Real vrel     = vmag / cs;
    if (drho_rel > mdrho) mdrho = drho_rel;
    if (dp_rel   > mdp)   mdp   = dp_rel;
    if (vrel     > mvrel) mvrel = vrel;
    s_drho += drho_rel;
    s_dp   += dp_rel;
  }, Kokkos::Max<Real>(linf_drho), Kokkos::Max<Real>(linf_dp),
     Kokkos::Max<Real>(linf_vrel), l1_drho, l1_dp);
  l1_drho /= static_cast<Real>(Ncell);
  l1_dp   /= static_cast<Real>(Ncell);

  std::cout.precision(6);
  std::cout << std::scientific
            << "[hydrostatic_r2] t           = " << pm->time << "\n"
            << "[hydrostatic_r2] cs_iso      = " << cs << "\n"
            << "[hydrostatic_r2] max|drho|/rho = " << linf_drho << "\n"
            << "[hydrostatic_r2] max|dp|/p     = " << linf_dp << "\n"
            << "[hydrostatic_r2] max|v|/cs     = " << linf_vrel << "\n"
            << "[hydrostatic_r2] L1 |drho|/rho = " << l1_drho << "\n"
            << "[hydrostatic_r2] L1 |dp|/p     = " << l1_dp
            << std::endl;
}

//----------------------------------------------------------------------------------------
// "solid_body_rotation": measure the one-step centrifugal acceleration from
// the spherical FV + geometric source path. Compare to the analytic predictions:
//   d(rho v_r) / dt   ~= rho * Omega^2 * r * sin^2(theta)
//   d(rho v_th) / dt  ~= rho * Omega^2 * r * sin(theta) * cos(theta)
//   d(rho v_phi) / dt ~= 0
// These are the centrifugal "force" terms in spherical components for
// solid-body rotation in absence of pressure gradient or gravity.

void SolidBodyRotationFinalize(ParameterInput *pin, Mesh *pm) {
  if (g_mode != Mode::kSolidBodyRotation) return;
  auto *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pmbp->nmb_thispack - 1;
  const Real rho0 = g_rho0;
  const Real Om = g_Omega;
  auto u0 = pmbp->phydro->u0;
  auto geom = pmbp->pcoord->shell_geom;
  const Real t = pm->time;
  if (t <= 0.0) {
    std::cout << "[solid_body_rotation] t=0; skipping diagnostic" << std::endl;
    return;
  }

  Real linf_r = 0.0, linf_t = 0.0, linf_p = 0.0;
  Real l1_r = 0.0, l1_t = 0.0;
  int Ncell = (nmb1+1)*(ke-ks+1)*(je-js+1)*(ie-is+1);
  Kokkos::parallel_reduce("sbr_diag",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, Ncell),
  KOKKOS_LAMBDA(const int idx,
                Real &mr, Real &mt, Real &mp,
                Real &sr, Real &st) {
    int nx1 = ie - is + 1, nx2 = je - js + 1, nx3 = ke - ks + 1;
    int nji = nx2 * nx1, nkji = nx3 * nx2 * nx1;
    int m = idx / nkji;
    int k = (idx - m*nkji) / nji;
    int j = (idx - m*nkji - k*nji) / nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks; j += js;
    Real r = geom.r_vol(m, i);
    Real th = geom.theta_vol(m, j);
    Real sint = Kokkos::sin(th), cost = Kokkos::cos(th);
    // Initial conditions: v_r=0, v_th=0, v_phi=Om*r*sin(th).
    Real m_r_init = 0.0;
    Real m_t_init = 0.0;
    Real m_p_init = rho0 * Om * r * sint;
    // Measured numerical acceleration (Eulerian).
    Real a_r = (u0(m, IM1, k, j, i) - m_r_init) / t;
    Real a_t = (u0(m, IM2, k, j, i) - m_t_init) / t;
    Real a_p = (u0(m, IM3, k, j, i) - m_p_init) / t;
    // Analytic predictions.
    Real a_r_an = rho0 * Om * Om * r * sint * sint;
    Real a_t_an = rho0 * Om * Om * r * sint * cost;
    Real a_p_an = 0.0;
    Real er = Kokkos::fabs(a_r - a_r_an);
    Real et = Kokkos::fabs(a_t - a_t_an);
    Real ep = Kokkos::fabs(a_p - a_p_an);
    if (er > mr) mr = er;
    if (et > mt) mt = et;
    if (ep > mp) mp = ep;
    sr += er;
    st += et;
  }, Kokkos::Max<Real>(linf_r), Kokkos::Max<Real>(linf_t),
     Kokkos::Max<Real>(linf_p), l1_r, l1_t);
  l1_r /= static_cast<Real>(Ncell);
  l1_t /= static_cast<Real>(Ncell);
  // Normalise by typical analytic magnitude rho0 * Om^2 * r_typ.
  Real r_typ = 0.5 * (pm->mesh_size.x1min + pm->mesh_size.x1max);
  Real scale = rho0 * Om * Om * r_typ;

  std::cout.precision(6);
  std::cout << std::scientific
            << "[solid_body_rotation] t              = " << t << "\n"
            << "[solid_body_rotation] Omega          = " << Om << "\n"
            << "[solid_body_rotation] scale          = " << scale << "\n"
            << "[solid_body_rotation] Linf a_r err   = " << linf_r
            << "   (rel " << (linf_r/scale) << ")\n"
            << "[solid_body_rotation] Linf a_th err  = " << linf_t
            << "   (rel " << (linf_t/scale) << ")\n"
            << "[solid_body_rotation] Linf a_phi err = " << linf_p
            << "   (rel " << (linf_p/scale) << ")\n"
            << "[solid_body_rotation] L1   a_r  err  = " << l1_r << "\n"
            << "[solid_body_rotation] L1   a_th err  = " << l1_t
            << std::endl;
}

//----------------------------------------------------------------------------------------
// "keplerian_orbit": measure total Lz drift, mass drift, and max |v_r|/v_phi.

void KeplerianOrbitFinalize(ParameterInput *pin, Mesh *pm) {
  if (g_mode != Mode::kKeplerianOrbit) return;
  auto *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pmbp->nmb_thispack - 1;
  auto u0 = pmbp->phydro->u0;
  auto geom = pmbp->pcoord->shell_geom;

  Real local_Lz = 0.0, local_M = 0.0;
  Real local_max_vr_over_vphi = 0.0;
  Kokkos::parallel_reduce("kep_orbit_diag",
    Kokkos::RangePolicy<>(DevExeSpace(),
                          0, (nmb1+1)*(ke-ks+1)*(je-js+1)*(ie-is+1)),
  KOKKOS_LAMBDA(const int idx, Real &Lz_, Real &M_, Real &vrat) {
    int nx1 = ie - is + 1, nx2 = je - js + 1, nx3 = ke - ks + 1;
    int nji = nx2 * nx1, nkji = nx3 * nx2 * nx1;
    int m = idx / nkji;
    int k = (idx - m*nkji) / nji;
    int j = (idx - m*nkji - k*nji) / nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks; j += js;
    Real r = geom.r_vol(m, i);
    Real th = geom.theta_vol(m, j);
    Real sint = Kokkos::sin(th);
    Real V = geom.dr3_third(m,i)*geom.dcos_theta(m,j)*geom.dphi(m,k);
    Real rho = u0(m, IDN, k, j, i);
    Real vr   = u0(m, IM1, k, j, i) / rho;
    Real vphi = u0(m, IM3, k, j, i) / rho;
    Lz_ += rho * (r * sint) * vphi * V;
    M_  += rho * V;
    Real ratio = (Kokkos::fabs(vphi) > 1e-30) ?
                 Kokkos::fabs(vr) / Kokkos::fabs(vphi) : 0.0;
    if (ratio > vrat) vrat = ratio;
  }, local_Lz, local_M, Kokkos::Max<Real>(local_max_vr_over_vphi));

  Real Lz0 = g_L_init, M0 = g_M_init;
  Real dLz = (Lz0 != 0.0) ? (local_Lz - Lz0) / Lz0 : (local_Lz - Lz0);
  Real dM  = (M0 != 0.0) ? (local_M - M0) / M0 : (local_M - M0);
  std::cout.precision(6);
  std::cout << std::scientific
            << "[keplerian_orbit] t                 = " << pm->time << "\n"
            << "[keplerian_orbit] Lz(t)             = " << local_Lz << "\n"
            << "[keplerian_orbit] Lz(0)             = " << Lz0 << "\n"
            << "[keplerian_orbit] (Lz-Lz0)/Lz0      = " << dLz << "\n"
            << "[keplerian_orbit] M(t)              = " << local_M << "\n"
            << "[keplerian_orbit] (M-M0)/M0         = " << dM << "\n"
            << "[keplerian_orbit] max|v_r|/|v_phi|  = " << local_max_vr_over_vphi
            << std::endl;
}

//----------------------------------------------------------------------------------------
// "thin_disk": coarse vibe diagnostic -- track mass-weighted r and theta of
// the disk and total Lz/Mass for short runs.

void ThinDiskFinalize(ParameterInput *pin, Mesh *pm) {
  if (g_mode != Mode::kThinDisk) return;
  auto *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pmbp->nmb_thispack - 1;
  auto u0 = pmbp->phydro->u0;
  auto geom = pmbp->pcoord->shell_geom;

  Real mwr_num = 0.0, mwth_num = 0.0, mw_den = 0.0, totLz = 0.0;
  Kokkos::parallel_reduce("disk_diag",
    Kokkos::RangePolicy<>(DevExeSpace(),
                          0, (nmb1+1)*(ke-ks+1)*(je-js+1)*(ie-is+1)),
  KOKKOS_LAMBDA(const int idx, Real &nr_, Real &nth_, Real &dn_, Real &Lz_) {
    int nx1 = ie - is + 1, nx2 = je - js + 1, nx3 = ke - ks + 1;
    int nji = nx2 * nx1, nkji = nx3 * nx2 * nx1;
    int m = idx / nkji;
    int k = (idx - m*nkji) / nji;
    int j = (idx - m*nkji - k*nji) / nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks; j += js;
    Real r = geom.r_vol(m, i);
    Real th = geom.theta_vol(m, j);
    Real sint = Kokkos::sin(th);
    Real V = geom.dr3_third(m,i)*geom.dcos_theta(m,j)*geom.dphi(m,k);
    Real rho = u0(m, IDN, k, j, i);
    Real vphi = u0(m, IM3, k, j, i) / rho;
    nr_  += rho * r  * V;
    nth_ += rho * th * V;
    dn_  += rho      * V;
    Lz_  += rho * (r * sint) * vphi * V;
  }, mwr_num, mwth_num, mw_den, totLz);
  Real rbar  = (mw_den != 0.0) ? (mwr_num  / mw_den) : 0.0;
  Real thbar = (mw_den != 0.0) ? (mwth_num / mw_den) : 0.0;
  std::cout.precision(6);
  std::cout << std::scientific
            << "[thin_disk] t          = " << pm->time << "\n"
            << "[thin_disk] mass       = " << mw_den << "\n"
            << "[thin_disk] <r>_mass   = " << rbar
            << "   (target r_disk=" << g_disk_rdisk << ")\n"
            << "[thin_disk] <theta>_m  = " << thbar
            << "   (target pi/2 = " << M_PI*0.5 << ")\n"
            << "[thin_disk] total Lz   = " << totLz
            << std::endl;
}

//----------------------------------------------------------------------------------------
// Parker user radial BC: at the inner radial boundary clamp ghost zones to the
// analytic Parker profile. Outer radial boundary uses zero-gradient outflow.

void ParkerRadialBCs(Mesh *pm) {
  auto *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  const int ng = indcs.ng;
  const int is = indcs.is, ie = indcs.ie;
  const int nx1 = indcs.nx1;
  const int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  const int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
  const int nmb = pmbp->nmb_thispack;
  auto &mb_bcs = pmbp->pmb->mb_bcs;
  auto &mb_size = pmbp->pmb->mb_size;
  auto u0 = pmbp->phydro->u0;
  auto geom = pmbp->pcoord->shell_geom;
  auto &eos = pmbp->phydro->peos->eos_data;
  const Real gam = eos.gamma;
  const Real cs    = g_parker_csiso;
  const Real rc    = g_parker_rc;
  const Real r_ref = g_parker_r_inner;
  const Real rho_ref = g_parker_rho_inner;
  const Real M_ref = g_parker_M_inner;

  par_for("parker_radial_bc", DevExeSpace(),
          0, nmb-1, 0, n3-1, 0, n2-1,
  KOKKOS_LAMBDA(int m, int k, int j) {
    if (mb_bcs.d_view(m, BoundaryFace::inner_x1) == BoundaryFlag::user) {
      // Inner ghosts: analytic Parker profile from cell-centred geometry.
      for (int gi = 0; gi < ng; ++gi) {
        int igh = is - 1 - gi;
        Real r = geom.r_vol(m, igh);
        Real M = ParkerMachAtR(r, rc);
        Real vr  = M * cs;
        Real rho = rho_ref * (M_ref / M) * (r_ref / r) * (r_ref / r);
        Real p   = cs * cs * rho;
        u0(m, IDN, k, j, igh) = rho;
        u0(m, IM1, k, j, igh) = rho * vr;
        u0(m, IM2, k, j, igh) = 0.0;
        u0(m, IM3, k, j, igh) = 0.0;
        u0(m, IEN, k, j, igh) = p / (gam - 1.0) + 0.5 * rho * vr * vr;
      }
    }
    if (mb_bcs.d_view(m, BoundaryFace::outer_x1) == BoundaryFlag::user) {
      // Outer ghosts: simple zero-gradient outflow copied from the last active.
      for (int gi = 0; gi < ng; ++gi) {
        int igh = ie + 1 + gi;
        u0(m, IDN, k, j, igh) = u0(m, IDN, k, j, ie);
        u0(m, IM1, k, j, igh) = u0(m, IM1, k, j, ie);
        u0(m, IM2, k, j, igh) = u0(m, IM2, k, j, ie);
        u0(m, IM3, k, j, igh) = u0(m, IM3, k, j, ie);
        u0(m, IEN, k, j, igh) = u0(m, IEN, k, j, ie);
      }
    }
  });
}

//----------------------------------------------------------------------------------------
// "parker_isothermal": compare numerical state to the analytic Parker solution
// over the active mesh; report L1/L2/Linf relative errors in v_r and rho, plus
// angular variation at fixed r (max angular spread of rho and v_r divided by
// the cell's analytic value).

void ParkerIsothermalFinalize(ParameterInput *pin, Mesh *pm) {
  if (g_mode != Mode::kParkerIsothermal) return;
  auto *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pmbp->nmb_thispack - 1;
  auto u0 = pmbp->phydro->u0;
  auto geom = pmbp->pcoord->shell_geom;
  const Real cs    = g_parker_csiso;
  const Real rc    = g_parker_rc;
  const Real r_ref = g_parker_r_inner;
  const Real rho_ref = g_parker_rho_inner;
  const Real M_ref = g_parker_M_inner;
  int Ncell = (nmb1+1)*(ke-ks+1)*(je-js+1)*(ie-is+1);

  Real linf_v = 0.0, linf_rho = 0.0;
  Real l1_v = 0.0, l2_v = 0.0;
  Real l1_rho = 0.0, l2_rho = 0.0;
  Kokkos::parallel_reduce("parker_diag",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, Ncell),
  KOKKOS_LAMBDA(const int idx,
                Real &lv, Real &lr, Real &mxv, Real &mxr,
                Real &sv, Real &sr) {
    int nx1 = ie - is + 1, nx2 = je - js + 1, nx3 = ke - ks + 1;
    int nji = nx2 * nx1, nkji = nx3 * nx2 * nx1;
    int m = idx / nkji;
    int k = (idx - m*nkji) / nji;
    int j = (idx - m*nkji - k*nji) / nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks; j += js;
    Real r = geom.r_vol(m, i);
    Real M = ParkerMachAtR(r, rc);
    Real vr_an  = M * cs;
    Real rho_an = rho_ref * (M_ref / M) * (r_ref / r) * (r_ref / r);
    Real rho = u0(m, IDN, k, j, i);
    Real vr  = u0(m, IM1, k, j, i) / rho;
    Real ev = Kokkos::fabs((vr  - vr_an)  / vr_an);
    Real er = Kokkos::fabs((rho - rho_an) / rho_an);
    if (ev > mxv) mxv = ev;
    if (er > mxr) mxr = er;
    lv += ev;
    lr += er;
    sv += ev * ev;
    sr += er * er;
  }, l1_v, l1_rho, Kokkos::Max<Real>(linf_v),
     Kokkos::Max<Real>(linf_rho), l2_v, l2_rho);
  l1_v /= static_cast<Real>(Ncell);
  l1_rho /= static_cast<Real>(Ncell);
  l2_v = std::sqrt(l2_v / static_cast<Real>(Ncell));
  l2_rho = std::sqrt(l2_rho / static_cast<Real>(Ncell));

  std::cout.precision(6);
  std::cout << std::scientific
            << "[parker_isothermal] t           = " << pm->time << "\n"
            << "[parker_isothermal] cs_iso      = " << cs << "\n"
            << "[parker_isothermal] r_c         = " << rc << "\n"
            << "[parker_isothermal] M(r_inner)  = " << M_ref << "\n"
            << "[parker_isothermal] L1 |dv_r/v_r|  = " << l1_v << "\n"
            << "[parker_isothermal] L2 |dv_r/v_r|  = " << l2_v << "\n"
            << "[parker_isothermal] Linf|dv_r/v_r| = " << linf_v << "\n"
            << "[parker_isothermal] L1 |drho/rho|  = " << l1_rho << "\n"
            << "[parker_isothermal] L2 |drho/rho|  = " << l2_rho << "\n"
            << "[parker_isothermal] Linf|drho/rho| = " << linf_rho
            << std::endl;
}

}  // namespace
