//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file spherical_shell.cpp
//! \brief Allocates/populates spherical-shell geometry Views and provides a sanity check.
//! Geometry formulas implemented exactly per Athena++ spherical_polar.cpp (see srcpp).

#include <cmath>
#include <iostream>
#include <iomanip>
#include <limits>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "mesh/meshblock_pack.hpp"
#include "coordinates/cell_locations.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/spherical_shell.hpp"
#include "eos/eos.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

//----------------------------------------------------------------------------------------
// File-static registry for a process-global user-defined radial grid function.
// This is a deliberate compromise: AthenaK constructs Coordinates BEFORE the
// ProblemGenerator, so the pgen cannot register the hook through UserProblem.
// Instead the user code registers via a static initializer that runs before
// main(). One mapping per process; this fork supports a single spherical_shell.

namespace {
UserRadialGridFnPtr g_user_radial_grid_fn = nullptr;
}

void SetUserRadialGridFunc(UserRadialGridFnPtr fn) {
  g_user_radial_grid_fn = fn;
}

UserRadialGridFnPtr GetUserRadialGridFunc() {
  return g_user_radial_grid_fn;
}

//----------------------------------------------------------------------------------------
//! \fn ConstructSphericalShellGeometry
//! \brief Build per-MeshBlock 1D arrays of all spherical FV factors. Allocation happens
//! once at construction time; population is done with Kokkos device kernels so the data
//! lives on device for hot-path access.
//!
//! The radial face positions are computed via `grid_type`:
//!   uniform   : r_face linear in the logical x1 coordinate (as in prior tasks)
//!   log       : r_face uniform in ln(r) over [mesh.x1min, mesh.x1max]
//!   power_law : r_face = r_min + (r_max - r_min) * xi_norm^grid_alpha
//!   user      : r_face from the user-registered host function
//!               (GetUserRadialGridFunc()); evaluated on host, then deep-copied
//!               to device.
//! Theta and phi are always uniform in their logical coordinates.

void ConstructSphericalShellGeometry(MeshBlockPack *ppack,
                                     SphericalShellGeom &geom,
                                     RadialGridType grid_type,
                                     Real grid_alpha) {
  auto &indcs = ppack->pmesh->mb_indcs;
  const int ng  = indcs.ng;
  const int nx1 = indcs.nx1;
  const int nx2 = indcs.nx2;
  const int nx3 = indcs.nx3;
  const int nc1 = nx1 + 2*ng;
  const int nc2 = (nx2 > 1) ? (nx2 + 2*ng) : 1;
  const int nc3 = (nx3 > 1) ? (nx3 + 2*ng) : 1;
  const int nmb = ppack->nmb_thispack;

  geom.nmb = nmb;
  geom.nc1 = nc1;
  geom.nc2 = nc2;
  geom.nc3 = nc3;
  geom.ng  = ng;

  // Allocate device Views (allocation only; population follows below)
  Kokkos::realloc(geom.r_face,         nmb, nc1+1);
  Kokkos::realloc(geom.r_vol,          nmb, nc1);
  Kokkos::realloc(geom.dr,             nmb, nc1);
  Kokkos::realloc(geom.r2_face,        nmb, nc1+1);
  Kokkos::realloc(geom.dr2_half,       nmb, nc1);
  Kokkos::realloc(geom.dr3_third,      nmb, nc1);
  Kokkos::realloc(geom.inv_dr3_third,  nmb, nc1);
  Kokkos::realloc(geom.coord_src1_i,   nmb, nc1);
  Kokkos::realloc(geom.coord_src2_i,   nmb, nc1);

  Kokkos::realloc(geom.theta_face,     nmb, nc2+1);
  Kokkos::realloc(geom.theta_vol,      nmb, nc2);
  Kokkos::realloc(geom.dtheta,         nmb, nc2);
  Kokkos::realloc(geom.sin_theta_face, nmb, nc2+1);
  Kokkos::realloc(geom.sin_theta_vol,  nmb, nc2);
  Kokkos::realloc(geom.dcos_theta,     nmb, nc2);
  Kokkos::realloc(geom.coord_src1_j,   nmb, nc2);
  Kokkos::realloc(geom.coord_src2_j,   nmb, nc2);

  Kokkos::realloc(geom.phi_face,       nmb, nc3+1);
  Kokkos::realloc(geom.phi_center,     nmb, nc3);
  Kokkos::realloc(geom.dphi,           nmb, nc3);

  auto &mbsize = ppack->pmb->mb_size;
  const int nmb1 = nmb - 1;
  const bool multi_d = ppack->pmesh->multi_d;
  const bool three_d = ppack->pmesh->three_d;

  // Capture views for kernels.
  auto r_face  = geom.r_face;
  auto r_vol   = geom.r_vol;
  auto dr      = geom.dr;
  auto r2_face = geom.r2_face;
  auto dr2_h   = geom.dr2_half;
  auto dr3_t   = geom.dr3_third;
  auto inv_v_i = geom.inv_dr3_third;

  // ---- r-direction: faces, then cell-averaged quantities ----
  //
  // The MeshBlock's (x1min, x1max) is linearly partitioned from the global mesh
  // range (mesh.x1min, mesh.x1max). For nonuniform radial grids we keep that
  // logical partition and remap the logical face position xi to a physical
  // radius r. This produces globally consistent face radii across MeshBlocks
  // because every block sees the same global mapping.
  const Real mesh_x1min = ppack->pmesh->mesh_size.x1min;
  const Real mesh_x1max = ppack->pmesh->mesh_size.x1max;
  const Real mesh_x1len = mesh_x1max - mesh_x1min;
  if (grid_type == RadialGridType::uniform) {
    par_for("sph_geom_rfaces_uniform", DevExeSpace(), 0, nmb1, 0, nc1,
    KOKKOS_LAMBDA(int m, int i) {
      Real x1min = mbsize.d_view(m).x1min;
      Real x1max = mbsize.d_view(m).x1max;
      Real rf = LeftEdgeX(i - ng, nx1, x1min, x1max);
      r_face(m, i)  = rf;
      r2_face(m, i) = rf*rf;
    });
  } else if (grid_type == RadialGridType::log) {
    // r = r_min * (r_max/r_min)^xi_norm,   xi_norm = (xi - x1min)/(x1max - x1min)
    const Real log_ratio = Kokkos::log(mesh_x1max / mesh_x1min);
    par_for("sph_geom_rfaces_log", DevExeSpace(), 0, nmb1, 0, nc1,
    KOKKOS_LAMBDA(int m, int i) {
      Real x1min = mbsize.d_view(m).x1min;
      Real x1max = mbsize.d_view(m).x1max;
      Real xi = LeftEdgeX(i - ng, nx1, x1min, x1max);
      Real xi_norm = (xi - mesh_x1min) / mesh_x1len;
      Real rf = mesh_x1min * Kokkos::exp(xi_norm * log_ratio);
      r_face(m, i)  = rf;
      r2_face(m, i) = rf * rf;
    });
  } else if (grid_type == RadialGridType::power_law) {
    // r = r_min + (r_max - r_min) * xi_norm^alpha
    // alpha = 1.0 reduces to uniform; alpha > 1 packs cells near r_min;
    // alpha < 1 packs cells near r_max.
    const Real alpha = grid_alpha;
    par_for("sph_geom_rfaces_powerlaw", DevExeSpace(), 0, nmb1, 0, nc1,
    KOKKOS_LAMBDA(int m, int i) {
      Real x1min = mbsize.d_view(m).x1min;
      Real x1max = mbsize.d_view(m).x1max;
      Real xi = LeftEdgeX(i - ng, nx1, x1min, x1max);
      Real xi_norm = (xi - mesh_x1min) / mesh_x1len;
      Real rf;
      // pow on negative xi_norm (ghosts below mesh.x1min) is undefined for
      // fractional alpha; fall back to linear extrapolation outside [0,1].
      if (xi_norm >= 0.0 && xi_norm <= 1.0) {
        rf = mesh_x1min + (mesh_x1max - mesh_x1min) * Kokkos::pow(xi_norm, alpha);
      } else {
        rf = mesh_x1min + (mesh_x1max - mesh_x1min) * xi_norm;
      }
      r_face(m, i)  = rf;
      r2_face(m, i) = rf * rf;
    });
  } else if (grid_type == RadialGridType::user) {
    UserRadialGridFnPtr ufn = GetUserRadialGridFunc();
    if (ufn == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line "
                << __LINE__ << std::endl
                << "radial_grid=user but no user function was registered "
                << "via SetUserRadialGridFunc()" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    // Host-evaluate face radii (xi_norm computed on host) into a mirror View,
    // then deep_copy to device. This is one-time setup, not a hot path.
    auto h_r_face  = Kokkos::create_mirror_view(r_face);
    auto h_r2_face = Kokkos::create_mirror_view(r2_face);
    auto h_mbsize  = Kokkos::create_mirror_view(mbsize.d_view);
    Kokkos::deep_copy(h_mbsize, mbsize.d_view);
    for (int m = 0; m < nmb; ++m) {
      Real x1min = h_mbsize(m).x1min;
      Real x1max = h_mbsize(m).x1max;
      for (int i = 0; i <= nc1; ++i) {
        Real xi = LeftEdgeX(i - ng, nx1, x1min, x1max);
        Real xi_norm = (xi - mesh_x1min) / mesh_x1len;
        // Inside the active range we trust the user function; outside (ghosts)
        // we linearly extrapolate to avoid undefined behaviour from the user
        // mapping at xi_norm < 0 or > 1.
        Real rf;
        if (xi_norm >= 0.0 && xi_norm <= 1.0) {
          rf = ufn(xi_norm);
        } else {
          // Match slope at the nearest endpoint.
          Real eps = 1.0e-6;
          if (xi_norm < 0.0) {
            Real r0 = ufn(0.0);
            Real r1 = ufn(eps);
            Real slope = (r1 - r0) / eps;
            rf = r0 + slope * xi_norm;
          } else {
            Real r0 = ufn(1.0 - eps);
            Real r1 = ufn(1.0);
            Real slope = (r1 - r0) / eps;
            rf = r1 + slope * (xi_norm - 1.0);
          }
        }
        h_r_face(m, i) = rf;
        h_r2_face(m, i) = rf * rf;
      }
    }
    Kokkos::deep_copy(r_face,  h_r_face);
    Kokkos::deep_copy(r2_face, h_r2_face);
  }

  auto src1_i = geom.coord_src1_i;
  auto src2_i = geom.coord_src2_i;

  par_for("sph_geom_rcells", DevExeSpace(), 0, nmb1, 0, nc1-1,
  KOKKOS_LAMBDA(int m, int i) {
    Real rm = r_face(m, i);
    Real rp = r_face(m, i+1);
    Real rm2 = rm*rm, rp2 = rp*rp;
    Real rm3 = rm2*rm, rp3 = rp2*rp;
    Real drv = rp - rm;
    Real dr2h = 0.5*(rp2 - rm2);
    Real v_i = (1.0/3.0)*(rp3 - rm3);
    dr(m, i)     = drv;
    dr2_h(m, i)  = dr2h;
    dr3_t(m, i)  = v_i;
    inv_v_i(m, i)= 1.0 / v_i;
    // Volume-averaged r: <r> = (\int r dV)/(\int dV) within the cell.
    // For a spherical shell d V = r^2 dr d Omega, so
    //   <r> = (3/4) (rp^4 - rm^4) / (rp^3 - rm^3)
    // (Mignone 2014 eq. 17). This reduces to (rp+rm)/2 in the small-dr limit.
    Real num = 0.75 * (rp2*rp2 - rm2*rm2);
    Real den = (rp3 - rm3);
    r_vol(m, i) = num / den;
    // Athena++ source-term factors (see srcpp/coordinates/spherical_polar.cpp).
    //   src1_i = dr2_half / vol_i  -- area2/vol, "<2/r>/2"-like
    //   src2_i = dr / ((rm+rp) * vol_i)  -- normalises flux-area-weighted radial
    //   fluxes for the (-rho v_r v_t / r) and (-rho v_r v_p / r) source terms.
    src1_i(m, i) = dr2h / v_i;
    src2_i(m, i) = drv  / ((rm + rp) * v_i);
  });

  // ---- theta-direction ----
  auto theta_face     = geom.theta_face;
  auto theta_vol      = geom.theta_vol;
  auto dtheta         = geom.dtheta;
  auto sin_theta_face = geom.sin_theta_face;
  auto sin_theta_vol  = geom.sin_theta_vol;
  auto dcos_theta     = geom.dcos_theta;

  auto src1_j = geom.coord_src1_j;
  auto src2_j = geom.coord_src2_j;

  if (multi_d) {
    par_for("sph_geom_thfaces", DevExeSpace(), 0, nmb1, 0, nc2,
    KOKKOS_LAMBDA(int m, int j) {
      Real x2min = mbsize.d_view(m).x2min;
      Real x2max = mbsize.d_view(m).x2max;
      Real tf = LeftEdgeX(j - ng, nx2, x2min, x2max);
      theta_face(m, j)     = tf;
      sin_theta_face(m, j) = Kokkos::sin(tf);
    });

    par_for("sph_geom_thcells", DevExeSpace(), 0, nmb1, 0, nc2-1,
    KOKKOS_LAMBDA(int m, int j) {
      Real tm = theta_face(m, j);
      Real tp = theta_face(m, j+1);
      Real cm = Kokkos::cos(tm);
      Real cp = Kokkos::cos(tp);
      Real sm = sin_theta_face(m, j);
      Real sp = sin_theta_face(m, j+1);
      Real dcos = cm - cp; // > 0 for theta in (0,pi) with tp>tm
      dtheta(m, j)     = tp - tm;
      dcos_theta(m, j) = dcos;
      // Volume-weighted <theta> = (\int sin(theta) theta dV)/(\int dV)
      //                        = ((sin t - t cos t) eval'd) / (cos tm - cos tp)
      Real num = (sp - tp*cp) - (sm - tm*cm);
      theta_vol(m, j)    = num / dcos;
      sin_theta_vol(m, j)= Kokkos::sin(theta_vol(m, j));
      // Athena++ source-term factors (see srcpp/coordinates/spherical_polar.cpp).
      //   src1_j = (sp - sm) / dcos -- approximates <cot theta>
      //   src2_j = (sp - sm) / ((sm + sp) * dcos) -- weights theta-flux pairs
      //            for the (-rho v_t v_p cot/r) source term on phi momentum.
      src1_j(m, j) = (sp - sm) / dcos;
      src2_j(m, j) = (sp - sm) / ((sm + sp) * dcos);
    });
  } else {
    // 1D: theta range is the full [x2min, x2max] of the mesh, but only one cell.
    par_for("sph_geom_thfaces_1d", DevExeSpace(), 0, nmb1,
    KOKKOS_LAMBDA(int m) {
      Real tm = mbsize.d_view(m).x2min;
      Real tp = mbsize.d_view(m).x2max;
      theta_face(m, 0)     = tm;
      theta_face(m, 1)     = tp;
      Real sm = Kokkos::sin(tm), sp = Kokkos::sin(tp);
      Real cm = Kokkos::cos(tm), cp = Kokkos::cos(tp);
      Real dcos = cm - cp;
      sin_theta_face(m, 0) = sm;
      sin_theta_face(m, 1) = sp;
      dtheta(m, 0)         = tp - tm;
      dcos_theta(m, 0)     = dcos;
      Real num = (sp - tp*cp) - (sm - tm*cm);
      theta_vol(m, 0)      = num / dcos;
      sin_theta_vol(m, 0)  = Kokkos::sin(theta_vol(m, 0));
      src1_j(m, 0) = (sp - sm) / dcos;
      src2_j(m, 0) = (sp - sm) / ((sm + sp) * dcos);
    });
  }

  // ---- phi-direction ----
  auto phi_face   = geom.phi_face;
  auto phi_center = geom.phi_center;
  auto dphi       = geom.dphi;

  if (three_d) {
    par_for("sph_geom_phifaces", DevExeSpace(), 0, nmb1, 0, nc3,
    KOKKOS_LAMBDA(int m, int k) {
      Real x3min = mbsize.d_view(m).x3min;
      Real x3max = mbsize.d_view(m).x3max;
      phi_face(m, k) = LeftEdgeX(k - ng, nx3, x3min, x3max);
    });
    par_for("sph_geom_phicells", DevExeSpace(), 0, nmb1, 0, nc3-1,
    KOKKOS_LAMBDA(int m, int k) {
      Real pm = phi_face(m, k);
      Real pp = phi_face(m, k+1);
      dphi(m, k)       = pp - pm;
      phi_center(m, k) = 0.5*(pm + pp);
    });
  } else {
    par_for("sph_geom_phifaces_lo", DevExeSpace(), 0, nmb1,
    KOKKOS_LAMBDA(int m) {
      Real pm = mbsize.d_view(m).x3min;
      Real pp = mbsize.d_view(m).x3max;
      phi_face(m, 0)   = pm;
      phi_face(m, 1)   = pp;
      dphi(m, 0)       = pp - pm;
      phi_center(m, 0) = 0.5*(pm + pp);
    });
  }
}

//----------------------------------------------------------------------------------------
//! \fn VerifySphericalShellVolume
//! \brief Reduce-summed cell volumes over active cells and compare against the analytic
//! spherical-wedge volume V = (1/3)(r1^3 - r0^3)(cos t0 - cos t1)(p1 - p0).
//! Performed on device via Kokkos parallel_reduce, MPI-summed if parallel.

Real VerifySphericalShellVolume(MeshBlockPack *ppack,
                                const SphericalShellGeom &geom,
                                bool verbose) {
  auto &indcs = ppack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = ppack->nmb_thispack - 1;

  auto dr3_t      = geom.dr3_third;
  auto dcos_theta = geom.dcos_theta;
  auto dphi       = geom.dphi;

  Real local_vol = 0.0;
  Kokkos::parallel_reduce("sph_volume_check",
    Kokkos::RangePolicy<>(DevExeSpace(),
                          0,
                          (nmb1+1)*(ke-ks+1)*(je-js+1)*(ie-is+1)),
  KOKKOS_LAMBDA(const int &idx, Real &lsum) {
    int nx1 = ie - is + 1;
    int nx2 = je - js + 1;
    int nx3 = ke - ks + 1;
    int nji = nx2 * nx1;
    int nkji = nx3 * nx2 * nx1;
    int m = idx / nkji;
    int k = (idx - m*nkji) / nji;
    int j = (idx - m*nkji - k*nji) / nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;
    lsum += dr3_t(m, i) * dcos_theta(m, j) * dphi(m, k);
  }, local_vol);

  Real total_vol = local_vol;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(&local_vol, &total_vol, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif

  // Analytic wedge from mesh_size.
  auto &ms = ppack->pmesh->mesh_size;
  const Real r0 = ms.x1min, r1 = ms.x1max;
  const Real t0 = ms.x2min, t1 = ms.x2max;
  const Real p0 = ms.x3min, p1 = ms.x3max;
  const Real V_an = (1.0/3.0) * (r1*r1*r1 - r0*r0*r0)
                              * (std::cos(t0) - std::cos(t1))
                              * (p1 - p0);

  Real rel_err = std::abs(total_vol - V_an) / std::abs(V_an);
  if (verbose && global_variable::my_rank == 0) {
    std::cout << std::scientific << std::setprecision(6)
              << "[spherical_shell] geometry sanity check:\n"
              << "    sum(cell volumes) = " << total_vol << "\n"
              << "    analytic wedge V  = " << V_an     << "\n"
              << "    relative error    = " << rel_err  << std::endl;
  }
  return rel_err;
}

//----------------------------------------------------------------------------------------
//! \fn Coordinates::AddSphericalShellHydroSrcTerms
//! \brief Apply Newtonian hydro geometric source terms in spherical-shell coordinates.
//! Mirrors Athena++ srcpp/coordinates/spherical_polar.cpp::AddCoordTermsDivergence for
//! HYDRO ONLY (no MHD, no STS, no diffusion). Combines the "centripetal/curvature"
//! source terms (from primitives) with the flux-area-weighted advective source terms
//! (from radial and theta Riemann fluxes). Must be called after the explicit RK update
//! and before c2p, with the SAME fluxes that produced the divergence in this stage.
//!
//! Implemented terms (orthonormal spherical components, ideal-gas hydro):
//!   IM1: + dt * coord_src1_i * [ rho*(v_t^2 + v_p^2) + 2*p ]
//!   IM2: - dt * coord_src2_i * (r-^2 F1[IM2,i] + r+^2 F1[IM2,i+1])
//!        + dt * coord_src1_i * coord_src1_j * [ rho*v_p^2 + p ]
//!   IM3: - dt * coord_src2_i * (r-^2 F1[IM3,i] + r+^2 F1[IM3,i+1])
//!        - dt * coord_src1_i * coord_src2_j * (sin- F2[IM3,j] + sin+ F2[IM3,j+1])  (multi-d)
//!        - dt * coord_src1_i * coord_src1_j * rho*v_t*v_p                            (1-d theta)
//! Energy IEN has no geometric source term in this conservative split.

void Coordinates::AddSphericalShellHydroSrcTerms(
    const DvceArray5D<Real> &w0,
    const DvceArray5D<Real> &flx1,
    const DvceArray5D<Real> &flx2,
    const EOS_Data &eos,
    const Real dt,
    DvceArray5D<Real> &u0) {
  if (coord_system != CoordinateSystem::spherical_shell) return;

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pmy_pack->nmb_thispack - 1;
  const bool multi_d = pmy_pack->pmesh->multi_d;
  const bool is_ideal = eos.is_ideal;
  const Real iso_cs = eos.iso_cs;
  auto geom = shell_geom;

  par_for("sph_hydro_geom_src", DevExeSpace(),
          0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real rho = w0(m, IDN, k, j, i);
    const Real vr  = w0(m, IM1, k, j, i);
    const Real vt  = w0(m, IM2, k, j, i);
    const Real vp  = w0(m, IM3, k, j, i);
    const Real pgas = is_ideal ? eos.IdealGasPressure(w0(m, IEN, k, j, i))
                               : (iso_cs * iso_cs) * rho;

    const Real src1_i = geom.coord_src1_i(m, i);
    const Real src2_i = geom.coord_src2_i(m, i);
    const Real src1_j = geom.coord_src1_j(m, j);
    const Real src2_j = geom.coord_src2_j(m, j);
    const Real r2m = geom.r2_face(m, i);
    const Real r2p = geom.r2_face(m, i+1);

    // (i) Radial centripetal + pressure-curvature source on m_r.
    //     S(m_r) = <1/r>_FV * [ rho (v_t^2 + v_p^2) + 2 p ]
    Real m_ii = rho * (vt*vt + vp*vp) + 2.0 * pgas;
    u0(m, IM1, k, j, i) += dt * src1_i * m_ii;

    // (ii) Curvature contribution from radial fluxes for theta and phi momentum:
    //      S(m_t) -= <(rho v_r v_t)/r>  ~= src2_i * (r-^2 F1[IM2,i] + r+^2 F1[IM2,i+1])
    //      S(m_p) -= <(rho v_r v_p)/r>  similarly
    u0(m, IM2, k, j, i) -= dt * src2_i *
                          (r2m * flx1(m, IM2, k, j, i)
                         + r2p * flx1(m, IM2, k, j, i+1));
    u0(m, IM3, k, j, i) -= dt * src2_i *
                          (r2m * flx1(m, IM3, k, j, i)
                         + r2p * flx1(m, IM3, k, j, i+1));

    // (iii) Cotangent + pressure-curvature source on m_t (theta momentum):
    //      S(m_t) += <cot(theta)/r> * [ rho v_p^2 + p ]
    Real m_pp = rho * vp * vp + pgas;
    u0(m, IM2, k, j, i) += dt * src1_i * src1_j * m_pp;

    // (iv) Theta-direction advection of phi momentum:
    //      S(m_p) -= <cot(theta)/r * (rho v_t v_p)>
    //   In multi-D theta this is computed from the F2[IM3] flux pair (consistent
    //   with the FV scheme); in 1-D theta no flux exists so we use primitives.
    if (multi_d) {
      const Real sm = geom.sin_theta_face(m, j);
      const Real sp = geom.sin_theta_face(m, j+1);
      u0(m, IM3, k, j, i) -= dt * src1_i * src2_j *
                            (sm * flx2(m, IM3, k, j,   i)
                           + sp * flx2(m, IM3, k, j+1, i));
    } else {
      u0(m, IM3, k, j, i) -= dt * src1_i * src1_j * (rho * vt * vp);
    }
  });
}

//----------------------------------------------------------------------------------------
//! \fn Coordinates::AddSphericalShellMHDSrcTerms
//! \brief Newtonian MHD geometric source terms for spherical-shell coordinates.
//! Identical in structure to the hydro version but with magnetic-stress contributions
//! from the cell-centred field bcc (orthonormal spherical components). The radial- and
//! theta-flux pairings use the FULL MHD Riemann fluxes flx1/flx2 (which already include
//! the -B_i B_j magnetic-stress pieces), so no extra B-only flux terms are needed in
//! the pairings. Energy IEN has no geometric source.
//!
//! Mirrors Athena++ srcpp/coordinates/spherical_polar.cpp::AddCoordTermsDivergence
//! MHD path (NON_BAROTROPIC + MAGNETIC_FIELDS_ENABLED branches).
//!
//! Indexing: bcc(m, IBX/IBY/IBZ, k, j, i) holds B_r, B_theta, B_phi at cell centres.

void Coordinates::AddSphericalShellMHDSrcTerms(
    const DvceArray5D<Real> &w0,
    const DvceArray5D<Real> &bcc,
    const DvceArray5D<Real> &flx1,
    const DvceArray5D<Real> &flx2,
    const EOS_Data &eos,
    const Real dt,
    DvceArray5D<Real> &u0) {
  if (coord_system != CoordinateSystem::spherical_shell) return;

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pmy_pack->nmb_thispack - 1;
  const bool multi_d = pmy_pack->pmesh->multi_d;
  const bool is_ideal = eos.is_ideal;
  const Real iso_cs = eos.iso_cs;
  auto geom = shell_geom;

  par_for("sph_mhd_geom_src", DevExeSpace(),
          0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real rho = w0(m, IDN, k, j, i);
    const Real vr  = w0(m, IM1, k, j, i);
    const Real vt  = w0(m, IM2, k, j, i);
    const Real vp  = w0(m, IM3, k, j, i);
    const Real pgas = is_ideal ? eos.IdealGasPressure(w0(m, IEN, k, j, i))
                               : (iso_cs * iso_cs) * rho;
    const Real br  = bcc(m, IBX, k, j, i);
    const Real bt  = bcc(m, IBY, k, j, i);
    const Real bp  = bcc(m, IBZ, k, j, i);

    const Real src1_i = geom.coord_src1_i(m, i);
    const Real src2_i = geom.coord_src2_i(m, i);
    const Real src1_j = geom.coord_src1_j(m, j);
    const Real src2_j = geom.coord_src2_j(m, j);
    const Real r2m = geom.r2_face(m, i);
    const Real r2p = geom.r2_face(m, i+1);

    // (i) Radial centripetal + pressure-curvature source on m_r.
    //   Hydro:  rho (v_t^2 + v_p^2) + 2 p
    //   MHD +=  B_r^2                                  (Athena++)
    Real m_ii = rho * (vt*vt + vp*vp) + 2.0 * pgas + br*br;
    u0(m, IM1, k, j, i) += dt * src1_i * m_ii;

    // (ii) Radial-flux pairings for theta and phi momentum.
    //   MHD Riemann fluxes include -B_r B_t / -B_r B_p stress contributions,
    //   so the pairing form is identical to the hydro form.
    u0(m, IM2, k, j, i) -= dt * src2_i *
                          (r2m * flx1(m, IM2, k, j, i)
                         + r2p * flx1(m, IM2, k, j, i+1));
    u0(m, IM3, k, j, i) -= dt * src2_i *
                          (r2m * flx1(m, IM3, k, j, i)
                         + r2p * flx1(m, IM3, k, j, i+1));

    // (iii) Cotangent + pressure-curvature source on m_t.
    //   Hydro:  rho v_p^2 + p
    //   MHD +=  0.5 ( B_r^2 + B_t^2 - B_p^2 )           (Athena++)
    Real m_pp = rho * vp * vp + pgas
              + 0.5 * (br*br + bt*bt - bp*bp);
    u0(m, IM2, k, j, i) += dt * src1_i * src1_j * m_pp;

    // (iv) Theta-direction advection of phi momentum.
    if (multi_d) {
      const Real sm = geom.sin_theta_face(m, j);
      const Real sp = geom.sin_theta_face(m, j+1);
      u0(m, IM3, k, j, i) -= dt * src1_i * src2_j *
                            (sm * flx2(m, IM3, k, j,   i)
                           + sp * flx2(m, IM3, k, j+1, i));
    } else {
      // 1D-theta fallback: subtract <cot/r>*(rho v_t v_p - B_t B_p).
      Real m_ph = rho * vt * vp - bt * bp;
      u0(m, IM3, k, j, i) -= dt * src1_i * src1_j * m_ph;
    }
  });
}
