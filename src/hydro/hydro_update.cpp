//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro_update.cpp
//! \brief Performs explicit update of Hydro conserved variables (u0) for each stage of
//! the SSP RK integrators (e.g. RK1, RK2, RK3) implemented in AthenaK, using weighted
//! average and partial time step update of flux divergence. Source terms are added in
//! the HydroSrcTerms() function.
//!
//! For coord_system == cartesian, uses the simple uniform-spacing flux divergence.
//! For coord_system == spherical_shell, uses the area-weighted finite-volume divergence
//! with cell volumes from SphericalShellGeom (Athena++ srcpp/coordinates/spherical_polar.cpp
//! style). Both branches are dispatched once at the host level so the inner kernels are
//! free of coordinate-system branching.

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "eos/eos.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/spherical_shell.hpp"
#include "hydro.hpp"

namespace hydro {
//----------------------------------------------------------------------------------------
//! \fn  void Hydro::Update
//  \brief Explicit RK update including flux divergence terms

TaskStatus Hydro::RKUpdate(Driver *pdriver, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;

  Real &gam0 = pdriver->gam0[stage-1];
  Real &gam1 = pdriver->gam1[stage-1];
  Real beta_dt = (pdriver->beta[stage-1])*(pmy_pack->pmesh->dt);
  int nmb1 = pmy_pack->nmb_thispack - 1;
  int nvar = nhydro + nscalars;
  auto u0_ = u0;
  auto u1_ = u1;
  auto flx1 = uflx.x1f;
  auto flx2 = uflx.x2f;
  auto flx3 = uflx.x3f;

  int scr_level = 0;
  size_t scr_size = ScrArray1D<Real>::shmem_size(ncells1);

  if (pmy_pack->pcoord->coord_system == CoordinateSystem::cartesian) {
    auto &mbsize = pmy_pack->pmb->mb_size;
    par_for_outer("h_update",DevExeSpace(),scr_size,scr_level,
                   0,nmb1,0,nvar-1,ks,ke,js,je,
    KOKKOS_LAMBDA(TeamMember_t member, const int m, const int n,
                  const int k, const int j) {
      ScrArray1D<Real> divf(member.team_scratch(scr_level), ncells1);

      // compute dF1/dx1
      par_for_inner(member, is, ie, [&](const int i) {
        divf(i) = (flx1(m,n,k,j,i+1) - flx1(m,n,k,j,i))/mbsize.d_view(m).dx1;
      });
      member.team_barrier();

      // Add dF2/dx2
      if (multi_d) {
        par_for_inner(member, is, ie, [&](const int i) {
          divf(i) += (flx2(m,n,k,j+1,i) - flx2(m,n,k,j,i))/mbsize.d_view(m).dx2;
        });
        member.team_barrier();
      }

      // Add dF3/dx3
      if (three_d) {
        par_for_inner(member, is, ie, [&](const int i) {
          divf(i) += (flx3(m,n,k+1,j,i) - flx3(m,n,k,j,i))/mbsize.d_view(m).dx3;
        });
        member.team_barrier();
      }

      par_for_inner(member, is, ie, [&](const int i) {
        u0_(m,n,k,j,i) = gam0*u0_(m,n,k,j,i) + gam1*u1_(m,n,k,j,i) - beta_dt*divf(i);
      });
    });
  } else {
    // Spherical-shell finite-volume update.
    //   dU/dt = -(1/V) * [ A1+ F1+ - A1- F1-
    //                    + A2+ F2+ - A2- F2-
    //                    + A3+ F3+ - A3- F3- ]
    // with face areas and cell volumes from SphericalShellGeom.
    // Geometric momentum source terms are added separately in HydroSrcTerms.
    auto geom = pmy_pack->pcoord->shell_geom;
    par_for_outer("h_update_sph",DevExeSpace(),scr_size,scr_level,
                   0,nmb1,0,nvar-1,ks,ke,js,je,
    KOKKOS_LAMBDA(TeamMember_t member, const int m, const int n,
                  const int k, const int j) {
      ScrArray1D<Real> divf(member.team_scratch(scr_level), ncells1);

      // (m,j,k) factors hoisted out of the inner i-loop
      Real dcos_j   = geom.dcos_theta(m, j);
      Real dphi_k   = geom.dphi(m, k);
      Real sin_jm   = geom.sin_theta_face(m, j);
      Real sin_jp   = geom.sin_theta_face(m, j+1);
      Real dtheta_j = geom.dtheta(m, j);

      // dF1/dr (always present): area1 = r^2 * dcos * dphi, V = dr3_third*dcos*dphi
      // so the radial divergence reduces to (r+^2 F+ - r-^2 F-)/dr3_third.
      par_for_inner(member, is, ie, [&](const int i) {
        Real r2m = geom.r2_face(m, i);
        Real r2p = geom.r2_face(m, i+1);
        Real inv_dr3 = geom.inv_dr3_third(m, i);
        divf(i) = inv_dr3 * (r2p*flx1(m,n,k,j,i+1) - r2m*flx1(m,n,k,j,i));
      });
      member.team_barrier();

      // dF2/dtheta: area2 = dr2_half * sin(theta_face) * dphi, V = dr3_third*dcos*dphi
      // so contribution = (dr2_half/dr3_third) * (sin+ F+ - sin- F-)/dcos
      if (multi_d) {
        par_for_inner(member, is, ie, [&](const int i) {
          Real ratio = geom.dr2_half(m, i) * geom.inv_dr3_third(m, i);
          divf(i) += ratio * (sin_jp*flx2(m,n,k,j+1,i) - sin_jm*flx2(m,n,k,j,i)) / dcos_j;
        });
        member.team_barrier();
      }

      // dF3/dphi: area3 = dr2_half * dtheta, V = dr3_third * dcos * dphi
      // so contribution = (dr2_half/dr3_third) * dtheta * (F+ - F-) / (dcos * dphi)
      if (three_d) {
        Real factor = dtheta_j / (dcos_j * dphi_k);
        par_for_inner(member, is, ie, [&](const int i) {
          Real ratio = geom.dr2_half(m, i) * geom.inv_dr3_third(m, i);
          divf(i) += ratio * factor * (flx3(m,n,k+1,j,i) - flx3(m,n,k,j,i));
        });
        member.team_barrier();
      }

      par_for_inner(member, is, ie, [&](const int i) {
        u0_(m,n,k,j,i) = gam0*u0_(m,n,k,j,i) + gam1*u1_(m,n,k,j,i) - beta_dt*divf(i);
      });
    });
  }
  return TaskStatus::complete;
}
} // namespace hydro
