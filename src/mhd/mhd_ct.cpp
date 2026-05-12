//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mhd_ct.cpp
//  \brief Constrained-transport update of face-centred magnetic fields.
//  Cartesian path:    dB/dt = -Curl(E), uniform-spacing.
//  spherical_shell:   discrete Stokes loop with SphericalShellGeom face areas and
//                     edge lengths (see Athena++ srcpp/coordinates/spherical_polar.cpp).

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "srcterms/srcterms.hpp"
#include "driver/driver.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/spherical_shell.hpp"
#include "mhd.hpp"

namespace mhd {
//----------------------------------------------------------------------------------------
//! \fn  void MHD::CT
//  \brief Constrained Transport implementation of dB/dt = -Curl(E), where E=-(v X B)
//  To be clear, the edge-centered variable 'efld' stores E = -(v X B).
//  Temporal update uses multi-step SSP integrators, e.g. RK2, RK3

TaskStatus MHD::CT(Driver *pdriver, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;

  // capture class variables for the kernels
  Real &gam0 = pdriver->gam0[stage-1];
  Real &gam1 = pdriver->gam1[stage-1];
  Real beta_dt = (pdriver->beta[stage-1])*(pmy_pack->pmesh->dt);
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;
  auto e1 = efld.x1e;
  auto e2 = efld.x2e;
  auto e3 = efld.x3e;

  const bool is_spherical = (pmy_pack->pcoord->coord_system ==
                             CoordinateSystem::spherical_shell);

  if (!is_spherical) {
    auto &mbsize = pmy_pack->pmb->mb_size;

    //---- update B1 (only for 2D/3D problems)
    if (multi_d) {
      auto bx1f = b0.x1f;
      auto bx1f_old = b1.x1f;
      par_for("CT-b1", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie+1,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        bx1f(m,k,j,i) = gam0*bx1f(m,k,j,i) + gam1*bx1f_old(m,k,j,i);
        bx1f(m,k,j,i) -= beta_dt*(e3(m,k,j+1,i) - e3(m,k,j,i))/mbsize.d_view(m).dx2;
        if (three_d) {
          bx1f(m,k,j,i) += beta_dt*(e2(m,k+1,j,i) - e2(m,k,j,i))/mbsize.d_view(m).dx3;
        }
      });
    }

    //---- update B2 (curl terms in 1D and 3D problems)
    auto bx2f = b0.x2f;
    auto bx2f_old = b1.x2f;
    par_for("CT-b2", DevExeSpace(), 0, nmb1, ks, ke, js, je+1, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      bx2f(m,k,j,i) = gam0*bx2f(m,k,j,i) + gam1*bx2f_old(m,k,j,i);
      bx2f(m,k,j,i) += beta_dt*(e3(m,k,j,i+1) - e3(m,k,j,i))/mbsize.d_view(m).dx1;
      if (three_d) {
        bx2f(m,k,j,i) -= beta_dt*(e1(m,k+1,j,i) - e1(m,k,j,i))/mbsize.d_view(m).dx3;
      }
    });

    //---- update B3 (curl terms in 1D and 2D/3D problems)
    auto bx3f = b0.x3f;
    auto bx3f_old = b1.x3f;
    par_for("CT-b3", DevExeSpace(), 0, nmb1, ks, ke+1, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      bx3f(m,k,j,i) = gam0*bx3f(m,k,j,i) + gam1*bx3f_old(m,k,j,i);
      bx3f(m,k,j,i) -= beta_dt*(e2(m,k,j,i+1) - e2(m,k,j,i))/mbsize.d_view(m).dx1;
      if (multi_d) {
        bx3f(m,k,j,i) += beta_dt*(e1(m,k,j+1,i) - e1(m,k,j,i))/mbsize.d_view(m).dx2;
      }
    });

    return TaskStatus::complete;
  }

  // ----------------- spherical_shell branch -------------------------------------
  //
  // Spherical CT uses face areas A1, A2, A3 and edge lengths L1, L2, L3 from
  // SphericalShellGeom. The discrete Stokes loop for each face component:
  //
  //   dB1/dt (radial face) = -(1/A1) [ L3(j+1) E3(j+1) - L3(j) E3(j) ]      (theta loop)
  //                          +(1/A1) [ L2(k+1) E2(k+1) - L2(k) E2(k) ]      (phi loop)
  //   dB2/dt (theta face)  = +(1/A2) [ L3(i+1) E3(i+1) - L3(i) E3(i) ]      (r loop)
  //                          -(1/A2) [ L1(k+1) E1(k+1) - L1(k) E1(k) ]      (phi loop)
  //   dB3/dt (phi   face)  = -(1/A3) [ L2(i+1) E2(i+1) - L2(i) E2(i) ]      (r loop)
  //                          +(1/A3) [ L1(j+1) E1(j+1) - L1(j) E1(j) ]      (theta loop)
  //
  // L1 depends only on i; L2 on (m, i, j); L3 on (m, i, j, k). Where a length factor
  // does not vary across the difference we factor it out for efficiency.
  // Face areas / edge lengths use the existing SphericalShellGeom views -- no trig
  // is computed in these hot loops.
  auto geom = pmy_pack->pcoord->shell_geom;

  //---- update B1
  if (multi_d) {
    auto bx1f = b0.x1f;
    auto bx1f_old = b1.x1f;
    par_for("CT-b1-sph", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie+1,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      const Real A1 = SphFace1Area(geom, m, k, j, i);
      const Real inv_A1 = 1.0 / A1;
      // L3 at (m,k,j,i) and (m,k,j+1,i): both with same r_face, same dphi_k, but
      // different sin_theta_face.
      const Real L3_m = SphEdge3Length(geom, m, k, j,   i);
      const Real L3_p = SphEdge3Length(geom, m, k, j+1, i);
      bx1f(m,k,j,i) = gam0*bx1f(m,k,j,i) + gam1*bx1f_old(m,k,j,i);
      bx1f(m,k,j,i) -= beta_dt * inv_A1 *
                       (L3_p * e3(m,k,j+1,i) - L3_m * e3(m,k,j,i));
      if (three_d) {
        // L2 at (m,k,j,i) and (m,k+1,j,i): only depends on (m, j, i), so factor out.
        const Real L2 = SphEdge2Length(geom, m, k, j, i);
        bx1f(m,k,j,i) += beta_dt * inv_A1 * L2 *
                         (e2(m,k+1,j,i) - e2(m,k,j,i));
      }
    });
  }

  //---- update B2
  {
    auto bx2f = b0.x2f;
    auto bx2f_old = b1.x2f;
    par_for("CT-b2-sph", DevExeSpace(), 0, nmb1, ks, ke, js, je+1, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      const Real A2 = SphFace2Area(geom, m, k, j, i);
      const Real inv_A2 = 1.0 / A2;
      // L3 at (m,k,j,i) and (m,k,j,i+1): different r_face. Same sin_theta_face(j),
      // same dphi(k). Cannot factor out.
      const Real L3_im = SphEdge3Length(geom, m, k, j, i  );
      const Real L3_ip = SphEdge3Length(geom, m, k, j, i+1);
      bx2f(m,k,j,i) = gam0*bx2f(m,k,j,i) + gam1*bx2f_old(m,k,j,i);
      bx2f(m,k,j,i) += beta_dt * inv_A2 *
                       (L3_ip * e3(m,k,j,i+1) - L3_im * e3(m,k,j,i));
      if (three_d) {
        // L1 = dr(m,i): depends only on i, factor out across k.
        const Real L1 = SphEdge1Length(geom, m, k, j, i);
        bx2f(m,k,j,i) -= beta_dt * inv_A2 * L1 *
                         (e1(m,k+1,j,i) - e1(m,k,j,i));
      }
    });
  }

  //---- update B3
  {
    auto bx3f = b0.x3f;
    auto bx3f_old = b1.x3f;
    par_for("CT-b3-sph", DevExeSpace(), 0, nmb1, ks, ke+1, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      const Real A3 = SphFace3Area(geom, m, k, j, i);
      const Real inv_A3 = 1.0 / A3;
      // L2 at (m,k,j,i) and (m,k,j,i+1): different r_face. Same dtheta(j). Cannot
      // factor out.
      const Real L2_im = SphEdge2Length(geom, m, k, j, i  );
      const Real L2_ip = SphEdge2Length(geom, m, k, j, i+1);
      bx3f(m,k,j,i) = gam0*bx3f(m,k,j,i) + gam1*bx3f_old(m,k,j,i);
      bx3f(m,k,j,i) -= beta_dt * inv_A3 *
                       (L2_ip * e2(m,k,j,i+1) - L2_im * e2(m,k,j,i));
      if (multi_d) {
        // L1 = dr(m,i): depends only on i, factor out across j.
        const Real L1 = SphEdge1Length(geom, m, k, j, i);
        bx3f(m,k,j,i) += beta_dt * inv_A3 * L1 *
                         (e1(m,k,j+1,i) - e1(m,k,j,i));
      }
    });
  }

  return TaskStatus::complete;
}
} // namespace mhd
