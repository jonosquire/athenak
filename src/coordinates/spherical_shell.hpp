#ifndef COORDINATES_SPHERICAL_SHELL_HPP_
#define COORDINATES_SPHERICAL_SHELL_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file spherical_shell.hpp
//! \brief Geometry data and device-callable helpers for a logically-Cartesian
//! spherical-polar shell (r, theta, phi).
//!
//! This is a single-purpose, GPU-native container: per-MeshBlock 1D Kokkos Views of
//! exact spherical finite-volume factors (cell volume, face areas, physical widths,
//! and useful trig precomputations). It is allocated only when the user selects
//! the spherical_shell coordinate option.
//!
//! No magnetic-field/CT/AMR support is provided here -- this is hydro-only
//! infrastructure suitable for a static, single-level grid.

#include "athena.hpp"

class MeshBlockPack;
class ParameterInput;

//----------------------------------------------------------------------------------------
//! \enum RadialGridType
//! \brief Selects the radial-face spacing rule for spherical_shell.
//!   uniform   : r_face linear in the logical x1 coordinate (default; matches
//!               Task 1-3B behaviour).
//!   log       : r_face uniform in ln(r) over the global mesh [x1min, x1max].
//!   power_law : r_face = r_min + (r_max - r_min) * xi_norm^alpha, where alpha
//!               is configurable via <spherical_shell>/r_grid_alpha.
//!   r_cubed   : r_face^3 is linear in xi_norm. For a constant-density monopole
//!               v_A proportional to r^-2, this is uniform in Alfven travel time.
//!   user      : r_face supplied by a user host-side function pointer registered
//!               via SetUserRadialGridFunc() before Coordinates construction.
//!               See README-sph.md.

enum class RadialGridType {
  uniform,
  log,
  power_law,
  r_cubed,
  user
};

//! \typedef UserRadialGridFnPtr
//! \brief Optional user-supplied radial grid function.
//! Maps a normalised global logical position xi_norm in [0, 1] (xi_norm=0 at
//! mesh.x1min, xi_norm=1 at mesh.x1max) to a physical radius r > 0. Must be
//! strictly monotonically increasing. Called host-side once at setup; the
//! result is then uploaded to device. To use, the user code must call
//! `SetUserRadialGridFunc(&MyFunc)` from a static initializer in its
//! translation unit BEFORE main() runs (the pgen UserProblem() runs AFTER
//! Coordinates is constructed, so it cannot register the hook).
using UserRadialGridFnPtr = Real (*)(Real xi_norm);

// Register/clear the optional user-defined radial grid function. The
// registration is process-global; this fork only supports a single
// spherical_shell at a time, which is the intended pattern.
void SetUserRadialGridFunc(UserRadialGridFnPtr fn);
UserRadialGridFnPtr GetUserRadialGridFunc();

//----------------------------------------------------------------------------------------
//! \struct SphericalShellGeom
//! \brief Plain-data container holding device-resident Kokkos Views of all geometric
//! factors needed by spherical finite-volume hydro. Stored on Coordinates and captured
//! by value into device kernels.
//!
//! Index conventions (matching AthenaK):
//!   - first dimension is MeshBlock index (m) within the pack
//!   - second dimension is cell or face index, including ghost zones
//!   - "_face" arrays have length (ncells+1); "vol"/"center"/"d*" have length (ncells)
//!
//! Geometric formulas implemented (exact spherical finite-volume integrals):
//!   V_ijk      = dr3_third(m,i) * dcos_theta(m,j) * dphi(m,k)
//!   A1(face_i) = r2_face(m,i)   * dcos_theta(m,j) * dphi(m,k)
//!   A2(face_j) = dr2_half(m,i)  * sin_theta_face(m,j) * dphi(m,k)
//!   A3(face_k) = dr2_half(m,i)  * dtheta(m,j)
//!   ds1 = dr(m,i)
//!   ds2 = r_vol(m,i) * dtheta(m,j)
//!   ds3 = r_vol(m,i) * sin_theta_vol(m,j) * dphi(m,k)

struct SphericalShellGeom {
  int nmb = 0;
  int nc1 = 0, nc2 = 0, nc3 = 0;     // including ghost zones
  int ng = 0;                        // number of ghost cells

  // r-direction (size nc1 or nc1+1 in the i-dimension)
  DvceArray2D<Real> r_face;          // r_{i-1/2}                              (nmb, nc1+1)
  DvceArray2D<Real> r_vol;            // volume-averaged r at cell centre        (nmb, nc1)
  DvceArray2D<Real> dr;               // r_{i+1/2} - r_{i-1/2}                  (nmb, nc1)
  DvceArray2D<Real> r2_face;          // r_face^2                              (nmb, nc1+1)
  DvceArray2D<Real> dr2_half;         // 0.5*(r_+^2 - r_-^2)                   (nmb, nc1)
  DvceArray2D<Real> dr3_third;        // (1/3)*(r_+^3 - r_-^3) = cell radial vol (nmb, nc1)
  DvceArray2D<Real> inv_dr3_third;    // 1.0 / dr3_third                        (nmb, nc1)
  // Spherical source-term factors (Athena++ coord_src1_i_, coord_src2_i_):
  //   src1_i = dr2_half / dr3_third    = (3/2)(r+^2 - r-^2) / (r+^3 - r-^3)
  //          = volume-averaged 1/r times the area-to-volume ratio used for the
  //            "p / r" curvature source term and for centripetal terms
  //   src2_i = dr      / ((rm + rp) * dr3_third)
  //          appears multiplying flux-area-weighted radial fluxes in the
  //            (-rho v_r v_theta / r) and (-rho v_r v_phi / r) source terms
  DvceArray2D<Real> coord_src1_i;     // (nmb, nc1)
  DvceArray2D<Real> coord_src2_i;     // (nmb, nc1)

  // theta-direction (size nc2 or nc2+1 in the j-dimension)
  DvceArray2D<Real> theta_face;       // theta_{j-1/2}                         (nmb, nc2+1)
  DvceArray2D<Real> theta_vol;        // volume-weighted theta at cell centre  (nmb, nc2)
  DvceArray2D<Real> dtheta;           // theta_{j+1/2} - theta_{j-1/2}         (nmb, nc2)
  DvceArray2D<Real> sin_theta_face;   // sin(theta_face)                       (nmb, nc2+1)
  DvceArray2D<Real> sin_theta_vol;    // sin(theta_vol)                        (nmb, nc2)
  DvceArray2D<Real> dcos_theta;       // cos(theta_-) - cos(theta_+)  >= 0      (nmb, nc2)
  // Spherical source-term factors (Athena++ coord_src1_j_, coord_src2_j_):
  //   src1_j = (sp - sm) / dcos_theta   approximates <cot(theta)>
  //   src2_j = (sp - sm) / ((sm + sp) * dcos_theta)
  //          appears multiplying flux-area-weighted theta fluxes in the
  //            (-rho v_theta v_phi cot(theta) / r) source term
  DvceArray2D<Real> coord_src1_j;     // (nmb, nc2)
  DvceArray2D<Real> coord_src2_j;     // (nmb, nc2)

  // phi-direction (size nc3 or nc3+1 in the k-dimension)
  DvceArray2D<Real> phi_face;         // phi_{k-1/2}                           (nmb, nc3+1)
  DvceArray2D<Real> phi_center;       // 0.5*(phi_-+phi_+)                     (nmb, nc3)
  DvceArray2D<Real> dphi;             // phi_{k+1/2} - phi_{k-1/2}             (nmb, nc3)
};

//----------------------------------------------------------------------------------------
// Device-callable geometry accessors. Use as inline helpers inside Kokkos kernels.

KOKKOS_INLINE_FUNCTION
Real SphCellVolume(const SphericalShellGeom &g,
                   int m, int k, int j, int i) {
  return g.dr3_third(m,i) * g.dcos_theta(m,j) * g.dphi(m,k);
}

KOKKOS_INLINE_FUNCTION
Real SphInvCellVolume(const SphericalShellGeom &g,
                      int m, int k, int j, int i) {
  return g.inv_dr3_third(m,i) / (g.dcos_theta(m,j) * g.dphi(m,k));
}

KOKKOS_INLINE_FUNCTION
Real SphFace1Area(const SphericalShellGeom &g,
                  int m, int k, int j, int i) {
  // radial face at i-1/2 in the index convention (so i runs to nc1)
  return g.r2_face(m,i) * g.dcos_theta(m,j) * g.dphi(m,k);
}

KOKKOS_INLINE_FUNCTION
Real SphFace2Area(const SphericalShellGeom &g,
                  int m, int k, int j, int i) {
  // theta face at j-1/2
  return g.dr2_half(m,i) * g.sin_theta_face(m,j) * g.dphi(m,k);
}

KOKKOS_INLINE_FUNCTION
Real SphFace3Area(const SphericalShellGeom &g,
                  int m, int k, int j, int i) {
  // phi face at k-1/2 (independent of phi for an orthogonal sphere)
  return g.dr2_half(m,i) * g.dtheta(m,j);
}

KOKKOS_INLINE_FUNCTION
Real SphPhysWidth1(const SphericalShellGeom &g,
                   int /*m*/, int /*k*/, int /*j*/, int i,
                   int mb) {
  return g.dr(mb,i);
}

KOKKOS_INLINE_FUNCTION
Real SphPhysWidth2(const SphericalShellGeom &g,
                   int m, int /*k*/, int j, int i) {
  return g.r_vol(m,i) * g.dtheta(m,j);
}

KOKKOS_INLINE_FUNCTION
Real SphPhysWidth3(const SphericalShellGeom &g,
                   int m, int k, int j, int i) {
  return g.r_vol(m,i) * g.sin_theta_vol(m,j) * g.dphi(m,k);
}

//----------------------------------------------------------------------------------------
// Edge-length accessors for CT.
//
// In AthenaK's CT convention, the i,j,k indices of an edge match the indices of the
// adjacent CELL whose lower-corner sits on that edge -- i.e. Edge1(m,k,j,i) is the
// radial edge that runs from (i-1/2, j-1/2, k-1/2) toward (i+1/2, j-1/2, k-1/2). The
// edge thus sits at theta-FACE j and phi-FACE k.
//
// Spherical-polar edge lengths (matches Athena++ srcpp/coordinates/spherical_polar.cpp):
//   L1 = dr                                        -- radial edge
//   L2 = r_face * dtheta                           -- theta edge
//   L3 = r_face * sin(theta_face) * dphi           -- phi edge
//
// L1 depends only on i; L2 depends on (i, j); L3 depends on (i, j, k).

KOKKOS_INLINE_FUNCTION
Real SphEdge1Length(const SphericalShellGeom &g,
                    int m, int /*k*/, int /*j*/, int i) {
  return g.dr(m, i);
}

KOKKOS_INLINE_FUNCTION
Real SphEdge2Length(const SphericalShellGeom &g,
                    int m, int /*k*/, int j, int i) {
  return g.r_face(m, i) * g.dtheta(m, j);
}

KOKKOS_INLINE_FUNCTION
Real SphEdge3Length(const SphericalShellGeom &g,
                    int m, int k, int j, int i) {
  return g.r_face(m, i) * g.sin_theta_face(m, j) * g.dphi(m, k);
}

//----------------------------------------------------------------------------------------
// Setup and diagnostics (host side).

//! \brief Allocate and populate every Kokkos View in `geom` from MeshBlockPack metadata.
//! Radial face positions follow `grid_type`:
//!   uniform : linear in the logical x1 (existing behaviour)
//!   log     : uniform in ln(r) over the global mesh range
//!   user    : evaluated host-side via `user_fn(xi_norm)` and uploaded to device
//! Theta and phi remain uniform in their logical coordinates (only the radial
//! direction supports nonuniform spacing in this fork).
void ConstructSphericalShellGeometry(MeshBlockPack *ppack,
                                     SphericalShellGeom &geom,
                                     RadialGridType grid_type = RadialGridType::uniform,
                                     Real grid_alpha = 1.0);

//! \brief Sanity check: sum cell volumes over active cells across the pack and compare
//! against the analytic spherical-wedge volume from mesh_size. Prints relative error
//! on rank 0. Returns the relative error.
Real VerifySphericalShellVolume(MeshBlockPack *ppack,
                                const SphericalShellGeom &geom,
                                bool verbose = true);

#endif // COORDINATES_SPHERICAL_SHELL_HPP_
