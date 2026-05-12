#ifndef COORDINATES_COORDINATES_HPP_
#define COORDINATES_COORDINATES_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file coordinates.hpp
//! \brief implemention of light-weight coordinates class.  Provides data structure that
//! stores array of RegionSizes over (# of MeshBlocks), and inline functions for
//! computing positions.  In GR, also provides inline metric functions (currently only
//! Cartesian Kerr-Schild)

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/spherical_shell.hpp"

// forward declarations
struct EOS_Data;

// Enumerator for the excision method
enum class ExcisionScheme {
  fixed,
  lapse
};

//----------------------------------------------------------------------------------------
//! \enum CoordinateSystem
//! \brief Selector for the logical-Cartesian coordinate system used by hydro/MHD.
//! Default is cartesian (existing AthenaK behaviour). spherical_shell is the new
//! single-purpose spherical-polar option (r, theta, phi). Other curvilinear systems
//! are intentionally NOT supported -- this is a research fork, not a general framework.
//! GR coordinates (Cartesian Kerr-Schild) are handled separately via
//! is_general_relativistic / coord_data.

enum class CoordinateSystem {
  cartesian,
  spherical_shell
};

//----------------------------------------------------------------------------------------
//! \struct CoordData
//! \brief container for Coordinate variables and functions needed inside kernels. Storing
//! everything in a container makes them easier to capture, and pass to inline functions,
//! inside kernels.

struct CoordData {
  // following data is only used in GR calculations to compute metric
  bool is_minkowski;               // flag to specify Minkowski (flat) space
  Real bh_spin;                    // needed for GR metric
  bool bh_excise;                  // flag to specify excision
  Real rexcise;                    // excision radius (SKS)
  Real dexcise;                    // rest-mass density inside excised region
  Real pexcise;                    // pressure inside excised region
  Real flux_excise_r;              // reduce to first-order inside this radius
  ExcisionScheme excision_scheme;  // excision method
  Real excise_lapse;               // if excision_scheme = lapse, excise under this lapse
};

//----------------------------------------------------------------------------------------
//! \class Coordinates
//! \brief data and functions for coordinates

class Coordinates {
 public:
  explicit Coordinates(ParameterInput *pin, MeshBlockPack *ppack);
  ~Coordinates() {}

  // flags to denote relativistic dynamics in these coordinates
  bool is_special_relativistic = false;
  bool is_general_relativistic = false;
  bool is_dynamical_relativistic = false;

  // Logical-Cartesian coordinate system (cartesian or spherical_shell). Set from the
  // <coord> block in the input file. spherical_shell coordinates require the
  // companion shell_geom container below.
  CoordinateSystem coord_system = CoordinateSystem::cartesian;

  // Spherical-polar shell geometry (only populated when coord_system == spherical_shell).
  // Holds device-resident Kokkos Views of all finite-volume factors. Capture by value
  // into kernels.
  SphericalShellGeom shell_geom;

  // Radial-grid selection. Default is uniform (matches Task 1-3B behaviour).
  RadialGridType radial_grid = RadialGridType::uniform;

  // data needed to compute metric in GR
  CoordData coord_data;

  // excision masks
  DvceArray4D<bool> excision_floor;  // cell-centered mask for C2P flooring about horizon
  DvceArray4D<bool> excision_flux;   // cell-centered mask for FOFC about horizon

  // functions
  void CoordSrcTerms(const DvceArray5D<Real> &w0, const EOS_Data &eos, const Real dt,
                     DvceArray5D<Real> &u0);
  void CoordSrcTerms(const DvceArray5D<Real> &w0, const DvceArray5D<Real> &bcc,
                     const EOS_Data &eos, const Real dt, DvceArray5D<Real> &u0);

  // Spherical-shell geometric source terms for Newtonian hydro. Uses primitives
  // (w0) and the radial/theta Riemann fluxes to assemble Athena++'s flux-weighted
  // (-rho v_r v_t / r) etc. terms. See coordinates/spherical_shell.cpp.
  void AddSphericalShellHydroSrcTerms(const DvceArray5D<Real> &w0,
                                      const DvceArray5D<Real> &flx1,
                                      const DvceArray5D<Real> &flx2,
                                      const EOS_Data &eos,
                                      const Real dt,
                                      DvceArray5D<Real> &u0);

  // MHD overload: extends hydro source terms with magnetic-stress contributions.
  //   m_ii += B_r^2
  //   m_pp += 0.5 (B_r^2 + B_t^2 - B_p^2)
  //   1D-theta fallback m_ph -= B_t * B_p
  // flx1/flx2 are the MHD Riemann fluxes (already include -B_i B_j stress).
  // See Athena++ srcpp/coordinates/spherical_polar.cpp::AddCoordTermsDivergence.
  void AddSphericalShellMHDSrcTerms(const DvceArray5D<Real> &w0,
                                    const DvceArray5D<Real> &bcc,
                                    const DvceArray5D<Real> &flx1,
                                    const DvceArray5D<Real> &flx2,
                                    const EOS_Data &eos,
                                    const Real dt,
                                    DvceArray5D<Real> &u0);

  void SetExcisionMasks(DvceArray4D<bool> &floor, DvceArray4D<bool> &flux);

  void UpdateExcisionMasks();

 private:
  MeshBlockPack* pmy_pack;
};

#endif // COORDINATES_COORDINATES_HPP_
