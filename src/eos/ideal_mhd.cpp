//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ideal_mhd.cpp
//! \brief derived class that implements ideal gas EOS in nonrelativistic mhd

#include "athena.hpp"
#include "mhd/mhd.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/spherical_shell.hpp"
#include "eos.hpp"
#include "eos/ideal_c2p_mhd.hpp"

//----------------------------------------------------------------------------------------
// ctor: also calls EOS base class constructor

IdealMHD::IdealMHD(MeshBlockPack *pp, ParameterInput *pin) :
    EquationOfState("mhd", pp, pin) {
  eos_data.is_ideal = true;
  eos_data.gamma = pin->GetReal("mhd","gamma");
  eos_data.iso_cs = 0.0;
  eos_data.use_e = true;  // ideal gas EOS always uses internal energy
  eos_data.use_t = false;
  eos_data.sigma_max = pin->GetOrAddReal("mhd","sigma_max",(FLT_MAX));  // sigma ceiling
}

//----------------------------------------------------------------------------------------
//! \!fn void ConsToPrim()
//! \brief Converts conserved into primitive variables.  Operates over range of cells
//! given in argument list.

void IdealMHD::ConsToPrim(DvceArray5D<Real> &cons, const DvceFaceFld4D<Real> &b,
                          DvceArray5D<Real> &prim, DvceArray5D<Real> &bcc,
                          const bool only_testfloors,
                          const int il, const int iu, const int jl, const int ju,
                          const int kl, const int ku) {
  int &nmhd  = pmy_pack->pmhd->nmhd;
  int &nscal = pmy_pack->pmhd->nscalars;
  int &nmb = pmy_pack->nmb_thispack;
  auto &eos = eos_data;
  auto &fofc_ = pmy_pack->pmhd->fofc;
  // Spherical-shell volume-weighted face-to-cell-centre interpolation
  // (matches Athena++ Field::CalculateCellCenteredField). For Cartesian the
  // simple 0.5 average is used. `geom` is safe to capture unconditionally
  // because we only access its arrays inside the `is_spherical` branch;
  // see mhd_newdt.cpp for the same pattern.
  const bool is_spherical = (pmy_pack->pcoord->coord_system ==
                             CoordinateSystem::spherical_shell);
  auto geom = pmy_pack->pcoord->shell_geom;

  const int ni   = (iu - il + 1);
  const int nji  = (ju - jl + 1)*ni;
  const int nkji = (ku - kl + 1)*nji;
  const int nmkji = nmb*nkji;

  int nfloord_=0, nfloore_=0, nfloort_=0;
  Kokkos::parallel_reduce("mhd_c2p",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, int &sumd, int &sume, int &sumt) {
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/ni;
    int i = (idx - m*nkji - k*nji - j*ni) + il;
    j += jl;
    k += kl;

    // load single state conserved variables
    MHDCons1D u;
    u.d  = cons(m,IDN,k,j,i);
    u.mx = cons(m,IM1,k,j,i);
    u.my = cons(m,IM2,k,j,i);
    u.mz = cons(m,IM3,k,j,i);
    u.e  = cons(m,IEN,k,j,i);

    // load cell-centered fields into conserved state
    // use input CC fields if only testing floors with FOFC
    if (only_testfloors) {
      u.bx = bcc(m,IBX,k,j,i);
      u.by = bcc(m,IBY,k,j,i);
      u.bz = bcc(m,IBZ,k,j,i);
    // else use linear interpolation of face-centered fields. For
    // spherical_shell the interpolation point is the Mignone-2014
    // volume-weighted centroid (r_vol, theta_vol); for Cartesian we fall
    // back to the simple 0.5-average.
    } else if (is_spherical) {
      // Linear interp through the Mignone-2014 volume-weighted centroid.
      // Weights bcc_wx1, bcc_wx2 are precomputed at geometry setup so the
      // per-cell kernel has no division (GPU-friendly).
      const Real wlo1 = geom.bcc_wx1(m, i);
      u.bx = wlo1 * b.x1f(m,k,j,i) + (1.0 - wlo1) * b.x1f(m,k,j,i+1);
      const Real wlo2 = geom.bcc_wx2(m, j);
      u.by = wlo2 * b.x2f(m,k,j,i) + (1.0 - wlo2) * b.x2f(m,k,j+1,i);
      // phi: dV ~ dphi so phi_center = 0.5(phi_- + phi_+) is the right
      // volume centroid; the standard 0.5 average is already correct.
      u.bz = 0.5*(b.x3f(m,k,j,i) + b.x3f(m,k+1,j,i));
    } else {
      u.bx = 0.5*(b.x1f(m,k,j,i) + b.x1f(m,k,j,i+1));
      u.by = 0.5*(b.x2f(m,k,j,i) + b.x2f(m,k,j+1,i));
      u.bz = 0.5*(b.x3f(m,k,j,i) + b.x3f(m,k+1,j,i));
    }

    // call c2p function
    // (inline function in ideal_c2p_mhd.hpp file)
    HydPrim1D w;
    bool dfloor_used=false, efloor_used=false, tfloor_used=false;
    SingleC2P_IdealMHD(u, eos, w, dfloor_used, efloor_used, tfloor_used);

    // set FOFC flag and quit loop if this function called only to check floors
    if (only_testfloors) {
      if (dfloor_used || efloor_used || tfloor_used) {
        fofc_(m,k,j,i) = true;
        sumd++;  // use dfloor as counter for when either is true
      }
    } else {
      // update counter, reset conserved if floor was hit
      if (dfloor_used) {
        cons(m,IDN,k,j,i) = u.d;
        sumd++;
      }
      if (efloor_used) {
        cons(m,IEN,k,j,i) = u.e;
        sume++;
      }
      if (tfloor_used) {
        cons(m,IEN,k,j,i) = u.e;
        sumt++;
      }
      // store primitive state in 3D array
      prim(m,IDN,k,j,i) = w.d;
      prim(m,IVX,k,j,i) = w.vx;
      prim(m,IVY,k,j,i) = w.vy;
      prim(m,IVZ,k,j,i) = w.vz;
      prim(m,IEN,k,j,i) = w.e;
      // store cell-centered fields in 3D array
      bcc(m,IBX,k,j,i) = u.bx;
      bcc(m,IBY,k,j,i) = u.by;
      bcc(m,IBZ,k,j,i) = u.bz;
      // convert scalars (if any), always stored at end of cons and prim arrays.
      for (int n=nmhd; n<(nmhd+nscal); ++n) {
        // apply scalar floor
        if (cons(m,n,k,j,i) < 0.0) {
          cons(m,n,k,j,i) = 0.0;
        }
        prim(m,n,k,j,i) = cons(m,n,k,j,i)/u.d;
      }
    }
  }, Kokkos::Sum<int>(nfloord_), Kokkos::Sum<int>(nfloore_), Kokkos::Sum<int>(nfloort_));

  // store appropriate counters
  if (only_testfloors) {
    pmy_pack->pmesh->ecounter.nfofc += nfloord_;
  } else {
    pmy_pack->pmesh->ecounter.neos_dfloor += nfloord_;
    pmy_pack->pmesh->ecounter.neos_efloor += nfloore_;
    pmy_pack->pmesh->ecounter.neos_tfloor += nfloort_;
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \!fn void PrimToCons()
//! \brief Converts conserved into primitive variables.  Operates over range of cells
//! given in argument list.  Does not change cell- or face-centered magnetic fields.

void IdealMHD::PrimToCons(const DvceArray5D<Real> &prim, const DvceArray5D<Real> &bcc,
                          DvceArray5D<Real> &cons, const int il, const int iu,
                          const int jl, const int ju, const int kl, const int ku) {
  int &nmhd  = pmy_pack->pmhd->nmhd;
  int &nscal = pmy_pack->pmhd->nscalars;
  int &nmb = pmy_pack->nmb_thispack;

  par_for("mhd_p2c", DevExeSpace(), 0, (nmb-1), kl, ku, jl, ju, il, iu,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // load single state primitive variables
    MHDPrim1D w;
    w.d  = prim(m,IDN,k,j,i);
    w.vx = prim(m,IVX,k,j,i);
    w.vy = prim(m,IVY,k,j,i);
    w.vz = prim(m,IVZ,k,j,i);
    w.e  = prim(m,IEN,k,j,i);

    // load cell-centered fields into primitive state
    w.bx = bcc(m,IBX,k,j,i);
    w.by = bcc(m,IBY,k,j,i);
    w.bz = bcc(m,IBZ,k,j,i);

    // call p2c function
    HydCons1D u;
    SingleP2C_IdealMHD(w, u);

    // store conserved state in 3D array
    cons(m,IDN,k,j,i) = u.d;
    cons(m,IM1,k,j,i) = u.mx;
    cons(m,IM2,k,j,i) = u.my;
    cons(m,IM3,k,j,i) = u.mz;
    cons(m,IEN,k,j,i) = u.e;

    // convert scalars (if any), always stored at end of cons and prim arrays.
    for (int n=nmhd; n<(nmhd+nscal); ++n) {
      cons(m,n,k,j,i) = u.d*prim(m,n,k,j,i);
    }
  });

  return;
}
