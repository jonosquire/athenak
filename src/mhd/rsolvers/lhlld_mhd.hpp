#ifndef MHD_RSOLVERS_LHLLD_MHD_HPP_
#define MHD_RSOLVERS_LHLLD_MHD_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file lhlld_mhd.hpp
//! \brief Low-dissipation HLLD Riemann solver for ideal gas EOS in MHD.
//!
//! Ported from Athena++ srcpp/hydro/rsolvers/mhd/lhlld.cpp, Minoshima et al. (2021).
//! This is an adiabatic/ideal-gas solver only; Athena++ has no isothermal LHLLD path.

namespace mhd {

#define LHLLD_SMALL_NUMBER 1.0e-4

KOKKOS_INLINE_FUNCTION
void LHLLDVelocityDifferences(const RegionIndcs &indcs, const DvceArray5D<Real> &w,
                              const int m, const int k, const int j, const int i,
                              const int ivx, Real &dvn, Real &dvt) {
  if (indcs.nx3 > 1) {
    if (ivx == IVX) {
      dvn = w(m,IVX,k,j,i) - w(m,IVX,k,j,i-1);
      Real dvl = fmin(w(m,IVY,k,j+1,i-1) - w(m,IVY,k,j,  i-1),
                      w(m,IVY,k,j,  i-1) - w(m,IVY,k,j-1,i-1));
      Real dvr = fmin(w(m,IVY,k,j+1,i)   - w(m,IVY,k,j,  i),
                      w(m,IVY,k,j,  i)   - w(m,IVY,k,j-1,i));
      Real dwl = fmin(w(m,IVZ,k+1,j,i-1) - w(m,IVZ,k,  j,i-1),
                      w(m,IVZ,k,  j,i-1) - w(m,IVZ,k-1,j,i-1));
      Real dwr = fmin(w(m,IVZ,k+1,j,i)   - w(m,IVZ,k,  j,i),
                      w(m,IVZ,k,  j,i)   - w(m,IVZ,k-1,j,i));
      dvt = fmin(fmin(dvl, dvr), fmin(dwl, dwr));
    } else if (ivx == IVY) {
      dvn = w(m,IVY,k,j,i) - w(m,IVY,k,j-1,i);
      Real dvl = fmin(w(m,IVZ,k+1,j-1,i) - w(m,IVZ,k,  j-1,i),
                      w(m,IVZ,k,  j-1,i) - w(m,IVZ,k-1,j-1,i));
      Real dvr = fmin(w(m,IVZ,k+1,j,  i) - w(m,IVZ,k,  j,  i),
                      w(m,IVZ,k,  j,  i) - w(m,IVZ,k-1,j,  i));
      Real dwl = fmin(w(m,IVX,k,j-1,i+1) - w(m,IVX,k,j-1,i),
                      w(m,IVX,k,j-1,i)   - w(m,IVX,k,j-1,i-1));
      Real dwr = fmin(w(m,IVX,k,j,  i+1) - w(m,IVX,k,j,  i),
                      w(m,IVX,k,j,  i)   - w(m,IVX,k,j,  i-1));
      dvt = fmin(fmin(dvl, dvr), fmin(dwl, dwr));
    } else {
      dvn = w(m,IVZ,k,j,i) - w(m,IVZ,k-1,j,i);
      Real dvl = fmin(w(m,IVX,k-1,j,i+1) - w(m,IVX,k-1,j,i),
                      w(m,IVX,k-1,j,i)   - w(m,IVX,k-1,j,i-1));
      Real dvr = fmin(w(m,IVX,k,  j,i+1) - w(m,IVX,k,  j,i),
                      w(m,IVX,k,  j,i)   - w(m,IVX,k,  j,i-1));
      Real dwl = fmin(w(m,IVY,k-1,j+1,i) - w(m,IVY,k-1,j,  i),
                      w(m,IVY,k-1,j,  i) - w(m,IVY,k-1,j-1,i));
      Real dwr = fmin(w(m,IVY,k,  j+1,i) - w(m,IVY,k,  j,  i),
                      w(m,IVY,k,  j,  i) - w(m,IVY,k,  j-1,i));
      dvt = fmin(fmin(dvl, dvr), fmin(dwl, dwr));
    }
  } else if (indcs.nx2 > 1) {
    if (ivx == IVX) {
      dvn = w(m,IVX,k,j,i) - w(m,IVX,k,j,i-1);
      Real dvl = fmin(w(m,IVY,k,j+1,i-1) - w(m,IVY,k,j,  i-1),
                      w(m,IVY,k,j,  i-1) - w(m,IVY,k,j-1,i-1));
      Real dvr = fmin(w(m,IVY,k,j+1,i)   - w(m,IVY,k,j,  i),
                      w(m,IVY,k,j,  i)   - w(m,IVY,k,j-1,i));
      dvt = fmin(dvl, dvr);
    } else {
      dvn = w(m,IVY,k,j,i) - w(m,IVY,k,j-1,i);
      Real dvl = fmin(w(m,IVX,k,j-1,i+1) - w(m,IVX,k,j-1,i),
                      w(m,IVX,k,j-1,i)   - w(m,IVX,k,j-1,i-1));
      Real dvr = fmin(w(m,IVX,k,j,  i+1) - w(m,IVX,k,j,  i),
                      w(m,IVX,k,j,  i)   - w(m,IVX,k,j,  i-1));
      dvt = fmin(dvl, dvr);
    }
  } else {
    dvn = w(m,IVX,k,j,i) - w(m,IVX,k,j,i-1);
    dvt = dvn;
  }
}

//----------------------------------------------------------------------------------------
//! \fn

KOKKOS_INLINE_FUNCTION
void LHLLD(TeamMember_t const &member, const EOS_Data &eos,
     const RegionIndcs &indcs,const DualArray1D<RegionSize> &size,const CoordData &coord,
     const int m, const int k, const int j, const int il, const int iu, const int ivx,
     const ScrArray2D<Real> &wl, const ScrArray2D<Real> &wr,
     const ScrArray2D<Real> &bl, const ScrArray2D<Real> &br, const DvceArray4D<Real> &bx,
     const DvceArray5D<Real> &wcell,
     DvceArray5D<Real> flx, DvceArray4D<Real> ey, DvceArray4D<Real> ez) {
  if (!eos.is_ideal) Kokkos::abort("LHLLD is implemented only for ideal-gas MHD.\n");

  int ivy = IVX + ((ivx-IVX)+1)%3;
  int ivz = IVX + ((ivx-IVX)+2)%3;
  int iby = ((ivx-IVX) + 1)%3;
  int ibz = ((ivx-IVX) + 2)%3;
  Real spd[5];
  Real gm1 = eos.gamma - 1.0;
  Real igm1 = 1.0/gm1;

  par_for_inner(member, il, iu, [&](const int i) {
    Real &wl_idn=wl(IDN,i);
    Real &wl_ivx=wl(ivx,i);
    Real &wl_ivy=wl(ivy,i);
    Real &wl_ivz=wl(ivz,i);
    Real &wl_iby=bl(iby,i);
    Real &wl_ibz=bl(ibz,i);

    Real &wr_idn=wr(IDN,i);
    Real &wr_ivx=wr(ivx,i);
    Real &wr_ivy=wr(ivy,i);
    Real &wr_ivz=wr(ivz,i);
    Real &wr_iby=br(iby,i);
    Real &wr_ibz=br(ibz,i);

    Real wl_ipr = eos.IdealGasPressure(wl(IEN,i));
    Real wr_ipr = eos.IdealGasPressure(wr(IEN,i));
    Real &bxi = bx(m,k,j,i);

    Real bxsq = bxi*bxi;
    Real pbl = 0.5*(bxsq + (SQR(wl_iby) + SQR(wl_ibz)));
    Real pbr = 0.5*(bxsq + (SQR(wr_iby) + SQR(wr_ibz)));
    Real kel = 0.5*wl_idn*(SQR(wl_ivx) + (SQR(wl_ivy) + SQR(wl_ivz)));
    Real ker = 0.5*wr_idn*(SQR(wr_ivx) + (SQR(wr_ivy) + SQR(wr_ivz)));

    MHDCons1D ul,ur;
    ul.d  = wl_idn;
    ul.mx = wl_ivx*ul.d;
    ul.my = wl_ivy*ul.d;
    ul.mz = wl_ivz*ul.d;
    ul.e  = wl_ipr*igm1 + kel + pbl;
    ul.by = wl_iby;
    ul.bz = wl_ibz;

    ur.d  = wr_idn;
    ur.mx = wr_ivx*ur.d;
    ur.my = wr_ivy*ur.d;
    ur.mz = wr_ivz*ur.d;
    ur.e  = wr_ipr*igm1 + ker + pbr;
    ur.by = wr_iby;
    ur.bz = wr_ibz;

    Real cfl = eos.IdealMHDFastSpeed(wl_idn, wl_ipr, bxi, wl_iby, wl_ibz);
    Real cfr = eos.IdealMHDFastSpeed(wr_idn, wr_ipr, bxi, wr_iby, wr_ibz);

    spd[0] = fmin( wl_ivx-cfl, wr_ivx-cfr );
    spd[4] = fmax( wl_ivx+cfl, wr_ivx+cfr );
    Real cfmax = fmax(cfl, cfr);

    Real ptl = wl_ipr + pbl;
    Real ptr = wr_ipr + pbr;

    MHDCons1D fl,fr,flxi;
    fl.d  = ul.mx;
    fl.mx = ul.mx*wl_ivx + ptl - bxsq;
    fl.my = ul.my*wl_ivx - bxi*ul.by;
    fl.mz = ul.mz*wl_ivx - bxi*ul.bz;
    fl.e  = wl_ivx*(ul.e + ptl - bxsq) - bxi*(wl_ivy*ul.by + wl_ivz*ul.bz);
    fl.by = ul.by*wl_ivx - bxi*wl_ivy;
    fl.bz = ul.bz*wl_ivx - bxi*wl_ivz;

    fr.d  = ur.mx;
    fr.mx = ur.mx*wr_ivx + ptr - bxsq;
    fr.my = ur.my*wr_ivx - bxi*ur.by;
    fr.mz = ur.mz*wr_ivx - bxi*ur.bz;
    fr.e  = wr_ivx*(ur.e + ptr - bxsq) - bxi*(wr_ivy*ur.by + wr_ivz*ur.bz);
    fr.by = ur.by*wr_ivx - bxi*wr_ivy;
    fr.bz = ur.bz*wr_ivx - bxi*wr_ivz;

    Real sdl = spd[0] - wl_ivx;
    Real sdr = spd[4] - wr_ivx;
    Real sdld = sdl*ul.d;
    Real sdrd = sdr*ur.d;

    Real dvn, dvt;
    LHLLDVelocityDifferences(indcs, wcell, m, k, j, i, ivx, dvn, dvt);
    Real th1 = fmin(1.0, (cfmax - fmin(dvn, 0.0))/(cfmax - fmin(dvt, 0.0)));
    Real th = SQR(SQR(th1));

    spd[2] = (sdr*ur.mx - sdl*ul.mx + th*(ptl - ptr))/(sdrd - sdld);

    Real sdml   = spd[0] - spd[2];
    Real sdmr   = spd[4] - spd[2];
    Real sdml_inv = 1.0/sdml;
    Real sdmr_inv = 1.0/sdmr;

    MHDCons1D ulst,uldst,urdst,urst;
    ulst.d = sdld * sdml_inv;
    urst.d = sdrd * sdmr_inv;
    Real ulst_d_inv = 1.0/ulst.d;
    Real urst_d_inv = 1.0/urst.d;
    Real sqrtdl = sqrt(ulst.d);
    Real sqrtdr = sqrt(urst.d);

    spd[1] = spd[2] - fabs(bxi)/sqrtdl;
    spd[3] = spd[2] + fabs(bxi)/sqrtdr;

    Real clsq_arg = fmax(SQR(pbl + kel) - 2.0*kel*bxsq, 0.0);
    Real crsq_arg = fmax(SQR(pbr + ker) - 2.0*ker*bxsq, 0.0);
    Real clsq = ((pbl + kel) + sqrt(clsq_arg)) / ul.d;
    Real crsq = ((pbr + ker) + sqrt(crsq_arg)) / ur.d;
    Real chi = fmin(1.0, sqrt(fmax(clsq, crsq)) / cfmax);
    Real phi = chi * (2.0 - chi);
    Real ptst = (sdrd*ptl - sdld*ptr + phi*sdrd*sdld*(wr_ivx-wl_ivx))/(sdrd - sdld);

    ulst.mx = ulst.d * spd[2];
    if (fabs(sdld*sdml-bxsq) < (LHLLD_SMALL_NUMBER)*ptst) {
      ulst.my = ulst.d * wl_ivy;
      ulst.mz = ulst.d * wl_ivz;
      ulst.by = ul.by;
      ulst.bz = ul.bz;
    } else {
      Real tmp = bxi*(sdl - sdml)/(sdld*sdml - bxsq);
      ulst.my = ulst.d * (wl_ivy - ul.by*tmp);
      ulst.mz = ulst.d * (wl_ivz - ul.bz*tmp);
      tmp = (sdld*sdl - bxsq)/(sdld*sdml - bxsq);
      ulst.by = ul.by * tmp;
      ulst.bz = ul.bz * tmp;
    }
    Real vbstl = (ulst.mx*bxi+(ulst.my*ulst.by+ulst.mz*ulst.bz))*ulst_d_inv;
    ulst.e = (sdl*ul.e - ptl*wl_ivx + ptst*spd[2] +
              bxi*(wl_ivx*bxi + (wl_ivy*ul.by + wl_ivz*ul.bz) - vbstl))*sdml_inv;

    urst.mx = urst.d * spd[2];
    if (fabs(sdrd*sdmr - bxsq) < (LHLLD_SMALL_NUMBER)*ptst) {
      urst.my = urst.d * wr_ivy;
      urst.mz = urst.d * wr_ivz;
      urst.by = ur.by;
      urst.bz = ur.bz;
    } else {
      Real tmp = bxi*(sdr - sdmr)/(sdrd*sdmr - bxsq);
      urst.my = urst.d * (wr_ivy - ur.by*tmp);
      urst.mz = urst.d * (wr_ivz - ur.bz*tmp);
      tmp = (sdrd*sdr - bxsq)/(sdrd*sdmr - bxsq);
      urst.by = ur.by * tmp;
      urst.bz = ur.bz * tmp;
    }
    Real vbstr = (urst.mx*bxi+(urst.my*urst.by+urst.mz*urst.bz))*urst_d_inv;
    urst.e = (sdr*ur.e - ptr*wr_ivx + ptst*spd[2] +
              bxi*(wr_ivx*bxi + (wr_ivy*ur.by + wr_ivz*ur.bz) - vbstr))*sdmr_inv;

    if (0.5*bxsq < (LHLLD_SMALL_NUMBER)*ptst) {
      uldst = ulst;
      urdst = urst;
    } else {
      Real invsumd = 1.0/(sqrtdl + sqrtdr);
      Real bxsig = (bxi > 0.0 ? 1.0 : -1.0);

      uldst.d = ulst.d;
      urdst.d = urst.d;
      uldst.mx = ulst.mx;
      urdst.mx = urst.mx;

      Real tmp = invsumd*(sqrtdl*(ulst.my*ulst_d_inv) + sqrtdr*(urst.my*urst_d_inv) +
                          bxsig*(urst.by - ulst.by));
      uldst.my = uldst.d * tmp;
      urdst.my = urdst.d * tmp;

      tmp = invsumd*(sqrtdl*(ulst.mz*ulst_d_inv) + sqrtdr*(urst.mz*urst_d_inv) +
                     bxsig*(urst.bz - ulst.bz));
      uldst.mz = uldst.d * tmp;
      urdst.mz = urdst.d * tmp;

      tmp = invsumd*(sqrtdl*urst.by + sqrtdr*ulst.by +
                     bxsig*sqrtdl*sqrtdr*((urst.my*urst_d_inv) - (ulst.my*ulst_d_inv)));
      uldst.by = urdst.by = tmp;

      tmp = invsumd*(sqrtdl*urst.bz + sqrtdr*ulst.bz +
                     bxsig*sqrtdl*sqrtdr*((urst.mz*urst_d_inv) - (ulst.mz*ulst_d_inv)));
      uldst.bz = urdst.bz = tmp;

      tmp = spd[2]*bxi + (uldst.my*uldst.by + uldst.mz*uldst.bz)/uldst.d;
      uldst.e = ulst.e - sqrtdl*bxsig*(vbstl - tmp);
      urdst.e = urst.e + sqrtdr*bxsig*(vbstr - tmp);
    }

    uldst.d = spd[1] * (uldst.d - ulst.d);
    uldst.mx = spd[1] * (uldst.mx - ulst.mx);
    uldst.my = spd[1] * (uldst.my - ulst.my);
    uldst.mz = spd[1] * (uldst.mz - ulst.mz);
    uldst.e = spd[1] * (uldst.e - ulst.e);
    uldst.by = spd[1] * (uldst.by - ulst.by);
    uldst.bz = spd[1] * (uldst.bz - ulst.bz);

    ulst.d = spd[0] * (ulst.d - ul.d);
    ulst.mx = spd[0] * (ulst.mx - ul.mx);
    ulst.my = spd[0] * (ulst.my - ul.my);
    ulst.mz = spd[0] * (ulst.mz - ul.mz);
    ulst.e = spd[0] * (ulst.e - ul.e);
    ulst.by = spd[0] * (ulst.by - ul.by);
    ulst.bz = spd[0] * (ulst.bz - ul.bz);

    urdst.d = spd[3] * (urdst.d - urst.d);
    urdst.mx = spd[3] * (urdst.mx - urst.mx);
    urdst.my = spd[3] * (urdst.my - urst.my);
    urdst.mz = spd[3] * (urdst.mz - urst.mz);
    urdst.e = spd[3] * (urdst.e - urst.e);
    urdst.by = spd[3] * (urdst.by - urst.by);
    urdst.bz = spd[3] * (urdst.bz - urst.bz);

    urst.d = spd[4] * (urst.d  - ur.d);
    urst.mx = spd[4] * (urst.mx - ur.mx);
    urst.my = spd[4] * (urst.my - ur.my);
    urst.mz = spd[4] * (urst.mz - ur.mz);
    urst.e = spd[4] * (urst.e - ur.e);
    urst.by = spd[4] * (urst.by - ur.by);
    urst.bz = spd[4] * (urst.bz - ur.bz);

    if (spd[0] >= 0.0) {
      flxi = fl;
    } else if (spd[4] <= 0.0) {
      flxi = fr;
    } else if (spd[1] >= 0.0) {
      flxi.d = fl.d  + ulst.d;
      flxi.mx = fl.mx + ulst.mx;
      flxi.my = fl.my + ulst.my;
      flxi.mz = fl.mz + ulst.mz;
      flxi.e  = fl.e  + ulst.e;
      flxi.by = fl.by + ulst.by;
      flxi.bz = fl.bz + ulst.bz;
    } else if (spd[2] >= 0.0) {
      flxi.d = fl.d  + ulst.d + uldst.d;
      flxi.mx = fl.mx + ulst.mx + uldst.mx;
      flxi.my = fl.my + ulst.my + uldst.my;
      flxi.mz = fl.mz + ulst.mz + uldst.mz;
      flxi.e  = fl.e  + ulst.e + uldst.e;
      flxi.by = fl.by + ulst.by + uldst.by;
      flxi.bz = fl.bz + ulst.bz + uldst.bz;
    } else if (spd[3] > 0.0) {
      flxi.d = fr.d + urst.d + urdst.d;
      flxi.mx = fr.mx + urst.mx + urdst.mx;
      flxi.my = fr.my + urst.my + urdst.my;
      flxi.mz = fr.mz + urst.mz + urdst.mz;
      flxi.e  = fr.e + urst.e + urdst.e;
      flxi.by = fr.by + urst.by + urdst.by;
      flxi.bz = fr.bz + urst.bz + urdst.bz;
    } else {
      flxi.d = fr.d  + urst.d;
      flxi.mx = fr.mx + urst.mx;
      flxi.my = fr.my + urst.my;
      flxi.mz = fr.mz + urst.mz;
      flxi.e  = fr.e  + urst.e;
      flxi.by = fr.by + urst.by;
      flxi.bz = fr.bz + urst.bz;
    }

    flx(m,IDN,k,j,i) = flxi.d;
    flx(m,ivx,k,j,i) = flxi.mx;
    flx(m,ivy,k,j,i) = flxi.my;
    flx(m,ivz,k,j,i) = flxi.mz;
    flx(m,IEN,k,j,i) = flxi.e;
    ey(m,k,j,i) = -flxi.by;
    ez(m,k,j,i) =  flxi.bz;
  });

  return;
}
} // namespace mhd
#endif // MHD_RSOLVERS_LHLLD_MHD_HPP_
