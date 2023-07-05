//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file bvals_dustdiffusion.cpp
//! \brief implements boundary functions for DustDiffusion variables and utilities to manage
//! primitive/conservative variable relationship in a derived class of the
//! CellCenteredBoundaryVariable base class.

// C headers

// C++ headers

// Athena++ headers
#include "../../../athena.hpp"
#include "../../../dustfluids/dustfluids.hpp"
#include "../../../dustfluids/dustfluids_diffusion_cc/cell_center_diffusions.hpp"
#include "../../../mesh/mesh.hpp"
#include "../../../utils/buffer_utils.hpp"
#include "bvals_dustdiffusion.hpp"

//----------------------------------------------------------------------------------------
//! \fn DustDiffusionBoundaryVariable::DustDiffusionBoundaryVariable
//! \brief

DustDiffusionBoundaryVariable::DustDiffusionBoundaryVariable(
    MeshBlock *pmb, AthenaArray<Real> *var_dustdiffusion, AthenaArray<Real> *coarse_var,
    AthenaArray<Real> *var_flux,
    DustDiffusionBoundaryQuantity dustdiffusion_type) :
    CellCenteredBoundaryVariable(pmb, var_dustdiffusion, coarse_var, var_flux),
    dustdiffusion_type_(dustdiffusion_type) {
    flip_across_pole_ = flip_across_pole_dustdiffusion;
}

//----------------------------------------------------------------------------------------
//! \fn void DustDiffusionBoundaryVariable::SelectCoarseBuffer(DustDiffusionBoundaryQuantity type)
//! \brief

void DustDiffusionBoundaryVariable::SelectCoarseBuffer(DustDiffusionBoundaryQuantity dustdiffusion_type) {
  if (pmy_mesh_->multilevel) {
    switch (dustdiffusion_type) {
      case (DustDiffusionBoundaryQuantity::cons_diff): {
        coarse_buf = &(pmy_block_->pdustfluids->dfccdif.coarse_diff_mom_cc_);
        break;
      }
    }
  }
  dustdiffusion_type_ = dustdiffusion_type;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void DustDiffusionBoundaryVariable::SwapDustDiffusionQuantity
//! \brief
//! \todo (felker):
//! * make general (but restricted) setter fns in CellCentered and FaceCentered

void DustDiffusionBoundaryVariable::SwapDustDiffusionQuantity(AthenaArray<Real> &var_dustdiffusion,
                                              DustDiffusionBoundaryQuantity dustdiffusion_type) {
  var_cc = &var_dustdiffusion;
  SelectCoarseBuffer(dustdiffusion_type);
  return;
}


//----------------------------------------------------------------------------------------
//! \fn void DustDiffusionBoundaryVariable::SetBoundarySameLevel(Real *buf,
//!                                                      const NeighborBlock& nb)
//! \brief Set dustdiffusion boundary received from a block on the same level

void DustDiffusionBoundaryVariable::SetBoundarySameLevel(Real *buf,
                                                 const NeighborBlock& nb) {
  MeshBlock *pmb = pmy_block_;
  int si, sj, sk, ei, ej, ek;
  AthenaArray<Real> &var = *var_cc;

  if (nb.ni.ox1 == 0)     si = pmb->is,          ei = pmb->ie;
  else if (nb.ni.ox1 > 0) si = pmb->ie + 1,      ei = pmb->ie + NGHOST;
  else                    si = pmb->is - NGHOST, ei = pmb->is - 1;
  if (nb.ni.ox2 == 0)     sj = pmb->js,          ej = pmb->je;
  else if (nb.ni.ox2 > 0) sj = pmb->je + 1,      ej = pmb->je + NGHOST;
  else                    sj = pmb->js - NGHOST, ej = pmb->js - 1;
  if (nb.ni.ox3 == 0)     sk = pmb->ks,          ek = pmb->ke;
  else if (nb.ni.ox3 > 0) sk = pmb->ke + 1,      ek = pmb->ke + NGHOST;
  else                    sk = pmb->ks - NGHOST, ek = pmb->ks - 1;

  int p = 0;

  if (nb.polar) {
    for (int n=nl_; n<=nu_; ++n) {
      Real sign = 1.0;
      if (flip_across_pole_ != nullptr) sign = flip_across_pole_[n%4] ? -1.0 : 1.0;
      for (int k=sk; k<=ek; ++k) {
        for (int j=ej; j>=sj; --j) {
#pragma omp simd linear(p)
          for (int i=si; i<=ei; ++i) {
            var(n, k, j, i) = sign * buf[p++];
          }
        }
      }
    }
  } else {
    BufferUtility::UnpackData(buf, var, nl_, nu_, si, ei, sj, ej, sk, ek, p);
  }

  if (pbval_->shearing_box == 2) {
    // 2D shearing box in x-z plane: additional step to shift azimuthal velocity
    int sign[2]{1, -1};
    Real qomL = pbval_->qomL_;
    for (int upper=0; upper<2; upper++) {
      if ((pmb->loc.lx1 == pbval_->loc_shear[upper]) && (sign[upper]*nb.ni.ox1 < 0)) {
        for (int n=0; n<NDUSTFLUIDS; ++n) {
          int dust_id = n;
          int rho_id  = 4*dust_id;
          int v3_id   = rho_id + 3;
          for (int k=sk; k<=ek; ++k) {
            for (int j=sj; j<=ej; ++j) {
              for (int i=si; i<=ei; ++i) {
                var(v3_id, k, j, i) += sign[upper]*qomL*var(rho_id, k, j, i);
              }
            }
          }
        }
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void DustDiffusionBoundaryVariable::SetBoundaryFromCoarser(Real *buf,
//!                                                               const NeighborBlock& nb)
//! \brief Set cell-centered prolongation buffer received from a block on a coarser level

void DustDiffusionBoundaryVariable::SetBoundaryFromCoarser(Real *buf,
                                                          const NeighborBlock& nb) {
  MeshBlock *pmb = pmy_block_;
  int si, sj, sk, ei, ej, ek;
  int cng = pmb->cnghost;
  AthenaArray<Real> &coarse_var = *coarse_buf;

  if (nb.ni.ox1 == 0) {
    si = pmb->cis, ei = pmb->cie;
    if ((pmb->loc.lx1 & 1LL) == 0LL) ei += cng;
    else                             si -= cng;
  } else if (nb.ni.ox1 > 0)  {
    si = pmb->cie + 1,   ei = pmb->cie + cng;
  } else {
    si = pmb->cis - cng, ei = pmb->cis - 1;
  }
  if (nb.ni.ox2 == 0) {
    sj = pmb->cjs, ej = pmb->cje;
    if (pmb->block_size.nx2 > 1) {
      if ((pmb->loc.lx2 & 1LL) == 0LL) ej += cng;
      else                             sj -= cng;
    }
  } else if (nb.ni.ox2 > 0) {
    sj = pmb->cje + 1,   ej = pmb->cje + cng;
  } else {
    sj = pmb->cjs - cng, ej = pmb->cjs - 1;
  }
  if (nb.ni.ox3 == 0) {
    sk = pmb->cks, ek = pmb->cke;
    if (pmb->block_size.nx3 > 1) {
      if ((pmb->loc.lx3 & 1LL) == 0LL) ek += cng;
      else                             sk -= cng;
    }
  } else if (nb.ni.ox3 > 0)  {
    sk = pmb->cke + 1,   ek = pmb->cke + cng;
  } else {
    sk = pmb->cks - cng, ek = pmb->cks - 1;
  }

  int p = 0;
  if (nb.polar) {
    for (int n=nl_; n<=nu_; ++n) {
      Real sign = 1.0;
      if (flip_across_pole_ != nullptr) sign = flip_across_pole_[n%4] ? -1.0 : 1.0;
      for (int k=sk; k<=ek; ++k) {
        for (int j=ej; j>=sj; --j) {
#pragma omp simd linear(p)
          for (int i=si; i<=ei; ++i)
            coarse_var(n, k, j, i) = sign * buf[p++];
        }
      }
    }
  } else {
    BufferUtility::UnpackData(buf, coarse_var, nl_, nu_, si, ei, sj, ej, sk, ek, p);
  }
  return;
}


//----------------------------------------------------------------------------------------
//! \fn void CellCenteredBoundaryVariable::SetBoundaryFromFiner(Real *buf,
//!                                                             const NeighborBlock& nb)
//! \brief Set cell-centered boundary received from a block on a finer level

void DustDiffusionBoundaryVariable::SetBoundaryFromFiner(Real *buf,
                                                        const NeighborBlock& nb) {
  MeshBlock *pmb = pmy_block_;
  AthenaArray<Real> &var = *var_cc;
  // receive already restricted data
  int si, sj, sk, ei, ej, ek;

  if (nb.ni.ox1 == 0) {
    si = pmb->is, ei = pmb->ie;
    if (nb.ni.fi1 == 1)   si += pmb->block_size.nx1/2;
    else            ei -= pmb->block_size.nx1/2;
  } else if (nb.ni.ox1 > 0) {
    si = pmb->ie + 1,      ei = pmb->ie + NGHOST;
  } else {
    si = pmb->is - NGHOST, ei = pmb->is - 1;
  }
  if (nb.ni.ox2 == 0) {
    sj = pmb->js, ej = pmb->je;
    if (pmb->block_size.nx2 > 1) {
      if (nb.ni.ox1 != 0) {
        if (nb.ni.fi1 == 1) sj += pmb->block_size.nx2/2;
        else          ej -= pmb->block_size.nx2/2;
      } else {
        if (nb.ni.fi2 == 1) sj += pmb->block_size.nx2/2;
        else          ej -= pmb->block_size.nx2/2;
      }
    }
  } else if (nb.ni.ox2 > 0) {
    sj = pmb->je + 1,      ej = pmb->je + NGHOST;
  } else {
    sj = pmb->js - NGHOST, ej = pmb->js - 1;
  }
  if (nb.ni.ox3 == 0) {
    sk = pmb->ks, ek = pmb->ke;
    if (pmb->block_size.nx3 > 1) {
      if (nb.ni.ox1 != 0 && nb.ni.ox2 != 0) {
        if (nb.ni.fi1 == 1) sk += pmb->block_size.nx3/2;
        else          ek -= pmb->block_size.nx3/2;
      } else {
        if (nb.ni.fi2 == 1) sk += pmb->block_size.nx3/2;
        else          ek -= pmb->block_size.nx3/2;
      }
    }
  } else if (nb.ni.ox3 > 0) {
    sk = pmb->ke + 1,      ek = pmb->ke + NGHOST;
  } else {
    sk = pmb->ks - NGHOST, ek = pmb->ks - 1;
  }

  int p = 0;
  if (nb.polar) {
    for (int n=nl_; n<=nu_; ++n) {
      Real sign=1.0;
      if (flip_across_pole_ != nullptr) sign = flip_across_pole_[n%4] ? -1.0 : 1.0;
      for (int k=sk; k<=ek; ++k) {
        for (int j=ej; j>=sj; --j) {
#pragma omp simd linear(p)
          for (int i=si; i<=ei; ++i)
            var(n, k, j, i) = sign * buf[p++];
        }
      }
    }
  } else {
    BufferUtility::UnpackData(buf, var, nl_, nu_, si, ei, sj, ej, sk, ek, p);
  }
  return;
}
