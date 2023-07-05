//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file rk2implicit_integrator.cpp
//! Fully implicit Runge Kutta 2 drag time integrators

// C++ headers
#include <algorithm>   // min,max
#include <cstring>    // strcmp
#include <iostream>   // endl
#include <limits>
#include <sstream>

// Athena++ headers
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../../coordinates/coordinates.hpp"
#include "../../defs.hpp"
#include "../../mesh/mesh.hpp"
#include "../dustfluids.hpp"
#include "dust_gas_drag.hpp"


void DustGasDrag::RK2ImplicitFeedback(const int stage,
      const Real dt, const AthenaArray<Real> &stopping_time,
      const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
      AthenaArray<Real> &u, AthenaArray<Real> &cons_df) {

  int is = pmb_->is; int js = pmb_->js; int ks = pmb_->ks;
  int ie = pmb_->ie; int je = pmb_->je; int ke = pmb_->ke;

  bool Stage_I = (((orb_advection_  < 2) && (stage == 1)) ||
                  ((orb_advection_ == 2) && (stage == 2)));
  bool Drag_Work = (NON_BAROTROPIC_EOS && (!Dissipation_Flag));
  bool Drag_WorkDissipation = (NON_BAROTROPIC_EOS && Dissipation_Flag);

  const AthenaArray<Real> &w_n             = pmy_hydro_->w_n;
  const AthenaArray<Real> &prim_df_n       = pmy_dustfluids_->df_prim_n;
  const AthenaArray<Real> &stopping_time_n = pmy_dustfluids_->stopping_time_array_n;
  const AthenaArray<Real> &u_af_src        = pmy_hydro_->u_af_src;
  const AthenaArray<Real> &cons_df_af_src  = pmy_dustfluids_->df_cons_af_src;

  AthenaArray<Real> &Stage_I_delta_mom1 = pmy_dustfluids_->Stage_I_delta_mom1;
  AthenaArray<Real> &Stage_I_delta_mom2 = pmy_dustfluids_->Stage_I_delta_mom2;
  AthenaArray<Real> &Stage_I_delta_mom3 = pmy_dustfluids_->Stage_I_delta_mom3;

  AthenaArray<Real> &Stage_I_vel1 = pmy_dustfluids_->Stage_I_vel1;
  AthenaArray<Real> &Stage_I_vel2 = pmy_dustfluids_->Stage_I_vel2;
  AthenaArray<Real> &Stage_I_vel3 = pmy_dustfluids_->Stage_I_vel3;

  if (Stage_I) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          // Alias the primitives of gas at current stage
          const Real &gas_rho  = w(IDN, k, j, i);
          const Real &gas_vel1 = w(IM1, k, j, i);
          const Real &gas_vel2 = w(IM2, k, j, i);
          const Real &gas_vel3 = w(IM3, k, j, i);

          mom1_prim(0, i) = gas_rho*gas_vel1;
          mom2_prim(0, i) = gas_rho*gas_vel2;
          mom3_prim(0, i) = gas_rho*gas_vel3;

          // Combine the implicit drag part and the other explicit source terms
          Real &gas_mom1_bf_src = mom1_prim(0, i);
          Real &gas_mom2_bf_src = mom2_prim(0, i);
          Real &gas_mom3_bf_src = mom3_prim(0, i);

          temp_mom1(i) = gas_mom1_bf_src;
          temp_mom2(i) = gas_mom2_bf_src;
          temp_mom3(i) = gas_mom3_bf_src;
          temp_rho(i)  = gas_rho;

          delta_mom1_src(0, i) = (u_af_src(IM1, k, j, i) - gas_mom1_bf_src);
          delta_mom2_src(0, i) = (u_af_src(IM2, k, j, i) - gas_mom2_bf_src);
          delta_mom3_src(0, i) = (u_af_src(IM3, k, j, i) - gas_mom3_bf_src);

          temp_mom1(i) += delta_mom1_src(0, i);
          temp_mom2(i) += delta_mom2_src(0, i);
          temp_mom3(i) += delta_mom3_src(0, i);
        }

        for (int n=1; n<=NDUSTFLUIDS; ++n) {
          int dust_id = n - 1;
          int rho_id  = 4*dust_id;
          int v1_id   = rho_id + 1;
          int v2_id   = rho_id + 2;
          int v3_id   = rho_id + 3;
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            const Real &dust_rho  = prim_df(rho_id, k, j, i);
            const Real &dust_vel1 = prim_df(v1_id,  k, j, i);
            const Real &dust_vel2 = prim_df(v2_id,  k, j, i);
            const Real &dust_vel3 = prim_df(v3_id,  k, j, i);

            mom1_prim(n, i) = dust_rho*dust_vel1;
            mom2_prim(n, i) = dust_rho*dust_vel2;
            mom3_prim(n, i) = dust_rho*dust_vel3;

            Real &dust_mom1_bf_src = mom1_prim(n, i);
            Real &dust_mom2_bf_src = mom2_prim(n, i);
            Real &dust_mom3_bf_src = mom3_prim(n, i);

            delta_mom1_src(n, i) = (cons_df_af_src(v1_id, k, j, i) - dust_mom1_bf_src);
            delta_mom2_src(n, i) = (cons_df_af_src(v2_id, k, j, i) - dust_mom2_bf_src);
            delta_mom3_src(n, i) = (cons_df_af_src(v3_id, k, j, i) - dust_mom3_bf_src);

            alpha(n, i)       = 1.0/(stopping_time(dust_id, k, j, i) + TINY_NUMBER);
            qvalue(n, i)      = alpha(n, i)*dt;
            weight_dust(n, i) = 1.0/(1.0 + qvalue(n, i));
            weight_gas(n, i)  = qvalue(n, i)*weight_dust(n, i);

            temp_mom1(i) += weight_gas(n, i)*(dust_mom1_bf_src + delta_mom1_src(n, i));
            temp_mom2(i) += weight_gas(n, i)*(dust_mom2_bf_src + delta_mom2_src(n, i));
            temp_mom3(i) += weight_gas(n, i)*(dust_mom3_bf_src + delta_mom3_src(n, i));
            temp_rho(i)  += weight_gas(n, i)*dust_rho;
          }
        }

#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          // Alias the primitives of gas at current stage
          const Real &gas_rho  = w(IDN, k, j, i);

          // Alias the conserves of gas
          Real &gas_dens = u(IDN, k, j, i);
          Real &gas_mom1 = u(IM1, k, j, i);
          Real &gas_mom2 = u(IM2, k, j, i);
          Real &gas_mom3 = u(IM3, k, j, i);

          temp_inv_rho(i)    = 1.0/temp_rho(i);
          temp_total_vel1(i) = temp_mom1(i)*temp_inv_rho(i);
          temp_total_vel2(i) = temp_mom2(i)*temp_inv_rho(i);
          temp_total_vel3(i) = temp_mom3(i)*temp_inv_rho(i);

          delta_mom1(0, i) = gas_rho*temp_total_vel1(i) - mom1_prim(0, i);
          delta_mom2(0, i) = gas_rho*temp_total_vel2(i) - mom2_prim(0, i);
          delta_mom3(0, i) = gas_rho*temp_total_vel3(i) - mom3_prim(0, i);

          // Calculate the gas velocity before drags
          Real inv_gas_dens     = 0.0;
          Real gas_vel1_bf_drag = 0.0;
          Real gas_vel2_bf_drag = 0.0;
          Real gas_vel3_bf_drag = 0.0;

          if (Drag_Work) {
            inv_gas_dens     = 1.0/gas_dens;
            gas_vel1_bf_drag = gas_mom1*inv_gas_dens;
            gas_vel2_bf_drag = gas_mom2*inv_gas_dens;
            gas_vel3_bf_drag = gas_mom3*inv_gas_dens;

            Stage_I_vel1(0, k, j, i) = gas_vel1_bf_drag;
            Stage_I_vel2(0, k, j, i) = gas_vel2_bf_drag;
            Stage_I_vel3(0, k, j, i) = gas_vel3_bf_drag;
          }

          Real gas_delta_mom1_drag = (delta_mom1(0, i) - delta_mom1_src(0, i));
          Real gas_delta_mom2_drag = (delta_mom2(0, i) - delta_mom2_src(0, i));
          Real gas_delta_mom3_drag = (delta_mom3(0, i) - delta_mom3_src(0, i));

          Stage_I_delta_mom1(0, k, j, i) = gas_delta_mom1_drag;
          Stage_I_delta_mom2(0, k, j, i) = gas_delta_mom2_drag;
          Stage_I_delta_mom3(0, k, j, i) = gas_delta_mom3_drag;

          gas_mom1 += gas_delta_mom1_drag;
          gas_mom2 += gas_delta_mom2_drag;
          gas_mom3 += gas_delta_mom3_drag;

          // Update the energy of gas if gas is non barotropic.
          // dE_gas = dM_gas*(v_gas_before_drag + v_gas_after_drag)/2
          if (Drag_Work) {
            // Calculate the gas velocity after drags
            Real gas_vel1_af_drag = gas_mom1*inv_gas_dens;
            Real gas_vel2_af_drag = gas_mom2*inv_gas_dens;
            Real gas_vel3_af_drag = gas_mom3*inv_gas_dens;

            Real work_drag = 0.5*(gas_delta_mom1_drag*(gas_vel1_bf_drag + gas_vel1_af_drag) +
                                  gas_delta_mom2_drag*(gas_vel2_bf_drag + gas_vel2_af_drag) +
                                  gas_delta_mom3_drag*(gas_vel3_bf_drag + gas_vel3_af_drag));

            Real &gas_erg  = u(IEN, k, j, i);
            gas_erg       += work_drag;
          }
        }

        for (int n=1; n<=NDUSTFLUIDS; ++n) {
          int dust_id = n - 1;
          int rho_id  = 4*dust_id;
          int v1_id   = rho_id + 1;
          int v2_id   = rho_id + 2;
          int v3_id   = rho_id + 3;
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            const Real &dust_rho = prim_df(rho_id, k, j, i);

            // Alias the conserves of dust
            Real &dust_dens = cons_df(rho_id, k, j, i);
            Real &dust_mom1 = cons_df(v1_id,  k, j, i);
            Real &dust_mom2 = cons_df(v2_id,  k, j, i);
            Real &dust_mom3 = cons_df(v3_id,  k, j, i);

            // Calculate the dust velocity before drags
            Real inv_dust_dens     = 0.0;
            Real dust_vel1_bf_drag = 0.0;
            Real dust_vel2_bf_drag = 0.0;
            Real dust_vel3_bf_drag = 0.0;

            if (Drag_WorkDissipation) {
              inv_dust_dens     = 1.0/dust_dens;
              dust_vel1_bf_drag = dust_mom1*inv_dust_dens;
              dust_vel2_bf_drag = dust_mom2*inv_dust_dens;
              dust_vel3_bf_drag = dust_mom3*inv_dust_dens;

              Stage_I_vel1(n, k, j, i) = dust_vel1_bf_drag;
              Stage_I_vel2(n, k, j, i) = dust_vel2_bf_drag;
              Stage_I_vel3(n, k, j, i) = dust_vel3_bf_drag;
            }

            delta_mom1(n, i) = weight_dust(n, i)*(mom1_prim(n, i) + delta_mom1_src(n, i) + qvalue(n, i)*dust_rho*temp_total_vel1(i)) - mom1_prim(n, i);
            delta_mom2(n, i) = weight_dust(n, i)*(mom2_prim(n, i) + delta_mom2_src(n, i) + qvalue(n, i)*dust_rho*temp_total_vel2(i)) - mom2_prim(n, i);
            delta_mom3(n, i) = weight_dust(n, i)*(mom3_prim(n, i) + delta_mom3_src(n, i) + qvalue(n, i)*dust_rho*temp_total_vel3(i)) - mom3_prim(n, i);

            // Add the delta momentum caused by drags on dust conserves
            Real dust_delta_mom1_drag = (delta_mom1(n, i) - delta_mom1_src(n, i));
            Real dust_delta_mom2_drag = (delta_mom2(n, i) - delta_mom2_src(n, i));
            Real dust_delta_mom3_drag = (delta_mom3(n, i) - delta_mom3_src(n, i));

            Stage_I_delta_mom1(n, k, j, i) = dust_delta_mom1_drag;
            Stage_I_delta_mom2(n, k, j, i) = dust_delta_mom2_drag;
            Stage_I_delta_mom3(n, k, j, i) = dust_delta_mom3_drag;

            dust_mom1 += dust_delta_mom1_drag;
            dust_mom2 += dust_delta_mom2_drag;
            dust_mom3 += dust_delta_mom3_drag;

            // Add the energy dissipation of drags if gas is non barotropic.
            if (Drag_WorkDissipation) {
              // Calculate the dust velocity after drags
              Real dust_vel1_af_drag = dust_mom1*inv_dust_dens;
              Real dust_vel2_af_drag = dust_mom2*inv_dust_dens;
              Real dust_vel3_af_drag = dust_mom3*inv_dust_dens;

              Real dissipation = 0.5*(dust_delta_mom1_drag*(dust_vel1_bf_drag + dust_vel1_af_drag) +
                                      dust_delta_mom2_drag*(dust_vel2_bf_drag + dust_vel2_af_drag) +
                                      dust_delta_mom3_drag*(dust_vel3_bf_drag + dust_vel3_af_drag));

              Real &gas_erg  = u(IEN, k, j, i);
              gas_erg       -= dissipation;
            }
          }
        }
      }
    }
  } else {
    Real inv_dt = 1.0/(2.0*dt);
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        force_x1.ZeroClear();
        force_x2.ZeroClear();
        force_x3.ZeroClear();

        force_x1_n.ZeroClear();
        force_x2_n.ZeroClear();
        force_x3_n.ZeroClear();

        jacobi.ZeroClear();
        jacobi_n.ZeroClear();

        temp_A.ZeroClear();
        temp_B.ZeroClear();
        temp_C.ZeroClear();
        temp_D.ZeroClear();

        product.ZeroClear();
        idx_vector.ZeroClear();
        lu_matrix.ZeroClear();
        lambda.ZeroClear();
        lambda_inv.ZeroClear();

        delta_mom1.ZeroClear();
        delta_mom2.ZeroClear();
        delta_mom3.ZeroClear();

        delta_mom1_im.ZeroClear();
        delta_mom2_im.ZeroClear();
        delta_mom3_im.ZeroClear();

        delta_mom1_im_II.ZeroClear();
        delta_mom2_im_II.ZeroClear();
        delta_mom3_im_II.ZeroClear();

        inv_gas_rho.ZeroClear();
        inv_gas_rho_n.ZeroClear();
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          const Real &gas_rho = w(IDN, k, j, i);
          inv_gas_rho(i)      = 1.0/gas_rho;

          const Real &gas_rho_n  = w_n(IDN, k, j, i);
          const Real &gas_vel1_n = w_n(IM1, k, j, i);
          const Real &gas_vel2_n = w_n(IM2, k, j, i);
          const Real &gas_vel3_n = w_n(IM3, k, j, i);
          inv_gas_rho_n(i)       = 1.0/gas_rho_n;

          mom1_prim_n(0, i) = gas_rho_n*gas_vel1_n;
          mom2_prim_n(0, i) = gas_rho_n*gas_vel2_n;
          mom3_prim_n(0, i) = gas_rho_n*gas_vel3_n;

          temp_A(0, 0, i) = 1.0;
          temp_C(0, 0, i) = 1.0;
        }

        for (int idx=1; idx<=NDUSTFLUIDS; ++idx) {
          int dust_id = idx - 1;
          int rho_id  = 4*dust_id;
          int v1_id   = rho_id + 1;
          int v2_id   = rho_id + 2;
          int v3_id   = rho_id + 3;
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            // Alias the primitives of dust at stage n
            const Real &dust_rho_n  = prim_df_n(rho_id, k, j, i);
            const Real &dust_vel1_n = prim_df_n(v1_id, k, j, i);
            const Real &dust_vel2_n = prim_df_n(v2_id, k, j, i);
            const Real &dust_vel3_n = prim_df_n(v3_id, k, j, i);
            alpha_n(idx, i)         = 1.0/(stopping_time_n(dust_id, k, j, i) + TINY_NUMBER);
            epsilon_n(idx, i)       = dust_rho_n*inv_gas_rho_n(i);

            // Alias the primitives of dust at current stage
            const Real &dust_rho = prim_df(rho_id, k, j, i);
            alpha(idx, i)        = 1.0/(stopping_time(dust_id, k, j, i) + TINY_NUMBER);
            epsilon(idx, i)      = dust_rho*inv_gas_rho(i);

            mom1_prim_n(idx, i) = dust_rho_n*dust_vel1_n;
            mom2_prim_n(idx, i) = dust_rho_n*dust_vel2_n;
            mom3_prim_n(idx, i) = dust_rho_n*dust_vel3_n;

            Real &dust_mom1_bf_src = mom1_prim_n(idx, i);
            Real &dust_mom2_bf_src = mom2_prim_n(idx, i);
            Real &dust_mom3_bf_src = mom3_prim_n(idx, i);

            force_x1_n(idx, i) = alpha_n(idx, i)*(epsilon_n(idx, i)*mom1_prim_n(0, i) - dust_mom1_bf_src);
            force_x2_n(idx, i) = alpha_n(idx, i)*(epsilon_n(idx, i)*mom2_prim_n(0, i) - dust_mom2_bf_src);
            force_x3_n(idx, i) = alpha_n(idx, i)*(epsilon_n(idx, i)*mom3_prim_n(0, i) - dust_mom3_bf_src);

            force_x1(idx, i) = alpha(idx, i)*(epsilon(idx, i)*mom1_prim_n(0, i) - dust_mom1_bf_src);
            force_x2(idx, i) = alpha(idx, i)*(epsilon(idx, i)*mom2_prim_n(0, i) - dust_mom2_bf_src);
            force_x3(idx, i) = alpha(idx, i)*(epsilon(idx, i)*mom3_prim_n(0, i) - dust_mom3_bf_src);

            force_x1_n(0, i) -= force_x1_n(idx, i);
            force_x2_n(0, i) -= force_x2_n(idx, i);
            force_x3_n(0, i) -= force_x3_n(idx, i);

            force_x1(0, i) -= force_x1(idx, i);
            force_x2(0, i) -= force_x2(idx, i);
            force_x3(0, i) -= force_x3(idx, i);

            delta_mom1(idx, i) = -0.5*Stage_I_delta_mom1(idx, k, j, i);
            delta_mom2(idx, i) = -0.5*Stage_I_delta_mom2(idx, k, j, i);
            delta_mom3(idx, i) = -0.5*Stage_I_delta_mom3(idx, k, j, i);

            delta_mom1_src(idx, i) = (cons_df_af_src(v1_id, k, j, i) + delta_mom1(idx, i) - dust_mom1_bf_src);
            delta_mom2_src(idx, i) = (cons_df_af_src(v2_id, k, j, i) + delta_mom2(idx, i) - dust_mom2_bf_src);
            delta_mom3_src(idx, i) = (cons_df_af_src(v3_id, k, j, i) + delta_mom3(idx, i) - dust_mom3_bf_src);

            Real dust_delta_force_x1_src = delta_mom1_src(idx, i)*inv_dt;
            Real dust_delta_force_x2_src = delta_mom2_src(idx, i)*inv_dt;
            Real dust_delta_force_x3_src = delta_mom3_src(idx, i)*inv_dt;

            force_x1_n(idx, i) += dust_delta_force_x1_src;
            force_x2_n(idx, i) += dust_delta_force_x2_src;
            force_x3_n(idx, i) += dust_delta_force_x3_src;

            force_x1(idx, i) += dust_delta_force_x1_src;
            force_x2(idx, i) += dust_delta_force_x2_src;
            force_x3(idx, i) += dust_delta_force_x3_src;
          }
        }

#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          delta_mom1(0, i) = -0.5*Stage_I_delta_mom1(0, k, j, i);
          delta_mom2(0, i) = -0.5*Stage_I_delta_mom2(0, k, j, i);
          delta_mom3(0, i) = -0.5*Stage_I_delta_mom3(0, k, j, i);

          delta_mom1_src(0, i) = (u_af_src(IM1, k, j, i) + delta_mom1(0, i) - mom1_prim_n(0, i));
          delta_mom2_src(0, i) = (u_af_src(IM2, k, j, i) + delta_mom2(0, i) - mom2_prim_n(0, i));
          delta_mom3_src(0, i) = (u_af_src(IM3, k, j, i) + delta_mom3(0, i) - mom3_prim_n(0, i));

          Real gas_delta_force_x1_src = delta_mom1_src(0, i)*inv_dt;
          Real gas_delta_force_x2_src = delta_mom2_src(0, i)*inv_dt;
          Real gas_delta_force_x3_src = delta_mom3_src(0, i)*inv_dt;

          force_x1_n(0, i) += gas_delta_force_x1_src;
          force_x2_n(0, i) += gas_delta_force_x2_src;
          force_x3_n(0, i) += gas_delta_force_x3_src;

          force_x1(0, i) += gas_delta_force_x1_src;
          force_x2(0, i) += gas_delta_force_x2_src;
          force_x3(0, i) += gas_delta_force_x3_src;
        }

        // Calculate the jacobi matrix of the drag forces, df/dM|^(n)
        for (int idx=1; idx<=NDUSTFLUIDS; ++idx) {
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            jacobi(0, idx, i)   = epsilon(idx, i)*alpha(idx, i);
            jacobi_n(0, idx, i) = epsilon_n(idx, i)*alpha_n(idx, i);

            jacobi(idx, 0, i)   = alpha(idx, i);
            jacobi_n(idx, 0, i) = alpha_n(idx, i);

            jacobi(0, 0, i)   -= jacobi(0, idx, i);
            jacobi_n(0, 0, i) -= jacobi_n(0, idx, i);

            jacobi(idx, idx, i)   = -jacobi(idx, 0, i);
            jacobi_n(idx, idx, i) = -jacobi_n(idx, 0, i);
          }
        }

        Real double_dt = 2.0*dt;
        for (int idx=1; idx<=NDUSTFLUIDS; ++idx) {
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            temp_A(0,   idx, i) =     -double_dt*jacobi_n(0,   idx, i);
            temp_A(idx, 0,   i) =     -double_dt*jacobi_n(idx, 0,   i);
            temp_A(idx, idx, i) = 1.0 -double_dt*jacobi_n(idx, idx, i);

            temp_C(0,   idx, i) =     -double_dt*jacobi(0,   idx, i);
            temp_C(idx, 0,   i) =     -double_dt*jacobi(idx, 0,   i);
            temp_C(idx, idx, i) = 1.0 -double_dt*jacobi(idx, idx, i);
          }
        }

#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          temp_A(0, 0, i) = 1.0 - double_dt*jacobi_n(0, 0, i);
          temp_C(0, 0, i) = 1.0 - double_dt*jacobi(0,   0, i);
        }


        // Calculate the product matrix, df/dM|^(')*df/dM|^(n)
        for (int jdx=1; jdx<=NDUSTFLUIDS; ++jdx) {
          for (int idx=1; idx<=NDUSTFLUIDS; ++idx) {
#pragma omp simd
            for (int i=is; i<=ie; ++i) {
              product(jdx, idx, i) = jacobi_n(0, idx, i)*jacobi(jdx, 0, i);
            }
          }
        }

        for (int idx=1; idx<=NDUSTFLUIDS; ++idx) {
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            product(idx, idx, i) += jacobi(idx, idx, i)*jacobi_n(idx, idx, i);
          }
        }

        for (int idx=1; idx<=NDUSTFLUIDS; ++idx) {
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            product(0, idx, i) = jacobi_n(0, idx, i)*jacobi(0, 0, i) + jacobi_n(idx, idx, i)*jacobi(0, idx, i);
          }
        }

        for (int jdx=1; jdx<=NDUSTFLUIDS; ++jdx) {
          for (int idx=1; idx<=NDUSTFLUIDS; ++idx) {
#pragma omp simd
            for (int i=is; i<=ie; ++i) {
              product(jdx, 0, i) -= product(jdx, idx, i);
            }
          }
        }

        for (int idx=1; idx<=NDUSTFLUIDS; ++idx) {
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            product(0, 0, i) -= product(0, idx, i);
          }
        }

        //// calculate temp_A = I - h*jacobi
        //Add(1.0, -2.0*dt, jacobi_n, temp_A);

        //// calculate temp_C = I - h*jacobi
        //Add(1.0, -2.0*dt, jacobi, temp_C);

        // calculate lambda = temp_A + 0.5*h^2*product
        Add(temp_A, 2.0*SQR(dt), product, lambda);

        // cauculate the inverse matrix of lambda
        LUdecompose(lambda, idx_vector, lu_matrix);
        Inverse(idx_vector, lu_matrix, lambda, lambda_inv);

        // calculate temp_B = 0.5*h*lambda_inv
        Multiply(dt, lambda_inv, temp_B);

        // calculate temp_D = 0.5*h*lambda_inv*temp_C
        for(int jdx=0; jdx<=NDUSTFLUIDS; ++jdx) {
          for(int idx=0; idx<=NDUSTFLUIDS; ++idx) {
#pragma omp simd
            for (int i=is; i<=ie; ++i) {
              temp_D(jdx, idx, i) += dt*lambda_inv(jdx, 0, i)*temp_C(0, idx, i);
            }
          }
        }

        for(int jdx=0; jdx<=NDUSTFLUIDS; ++jdx) {
          for(int idx=1; idx<=NDUSTFLUIDS; ++idx) {
#pragma omp simd
            for (int i=is; i<=ie; ++i) {
              temp_D(jdx, 0, i) += dt*lambda_inv(jdx, idx, i)*temp_C(idx, 0, i);
            }
          }
        }

        for(int jdx=0; jdx<=NDUSTFLUIDS; ++jdx) {
          for(int idx=1; idx<=NDUSTFLUIDS; ++idx) {
#pragma omp simd
            for (int i=is; i<=ie; ++i) {
              temp_D(jdx, idx, i) += dt*lambda_inv(jdx, idx, i)*temp_C(idx, idx, i);
            }
          }
        }

        //Multiply(lambda_inv, temp_C, temp_D);
        //Multiply(dt, temp_D);

        // Delta_M_im = temp_B*force
        MultiplyVector(temp_B, force_x1, delta_mom1_im);
        MultiplyVector(temp_B, force_x2, delta_mom2_im);
        MultiplyVector(temp_B, force_x3, delta_mom3_im);

        // Delta_M_im_II = temp_D*force_n
        MultiplyVector(temp_D, force_x1_n, delta_mom1_im_II);
        MultiplyVector(temp_D, force_x2_n, delta_mom2_im_II);
        MultiplyVector(temp_D, force_x3_n, delta_mom3_im_II);

#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          // Alias the primitives of gas at current stage
          const Real &gas_vel1 = w(IM1, k, j, i);
          const Real &gas_vel2 = w(IM2, k, j, i);
          const Real &gas_vel3 = w(IM3, k, j, i);

          // Alias the conserves of gas
          Real &gas_dens = u(IDN, k, j, i);
          Real &gas_mom1 = u(IM1, k, j, i);
          Real &gas_mom2 = u(IM2, k, j, i);
          Real &gas_mom3 = u(IM3, k, j, i);

          Real delta_gas_mom1_drag = delta_mom1_im(0, i) + delta_mom1_im_II(0, i) - delta_mom1_src(0, i);
          Real delta_gas_mom2_drag = delta_mom2_im(0, i) + delta_mom2_im_II(0, i) - delta_mom2_src(0, i);
          Real delta_gas_mom3_drag = delta_mom3_im(0, i) + delta_mom3_im_II(0, i) - delta_mom3_src(0, i);

          // Add the delta momentum caused by drags on gas conserves
          gas_mom1 += delta_mom1(0, i);
          gas_mom2 += delta_mom2(0, i);
          gas_mom3 += delta_mom3(0, i);

          gas_mom1 += delta_gas_mom1_drag;
          gas_mom2 += delta_gas_mom2_drag;
          gas_mom3 += delta_gas_mom3_drag;

          // Update the energy of gas if gas is non barotropic.
          // dE_gas = dM_gas*(v_gas_before_drag + v_gas_after_drag)/2
          if (Drag_Work) {
            // Calculate the gas velocity after drags
            Real inv_gas_dens     = 1.0/gas_dens;
            Real gas_vel1_af_drag = gas_mom1*inv_gas_dens;
            Real gas_vel2_af_drag = gas_mom2*inv_gas_dens;
            Real gas_vel3_af_drag = gas_mom3*inv_gas_dens;

            Real work_drag_I = 0.5*(delta_mom1(0, i)*(Stage_I_vel1(0, k, j, i) + gas_vel1) +
                                    delta_mom2(0, i)*(Stage_I_vel2(0, k, j, i) + gas_vel2) +
                                    delta_mom3(0, i)*(Stage_I_vel3(0, k, j, i) + gas_vel3));

            Real work_drag = 0.5*(delta_gas_mom1_drag*(Stage_I_vel1(0, k, j, i) + gas_vel1_af_drag) +
                                  delta_gas_mom2_drag*(Stage_I_vel2(0, k, j, i) + gas_vel2_af_drag) +
                                  delta_gas_mom3_drag*(Stage_I_vel3(0, k, j, i) + gas_vel3_af_drag));

            Real &gas_erg  = u(IEN, k, j, i);
            gas_erg       += (work_drag_I + work_drag);
          }
        }

        for (int idx=1; idx<=NDUSTFLUIDS; ++idx) {
          int dust_id = idx - 1;
          int rho_id  = 4*dust_id;
          int v1_id   = rho_id + 1;
          int v2_id   = rho_id + 2;
          int v3_id   = rho_id + 3;
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            // Alias the primitives of dust
            const Real &dust_vel1 = prim_df(v1_id, k, j, i);
            const Real &dust_vel2 = prim_df(v2_id, k, j, i);
            const Real &dust_vel3 = prim_df(v3_id, k, j, i);

            // Alias the conserves of dust
            Real &dust_dens = cons_df(rho_id, k, j, i);
            Real &dust_mom1 = cons_df(v1_id,  k, j, i);
            Real &dust_mom2 = cons_df(v2_id,  k, j, i);
            Real &dust_mom3 = cons_df(v3_id,  k, j, i);

            Real delta_dust_mom1_drag = delta_mom1_im(idx, i) + delta_mom1_im_II(idx, i) - delta_mom1_src(idx, i);
            Real delta_dust_mom2_drag = delta_mom2_im(idx, i) + delta_mom2_im_II(idx, i) - delta_mom2_src(idx, i);
            Real delta_dust_mom3_drag = delta_mom3_im(idx, i) + delta_mom3_im_II(idx, i) - delta_mom3_src(idx, i);

            // Add the delta momentum caused by drags on dust conserves
            dust_mom1 += delta_mom1(idx, i);
            dust_mom2 += delta_mom2(idx, i);
            dust_mom3 += delta_mom3(idx, i);

            dust_mom1 += delta_dust_mom1_drag;
            dust_mom2 += delta_dust_mom2_drag;
            dust_mom3 += delta_dust_mom3_drag;

            // Add the energy dissipation of drags if gas is non barotropic.
            // dE_gas = dM_dust*(v_dust_bf_drag + v_dust_af_drag)/2
            if (Drag_WorkDissipation) {
              // Calculate the dust velocity after drags
              Real inv_dust_dens     = 1.0/dust_dens;
              Real dust_vel1_af_drag = dust_mom1*inv_dust_dens;
              Real dust_vel2_af_drag = dust_mom2*inv_dust_dens;
              Real dust_vel3_af_drag = dust_mom3*inv_dust_dens;

              Real dissipation_I = 0.5*(delta_mom1(idx, i)*(Stage_I_vel1(idx, k, j, i) + dust_vel1) +
                                        delta_mom2(idx, i)*(Stage_I_vel2(idx, k, j, i) + dust_vel2) +
                                        delta_mom3(idx, i)*(Stage_I_vel3(idx, k, j, i) + dust_vel3));

              Real dissipation = 0.5*(delta_dust_mom1_drag*(Stage_I_vel1(idx, k, j, i) + dust_vel1_af_drag) +
                                      delta_dust_mom2_drag*(Stage_I_vel2(idx, k, j, i) + dust_vel2_af_drag) +
                                      delta_dust_mom3_drag*(Stage_I_vel3(idx, k, j, i) + dust_vel3_af_drag));

              Real &gas_erg  = u(IEN, k, j, i);
              gas_erg       -= (dissipation_I + dissipation);
            }
          }
        }
      }
    }
  }
  return;
}


void DustGasDrag::RK2ImplicitNoFeedback(const int stage,
      const Real dt, const AthenaArray<Real> &stopping_time,
      const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
      const AthenaArray<Real> &u, AthenaArray<Real> &cons_df) {

  int is = pmb_->is; int js = pmb_->js; int ks = pmb_->ks;
  int ie = pmb_->ie; int je = pmb_->je; int ke = pmb_->ke;

  bool Stage_I = (((orb_advection_  < 2) && (stage == 1)) ||
                  ((orb_advection_ == 2) && (stage == 2)));

  const AthenaArray<Real> &w_n             = pmy_hydro_->w_n;
  const AthenaArray<Real> &prim_df_n       = pmy_dustfluids_->df_prim_n;
  const AthenaArray<Real> &stopping_time_n = pmy_dustfluids_->stopping_time_array_n;
  const AthenaArray<Real> &cons_df_af_src  = pmy_dustfluids_->df_cons_af_src;

  AthenaArray<Real> &Stage_I_delta_mom1 = pmy_dustfluids_->Stage_I_delta_mom1;
  AthenaArray<Real> &Stage_I_delta_mom2 = pmy_dustfluids_->Stage_I_delta_mom2;
  AthenaArray<Real> &Stage_I_delta_mom3 = pmy_dustfluids_->Stage_I_delta_mom3;

  if (Stage_I) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        alpha.ZeroClear();
        qvalue.ZeroClear();
        weight_dust.ZeroClear();
        weight_gas.ZeroClear();
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          // Alias the primitives of gas at current stage
          const Real &gas_rho  = w(IDN, k, j, i);
          const Real &gas_vel1 = w(IM1, k, j, i);
          const Real &gas_vel2 = w(IM2, k, j, i);
          const Real &gas_vel3 = w(IM3, k, j, i);

          mom1_prim(0, i) = gas_rho*gas_vel1;
          mom2_prim(0, i) = gas_rho*gas_vel2;
          mom3_prim(0, i) = gas_rho*gas_vel3;
        }

        for (int n=1; n<=NDUSTFLUIDS; ++n) {
          int dust_id = n - 1;
          int rho_id  = 4*dust_id;
          int v1_id   = rho_id + 1;
          int v2_id   = rho_id + 2;
          int v3_id   = rho_id + 3;
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            // Alias the primitives of gas at current stage
            const Real &gas_vel1 = w(IM1, k, j, i);
            const Real &gas_vel2 = w(IM2, k, j, i);
            const Real &gas_vel3 = w(IM3, k, j, i);

            const Real &dust_rho  = prim_df(rho_id, k, j, i);
            const Real &dust_vel1 = prim_df(v1_id,  k, j, i);
            const Real &dust_vel2 = prim_df(v2_id,  k, j, i);
            const Real &dust_vel3 = prim_df(v3_id,  k, j, i);

            mom1_prim(n, i) = dust_rho*dust_vel1;
            mom2_prim(n, i) = dust_rho*dust_vel2;
            mom3_prim(n, i) = dust_rho*dust_vel3;

            Real &dust_mom1_bf_src = mom1_prim(n, i);
            Real &dust_mom2_bf_src = mom2_prim(n, i);
            Real &dust_mom3_bf_src = mom3_prim(n, i);

            delta_mom1_src(n, i) = (cons_df_af_src(v1_id, k, j, i) - dust_mom1_bf_src);
            delta_mom2_src(n, i) = (cons_df_af_src(v2_id, k, j, i) - dust_mom2_bf_src);
            delta_mom3_src(n, i) = (cons_df_af_src(v3_id, k, j, i) - dust_mom3_bf_src);

            alpha(n, i)       = 1.0/(stopping_time(dust_id, k, j, i) + TINY_NUMBER);
            qvalue(n, i)      = alpha(n, i)*dt;
            weight_dust(n, i) = 1.0/(1.0 + qvalue(n, i));

            // Alias the conserves of dust
            Real &dust_mom1 = cons_df(v1_id,  k, j, i);
            Real &dust_mom2 = cons_df(v2_id,  k, j, i);
            Real &dust_mom3 = cons_df(v3_id,  k, j, i);

            delta_mom1(n, i) = weight_dust(n, i)*(mom1_prim(n, i) + delta_mom1_src(n, i) + qvalue(n, i)*dust_rho*gas_vel1) - mom1_prim(n, i);
            delta_mom2(n, i) = weight_dust(n, i)*(mom2_prim(n, i) + delta_mom2_src(n, i) + qvalue(n, i)*dust_rho*gas_vel2) - mom2_prim(n, i);
            delta_mom3(n, i) = weight_dust(n, i)*(mom3_prim(n, i) + delta_mom3_src(n, i) + qvalue(n, i)*dust_rho*gas_vel3) - mom3_prim(n, i);

            // Add the delta momentum caused by drags on dust conserves
            Real dust_delta_mom1_drag = (delta_mom1(n, i) - delta_mom1_src(n, i));
            Real dust_delta_mom2_drag = (delta_mom2(n, i) - delta_mom2_src(n, i));
            Real dust_delta_mom3_drag = (delta_mom3(n, i) - delta_mom3_src(n, i));

            Stage_I_delta_mom1(n, k, j, i) = dust_delta_mom1_drag;
            Stage_I_delta_mom2(n, k, j, i) = dust_delta_mom2_drag;
            Stage_I_delta_mom3(n, k, j, i) = dust_delta_mom3_drag;

            dust_mom1 += dust_delta_mom1_drag;
            dust_mom2 += dust_delta_mom2_drag;
            dust_mom3 += dust_delta_mom3_drag;
          }
        }
      }
    }
  } else {
    Real inv_dt = 1.0/(2.0*dt);
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        force_x1.ZeroClear();
        force_x2.ZeroClear();
        force_x3.ZeroClear();

        force_x1_n.ZeroClear();
        force_x2_n.ZeroClear();
        force_x3_n.ZeroClear();

        jacobi.ZeroClear();
        jacobi_n.ZeroClear();

        temp_A.ZeroClear();
        temp_B.ZeroClear();
        temp_C.ZeroClear();
        temp_D.ZeroClear();

        product.ZeroClear();
        idx_vector.ZeroClear();
        lu_matrix.ZeroClear();
        lambda.ZeroClear();
        lambda_inv.ZeroClear();

        delta_mom1.ZeroClear();
        delta_mom2.ZeroClear();
        delta_mom3.ZeroClear();

        delta_mom1_im.ZeroClear();
        delta_mom2_im.ZeroClear();
        delta_mom3_im.ZeroClear();

        delta_mom1_im_II.ZeroClear();
        delta_mom2_im_II.ZeroClear();
        delta_mom3_im_II.ZeroClear();

        inv_gas_rho.ZeroClear();
        inv_gas_rho_n.ZeroClear();

        mom1_prim_n.ZeroClear();
        mom2_prim_n.ZeroClear();
        mom3_prim_n.ZeroClear();
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          const Real &gas_rho = w(IDN, k, j, i);
          inv_gas_rho(i)      = 1.0/gas_rho;

          const Real &gas_rho_n  = w_n(IDN, k, j, i);
          const Real &gas_vel1_n = w_n(IM1, k, j, i);
          const Real &gas_vel2_n = w_n(IM2, k, j, i);
          const Real &gas_vel3_n = w_n(IM3, k, j, i);
          inv_gas_rho_n(i)       = 1.0/gas_rho_n;

          mom1_prim_n(0, i) = gas_rho_n*gas_vel1_n;
          mom2_prim_n(0, i) = gas_rho_n*gas_vel2_n;
          mom3_prim_n(0, i) = gas_rho_n*gas_vel3_n;

          temp_A(0, 0, i) = 1.0;
          temp_C(0, 0, i) = 1.0;
        }

        Real double_dt = 2.0*dt;
        for (int idx=1; idx<=NDUSTFLUIDS; ++idx) {
          int dust_id = idx - 1;
          int rho_id  = 4*dust_id;
          int v1_id   = rho_id + 1;
          int v2_id   = rho_id + 2;
          int v3_id   = rho_id + 3;
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            // Alias the primitives of dust at stage n
            const Real &dust_rho_n  = prim_df_n(rho_id, k, j, i);
            const Real &dust_vel1_n = prim_df_n(v1_id, k, j, i);
            const Real &dust_vel2_n = prim_df_n(v2_id, k, j, i);
            const Real &dust_vel3_n = prim_df_n(v3_id, k, j, i);
            alpha_n(idx, i)         = 1.0/(stopping_time_n(dust_id, k, j, i) + TINY_NUMBER);
            epsilon_n(idx, i)       = dust_rho_n*inv_gas_rho_n(i);

            // Alias the primitives of dust at current stage
            const Real &dust_rho = prim_df(rho_id, k, j, i);
            alpha(idx, i)        = 1.0/(stopping_time(dust_id, k, j, i) + TINY_NUMBER);
            epsilon(idx, i)      = dust_rho*inv_gas_rho(i);

            mom1_prim_n(idx, i) = dust_rho_n*dust_vel1_n;
            mom2_prim_n(idx, i) = dust_rho_n*dust_vel2_n;
            mom3_prim_n(idx, i) = dust_rho_n*dust_vel3_n;

            Real &dust_mom1_bf_src = mom1_prim_n(idx, i);
            Real &dust_mom2_bf_src = mom2_prim_n(idx, i);
            Real &dust_mom3_bf_src = mom3_prim_n(idx, i);

            force_x1_n(idx, i) = alpha_n(idx, i)*(epsilon_n(idx, i)*mom1_prim_n(0, i) - dust_mom1_bf_src);
            force_x2_n(idx, i) = alpha_n(idx, i)*(epsilon_n(idx, i)*mom2_prim_n(0, i) - dust_mom2_bf_src);
            force_x3_n(idx, i) = alpha_n(idx, i)*(epsilon_n(idx, i)*mom3_prim_n(0, i) - dust_mom3_bf_src);

            force_x1(idx, i) = alpha(idx, i)*(epsilon(idx, i)*mom1_prim_n(0, i) - dust_mom1_bf_src);
            force_x2(idx, i) = alpha(idx, i)*(epsilon(idx, i)*mom2_prim_n(0, i) - dust_mom2_bf_src);
            force_x3(idx, i) = alpha(idx, i)*(epsilon(idx, i)*mom3_prim_n(0, i) - dust_mom3_bf_src);

            delta_mom1(idx, i) = -0.5*Stage_I_delta_mom1(idx, k, j, i);
            delta_mom2(idx, i) = -0.5*Stage_I_delta_mom2(idx, k, j, i);
            delta_mom3(idx, i) = -0.5*Stage_I_delta_mom3(idx, k, j, i);

            delta_mom1_src(idx, i) = (cons_df_af_src(v1_id, k, j, i) + delta_mom1(idx, i) - dust_mom1_bf_src);
            delta_mom2_src(idx, i) = (cons_df_af_src(v2_id, k, j, i) + delta_mom2(idx, i) - dust_mom2_bf_src);
            delta_mom3_src(idx, i) = (cons_df_af_src(v3_id, k, j, i) + delta_mom3(idx, i) - dust_mom3_bf_src);

            Real dust_delta_force_x1_src = delta_mom1_src(idx, i)*inv_dt;
            Real dust_delta_force_x2_src = delta_mom2_src(idx, i)*inv_dt;
            Real dust_delta_force_x3_src = delta_mom3_src(idx, i)*inv_dt;

            force_x1_n(idx, i) += dust_delta_force_x1_src;
            force_x2_n(idx, i) += dust_delta_force_x2_src;
            force_x3_n(idx, i) += dust_delta_force_x3_src;

            force_x1(idx, i) += dust_delta_force_x1_src;
            force_x2(idx, i) += dust_delta_force_x2_src;
            force_x3(idx, i) += dust_delta_force_x3_src;

            jacobi(idx, 0, i)   = alpha(idx, i);
            jacobi_n(idx, 0, i) = alpha_n(idx, i);

            jacobi(idx, idx, i)   = -jacobi(idx, 0, i);
            jacobi_n(idx, idx, i) = -jacobi_n(idx, 0, i);

            product(idx, idx, i) = jacobi(idx, idx, i)*jacobi_n(idx, idx, i);
            product(idx, 0, i)   = -product(idx, idx, i);

            temp_A(idx, 0,   i) =     -double_dt*jacobi_n(idx, 0,   i);
            temp_A(idx, idx, i) = 1.0 -double_dt*jacobi_n(idx, idx, i);

            temp_C(idx, 0,   i) =     -double_dt*jacobi(idx, 0,   i);
            temp_C(idx, idx, i) = 1.0 -double_dt*jacobi(idx, idx, i);
          }
        }

        //// calculate temp_A = I - h*jacobi_n
        //Add(1.0, -2.0*dt, jacobi_n, temp_A);

        //// calculate temp_C = I - h*jacobi
        //Add(1.0, -2.0*dt, jacobi, temp_C);

        // calculate lambda = temp_A + 0.5*h^2*product
        Add(temp_A, 2.0*SQR(dt), product, lambda);

        // cauculate the inverse matrix of lambda
        LUdecompose(lambda, idx_vector, lu_matrix);
        Inverse(idx_vector, lu_matrix, lambda, lambda_inv);

        // calculate temp_B = 0.5*h*lambda_inv
        Multiply(dt, lambda_inv, temp_B);

        // calculate temp_D = 0.5*h*lambda_inv*temp_C
        for(int jdx=0; jdx<=NDUSTFLUIDS; ++jdx) {
          for(int idx=0; idx<=NDUSTFLUIDS; ++idx) {
#pragma omp simd
            for (int i=is; i<=ie; ++i) {
              temp_D(jdx, idx, i) += dt*lambda_inv(jdx, 0, i)*temp_C(0, idx, i);
            }
          }
        }

        for(int jdx=0; jdx<=NDUSTFLUIDS; ++jdx) {
          for(int idx=1; idx<=NDUSTFLUIDS; ++idx) {
#pragma omp simd
            for (int i=is; i<=ie; ++i) {
              temp_D(jdx, 0, i) += dt*lambda_inv(jdx, idx, i)*temp_C(idx, 0, i);
            }
          }
        }

        for(int jdx=0; jdx<=NDUSTFLUIDS; ++jdx) {
          for(int idx=1; idx<=NDUSTFLUIDS; ++idx) {
#pragma omp simd
            for (int i=is; i<=ie; ++i) {
              temp_D(jdx, idx, i) += dt*lambda_inv(jdx, idx, i)*temp_C(idx, idx, i);
            }
          }
        }

        //Multiply(lambda_inv, temp_C, temp_D);
        //Multiply(dt, temp_D);

        // Delta_M_im = temp_B*force
        MultiplyVector(temp_B, force_x1, delta_mom1_im);
        MultiplyVector(temp_B, force_x2, delta_mom2_im);
        MultiplyVector(temp_B, force_x3, delta_mom3_im);

        // Delta_M_im_II = temp_D*force_n
        MultiplyVector(temp_D, force_x1_n, delta_mom1_im_II);
        MultiplyVector(temp_D, force_x2_n, delta_mom2_im_II);
        MultiplyVector(temp_D, force_x3_n, delta_mom3_im_II);

        for (int idx=1; idx<=NDUSTFLUIDS; ++idx) {
          int dust_id = idx - 1;
          int rho_id  = 4*dust_id;
          int v1_id   = rho_id + 1;
          int v2_id   = rho_id + 2;
          int v3_id   = rho_id + 3;
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            // Alias the conserves of dust
            Real &dust_mom1 = cons_df(v1_id,  k, j, i);
            Real &dust_mom2 = cons_df(v2_id,  k, j, i);
            Real &dust_mom3 = cons_df(v3_id,  k, j, i);

            Real delta_dust_mom1_drag = delta_mom1_im(idx, i) + delta_mom1_im_II(idx, i) - delta_mom1_src(idx, i);
            Real delta_dust_mom2_drag = delta_mom2_im(idx, i) + delta_mom2_im_II(idx, i) - delta_mom2_src(idx, i);
            Real delta_dust_mom3_drag = delta_mom3_im(idx, i) + delta_mom3_im_II(idx, i) - delta_mom3_src(idx, i);

            // Add the delta momentum caused by drags on dust conserves
            dust_mom1 += delta_mom1(idx, i);
            dust_mom2 += delta_mom2(idx, i);
            dust_mom3 += delta_mom3(idx, i);

            dust_mom1 += delta_dust_mom1_drag;
            dust_mom2 += delta_dust_mom2_drag;
            dust_mom3 += delta_dust_mom3_drag;
          }
        }
      }
    }
  }
  return;
}
