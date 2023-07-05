//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file disk.cpp
//! \brief Initializes stratified Keplerian accretion disk in both cylindrical
//! coordinates.  Initial conditions are in vertical hydrostatic eqm.

// C headers

// C++ headers
#include <algorithm>  // min
#include <cmath>      // sqrt
#include <cstdlib>    // srand
#include <cstring>    // strcmp()
#include <fstream>
#include <iostream>   // endl
#include <limits>
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../bvals/bvals.hpp"
#include "../coordinates/coordinates.hpp"
#include "../dustfluids/dustfluids.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../hydro/hydro_diffusion/hydro_diffusion.hpp"
#include "../mesh/mesh.hpp"
#include "../orbital_advection/orbital_advection.hpp"
#include "../parameter_input.hpp"
#include "../scalars/scalars.hpp"

namespace {
Real PoverRho(const Real rad, const Real phi, const Real z);
Real DenProfileCyl_gas(const Real rad, const Real phi, const Real z);
Real VelProfileCyl_gas(const Real rad, const Real phi, const Real z);
Real DenProfileCyl_dust(const Real rad, const Real phi, const Real z,
                        const Real den_ratio, const Real H_ratio);
Real VelProfileCyl_dust(const Real rad, const Real phi, const Real z);

void GetCylCoord(Coordinates *pco, Real &rad, Real &phi, Real &z, int i, int j, int k);
void GetPlanetAcc(const int order, Real &rad, Real &phi, Real &z, int i, int j, int k);
void Vr_interpolate_outer_nomatter(const Real r_active, const Real r_ghost,
    const Real sigma_active, const Real sigma_ghost,
    const Real vr_active, Real &vr_ghost);

// problem parameters which are useful to make global to this file
Real rad_soft, gmp, inv_sqrt2gmp, rad_planet, phi_planet_0, z_planet,
t0_planet, time_drag, vk_planet, omega_planet, inv_omega_planet, cs_planet,
gm0, r0, rho0, dslope, p0_over_r0, pslope, beta, gMth, nu_alpha,
t_planet_growth, dfloor, dffloor, Omega0, user_dt, A_gap,
x1min, x1max, damping_rate, Hill_radius, accretion_radius, accretion_rate,
radius_inner_damping, radius_outer_damping, inner_ratio_region,
outer_ratio_region, inner_width_damping, outer_width_damping,
refine_factor, derefine_factor, refine_rad_min, refine_rad_max;

Real initial_D2G[NDUSTFLUIDS], Stokes_number[NDUSTFLUIDS], Hratio[NDUSTFLUIDS],
weight_dust[NDUSTFLUIDS];

bool Damping_Flag, Isothermal_Flag, MassTransfer_Flag,
     RadiativeConduction_Flag, TransferFeedback_Flag;
int PlanetaryGravityOrder;

// User-defined Sources
void MySource(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s);
void ThermalRelaxation(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s);
void MassTransferWithinHill(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s);
void PlanetaryGravity(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s);

// User-defined Stopping time
void MyStoppingTime(MeshBlock *pmb, const Real time, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, AthenaArray<Real> &stopping_time_array);

// User-defined dust diffusivity
void MyDustDiffusivity(DustFluids *pdf, MeshBlock *pmb,
      const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
      const AthenaArray<Real> &stopping_time,
      AthenaArray<Real> &nu_dust, AthenaArray<Real> &cs_dust,
      int is, int ie, int js, int je, int ks, int ke);

// User-defined condutivity
void RadiativeCondution(HydroDiffusion *phdif, MeshBlock *pmb,
    const AthenaArray<Real> &w, const AthenaArray<Real> &bc,
    int is, int ie, int js, int je, int ks, int ke);

// Adapative Refinement condition
int RefinementCondition(MeshBlock *pmb);
} // namespace

// User-defined boundary conditions for disk simulations
void DiskInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                AthenaArray<Real> &prim_df, FaceField &b, Real time, Real dt,
                int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                AthenaArray<Real> &prim_df, FaceField &b, Real time, Real dt,
                int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                AthenaArray<Real> &prim_df, FaceField &b, Real time, Real dt,
                int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                AthenaArray<Real> &prim_df, FaceField &b, Real time, Real dt,
                int il, int iu, int jl, int ju, int kl, int ku, int ngh);

void LocalIsothermalEOS(MeshBlock *pmb, int il, int iu, int jl,
    int ju, int kl, int ku, AthenaArray<Real> &prim, AthenaArray<Real> &cons);
void InnerWaveDampingGas(MeshBlock *pmb, const Real time, const Real dt, int il, int iu,
    int jl, int ju, int kl, int ku, AthenaArray<Real> &prim, AthenaArray<Real> &cons);
void OuterWaveDampingGas(MeshBlock *pmb, const Real time, const Real dt, int il, int iu,
    int jl, int ju, int kl, int ku, AthenaArray<Real> &prim, AthenaArray<Real> &cons);
void FixedDust(MeshBlock *pmb, int il, int iu, int jl, int ju, int kl, int ku,
    AthenaArray<Real> &prim_df, AthenaArray<Real> &cons_df);

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//! \brief Function to initialize problem-specific data in mesh class.  Can also be used
//! to initialize variables which are global to (and therefore can be passed to) other
//! functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {

  // Get parameters for gravitatonal potential of central point mass
  gm0                      = pin->GetOrAddReal("problem", "GM", 0.0);
  r0                       = pin->GetOrAddReal("problem", "r0", 1.0);
  Damping_Flag             = pin->GetBoolean("problem", "Damping_Flag");
  Isothermal_Flag          = pin->GetBoolean("problem", "Isothermal_Flag");
  MassTransfer_Flag        = pin->GetOrAddBoolean("problem", "MassTransfer_Flag", false);
  RadiativeConduction_Flag = pin->GetOrAddBoolean("problem", "RadiativeConduction_Flag", false);
  TransferFeedback_Flag    = pin->GetOrAddBoolean("problem", "TransferFeedback_Flag", true);

  // Get parameters for initial density and velocity
  rho0   = pin->GetReal("problem", "rho0");
  dslope = pin->GetOrAddReal("problem", "dslope", -1.0);
  A_gap  = pin->GetOrAddReal("problem", "A_gap", 0.0);

  // Get parameters of initial pressure and cooling parameters
  if (NON_BAROTROPIC_EOS) {
    p0_over_r0 = pin->GetOrAddReal("problem", "p0_over_r0", 0.0025);
    pslope     = pin->GetOrAddReal("problem", "pslope",     -0.5);
    beta       = pin->GetOrAddReal("problem", "beta", 0.0);
    if (beta < 0.0) beta = 0.0;
  } else {
    p0_over_r0 = SQR(pin->GetReal("hydro", "iso_sound_speed"));
  }

  Real float_min = std::numeric_limits<float>::min();
  dfloor    = pin->GetOrAddReal("hydro", "dfloor", (1024*(float_min)));
  nu_alpha  = pin->GetOrAddReal("problem", "nu_alpha", 0.0);
  dffloor   = pin->GetOrAddReal("dust", "dffloor", (1024*(float_min)));
  time_drag = pin->GetOrAddReal("dust", "time_drag", 0.0);
  Omega0    = pin->GetOrAddReal("orbital_advection", "Omega0", 0.0);

  // Dust to gas ratio && dust stopping time
  if (NDUSTFLUIDS > 0) {
    for (int n=0; n<NDUSTFLUIDS; n++) {
      initial_D2G[n]   = pin->GetReal("dust", "initial_D2G_" + std::to_string(n+1));
      Stokes_number[n] = pin->GetReal("dust", "Stokes_number_" + std::to_string(n+1));
      Hratio[n]        = pin->GetReal("dust", "Hratio_" + std::to_string(n+1));
      weight_dust[n]   = 2.0/(Stokes_number[n] + SQR(1.0+initial_D2G[n])/Stokes_number[n]);
    }
  }


  // The parameters of one planet
  rad_planet   = pin->GetOrAddReal("problem",  "rad_planet", 1.0);            // radial position of the planet
  phi_planet_0 = pin->GetOrAddReal("problem",  "phi_planet", 0.0);            // azimuthal position of the planet
  z_planet     = pin->GetOrAddReal("problem",  "z_planet",   0.0);            // vertical position of the planet
  t0_planet    = (pin->GetOrAddReal("problem", "t0_planet",  0.0))*TWO_PI*r0; // time to put in the planet
  gmp          = pin->GetOrAddReal("problem",  "GMp",        0.0);            // GM of the planet
  user_dt      = pin->GetOrAddReal("problem",  "user_dt",    0.0);

  vk_planet        = std::sqrt(gm0/rad_planet);
  omega_planet     = vk_planet/rad_planet;
  inv_omega_planet = 1.0/omega_planet;

  PlanetaryGravityOrder = pin->GetOrAddInteger("problem", "PlanetaryGravityOrder", 2);
  if ((PlanetaryGravityOrder != 2) || (PlanetaryGravityOrder != 4))
    PlanetaryGravityOrder = 2;

  if (NON_BAROTROPIC_EOS)
    cs_planet = std::sqrt(p0_over_r0*std::pow(rad_planet/r0, pslope));
  else
    cs_planet = std::sqrt(p0_over_r0);

  if (gmp != 0.0) inv_sqrt2gmp = 1.0/std::sqrt(2.0*gmp);
  gMth = gm0*cs_planet*cs_planet*cs_planet/(vk_planet*vk_planet*vk_planet);

  t_planet_growth  = pin->GetOrAddReal("problem", "t_planet_growth", 0.0)*TWO_PI*r0; // orbital number to grow the planet
  t_planet_growth *= inv_omega_planet*(gmp/gMth);
  t_planet_growth += t0_planet;

  Hill_radius = (std::pow(gmp/gm0*ONE_3RD, ONE_3RD)*rad_planet);

  rad_soft  = pin->GetOrAddReal("problem", "rs", 0.6); // softening length of the gravitational potential of planets
  rad_soft *= Hill_radius;

  accretion_radius  = pin->GetOrAddReal("problem", "accretion_radius", 0.3); // Accretion radius of planets
  accretion_radius *= Hill_radius;

  if (accretion_radius < rad_soft) accretion_radius = 1.1*rad_soft;

  accretion_rate = pin->GetOrAddReal("problem", "accretion_rate", 0.1); // Accretion radius of planets

  // parameters of refinement
  refine_factor   = pin->GetOrAddReal("problem", "refine_factor",   0.01);
  derefine_factor = pin->GetOrAddReal("problem", "derefine_factor", 0.005);
  refine_rad_min  = pin->GetOrAddReal("problem", "refine_rad_min",  0.2);
  refine_rad_max  = pin->GetOrAddReal("problem", "refine_rad_max",  2.0);

  // The parameters of damping zones
  x1min = pin->GetReal("mesh", "x1min");
  x1max = pin->GetReal("mesh", "x1max");

  //ratio of the orbital periods between the edge of the wave-killing zone and the corresponding edge of the mesh
  inner_ratio_region = pin->GetOrAddReal("problem", "inner_dampingregion_ratio", 1.5);
  outer_ratio_region = pin->GetOrAddReal("problem", "outer_dampingregion_ratio", 1.2);

  radius_inner_damping = x1min*std::pow(inner_ratio_region, TWO_3RD);
  radius_outer_damping = x1max*std::pow(outer_ratio_region, -TWO_3RD);

  inner_width_damping = radius_inner_damping - x1min;
  outer_width_damping = x1max - radius_outer_damping;

  // The normalized wave damping timescale, in unit of dynamical timescale.
  damping_rate = pin->GetOrAddReal("problem", "damping_rate", 1.0);

  // enroll user-defined boundary condition
  if (mesh_bcs[BoundaryFace::inner_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x1, DiskInnerX1);
  }
  if (mesh_bcs[BoundaryFace::outer_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x1, DiskOuterX1);
  }
  if (mesh_bcs[BoundaryFace::inner_x2] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x2, DiskInnerX2);
  }
  if (mesh_bcs[BoundaryFace::outer_x2] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x2, DiskOuterX2);
  }

  if (NDUSTFLUIDS > 0) {
    // Enroll user-defined dust stopping time
    EnrollUserDustStoppingTime(MyStoppingTime);
    // Enroll user-defined dust diffusivity
    EnrollDustDiffusivity(MyDustDiffusivity);
  }

  // Enroll planetary gravity, damping zone, local isothermal and beta cooling
  EnrollUserExplicitSourceFunction(MySource);

  // Enroll user-defined thermal conduction
  if (!(Isothermal_Flag && NON_BAROTROPIC_EOS) && (beta > 0.0) && (RadiativeConduction_Flag))
    EnrollConductionCoefficient(RadiativeCondution);

  // Enroll user-defined AMR criterion
  if (adaptive)
    EnrollUserRefinementCondition(RefinementCondition);

  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Initializes Keplerian accretion disk.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {

  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") != 0) {
    std::stringstream msg;
    msg << "This problem file must be setup in the cylindrical coordinate!" << std::endl;
    ATHENA_ERROR(msg);
  }

  if ((block_size.nx2 == 1) || (block_size.nx3 > 1)) {
    std::stringstream msg;
    msg << "This problem file must be setup in 2D!" << std::endl;
    ATHENA_ERROR(msg);
  }

  Real igm1 = 1.0/(peos->GetGamma() - 1.0);
  OrbitalVelocityFunc &vK = porb->OrbitalVelocity;

  Real orb_defined;
  (porb->orbital_advection_defined) ? orb_defined = 1.0 : orb_defined = 0.0;

    // Initialize density and momenta
  for (int k=ks; k<=ke; ++k) {
    Real x3 = pcoord->x3v(k);
    for (int j=js; j<=je; ++j) {
      Real x2 = pcoord->x2v(j);
      for (int i=is; i<=ie; ++i) {
        Real x1 = pcoord->x1v(i);
        Real rad, phi, z;

        GetCylCoord(pcoord, rad, phi, z, i, j, k); // convert to cylindrical coordinates
        Real cs_square = PoverRho(rad, phi, z);
        Real omega_dyn = std::sqrt(gm0/(rad*rad*rad));
        Real vel_K     = vK(porb, x1, x2, x3);

        // compute initial conditions in cylindrical coordinates
        Real den_gas      = DenProfileCyl_gas(rad, phi, z);
        Real vis_vel_r    = -1.5*(nu_alpha*cs_square/rad/omega_dyn);
        Real vel_gas_phi  = VelProfileCyl_gas(rad, phi, z);
        vel_gas_phi      -= orb_defined*vK(porb, x1, x2, x3);
        Real vel_gas_z    = 0.0;

        Real pre_diff = (pslope + dslope)*cs_square;
        
        Real &gas_dens = phydro->u(IDN, k, j, i);
        Real &gas_mom1 = phydro->u(IM1, k, j, i);
        Real &gas_mom2 = phydro->u(IM2, k, j, i);
        Real &gas_mom3 = phydro->u(IM3, k, j, i);

        gas_dens = den_gas;
        gas_mom1 = den_gas*vis_vel_r;
        gas_mom2 = den_gas*vel_gas_phi;
        gas_mom3 = den_gas*vel_gas_z;

        if (NON_BAROTROPIC_EOS) {
          phydro->u(IEN, k, j, i)  = cs_square*phydro->u(IDN, k, j, i)*igm1;
          phydro->u(IEN, k, j, i) += 0.5*(SQR(phydro->u(IM1, k, j, i))+SQR(phydro->u(IM2, k, j, i))
                                        + SQR(phydro->u(IM3, k, j, i)))/phydro->u(IDN, k, j, i);
        }

        if (NDUSTFLUIDS > 0) {
          for (int n=0; n<NDUSTFLUIDS; ++n) {
            int dust_id = n;
            int rho_id  = 4*dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + 2;
            int v3_id   = rho_id + 3;

            // compute initial conditions in cylindrical coordinates
            Real den_dust      = DenProfileCyl_dust(rad, phi, z, initial_D2G[dust_id], Hratio[dust_id]);
            Real vel_dust_r    = weight_dust[dust_id]*pre_diff/(2.0*rad*omega_dyn);
            Real vel_dust_phi  = VelProfileCyl_dust(rad, phi, z);
            vel_dust_phi      -= orb_defined*vK(porb, x1, x2, x3);
            Real vel_dust_z    = 0.0;

            pdustfluids->df_cons(rho_id, k, j, i) = den_dust;
            pdustfluids->df_cons(v1_id,  k, j, i) = den_dust*vel_dust_r;
            pdustfluids->df_cons(v2_id,  k, j, i) = den_dust*vel_dust_phi;
            pdustfluids->df_cons(v3_id,  k, j, i) = den_dust*vel_dust_z;
          }
        }

        if (NSCALARS > 0) {
          for (int n=0; n<NSCALARS; ++n) {
            pscalars->s(n, k, j, i) = den_gas;
          }
        }
      }
    }
  }
  return;
}


namespace {
//----------------------------------------------------------------------------------------
void MySource(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s) {

  if ((gmp > 0.0) && (time >= t0_planet))
    PlanetaryGravity(pmb, time, dt, prim, prim_df, prim_s, bcc, cons, cons_df, cons_s);

  if ((gmp > 0.0) && (time >= t0_planet) && MassTransfer_Flag)
    MassTransferWithinHill(pmb, time, dt, prim, prim_df, prim_s, bcc, cons, cons_df, cons_s);

  if ((!Isothermal_Flag) && (beta > 0.0) && NON_BAROTROPIC_EOS)
    ThermalRelaxation(pmb, time, dt, prim, prim_df, prim_s, bcc, cons, cons_df, cons_s);

  return;
}


void MyStoppingTime(MeshBlock *pmb, const Real time, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, AthenaArray<Real> &stopping_time) {

  int nc1 = pmb->ncells1;
  AthenaArray<Real> rad_arr, phi_arr, z_arr;
  rad_arr.NewAthenaArray(nc1);
  phi_arr.NewAthenaArray(nc1);
  z_arr.NewAthenaArray(nc1);

  Real inv_sqrt_gm0 = 1.0/std::sqrt(gm0);

  for (int n=0; n<NDUSTFLUIDS; ++n) {
    int dust_id = n;
    for (int k=pmb->ks; k<=pmb->ke; ++k) {
      for (int j=pmb->js; j<=pmb->je; ++j) {
#pragma omp simd
        for (int i=pmb->is; i<=pmb->ie; ++i) {
          GetCylCoord(pmb->pcoord, rad_arr(i), phi_arr(i), z_arr(i), i, j, k);
          //rad_arr(i) = pmb->pcoord->x1v(i);

          Real &st_time = stopping_time(dust_id, k, j, i);
          //Constant Stokes number in disk problems
          st_time = Stokes_number[dust_id]*std::sqrt(rad_arr(i)*rad_arr(i)*rad_arr(i))*inv_sqrt_gm0;
        }
      }
    }
  }
  return;
}


void MyDustDiffusivity(DustFluids *pdf, MeshBlock *pmb,
      const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
      const AthenaArray<Real> &stopping_time, AthenaArray<Real> &nu_dust,
      AthenaArray<Real> &cs_dust, int is, int ie, int js, int je, int ks, int ke) {

    int nc1 = pmb->ncells1;
    AthenaArray<Real> rad_arr, phi_arr, z_arr;
    rad_arr.NewAthenaArray(nc1);
    phi_arr.NewAthenaArray(nc1);
    z_arr.NewAthenaArray(nc1);

    Real inv_sqrt_gm0 = 1.0/std::sqrt(gm0);
    Real gamma = pmb->peos->GetGamma();

    for (int n=0; n<NDUSTFLUIDS; n++) {
      int dust_id = n;
      for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            GetCylCoord(pmb->pcoord, rad_arr(i), phi_arr(i), z_arr(i), i, j, k);
            //rad_arr(i) = pmb->pcoord->x1v(i);

            const Real &gas_pre = w(IPR, k, j, i);
            const Real &gas_den = w(IDN, k, j, i);

            Real inv_Omega_K = std::pow(rad_arr(i), 1.5)*inv_sqrt_gm0;
            Real nu_gas      = nu_alpha*inv_Omega_K*gamma*gas_pre/gas_den;

            Real &diffusivity = nu_dust(dust_id, k, j, i);
            diffusivity       = nu_gas/(1.0 + SQR(Stokes_number[dust_id]));

            Real &soundspeed  = cs_dust(dust_id, k, j, i);
            soundspeed        = std::sqrt(diffusivity*inv_Omega_K);
          }
        }
      }
    }
  return;
}


void PlanetaryGravity(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s) {

  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  Real phi_planet_move    = omega_planet*time + phi_planet_0;
  Real planet_gm;
  int nc1 = pmb->ncells1;

  phi_planet_move -= Omega0*time;

  if (time >= t0_planet) {
    AthenaArray<Real> distance_square;
    AthenaArray<Real> x_dis, y_dis, z_dis, rad_dis, phi_dis;
    AthenaArray<Real> rad_arr, phi_arr, z_arr;
    AthenaArray<Real> acc_R, acc_phi, acc_z, indirect_acc_r, indirect_acc_phi, gravity;

    distance_square.NewAthenaArray(nc1);

    x_dis.NewAthenaArray(nc1);
    y_dis.NewAthenaArray(nc1);
    z_dis.NewAthenaArray(nc1);

    rad_dis.NewAthenaArray(nc1);
    phi_dis.NewAthenaArray(nc1);

    rad_arr.NewAthenaArray(nc1);
    phi_arr.NewAthenaArray(nc1);
    z_arr.NewAthenaArray(nc1);

    acc_R.NewAthenaArray(nc1);
    acc_phi.NewAthenaArray(nc1);
    acc_z.NewAthenaArray(nc1);

    indirect_acc_r.NewAthenaArray(nc1);
    indirect_acc_phi.NewAthenaArray(nc1);
    gravity.NewAthenaArray(nc1);

    (t_planet_growth > t0_planet) ? (planet_gm = gmp*std::sin(0.5*PI*std::min(time/t_planet_growth, 1.0))) : (planet_gm = gmp);
    //planet_gm = gmp;

    for (int k=pmb->ks; k<=pmb->ke; ++k) {
      for (int j=pmb->js; j<=pmb->je; ++j) {
#pragma omp simd
        for (int i=pmb->is; i<=pmb->ie; ++i) {
          GetCylCoord(pmb->pcoord, rad_arr(i), phi_arr(i), z_arr(i), i, j, k);

          x_dis(i) = rad_arr(i)*std::cos(phi_arr(i)) - rad_planet*std::cos(phi_planet_move);
          y_dis(i) = rad_arr(i)*std::sin(phi_arr(i)) - rad_planet*std::sin(phi_planet_move);
          z_dis(i) = z_arr(i) - z_planet;

          rad_dis(i) = rad_arr(i) - rad_planet*std::cos(phi_arr(i) - phi_planet_move);
          phi_dis(i) = rad_planet*std::sin(phi_arr(i) - phi_planet_move);

          distance_square(i) = SQR(x_dis(i)) + SQR(y_dis(i)) + SQR(z_dis(i));

          indirect_acc_r(i)   =  planet_gm*std::cos(phi_arr(i) - phi_planet_move)/SQR(rad_planet);
          indirect_acc_phi(i) = -planet_gm*std::sin(phi_arr(i) - phi_planet_move)/SQR(rad_planet);

          gravity(i) = ((PlanetaryGravityOrder == 2)) ?
          (planet_gm/std::pow(distance_square(i)+SQR(rad_soft), 1.5)) :
          (planet_gm*(5.0*SQR(rad_soft)+2.0*distance_square(i))/(2.0*std::pow(SQR(rad_soft)+distance_square(i), 2.5)));

          acc_R(i)   = gravity(i)*rad_dis(i) + indirect_acc_r(i);
          acc_phi(i) = gravity(i)*phi_dis(i) + indirect_acc_phi(i);
          acc_z(i)   = gravity(i)*z_dis(i);

          const Real &gas_rho  = prim(IDN, k, j, i);
          const Real &gas_vel1 = prim(IM1, k, j, i);
          const Real &gas_vel2 = prim(IM2, k, j, i);
          const Real &gas_vel3 = prim(IM3, k, j, i);

          Real &gas_mom1 = cons(IM1, k, j, i);
          Real &gas_mom2 = cons(IM2, k, j, i);
          Real &gas_mom3 = cons(IM3, k, j, i);

          Real delta_mom1 = -dt*gas_rho*acc_R(i);
          Real delta_mom2 = -dt*gas_rho*acc_phi(i);
          Real delta_mom3 = -dt*gas_rho*acc_z(i);

          gas_mom1 += delta_mom1;
          gas_mom2 += delta_mom2;
          gas_mom3 += delta_mom3;

          if (NON_BAROTROPIC_EOS && (!Isothermal_Flag)) {
            Real &gas_erg  = cons(IEN, k, j, i);
            gas_erg       += (delta_mom1*gas_vel1 + delta_mom2*gas_vel2 + delta_mom3*gas_vel3);
          }
        }

        if (NDUSTFLUIDS > 0) {
          for (int n=0; n<NDUSTFLUIDS; ++n) {
            int dust_id = n;
            int rho_id  = 4*dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + 2;
            int v3_id   = rho_id + 3;
#pragma omp simd
            for (int i=pmb->is; i<=pmb->ie; ++i) {
              const Real &dust_rho = prim_df(rho_id, k, j, i);

              Real &dust_mom1 = cons_df(v1_id, k, j, i);
              Real &dust_mom2 = cons_df(v2_id, k, j, i);
              Real &dust_mom3 = cons_df(v3_id, k, j, i);

              Real delta_dust_mom1 = -dt*dust_rho*acc_R(i);
              Real delta_dust_mom2 = -dt*dust_rho*acc_phi(i);
              Real delta_dust_mom3 = -dt*dust_rho*acc_z(i);

              dust_mom1 += delta_dust_mom1;
              dust_mom2 += delta_dust_mom2;
              dust_mom3 += delta_dust_mom3;
            }
          }
        }
      }
    }
  }
  return;
}



// Mass Remove within Hill
void MassTransferWithinHill(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s) {

  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  Real phi_planet_move = omega_planet*time + phi_planet_0;
  phi_planet_move -= Omega0*time;

  int nc1 = pmb->ncells1;

  Real igm1 = 1.0/(pmb->peos->GetGamma() - 1.0);
  Real inv_rad_soft   = 1.0/rad_soft;
  Real inv_rad_soft_3 = 1.0/(rad_soft*rad_soft*rad_soft);

  AthenaArray<Real> distance_square, distance, time_freefall, remove_percent;
  AthenaArray<Real> x_dis, y_dis, z_dis;
  AthenaArray<Real> rad_arr, phi_arr, z_arr;

  distance_square.NewAthenaArray(nc1);
  distance.NewAthenaArray(nc1);
  time_freefall.NewAthenaArray(nc1);
  remove_percent.NewAthenaArray(nc1);

  x_dis.NewAthenaArray(nc1);
  y_dis.NewAthenaArray(nc1);
  z_dis.NewAthenaArray(nc1);

  rad_arr.NewAthenaArray(nc1);
  phi_arr.NewAthenaArray(nc1);
  z_arr.NewAthenaArray(nc1);

  if (time >= t0_planet) {
    for (int k=pmb->ks; k<=pmb->ke; ++k) {
      for (int j=pmb->js; j<=pmb->je; ++j) {
#pragma omp simd
        for (int i=pmb->is; i<=pmb->ie; ++i) {
          GetCylCoord(pmb->pcoord, rad_arr(i), phi_arr(i), z_arr(i), i, j, k);
          x_dis(i) = rad_arr(i)*std::cos(phi_arr(i)) - rad_planet*std::cos(phi_planet_move);
          y_dis(i) = rad_arr(i)*std::sin(phi_arr(i)) - rad_planet*std::sin(phi_planet_move);
          z_dis(i) = z_arr(i) - z_planet;

          distance_square(i) = SQR(x_dis(i)) + SQR(y_dis(i)) + SQR(z_dis(i));
          distance(i)        = std::sqrt(distance_square(i));

          if ((distance(i) > rad_soft) && (distance(i) <= accretion_radius)) {
            time_freefall(i)  = std::sqrt(distance_square(i)*distance(i))*inv_sqrt2gmp;
            remove_percent(i) = -accretion_rate*std::max(dt/time_freefall(i), 1.0);

            const Real &gas_rho  = prim(IDN, k, j, i);
            const Real &gas_vel1 = prim(IM1, k, j, i);
            const Real &gas_vel2 = prim(IM2, k, j, i);
            const Real &gas_vel3 = prim(IM3, k, j, i);

            Real &gas_dens = cons(IDN, k, j, i);
            Real &gas_mom1 = cons(IM1, k, j, i);
            Real &gas_mom2 = cons(IM2, k, j, i);
            Real &gas_mom3 = cons(IM3, k, j, i);

            Real delta_gas_dens = remove_percent(i)*gas_rho;
            Real delta_gas_mom1 = delta_gas_dens*gas_vel1;
            Real delta_gas_mom2 = delta_gas_dens*gas_vel2;
            Real delta_gas_mom3 = delta_gas_dens*gas_vel3;

            gas_dens += delta_gas_dens;
            gas_mom1 += delta_gas_mom1;
            gas_mom2 += delta_gas_mom2;
            gas_mom3 += delta_gas_mom3;

            if (NON_BAROTROPIC_EOS && (!Isothermal_Flag)) {
              Real &gas_erg  = cons(IEN, k, j, i);
              gas_erg       += (delta_gas_mom1*gas_vel1 + delta_gas_mom2*gas_vel2 + delta_gas_mom3*gas_vel3);
            }
          }
        }

        if (NDUSTFLUIDS > 0) {
          for (int n=0; n<NDUSTFLUIDS; ++n) {
            int dust_id = n;
            int rho_id  = 4*dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + 2;
            int v3_id   = rho_id + 3;
#pragma omp simd
            for (int i=pmb->is; i<=pmb->ie; ++i) {
              if ((distance(i) > rad_soft) && (distance(i) <= accretion_radius)) {
                const Real &dust_rho  = prim_df(rho_id, k, j, i);
                const Real &dust_vel1 = prim_df(v1_id,  k, j, i);
                const Real &dust_vel2 = prim_df(v2_id,  k, j, i);
                const Real &dust_vel3 = prim_df(v3_id,  k, j, i);

                Real &dust_dens = cons_df(rho_id, k, j, i);
                Real &dust_mom1 = cons_df(v1_id,  k, j, i);
                Real &dust_mom2 = cons_df(v2_id,  k, j, i);
                Real &dust_mom3 = cons_df(v3_id,  k, j, i);

                const Real &gas_vel1 = prim(IM1, k, j, i);
                const Real &gas_vel2 = prim(IM2, k, j, i);
                const Real &gas_vel3 = prim(IM3, k, j, i);

                Real &gas_mom1 = cons(IM1, k, j, i);
                Real &gas_mom2 = cons(IM2, k, j, i);
                Real &gas_mom3 = cons(IM3, k, j, i);

                Real delta_dust_dens = remove_percent(i)*dust_rho;
                Real delta_dust_mom1 = delta_dust_dens*dust_vel1;
                Real delta_dust_mom2 = delta_dust_dens*dust_vel2;
                Real delta_dust_mom3 = delta_dust_dens*dust_vel3;

                dust_dens += delta_dust_dens;
                dust_mom1 += delta_dust_mom1;
                dust_mom2 += delta_dust_mom2;
                dust_mom3 += delta_dust_mom3;

                if (TransferFeedback_Flag) {
                  gas_mom1 -= delta_dust_mom1;
                  gas_mom2 -= delta_dust_mom2;
                  gas_mom3 -= delta_dust_mom3;
                }

                if (TransferFeedback_Flag && NON_BAROTROPIC_EOS && (!Isothermal_Flag)) {
                  Real work_remove = -(delta_dust_mom1*gas_vel1 + delta_dust_mom2*gas_vel2
                                     + delta_dust_mom3*gas_vel3);

                  Real accretion_luminosity = gmp*inv_rad_soft*delta_dust_dens;

                  Real &gas_erg  = cons(IEN, k, j, i);
                  gas_erg       += (work_remove + accretion_luminosity*inv_rad_soft_3);
                }
              }
            }
          }
        }
      }
    }
  }
  return;
}


void ThermalRelaxation(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s) {

  int is = pmb->is; int ie = pmb->ie;
  int js = pmb->js; int je = pmb->je;
  int ks = pmb->ks; int ke = pmb->ke;

  int nc1 = pmb->ncells1;

  Real inv_beta = 1.0/beta;
  Real mygam    = pmb->peos->GetGamma();
  Real igm1     = 1.0/(mygam - 1.0);

  AthenaArray<Real> rad_arr, phi_arr, z_arr;
  rad_arr.NewAthenaArray(nc1);
  phi_arr.NewAthenaArray(nc1);
  z_arr.NewAthenaArray(nc1);

  for (int k=ks; k<=ke; ++k) { // include ghost zone
    for (int j=js; j<=je; ++j) { // prim, cons
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        GetCylCoord(pmb->pcoord, rad_arr(i), phi_arr(i), z_arr(i), i, j, k);

        const Real &gas_rho = prim(IDN, k, j, i);
        const Real &gas_pre = prim(IPR, k, j, i);

        Real omega_dyn      = std::sqrt(gm0/(rad_arr(i)*rad_arr(i)*rad_arr(i)));
        Real inv_t_cool     = omega_dyn*inv_beta;
        Real cs_square_init = PoverRho(rad_arr(i), phi_arr(i), z_arr(i));

        Real &gas_erg   = cons(IEN, k, j, i);
        Real delta_erg  = (gas_pre - gas_rho*cs_square_init)*igm1*inv_t_cool*dt;
        gas_erg        -= delta_erg;
      }
    }
  }
  return;
}


void RadiativeCondution(HydroDiffusion *phdif, MeshBlock *pmb,
    const AthenaArray<Real> &w, const AthenaArray<Real> &bc,
    int is, int ie, int js, int je, int ks, int ke) {

  int nc1 = pmb->ncells1;

  Real inv_beta = 1.0/beta;
  Real mygam    = pmb->peos->GetGamma();
  Real igm1     = 1.0/(mygam - 1.0);

  AthenaArray<Real> rad_arr, phi_arr, z_arr;
  rad_arr.NewAthenaArray(nc1);
  phi_arr.NewAthenaArray(nc1);
  z_arr.NewAthenaArray(nc1);

  for (int k=ks; k<=ke; ++k) { // include ghost zone
    for (int j=js; j<=je; ++j) { // prim, cons
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        GetCylCoord(pmb->pcoord, rad_arr(i), phi_arr(i), z_arr(i), i, j, k);

        const Real &gas_rho = w(IDN, k, j, i);
        const Real &gas_pre = w(IPR, k, j, i);

        Real inv_omega_dyn = std::sqrt((rad_arr(i)*rad_arr(i)*rad_arr(i))/gm0);
        Real internal_erg  = gas_rho*gas_pre*igm1;

        Real &kappa = phdif->kappa(HydroDiffusion::DiffProcess::aniso, k, j, i);
        kappa = internal_erg*inv_omega_dyn*inv_beta;
      }
    }
  }
  return;
}


void GetCylCoord(Coordinates *pco,Real &rad,Real &phi,Real &z,int i,int j,int k) {
  rad = pco->x1v(i);
  phi = pco->x2v(j);
  z   = pco->x3v(k);
  return;
}

//----------------------------------------------------------------------------------------
//! computes density in cylindrical coordinates
Real DenProfileCyl_gas(const Real rad, const Real phi, const Real z) {
  Real den;
  Real p_over_r = p0_over_r0;
  if (NON_BAROTROPIC_EOS) p_over_r = PoverRho(rad, phi, z);
  Real denmid = rho0*std::pow(rad/r0,dslope);
  Real dentem = denmid*std::exp(gm0/p_over_r*(1./std::sqrt(SQR(rad)+SQR(z))-1./rad));
  den = dentem;
  return std::max(den, dfloor);
}

Real DenProfileCyl_dust(const Real rad, const Real phi, const Real z, const Real den_ratio, const Real H_ratio) {
  Real den;
  Real p_over_r = p0_over_r0;
  if (NON_BAROTROPIC_EOS) p_over_r = PoverRho(rad, phi, z);
  Real denmid = den_ratio*rho0*std::pow(rad/r0,dslope);
  Real dentem = denmid*std::exp(gm0/(SQR(H_ratio)*p_over_r)*(1./std::sqrt(SQR(rad)+SQR(z))-1./rad));
  den = dentem;
  return std::max(den, dffloor);
}

//----------------------------------------------------------------------------------------
//! computes pressure/density in cylindrical coordinates
Real PoverRho(const Real rad, const Real phi, const Real z) {
  Real poverr;
  poverr = p0_over_r0*std::pow(rad/r0, pslope);
  return poverr;
}

//----------------------------------------------------------------------------------------
//! computes rotational velocity in cylindrical coordinates
Real VelProfileCyl_gas(const Real rad, const Real phi, const Real z) {
  Real p_over_r = PoverRho(rad, phi, z);
  Real vel = (dslope+pslope)*p_over_r/(gm0/rad) + (1.0+pslope)
             - pslope*rad/std::sqrt(rad*rad+z*z);
  vel = std::sqrt(gm0/rad)*std::sqrt(vel) - rad*Omega0;
  return vel;
}

Real VelProfileCyl_dust(const Real rad, const Real phi, const Real z) {
  Real dis = std::sqrt(SQR(rad) + SQR(z));
  Real vel = std::sqrt(gm0/dis) - rad*Omega0;
  return vel;
}

Real UserOrbitalVelocity(OrbitalAdvection *porb, Real x1, Real x2, Real x3) {
  return std::sqrt(porb->gm/x1)-porb->Omega0*x1;
}

Real UserOrbitalVelocity_r(OrbitalAdvection *porb, Real x1, Real x2, Real x3) {
  return -0.5*std::sqrt(porb->gm/x1)/x1-porb->Omega0;
}

Real UserOrbitalVelocity_z(OrbitalAdvection *porb, Real x1, Real x2, Real x3) {
  return 0.0;
}

int RefinementCondition(MeshBlock *pmb) {
  AthenaArray<Real> &w = pmb->phydro->w;
  AthenaArray<Real> &df_prim = pmb->pdustfluids->df_prim;
      Real rad(0.0), phi(0.0), z(0.0);
  Real maxeps  = 0.0;
  Real max_rad = 0.0;
  Real min_rad = 100.0;

  int k = pmb->ks;
  for (int j=pmb->js; j<=pmb->je; j++) {
    for (int i=pmb->is; i<=pmb->ie; i++) {
      GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k); // convert to cylindrical coordinates
      Real epsr_g = (std::abs(w(IDN,k,j,i+1) - 2.0*w(IDN,k,j,i) + w(IDN,k,j,i-1))
                   + std::abs(w(IDN,k,j+1,i) - 2.0*w(IDN,k,j,i) + w(IDN,k,j-1,i)))/w(IDN,k,j,i);

      Real epsp = (std::abs(w(IPR,k,j,i+1) - 2.0*w(IPR,k,j,i) + w(IPR,k,j,i-1))
                 + std::abs(w(IPR,k,j+1,i) - 2.0*w(IPR,k,j,i) + w(IPR,k,j-1,i)))/w(IPR,k,j,i);

      Real epsr_d = 0;
      if (NDUSTFLUIDS > 0) {
        for (int n = 0; n<NDUSTFLUIDS; n++) {
          int dust_id = n;
          int rho_id  = 4*dust_id;
          int v1_id   = rho_id + 1;
          int v2_id   = rho_id + 2;
          int v3_id   = rho_id + 3;

          Real epsr_temp = (std::abs(df_prim(rho_id,k,j,i+1) - 2.0*df_prim(rho_id,k,j,i) + df_prim(rho_id,k,j,i-1))
                      + std::abs(df_prim(rho_id,k,j+1,i) - 2.0*df_prim(rho_id,k,j,i) + df_prim(rho_id,k,j-1,i)))
                      /df_prim(rho_id,k,j,i);

          epsr_d = std::max(epsr_d, epsr_temp);
        }
      }

      Real eps = std::max(std::max(epsr_g, epsr_d), epsp);
      maxeps   = std::max(maxeps, eps);
      max_rad  = std::max(max_rad, rad);
      min_rad  = std::min(min_rad, rad);
    }
  }

  Real inv_mean_rad = 2.0/(min_rad + max_rad);

  // refine : curvature > 0.01
  if ((max_rad >= refine_rad_min) && ( min_rad <= refine_rad_max ) && (maxeps > (refine_factor*inv_mean_rad))) return 1;
  // derefinement: curvature < 0.005
  if ((max_rad < refine_rad_min) || ( min_rad > refine_rad_max ) || (maxeps < (derefine_factor*inv_mean_rad))) return -1;
  // otherwise, stay
  return 0;
}


void Vr_interpolate_outer_nomatter(const Real r_active, const Real r_ghost, const Real sigma_active,
    const Real sigma_ghost, const Real vr_active, Real &vr_ghost) {
  //if (sigma_active < TINY_NUMBER)
    //vr_ghost = vr_active >= 0.0 ? ((sigma_active+TINY_NUMBER)*r_active*vr_active)/(sigma_ghost*r_ghost) : 0.0;
  //else
  //vr_ghost = vr_active >= 0.0 ? (sigma_active*r_active*vr_active)/(sigma_ghost*r_ghost) : 0.0;
  vr_ghost = (sigma_active*r_active*vr_active)/(sigma_ghost*r_ghost);
  return;
}

} // namespace

//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values
void DiskInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh) {

  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;

  Real orb_defined;
  (pmb->porb->orbital_advection_defined) ? orb_defined = 1.0 : orb_defined = 0.0;

  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=1; i<=ngh; ++i) {
        Real rad_ghost, phi_ghost, z_ghost;
        GetCylCoord(pco, rad_ghost, phi_ghost, z_ghost, il-i, j, k);

        Real cs_square = PoverRho(rad_ghost, phi_ghost, z_ghost);
        Real omega_dyn = std::sqrt(gm0/(rad_ghost*rad_ghost*rad_ghost));
        Real vel_K     = vK(pmb->porb, pco->x1v(il-i), pco->x2v(j), pco->x3v(k));
        Real pre_diff  = (pslope + dslope)*cs_square;

        Real &gas_rho_ghost  = prim(IDN, k, j, il-i);
        Real &gas_vel1_ghost = prim(IM1, k, j, il-i);
        Real &gas_vel2_ghost = prim(IM2, k, j, il-i);
        Real &gas_vel3_ghost = prim(IM3, k, j, il-i);

        Real vis_vel_r    = -1.5*(nu_alpha*cs_square/rad_ghost/omega_dyn);
        Real vel_gas_phi  = VelProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
        vel_gas_phi      -= orb_defined*vel_K;
        Real vel_gas_z    = 0.0;

        gas_rho_ghost  = DenProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
        gas_vel1_ghost = vis_vel_r;
        gas_vel2_ghost = vel_gas_phi;
        gas_vel3_ghost = vel_gas_z;

        if (NON_BAROTROPIC_EOS) {
          Real &gas_pres_ghost = prim(IEN, k, j, il-i);
          gas_pres_ghost       = cs_square*gas_rho_ghost;
        }

        if (NDUSTFLUIDS > 0) {
          for (int n=0; n<NDUSTFLUIDS; ++n) {
            int dust_id = n;
            int rho_id  = 4*dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + 2;
            int v3_id   = rho_id + 3;

            Real &dust_rho_ghost  = prim_df(rho_id, k, j, il-i);
            Real &dust_vel1_ghost = prim_df(v1_id,  k, j, il-i);
            Real &dust_vel2_ghost = prim_df(v2_id,  k, j, il-i);
            Real &dust_vel3_ghost = prim_df(v3_id,  k, j, il-i);

            dust_rho_ghost     = DenProfileCyl_dust(rad_ghost, phi_ghost, z_ghost, initial_D2G[dust_id], Hratio[dust_id]);
            Real vel_dust_r    = weight_dust[dust_id]*pre_diff/(2.0*rad_ghost*omega_dyn);
            Real vel_dust_phi  = VelProfileCyl_dust(rad_ghost, phi_ghost, z_ghost);
            vel_dust_phi      -= orb_defined*vel_K;
            Real vel_dust_z    = 0.0;

            dust_vel1_ghost = vel_dust_r;
            dust_vel2_ghost = vel_dust_phi;
            dust_vel3_ghost = vel_dust_z;
          }
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values

void DiskOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh) {

  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;

  Real orb_defined;
  (pmb->porb->orbital_advection_defined) ? orb_defined = 1.0 : orb_defined = 0.0;

  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=1; i<=ngh; ++i) {
        Real rad_ghost, phi_ghost, z_ghost;
        GetCylCoord(pco, rad_ghost, phi_ghost, z_ghost, iu+i, j, k);

        Real cs_square = PoverRho(rad_ghost, phi_ghost, z_ghost);
        Real omega_dyn = std::sqrt(gm0/(rad_ghost*rad_ghost*rad_ghost));
        Real vis_vel_r = -1.5*(nu_alpha*cs_square/rad_ghost/omega_dyn);
        Real vel_K     = vK(pmb->porb, pco->x1v(iu+i), pco->x2v(j), pco->x3v(k));
        Real pre_diff  = (pslope + dslope)*cs_square;

        Real &gas_rho_ghost  = prim(IDN, k, j, iu+i);
        Real &gas_vel1_ghost = prim(IM1, k, j, iu+i);
        Real &gas_vel2_ghost = prim(IM2, k, j, iu+i);
        Real &gas_vel3_ghost = prim(IM3, k, j, iu+i);

        gas_rho_ghost     = DenProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
        Real vel_gas_phi  = VelProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
        vel_gas_phi      -= orb_defined*vel_K;
        Real vel_gas_z    = 0.0;

        gas_vel1_ghost = vis_vel_r;
        gas_vel2_ghost = vel_gas_phi;
        gas_vel3_ghost = vel_gas_z;

        if (NON_BAROTROPIC_EOS) {
          Real &gas_pres_ghost = prim(IEN, k, j, iu+i);
          gas_pres_ghost       = cs_square*gas_rho_ghost;
        }

        if (NDUSTFLUIDS > 0) {
          for (int n=0; n<NDUSTFLUIDS; ++n) {
            int dust_id = n;
            int rho_id  = 4*dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + 2;
            int v3_id   = rho_id + 3;

            Real &dust_rho_ghost  = prim_df(rho_id, k, j, iu+i);
            Real &dust_vel1_ghost = prim_df(v1_id,  k, j, iu+i);
            Real &dust_vel2_ghost = prim_df(v2_id,  k, j, iu+i);
            Real &dust_vel3_ghost = prim_df(v3_id,  k, j, iu+i);

            dust_rho_ghost     = DenProfileCyl_dust(rad_ghost, phi_ghost, z_ghost, initial_D2G[dust_id], Hratio[dust_id]);
            Real vel_dust_r    = weight_dust[dust_id]*pre_diff/(2.0*rad_ghost*omega_dyn);
            Real vel_dust_phi  = VelProfileCyl_dust(rad_ghost, phi_ghost, z_ghost);
            vel_dust_phi      -= orb_defined*vel_K;
            Real vel_dust_z    = 0.0;

            dust_vel1_ghost = vel_dust_r;
            dust_vel2_ghost = vel_dust_phi;
            dust_vel3_ghost = vel_dust_z;
          }
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values

void DiskInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh) {

  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;

  Real orb_defined;
  (pmb->porb->orbital_advection_defined) ? orb_defined = 1.0 : orb_defined = 0.0;

  for (int k=kl; k<=ku; ++k) {
    for (int j=1; j<=ngh; ++j) {
      for (int i=il; i<=iu; ++i) {
        Real rad_ghost, phi_ghost, z_ghost;
        GetCylCoord(pco, rad_ghost, phi_ghost, z_ghost, i, jl-j, k);

        Real cs_square   = PoverRho(rad_ghost, phi_ghost, z_ghost);
        Real omega_dyn   = std::sqrt(gm0/(rad_ghost*rad_ghost*rad_ghost));
        Real vis_vel_r = -1.5*(nu_alpha*cs_square/rad_ghost/omega_dyn);
        Real vel_K       = vK(pmb->porb, pco->x1v(i), pco->x2v(jl-j), pco->x3v(k));
        Real pre_diff    = (pslope + dslope)*cs_square;

        Real &gas_rho_ghost  = prim(IDN, k, jl-j, i);
        Real &gas_vel1_ghost = prim(IM1, k, jl-j, i);
        Real &gas_vel2_ghost = prim(IM2, k, jl-j, i);
        Real &gas_vel3_ghost = prim(IM3, k, jl-j, i);

        gas_rho_ghost     = DenProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
        Real vel_gas_phi  = VelProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
        vel_gas_phi      -= orb_defined*vel_K;
        Real vel_gas_z    = 0.0;

        gas_vel1_ghost = vis_vel_r;
        gas_vel2_ghost = vel_gas_phi;
        gas_vel3_ghost = vel_gas_z;

        if (NON_BAROTROPIC_EOS) {
          Real &gas_pres_ghost = prim(IEN, k, jl-j, i);
          gas_pres_ghost       = cs_square*gas_rho_ghost;
        }

        if (NDUSTFLUIDS > 0) {
          for (int n=0; n<NDUSTFLUIDS; ++n) {
            int dust_id = n;
            int rho_id  = 4*dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + 2;
            int v3_id   = rho_id + 3;

            Real &dust_rho_ghost  = prim_df(rho_id, k, jl-j, i);
            Real &dust_vel1_ghost = prim_df(v1_id,  k, jl-j, i);
            Real &dust_vel2_ghost = prim_df(v2_id,  k, jl-j, i);
            Real &dust_vel3_ghost = prim_df(v3_id,  k, jl-j, i);

            dust_rho_ghost     = DenProfileCyl_dust(rad_ghost, phi_ghost, z_ghost, initial_D2G[dust_id], Hratio[dust_id]);
            Real vel_dust_r    = weight_dust[dust_id]*pre_diff/(2.0*rad_ghost*omega_dyn);
            Real vel_dust_phi  = VelProfileCyl_dust(rad_ghost, phi_ghost, z_ghost);
            vel_dust_phi      -= orb_defined*vel_K;
            Real vel_dust_z    = 0.0;

            dust_vel1_ghost = vel_dust_r;
            dust_vel2_ghost = vel_dust_phi;
            dust_vel3_ghost = vel_dust_z;
          }
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values

void DiskOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh) {

  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;

  Real orb_defined;
  (pmb->porb->orbital_advection_defined) ? orb_defined = 1.0 : orb_defined = 0.0;

  for (int k=kl; k<=ku; ++k) {
    for (int j=1; j<=ngh; ++j) {
      for (int i=il; i<=iu; ++i) {
        Real rad_ghost, phi_ghost, z_ghost;
        GetCylCoord(pco, rad_ghost, phi_ghost, z_ghost, i, ju+j, k);

        Real cs_square   = PoverRho(rad_ghost, phi_ghost, z_ghost);
        Real omega_dyn   = std::sqrt(gm0/(rad_ghost*rad_ghost*rad_ghost));
        Real vis_vel_r = -1.5*(nu_alpha*cs_square/rad_ghost/omega_dyn);
        Real vel_K       = vK(pmb->porb, pco->x1v(i), pco->x2v(ju+j), pco->x3v(k));
        Real pre_diff    = (pslope + dslope)*cs_square;

        Real &gas_rho_ghost  = prim(IDN, k, ju+j, i);
        Real &gas_vel1_ghost = prim(IM1, k, ju+j, i);
        Real &gas_vel2_ghost = prim(IM2, k, ju+j, i);
        Real &gas_vel3_ghost = prim(IM3, k, ju+j, i);

        gas_rho_ghost     = DenProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
        Real vel_gas_phi  = VelProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
        vel_gas_phi      -= orb_defined*vel_K;
        Real vel_gas_z    = 0.0;

        gas_vel1_ghost = vis_vel_r;
        gas_vel2_ghost = vel_gas_phi;
        gas_vel3_ghost = vel_gas_z;

        if (NON_BAROTROPIC_EOS) {
          Real &gas_pres_ghost = prim(IEN, k, ju+j, i);
          gas_pres_ghost       = cs_square*gas_rho_ghost;
        }

        if (NDUSTFLUIDS > 0) {
          for (int n=0; n<NDUSTFLUIDS; ++n) {
            int dust_id = n;
            int rho_id  = 4*dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + 2;
            int v3_id   = rho_id + 3;

            Real &dust_rho_ghost  = prim_df(rho_id, k, ju+j, i);
            Real &dust_vel1_ghost = prim_df(v1_id,  k, ju+j, i);
            Real &dust_vel2_ghost = prim_df(v2_id,  k, ju+j, i);
            Real &dust_vel3_ghost = prim_df(v3_id,  k, ju+j, i);

            dust_rho_ghost     = DenProfileCyl_dust(rad_ghost, phi_ghost, z_ghost, initial_D2G[dust_id], Hratio[dust_id]);
            Real vel_dust_r    = weight_dust[dust_id]*pre_diff/(2.0*rad_ghost*omega_dyn);
            Real vel_dust_phi  = VelProfileCyl_dust(rad_ghost, phi_ghost, z_ghost);
            vel_dust_phi      -= orb_defined*vel_K;
            Real vel_dust_z    = 0.0;

            dust_vel1_ghost = vel_dust_r;
            dust_vel2_ghost = vel_dust_phi;
            dust_vel3_ghost = vel_dust_z;
          }
        }
      }
    }
  }
}


void LocalIsothermalEOS(MeshBlock *pmb, int il, int iu, int jl,
    int ju, int kl, int ku, AthenaArray<Real> &prim, AthenaArray<Real> &cons) {

  Real mygam = pmb->peos->GetGamma();
  Real igm1  = 1.0/(mygam - 1.0);

  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
#pragma omp simd
      for (int i=il; i<=iu; ++i) {
        Real rad, phi, z;
        GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);

        Real &gas_pres = prim(IPR, k, j, i);

        Real &gas_dens = cons(IDN, k, j, i);
        Real &gas_mom1 = cons(IM1, k, j, i);
        Real &gas_mom2 = cons(IM2, k, j, i);
        Real &gas_mom3 = cons(IM3, k, j, i);
        Real &gas_erg  = cons(IEN, k, j, i);

        Real inv_gas_dens = 1.0/gas_dens;
        gas_pres = PoverRho(rad, phi, z)*gas_dens;
        gas_erg  = gas_pres*igm1 + 0.5*(SQR(gas_mom1) + SQR(gas_mom2) + SQR(gas_mom3))*inv_gas_dens;
      }
    }
  }
  return;
}


void InnerWaveDampingGas(MeshBlock *pmb, const Real time, const Real dt, int il, int iu,
    int jl, int ju, int kl, int ku, AthenaArray<Real> &prim, AthenaArray<Real> &cons) {

  Real mygam = pmb->peos->GetGamma();
  Real igm1  = 1.0/(mygam - 1.0);

  Real inv_inner_damp = 1.0/inner_width_damping;
  Real inv_outer_damp = 1.0/outer_width_damping;

  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;

  Real orb_defined;
  (pmb->porb->orbital_advection_defined) ? orb_defined = 1.0 : orb_defined = 0.0;

  int nc1 = pmb->ncells1;

  AthenaArray<Real> omega_dyn, R_func, inv_damping_tau;
  AthenaArray<Real> rad_arr, phi_arr, z_arr;

  omega_dyn.NewAthenaArray(nc1);
  R_func.NewAthenaArray(nc1);
  inv_damping_tau.NewAthenaArray(nc1);
  rad_arr.NewAthenaArray(nc1);
  phi_arr.NewAthenaArray(nc1);
  z_arr.NewAthenaArray(nc1);

  for (int k=kl; k<=ku; ++k) {
    Real x3 = pmb->pcoord->x3v(k);
    for (int j=jl; j<=ju; ++j) {
      Real x2 = pmb->pcoord->x2v(j);
#pragma omp simd
      for (int i=il; i<=iu; ++i) {
        Real x1 = pmb->pcoord->x1v(i);
        GetCylCoord(pmb->pcoord, rad_arr(i), phi_arr(i), z_arr(i), i, j, k); // convert to cylindrical coordinates

        if (rad_arr(i) <= radius_inner_damping) {
          // See de Val-Borro et al. 2006 & 2007
          omega_dyn(i)       = std::sqrt(gm0/(rad_arr(i)*rad_arr(i)*rad_arr(i)));
          R_func(i)          = SQR((rad_arr(i) - radius_inner_damping)*inv_inner_damp);
          inv_damping_tau(i) = (damping_rate*omega_dyn(i));

          Real gas_rho_0   = DenProfileCyl_gas(rad_arr(i), phi_arr(i), z_arr(i));
          Real cs_square_0 = PoverRho(rad_arr(i), phi_arr(i), z_arr(i));

          Real vis_vel_r    = -1.5*(nu_alpha*cs_square_0/rad_arr(i)/omega_dyn(i));
          Real vel_gas_phi  = VelProfileCyl_gas(rad_arr(i), phi_arr(i), z_arr(i));
          vel_gas_phi      -= orb_defined*vK(pmb->porb, x1, x2, x3);
          Real vel_gas_z    = 0.0;

          Real gas_vel1_0 = vis_vel_r;
          Real gas_vel2_0 = vel_gas_phi;
          Real gas_vel3_0 = vel_gas_z;
          Real gas_pre_0  = cs_square_0*gas_rho_0;

          Real &gas_rho  = prim(IDN, k, j, i);
          Real &gas_vel1 = prim(IM1, k, j, i);
          Real &gas_vel2 = prim(IM2, k, j, i);
          Real &gas_vel3 = prim(IM3, k, j, i);
          Real &gas_pre  = prim(IPR, k, j, i);

          Real &gas_dens = cons(IDN, k, j, i);
          Real &gas_mom1 = cons(IM1, k, j, i);
          Real &gas_mom2 = cons(IM2, k, j, i);
          Real &gas_mom3 = cons(IM3, k, j, i);
          Real &gas_erg  = cons(IEN, k, j, i);

          Real delta_gas_rho  = (gas_rho_0  - gas_rho )*R_func(i)*inv_damping_tau(i)*dt;
          Real delta_gas_vel1 = (gas_vel1_0 - gas_vel1)*R_func(i)*inv_damping_tau(i)*dt;
          Real delta_gas_vel2 = (gas_vel2_0 - gas_vel2)*R_func(i)*inv_damping_tau(i)*dt;
          Real delta_gas_vel3 = (gas_vel3_0 - gas_vel3)*R_func(i)*inv_damping_tau(i)*dt;
          Real delta_gas_pre  = (gas_pre_0  - gas_pre )*R_func(i)*inv_damping_tau(i)*dt;

          gas_rho  += delta_gas_rho;
          gas_vel1 += delta_gas_vel1;
          gas_vel2 += delta_gas_vel2;
          gas_vel3 += delta_gas_vel3;
          gas_pre  += delta_gas_pre;

          gas_dens = gas_rho;
          gas_mom1 = gas_dens*gas_vel1;
          gas_mom2 = gas_dens*gas_vel2;
          gas_mom3 = gas_dens*gas_vel3;

          Real Ek = 0.5*(SQR(gas_mom1) + SQR(gas_mom2) + SQR(gas_mom3));
          gas_erg = gas_pre*igm1 + Ek/gas_dens;
        }
      }
    }
  }
  return;
}


void OuterWaveDampingGas(MeshBlock *pmb, const Real time, const Real dt, int il, int iu,
    int jl, int ju, int kl, int ku, AthenaArray<Real> &prim, AthenaArray<Real> &cons) {

  Real mygam = pmb->peos->GetGamma();
  Real igm1  = 1.0/(mygam - 1.0);

  Real inv_inner_damp = 1.0/inner_width_damping;
  Real inv_outer_damp = 1.0/outer_width_damping;

  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;

  Real orb_defined;
  (pmb->porb->orbital_advection_defined) ? orb_defined = 1.0 : orb_defined = 0.0;

  int nc1 = pmb->ncells1;

  AthenaArray<Real> omega_dyn, R_func, inv_damping_tau;
  AthenaArray<Real> rad_arr, phi_arr, z_arr;

  omega_dyn.NewAthenaArray(nc1);
  R_func.NewAthenaArray(nc1);
  inv_damping_tau.NewAthenaArray(nc1);
  rad_arr.NewAthenaArray(nc1);
  phi_arr.NewAthenaArray(nc1);
  z_arr.NewAthenaArray(nc1);

  for (int k=kl; k<=ku; ++k) {
    Real x3 = pmb->pcoord->x3v(k);
    for (int j=jl; j<=ju; ++j) {
      Real x2 = pmb->pcoord->x2v(j);
#pragma omp simd
      for (int i=il; i<=iu; ++i) {
        Real x1 = pmb->pcoord->x1v(i);
        GetCylCoord(pmb->pcoord, rad_arr(i), phi_arr(i), z_arr(i), i, j, k); // convert to cylindrical coordinates

        if (rad_arr(i) >= radius_outer_damping) {
          // See de Val-Borro et al. 2006 & 2007
          omega_dyn(i)       = std::sqrt(gm0/(rad_arr(i)*rad_arr(i)*rad_arr(i)));
          R_func(i)          = SQR((rad_arr(i) - radius_outer_damping)*inv_outer_damp);
          inv_damping_tau(i) = (damping_rate*omega_dyn(i));

          Real gas_rho_0   = DenProfileCyl_gas(rad_arr(i), phi_arr(i), z_arr(i));
          Real cs_square_0 = PoverRho(rad_arr(i), phi_arr(i), z_arr(i));

          Real vis_vel_r    = -1.5*(nu_alpha*cs_square_0/rad_arr(i)/omega_dyn(i));
          Real vel_gas_phi  = VelProfileCyl_gas(rad_arr(i), phi_arr(i), z_arr(i));
          vel_gas_phi      -= orb_defined*vK(pmb->porb, x1, x2, x3);
          Real vel_gas_z    = 0.0;

          Real gas_vel1_0 = vis_vel_r;
          Real gas_vel2_0 = vel_gas_phi;
          Real gas_vel3_0 = vel_gas_z;
          Real gas_pre_0  = cs_square_0*gas_rho_0;

          Real &gas_dens = cons(IDN, k, j, i);
          Real &gas_mom1 = cons(IM1, k, j, i);
          Real &gas_mom2 = cons(IM2, k, j, i);
          Real &gas_mom3 = cons(IM3, k, j, i);
          Real &gas_erg  = cons(IEN, k, j, i);

          Real &gas_rho  = prim(IDN, k, j, i);
          Real &gas_vel1 = prim(IM1, k, j, i);
          Real &gas_vel2 = prim(IM2, k, j, i);
          Real &gas_vel3 = prim(IM3, k, j, i);
          Real &gas_pre  = prim(IPR, k, j, i);

          Real delta_gas_rho  = (gas_rho_0  - gas_rho )*R_func(i)*inv_damping_tau(i)*dt;
          Real delta_gas_vel1 = (gas_vel1_0 - gas_vel1)*R_func(i)*inv_damping_tau(i)*dt;
          Real delta_gas_vel2 = (gas_vel2_0 - gas_vel2)*R_func(i)*inv_damping_tau(i)*dt;
          Real delta_gas_vel3 = (gas_vel3_0 - gas_vel3)*R_func(i)*inv_damping_tau(i)*dt;
          Real delta_gas_pre  = (gas_pre_0  - gas_pre )*R_func(i)*inv_damping_tau(i)*dt;

          gas_rho  += delta_gas_rho;
          gas_vel1 += delta_gas_vel1;
          gas_vel2 += delta_gas_vel2;
          gas_vel3 += delta_gas_vel3;
          gas_pre  += delta_gas_pre;

          gas_dens = gas_rho;
          gas_mom1 = gas_dens*gas_vel1;
          gas_mom2 = gas_dens*gas_vel2;
          gas_mom3 = gas_dens*gas_vel3;

          Real Ek = 0.5*(SQR(gas_mom1) + SQR(gas_mom2) + SQR(gas_mom3));
          gas_erg = gas_pre*igm1 + Ek/gas_dens;
        }
      }
    }
  }
  return;
}



void FixedDust(MeshBlock *pmb, int il, int iu, int jl, int ju, int kl, int ku,
    AthenaArray<Real> &prim_df, AthenaArray<Real> &cons_df) {

  int nc1 = pmb->ncells1;
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;

  Real orb_defined;
  (pmb->porb->orbital_advection_defined) ? orb_defined = 1.0 : orb_defined = 0.0;

  AthenaArray<Real> rad_arr, phi_arr, z_arr, cs_square, vel_K, pre_diff,
                    den_dust, vel_dust_r, vel_dust_phi, vel_dust_z, omega_dyn;

  rad_arr.NewAthenaArray(nc1);
  phi_arr.NewAthenaArray(nc1);
  z_arr.NewAthenaArray(nc1);
  cs_square.NewAthenaArray(nc1);
  vel_K.NewAthenaArray(nc1);
  pre_diff.NewAthenaArray(nc1);
  den_dust.NewAthenaArray(nc1);
  vel_dust_r.NewAthenaArray(nc1);
  vel_dust_phi.NewAthenaArray(nc1);
  vel_dust_z.NewAthenaArray(nc1);
  omega_dyn.NewAthenaArray(nc1);

  for (int k=kl; k<=ku; ++k) {
    Real x3 = pmb->pcoord->x3v(k);
    for (int j=jl; j<=ju; ++j) {
      Real x2 = pmb->pcoord->x2v(j);
#pragma omp simd
      for (int i=il; i<=iu; ++i) {
        Real x1 = pmb->pcoord->x1v(i);
        GetCylCoord(pmb->pcoord, rad_arr(i), phi_arr(i), z_arr(i), i, j, k);
        cs_square(i) = PoverRho(rad_arr(i), phi_arr(i), z_arr(i));
        vel_K(i)     = vK(pmb->porb, x1, x2, x3);
        pre_diff(i)  = (pslope + dslope)*cs_square(i);
        omega_dyn(i) = std::sqrt(gm0/(rad_arr(i)*rad_arr(i)*rad_arr(i)));
      }

      for (int n=0; n<NDUSTFLUIDS; ++n) {
        int dust_id = n;
        int rho_id  = 4*dust_id;
        int v1_id   = rho_id + 1;
        int v2_id   = rho_id + 2;
        int v3_id   = rho_id + 3;
#pragma omp simd
        for (int i=il; i<=iu; ++i) {
          //den_dust(i)      = DenProfileCyl_dust(rad_arr(i), phi_arr(i), z_arr(i),
                              //initial_D2G[dust_id], Hratio[dust_id]);

          const Real &gas_rho = pmb->phydro->w(IDN, k, j, i);
          den_dust(i)         = initial_D2G[dust_id]*gas_rho;

          vel_dust_r(i)    = weight_dust[dust_id]*pre_diff(i)/(2.0*rad_arr(i)*omega_dyn(i));
          vel_dust_phi(i)  = VelProfileCyl_dust(rad_arr(i), phi_arr(i), z_arr(i));
          vel_dust_phi(i) -= orb_defined*vel_K(i);
          vel_dust_z(i)    = 0.0;

          Real &dust_rho  = prim_df(rho_id, k, j, i);
          Real &dust_vel1 = prim_df(v1_id,  k, j, i);
          Real &dust_vel2 = prim_df(v2_id,  k, j, i);
          Real &dust_vel3 = prim_df(v3_id,  k, j, i);

          Real &dust_dens = cons_df(rho_id, k, j, i);
          Real &dust_mom1 = cons_df(v1_id,  k, j, i);
          Real &dust_mom2 = cons_df(v2_id,  k, j, i);
          Real &dust_mom3 = cons_df(v3_id,  k, j, i);

          dust_rho  = den_dust(i);
          dust_vel1 = vel_dust_r(i);
          dust_vel2 = vel_dust_phi(i);
          dust_vel3 = vel_dust_z(i);

          dust_dens = den_dust(i);
          dust_mom1 = den_dust(i)*vel_dust_r(i);
          dust_mom2 = den_dust(i)*vel_dust_phi(i);
          dust_mom3 = den_dust(i)*vel_dust_z(i);
        }
      }
    }
  }
  return;
}


void MeshBlock::UserWorkInLoop() {

  Real &time = pmy_mesh->time;
  Real &dt   = pmy_mesh->dt;
  Real mygam = peos->GetGamma();
  Real igm1  = 1.0/(mygam - 1.0);
  int dk     = NGHOST;
  if (block_size.nx3 == 1) dk = 0;

  int kl = ks - dk;     int ku = ke + dk;
  int jl = js - NGHOST; int ju = je + NGHOST;
  int il = is - NGHOST; int iu = ie + NGHOST;

  if (Isothermal_Flag)
    LocalIsothermalEOS(this, il, iu, jl, ju, kl, ku, phydro->w, phydro->u);

  if (Damping_Flag) {
    InnerWaveDampingGas(this, time, dt, il, iu, jl, ju, kl, ku, phydro->w, phydro->u);
    OuterWaveDampingGas(this, time, dt, il, iu, jl, ju, kl, ku, phydro->w, phydro->u);
  }

  if ((NDUSTFLUIDS > 0) && (time_drag != 0.0) && (time < time_drag))
    FixedDust(this, il, iu, jl, ju, kl, ku, pdustfluids->df_prim, pdustfluids->df_cons);

  return;
}
