//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file disk.cpp
//  \brief Initializes Keplerian disk with the option planet and multi-dust
// in the cylindrical coordinates.

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
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../dustfluids/dustfluids.hpp"
#include "../orbital_advection/orbital_advection.hpp"
#include "../utils/utils.hpp" // ran2()

namespace{
//function 
Real logr(Real x, RegionSize rs);
void GetCylCoord(Coordinates *pco,Real &rad,Real &phi,Real &z,int i,int j,int k);
void MyStoppingTime(MeshBlock *pmb, const Real time, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, AthenaArray<Real> &stopping_time);
Real PoverRho(const Real rad, const Real phi, const Real z);
Real Keplerian_velocity(const Real rad);
void GetDustDensityProfile(Real a_d[NDUSTFLUIDS], const AthenaArray<Real> &u, 
    const AthenaArray<Real> &df_cons, Real D2GRatio,
    int i, int j, int k);
void LocalIsothermalEOS(MeshBlock *pmb, int il, int iu, int jl,
    int ju, int kl, int ku, AthenaArray<Real> &prim, AthenaArray<Real> &cons);

// parameter
Real Hratio_gas,D2GRatio; 
Real zeta;
Real gm0, r0, amp;
Real igm1;

// Array
Real a_d[NDUSTFLUIDS];
Real initial_D2G[NDUSTFLUIDS];
Real Hratio[NDUSTFLUIDS];

// Flag
bool Isothermal_Flag;
}


//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//! \brief Function to initialize problem-specific data in mesh class.  Can also be used
//! to initialize variables which are global to (and therefore can be passed to) other
//! functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  
  Real x1rat = pin->GetReal("mesh", "x1rat");
  if (x1rat==-1.0) 
    EnrollUserMeshGenerator(X1DIR, logr);

  // Enroll user-defined dust stopping time
  if (NDUSTFLUIDS > 0)   
  	EnrollUserDustStoppingTime(MyStoppingTime);

  // Get parameters for gravitatonal potential of central point mass
  gm0 = pin->GetOrAddReal("problem", "GM", 1.0); // should change to the correct unit
  r0  = pin->GetOrAddReal("problem", "r0", 1.0);
  amp = pin->GetOrAddReal("problem", "amp", 0.01);
  zeta   = pin->GetReal("problem", "POWER_ZETA");
  Hratio_gas = pin->GetReal("problem", "Hratio_gas");
  D2GRatio = pin->GetOrAddReal("dust", "D2GRatio", 0.01);

  // Get the flag for isothermal EOS
  Isothermal_Flag  = pin->GetOrAddBoolean("problem", "Isothermal_Flag", true);

  return;
}


//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Set up the initial condition
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {

  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") != 0) {
    std::stringstream msg;
    msg << "ProblemGenerator::planet2D_coag must be setup in the cylindrical coordinate!" << std::endl;
    ATHENA_ERROR(msg);
  }
  if ((block_size.nx2 == 1) || (block_size.nx3 > 1)) {
    std::stringstream msg;
    msg << "ProblemGenerator::planet2D_coag must be setup in 2D!" << std::endl;
    ATHENA_ERROR(msg);
  }

  std::int64_t iseed = -1 - gid; // random seed

  // Initialize the dust physicsal properties : 
  //  a_d(size), initial_D2G(dust to gas ratio)
  if (NDUSTFLUIDS > 0) {  
    Real a_dust_min = pin->GetReal("dust", "a_dust_min");
    Real a_dust_max = pin->GetReal("dust", "a_dust_max");
    Real da = std::log10(a_dust_max/std::log10(a_dust_min))/NDUSTFLUIDS;
    for (int n=0; n<NDUSTFLUIDS; ++n) {
      a_d[n] = pow(10, std::log10(a_dust_min) + n*da);
      //initial_D2G[n] = pin->GetOrAddReal("dust", "initial_D2G_" + std::to_string(n+1), 0.01);
      //Hratio[n]      = pin->GetReal("dust", "Hratio_" + std::to_string(n+1)); 
    }
  }

  Real beta   = pin->GetReal("problem", "POWER_BETA");
  Real M0     = pin->GetReal("problem", "M_DISK");
  Real rc     = pin->GetReal("problem", "rc_exp_disk");
  bool exp_disk = pin->GetBoolean("problem", "i_exp_disk");
  amp = pin->GetOrAddReal("problem", "amp", 0.01);
  Real gamma_gas = peos->GetGamma();
  Real igm1      = 1.0/(gamma_gas - 1.0);

  // Initialize density and velocity for dust and gas
  for (int k=ks; k<=ke; ++k){
    Real x3 = pcoord->x3v(k);
    for (int j=js; j<=je; ++j){
      Real x2 = pcoord->x2v(j);
      for (int i=is; i<=ie; ++i){
        Real x1 = pcoord->x1v(i);

        Real &gas_dens = phydro->u(IDN, k, j, i);
        Real &gas_mom1 = phydro->u(IM1, k, j, i);
        Real &gas_mom2 = phydro->u(IM2, k, j, i);
        Real &gas_mom3 = phydro->u(IM3, k, j, i);
				
        // compute the gas density and the velocity profile
        Real rad, phi, z;
        GetCylCoord(pcoord, rad, phi, z, i, j, k);

        // rho(r) = M0* (r/rc)^{-beta}*exp(-r(r/rc)^{2-beta})
        if (exp_disk){
          gas_dens = M0*std::pow(rad/rc,-beta)*std::exp(-pow(rad/rc,2-beta));
        }else{
          gas_dens = M0*std::pow(rad/rc,-beta);
        }

        // compute the velocity profile
        Real cs_square   = PoverRho(rad, phi, z);
        Real vis_vel_r   = 0.0; //-1.5*(nu_alpha*cs_square/rad/omega_dyn);
        Real vel_gas_phi = Keplerian_velocity(rad);
        Real vel_gas_z   = 0.0;

        Real delta_gas_vel1 = amp*std::sqrt(cs_square)*(ran2(&iseed) - 0.5);
        Real delta_gas_vel2 = amp*std::sqrt(cs_square)*(ran2(&iseed) - 0.5);
        Real delta_gas_vel3 = amp*std::sqrt(cs_square)*(ran2(&iseed) - 0.5);

        gas_mom1 = gas_dens*(vis_vel_r + delta_gas_vel1);
        gas_mom2 = gas_dens*(vel_gas_phi + delta_gas_vel2);
        gas_mom3 = gas_dens*(vel_gas_z + delta_gas_vel3);

        if (NON_BAROTROPIC_EOS) {
          phydro->u(IEN, k, j, i)  = cs_square*phydro->u(IDN, k, j, i)*igm1;
          phydro->u(IEN, k, j, i) += 0.5*(SQR(phydro->u(IM1, k, j, i))+SQR(phydro->u(IM2, k, j, i))
                                        + SQR(phydro->u(IM3, k, j, i)))/phydro->u(IDN, k, j, i);
        }

        // Initialize the dust density and momentum
        if (NDUSTFLUIDS > 0) { 
          GetDustDensityProfile(a_d, phydro->u, pdustfluids->df_cons, D2GRatio, i, j, k);
          for (int n=0; n<NDUSTFLUIDS; ++n) {
            int dust_id = n;
            int rho_id  = 4*dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + 2;
            int v3_id   = rho_id + 3;
            Real dust_den = pdustfluids->df_cons(rho_id, k, j, i);
            pdustfluids->df_cons(v1_id,  k, j, i) = dust_den*vis_vel_r;
            pdustfluids->df_cons(v2_id,  k, j, i) = dust_den*vel_gas_phi;
            pdustfluids->df_cons(v3_id,  k, j, i) = dust_den*vel_gas_z;
          }
        }
      }
    }
  }
  return;
}

//========================================================================================
//! \fn void Mesh::UserWorkAfterLoop()
//! \brief correct the sound speed for gas
//========================================================================================
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

  return;
}


//========================================================================================
//! \fn void Mesh::UserWorkAfterLoop(ParameterInput *pin)
//! \brief Check radius of sphere to make sure it is round
//========================================================================================

void Mesh::UserWorkAfterLoop(ParameterInput *pin) {
}


//self define function
namespace {

//----------------------------------------------------------------------------------------
//! transform to cylindrical coordinate

void GetCylCoord(Coordinates *pco,Real &rad,Real &phi,Real &z,int i,int j,int k) {
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    rad=pco->x1v(i);
    phi=pco->x2v(j);
    z=pco->x3v(k);
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    rad=std::abs(pco->x1v(i)*std::sin(pco->x2v(j)));
    phi=pco->x3v(k);
    z=pco->x1v(i)*std::cos(pco->x2v(j));
  }
  return;
}


// Create a grid for x-1 dim in log 10 space and return in real space
Real logr(Real x, RegionSize rs){
  Real dr = std::log10(rs.x1max/rs.x1min)/rs.nx1;
  return std::pow(10, log10(rs.x1min) + x*dr);
}

// function of setting a kepplerian velocity profile
Real Keplerian_velocity(const Real rad) {
  Real vk = std::sqrt(gm0/rad);
  return vk;
}

//========================================================================================
//! \fn void Mesh::MyStoppingTime(MeshBlock *pmb, const Real time, const AthenaArray<Real> &prim,
//    const AthenaArray<Real> &prim_df, AthenaArray<Real> &stopping_time) {
//! \brief Compute the stoping time assuming the epstin regime
//========================================================================================
void MyStoppingTime(MeshBlock *pmb, const Real time, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, AthenaArray<Real> &stopping_time) {

  for (int n=0; n<NDUSTFLUIDS; ++n) {
    int dust_id = n;
    int rho_id  = 4*dust_id;
    Real a_dust = a_d[dust_id];
    for (int k=pmb->ks; k<=pmb->ke; ++k) {
      for (int j=pmb->js; j<=pmb->je; ++j) {
#pragma omp simd
        for (int i=pmb->is; i<=pmb->ie; ++i) {
          Real gas_dens = prim(IDN, k, j, i);
          Real dust_dens = prim_df(rho_id, k, j, i);
          //GetCylCoord(pmb->pcoord, rad_arr(i), phi_arr(i), z_arr(i), i, j, k);
          Real &st_time = stopping_time(dust_id, k, j, i);
          st_time = dust_dens*a_dust/gas_dens*PI/2;
        }
      }
    }
  }
  return;
}

//========================================================================================
//! \fn MySource(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
//    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
//    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s) 
//! \brief added the gravitional source from star and planet
//========================================================================================
void MySource(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s) {

  // Adding Plantary Gravity
  //if ((gmp > 0.0) && (time >= t0_planet))
  //  PlanetaryGravity(pmb, time, dt, prim, prim_df, prim_s, bcc, cons, cons_df, cons_s);

  return;
}

//----------------------------------------------------------------------------------------
//! computes pressure/density in cylindrical coordinates
Real PoverRho(const Real rad, const Real phi, const Real z) {
  Real poverr;
  // cs^2 = T0*(r/r0)^(zeta)
  poverr = pow(Hratio_gas,2)*std::pow(rad/r0, zeta);
  return poverr;
}

void PlanetaryGravity(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s) {
}

// function to compute the dust density profile
void GetDustDensityProfile(Real a_d[NDUSTFLUIDS], const AthenaArray<Real> &u, 
    const AthenaArray<Real> &df_cons, Real D2GRatio, 
    int i, int j, int k) {
  Real sum_ri_rmin = 0.0;
  Real a_min = a_d[0];
  for (int a_i=0; i<NDUSTFLUIDS; ++a_i)
    sum_ri_rmin += std::sqrt(a_d[a_i]/a_min);  

  for (int n=0; n<NDUSTFLUIDS; ++n) {
    int dust_id = n;
    int rho_id  = 4*dust_id;
    Real a_dust = a_d[dust_id];
    Real gas_dens =  u(IDN, k, j, i);
    Real dust_dens = df_cons(rho_id, k, j, i);
    Real D0 = D2GRatio*gas_dens;
    dust_dens = D0/sum_ri_rmin*std::sqrt(a_dust/a_min);
  }
  return;
}

// function to set the local isothermal condition after each time step
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
        // gas pressure = cs^2 * gas density
        gas_pres = PoverRho(rad, phi, z)*gas_dens;
        gas_erg  = gas_pres*igm1 + 0.5*(SQR(gas_mom1) + SQR(gas_mom2) + SQR(gas_mom3))*inv_gas_dens;
      }
    }
  }
  return;
}

}