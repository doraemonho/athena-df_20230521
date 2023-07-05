//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file disk.cpp
//  \brief Initializes stratified Keplerian accretion disk in both cylindrical and
//  spherical polar coordinates.  Initial conditions are in vertical hydrostatic eqm.

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
// #include "../dustfluids/dustfluids.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../orbital_advection/orbital_advection.hpp"
#include "../parameter_input.hpp"

namespace {
void GetCylCoord(Coordinates *pco,Real &rad,Real &phi,Real &z,int i,int j,int k);
void GetSphCoord(Coordinates *pco,Real &rad,Real &theta,Real &phi,int i,int j,int k);
void Car2Sph(Real rad, Real theta,Real phi, Real vx, Real vy, Real vz, Real &v1, Real &v2, Real &v3);
void Sph2Car(Real rad, Real theta,Real phi, Real &vx, Real &vy, Real &vz, Real v1, Real v2, Real v3);
// Real DenProfileCyl(const Real rad, const Real phi, const Real z);
// Real PoverR(const Real rad, const Real phi, const Real z);
// void VelProfileCyl(const Real rad, const Real phi, const Real z,
//                    Real &v1, Real &v2, Real &v3);
void Get_vel_new_fromMC(AthenaArray<Real> gas_vel_array,
    Real rho_g, Real rho_g1, AthenaArray<Real> rho_d_array, AthenaArray<Real> rho_d_array1, Real drho,
    AthenaArray<Real> rho_ratio_array, AthenaArray<Real> dust_vel_array, bool istracer[NDUSTFLUIDS],
    AthenaArray<Real> &gas_vel_array1, Real &E_kg, AthenaArray<Real> &E_kd_array);
void Get_E_kg(AthenaArray<Real> gas_vel_array, Real &E_kg);
Real Get_T_rhoe_g(Real rhoe_g, Real rho_g, Real fv);
Real Get_T_rhoe(Real rhoe, Real rho_g, AthenaArray<Real> rho_d_array, Real fv);
Real Get_rhomu_d(AthenaArray<Real> rho_d_array);
Real Get_rhoe(Real rhoE_total, Real rho_g, Real E_kg, AthenaArray<Real> rho_d_array, AthenaArray<Real> E_kd_array);
Real Get_Z(Real rho_g, Real T);
void phase_trans(Real rhoe, Real rho_g, AthenaArray<Real> rho_I, Real rho_v, Real &drho);
Real Get_mu(Real fv);
// problem parameters which are useful to make global to this file
Real mplanet, tsoft, rsoft, gamma_gas, x1min, x1max;
int nx1, nx2, nx3;
bool size_change, mom_correct_Flag;
Real dfloor, pfloor, float_min;
Real min_tol, max_dfvdt, dust_start_injection, injection_Tsoft;
// pebble property
Real f_ICE_inter0, m_p0, rho_sil_inter, rho_ice_inter;

// Dust fluid module
Real initial_D2G[NDUSTFLUIDS], Stokes_number[NDUSTFLUIDS], Hratio[NDUSTFLUIDS], boundary_D2G[NDUSTFLUIDS], const_stopping_time[NDUSTFLUIDS];
Real const_nu_dust[NDUSTFLUIDS];
// User Sources
void MySource(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s);

void PlanetaryGravity(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s);

void copy_velocity(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
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

Real MyMeshSpacingX1(Real x, RegionSize rs);
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
void DiskInnerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                AthenaArray<Real> &prim_df, FaceField &b, Real time, Real dt,
                int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskOuterX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                AthenaArray<Real> &prim_df, FaceField &b, Real time, Real dt,
                int il, int iu, int jl, int ju, int kl, int ku, int ngh);

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in mesh class.  Can also be used
//  to initialize variables which are global to (and therefore can be passed to) other
//  functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  // Get parameters for gravitatonal potential of central poix1minnt mass
  mplanet = pin->GetReal("problem","MPLANET");
  tsoft = pin->GetReal("problem","TSOFT");
  rsoft = pin->GetOrAddReal("problem","RSOFT",0.0);
  gamma_gas = pin->GetReal("hydro","gamma");
  x1min = pin->GetReal("mesh","x1min");
  x1max = pin->GetReal("mesh","x1max");
  nx1 = pin->GetInteger("mesh","nx1");
  nx2 = pin->GetInteger("mesh","nx2");
  nx3 = pin->GetInteger("mesh","nx3");
  size_change    = pin->GetBoolean("problem",   "size_change");
  mom_correct_Flag = pin->GetBoolean("problem",   "mom_correct_Flag");

  //dustfluid module
  // Dust to gas ratio && dust stopping time
  if (NDUSTFLUIDS > 0) {
    for (int n=0; n<NDUSTFLUIDS; n++) {
      initial_D2G[n]   = pin->GetReal("dust", "initial_D2G_" + std::to_string(n+1));
      Stokes_number[n] = pin->GetReal("dust", "stopping_time_" + std::to_string(n+1));
      boundary_D2G[n] = pin->GetReal("dust", "boundary_D2G_" + std::to_string(n+1));
      const_stopping_time[n] = pin->GetReal("dust", "stopping_time_" + std::to_string(n+1));
      const_nu_dust[n] = pin->GetReal("dust", "nu_dust_" + std::to_string(n+1));
      // Hratio[n]        = pin->GetReal("dust", "Hratio_" + std::to_string(n+1));
    }
  }

  // minimum tolerance of dfv
  min_tol = pin->GetOrAddReal("problem", "min_tol", 1.e-7);
  max_dfvdt = pin->GetOrAddReal("problem", "max_dfvdt", 1.e-5);
  dust_start_injection = pin->GetReal("problem", "dust_start_injection");
  injection_Tsoft = pin->GetReal("problem", "injection_Tsoft");
  // pebble properties
  f_ICE_inter0 = pin->GetOrAddReal("problem", "f_ICE_inter0", 0.5);
  rho_sil_inter = pin->GetOrAddReal("problem", "rho_sil_inter", 3.0); // cgs, internal density of silicate
  rho_ice_inter = pin->GetOrAddReal("problem", "rho_ice_inter", 1.0); // cgs, internal density of ice
  m_p0 = -1.0;

  float_min = std::numeric_limits<float>::min();
  dfloor=pin->GetOrAddReal("hydro","dfloor",(1024*(float_min)));
  pfloor=pin->GetOrAddReal("hydro","pfloor",(1024*(float_min)));

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
  // if (mesh_bcs[BoundaryFace::inner_x3] == GetBoundaryFlag("user")) {
  //   EnrollUserBoundaryFunction(BoundaryFace::inner_x3, DiskInnerX3);
  // }
  // if (mesh_bcs[BoundaryFace::outer_x3] == GetBoundaryFlag("user")) {
  //   EnrollUserBoundaryFunction(BoundaryFace::outer_x3, DiskOuterX3);
  // }

  // Enroll planetary and star force
  EnrollUserExplicitSourceFunction(MySource);

  // Enroll userdef mesh, x1
  if (pin->GetReal("mesh","x1rat") < 0.0){
    EnrollUserMeshGenerator(X1DIR, MyMeshSpacingX1);
  }
  // gas property output
  std::cout << "a_semi=" << a_semi <<std::endl;
  std::cout << "T_a=" << T_a<<std::endl;
  std::cout << "Cd_water=" << Cd_water<<std::endl;
  
  return;
}


// enroll user defined output variables
void MeshBlock::InitUserMeshBlockData(ParameterInput *pin){
  AllocateUserOutputVariables(2);
  SetUserOutputVariableName(0,"Tem");
  SetUserOutputVariableName(1,"gamma");
  // SetUserOutputVariableName(2,"st");
  // SetUserOutputVariableName(3,"m_p");
  // SetUserOutputVariableName(3,"dfvdt");
  // SetUserOutputVariableName(4,"nu");
  // SetUserOutputVariableName(5,"mass_cons");
  // SetUserOutputVariableName(6,"mom1_cons");
  // SetUserOutputVariableName(7,"mom2_cons");
  // SetUserOutputVariableName(8,"erg_cons");
  // SetUserOutputVariableName(4,"dif");
  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Initializes Keplerian accretion disk.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  Real rad(0.0), theta(0.0), phi(0.0);
  Real v1, v2, v3;
  Real x, y, z;
  Real vx, vy, vz;
  Real rhov;
  if (PHASE_CHANGE) {
    rhov = initial_D2G[NDUSTFLUIDS-1];
  } else {
    rhov = 0.;
  }
  
  //  Initialize density and momenta
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
          // assign spherical coordinate
          GetSphCoord(pcoord,rad,theta,phi,i,j,k);
          x = rad* std::sin(theta)* std::cos(phi);
          y = rad* std::sin(theta)* std::sin(phi);
          z = rad* std::cos(theta);
        } else if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
          x = pcoord->x1v(i);
          y = pcoord->x1v(j);
          z = pcoord->x1v(k);
        }
        
        // Bethune & Rafikov 2019
        // z = (z > 4.0) ? 4.0 : z;
        z = 0.0;
        // assign velocity in cartesian coordinate
        vx = 0.;
        vy = -1.5*x;
        vz = 0.;
        
        if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
          // transfer from cartesian to spherical
          Car2Sph(rad, theta, phi, vx, vy, vz, v1, v2, v3);
        } else if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
          v1 = vx;
          v2 = vy;
          v3 = vz;
        }
        // 
        phydro->u(IDN,k,j,i) = (1.0 + rhov)* std::exp(-0.5* SQR(z)); // stratified density.
        phydro->u(IM1,k,j,i) = phydro->u(IDN,k,j,i)*v1;
        phydro->u(IM2,k,j,i) = phydro->u(IDN,k,j,i)*v2;
        phydro->u(IM3,k,j,i) = phydro->u(IDN,k,j,i)*v3;
        
        if (NON_BAROTROPIC_EOS) {
          Real prs = (1.0 + rhov*mu_xy/mu_z)* std::exp(-0.5* SQR(z));
          phydro->u(IEN,k,j,i) = prs/(gamma_gas - 1.0);
          phydro->u(IEN,k,j,i) += 0.5*(SQR(phydro->u(IM1,k,j,i)) + SQR(phydro->u(IM2,k,j,i)) + SQR(phydro->u(IM3,k,j,i)))/phydro->u(IDN,k,j,i);
          phydro->Tem(k,j,i) = mu_xy*KELVIN;
        }
        
        if (NDUSTFLUIDS > 0) {
          for (int n=0; n<NDUSTFLUIDS; ++n) {
            int dust_id = n;
            int rho_id  = 4*dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + 2;
            int v3_id   = rho_id + 3;

            Real d2g = initial_D2G[n]; // dust to gas ratio
            Real den_dust = d2g*phydro->u(IDN,k,j,i); // dust density

            pdustfluids->df_cons(rho_id, k, j, i) = den_dust;
            pdustfluids->df_cons(v1_id,  k, j, i) = v1*den_dust;
            pdustfluids->df_cons(v2_id, k, j, i) = v2*den_dust;
            pdustfluids->df_cons(v3_id, k, j, i) = v3*den_dust;
            // initialize pebble stopping time and size.
            if(!pdustfluids->istracer[dust_id]){
              pdustfluids->stopping_time_array(dust_id,k,j,i) = const_stopping_time[dust_id];
            }
					}
				}
        
      }
    }
  }
  return;
}

//! \warning not modifed for 3D
// void MeshBlock::UserWorkInLoop(){

//   // copy gas velocity to tracer;
//   for (int k=ks; k<=ke; ++k) {
//     for (int j=js-NGHOST; j<=je+NGHOST; ++j) {
//       for (int i=is-NGHOST; i<=ie+NGHOST; ++i) {
//         // pre-calculation
//         const Real &gas_den = phydro->w(IDN,k ,j ,i);
//         const Real &gas_vel1 = phydro->w(IVX, k, j, i);
//         const Real &gas_vel2 = phydro->w(IVY, k, j, i);

//         if (NDUSTFLUIDS > 0) {
//           for (int n=0; n<NDUSTFLUIDS; ++n) {
//             int dust_id = n;
//             int rho_id  = 4*dust_id;
//             int v1_id   = rho_id + 1;
//             int v2_id   = rho_id + 2;
//             int v3_id   = rho_id + 3;
            
//             if(pdustfluids->istracer[n]){
//               const Real &dust_rho  = pdustfluids->df_prim(rho_id, k, j, i);
//               Real &dust_vel1 = pdustfluids->df_prim(v1_id,  k, j, i);
//               Real &dust_vel2 = pdustfluids->df_prim(v2_id,  k, j, i);

//               const Real &dust_den  = pdustfluids->df_cons(rho_id, k, j, i);
//               Real &dust_mom1 = pdustfluids->df_cons(v1_id,  k, j, i);
//               Real &dust_mom2 = pdustfluids->df_cons(v2_id,  k, j, i);

//               dust_vel1 = gas_vel1;
//               dust_vel2 = gas_vel2;
//               dust_mom1 = dust_den*gas_vel1;
//               dust_mom2 = dust_den*gas_vel2;
//             }
//           }
//         }
//       }
//     }
//   }

//   if(PHASE_CHANGE){
//       // used for cooling sub-cycle
//     for (int k=ks; k<=ke; ++k) {
//       for (int j=js-NGHOST; j<=je+NGHOST; ++j) {
//         for (int i=is-NGHOST; i<=ie+NGHOST; ++i) {
//           // store some constants of pebbles. (No vapor included here to keep the code safe)
//           AthenaArray<Real> E_kd_array, rho_d_array0, rho_d_array, rho_d_array1, rho_ratio_array;
//           E_kd_array.NewAthenaArray(NDUSTFLUIDS-1);
//           rho_d_array0.NewAthenaArray(NDUSTFLUIDS-1);
//           rho_d_array.NewAthenaArray(NDUSTFLUIDS-1);
//           rho_d_array1.NewAthenaArray(NDUSTFLUIDS-1);
//           rho_ratio_array.NewAthenaArray(NDUSTFLUIDS-1);
//           // store velocity
//           AthenaArray<Real> gas_vel_array0, gas_vel_array, gas_vel_array1, dust_vel_array;
//           gas_vel_array0.NewAthenaArray(NDIM);
//           gas_vel_array.NewAthenaArray(NDIM);
//           gas_vel_array1.NewAthenaArray(NDIM);
//           dust_vel_array.NewAthenaArray(NDUSTFLUIDS-1,NDIM);

//           // pre-calculation
//           Real &gas_rho  = phydro->w(IDN, k, j, i);
//           Real &gas_vel1 = phydro->w(IVX, k, j, i);
//           Real &gas_vel2 = phydro->w(IVY, k, j, i);
//           Real &pres      = phydro->w(IPR, k, j, i);

//           Real &rho_g  = phydro->u(IDN, k, j, i);
//           Real &gas_mom1 = phydro->u(IM1, k, j, i);
//           Real &gas_mom2 = phydro->u(IM2, k, j, i);
//           // Real &gas_mom3 = cons(IM3, k, j, i);
//           Real &gas_erg  = phydro->u(IEN, k, j, i);
//           // gas kinetic energy
//           Real E_kg = 0.5*(SQR(gas_vel1)+SQR(gas_vel2));
//           // store initial density, velocity 
//           gas_vel_array(0) = gas_vel1;
//           gas_vel_array(1) = gas_vel2;
//           gas_vel_array1 = gas_vel_array;
          
//           // pebble 1:ice 
//           int dust_id = 0;
//           int rho_id  = 4*dust_id;
//           int v1_id   = rho_id + 1;
//           int v2_id   = rho_id + 2;
//           int v3_id   = rho_id + 3;

//           Real &d1_rho  = pdustfluids->df_prim(rho_id, k, j, i);
//           Real &d1_vel1 = pdustfluids->df_prim(v1_id,  k, j, i);
//           Real &d1_vel2 = pdustfluids->df_prim(v2_id,  k, j, i);

//           Real &rho_I  = pdustfluids->df_cons(rho_id, k, j, i);
//           Real &d1_mom1 = pdustfluids->df_cons(v1_id,  k, j, i);
//           Real &d1_mom2 = pdustfluids->df_cons(v2_id,  k, j, i);
//           // Real &dust_mom3 = cons_df(v3_id,  k, j, i);
//           // ICE kinetic energy
//           E_kd_array(0) = 0.5*(SQR(d1_vel1)+SQR(d1_vel2));
//           rho_d_array(0) = rho_I;
//           dust_vel_array(dust_id,0) = d1_vel1;
//           dust_vel_array(dust_id,1) = d1_vel2;

//           // pebble 2, dust, st = 0, ice:
//           dust_id = 1;
//           rho_id  = 4*dust_id;
//           v1_id   = rho_id + 1;
//           v2_id   = rho_id + 2;
//           v3_id   = rho_id + 3;

//           Real &d2_rho  = pdustfluids->df_prim(rho_id, k, j, i);
//           Real &d2_vel1 = pdustfluids->df_prim(v1_id,  k, j, i);
//           Real &d2_vel2 = pdustfluids->df_prim(v2_id,  k, j, i);

//           Real &rho_d2  = pdustfluids->df_cons(rho_id, k, j, i);
//           Real &d2_mom1 = pdustfluids->df_cons(v1_id,  k, j, i);
//           Real &d2_mom2 = pdustfluids->df_cons(v2_id,  k, j, i);
//           // Real &dust_mom3 = cons_df(v3_id,  k, j, i);
//           // ICE kinetic energy
//           E_kd_array(1) = 0.5*(SQR(d2_vel1)+SQR(d2_vel2));
//           rho_d_array(1) = rho_d2;
//           dust_vel_array(dust_id,0) = d2_vel1;
//           dust_vel_array(dust_id,1) = d2_vel2;
          
//           // tracer particle &  vapor fraction:
//           dust_id = NDUSTFLUIDS-1;
//           rho_id  = 4*dust_id;
//           v1_id   = rho_id + 1;
//           v2_id   = rho_id + 2;
//           v3_id   = rho_id + 3;
          
//           Real &v1_rho = pdustfluids->df_prim(rho_id, k, j, i);
//           Real &v1_vel1 = pdustfluids->df_prim(v1_id, k, j, i);
//           Real &v1_vel2 = pdustfluids->df_prim(v2_id, k, j, i);

//           Real &rho_v  = pdustfluids->df_cons(rho_id, k, j, i);
//           Real &v1_mom1 = pdustfluids->df_cons(v1_id, k, j, i);
//           Real &v1_mom2 = pdustfluids->df_cons(v2_id, k, j, i);

//           Real fv = rho_v/rho_g;
//           Real fv0 = fv;

//           if(std::isnan(rho_g)){
//             std::cout << "rho_g is nan \n" << std::endl;
//             quick_exit(1);
//           }
//           if(std::isnan(gas_mom1)){
//               std::cout << "gas_mom1 is nan \n" << std::endl;
//               quick_exit(1);
//           }
//           if(std::isnan(gas_mom2)){
//               std::cout << "gas_mom2 is nan \n" << std::endl;
//               quick_exit(1);
//           }
//           if(std::isnan(gas_erg)){
//               std::cout << "gas_erg is nan \n" << std::endl;
//               quick_exit(1);
//           }

//           // dust array initiation
//           rho_d_array1 = rho_d_array;

//           // // verify mass/momentum/energy conservation
//           // Real mass_total1 = rho_I + rho_d2 + rho_g;
//           // Real mom1_total1 = gas_mom1 + d1_mom1 + d2_mom1;
//           // Real mom2_total1 = gas_mom2 + d1_mom2 + d2_mom2;

//           // store the total energy: the conserved quantity
//           Real rhoe_g = gas_erg-E_kg*rho_g;
//           Real Tem = Get_T_rhoe_g(rhoe_g,rho_g,fv);
//           Real rhomu_d = Get_rhomu_d(rho_d_array); // chemical potential of dust. mu_g is set to be 0.
//           Real rhoE_total = gas_erg + rhomu_d + Cd_water*(rho_I + rho_d2)*Tem;
//           for (int m = 0; m<= NDUSTFLUIDS-2; ++m){
//             rhoE_total += E_kd_array(m)*rho_d_array(m);
//           }

//           // std::cout << "tem =" << Tem << std::endl;
          
//           // // calculate analytical evaporation rate (schoonenberg+ 2017);
//           // Real P_z = rho_v/(KELVIN*mu_z)*Tem;
//           // Real P_eq = P_eq0*std::exp(-T_a/Tem);
//           // Real dpebdt = 3.0*std::sqrt(8.0/PI)*rho_v*(rho_d2+rho_I)*(1.0-P_eq/P_z);
//           // dpebdt /= (0.7*1.0/(UNIT_DENSITY*UNIT_LENGTH));
//           // pdustfluids->dfvdt_calc(k,j,i) = dpebdt*(1.0-fv0)/rho_g0;
          
//           // define variables used in intermediate steps
//           Real drho, rho_g1, rho_v1;
//           Real fx0, fx1, fx2, x0, x1, x2; // secant points
//           Real rhoe, rhoe1; // total internal energy density
//           rho_g1 = rho_g;

//           // first calculation of phase change quantity and save their initial value
//           rhoe = Get_rhoe(rhoE_total,rho_g,E_kg,rho_d_array,E_kd_array);
//           phase_trans(rhoe,rho_g,rho_d_array,rho_v, drho);
//           fx0 = drho;
//           x0 = rho_v;
//           // Loop starts
//           Real sign = (drho > 0. ? 1.0:-1.0);
//           Real dt = pmy_mesh->dt;
//           bool limiter = true;

//           //***** first step of secant: define the root range *******//
//           // supplement
//           Real rho_d_supply = 0.0;
//           AthenaArray<Real> rho_d_supply_array;
//           rho_d_supply_array.NewAthenaArray(NDUSTFLUIDS-1);

//           for (int m = 0; m<= NDUSTFLUIDS-2; ++m){
//             rho_d_supply_array(m) = rho_d_array(m) - dfloor;
//             // still need a floor, which should be much smaller than density floor
//             rho_d_supply_array(m) = (rho_d_supply_array(m) < 1.e-21) ? 1.e-21 : rho_d_supply_array(m);
//             rho_d_supply += rho_d_supply_array(m);
//           }

//           Real rho_v_supply = rho_v - dfloor;
//           rho_v_supply = (rho_v_supply < 1.e-21) ? 1.e-21 : rho_v_supply;
//           // only condense to become st = 0 dusts.
//           for (int m = 0; m<= NDUSTFLUIDS-2; ++m){
//             if(sign > 0.){
//               rho_ratio_array(m) = rho_d_supply_array(m)/rho_d_supply;
//             }else{
//               rho_ratio_array(0) = 0.0;
//               rho_ratio_array(1) = 1.0;
//             }
//           }

//           Real drho_supply = (sign > 0.0 ? rho_d_supply : -rho_v_supply);
//           rho_g1 = rho_g + drho_supply;
//           rho_v1 = rho_v + drho_supply;
//           for (int m = 0; m<= NDUSTFLUIDS-2; ++m){
//             rho_d_array1(m) = rho_d_array(m) - drho_supply*rho_ratio_array(m);
//           }

//           // momentum conservation, update E_kg, E_kd_array:
//           if(mom_correct_Flag){
//             Get_vel_new_fromMC(gas_vel_array,rho_g,rho_g1,rho_d_array,rho_d_array1, \
//               drho_supply, rho_ratio_array, dust_vel_array, pdustfluids->istracer, gas_vel_array1, E_kg, E_kd_array);
//           }
          
//           // update rhoe
//           rhoe1 = Get_rhoe(rhoE_total,rho_g1,E_kg,rho_d_array1,E_kd_array);
//           phase_trans(rhoe1,rho_g1,rho_d_array1,rho_v1,drho);
//           fx1 = drho;
//           x1 = rho_v1;

//           // bisect prepare: store inital values
//           Real rho_left = rho_v;
//           Real rho_right = rho_v1;
//           Real rho_g0 = rho_g;
//           Real rho_v0 = rho_v;
//           gas_vel_array0 = gas_vel_array;
//           rho_d_array0 = rho_d_array;
          
//           //***** 2nd step of secant: start from [rhov,rhov1]  *******//
//           // (drho*sign) > 0.
//           rho_g = rho_g1;
//           rho_v = rho_v1;
//           rho_d_array = rho_d_array1;
//           gas_vel_array = gas_vel_array1;

//           if( (drho*sign) < 0.){
//             // secant
//             Real drho_adp;
//             Real f_err = 1.0;
//             int nite = 0;
//             bool bisect = false;

//             while(f_err>min_tol){
//               nite += 1;
//               if(nite > 100){
//                 std::cout << "nite > 100 in secant, break" <<std::endl;
//                 bisect = true;
//                 break;
//                 // quick_exit(1);
//               }
//               drho_adp = -fx1/(fx1-fx0)*(x1-x0);

//               if(std::isnan(drho_adp) or std::isinf(drho_adp)){
//                 std::cout << "drho_adp = " << drho_adp<< std::endl;
//                 bisect = true;
//                 break;
//               }

//               x2 = x1 + drho_adp;

//               rho_g1 = rho_g + drho_adp;
//               rho_v1 = rho_v + drho_adp;
//               // only condense to become st = 0 dusts (already implemented in [phase_trans]).
//               for (int m = 0; m<= NDUSTFLUIDS-2; ++m){
//                 rho_d_array1(m) = rho_d_array(m) - drho_adp*rho_ratio_array(m);
//               }

//               // momentum conservation, update E_kg, E_kd_array:
//               if(mom_correct_Flag){
//                 Get_vel_new_fromMC(gas_vel_array,rho_g,rho_g1,rho_d_array,rho_d_array1, \
//                   drho_adp, rho_ratio_array, dust_vel_array, pdustfluids->istracer, gas_vel_array1, E_kg, E_kd_array);
//               }
              
//               // update rhoe
//               rhoe1 = Get_rhoe(rhoE_total,rho_g1,E_kg,rho_d_array1,E_kd_array);
//               phase_trans(rhoe1,rho_g1,rho_d_array1, rho_v1, drho);
              
//               // update secant point:
//               fx2 = drho;
//               x0 = x1;
//               fx0 = fx1;
//               x1 = x2;
//               fx1 = fx2;

//               // update physical value:
//               rho_g = rho_g1;
//               rho_v = rho_v1;
//               rho_d_array = rho_d_array1;
//               gas_vel_array = gas_vel_array1;

//               // secant method root error fraction:
//               f_err = fabs(drho)/rho_v;
//             }
            
//             if(bisect == true){
//               // bisection
//               std::cout << "bisect in" <<std::endl;
//               Real rho_mid;
//               rho_g = rho_g0;
//               rho_v = rho_v0;
//               rho_d_array = rho_d_array0;
//               gas_vel_array = gas_vel_array0; 
//               int nite = 0;

//               while(f_err>min_tol){
//                 nite += 1;
//                 if(nite > 1000){
//                   std::cout << "nite > 1000 in bisect, break" <<std::endl;
//                   quick_exit(1);
//                 }
//                 rho_mid = (rho_left+rho_right)/2.0;
//                 drho_adp = rho_mid-rho_v;

//                 rho_g1 = rho_g + drho_adp;
//                 rho_v1 = rho_v + drho_adp;
//                 // only condense to become st = 0 dusts (already implemented in [phase_trans]).
//                 for (int m = 0; m<= NDUSTFLUIDS-2; ++m){
//                   rho_d_array1(m) = rho_d_array(m) - drho_adp*rho_ratio_array(m);
//                 }

//                 // momentum conservation, update E_kg, E_kd_array:
//                 Get_vel_new_fromMC(gas_vel_array,rho_g,rho_g1,rho_d_array,rho_d_array1, \
//                     drho_adp, rho_ratio_array, dust_vel_array, pdustfluids->istracer, gas_vel_array1, E_kg, E_kd_array);
                
//                 // update rhoe
//                 rhoe1 = Get_rhoe(rhoE_total,rho_g1,E_kg,rho_d_array1,E_kd_array);
//                 phase_trans(rhoe1,rho_g1,rho_d_array1,rho_v1, drho);
                
//                 if( (drho*sign) > 0.){
//                   rho_left = rho_v1;
//                 }else{
//                   rho_right = rho_v1;
//                 }

//                 rho_g = rho_g1;
//                 rho_v = rho_v1;
//                 rho_d_array = rho_d_array1;
//                 gas_vel_array = gas_vel_array1;

//                 // bisection root error fraction
//                 f_err = fabs(rho_right-rho_left)/rho_g;
//               }

//               std::cout << "nite="<< nite <<std::endl;
//             }
//           }

//           // calculate dfvdt
//           pdustfluids->dfv_dt(k,j,i) = (rho_v/rho_g-fv0)/dt;
//           // cons update
//           rho_I = rho_d_array(0);
//           rho_d2 = rho_d_array(1);

//           // apply floor value
//           // rho_g = (rho_g > dfloor ) ? rho_g : dfloor;
//           // rho_I = (rho_I > dfloor ) ? rho_I : dfloor;
//           // rho_d2 = (rho_d2 > dfloor ) ? rho_d2 : dfloor;
//           // rho_v = (rho_v > dfloor ) ? rho_v : dfloor;
//           // rho_v = (rho_v > initial_D2G[0]*rho_g/(1.0+initial_D2G[0])) ? initial_D2G[0]*rho_g/(1.0+initial_D2G[0]) : rho_v;
//           rho_d_array(0) = rho_I;
//           rho_d_array(1) = rho_d2;
          
//           // calculate pressure
//           fv = rho_v/rho_g;
//           rhoe = Get_rhoe(rhoE_total,rho_g,E_kg,rho_d_array,E_kd_array);
//           Real T1 = Get_T_rhoe(rhoe,rho_g,rho_d_array,fv);
//           Real mu1 = Get_mu(fv);
//           Real prs = rho_g*T1/(mu1*KELVIN);
//           phydro->Tem(k,j,i) = T1; // save temperature value

//           // get rhoe_g from rhoe
//           rhoe_g = rhoe;
//           for (int m = 0; m<= NDUSTFLUIDS-2; ++m){
//             rhoe_g -= rho_d_array(m)*Cd_water*T1;
//           }

//           // Prim update: gas density, pressure. ICE pebble density. fv
//           gas_rho = rho_g;
//           pres = prs;
//           d1_rho = rho_I;
//           d2_rho = rho_d2;
//           v1_rho = rho_v;
//           gas_vel1 = gas_vel_array(0);
//           gas_vel2 = gas_vel_array(1);
//           d2_vel1 = gas_vel1;
//           d2_vel2 = gas_vel2;
//           v1_vel1 = gas_vel1;
//           v1_vel2 = gas_vel2;

//           // Cons update: gas energy, momentum. dust1, 2 momentum.
//           gas_erg = rhoe_g + rho_g*E_kg;
//           gas_mom1 = rho_g*gas_vel1;
//           gas_mom2 = rho_g*gas_vel2;
          
//           d1_mom1 = rho_I*d1_vel1;
//           d1_mom2 = rho_I*d1_vel2;
//           d2_mom1 = rho_d2*d2_vel1;
//           d2_mom2 = rho_d2*d2_vel2;
//           v1_mom1 = rho_v*v1_vel1;
//           v1_mom2 = rho_v*v1_vel2;

//           // // verify mass/momentum/energy conservation:
//           // Real mass_total2 = rho_I + rho_d2 + rho_g;
//           // Real mom1_total2 = gas_mom1 + d1_mom1 + d2_mom1;
//           // Real mom2_total2 = gas_mom2 + d1_mom2 + d2_mom2;
//           // Real rhomu_d2 = Get_rhomu_d(rho_d_array); // chemical potential of dust. mu_g is set to be 0.
//           // Real erg_total2 = gas_erg + rhomu_d2 + Cd_water*(rho_I + rho_d2)*T1;
//           // for (int m = 0; m<= NDUSTFLUIDS-2; ++m){
//           //   erg_total2 += E_kd_array(m)*rho_d_array(m);
//           // }

//           // pdustfluids->cons_verify(0,k,j,i) = (mass_total2-mass_total1)/mass_total2;
//           // pdustfluids->cons_verify(1,k,j,i) = (mom1_total2-mom1_total1)/mom1_total2;
//           // pdustfluids->cons_verify(2,k,j,i) = (mom2_total2-mom2_total1)/mom2_total2;
//           // pdustfluids->cons_verify(3,k,j,i) = (erg_total2-rhoE_total)/erg_total2;

//           // // update viscosity
//           // Real T0 = KELVIN*mu_xy;
//           // phydro->hdif.nu(HydroDiffusion::DiffProcess::iso,k,j,i) = phydro->hdif.nu_iso*T1/T0;
          
//         }
//       }
//     }
//   }

//   // update stopping time
//   for (int k=ks; k<=ke; ++k) {
//     for (int j=js-NGHOST; j<=je+NGHOST; ++j) {
//       for (int i=is-NGHOST; i<=ie+NGHOST; ++i) {
//         // pre-calculation
//         Real rho_g = phydro->w(IDN,k ,j ,i);

//         // icy pebble
//         int dust_id = 0;
//         int rho_id  = 4*dust_id;
//         Real rho_I = pdustfluids->df_prim(rho_id, k, j, i);

//         // silicon pebble
//         dust_id = 2;
//         rho_id  = 4*dust_id;
//         Real rho_sil = pdustfluids->df_prim(rho_id, k, j, i);

//         // vapor
//         dust_id = NDUSTFLUIDS-1;
//         rho_id  = 4*dust_id;
//         Real rho_vap = pdustfluids->df_prim(rho_id, k, j, i);

//         Real fv = rho_vap/rho_g;

//         // convert to cgs:
//         rho_g *= UNIT_DENSITY;
//         rho_I *= UNIT_DENSITY;
//         rho_sil *= UNIT_DENSITY;
//         rho_vap *= UNIT_DENSITY;

//         // calc mean free path
//         Real mu1 = Get_mu(fv);
//         Real mu_cgs = mu1*CONST_amu;
//         Real sigma_mol = 2.e-15; // collisional cross-section of H2, in cgs.
//         Real l_mfp = mu_cgs/(std::sqrt(2)*rho_g*sigma_mol);

//         // decide initial size and mass of pebble
//         if(m_p0 < 0.0){
//           Real rho_p_inter0 = rho_ice_inter*rho_sil_inter/((1.0-f_ICE_inter0)*rho_ice_inter + f_ICE_inter0*rho_sil_inter);
//           Real t_stop0 = const_stopping_time[0]*UNIT_TIME;
//           Real vth0 = std::sqrt(8.0/PI)*UNIT_VELOCITY;
//           Real s_p0 = t_stop0*vth0*UNIT_DENSITY/(rho_p_inter0);
//           m_p0 = FOUR_3RD*PI*std::pow(s_p0,3.0)*rho_p_inter0;

//           std::cout << " Initial pebble properties: " <<std::endl;
//           std::cout << "s_p0=" << s_p0 <<std::endl;
//           std::cout << "m_p0=" << m_p0 <<std::endl;
//         }
        
//         // calc the size of pebble
//         Real rho_Np = rho_sil/(1.0-f_ICE_inter0)/m_p0;
//         Real f_ice = rho_I/(rho_sil+rho_I);
//         Real f_sil = rho_sil/(rho_sil+rho_I);
//         Real rho_p_inter = rho_ice_inter*rho_sil_inter/(f_ice*rho_sil_inter + f_sil*rho_ice_inter);
//         Real m_p = (rho_I + rho_sil)/(rho_Np);
//         Real s_p = m_p/(FOUR_3RD*PI*rho_p_inter);
//         s_p = std::pow(s_p,ONE_3RD);

//         Real t_stop, tem;
//         tem = phydro->Tem(k,j,i);
//         Real cs = std::sqrt(tem/(KELVIN*mu1));
//         Real vth = std::sqrt(8.0/PI)*cs*UNIT_VELOCITY;
//         if(s_p < (9.0/4.0*l_mfp)){
//           // Epstein regime
//           t_stop = rho_p_inter*s_p/(vth*rho_g);
//         }else{
//           // Stokes regime
//           t_stop = 4.0*rho_p_inter*SQR(s_p)/(9.0*vth*rho_g*l_mfp);
//         }

//         t_stop /= UNIT_TIME;

//         // update values:
//         pdustfluids->stopping_time_array(0,k,j,i) = t_stop; // icy pebble
//         pdustfluids->stopping_time_array(2,k,j,i) = t_stop; // sil pebble
//         pdustfluids->m_p_array(k,j,i) = m_p;
//       }
//     }
//   }

//   return;
// }


void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin){
  for (int k=ks-NGHOST; k<=ke+NGHOST; ++k) {
    for (int j=js-NGHOST; j<=je+NGHOST; ++j) {
      for (int i=is-NGHOST; i<=ie+NGHOST; ++i) {

        const Real &rho_g = phydro->w(IDN,k,j,i);
        const Real &press = phydro->w(IPR,k,j,i);

        Real Tem, gm; 
        if(PHASE_CHANGE){
          Tem = phydro->Tem(k,j,i);
          const Real &rho_v = pdustfluids->df_prim(4*(NDUSTFLUIDS - 1),k,j,i);
          Real E_kg = 0.5*(SQR(phydro->u(IM1,k,j,i)) + SQR(phydro->u(IM2,k,j,i)) + SQR(phydro->u(IM3,k,j,i)))/phydro->u(IDN,k,j,i);
          Real rhoe_g = phydro->u(IEN,k,j,i) - E_kg;
          Real fv = rho_v/rho_g;
          Real mu = Get_mu(fv);
          Real prs = rho_g*Tem/(mu*KELVIN);
          gm = prs/rhoe_g + 1.0;
        }else{
          Tem = press/rho_g*KELVIN*mu_xy;
          gm = gamma_gas;
        }
        // Real T = Get_T_rhoe_g(rhoe_g,rho_g,fv);
        
        // output
        user_out_var(0,k,j,i) = Tem;
        user_out_var(1,k,j,i) = gm;
        // user_out_var(2,k,j,i) = pdustfluids->stopping_time_array(0,k,j,i);
        // user_out_var(3,k,j,i) = pdustfluids->m_p_array(k,j,i);
        // user_out_var(4,k,j,i) = phydro->hdif.nu(HydroDiffusion::DiffProcess::iso,k,j,i);
        // user_out_var(5,k,j,i) = pdustfluids->cons_verify(0,k,j,i);
        // user_out_var(6,k,j,i) = pdustfluids->cons_verify(1,k,j,i);
        // user_out_var(7,k,j,i) = pdustfluids->cons_verify(2,k,j,i);
        // user_out_var(8,k,j,i) = pdustfluids->cons_verify(3,k,j,i);
        // user_out_var(4,k,j,i) = pdustfluids->nu_dustfluids_array(0,k,j,i);
      }
    }
  }
  return;
}


namespace {
//----------------------------------------------------------------------------------------
//!\f transform to cylindrical coordinate

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

void GetSphCoord(Coordinates *pco,Real &rad,Real &theta,Real &phi,int i,int j,int k) {
  rad = pco->x1v(i);
  theta = pco->x2v(j);
  phi = pco->x3v(k);

  return;
}

void Car2Sph(Real rad, Real theta,Real phi, Real vx, Real vy, Real vz, Real &v1, Real &v2, Real &v3) {
  // transfer vectors from cartesian to spherical 
  v1 = std::sin(theta)*std::cos(phi)*vx + std::sin(theta)*std::sin(phi)*vy + std::cos(theta)*vz;
  v2 = std::cos(theta)*std::cos(phi)*vx + std::cos(theta)*std::sin(phi)*vy - std::sin(theta)*vz;
  v3 = -std::sin(phi)*vx + std::cos(phi)*vy;

  return;
}

void Sph2Car(Real rad, Real theta,Real phi, Real &vx, Real &vy, Real &vz, Real v1, Real v2, Real v3) {
  // transfer vectors from spherical to cartesian
  vx = std::sin(theta)*std::cos(phi)*v1 + std::cos(theta)*std::cos(phi)*v2 - std::sin(phi)*v3;
  vy = std::sin(theta)*std::sin(phi)*v1 + std::cos(theta)*std::sin(phi)*v2 + std::cos(phi)*v3;
  vz = std::cos(theta)*v1 - std::sin(theta)*v2;

  return;
}


Real Get_T_rhoe_g(Real rhoe_g, Real rho_g, Real fv){
  Real fx = 1.0-fv; // He/H fraction
  Real e = rhoe_g/rho_g;
  Real T = e*KELVIN/(fx/mu_H2*0.71*2.5+fx/mu_He*0.29*1.5+fv/mu_z*3.0);
  return T;
}

Real Get_T_rhoe(Real rhoe, Real rho_g, AthenaArray<Real> rho_d_array, Real fv){
  Real fx = 1.0-fv; // He/H fraction
  Real bottom = rho_g/KELVIN*(fx/mu_H2*0.71*2.5+fx/mu_He*0.29*1.5+fv/mu_z*3.0);
  for (int m = 0; m<= NDUSTFLUIDS-2; ++m){
    bottom += rho_d_array(m)*Cd_water;
  }
  Real T = rhoe/bottom;

  return T;
}

Real Get_rhomu_d(AthenaArray<Real> rho_d_array){
  Real rhomu_d = 0.0;
  for (int m = 0; m<= NDUSTFLUIDS-2; ++m){
    rhomu_d += rho_d_array(m)*(-L_heat);
  }
  return rhomu_d;
}

Real Get_rhoe(Real rhoE_total, Real rho_g, Real E_kg, AthenaArray<Real> rho_d_array, AthenaArray<Real> E_kd_array){
  Real rhoe = 0.0;
  Real rhomu_d = Get_rhomu_d(rho_d_array);
  rhoe = rhoE_total- rho_g*E_kg- rhomu_d;

  for (int m = 0; m<= NDUSTFLUIDS-2; ++m){
    rhoe -= E_kd_array(m)*rho_d_array(m);
  }

  return rhoe;
}

Real Get_Z(Real rho_g, Real T){
  Real z,P_eq,rhoz,kB_mp;
  
  P_eq = P_eq0*exp(-T_a/T);
  kB_mp = 1.0/KELVIN;
  rhoz = P_eq * mu_z /(T*kB_mp);

  // z = rhoz/rho_g;
  // z = 1/(1+(1/(z+1e-8)));  // softening of z profile
  
  return rhoz;
}

Real Get_mu(Real fv){
  return 1./((1.-fv)/mu_xy + fv / mu_z);
}

void Get_E_kg(AthenaArray<Real> gas_vel_array, Real &E_kg){
  E_kg = 0.0;
  for (int n = 0; n<= NDIM-1; ++n){
    E_kg += 0.5*SQR(gas_vel_array(n));
  }
  return;
}

void Get_vel_new_fromMC(AthenaArray<Real> gas_vel_array,
    Real rho_g, Real rho_g1, AthenaArray<Real> rho_d_array, AthenaArray<Real> rho_d_array1, Real drho,
    AthenaArray<Real> rho_ratio_array, AthenaArray<Real> dust_vel_array, bool istracer[NDUSTFLUIDS],
    AthenaArray<Real> &gas_vel_array1, Real &E_kg, AthenaArray<Real> &E_kd_array){
  // momentum conservation, update E_kg, E_kd_array:
  for (int n = 0; n<= NDIM-1; ++n){
    gas_vel_array1(n) = (rho_g)*gas_vel_array(n);
  }
  
  Real denom_gas_vel_new = rho_g1;

  for (int m = 0; m<= NDUSTFLUIDS-2; ++m){
    if(istracer[m]){
      for (int n = 0; n<= NDIM-1; ++n){
        gas_vel_array1(n) += rho_d_array(m)*gas_vel_array(n);
      }
      denom_gas_vel_new += rho_d_array1(m);
    }else{
      for (int n = 0; n<= NDIM-1; ++n){
        gas_vel_array1(n) += drho*rho_ratio_array(m)*dust_vel_array(m,n);
      }
    }
  }

  for (int n = 0; n<= NDIM-1; ++n){
    gas_vel_array1(n) /= denom_gas_vel_new;
  }

  E_kg = 0.0;
  for (int n = 0; n<= NDIM-1; ++n){
    E_kg += 0.5*SQR(gas_vel_array1(n));
  }

  for (int m = 0; m<= NDUSTFLUIDS-2; ++m){
    if(istracer[m]){
      E_kd_array(m) = E_kg;
    }
  }

  // Real mom1 = gas_vel_array(0)*(rho_d_array(1)+rho_g) + rho_d_array(0)*dust_vel_array(0,0);
  // Real mom2 = gas_vel_array(1)*(rho_d_array(1)+rho_g) + rho_d_array(0)*dust_vel_array(0,1);

  // Real mom1_after = gas_vel_array1(0)*(rho_d_array1(1)+rho_g1) + rho_d_array1(0)*dust_vel_array(0,0);
  // Real mom2_after = gas_vel_array1(1)*(rho_d_array1(1)+rho_g1) + rho_d_array1(0)*dust_vel_array(0,1);

  // std::cout << "dif_mom1=" << mom1-mom1_after <<std::endl;
  // std::cout << "dif_mom2=" << mom2-mom2_after <<std::endl;

  return;
}


void phase_trans(Real rhoe, Real rho_g, AthenaArray<Real> rho_I, Real rho_v, Real &drho){
  Real T;
  Real fv, rhoz;

  fv = rho_v/rho_g;

  T = Get_T_rhoe(rhoe,rho_g,rho_I,fv);
  rhoz = Get_Z(rho_g,T);
  drho = rhoz-rho_v;

  if(std::isnan(drho)){
    std::cout << "drho = Nan" <<std::endl;
    std::cout << "T = " << T <<std::endl;
    std::cout << "rhoz = " << rhoz <<std::endl;
    std::cout << "rhoe = " << rhoe <<std::endl;
    std::cout << "rho_g = " << rho_g <<std::endl;
    std::cout << "rho_I = " << rho_I(0)+rho_I(1) <<std::endl;
    std::cout << "fv = " << fv <<std::endl;
    quick_exit(1);
  }

  return;
}

void MySource(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s) {

  // copy_velocity(pmb, time, dt, prim, prim_df, bcc, cons, cons_df);

  // PlanetaryGravity(pmb, time, dt, prim, prim_df, prim_s, bcc, cons, cons_df, cons_s);

  return;
}

void MyStoppingTime(MeshBlock *pmb, const Real time, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, AthenaArray<Real> &stopping_time) {
  
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
#pragma omp simd
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        Real rho_g = prim(IDN,k,j,i);

        // icy pebble
        int dust_id = 0;
        int rho_id  = 4*dust_id;
        Real rho_I = prim_df(rho_id, k, j, i);

        // silicon pebble
        dust_id = 2;
        rho_id  = 4*dust_id;
        Real rho_sil = prim_df(rho_id, k, j, i);

        // vapor
        dust_id = NDUSTFLUIDS-1;
        rho_id  = 4*dust_id;
        Real rho_vap = prim_df(rho_id, k, j, i);

        Real fv = rho_vap/rho_g;

        // convert to cgs:
        rho_g *= UNIT_DENSITY;
        rho_I *= UNIT_DENSITY;
        rho_sil *= UNIT_DENSITY;
        rho_vap *= UNIT_DENSITY;

        // calc mean free path
        Real mu1 = Get_mu(fv);
        Real mu_cgs = mu1*CONST_amu;
        Real sigma_mol = 2.e-15; // collisional cross-section of H2, in cgs.
        Real l_mfp = mu_cgs/(std::sqrt(2)*rho_g*sigma_mol);
        
        // calc the size of pebble
        Real rho_Np = rho_sil/(1.0-f_ICE_inter0)/m_p0;
        Real f_ice = rho_I/(rho_sil+rho_I);
        Real f_sil = rho_sil/(rho_sil+rho_I);
        Real rho_p_inter = rho_ice_inter*rho_sil_inter/(f_ice*rho_sil_inter + f_sil*rho_ice_inter);
        Real m_p = (rho_I + rho_sil)/(rho_Np);
        Real s_p = m_p/(FOUR_3RD*PI*rho_p_inter);
        s_p = std::pow(s_p,ONE_3RD);

        Real t_stop, tem;
        tem = pmb->phydro->Tem(k,j,i);
        Real cs = std::sqrt(tem/(KELVIN*mu1));
        Real vth = std::sqrt(8.0/PI)*cs*UNIT_VELOCITY;
        if(s_p < (9.0/4.0*l_mfp)){
          // Epstein regime
          t_stop = rho_p_inter*s_p/(vth*rho_g);
        }else{
          // Stokes regime
          t_stop = 4.0*rho_p_inter*SQR(s_p)/(9.0*vth*rho_g*l_mfp);
        }

        t_stop /= UNIT_TIME;

        // update values:
        stopping_time(0,k,j,i) = t_stop; // icy pebble
        stopping_time(2,k,j,i) = t_stop; // sil pebble
        pmb->pdustfluids->m_p_array(k,j,i) = m_p;
      }
    }
  }

  return;
}

void MyDustDiffusivity(DustFluids *pdf, MeshBlock *pmb,
      const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
      const AthenaArray<Real> &stopping_time, AthenaArray<Real> &nu_dust,
      AthenaArray<Real> &cs_dust, int is, int ie, int js, int je, int ks, int ke) {

    Real inv_eddy_time = 1.0; // 1/Omega
    for (int dust_id=0; dust_id<NDUSTFLUIDS; ++dust_id) {
      for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
#pragma omp simd
          for (int i=is; i<=ie; ++i) {

            Real &diffusivity = nu_dust(dust_id, k, j, i);
            diffusivity = const_nu_dust[dust_id];
            
            // if pebbles, diffusivity change with st.
            if(!pmb->pdustfluids->istracer[dust_id]){
              Real st = stopping_time(dust_id, k, j, i)*inv_eddy_time;
              diffusivity = diffusivity/(1.0+SQR(st));
            }
            
            Real &soundspeed  = cs_dust(dust_id, k, j, i);
            soundspeed        = std::sqrt(diffusivity*inv_eddy_time);
          }
        }
      }
    }

  return;
}

void copy_velocity(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s) {
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    // Real x3 = pmb->pcoord->x3v(k);
    for (int j=pmb->js; j<=pmb->je; ++j) {
      // Real x2 = pmb->pcoord->x2v(j);
#pragma omp simd
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        Real &gas_den  = cons(IDN, k, j, i);
        Real &gas_mom1 = cons(IM1, k, j, i);
        Real &gas_mom2 = cons(IM2, k, j, i);
        // Real &gas_mom3 = cons(IM3, k, j, i);
        if (NDUSTFLUIDS > 0) {

          const Real &gas_vel1 = prim(IVX, k, j, i);
          const Real &gas_vel2 = prim(IVY, k, j, i);
						for (int n=0; n<NDUSTFLUIDS; ++n) {
							int dust_id = n;
							int rho_id  = 4*dust_id;
							int v1_id   = rho_id + 1;
							int v2_id   = rho_id + 2;
							int v3_id   = rho_id + 3;

							const Real &dust_rho  = prim_df(rho_id, k, j, i);
							// Real &dust_vel1 = prim_df(v1_id,  k, j, i);
							// Real &dust_vel2 = prim_df(v2_id,  k, j, i);
							// Real &dust_vel3 = prim_df(v3_id,  k, j, i);

							Real &dust_den  = cons_df(rho_id, k, j, i);
							Real &dust_mom1 = cons_df(v1_id,  k, j, i);
							Real &dust_mom2 = cons_df(v2_id,  k, j, i);
							// Real &dust_mom3 = cons_df(v3_id,  k, j, i);
              if(pmb->pdustfluids->istracer[dust_id]){
                // copy gas velocity to tracer
                dust_mom1 = dust_den/gas_den*gas_mom1;
                dust_mom2 = dust_den/gas_den*gas_mom2;
                // dust_vel1 = gas_vel1;
                // dust_vel2 = gas_vel2;
              }
            }
          }
      }
    }
  }
  return;
}


// Add planet
void PlanetaryGravity(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s) {

  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
#pragma omp simd
      for (int i=pmb->is; i<=pmb->ie; ++i) {

        Real rad, theta, phi;
        Real x, y, z;
        Real vx, vy, vz;
        // assign spherical coordinate
        GetSphCoord(pmb->pcoord,rad,theta,phi,i,j,k);
        x = rad* std::sin(theta)* std::cos(phi);
        y = rad* std::sin(theta)* std::sin(phi);
        z = rad* std::cos(theta);
        // Bethune & Rafikov 2019
        // z = (z > 4.0) ? 4.0 : z;
        z = 0.0;

        const Real &gas_rho  = prim(IDN, k, j, i);
        const Real &gas_vel1 = prim(IVX, k, j, i);
        const Real &gas_vel2 = prim(IVY, k, j, i);
        const Real &gas_vel3 = prim(IVZ, k, j, i);

        // get cartesian velocities
        Sph2Car(rad, theta, phi,vx,vy,vz, gas_vel1,gas_vel2,gas_vel3);
        // coriolis and tidal force
        Real Fcorx_gas = 2.0*vy;
        Real Fcory_gas = -2.0*vx;
        Real Ftidx = 3.0*x;
        Real Ftidz = -z;

        Real acc_x_gas = Fcorx_gas + Ftidx;
        Real acc_y_gas = Fcory_gas;
        Real acc_z_gas = Ftidz;

        Real acc_r_gas, acc_theta_gas, acc_phi_gas;
        Car2Sph(rad, theta, phi, acc_x_gas, acc_y_gas, acc_z_gas, acc_r_gas, acc_theta_gas, acc_phi_gas);
        
        // Fung+ 2019, Zhu+ 2021
        Real fsoft = SQR(rad-x1min)/(SQR(rad-x1min)+SQR(rsoft));
        // Real Fplanet = -mplanet/(rad*rad+ rsoft*rsoft)*(1.0-std::exp(-0.5*SQR(time/tsoft)));
        Real Fplanet = -mplanet/(rad*rad)*fsoft*(1.0-std::exp(-0.5*SQR(time/tsoft)));
        acc_r_gas += Fplanet; // planet gravity
        
        // momentum change
        Real &gas_den  = cons(IDN, k, j, i);
        Real &gas_mom1 = cons(IM1, k, j, i);
        Real &gas_mom2 = cons(IM2, k, j, i);
        Real &gas_mom3 = cons(IM3, k, j, i);

        Real delta_mom1 = dt*gas_rho*acc_r_gas;
        Real delta_mom2 = dt*gas_rho*acc_theta_gas;
        Real delta_mom3 = dt*gas_rho*acc_phi_gas;

        gas_mom1 += delta_mom1;
        gas_mom2 += delta_mom2;
        gas_mom3 += delta_mom3;

        // erg change
        Real fl1 = pmb->phydro->flux[0](k,j,i);
        Real fr1 = pmb->phydro->flux[0](k,j,i+1);
        Real fl2 = pmb->phydro->flux[1](k,j,i);
        Real fr2 = pmb->phydro->flux[1](k,j+1,i);
        Real fl3 = pmb->phydro->flux[2](k,j,i);
        Real fr3 = pmb->phydro->flux[2](k+1,j,i);

        Real gwork_x = Ftidx;
        Real gwork_y = 0.0;
        Real gwork_z = Ftidz;
        Real gwork_r, gwork_theta, gwork_phi;
        
        Car2Sph(rad,theta,phi, gwork_x, gwork_y, gwork_z, gwork_r, gwork_theta, gwork_phi);
        gwork_r += Fplanet; // planet gravity
        
        if (NON_BAROTROPIC_EOS) {
          Real &gas_erg  = cons(IEN, k, j, i);
          gas_erg += dt*0.5*(fl1+fr1)*gwork_r + dt*0.5*(fl2+fr2)*gwork_theta + dt*0.5*(fl3+fr3)*gwork_phi;
          // gas_erg       += (gwork_r*gas_vel1 + gwork_phi*gas_vel2)*dt;
          // gas_erg       += (delta_mom1*gas_vel1 + delta_mom2*gas_vel2 + delta_mom3*gas_vel3);
        }

        if (NDUSTFLUIDS > 0) {
						for (int n=0; n<NDUSTFLUIDS; ++n) {
							int dust_id = n;
							int rho_id  = 4*dust_id;
							int v1_id   = rho_id + 1;
							int v2_id   = rho_id + 2;
							int v3_id   = rho_id + 3;

							const Real &dust_rho  = prim_df(rho_id, k, j, i);
							const Real &dust_vel1 = prim_df(v1_id,  k, j, i);
							const Real &dust_vel2 = prim_df(v2_id,  k, j, i);
							const Real &dust_vel3 = prim_df(v3_id,  k, j, i);

							Real &dust_den  = cons_df(rho_id, k, j, i);
							Real &dust_mom1 = cons_df(v1_id,  k, j, i);
							Real &dust_mom2 = cons_df(v2_id,  k, j, i);
							Real &dust_mom3 = cons_df(v3_id,  k, j, i);

              if(pmb->pdustfluids->istracer[dust_id]){
                // copy gas velocity to tracer
                dust_mom1 = dust_den/gas_den*gas_mom1;
                dust_mom2 = dust_den/gas_den*gas_mom2;
                dust_mom3 = dust_den/gas_den*gas_mom3;
              }else{
                Real dust_vx, dust_vy, dust_vz;
                Sph2Car(rad, theta, phi, dust_vx,dust_vy,dust_vz, dust_vel1,dust_vel2,dust_vel3);
                // coriolis and tidal force
                Real Fcorx_dust = 2.0*dust_vy;
                Real Fcory_dust = -2.0*dust_vx;
                Real acc_x_dust = Fcorx_dust + Ftidx;
                Real acc_y_dust = Fcory_dust;
                Real acc_z_dust = Ftidz;

                Real acc_r_dust, acc_theta_dust, acc_phi_dust;
                Car2Sph(rad, theta, phi, acc_x_dust, acc_y_dust, acc_z_dust, acc_r_dust, acc_theta_dust, acc_phi_dust);
                acc_r_dust += Fplanet; // planet gravity

                Real delta_mom_dust_1 = dt*dust_rho*acc_r_dust;
                Real delta_mom_dust_2 = dt*dust_rho*acc_theta_dust;
                Real delta_mom_dust_3 = dt*dust_rho*acc_phi_dust;

                dust_mom1 += delta_mom_dust_1;
                dust_mom2 += delta_mom_dust_2;
                dust_mom3 += delta_mom_dust_3;
              }
            }	
			  }
      }
    }
  }
  return;
}

Real MyMeshSpacingX1(Real x, RegionSize rs){
  // log-spaced mesh
  Real tmp = x*(std::log10(rs.x1max)-std::log10(rs.x1min)) + std::log10(rs.x1min);
  
  return std::pow(10.0,tmp);
}

}

void DiskInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad(0.0), theta(0.0), phi(0.0);
  Real z;

  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=1; i<=ngh; ++i) {
        // assign spherical coordinate
        GetSphCoord(pco,rad,theta,phi,il-i,j,k);
        // Real f_time = 1.0-std::exp(-0.5*SQR(time/injection_Tsoft));
        prim(IDN,k,j,il-i) = prim(IDN,k,j,il+i-1);
        prim(IVX,k,j,il-i) = -prim(IVX,k,j,il+i-1);
        // prim(IVX,k,j,il-i) = (-2.0*f_time + 1.0)* prim(IVX,k,j,il+i-1);
        // prim(IVY,k,j,il-i) = prim(IVY,k,j,il+i-1);
        prim(IVY,k,j,il-i) = 0.0;
        prim(IVZ,k,j,il-i) = -3.0/4.0*rad*std::sin(theta);

        if (NON_BAROTROPIC_EOS){
          prim(IPR,k,j,il-i) = prim(IPR,k,j,il+i-1);
          // pmb->phydro->g_gamma(0,k,j,il-i) = pmb->phydro->g_gamma(0,k,j,il+i-1);
          // prim(IPR,k,j,il-i) = pow(prim(IDN,k,j,il-i),gamma_gas);
        }

        if (NDUSTFLUIDS > 0) {
          for (int n=0; n<NDUSTFLUIDS; ++n) {
            int dust_id = n;
            int rho_id  = 4*dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + 2;
            // int v3_id   = rho_id + 3;

            if(dust_id >= NDUSTFLUIDS-1){
              // reflective for vapor tracer
              prim_df(rho_id,k,j,il-i) = prim_df(rho_id,k,j,il+i-1);
              prim_df(v1_id,k,j,il-i) = -prim_df(v1_id,k,j,il+i-1);
              // prim_df(v1_id,k,j,il-i) = (-2.0*f_time + 1.0)* prim_df(v1_id,k,j,il+i-1);
              prim_df(v2_id,k,j,il-i) = prim_df(v2_id,k,j,il+i-1);
            }else{
              // free outflow for pebble and dust
              prim_df(rho_id,k,j,il-i) = prim_df(rho_id,k,j,il+i-1);
              prim_df(v1_id,k,j,il-i) = prim_df(v1_id,k,j,il+i-1);
              prim_df(v2_id,k,j,il-i) = prim_df(v2_id,k,j,il+i-1);
            }
          }
        }
      }
    }
  }
}



void DiskOuterX1(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad(0.0), theta(0.0), phi(0.0);
  Real v1, v2, v3;
  Real x, y, z;
  Real vx, vy, vz;
  Real rhov;
  if (PHASE_CHANGE) {
    rhov = initial_D2G[NDUSTFLUIDS-1];
  } else {
    rhov = 0.;
  }

  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=1; i<=ngh; ++i) {
        if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
          // assign spherical coordinate
          GetSphCoord(pco,rad,theta,phi,iu+i,j,k);
          x = rad* std::sin(theta)* std::cos(phi);
          y = rad* std::sin(theta)* std::sin(phi);
          z = rad* std::cos(theta);
        } else if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
          x = pco->x1v(iu+i);
          y = pco->x1v(j);
          z = pco->x1v(k);
        }
        // Bethune & Rafikov 2019
        // z = (z > 4.0) ? 4.0 : z;
        z = 0.0;
        // assign velocity in cartesian coordinate
        vx = 0.;
        vy = -1.5*x;
        vz = 0.;

        if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
          // transfer from cartesian to spherical
          Car2Sph(rad, theta, phi, vx, vy, vz, v1, v2, v3);
        } else if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
          v1 = vx;
          v2 = vy;
          v3 = vz;
        }
        //
        prim(IDN,k,j,iu+i) = (1.0 + rhov)* std::exp(-0.5* SQR(z)); // stratified density.
        prim(IVX,k,j,iu+i) = v1;
        prim(IVY,k,j,iu+i) = v2;
        prim(IVZ,k,j,iu+i) = v3;

        if (NON_BAROTROPIC_EOS)
          prim(IPR,k,j,iu+i) = (1.0 + rhov*mu_xy/mu_z)* std::exp(-0.5* SQR(z));

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
            
            Real den_dust;
            // inject pebbles for inflow. While free outflow 
	          bool inject_bd = (v1<0.0);

            if(inject_bd){
              den_dust = boundary_D2G[n]*prim(IDN,k,j,iu+i);
            }else{
              den_dust = prim_df(rho_id, k, j, iu-i);
            }
            
            dust_rho_ghost = den_dust;
            dust_vel1_ghost = v1;
            dust_vel2_ghost = v2;
            dust_vel3_ghost = v3;
					}
				}
      }
    }
  }
}

void DiskInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                  AthenaArray<Real> &prim_df, FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad(0.0), theta(0.0), phi(0.0);
  Real v1, v2, v3;
  Real x, y, z;
  Real vx, vy, vz;
  Real rhov;
  if (PHASE_CHANGE) {
    rhov = initial_D2G[NDUSTFLUIDS-1];
  } else {
    rhov = 0.;
  }
  for (int k=kl; k<=ku; ++k) {
    for (int j=1; j<=ngh; ++j) {
      for (int i=il; i<=iu; ++i) {
        if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
          // assign spherical coordinate
          GetSphCoord(pco,rad,theta,phi,i,jl-j,k);
          x = rad* std::sin(theta)* std::cos(phi);
          y = rad* std::sin(theta)* std::sin(phi);
          z = rad* std::cos(theta);
        } else if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
          x = pco->x1v(i);
          y = pco->x1v(jl-j);
          z = pco->x1v(k);
        }
        // Bethune & Rafikov 2019
        // z = (z > 4.0) ? 4.0 : z;
        z = 0.0;
        // assign velocity in cartesian coordinate
        vx = 0.;
        vy = -1.5*x;
        vz = 0.;

        if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
          // transfer from cartesian to spherical
          Car2Sph(rad, theta, phi, vx, vy, vz, v1, v2, v3);
        } else if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
          v1 = vx;
          v2 = vy;
          v3 = vz;
        }
        //
        prim(IDN,k,jl-j,i) = (1.0 + rhov)* std::exp(-0.5* SQR(z)); // stratified density.
        prim(IVX,k,jl-j,i) = v1;
        prim(IVY,k,jl-j,i) = v2;
        prim(IVZ,k,jl-j,i) = v3;

        if (NON_BAROTROPIC_EOS)
          prim(IPR,k,jl-j,i) = (1.0 + rhov*mu_xy/mu_z)* std::exp(-0.5* SQR(z));
      }
    }
  }

}

void DiskOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad(0.0), theta(0.0), phi(0.0);
  Real v1, v2, v3;
  Real x, y, z;
  Real vx, vy, vz;
  Real rhov;
  if (PHASE_CHANGE) {
    rhov = initial_D2G[NDUSTFLUIDS-1];
  } else {
    rhov = 0.;
  }
  for (int k=kl; k<=ku; ++k) {
    for (int j=1; j<=ngh; ++j) {
      for (int i=il; i<=iu; ++i) {
        if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
          // assign spherical coordinate
          GetSphCoord(pco,rad,theta,phi,i,ju+j,k);
          x = rad* std::sin(theta)* std::cos(phi);
          y = rad* std::sin(theta)* std::sin(phi);
          z = rad* std::cos(theta);
        } else if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
          x = pco->x1v(i);
          y = pco->x1v(ju+j);
          z = pco->x1v(k);
        }
        // Bethune & Rafikov 2019
        // z = (z > 4.0) ? 4.0 : z;
        z = 0.0;
        // assign velocity in cartesian coordinate
        vx = 0.;
        vy = -1.5*x;
        vz = 0.;

        if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
          // transfer from cartesian to spherical
          Car2Sph(rad, theta, phi, vx, vy, vz, v1, v2, v3);
        } else if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
          v1 = vx;
          v2 = vy;
          v3 = vz;
        }
        //
        prim(IDN,k,ju+j,i) = (1.0 + rhov)* std::exp(-0.5* SQR(z)); // stratified density.
        prim(IVX,k,ju+j,i) = v1;
        prim(IVY,k,ju+j,i) = v2;
        prim(IVZ,k,ju+j,i) = v3;

        if (NON_BAROTROPIC_EOS)
          prim(IPR,k,ju+j,i) = (1.0 + rhov*mu_xy/mu_z)* std::exp(-0.5* SQR(z));
      }
    }
  }

}