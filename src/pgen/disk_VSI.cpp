//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file disk.cpp
//! \brief Initializes stratified Keplerian accretion disk in both cylindrical and
//! spherical polar coordinates.  Initial conditions are in vertical hydrostatic eqm.

// C headers

// C++ headers
#include <algorithm>  // min
#include <cfloat>     // FLT_MIN
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
#include "../utils/utils.hpp" // ran2()

#if (!NON_BAROTROPIC_EOS)
#error "This problem generator requires NON_BAROTROPIC_EOS!"
#endif

namespace {

void GetCylCoordInSpherical(Coordinates *pco, Real &rad, Real &phi, Real &z,
                            int i, int j, int k);
Real PoverRho(const Real rad, const Real phi, const Real z);
Real DenProfileCyl_gas(const Real rad, const Real phi, const Real z);
Real VelProfileCyl_gas(const Real rad, const Real phi, const Real z);
Real DenProfileCyl_dust(const Real rad, const Real phi, const Real z,
                        const Real den_ratio, const Real H_ratio);
Real VelProfileCyl_dust(const Real rad, const Real phi, const Real z);

Real initial_D2G[NDUSTFLUIDS], Stokes_number[NDUSTFLUIDS], Hratio[NDUSTFLUIDS],
weight_dust[NDUSTFLUIDS];

Real gm0, r0, rho0, pvalue, cs2_0, qvalue, beta, nu_alpha, dfloor, dffloor, rho_floor, Omega0,
amp, time_drag, time_refine, refine_theta_upper, refine_theta_lower, refine_r_min, refine_r_max,
x1min, x1max, x2min, x2max, damping_rate, radius_inner_damping, radius_outer_damping,
theta_upper_damping, theta_lower_damping, upper_altitude_damping, lower_altitude_damping,
inner_ratio_region, outer_ratio_region, inner_width_damping, outer_width_damping,
prev_time, curr_time, next_time, edt;

bool Inner_Gas_Damping_Flag, Outer_Gas_Damping_Flag, Theta_Gas_Damping_Flag,
Inner_Dust_Damping_Flag, Outer_Dust_Damping_Flag,
Isothermal_Flag, RadiativeConduction_Flag;

int n0, nvar, dowrite, file_number, out_level;
std::string basename;

// User Sources
void MySource(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s);
void DensityFloor(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s);
void ThermalRelaxation(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
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
// User-defined history
Real DustFluidsRatioMaximum(MeshBlock *pmb, int iout);
// Adaptive Mesh Refinement Condition
int RefinementCondition(MeshBlock *pmb);
void Swap4Bytes(void *vdat);
void Vr_outflow(const Real r_active, const Real r_ghost, const Real rho_active,
                    const Real rho_ghost, const Real vr_active, Real &vr_ghost);
void Vr_Mdot(const Real r_active, const Real r_ghost, const Real rho_active,
                    const Real rho_ghost, const Real vr_active, Real &vr_ghost);
}
// namespace

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

void AccumulateData(MeshBlock *pmb);
void DoOutput(MeshBlock *pmb, int dlevel);
int IsBigEndianOutput();

void LocalIsothermalEOS(MeshBlock *pmb, int il, int iu, int jl,
    int ju, int kl, int ku, AthenaArray<Real> &prim, AthenaArray<Real> &cons);
void InnerWaveDampingGas(MeshBlock *pmb, const Real time, const Real dt, int il, int iu,
    int jl, int ju, int kl, int ku, AthenaArray<Real> &prim, AthenaArray<Real> &cons);
void OuterWaveDampingGas(MeshBlock *pmb, const Real time, const Real dt, int il, int iu,
    int jl, int ju, int kl, int ku, AthenaArray<Real> &prim, AthenaArray<Real> &cons);
void UpperWaveDampingGas(MeshBlock *pmb, const Real time, const Real dt, int il, int iu,
    int jl, int ju, int kl, int ku, AthenaArray<Real> &prim, AthenaArray<Real> &cons);
void LowerWaveDampingGas(MeshBlock *pmb, const Real time, const Real dt, int il, int iu,
    int jl, int ju, int kl, int ku, AthenaArray<Real> &prim, AthenaArray<Real> &cons);
void InnerWaveDampingDust(MeshBlock *pmb, const Real time, const Real dt, int il, int iu,
    int jl, int ju, int kl, int ku, AthenaArray<Real> &prim_df, AthenaArray<Real> &cons_df);
void OuterWaveDampingDust(MeshBlock *pmb, const Real time, const Real dt, int il, int iu,
    int jl, int ju, int kl, int ku, AthenaArray<Real> &prim_df, AthenaArray<Real> &cons_df);
void FixedDust(MeshBlock *pmb, int il, int iu, int jl, int ju, int kl, int ku,
      const AthenaArray<Real> &w, AthenaArray<Real> &prim_df,
      const AthenaArray<Real> &u, AthenaArray<Real> &cons_df);

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//! \brief Function to initialize problem-specific data in mesh class.  Can also be used
//! to initialize variables which are global to (and therefore can be passed to) other
//! functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {

  if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") != 0) {
    std::stringstream msg;
    msg << "This problem file must be setup in the spherical_polar coordinate!" << std::endl;
    ATHENA_ERROR(msg);
  }

  std::string basename;
  basename = pin->GetString("job", "problem_id");

  // Get parameters for gravitatonal potential of central point mass
  gm0 = pin->GetOrAddReal("problem", "GM", 0.0);
  r0  = pin->GetOrAddReal("problem", "r0", 1.0);
  amp = pin->GetOrAddReal("problem", "amp", 0.01);

  Isothermal_Flag          = pin->GetBoolean("problem", "Isothermal_Flag");
  RadiativeConduction_Flag = pin->GetOrAddBoolean("problem", "RadiativeConduction_Flag", false);
  Inner_Gas_Damping_Flag   = pin->GetOrAddBoolean("problem", "Inner_Gas_Damping_Flag",   false);
  Outer_Gas_Damping_Flag   = pin->GetOrAddBoolean("problem", "Outer_Gas_Damping_Flag",   false);
  Theta_Gas_Damping_Flag   = pin->GetOrAddBoolean("problem", "Theta_Gas_Damping_Flag",   false);
  Inner_Dust_Damping_Flag  = pin->GetOrAddBoolean("problem", "Inner_Dust_Damping_Flag",  false);
  Outer_Dust_Damping_Flag  = pin->GetOrAddBoolean("problem", "Outer_Dust_Damping_Flag",  false);

  // Get parameters for initial density and velocity
  rho0   = pin->GetReal("problem", "rho0");
  pvalue = pin->GetOrAddReal("problem", "pvalue", -1.0);

  // Get parameters of initial pressure and cooling parameters
  cs2_0  = pin->GetOrAddReal("problem", "cs2_0",  0.0025);
  qvalue = pin->GetOrAddReal("problem", "qvalue", -0.5);
  beta   = pin->GetOrAddReal("problem", "beta",   0.0);
  if (beta < 0.0) beta = 0.0;

  Real float_min = std::numeric_limits<float>::min();
  dfloor    = pin->GetOrAddReal("hydro", "dfloor",  (1024*(float_min)));
  nu_alpha  = pin->GetOrAddReal("problem", "nu_alpha", 0.0);
  dffloor   = pin->GetOrAddReal("dust", "dffloor", (1024*(float_min)));
  Omega0    = pin->GetOrAddReal("orbital_advection", "Omega0", 0.0);
  rho_floor = pin->GetOrAddReal("dust", "rho_floor", 1e-10);

  // Dust to gas ratio && dust stopping time
  if (NDUSTFLUIDS > 0) {
    for (int n=0; n<NDUSTFLUIDS; n++) {
      initial_D2G[n]   = pin->GetReal("dust", "initial_D2G_" + std::to_string(n+1));
      Stokes_number[n] = pin->GetReal("dust", "Stokes_number_" + std::to_string(n+1));
      Hratio[n]        = pin->GetReal("dust", "Hratio_" + std::to_string(n+1));
      weight_dust[n]   = 2.0/(Stokes_number[n] + SQR(1.0+initial_D2G[n])/Stokes_number[n]);
    }
  }

  time_drag   = pin->GetOrAddReal("dust", "time_drag", 0.0);
  time_refine = pin->GetOrAddReal("problem", "time_refine", time_drag);

  // The parameters of damping zones
  x1min = pin->GetReal("mesh", "x1min");
  x1max = pin->GetReal("mesh", "x1max");

  x2min = pin->GetReal("mesh", "x2min");
  x2max = pin->GetReal("mesh", "x2max");

  refine_theta_lower = pin->GetOrAddReal("problem", "refine_theta_lower", 0.5*PI - std::sqrt(cs2_0));
  refine_theta_upper = pin->GetOrAddReal("problem", "refine_theta_upper", 0.5*PI + std::sqrt(cs2_0));

  //ratio of the orbital periods between the edge of the wave-killing zone and the corresponding edge of the mesh
  inner_ratio_region = pin->GetOrAddReal("problem", "inner_dampingregion_ratio", 1.2);
  outer_ratio_region = pin->GetOrAddReal("problem", "outer_dampingregion_ratio", 1.2);

  radius_inner_damping = x1min*pow(inner_ratio_region, TWO_3RD);
  radius_outer_damping = x1max*pow(outer_ratio_region, -TWO_3RD);

  inner_width_damping = radius_inner_damping - x1min;
  outer_width_damping = x1max - radius_outer_damping;

  upper_altitude_damping = 1.5*std::sqrt(cs2_0);
  lower_altitude_damping = 1.5*std::sqrt(cs2_0);

  theta_upper_damping = x2min + upper_altitude_damping;
  theta_lower_damping = x2max - upper_altitude_damping;

  refine_r_min = pin->GetOrAddReal("problem", "refine_r_min", radius_inner_damping);
  refine_r_max = pin->GetOrAddReal("problem", "refine_r_max", radius_outer_damping);

  // The normalized wave damping timescale, in unit of dynamical timescale.
  damping_rate = pin->GetOrAddReal("problem", "damping_rate", 1.0);

  // initialize user output
  n0          = 4;
  nvar        = 17 + 2*(NDUSTFLUIDS);
  dowrite     = 0;
  file_number = pin->GetOrAddInteger("problem", "file_number", 0);
  out_level   = pin->GetOrAddInteger("problem", "out_level",   0);
  edt         = pin->GetOrAddReal("problem", "Edt", 100);
  prev_time   = time;
  curr_time   = time;
  next_time   = pin->GetOrAddReal("problem", "next_time", time+edt);

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

    //// Enroll user-defined dust diffusivity
    //EnrollDustDiffusivity(MyDustDiffusivity);

    // Enroll user-defined history file
    AllocateUserHistoryOutput(1);
    EnrollUserHistoryOutput(0, DustFluidsRatioMaximum, "RatioMax", UserHistoryOperation::max);
  }

  // Enroll Thermal Relaxiation and Dust Density Floor
  EnrollUserExplicitSourceFunction(MySource);

  // Enroll user-defined thermal conduction
  if ((!Isothermal_Flag) && (beta > 0.0) && (RadiativeConduction_Flag))
    EnrollConductionCoefficient(RadiativeCondution);

  // Enroll user-defined AMR criterion
  if (adaptive)
    EnrollUserRefinementCondition(RefinementCondition);

  return;
}


void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {

  //AllocateUserOutputVariables2D(1);
  //SetUserOutputVariableName_2D(0, "gas_dens_2D");
  //AllocateUserOutputVariables1D(1);
  //SetUserOutputVariableName_1D(0, "gas_dens_1D");
  AllocateRealUserMeshBlockDataField(n0+nvar);
// List of User field:
//     0: temperature profile (1d array)
//     1: initial density profile (1d array)
//     2: uniform output grid (x1)
//     3: uniform output grid (x2)
//  4-35: azimuthally-averaged, and time-averaged diagnostics (2d arrays)
//  4- 5: rho, pres
//  6- 8: rhov1-rhov3
//  9-11: rhov1sqr - rhov3sqr
// 12-14: rhov1v2, rhov2v3, rhov3v1
// 15-17: kinflx1-kinflx3
// 18-20: enpflx1-enpflx3
// 21-23: b1-b3
// 24-26: b1sqr - b3sqr
// 27-29: b1b2,b2b3,b3b1
// 30-32: poyflx1-poyflx3
// 33-35: ey1, ey2, eyn

  ruser_meshblock_data[0].NewAthenaArray(block_size.nx2+2*NGHOST);
  ruser_meshblock_data[1].NewAthenaArray(block_size.nx2+2*NGHOST);

  int current_level = loc.level - pmy_mesh->root_level;
  int dlevel = current_level - out_level;
  int mynx1, mynx2, maxnx1, maxnx2, mynx1t, mynx2t;

  if (dlevel > 0) { // need downsampling
    mynx1 = block_size.nx1>>dlevel;
    mynx2 = block_size.nx2>>dlevel;
  } else { // just need to copy
    mynx1 = block_size.nx1<<(-dlevel);
    mynx2 = block_size.nx2<<(-dlevel);
  }
  maxnx1 = block_size.nx1<<out_level;
  maxnx2 = block_size.nx2<<out_level;
  ruser_meshblock_data[2].NewAthenaArray(maxnx1+1);
  ruser_meshblock_data[3].NewAthenaArray(maxnx2+1);

  mynx1t = pmy_mesh->mesh_size.nx1<<out_level;
  mynx2t = pmy_mesh->mesh_size.nx2<<out_level;

  for (int n=n0; n<n0+nvar; ++n)
    ruser_meshblock_data[n].NewAthenaArray(block_size.nx2+1, block_size.nx1+1);


// set initial density profile
  //int dlev = pmy_mesh->root_level+finest_lev-loc.level;
  //int ifac = 1<<dlev;

  //for (int j=js;j<=je;++j) {
    //int j1=ifac*((j-js)+loc.lx2*block_size.nx2);
    //int j2=j1+ifac-1;
    //Real val=1.0;
    //for (int t=j1; t<=j2; ++t)
      //val *= pmy_mesh->ruser_mesh_data[1](t);
    //for (int t=0; t<dlev; ++t)
      //val = sqrt(val);
    //ruser_meshblock_data[1](j) = val;
  //}

// set the output grid
  for (int i=0;i<=mynx1;++i) {
    Real rx = (Real)(i+loc.lx1*mynx1)/(Real)(mynx1t);
    ruser_meshblock_data[2](i) = pmy_mesh->MeshGenerator_[X1DIR](rx,pmy_mesh->mesh_size);
  }
  for (int j=0;j<=mynx2;++j) {
    Real rx = (Real)(j+loc.lx2*mynx2)/(Real)(mynx2t);
    ruser_meshblock_data[3](j) = pmy_mesh->MeshGenerator_[X2DIR](rx,pmy_mesh->mesh_size);
  }

// clear the rest of the data array
  for (int n=n0; n<n0+nvar; n++)
    for (int j=0;j<block_size.nx2;j++)
      for (int i=0;i<block_size.nx1;i++)
        ruser_meshblock_data[n](j, i)=0.0;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Initializes Keplerian accretion disk.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {

  std::int64_t iseed = -1 - gid;
  Real gamma_gas = peos->GetGamma();
  Real igm1      = 1.0/(gamma_gas - 1.0);
  OrbitalVelocityFunc &vK = porb->OrbitalVelocity;

  Real orb_defined;
  (porb->orbital_advection_defined) ? orb_defined = 1.0 : orb_defined = 0.0;

  //  Initialize density and momenta
  for (int k=ks; k<=ke; ++k) {
    Real x3 = pcoord->x3v(k);
    for (int j=js; j<=je; ++j) {
      Real x2 = pcoord->x2v(j);
      for (int i=is; i<=ie; ++i) {
        Real x1 = pcoord->x1v(i);

        Real rad, phi, z;
        GetCylCoordInSpherical(pcoord, rad, phi, z, i, j, k);

        Real cs_square   = PoverRho(rad, phi, z);
        Real omega_dyn   = std::sqrt(gm0/(rad*rad*rad));
        Real vel_K       = vK(porb, x1, x2, x3);
        Real vis_vel_cyl = -1.5*(nu_alpha*cs_square/rad/omega_dyn);

        Real den_gas        = DenProfileCyl_gas(rad, phi, z);
        Real vis_vel_r      = vis_vel_cyl*std::sin(x2);
        Real vel_gas_phi    = VelProfileCyl_gas(rad, phi, z);
        vel_gas_phi        -= orb_defined*vK(porb, x1, x2, x3);
        Real vel_gas_theta  = vis_vel_cyl*std::cos(x2);

        Real pre_diff = (qvalue + pvalue)*cs_square;

        Real delta_gas_vel1 = amp*std::sqrt(cs_square)*(ran2(&iseed) - 0.5);
        Real delta_gas_vel2 = amp*std::sqrt(cs_square)*(ran2(&iseed) - 0.5);
        Real delta_gas_vel3 = amp*std::sqrt(cs_square)*(ran2(&iseed) - 0.5);

        phydro->u(IDN, k, j, i) = den_gas;
        phydro->u(IM1, k, j, i) = den_gas*(vis_vel_r     + delta_gas_vel1);
        phydro->u(IM2, k, j, i) = den_gas*(vel_gas_theta + delta_gas_vel2);
        phydro->u(IM3, k, j, i) = den_gas*(vel_gas_phi   + delta_gas_vel3);

        phydro->u(IEN, k, j, i)  = cs_square*phydro->u(IDN, k, j, i)*igm1;
        phydro->u(IEN, k, j, i) += 0.5*(SQR(phydro->u(IM1, k, j, i))+SQR(phydro->u(IM2, k, j, i))
                                      + SQR(phydro->u(IM3, k, j, i)))/den_gas;

        if (NDUSTFLUIDS > 0) {
          for (int n=0; n<NDUSTFLUIDS; ++n) {
            int dust_id = n;
            int rho_id  = 4*dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + 2;
            int v3_id   = rho_id + 3;

            Real den_dust        = DenProfileCyl_dust(rad, phi, z, initial_D2G[dust_id], Hratio[dust_id]);
            Real vel_dust_cyl    = weight_dust[dust_id]*pre_diff/(2.0*rad*omega_dyn);
            Real vel_dust_phi    = VelProfileCyl_dust(rad, phi, z);
            vel_dust_phi        -= orb_defined*vK(porb, x1, x2, x3);
            Real vel_dust_r      = vel_dust_cyl*std::sin(x2);
            Real vel_dust_theta  = vel_dust_cyl*std::cos(x2);

            pdustfluids->df_cons(rho_id, k, j, i) = den_dust;
            pdustfluids->df_cons(v1_id,  k, j, i) = den_dust*vel_dust_r;
            pdustfluids->df_cons(v2_id,  k, j, i) = den_dust*vel_dust_theta;
            pdustfluids->df_cons(v3_id,  k, j, i) = den_dust*vel_dust_phi;
          }
        }

        if (NSCALARS > 0) {
          for (int n=0; n<NSCALARS; ++n) {
            pscalars->s(n, k, j, i) = DenProfileCyl_dust(rad, phi, z, 1., 0.1);
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

  if ((!Isothermal_Flag) && (beta > 0.0))
    ThermalRelaxation(pmb, time, dt, prim, prim_df, prim_s, bcc, cons, cons_df, cons_s);

  //DensityFloor(pmb, time, dt, prim, prim_df, prim_s, bcc, cons, cons_df, cons_s);

  return;
}


//!\f User source term to set density floor and adjust temperature
void DensityFloor(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s) {

  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

  for (int k=ks; k<=ke; ++k) {
#pragma omp for schedule(static)
    for (int j=js; j<=je; ++j) {
#pragma simd
      for (int i=is; i<=ie; ++i) {
        Real rad, phi, z;
        Real x1 = pmb->pcoord->x1v(i);
        GetCylCoordInSpherical(pmb->pcoord, rad, phi, z, i, j, k);
        cons(IDN, k, j, i) = std::max(cons(IDN, k, j, i), rho_floor*pow(rad/r0, pvalue));
      }
    }
  }

	if (NDUSTFLUIDS > 0) {
		for (int n=0; n<NDUSTFLUIDS; ++n) {
			int dust_id = n;
			int rho_id  = 4*dust_id;
			for (int k=ks; k<=ke; ++k) {
#pragma omp for schedule(static)
				for (int j=js; j<=je; ++j) {
#pragma simd
					for (int i=is; i<=ie; ++i) {
            Real rad, phi, z;
            Real x1 = pmb->pcoord->x1v(i);
            GetCylCoordInSpherical(pmb->pcoord, rad, phi, z, i, j, k);
            cons_df(rho_id, k, j, i) = std::max(cons_df(rho_id, k, j, i), rho_floor*pow(rad/r0, pvalue));
					}
				}
			}
		}
	}
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
          //Constant Stokes number in disk problems
          Real &st_time = stopping_time(dust_id, k, j, i);
          GetCylCoordInSpherical(pmb->pcoord, rad_arr(i), phi_arr(i), z_arr(i), i, j, k);
          st_time = Stokes_number[dust_id]*std::pow(rad_arr(i), 1.5)*inv_sqrt_gm0;
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
          GetCylCoordInSpherical(pmb->pcoord, rad_arr(i), phi_arr(i), z_arr(i), i, j, k);

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


void ThermalRelaxation(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s) {

  int is = pmb->is; int ie = pmb->ie;
  int js = pmb->js; int je = pmb->je;
  int ks = pmb->ks; int ke = pmb->ke;

  Real inv_beta  = 1.0/beta;
  //Real inv_beta = (beta > 1.0e-16) ? 1.0/beta : 1.0e16;
  Real gamma_gas = pmb->peos->GetGamma();
  Real igm1      = 1.0/(gamma_gas - 1.0);

  for (int k=ks; k<=ke; ++k) { // include ghost zone
    for (int j=js; j<=je; ++j) { // prim, cons
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        Real rad, phi, z;
        GetCylCoordInSpherical(pmb->pcoord, rad, phi, z, i, j, k);

        const Real &gas_rho = prim(IDN, k, j, i);
        const Real &gas_pre = prim(IPR, k, j, i);

        Real &gas_dens = cons(IDN, k, j, i);
        Real &gas_erg  = cons(IEN, k, j, i);

        Real omega_dyn      = std::sqrt(gm0/(rad*rad*rad));
        Real inv_t_cool     = omega_dyn*inv_beta;
        Real cs_square_init = PoverRho(rad, phi, z);

        Real delta_erg  = (gas_pre - gas_rho*cs_square_init)*igm1*inv_t_cool*dt;
        gas_erg        -= delta_erg;
        //Real delta_erg  = (gas_pre - gas_rho*cs_square_init)*igm1*(1.0 - exp(-omega_dyn*inv_beta*dt));
        //gas_erg        -= delta_erg;
      }
    }
  }
  return;
}


Real DustFluidsRatioMaximum(MeshBlock *pmb, int iout) {

  Real ratio_maximum = 0.0;

  int is = pmb->is, ie = pmb->ie, js = pmb->js,
      je = pmb->je, ks = pmb->ks, ke = pmb->ke;

  AthenaArray<Real> &df_prim = pmb->pdustfluids->df_prim;
  AthenaArray<Real> &w       = pmb->phydro->w;

  for (int n=0; n<NDUSTFLUIDS; n++) {
    int dust_id = n;
    int rho_id  = 4*dust_id;
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie; i++) {
          Real &gas_rho  = w(IDN, k, j, i);
          Real &dust_rho = df_prim(rho_id, k, j, i);
          ratio_maximum  = std::max(ratio_maximum, dust_rho/gas_rho);
        }
      }
    }
  }
  return ratio_maximum;
}


void RadiativeCondution(HydroDiffusion *phdif, MeshBlock *pmb,
    const AthenaArray<Real> &w, const AthenaArray<Real> &bc,
    int is, int ie, int js, int je, int ks, int ke) {

  Real inv_beta  = 1.0/beta;
  Real gamma_gas = pmb->peos->GetGamma();
  Real igm1      = 1.0/(gamma_gas - 1.0);

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        Real rad, phi, z;
        GetCylCoordInSpherical(pmb->pcoord, rad, phi, z, i, j, k);

        const Real &gas_rho = w(IDN, k, j, i);
        const Real &gas_pre = w(IPR, k, j, i);

        Real inv_omega_dyn   = std::sqrt((rad*rad*rad)/gm0);
        Real internal_erg    = gas_rho*gas_pre*igm1;
        Real kappa_radiative = internal_erg*inv_omega_dyn*inv_beta;

        phdif->kappa(HydroDiffusion::DiffProcess::aniso, k, j, i) = kappa_radiative;
      }
    }
  }
  return;
}


void GetCylCoordInSpherical(Coordinates *pco,Real &rad,Real &phi,Real &z,int i,int j,int k) {
  rad = std::abs(pco->x1v(i)*std::sin(pco->x2v(j)));
  phi = pco->x3v(k);
  z   = pco->x1v(i)*std::cos(pco->x2v(j));
  return;
}


Real DenProfileCyl_gas(const Real rad, const Real phi, const Real z) {
  Real den;
  Real cs2    = PoverRho(rad, phi, z);
  Real denmid = rho0*std::pow(rad/r0, pvalue);
  Real dentem = denmid*std::exp(gm0/cs2*(1./std::sqrt(SQR(rad)+SQR(z))-1./rad));
  den         = dentem;
  return std::max(den, dfloor);
}


Real DenProfileCyl_dust(const Real rad, const Real phi, const Real z, const Real den_ratio, const Real H_ratio) {
  Real den;
  Real cs2    = PoverRho(rad, phi, z);
  Real denmid = den_ratio*rho0*std::pow(rad/r0, pvalue);
  Real dentem = denmid*std::exp(gm0/(SQR(H_ratio)*cs2)*(1./std::sqrt(SQR(rad)+SQR(z))-1./rad));
  den         = dentem;
  return std::max(den, dffloor);
}


Real PoverRho(const Real rad, const Real phi, const Real z) {
  Real poverr;
  poverr = cs2_0*std::pow(rad/r0, qvalue);
  return poverr;
}


Real VelProfileCyl_gas(const Real rad, const Real phi, const Real z) {
  Real cs2 = PoverRho(rad, phi, z);
  Real vel = (pvalue+qvalue)*cs2/(gm0/rad) + (1.0+qvalue) - qvalue*rad/std::sqrt(rad*rad+z*z);
  vel      = std::sqrt(gm0/rad)*std::sqrt(vel) - rad*Omega0;
  return vel;
}


Real VelProfileCyl_dust(const Real rad, const Real phi, const Real z) {
  Real dis = std::sqrt(SQR(rad) + SQR(z));
  Real vel = std::sqrt(gm0/dis) - rad*Omega0;
  return vel;
}


int RefinementCondition(MeshBlock *pmb) {
  Real time = pmb->pmy_mesh->time;

  bool time_flag = (time >= time_refine);

  if (time_flag) {
    for (int k=pmb->ks; k<=pmb->ke; k++) {
      for (int j=pmb->js; j<=pmb->je; j++) {
        for (int i=pmb->is; i<=pmb->ie; i++) {
          Real &rad   = pmb->pcoord->x1v(i);
          Real &theta = pmb->pcoord->x2v(j);

          bool theta_lower = (theta >= refine_theta_lower);
          bool theta_upper = (theta <= refine_theta_upper);
          bool rad_min     = (rad >= refine_r_min);
          bool rad_max     = (rad <= refine_r_max);

          if (theta_lower && theta_upper && rad_min && rad_max)
            return 1;
        }
      }
    }
  }

  return 0;
}


// Endianness: used for writing binary output
void Swap4Bytes(void *vdat) {
  char tmp, *dat = (char *) vdat;
  tmp = dat[0];  dat[0] = dat[3];  dat[3] = tmp;
  tmp = dat[1];  dat[1] = dat[2];  dat[2] = tmp;
}


void Vr_outflow(const Real r_active, const Real r_ghost, const Real rho_active,
               const Real rho_ghost, const Real vr_active, Real &vr_ghost) {

  vr_ghost = (vr_active <= 0.0) ? ((rho_active*SQR(r_active)*vr_active)/(rho_ghost*SQR(r_ghost))) : 0.0;
  return;
}


void Vr_Mdot(const Real r_active, const Real r_ghost, const Real rho_active,
               const Real rho_ghost, const Real vr_active, Real &vr_ghost) {

  vr_ghost = rho_active*SQR(r_active)*vr_active/(rho_ghost*SQR(r_ghost));
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
        Real rad_active, phi_active, z_active;
        GetCylCoordInSpherical(pco, rad_ghost,  phi_ghost,  z_ghost,  il-i, j, k);
        GetCylCoordInSpherical(pco, rad_active, phi_active, z_active, il,   j, k);

        Real cs_square   = PoverRho(rad_ghost, phi_ghost, z_ghost);
        Real omega_dyn   = std::sqrt(gm0/(rad_ghost*rad_ghost*rad_ghost));
        Real vis_vel_cyl = -1.5*(nu_alpha*cs_square/rad_ghost/omega_dyn);
        Real vel_K       = vK(pmb->porb, pco->x1v(il-i), pco->x2v(j), pco->x3v(k));
        Real pre_diff    = (qvalue + pvalue)*cs_square;

        Real &gas_rho_ghost  = prim(IDN, k, j, il-i);
        Real &gas_vel1_ghost = prim(IM1, k, j, il-i);
        Real &gas_vel2_ghost = prim(IM2, k, j, il-i);
        Real &gas_vel3_ghost = prim(IM3, k, j, il-i);
        Real &gas_pres_ghost = prim(IEN, k, j, il-i);

        Real &gas_rho_active  = prim(IDN, k, j, il);
        Real &gas_vel1_active = prim(IM1, k, j, il);
        Real &gas_vel2_active = prim(IM2, k, j, il);
        Real &gas_vel3_active = prim(IM3, k, j, il);
        Real &gas_pres_active = prim(IEN, k, j, il);

        gas_rho_ghost     = DenProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
        Real vel_gas_phi  = VelProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
        vel_gas_phi      -= orb_defined*vel_K;

        //gas_vel1_ghost = vis_vel_cyl*std::sin(pco->x2v(j));
        Vr_outflow(rad_active, rad_ghost, gas_rho_active, gas_rho_ghost,
                                          gas_vel1_active, gas_vel1_ghost);
        gas_vel2_ghost = vis_vel_cyl*std::cos(pco->x2v(j));
        gas_vel3_ghost = vel_gas_phi;
        gas_pres_ghost = cs_square*gas_rho_ghost;

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

            dust_rho_ghost       = DenProfileCyl_dust(rad_ghost, phi_ghost, z_ghost, initial_D2G[dust_id], Hratio[dust_id]);
            Real vel_dust_cyl    = weight_dust[dust_id]*pre_diff/(2.0*rad_ghost*omega_dyn);
            Real vel_dust_phi    = VelProfileCyl_dust(rad_ghost, phi_ghost, z_ghost);
            vel_dust_phi        -= orb_defined*vel_K;
            Real vel_dust_r      = vel_dust_cyl*std::sin(pco->x2v(j));
            Real vel_dust_theta  = vel_dust_cyl*std::cos(pco->x2v(j));

            dust_vel1_ghost = vel_dust_r;
            dust_vel2_ghost = vel_dust_theta;
            dust_vel3_ghost = vel_dust_phi;
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
        GetCylCoordInSpherical(pco, rad_ghost, phi_ghost, z_ghost, iu+i, j, k);

        Real cs_square   = PoverRho(rad_ghost, phi_ghost, z_ghost);
        Real omega_dyn   = std::sqrt(gm0/(rad_ghost*rad_ghost*rad_ghost));
        Real vis_vel_cyl = -1.5*(nu_alpha*cs_square/rad_ghost/omega_dyn);
        Real vel_K       = vK(pmb->porb, pco->x1v(iu+i), pco->x2v(j), pco->x3v(k));
        Real pre_diff    = (qvalue + pvalue)*cs_square;

        Real &gas_rho_ghost  = prim(IDN, k, j, iu+i);
        Real &gas_vel1_ghost = prim(IM1, k, j, iu+i);
        Real &gas_vel2_ghost = prim(IM2, k, j, iu+i);
        Real &gas_vel3_ghost = prim(IM3, k, j, iu+i);
        Real &gas_pres_ghost = prim(IEN, k, j, iu+i);

        gas_rho_ghost     = DenProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
        Real vel_gas_phi  = VelProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
        vel_gas_phi      -= orb_defined*vel_K;

        gas_vel1_ghost = vis_vel_cyl*std::sin(pco->x2v(j));
        gas_vel2_ghost = vis_vel_cyl*std::cos(pco->x2v(j));
        gas_vel3_ghost = vel_gas_phi;
        gas_pres_ghost = cs_square*gas_rho_ghost;

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

            dust_rho_ghost       = DenProfileCyl_dust(rad_ghost, phi_ghost, z_ghost, initial_D2G[dust_id], Hratio[dust_id]);
            Real vel_dust_cyl    = weight_dust[dust_id]*pre_diff/(2.0*rad_ghost*omega_dyn);
            Real vel_dust_phi    = VelProfileCyl_dust(rad_ghost, phi_ghost, z_ghost);
            vel_dust_phi        -= orb_defined*vel_K;
            Real vel_dust_r      = vel_dust_cyl*std::sin(pco->x2v(j));
            Real vel_dust_theta  = vel_dust_cyl*std::cos(pco->x2v(j));

            dust_vel1_ghost = vel_dust_r;
            dust_vel2_ghost = vel_dust_theta;
            dust_vel3_ghost = vel_dust_phi;
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
        Real rad_active, phi_active, z_active;
        GetCylCoordInSpherical(pco, rad_ghost,  phi_ghost,  z_ghost,  i, jl-j, k);
        GetCylCoordInSpherical(pco, rad_active, phi_active, z_active, i, jl,   k);

        Real cs_square   = PoverRho(rad_ghost, phi_ghost, z_ghost);
        Real omega_dyn   = std::sqrt(gm0/(rad_ghost*rad_ghost*rad_ghost));
        Real vis_vel_cyl = -1.5*(nu_alpha*cs_square/rad_ghost/omega_dyn);
        Real vel_K       = vK(pmb->porb, pco->x1v(i), pco->x2v(jl-j), pco->x3v(k));
        Real pre_diff    = (qvalue + pvalue)*cs_square;

        Real &gas_rho_ghost  = prim(IDN, k, jl-j, i);
        Real &gas_vel1_ghost = prim(IM1, k, jl-j, i);
        Real &gas_vel2_ghost = prim(IM2, k, jl-j, i);
        Real &gas_vel3_ghost = prim(IM3, k, jl-j, i);
        Real &gas_pres_ghost = prim(IEN, k, jl-j, i);

        Real &gas_rho_active  = prim(IDN, k, jl, i);
        Real &gas_vel1_active = prim(IM1, k, jl, i);
        Real &gas_vel2_active = prim(IM2, k, jl, i);
        Real &gas_vel3_active = prim(IM3, k, jl, i);
        Real &gas_pres_active = prim(IEN, k, jl, i);

        gas_rho_ghost     = DenProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
        Real vel_gas_phi  = VelProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
        vel_gas_phi      -= orb_defined*vel_K;

        //gas_vel1_ghost = vis_vel_cyl*std::sin(pco->x2v(jl-j));
        //gas_vel2_ghost = gas_vel2_active;
        Vr_Mdot(rad_active, rad_ghost, gas_rho_active, gas_rho_ghost,
                gas_vel1_active, gas_vel1_ghost);
        gas_vel2_ghost = vis_vel_cyl*std::cos(pco->x2v(jl-j));
        gas_vel3_ghost = vel_gas_phi;
        gas_pres_ghost = cs_square*gas_rho_ghost;

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

            dust_rho_ghost       = DenProfileCyl_dust(rad_ghost, phi_ghost, z_ghost, initial_D2G[dust_id], Hratio[dust_id]);
            Real vel_dust_cyl    = weight_dust[dust_id]*pre_diff/(2.0*rad_ghost*omega_dyn);
            Real vel_dust_phi    = VelProfileCyl_dust(rad_ghost, phi_ghost, z_ghost);
            vel_dust_phi        -= orb_defined*vel_K;
            Real vel_dust_r      = vel_dust_cyl*std::sin(pco->x2v(jl-j));
            Real vel_dust_theta  = vel_dust_cyl*std::cos(pco->x2v(jl-j));

            dust_vel1_ghost = vel_dust_r;
            dust_vel2_ghost = vel_dust_theta;
            dust_vel3_ghost = vel_dust_phi;
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
        Real rad_active, phi_active, z_active;
        GetCylCoordInSpherical(pco, rad_ghost,  phi_ghost,  z_ghost,  i, ju+j, k);
        GetCylCoordInSpherical(pco, rad_active, phi_active, z_active, i, ju,   k);

        Real cs_square   = PoverRho(rad_ghost, phi_ghost, z_ghost);
        Real omega_dyn   = std::sqrt(gm0/(rad_ghost*rad_ghost*rad_ghost));
        Real vis_vel_cyl = -1.5*(nu_alpha*cs_square/rad_ghost/omega_dyn);
        Real vel_K       = vK(pmb->porb, pco->x1v(i), pco->x2v(ju+j), pco->x3v(k));
        Real pre_diff    = (qvalue + pvalue)*cs_square;

        Real &gas_rho_ghost  = prim(IDN, k, ju+j, i);
        Real &gas_vel1_ghost = prim(IM1, k, ju+j, i);
        Real &gas_vel2_ghost = prim(IM2, k, ju+j, i);
        Real &gas_vel3_ghost = prim(IM3, k, ju+j, i);
        Real &gas_pres_ghost = prim(IEN, k, ju+j, i);

        Real &gas_rho_active  = prim(IDN, k, ju, i);
        Real &gas_vel1_active = prim(IM1, k, ju, i);
        Real &gas_vel2_active = prim(IM2, k, ju, i);
        Real &gas_vel3_active = prim(IM3, k, ju, i);
        Real &gas_pres_active = prim(IEN, k, ju, i);

        gas_rho_ghost     = DenProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
        Real vel_gas_phi  = VelProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
        vel_gas_phi      -= orb_defined*vel_K;

        //gas_vel1_ghost = vis_vel_cyl*std::sin(pco->x2v(ju+j));
        //gas_vel2_ghost = gas_vel2_active;
        Vr_Mdot(rad_active, rad_ghost, gas_rho_active, gas_rho_ghost,
                gas_vel1_active, gas_vel1_ghost);
        gas_vel2_ghost = vis_vel_cyl*std::cos(pco->x2v(ju+j));
        gas_vel3_ghost = vel_gas_phi;
        gas_pres_ghost = cs_square*gas_rho_ghost;

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

            dust_rho_ghost       = DenProfileCyl_dust(rad_ghost, phi_ghost, z_ghost, initial_D2G[dust_id], Hratio[dust_id]);
            Real vel_dust_cyl    = weight_dust[dust_id]*pre_diff/(2.0*rad_ghost*omega_dyn);
            Real vel_dust_phi    = VelProfileCyl_dust(rad_ghost, phi_ghost, z_ghost);
            vel_dust_phi        -= orb_defined*vel_K;
            Real vel_dust_r      = vel_dust_cyl*std::sin(pco->x2v(ju+j));
            Real vel_dust_theta  = vel_dust_cyl*std::cos(pco->x2v(ju+j));

            dust_vel1_ghost = vel_dust_r;
            dust_vel2_ghost = vel_dust_theta;
            dust_vel3_ghost = vel_dust_phi;
          }
        }
      }
    }
  }
}



//--------------------------------------------------------------------------------------
//!\f: User-defined output function: resampling data to a uniform grid
void DoOutput(MeshBlock *pmb, int dlevel) {

  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;
  int nx1 = pmb->block_size.nx1;
  int nx2 = pmb->block_size.nx2;
  int nx3 = pmb->block_size.nx3;
  int mynx1, mynx2; // output grid size
  int indx1, indx2; // starting index in global grid

  Real time = pmb->pmy_mesh->time+pmb->pmy_mesh->dt;
  Real fac  = 1.0/((time-prev_time)*nx3+TINY_NUMBER);
  Real wei  = 1.0;
  if (nx3 > 1)
    wei = (pmb->block_size.x3max-pmb->block_size.x3min)/
          (pmb->pmy_mesh->mesh_size.x3max-pmb->pmy_mesh->mesh_size.x3min);
  fac *= wei;

  AthenaArray<Real> vol;
  AthenaArray<Real> myx1v, myx2v;
  AthenaArray<float> newdata;

  int nfac = 1;
  if (dlevel > 0) { // need downsampling
    mynx1 = nx1>>dlevel;
    mynx2 = nx2>>dlevel;
    nfac  = nfac<<dlevel;
    indx1 = pmb->loc.lx1*nx1/nfac;
    indx2 = pmb->loc.lx2*nx2/nfac;
  } else { // just need to copy
    mynx1 = nx1<<(-dlevel);
    mynx2 = nx2<<(-dlevel);
    nfac  = nfac<<(-dlevel);
    indx1 = pmb->loc.lx1*nx1*nfac;
    indx2 = pmb->loc.lx2*nx2*nfac;
  }

  newdata.NewAthenaArray(mynx2,mynx1);
  myx1v.NewAthenaArray(mynx1);
  myx2v.NewAthenaArray(mynx2);
  vol.NewAthenaArray(pmb->block_size.nx1+2*NGHOST);

  for (int i=0;i<mynx1;++i) {
    Real r1  = pmb->ruser_meshblock_data[2](i);
    Real r2  = pmb->ruser_meshblock_data[2](i+1);
    myx1v(i) = 0.75*(pow(r2,4) - pow(r1,4))/(pow(r2,3) - pow(r1,3)); // TODO
  }

  for (int j=0;j<mynx2;++j) {
    Real t1  = pmb->ruser_meshblock_data[3](j);
    Real t2  = pmb->ruser_meshblock_data[3](j+1);
    myx2v(j) = ((sin(t2) - t2*cos(t2)) - (sin(t1) - t1*cos(t1)))/(cos(t1) - cos(t2)); // TODO
  }

  std::stringstream msg;
  int big_end = IsBigEndianOutput(); // =1 on big endian machine

  float *data;
  int ndata = std::max(mynx1, mynx2)+1;
  data = new float[ndata];

  // construct file name
  std::string fname;
  char number[6];
  sprintf(number, "%05d", file_number);
  char blockid[12];
  sprintf(blockid, "block%d", pmb->gid);

  std::string dirname = "2D_avg_";
  std::string cmd("mkdir -p " + dirname + number);
  system(cmd.c_str());

  fname.assign(dirname);
  fname.append(number);
  fname.append("/");
  fname.append("VSI");
  //fname.assign("VSI");
  fname.append(".");
  fname.append(blockid);
  fname.append(".");
  fname.append("avg");
  fname.append(".");
  fname.append(number);
  fname.append(".vtk");

  // open file
  FILE *pfile;
  if ((pfile = fopen(fname.c_str(), "w")) == NULL) {
    msg << "### FATAL ERROR in function [AvgOutput]"
        << std::endl << "Output file '" << fname << "' could not be opened" <<std::endl;
    throw std::runtime_error(msg.str().c_str());
  }

  // output file
  fprintf(pfile, "# vtk DataFile Version 2.0\n");
  fprintf(pfile, "# Athena++ data at time= %f DeltaT= %f indx1= %d indx2= %d indx3= %d\n",
                                    time, time-prev_time, indx1, indx2, 0);
  fprintf(pfile, "BINARY\n");

  fprintf(pfile, "DATASET RECTILINEAR_GRID\n");
  fprintf(pfile, "DIMENSIONS %d %d %d\n", mynx1+1, mynx2+1, 2);

  fprintf(pfile, "X_COORDINATES %d float\n", mynx1+1);
  for (int i=0; i<=mynx1; ++i) {
    data[i] = (float)(pmb->ruser_meshblock_data[2](i));
  }
  if (!big_end) {for (int i=0; i<=mynx1; ++i) Swap4Bytes(&data[i]);}
  fwrite(data, sizeof(float), (size_t)(mynx1+1), pfile);

  fprintf(pfile,"\nY_COORDINATES %d float\n",mynx2+1);
  for (int j=0; j<=mynx2; ++j) {
    data[j] = (float)(pmb->ruser_meshblock_data[3](j));
  }
  if (!big_end) {for (int j=0; j<=mynx2; ++j) Swap4Bytes(&data[j]);}
  fwrite(data, sizeof(float), (size_t)(mynx2+1), pfile);

  fprintf(pfile, "\nZ_COORDINATES %d float\n", 2);
  data[0] = (float)(pmb->pmy_mesh->mesh_size.x3min);
  data[1] = (float)(pmb->pmy_mesh->mesh_size.x3max);
  if (!big_end) {for (int k=0; k<=1; ++k) Swap4Bytes(&data[k]);}
  fwrite(data, sizeof(float), (size_t)(2), pfile);

  fprintf(pfile, "\nCELL_DATA %d", mynx1*mynx2);

  std::string vname;
  for (int n=0; n<nvar; ++n) {
    switch (n) {
      case 0:
        vname.assign("rho");
        break;
      case 1:
        vname.assign("pres");
        break;
      case 2:
        vname.assign("rhov1");
        break;
      case 3:
        vname.assign("rhov2");
        break;
      case 4:
        vname.assign("rhov3");
        break;
      case 5:
        vname.assign("rhov1sqr");
        break;
      case 6:
        vname.assign("rhov2sqr");
        break;
      case 7:
        vname.assign("rhov3sqr");
        break;
      case 8:
        vname.assign("rhov1v2");
        break;
      case 9:
        vname.assign("rhov2v3");
        break;
      case 10:
        vname.assign("rhov3v1");
        break;
      case 11:
        vname.assign("kinflx1");
        break;
      case 12:
        vname.assign("kinflx2");
        break;
      case 13:
        vname.assign("kinflx3");
        break;
      case 14:
        vname.assign("enpflx1");
        break;
      case 15:
        vname.assign("enpflx2");
        break;
      case 16:
        vname.assign("enpflx3");
        break;
      case (17):
        vname.assign("ratio_" + std::to_string(1));
        break;
      case (18):
        vname.assign("max_ratio_" + std::to_string(1));
        break;
      case (19):
        vname.assign("ratio_" + std::to_string(2));
        break;
      case (20):
        vname.assign("max_ratio_" + std::to_string(2));
        break;
      default:
        std::stringstream msg;
        msg << "### FATAL ERROR in averaged data output." << std::endl;
        throw std::runtime_error(msg.str().c_str());
    }

    if (dlevel > 0) {
      for (int j=0; j<mynx2; ++j) {
        int myjs = j*nfac;
        int myje = myjs+nfac-1;
        for (int i=0; i<mynx1; ++i) {
          int myis  = i*nfac;
          int myie  = myis+nfac-1;
          Real val  = 0.0;
          Real vtot = 0.0;
          for (int myj=myjs; myj<=myje; ++myj) {
            pmb->pcoord->CellVolume(0,myj+NGHOST,myis+NGHOST,myie+NGHOST,vol);
#pragma simd
            for (int myi=myis; myi<=myie; ++myi) {
              val += pmb->ruser_meshblock_data[n0+n](myj,myi)*vol(myi+NGHOST);
              vtot+= vol(myi+NGHOST);
            }
          }
          newdata(j,i) = (float)(val/vtot);
        }
      }
    } else { //just copy
      for (int j=0; j<mynx2; ++j) {
        int myj = j/nfac;
        for (int i=0; i<mynx1; ++i) {
          int myi = i/nfac;
          newdata(j,i) = pmb->ruser_meshblock_data[n0+n](myj,myi);
        }
      }
    }

    fprintf(pfile, "\nSCALARS %s float\n", vname.c_str());
    fprintf(pfile, "LOOKUP_TABLE default\n");
    for (int j=0; j<mynx2; ++j) {
#pragma simd
      for (int i=0; i<mynx1; ++i)
        data[i] = newdata(j, i)*fac;

      if (!big_end) {for (int i=0; i<mynx1; ++i) Swap4Bytes(&data[i]);}
      fwrite(data, sizeof(float), (size_t)mynx1, pfile);
    }
  }

  fclose(pfile);

  delete data;
  newdata.DeleteAthenaArray();
  myx1v.DeleteAthenaArray();
  myx2v.DeleteAthenaArray();
  vol.DeleteAthenaArray();

  return;
}


int IsBigEndianOutput() {
  short int n = 1;
  char *ep = (char *)&n;
  return (*ep == 0); // Returns 1 on a big endian machine
}


void AccumulateData(MeshBlock *pmb) {
  Hydro *phydro           = pmb->phydro;
  Field *pfield           = pmb->pfield;
  DustFluids *pdustfluids = pmb->pdustfluids;

  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;
  int nx1 = pmb->block_size.nx1;
  int nx2 = pmb->block_size.nx2;
  Real dt    = pmb->pmy_mesh->dt;
  Real mygam = pmb->peos->GetGamma();
  Real igm1  = 1.0/(mygam-1.0);
  int nthreads = pmb->pmy_mesh->GetNumMeshThreads();

#pragma omp parallel default(shared) num_threads(nthreads)
  {
    for (int k=ks; k<=ke; ++k) {
#pragma omp for schedule(static)
      for (int j=js; j<=je; ++j) {
        Real x2 = pmb->pcoord->x2v(j);
        Real sintheta = std::sin(x2);
        Real costheta = std::cos(x2);
        int  myj = j-js;
    //#pragma simd
        for (int i=is; i<=ie; ++i) {
          int myi = i-is;
          Real &gas_rho  = phydro->w(IDN, k, j, i);
          Real &gas_vel1 = phydro->w(IM1, k, j, i);
          Real &gas_vel2 = phydro->w(IM2, k, j, i);
          Real &gas_vel3 = phydro->w(IM3, k, j, i);
          Real &gas_pres = phydro->w(IPR, k, j, i);

          Real gas_velz = gas_vel1*costheta - gas_vel2*sintheta;
          Real gas_velR = gas_vel1*sintheta + gas_vel2*costheta;
          // rho, pres
          // rhov1-rhov3
          // rhov1sqr - rhov3sqr
          // rhov1v2, rhov2v3, rhov3v1
          // kinflx1-kinflx3
          // enpflx1-enpflx3
          // b1-b3
          // b1sqr - b3sqr
          // b1b2,b2b3,b3b1
          // poyflx1-poyflx3
          // ey1, ey2, eyn
          int n = n0;
          pmb->ruser_meshblock_data[n](myj, myi) += dt*gas_rho;
          n++;
          pmb->ruser_meshblock_data[n](myj, myi) += dt*gas_pres;
          n++;
          pmb->ruser_meshblock_data[n](myj, myi) += dt*gas_rho*gas_vel1;
          n++;
          pmb->ruser_meshblock_data[n](myj, myi) += dt*gas_rho*gas_vel2;
          n++;
          pmb->ruser_meshblock_data[n](myj, myi) += dt*gas_rho*gas_vel3;
          n++;
          pmb->ruser_meshblock_data[n](myj, myi) += dt*gas_rho*SQR(gas_vel1);
          n++;
          pmb->ruser_meshblock_data[n](myj, myi) += dt*gas_rho*SQR(gas_vel2);
          n++;
          pmb->ruser_meshblock_data[n](myj, myi) += dt*gas_rho*SQR(gas_vel3);
          n++;
          pmb->ruser_meshblock_data[n](myj, myi) += dt*gas_rho*gas_vel1*gas_vel2;
          n++;
          pmb->ruser_meshblock_data[n](myj, myi) += dt*gas_rho*gas_vel2*gas_vel3;
          n++;
          pmb->ruser_meshblock_data[n](myj, myi) += dt*gas_rho*gas_vel3*gas_vel1;
          n++;
          // kinetic energy flux
          Real rhovsq = 0.5*gas_rho*(SQR(gas_vel1)+SQR(gas_vel2)+SQR(gas_vel3));
          pmb->ruser_meshblock_data[n](myj, myi) += dt*rhovsq*gas_vel1;
          n++;
          pmb->ruser_meshblock_data[n](myj, myi) += dt*rhovsq*gas_vel2;
          n++;
          pmb->ruser_meshblock_data[n](myj, myi) += dt*rhovsq*gas_vel3;
          n++;
          // enthalpy flux
          Real enp = mygam*igm1*gas_pres;
          pmb->ruser_meshblock_data[n](myj, myi) += dt*enp*gas_vel1;
          n++;
          pmb->ruser_meshblock_data[n](myj, myi) += dt*enp*gas_vel2;
          n++;
          pmb->ruser_meshblock_data[n](myj, myi) += dt*enp*gas_vel3;
          n++;

          for (int m=0; m<NDUSTFLUIDS; ++m) {
            int dust_id = m;
            int rho_id  = 4*dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + 2;
            int v3_id   = rho_id + 3;
            Real &dust_rho = pdustfluids->df_prim(rho_id, k, j, i);

            Real ratio = dust_rho/gas_rho;
            pmb->ruser_meshblock_data[n](myj, myi) += dt*ratio;
            n++;

            Real max_ratio = (pmb->ruser_meshblock_data[n](myj, myi));
            max_ratio = std::max(ratio, max_ratio);
            pmb->ruser_meshblock_data[n](myj, myi) = max_ratio;
            n++;
          }
        }
      }
    }
  }

  return;
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
        GetCylCoordInSpherical(pmb->pcoord, rad, phi, z, i, j, k);

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
        GetCylCoordInSpherical(pmb->pcoord, rad_arr(i), phi_arr(i), z_arr(i), i, j, k); // convert to cylindrical coordinates

        if (rad_arr(i) <= radius_inner_damping) {
          // See de Val-Borro et al. 2006 & 2007
          omega_dyn(i)       = std::sqrt(gm0/(rad_arr(i)*rad_arr(i)*rad_arr(i)));
          R_func(i)          = SQR((rad_arr(i) - radius_inner_damping)*inv_inner_damp);
          inv_damping_tau(i) = (damping_rate*omega_dyn(i));

          Real gas_rho_0   = DenProfileCyl_gas(rad_arr(i), phi_arr(i), z_arr(i));
          Real cs_square_0 = PoverRho(rad_arr(i), phi_arr(i), z_arr(i));

          Real vis_vel_cyl  = -1.5*(nu_alpha*cs_square_0/rad_arr(i)/omega_dyn(i));
          Real vel_gas_phi  = VelProfileCyl_gas(rad_arr(i), phi_arr(i), z_arr(i));
          vel_gas_phi      -= orb_defined*vK(pmb->porb, x1, x2, x3);

          Real gas_vel1_0 = vis_vel_cyl*std::sin(x2);
          Real gas_vel2_0 = vis_vel_cyl*std::cos(x2);
          Real gas_vel3_0 = vel_gas_phi;
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
        GetCylCoordInSpherical(pmb->pcoord, rad_arr(i), phi_arr(i), z_arr(i), i, j, k); // convert to cylindrical coordinates

        if (rad_arr(i) >= radius_outer_damping) {
          // See de Val-Borro et al. 2006 & 2007
          omega_dyn(i)       = std::sqrt(gm0/(rad_arr(i)*rad_arr(i)*rad_arr(i)));
          R_func(i)          = SQR((rad_arr(i) - radius_outer_damping)*inv_outer_damp);
          inv_damping_tau(i) = (damping_rate*omega_dyn(i));

          Real gas_rho_0   = DenProfileCyl_gas(rad_arr(i), phi_arr(i), z_arr(i));
          Real cs_square_0 = PoverRho(rad_arr(i), phi_arr(i), z_arr(i));

          Real vis_vel_cyl  = -1.5*(nu_alpha*cs_square_0/rad_arr(i)/omega_dyn(i));
          Real vel_gas_phi  = VelProfileCyl_gas(rad_arr(i), phi_arr(i), z_arr(i));
          vel_gas_phi      -= orb_defined*vK(pmb->porb, x1, x2, x3);

          Real gas_vel1_0 = vis_vel_cyl*std::sin(x2);
          Real gas_vel2_0 = vis_vel_cyl*std::cos(x2);
          Real gas_vel3_0 = vel_gas_phi;
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


void UpperWaveDampingGas(MeshBlock *pmb, const Real time, const Real dt, int il, int iu,
    int jl, int ju, int kl, int ku, AthenaArray<Real> &prim, AthenaArray<Real> &cons) {

  Real mygam = pmb->peos->GetGamma();
  Real igm1  = 1.0/(mygam - 1.0);

  Real inv_upper_damp = 1.0/upper_altitude_damping;
  Real inv_lower_damp = 1.0/lower_altitude_damping;

  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;

  Real orb_defined;
  (pmb->porb->orbital_advection_defined) ? orb_defined = 1.0 : orb_defined = 0.0;

  int nc1 = pmb->ncells1;

  AthenaArray<Real> omega_dyn, Theta_func, inv_damping_tau;
  AthenaArray<Real> rad_arr, phi_arr, z_arr;

  omega_dyn.NewAthenaArray(nc1);
  Theta_func.NewAthenaArray(nc1);
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
        GetCylCoordInSpherical(pmb->pcoord, rad_arr(i), phi_arr(i), z_arr(i), i, j, k); // convert to cylindrical coordinates

        if (x2 <= theta_upper_damping) {
          // See de Val-Borro et al. 2006 & 2007
          omega_dyn(i)       = std::sqrt(gm0/(rad_arr(i)*rad_arr(i)*rad_arr(i)));
          Theta_func(i)      = SQR((x2 - theta_upper_damping)*inv_upper_damp);
          inv_damping_tau(i) = (damping_rate*omega_dyn(i));

          Real gas_rho_0   = DenProfileCyl_gas(rad_arr(i), phi_arr(i), z_arr(i));
          Real cs_square_0 = PoverRho(rad_arr(i), phi_arr(i), z_arr(i));

          Real vis_vel_cyl  = -1.5*(nu_alpha*cs_square_0/rad_arr(i)/omega_dyn(i));
          Real vel_gas_phi  = VelProfileCyl_gas(rad_arr(i), phi_arr(i), z_arr(i));
          vel_gas_phi      -= orb_defined*vK(pmb->porb, x1, x2, x3);

          Real gas_vel1_0 = vis_vel_cyl*std::sin(x2);
          Real gas_vel2_0 = vis_vel_cyl*std::cos(x2);
          Real gas_vel3_0 = vel_gas_phi;
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

          Real delta_gas_rho  = (gas_rho_0  - gas_rho )*Theta_func(i)*inv_damping_tau(i)*dt;
          Real delta_gas_vel1 = (gas_vel1_0 - gas_vel1)*Theta_func(i)*inv_damping_tau(i)*dt;
          Real delta_gas_vel2 = (gas_vel2_0 - gas_vel2)*Theta_func(i)*inv_damping_tau(i)*dt;
          Real delta_gas_vel3 = (gas_vel3_0 - gas_vel3)*Theta_func(i)*inv_damping_tau(i)*dt;
          Real delta_gas_pre  = (gas_pre_0  - gas_pre )*Theta_func(i)*inv_damping_tau(i)*dt;

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


void LowerWaveDampingGas(MeshBlock *pmb, const Real time, const Real dt, int il, int iu,
    int jl, int ju, int kl, int ku, AthenaArray<Real> &prim, AthenaArray<Real> &cons) {

  Real mygam = pmb->peos->GetGamma();
  Real igm1  = 1.0/(mygam - 1.0);

  Real inv_upper_damp = 1.0/upper_altitude_damping;
  Real inv_lower_damp = 1.0/lower_altitude_damping;

  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;

  Real orb_defined;
  (pmb->porb->orbital_advection_defined) ? orb_defined = 1.0 : orb_defined = 0.0;

  int nc1 = pmb->ncells1;

  AthenaArray<Real> omega_dyn, Theta_func, inv_damping_tau;
  AthenaArray<Real> rad_arr, phi_arr, z_arr;

  omega_dyn.NewAthenaArray(nc1);
  Theta_func.NewAthenaArray(nc1);
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
        GetCylCoordInSpherical(pmb->pcoord, rad_arr(i), phi_arr(i), z_arr(i), i, j, k); // convert to cylindrical coordinates

        if (x2 >= theta_lower_damping) {
          // See de Val-Borro et al. 2006 & 2007
          omega_dyn(i)       = std::sqrt(gm0/(rad_arr(i)*rad_arr(i)*rad_arr(i)));
          Theta_func(i)      = SQR((x2 - theta_lower_damping)*inv_lower_damp);
          inv_damping_tau(i) = (damping_rate*omega_dyn(i));

          Real gas_rho_0   = DenProfileCyl_gas(rad_arr(i), phi_arr(i), z_arr(i));
          Real cs_square_0 = PoverRho(rad_arr(i), phi_arr(i), z_arr(i));

          Real vis_vel_cyl  = -1.5*(nu_alpha*cs_square_0/rad_arr(i)/omega_dyn(i));
          Real vel_gas_phi  = VelProfileCyl_gas(rad_arr(i), phi_arr(i), z_arr(i));
          vel_gas_phi      -= orb_defined*vK(pmb->porb, x1, x2, x3);

          Real gas_vel1_0 = vis_vel_cyl*std::sin(x2);
          Real gas_vel2_0 = vis_vel_cyl*std::cos(x2);
          Real gas_vel3_0 = vel_gas_phi;
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

          Real delta_gas_rho  = (gas_rho_0  - gas_rho )*Theta_func(i)*inv_damping_tau(i)*dt;
          Real delta_gas_vel1 = (gas_vel1_0 - gas_vel1)*Theta_func(i)*inv_damping_tau(i)*dt;
          Real delta_gas_vel2 = (gas_vel2_0 - gas_vel2)*Theta_func(i)*inv_damping_tau(i)*dt;
          Real delta_gas_vel3 = (gas_vel3_0 - gas_vel3)*Theta_func(i)*inv_damping_tau(i)*dt;
          Real delta_gas_pre  = (gas_pre_0  - gas_pre )*Theta_func(i)*inv_damping_tau(i)*dt;

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



void InnerWaveDampingDust(MeshBlock *pmb, const Real time, const Real dt, int il, int iu,
    int jl, int ju, int kl, int ku, AthenaArray<Real> &prim_df, AthenaArray<Real> &cons_df) {

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
      for (int n=0; n<NDUSTFLUIDS; ++n) {
        int dust_id = n;
        int rho_id  = 4*dust_id;
        int v1_id   = rho_id + 1;
        int v2_id   = rho_id + 2;
        int v3_id   = rho_id + 3;
#pragma omp simd
        for (int i=il; i<=iu; ++i) {
          Real x1 = pmb->pcoord->x1v(i);
          GetCylCoordInSpherical(pmb->pcoord, rad_arr(i), phi_arr(i), z_arr(i), i, j, k); // convert to cylindrical coordinates

          if (rad_arr(i) <= radius_inner_damping) {
            // See de Val-Borro et al. 2006 & 2007
            omega_dyn(i)       = std::sqrt(gm0/(rad_arr(i)*rad_arr(i)*rad_arr(i)));
            R_func(i)          = SQR((rad_arr(i) - radius_inner_damping)*inv_inner_damp);
            inv_damping_tau(i) = (damping_rate*omega_dyn(i));

            Real cs_square_0 = PoverRho(rad_arr(i), phi_arr(i), z_arr(i));
            Real pre_diff    = (qvalue + pvalue)*cs_square_0;

            Real vel_K           = vK(pmb->porb, x1, x2, x3);
            Real vel_dust_cyl    = weight_dust[dust_id]*pre_diff/(2.0*rad_arr(i)*omega_dyn(i));
            Real vel_dust_phi    = VelProfileCyl_dust(rad_arr(i), phi_arr(i), z_arr(i));
            vel_dust_phi        -= orb_defined*vel_K;
            Real vel_dust_r      = vel_dust_cyl*std::sin(x2);
            Real vel_dust_theta  = vel_dust_cyl*std::cos(x2);

            Real dust_rho_0  = DenProfileCyl_dust(rad_arr(i), phi_arr(i), z_arr(i), initial_D2G[dust_id], Hratio[dust_id]);
            Real dust_vel1_0 = vel_dust_r;
            Real dust_vel2_0 = vel_dust_theta;
            Real dust_vel3_0 = vel_dust_phi;

            Real &dust_rho  = prim_df(rho_id, k, j, i);
            Real &dust_vel1 = prim_df(v1_id,  k, j, i);
            Real &dust_vel2 = prim_df(v2_id,  k, j, i);
            Real &dust_vel3 = prim_df(v3_id,  k, j, i);

            Real &dust_dens = cons_df(rho_id, k, j, i);
            Real &dust_mom1 = cons_df(v1_id,  k, j, i);
            Real &dust_mom2 = cons_df(v2_id,  k, j, i);
            Real &dust_mom3 = cons_df(v3_id,  k, j, i);

            Real delta_dust_rho  = (dust_rho_0  - dust_rho )*R_func(i)*inv_damping_tau(i)*dt;
            Real delta_dust_vel1 = (dust_vel1_0 - dust_vel1)*R_func(i)*inv_damping_tau(i)*dt;
            Real delta_dust_vel2 = (dust_vel2_0 - dust_vel2)*R_func(i)*inv_damping_tau(i)*dt;
            Real delta_dust_vel3 = (dust_vel3_0 - dust_vel3)*R_func(i)*inv_damping_tau(i)*dt;

            dust_rho  += delta_dust_rho;
            dust_vel1 += delta_dust_vel1;
            dust_vel2 += delta_dust_vel2;
            dust_vel3 += delta_dust_vel3;

            dust_dens = dust_rho;
            dust_mom1 = dust_dens*dust_vel1;
            dust_mom2 = dust_dens*dust_vel2;
            dust_mom3 = dust_dens*dust_vel3;
          }
        }
      }
    }
  }
  return;
}


void OuterWaveDampingDust(MeshBlock *pmb, const Real time, const Real dt, int il, int iu,
    int jl, int ju, int kl, int ku, AthenaArray<Real> &prim_df, AthenaArray<Real> &cons_df) {

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
      for (int n=0; n<NDUSTFLUIDS; ++n) {
        int dust_id = n;
        int rho_id  = 4*dust_id;
        int v1_id   = rho_id + 1;
        int v2_id   = rho_id + 2;
        int v3_id   = rho_id + 3;
#pragma omp simd
        for (int i=il; i<=iu; ++i) {
          Real x1 = pmb->pcoord->x1v(i);
          GetCylCoordInSpherical(pmb->pcoord, rad_arr(i), phi_arr(i), z_arr(i), i, j, k); // convert to cylindrical coordinates

          if (rad_arr(i) >= radius_outer_damping) {
            // See de Val-Borro et al. 2006 & 2007
            omega_dyn(i)       = std::sqrt(gm0/(rad_arr(i)*rad_arr(i)*rad_arr(i)));
            R_func(i)          = SQR((rad_arr(i) - radius_outer_damping)*inv_outer_damp);
            inv_damping_tau(i) = (damping_rate*omega_dyn(i));

            Real cs_square_0 = PoverRho(rad_arr(i), phi_arr(i), z_arr(i));
            Real pre_diff    = (qvalue + pvalue)*cs_square_0;

            Real vel_K           = vK(pmb->porb, x1, x2, x3);
            Real vel_dust_cyl    = weight_dust[dust_id]*pre_diff/(2.0*rad_arr(i)*omega_dyn(i));
            Real vel_dust_phi    = VelProfileCyl_dust(rad_arr(i), phi_arr(i), z_arr(i));
            vel_dust_phi        -= orb_defined*vel_K;
            Real vel_dust_r      = vel_dust_cyl*std::sin(x2);
            Real vel_dust_theta  = vel_dust_cyl*std::cos(x2);

            Real dust_rho_0  = DenProfileCyl_dust(rad_arr(i), phi_arr(i), z_arr(i), initial_D2G[dust_id], Hratio[dust_id]);
            Real dust_vel1_0 = vel_dust_r;
            Real dust_vel2_0 = vel_dust_theta;
            Real dust_vel3_0 = vel_dust_phi;

            Real &dust_rho  = prim_df(rho_id, k, j, i);
            Real &dust_vel1 = prim_df(v1_id,  k, j, i);
            Real &dust_vel2 = prim_df(v2_id,  k, j, i);
            Real &dust_vel3 = prim_df(v3_id,  k, j, i);

            Real &dust_dens = cons_df(rho_id, k, j, i);
            Real &dust_mom1 = cons_df(v1_id,  k, j, i);
            Real &dust_mom2 = cons_df(v2_id,  k, j, i);
            Real &dust_mom3 = cons_df(v3_id,  k, j, i);

            Real delta_dust_rho  = (dust_rho_0  - dust_rho )*R_func(i)*inv_damping_tau(i)*dt;
            Real delta_dust_vel1 = (dust_vel1_0 - dust_vel1)*R_func(i)*inv_damping_tau(i)*dt;
            Real delta_dust_vel2 = (dust_vel2_0 - dust_vel2)*R_func(i)*inv_damping_tau(i)*dt;
            Real delta_dust_vel3 = (dust_vel3_0 - dust_vel3)*R_func(i)*inv_damping_tau(i)*dt;

            dust_rho  += delta_dust_rho;
            dust_vel1 += delta_dust_vel1;
            dust_vel2 += delta_dust_vel2;
            dust_vel3 += delta_dust_vel3;

            dust_dens = dust_rho;
            dust_mom1 = dust_dens*dust_vel1;
            dust_mom2 = dust_dens*dust_vel2;
            dust_mom3 = dust_dens*dust_vel3;
          }
        }
      }
    }
  }
  return;
}


void FixedDust(MeshBlock *pmb, int il, int iu, int jl, int ju, int kl, int ku,
              const AthenaArray<Real> &w, AthenaArray<Real> &prim_df,
              const AthenaArray<Real> &u, AthenaArray<Real> &cons_df) {

  int nc1 = pmb->ncells1;
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;

  Real orb_defined;
  (pmb->porb->orbital_advection_defined) ? orb_defined = 1.0 : orb_defined = 0.0;

  AthenaArray<Real> rad_arr, phi_arr, z_arr, cs_square, vel_K, pre_diff,
  den_dust, vel_dust_cyl, vel_dust_phi, omega_dyn, vel_dust_r, vel_dust_theta;

  rad_arr.NewAthenaArray(nc1);
  phi_arr.NewAthenaArray(nc1);
  z_arr.NewAthenaArray(nc1);
  cs_square.NewAthenaArray(nc1);
  vel_K.NewAthenaArray(nc1);
  pre_diff.NewAthenaArray(nc1);
  den_dust.NewAthenaArray(nc1);
  vel_dust_cyl.NewAthenaArray(nc1);
  vel_dust_r.NewAthenaArray(nc1);
  vel_dust_phi.NewAthenaArray(nc1);
  vel_dust_theta.NewAthenaArray(nc1);
  omega_dyn.NewAthenaArray(nc1);

  for (int k=kl; k<=ku; ++k) {
    Real x3 = pmb->pcoord->x3v(k);
    for (int j=jl; j<=ju; ++j) {
      Real x2 = pmb->pcoord->x2v(j);
#pragma omp simd
      for (int i=il; i<=iu; ++i) {
        Real x1 = pmb->pcoord->x1v(i);
        GetCylCoordInSpherical(pmb->pcoord, rad_arr(i), phi_arr(i), z_arr(i), i, j, k);

        cs_square(i) = PoverRho(rad_arr(i), phi_arr(i), z_arr(i));
        vel_K(i)     = vK(pmb->porb, x1, x2, x3);
        pre_diff(i)  = (qvalue + pvalue)*cs_square(i);
        omega_dyn(i) = std::sqrt(gm0/(rad_arr(i)*rad_arr(i)*rad_arr(i)));
      }

      if (NDUSTFLUIDS > 0) {
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
            const Real &gas_rho = w(IDN, k, j, i);

            den_dust(i)        = initial_D2G[dust_id]*gas_rho;
            vel_dust_cyl(i)    = weight_dust[dust_id]*pre_diff(i)/(2.0*rad_arr(i)*omega_dyn(i));
            vel_dust_phi(i)    = VelProfileCyl_dust(rad_arr(i), phi_arr(i), z_arr(i));
            vel_dust_phi(i)   -= orb_defined*vel_K(i);
            vel_dust_r(i)      = vel_dust_cyl(i)*std::sin(x2);
            vel_dust_theta(i)  = vel_dust_cyl(i)*std::cos(x2);

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
            dust_vel2 = vel_dust_theta(i);
            dust_vel3 = vel_dust_phi(i);

            dust_dens = den_dust(i);
            dust_mom1 = den_dust(i)*vel_dust_r(i);
            dust_mom2 = den_dust(i)*vel_dust_theta(i);
            dust_mom3 = den_dust(i)*vel_dust_phi(i);
          }
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

  int kl = ks - dk;
  int ku = ke + dk;
  int jl = js - NGHOST;
  int ju = je + NGHOST;
  int il = is - NGHOST;
  int iu = ie + NGHOST;

  if (Inner_Gas_Damping_Flag)
    InnerWaveDampingGas(this, time, dt, il, iu, jl, ju, kl, ku, phydro->w, phydro->u);

  if (Outer_Gas_Damping_Flag)
    OuterWaveDampingGas(this, time, dt, il, iu, jl, ju, kl, ku, phydro->w, phydro->u);

  if (Theta_Gas_Damping_Flag) {
    UpperWaveDampingGas(this, time, dt, il, iu, jl, ju, kl, ku, phydro->w, phydro->u);
    LowerWaveDampingGas(this, time, dt, il, iu, jl, ju, kl, ku, phydro->w, phydro->u);
  }

  if (NDUSTFLUIDS > 0) {
    if (Inner_Dust_Damping_Flag)
      InnerWaveDampingDust(this, time, dt, il, iu, jl, ju, kl, ku,
          pdustfluids->df_prim, pdustfluids->df_cons);

    if (Outer_Dust_Damping_Flag)
      OuterWaveDampingDust(this, time, dt, il, iu, jl, ju, kl, ku,
          pdustfluids->df_prim, pdustfluids->df_cons);
  }

  if (Isothermal_Flag)
    LocalIsothermalEOS(this, il, iu, jl, ju, kl, ku, phydro->w, phydro->u);

  if ((time_drag != 0.0) && (time < time_drag))
    FixedDust(this, il, iu, jl, ju, kl, ku, phydro->w, pdustfluids->df_prim,
                                            phydro->u, pdustfluids->df_cons);

  // User-defined output
   //Accumulate data
  AccumulateData(this);

  if (((time+dt >= next_time) && (time < next_time))
    || ((time+dt >= pmy_mesh->tlim) && (curr_time < time+dt))) {
    file_number++;
    curr_time  = time+dt;
    next_time += edt;
    dowrite    = 1;
  }

  if ((time+dt > curr_time) && (time+dt < next_time) && (dowrite == 1)) {
    dowrite = 0;
    prev_time = curr_time;
  }

  if (dowrite == 1) {
    int current_level = loc.level - pmy_mesh->root_level;
    int dlevel = current_level - out_level;

    DoOutput(this, dlevel);

// clean array
    for (int n=n0; n<n0+nvar; ++n) {
      for (int j=0; j<=block_size.nx2; ++j) {
#pragma simd
        for (int i=0; i<=block_size.nx1; ++i) {
          ruser_meshblock_data[n](j, i)=0.0;
        }
      }
    }
  }
  return;
}


//void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin)
//{
  //for(int j=js; j<=je; j++) {
    //for(int i=is; i<=ie; i++) {
      //user_out_var_2D(0, j, i) = phydro->w(IDN, 0, j, i);
    //}
  //}

  //int index = (je + js)/2;

  //for(int i=is; i<=ie; i++) {
    //user_out_var_1D(0, i) = phydro->w(IDN, 0, index, i);
  //}
  //return;
//}
