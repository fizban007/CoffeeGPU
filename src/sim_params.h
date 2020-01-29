#ifndef _SIM_PARAMS_H_
#define _SIM_PARAMS_H_

#include <string>
#include "data/typedefs.h"

namespace Coffee {

////////////////////////////////////////////////////////////////////////////////
///  This is the parameter class that contains all of the global simulation
///  parameters. This structure will be copied to the GPU and will be accessible
///  to all device kernels.
////////////////////////////////////////////////////////////////////////////////
struct sim_params {
  // Simulation parameters
  Scalar dt = 0.01;   ///< Time step size in numerical units
  uint64_t max_steps = 10000; ///< Total number of timesteps
  int data_interval = 100; ///< How many steps between data outputs
  bool periodic_boundary[3] = {false}; ///< Periodic boundary condition
  int downsample = 1; ///< How much to downsample from simulation grid to data
                      ///output grid

  // Grid parameters
  int N[3] = {1};  ///< Number of cells in each direction (excluding guard cells)
  int guard[3] = {0}; ///< Number of guard cells at each boundary
  Scalar lower[3] = {0.0}; ///< Lower limits of the simulation box
  Scalar size[3] = {1.0}; ///< Sizes of the simulation box
  int nodes[3] = {1}; ///< Number of nodes in all directions
  int shift_ghost = 0; ///< Number of ghost cells to be updated in field_solver

  // Problem specific
  int kn[3] = {1, 1, 0};

  // GR parameters
  Scalar a = 0.99; ///<Black hole spin

  bool calc_current = true; ///< Whether to calculate force-free current or not
  bool clean_ep = true; ///< Whether to clean E_parallel or not
  bool check_egb = true; ///< Whether to clean E>B or not

  // Absorbing layer
  int pml[3] = {60, 60, 60}; ///< Number of cells in absorbing layer at each boundary 
  int pmllen = 10; ///< Absorbing layer resistivety variation length scale in terms of cells
  Scalar sigpml = 0.1; ///< Parameter for the resistivity in the absorbing layer

  // EZ scheme
  bool divB_clean = true; ///< Whether or not to do divergence cleaning
  Scalar ch2 = 1.0; ///< Dedner wave speed (squared)
  Scalar tau = 0.02; ///< Dedner damping time
  Scalar KOeps = 0.50; ///< Small parameter in Kreiss-Oliger dissipation (range [0-1])
  bool KO_geometry = true; ///< Whether or not to include geometric factors in KO

  // Pulsar problem
  bool pulsar = true; ///< Whether or not to use pulsar setup
  Scalar radius = 10.0; ///< Radius of the pulsar
  Scalar omega = 0.001; ///< Angular velocity of the pulsar
  Scalar b0 = 1e3; ///< Magnetic field at stellar surface
  // Scalar alpha = 0.0; ///< Inclination of the magnetic dipole moment
  // Dipole parameters
  Scalar p1 = 0.0;
  Scalar p2 = 0.0;
  Scalar p3 = 1.0;
  // Quadrupole component
  Scalar q11 = 0.0;
  Scalar q12 = 0.0;
  Scalar q13 = 0.0;
  Scalar q22 = 0.0;
  Scalar q23 = 0.0;
  Scalar q_offset_x = 0.0;
  Scalar q_offset_y = 0.0;
  Scalar q_offset_z = 0.0;


  // Pulsar with Alfven wave perturbation
  Scalar tp_start = 0.0; ///< Time for the perturbation to start
  Scalar tp_end = 0.5; ///< Time for the perturbation to end
  Scalar rpert1 = 5.0; ///< Perturbation range
  Scalar rpert2 = 10.0; ///< Perturbation range
  Scalar dw0 = 0.001; ///< Amplitude of the angular velocity perturbation

  int skymap_Nth = 256;
  int skymap_Nph = 512;
};

/// Reads a toml config file, parses the above data structure, and returns it.
sim_params parse_config(const std::string& filename);

}

#endif  // _SIM_PARAMS_H_
