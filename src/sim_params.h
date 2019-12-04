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
  Scalar ch2 = 1.0; ///< Dedner wave speed (squared)
  Scalar tau = 0.02; ///< Dedner damping time
  Scalar KOeps = 0.50; ///< Small parameter in Kreiss-Oliger dissipation (range [0-1])

  // Pulsar problem
  Scalar radius = 10.0; ///< Radius of the pulsar
  Scalar omega = 0.001; ///< Angular velocity of the pulsar
  Scalar b0 = 1e3; ///< Magnetic field at stellar surface

  // Pulsar with Alfven wave perturbation
  Scalar tp_start = 0.0; ///< Time for the perturbation to start
  Scalar tp_end = 0.5; ///< Time for the perturbation to end
  Scalar rpert = 10.0; ///< Perturbation range
  Scalar dw0 = 0.001; ///< Amplitude of the angular velocity perturbation

};

/// Reads a toml config file, parses the above data structure, and returns it.
sim_params parse_config(const std::string& filename);

}

#endif  // _SIM_PARAMS_H_
