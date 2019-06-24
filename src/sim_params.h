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
};


/// Reads a toml config file, parses the above data structure, and returns it.
sim_params parse_config(const std::string& filename);

}

#endif  // _SIM_PARAMS_H_
