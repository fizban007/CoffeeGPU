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

  // Problem specific
  bool vacuum = true; ///< Whether the evolution is in vacuum or not
  bool resistive =true; ///< Whether to use resistive formalism or not
  // Parameter for resistive formalism
  Scalar sigsq = 1.0; ///< Conductivity squared
  int subsamp = 10; ///< Number of substeps in the stiff part
  // disk setup
  Scalar r1 = 1.0; ///< Radius of the central compact object
  Scalar r2 = 1.44; ///< Inner boundary of the accretion disk
  Scalar eta = 0.0; ///< Resistivity of the membrane
  Scalar omega0 = 0.48; ///< Angular velocity of the central compact object
  Scalar omegad0 = 0.0; ///< Angular velocity of the inner boundary of the accretion disk
  // Initial current distribution in the disk
  Scalar j0 = 1000.0; ///< Magnitude of the initial current in the disk
  Scalar wid = 5.0; ///< Disk current distribution length scale
  Scalar alpha = 1.0; ///< How fast current decreases with radius
  int vacstep = 1000; ///< Number of time steps for vacuum evolution
  // Absorbing layer
  int pml[3] = {60, 60, 60}; ///< Number of cells in absorbing layer at each boundary 
  int pmllen = 10; ///< Absorbing layer resistivety variation length scale in terms of cells
  Scalar sigpml = 0.1; ///< Parameter for the resistivity in the absorbing layer
};


/// Reads a toml config file, parses the above data structure, and returns it.
sim_params parse_config(const std::string& filename);

}

#endif  // _SIM_PARAMS_H_
