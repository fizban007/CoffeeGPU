#ifndef _CONSTANT_MEM_H_
#define _CONSTANT_MEM_H_

#include "sim_params.h"
#include "data/grid.h"

namespace Coffee {

// This is the simulation parameters in constant memory
extern __device__ __constant__ sim_params dev_params;

// This is the grid parameters in constant memory
extern __device__ __constant__ Grid dev_grid;

// Copy a given parameter struct to constant memory. This can be used to update
// the parameters on device even after initializing the simulation
void init_dev_params(const sim_params& params);

// Copy a given grid configuration to constant memory.
void init_dev_grid(const Grid& g);

}  // namespace Coffee

#endif  // _CONSTANT_MEM_H_
