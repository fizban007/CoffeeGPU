#ifndef _CONSTANT_MEM_H_
#define _CONSTANT_MEM_H_

#include "sim_params.h"
#include "data/grid.h"

namespace Coffee {

// This is the simulation parameters in constant memory
extern __device__ __constant__ sim_params dev_params;

// This is the grid parameters in constant memory
extern __device__ __constant__ Grid dev_grid;

}  // namespace Coffee

#endif  // _CONSTANT_MEM_H_
