#ifndef _CONSTANT_MEM_H_
#define _CONSTANT_MEM_H_

#include "sim_params.h"

namespace Coffee {

// This is the simulation parameters in constant memory
extern __device__ __constant__ sim_params dev_params;


}

#endif  // _CONSTANT_MEM_H_
