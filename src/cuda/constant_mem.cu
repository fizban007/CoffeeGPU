#include "cuda/constant_mem.h"
#include "cuda/cuda_utility.h"

namespace Coffee {

__constant__ sim_params dev_params;

void
init_dev_params(const sim_params& params) {
  const sim_params* p = &params;
  CudaSafeCall(cudaMemcpyToSymbol(dev_params, (void*)p,
                                  sizeof(sim_params)));
}



}
