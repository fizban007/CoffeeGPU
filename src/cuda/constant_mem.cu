#include "cuda/constant_mem.h"
#include "cuda/constant_mem_func.h"
#include "cuda/cuda_utility.h"

namespace Coffee {

__constant__ sim_params dev_params;
__constant__ Grid dev_grid;

void
init_dev_params(const sim_params& params) {
  const sim_params* p = &params;
  CudaSafeCall(
      cudaMemcpyToSymbol(dev_params, (void*)p, sizeof(sim_params)));
}

void
init_dev_grid(const Grid& g) {
  const Grid* p = &g;
  CudaSafeCall(cudaMemcpyToSymbol(dev_grid, (void*)p, sizeof(Grid)));
}

}  // namespace Coffee
