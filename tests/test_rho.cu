#include "catch.hpp"
#include "algorithms/field_solver.h"
#include "cuda/constant_mem_func.h"
#include "cuda/constant_mem.h"
#include "data/typedefs.h"
#include "data/vec3.h"
#include "data/grid.h"

using namespace Coffee;

namespace Coffee {
// Load the rho kernel
__global__ void
kernel_compute_rho(const Scalar *ex, const Scalar *ey, const Scalar *ez,
                   Scalar *rho);
}

// Create my own grid
TEST_CASE("blah", "[a]") { 
  Grid grid;
  for (int i = 0; i < 3; ++i) {
    grid.delta[i] = 1.0;
    grid.inv_delta[i] = 1.0/grid.delta[i];
    grid.dims[i] = 10;
    grid.guard[i] = 2;
  }
  init_dev_grid(grid);
  multi_array<float> E(10, 10, 10);
  for (int k = 0; k < 10; ++k) {
    for (int j = 0; j < 10; ++j) {
      for (int i = 0; i < 10; ++i) {
        E(i, j, k) = i + j + k;
      }
    }
  }
  E.sync_to_device();
  multi_array<float> Rho(10, 10, 10);
  
  dim3 gridSize(8, 16, 16);
  dim3 blockSize(32, 4, 4);

  kernel_compute_rho<<<gridSize, blockSize>>>(E.dev_ptr(),E.dev_ptr(),E.dev_ptr(),Rho.dev_ptr());
  Rho.sync_to_host();

  for (int k = 2; k < 8; k++) {
    for (int j = 2; j < 8; j++) {
      for (int i = 2; i < 8; i++) {
        CHECK(Rho(i,j,k) == Approx(3.0f));

      }
    }
  }

}

