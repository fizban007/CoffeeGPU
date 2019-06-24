#include "data/typedefs.h"

namespace Coffee {

__host__ __device__ Scalar interpolate(Scalar* f, Index idx, Stagger in, Stagger out) {

  int di_m = (in[0] == out[0] ? 0 : - out[0]);
  int di_p = (in[0] == out[0] ? 0 : 1 - out[0]);
  int dj_m = (in[1] == out[1] ? 0 : - out[1]);
  int dj_p = (in[1] == out[1] ? 0 : 1 - out[1]);
  int dk_m = (in[2] == out[2] ? 0 : - out[2]);
  int dk_p = (in[2] == out[2] ? 0 : 1 - out[2]);
  int idx_lin = idx.x + idx.y * dev_grid.dims[0] + idx.z * dev_grid.dims[0] * dev_grid.dims[1]

  Scalar f11 = 0.5 * (f[idx_lin + di_p + dj_p * dev_grid.dims[0] + dk_m * dev_grid.dims[0] * dev_grid.dims[1]]
                     + f[idx_lin + di_p + dj_p * dev_grid.dims[0] + dk_p * dev_grid.dims[0] * dev_grid.dims[1]]);
  Scalar f10 = 0.5 * (f[idx_lin + di_p + dj_m * dev_grid.dims[0] + dk_m * dev_grid.dims[0] * dev_grid.dims[1]]
                     + f[idx_lin + di_p + dj_m * dev_grid.dims[0] + dk_p * dev_grid.dims[0] * dev_grid.dims[1]]);
  Scalar f01 = 0.5 * (f[idx_lin + di_m + dj_p * dev_grid.dims[0] + dk_m * dev_grid.dims[0] * dev_grid.dims[1]]
                     + f[idx_lin + di_m + dj_p * dev_grid.dims[0] + dk_p * dev_grid.dims[0] * dev_grid.dims[1]]);
  Scalar f00 = 0.5 * (f[idx_lin + di_m + dj_m * dev_grid.dims[0] + dk_m * dev_grid.dims[0] * dev_grid.dims[1]]
                     + f[idx_lin + di_m + dj_m * dev_grid.dims[0] + dk_p * dev_grid.dims[0] * dev_grid.dims[1]]);
  Scalar f1 = 0.5 * (f11 + f10);
  Scalar f0 = 0.5 * (f01 + f00);
  return 0.5 * (f1 + f0);

}

}
