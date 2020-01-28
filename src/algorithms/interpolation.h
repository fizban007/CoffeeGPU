#include "cuda/cuda_control.h"
#include "data/stagger.h"
#include "data/typedefs.h"
#include "data/vec3.h"
#include <cstdint>

namespace Coffee {

template <typename T>
HOST_DEVICE T
interpolate(T* f, std::size_t idx_lin, Stagger in, Stagger out,
            int dim0, int dim1) {
  int di_m = (in[0] == out[0] ? 0 : -out[0]);
  int di_p = (in[0] == out[0] ? 0 : 1 - out[0]);
  int dj_m = (in[1] == out[1] ? 0 : -out[1]);
  int dj_p = (in[1] == out[1] ? 0 : 1 - out[1]);
  int dk_m = (in[2] == out[2] ? 0 : -out[2]);
  int dk_p = (in[2] == out[2] ? 0 : 1 - out[2]);

  Scalar f11 =
      0.5 * (f[idx_lin + di_p + dj_p * dim0 + dk_m * dim0 * dim1] +
             f[idx_lin + di_p + dj_p * dim0 + dk_p * dim0 * dim1]);
  Scalar f10 =
      0.5 * (f[idx_lin + di_p + dj_m * dim0 + dk_m * dim0 * dim1] +
             f[idx_lin + di_p + dj_m * dim0 + dk_p * dim0 * dim1]);
  Scalar f01 =
      0.5 * (f[idx_lin + di_m + dj_p * dim0 + dk_m * dim0 * dim1] +
             f[idx_lin + di_m + dj_p * dim0 + dk_p * dim0 * dim1]);
  Scalar f00 =
      0.5 * (f[idx_lin + di_m + dj_m * dim0 + dk_m * dim0 * dim1] +
             f[idx_lin + di_m + dj_m * dim0 + dk_p * dim0 * dim1]);
  Scalar f1 = 0.5 * (f11 + f10);
  Scalar f0 = 0.5 * (f01 + f00);
  return 0.5 * (f1 + f0);
}

template <typename T>
HOST_DEVICE T
interpolate2d(T* f, std::size_t idx_lin, Stagger in, Stagger out,
              int dim0) {
  int di_m = (in[0] == out[0] ? 0 : -out[0]);
  int di_p = (in[0] == out[0] ? 0 : 1 - out[0]);
  int dj_m = (in[1] == out[1] ? 0 : -out[1]);
  int dj_p = (in[1] == out[1] ? 0 : 1 - out[1]);

  Scalar f1 = 0.5 * (f[idx_lin + di_p + dj_p * dim0] +
                     f[idx_lin + di_p + dj_m * dim0]);
  Scalar f0 = 0.5 * (f[idx_lin + di_m + dj_p * dim0] +
                     f[idx_lin + di_m + dj_m * dim0]);
  return 0.5 * (f1 + f0);
}

template <typename T>
HOST_DEVICE T
interpolate(T* f, std::size_t idx_lin, Vec3<float> out, int dim0,
            int dim1) {
  int di_m = 0;
  int di_p = 1;
  int dj_m = 0;
  int dj_p = 1;
  int dk_m = 0;
  int dk_p = 1;

  Scalar f11 =
      (1.0f - out.z) *
          f[idx_lin + di_p + dj_p * dim0 + dk_m * dim0 * dim1] +
      out.z * f[idx_lin + di_p + dj_p * dim0 + dk_p * dim0 * dim1];
  Scalar f10 =
      (1.0f - out.z) *
          f[idx_lin + di_p + dj_m * dim0 + dk_m * dim0 * dim1] +
      out.z * f[idx_lin + di_p + dj_m * dim0 + dk_p * dim0 * dim1];
  Scalar f01 =
      (1.0f - out.z) *
          f[idx_lin + di_m + dj_p * dim0 + dk_m * dim0 * dim1] +
      out.z * f[idx_lin + di_m + dj_p * dim0 + dk_p * dim0 * dim1];
  Scalar f00 =
      (1.0f - out.z) *
          f[idx_lin + di_m + dj_m * dim0 + dk_m * dim0 * dim1] +
      out.z * f[idx_lin + di_m + dj_m * dim0 + dk_p * dim0 * dim1];
  Scalar f1 = out.y * f11 + (1.0f - out.y) * f10;
  Scalar f0 = out.y * f01 + (1.0f - out.y) * f00;
  return out.x * f1 + (1.0f - out.x) * f0;
}

template <typename T>
HOST_DEVICE T
interpolate(T* f, Index idx, Stagger in, Stagger out, int dim0,
            int dim1) {
  std::size_t idx_lin = idx.x + idx.y * dim0 + idx.z * dim0 * dim1;
  return interpolate(f, idx_lin, in, out, dim0, dim1);
}

template <int DIM1, int DIM0, typename T>
HOST_DEVICE T
interpolate(T (*f)[DIM1][DIM0], int i, int j, int k, Stagger in,
            Stagger out) {
  int di_m = (in[0] == out[0] ? 0 : -out[0]);
  int di_p = (in[0] == out[0] ? 0 : 1 - out[0]);
  int dj_m = (in[1] == out[1] ? 0 : -out[1]);
  int dj_p = (in[1] == out[1] ? 0 : 1 - out[1]);
  int dk_m = (in[2] == out[2] ? 0 : -out[2]);
  int dk_p = (in[2] == out[2] ? 0 : 1 - out[2]);

  Scalar f11 = 0.5 * (f[k + dk_p][j + dj_p][i + di_p] +
                      f[k + dk_m][j + dj_p][i + di_p]);
  Scalar f10 = 0.5 * (f[k + dk_p][j + dj_m][i + di_p] +
                      f[k + dk_m][j + dj_m][i + di_p]);
  Scalar f01 = 0.5 * (f[k + dk_p][j + dj_p][i + di_m] +
                      f[k + dk_m][j + dj_p][i + di_m]);
  Scalar f00 = 0.5 * (f[k + dk_p][j + dj_m][i + di_m] +
                      f[k + dk_m][j + dj_m][i + di_m]);
  Scalar f1 = 0.5 * (f11 + f10);
  Scalar f0 = 0.5 * (f01 + f00);
  return 0.5 * (f1 + f0);
}

}  // namespace Coffee
