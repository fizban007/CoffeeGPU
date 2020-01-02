#include "cuda/constant_mem.h"
#include "cuda/constant_mem_func.h"
#include "cuda/cuda_utility.h"
#include "cuda/cuda_control.h"
#include "field_solver_gr_EZ.h"
#include "interpolation.h"
#include "metric.h"
#include "utils/timer.h"
#include "utils/nvproftool.h"

#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 2
#define BLOCK_SIZE_Z 2


#define TINY 1e-7

namespace Coffee {

// static dim3 gridSize(8, 16, 16);
static dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);

static dim3 blockGroupSize;

HOST_DEVICE Scalar get_R2(Scalar x, Scalar y, Scalar z) {
  return x * x + y * y + z * z;
}

HOST_DEVICE Scalar get_r(Scalar a, Scalar x, Scalar y, Scalar z) {
  Scalar R2 = get_R2(x, y, z);
  return std::sqrt((R2 - a * a + std::sqrt(square(R2 - a * a) + 4.0 * square(a * z) + TINY)) / 2.0);
}

HOST_DEVICE Scalar get_g() { return -1.0; }

HOST_DEVICE Scalar get_beta_d1(Scalar a, Scalar x, Scalar y, Scalar z) {
  Scalar r = get_r(a, x, y, z);
  return 2.0 * r * r * r * (r * x + a * y) / (a * a + r * r + TINY) / (r * r * r * r + a * a * z * z + TINY);
}

HOST_DEVICE Scalar get_beta_d2(Scalar a, Scalar x, Scalar y, Scalar z) {
  Scalar r = get_r(a, x, y, z);
  return 2.0 * r * r * r * (- a * x + r * y) / (a * a + r * r + TINY) / (r * r * r * r + a * a * z * z + TINY);
}

HOST_DEVICE Scalar get_beta_d3(Scalar a, Scalar x, Scalar y, Scalar z) {
  Scalar r = get_r(a, x, y, z);
  return 2.0 * r * r * z / (r * r * r * r + a * a * z * z + TINY);
}

HOST_DEVICE Scalar get_beta_u1(Scalar a, Scalar x, Scalar y, Scalar z) {
  Scalar r = get_r(a, x, y, z);
  return 2.0 * r * r * r * (r * x + a * y) / (a * a + r * r + TINY) / (2.0 * r * r * r + r * r * r * r + a * a * z * z + TINY);
}

HOST_DEVICE Scalar get_beta_u2(Scalar a, Scalar x, Scalar y, Scalar z) {
  Scalar r = get_r(a, x, y, z);
  return 2.0 * r * r * r * (- a * x + r * y) / (a * a + r * r + TINY) / (2.0 * r * r * r + r * r * r * r + a * a * z * z + TINY);
}

HOST_DEVICE Scalar get_beta_u3(Scalar a, Scalar x, Scalar y, Scalar z) {
  Scalar r = get_r(a, x, y, z);
  return 2.0 * r * r * z / (2.0 * r * r * r + r * r * r * r + a * a * z * z + TINY);
}

HOST_DEVICE Scalar get_gamma(Scalar a, Scalar x, Scalar y, Scalar z) {
  Scalar r = get_r(a, x, y, z);
  return 1.0 + 2.0 * r * r * r / (r * r * r * r + a * a * z * z + TINY);
}

HOST_DEVICE Scalar get_sqrt_gamma(Scalar a, Scalar x, Scalar y, Scalar z) {
  return std::sqrt(get_gamma(a, x, y, z));
}

HOST_DEVICE Scalar get_alpha(Scalar a, Scalar x, Scalar y, Scalar z) {
  Scalar r = get_r(a, x, y, z);
  return std::sqrt(1.0 / (1.0 + 2.0 * r * r * r / (r * r * r * r + a * a * z * z + TINY)));
}

HOST_DEVICE Scalar get_gamma_d11(Scalar a, Scalar x, Scalar y, Scalar z) {
  Scalar r = get_r(a, x, y, z);
  return 1.0 + 2.0 * r * r * r * square(r * x + a * y) / square(a * a + r * r + TINY) / (r * r * r * r + a * a * z * z + TINY);
}

HOST_DEVICE Scalar get_gamma_d12(Scalar a, Scalar x, Scalar y, Scalar z) {
  Scalar r = get_r(a, x, y, z);
  return 2.0 * r * r * r * (r * x + a * y) * (- a * x + r * y) / square(a * a + r * r + TINY) / (r * r * r * r + a * a * z * z + TINY);
}

HOST_DEVICE Scalar get_gamma_d13(Scalar a, Scalar x, Scalar y, Scalar z) {
  Scalar r = get_r(a, x, y, z);
  return 2.0 * r * r * (r * x + a * y) * z / (a * a + r * r + TINY) / (r * r * r * r + a * a * z * z + TINY);
}

HOST_DEVICE Scalar get_gamma_d22(Scalar a, Scalar x, Scalar y, Scalar z) {
  Scalar r = get_r(a, x, y, z);
  return 1.0 + 2.0 * r * r * r * square(a * x - r * y) / square(a * a + r * r + TINY) / (r * r * r * r + a * a * z * z + TINY);
}

HOST_DEVICE Scalar get_gamma_d23(Scalar a, Scalar x, Scalar y, Scalar z) {
  Scalar r = get_r(a, x, y, z);
  return 2.0 * r * r * (- a * x + r * y) * z / (a * a + r * r + TINY) / (r * r * r * r + a * a * z * z + TINY);
}

HOST_DEVICE Scalar get_gamma_d33(Scalar a, Scalar x, Scalar y, Scalar z) {
  Scalar r = get_r(a, x, y, z);
  return 1.0 + 2.0 * r * z * z / (r * r * r * r + a * a * z * z + TINY);
}

HOST_DEVICE Scalar get_gamma_u11(Scalar a, Scalar x, Scalar y, Scalar z) {
  Scalar r = get_r(a, x, y, z);
  return 1.0 - 2.0 * r * r * r * square(r * x + a * y) / square(a * a + r * r + TINY) / (2.0 * r * r * r + r * r * r * r + a * a * z * z + TINY);
}

HOST_DEVICE Scalar get_gamma_u12(Scalar a, Scalar x, Scalar y, Scalar z) {
  Scalar r = get_r(a, x, y, z);
  return - 2.0 * r * r * r * (r * x + a * y) * (- a * x + r * y) / square(a * a + r * r + TINY) / (2.0 * r * r * r + r * r * r * r + a * a * z * z + TINY);
}

HOST_DEVICE Scalar get_gamma_u13(Scalar a, Scalar x, Scalar y, Scalar z) {
  Scalar r = get_r(a, x, y, z);
  return - 2.0 * r * r * (r * x + a * y) * z / (a * a + r * r + TINY) / (2.0 * r * r * r + r * r * r * r + a * a * z * z + TINY);
}

HOST_DEVICE Scalar get_gamma_u22(Scalar a, Scalar x, Scalar y, Scalar z) {
  Scalar r = get_r(a, x, y, z);
  return 1.0 - 2.0 * r * r * r * square(a * x - r * y) / square(a * a + r * r + TINY) / (2.0 * r * r * r + r * r * r * r + a * a * z * z + TINY);
}

HOST_DEVICE Scalar get_gamma_u23(Scalar a, Scalar x, Scalar y, Scalar z) {
  Scalar r = get_r(a, x, y, z);
  return 2.0 * r * r * (a * x - r * y) * z / (a * a + r * r + TINY) / (2.0 * r * r * r + r * r * r * r + a * a * z * z + TINY);
}

HOST_DEVICE Scalar get_gamma_u33(Scalar a, Scalar x, Scalar y, Scalar z) {
  Scalar r = get_r(a, x, y, z);
  return 1.0 - 2.0 * r * z * z / (2.0 * r * r * r + r * r * r * r + a * a * z * z + TINY);
}

__device__ inline Scalar
diff1x4(const Scalar *f, int ijk) {
  return (f[ijk - 2] - 8 * f[ijk - 1] + 8 * f[ijk + 1] - f[ijk + 2]) /
         12.0;
}

__device__ inline Scalar
diff1y4(const Scalar *f, int ijk) {
  int s = dev_grid.dims[0];
  return (f[ijk - 2 * s] - 8 * f[ijk - 1 * s] + 8 * f[ijk + 1 * s] -
          f[ijk + 2 * s]) /
         12.0;
}

__device__ inline Scalar
diff1z4(const Scalar *f, int ijk) {
  int s = dev_grid.dims[0] * dev_grid.dims[1];
  return (f[ijk - 2 * s] - 8 * f[ijk - 1 * s] + 8 * f[ijk + 1 * s] -
          f[ijk + 2 * s]) /
         12.0;
}

__device__ inline Scalar
diff4x2(const Scalar *f, int ijk) {
  return (f[ijk - 2] - 4 * f[ijk - 1] + 6 * f[ijk] - 4 * f[ijk + 1] +
          f[ijk + 2]);
}

__device__ inline Scalar
diff4y2(const Scalar *f, int ijk) {
  int s = dev_grid.dims[0];
  return (f[ijk - 2 * s] - 4 * f[ijk - 1 * s] + 6 * f[ijk] -
          4 * f[ijk + 1 * s] + f[ijk + 2 * s]);
}

__device__ inline Scalar
diff4z2(const Scalar *f, int ijk) {
  int s = dev_grid.dims[0] * dev_grid.dims[1];
  return (f[ijk - 2 * s] - 4 * f[ijk - 1 * s] + 6 * f[ijk] -
          4 * f[ijk + 1 * s] + f[ijk + 2 * s]);
}

__device__ inline Scalar
diff6x2(const Scalar *f, int ijk) {
  return (f[ijk - 3] - 6 * f[ijk - 2] + 15 * f[ijk - 1] - 20 * f[ijk] +
          15 * f[ijk + 1] - 6 * f[ijk + 2] + f[ijk + 3]);
}

__device__ inline Scalar
diff6y2(const Scalar *f, int ijk) {
  int s = dev_grid.dims[0];
  return (f[ijk - 3 * s] - 6 * f[ijk - 2 * s] + 15 * f[ijk - 1 * s] -
          20 * f[ijk] + 15 * f[ijk + 1 * s] - 6 * f[ijk + 2 * s] +
          f[ijk + 3 * s]);
}

__device__ inline Scalar
diff6z2(const Scalar *f, int ijk) {
  int s = dev_grid.dims[0] * dev_grid.dims[1];
  return (f[ijk - 3 * s] - 6 * f[ijk - 2 * s] + 15 * f[ijk - 1 * s] -
          20 * f[ijk] + 15 * f[ijk + 1 * s] - 6 * f[ijk + 2 * s] +
          f[ijk + 3 * s]);
}

__device__ inline Scalar
dfdx(const Scalar *f, int ijk) {
  return diff1x4(f, ijk) / dev_grid.delta[0];
}

__device__ inline Scalar
dfdy(const Scalar *f, int ijk) {
  return diff1y4(f, ijk) / dev_grid.delta[1];
}

__device__ inline Scalar
dfdz(const Scalar *f, int ijk) {
  return diff1z4(f, ijk) / dev_grid.delta[2];
}

__device__ Scalar
div4(const Scalar *fx, const Scalar *fy, const Scalar *fz, int ijk,
     Scalar x, Scalar y, Scalar z) {
  Scalar tmpx =
      (fx[ijk - 2] * get_sqrt_gamma(dev_params.a,
                                    x - 2.0 * dev_grid.delta[0], y, z) -
       8 * fx[ijk - 1] *
           get_sqrt_gamma(dev_params.a, x - 1.0 * dev_grid.delta[0], y,
                          z) +
       8 * fx[ijk + 1] *
           get_sqrt_gamma(dev_params.a, x + 1.0 * dev_grid.delta[0], y,
                          z) -
       fx[ijk + 2] * get_sqrt_gamma(dev_params.a,
                                    x + 2.0 * dev_grid.delta[0], y,
                                    z)) /
      12.0 / dev_grid.delta[0];
  int s = dev_grid.dims[0];
  Scalar tmpy =
      (fy[ijk - 2 * s] * get_sqrt_gamma(dev_params.a, x,
                                       y - 2.0 * dev_grid.delta[1], z) -
       8 * fy[ijk - 1 * s] *
           get_sqrt_gamma(dev_params.a, x, y - 1.0 * dev_grid.delta[1],
                          z) +
       8 * fy[ijk + 1 * s] *
           get_sqrt_gamma(dev_params.a, x, y + 1.0 * dev_grid.delta[1],
                          z) -
       fy[ijk + 2 * s] * get_sqrt_gamma(dev_params.a, x,
                                       y + 2.0 * dev_grid.delta[1],
                                       z)) /
      12.0 / dev_grid.delta[1];
  s = dev_grid.dims[0] * dev_grid.dims[1];
  Scalar tmpz =
      (fz[ijk - 2 * s] * get_sqrt_gamma(dev_params.a, x, y,
                                       z - 2.0 * dev_grid.delta[2]) -
       8 * fz[ijk - 1 * s] *
           get_sqrt_gamma(dev_params.a, x, y,
                          z - 1.0 * dev_grid.delta[2]) +
       8 * fz[ijk + 1 * s] *
           get_sqrt_gamma(dev_params.a, x, y,
                          z + 1.0 * dev_grid.delta[2]) -
       fz[ijk + 2 * s] * get_sqrt_gamma(dev_params.a, x, y,
                                       z + 2.0 * dev_grid.delta[2])) /
      12.0 / dev_grid.delta[2];
  return (tmpx + tmpy + tmpz) / get_sqrt_gamma(dev_params.a, x, y, z);
}

__device__ inline Scalar
KO(const Scalar *f, int ijk) {
  if (FFE_DISSIPATION_ORDER == 4)
    return diff4x2(f, ijk) + diff4y2(f, ijk) + diff4z2(f, ijk);
  if (FFE_DISSIPATION_ORDER == 6)
    return diff6x2(f, ijk) + diff6y2(f, ijk) + diff6z2(f, ijk);
}

__global__ void
kernel_compute_E_gr_thread(const Scalar *Dx, const Scalar *Dy, const Scalar *Dz,
                          const Scalar *Bx, const Scalar *By, const Scalar *Bz,
                          Scalar *Ex, Scalar *Ey, Scalar *Ez, int shift) {
  size_t ijk;
  Scalar Ddx, Ddy, Ddz;
  Scalar x, y, z;

  int i = threadIdx.x + blockIdx.x * blockDim.x + dev_grid.guard[0] - shift;
  int j = threadIdx.y + blockIdx.y * blockDim.y + dev_grid.guard[1] - shift;
  int k = threadIdx.z + blockIdx.z * blockDim.z + dev_grid.guard[2] - shift;
  if (i < dev_grid.dims[0] - dev_grid.guard[0] + shift &&
      j < dev_grid.dims[1] - dev_grid.guard[1] + shift &&
      k < dev_grid.dims[2] - dev_grid.guard[2] + shift) {
    ijk = i + j * dev_grid.dims[0] +
          k * dev_grid.dims[0] * dev_grid.dims[1];
    
    x = dev_grid.pos(0, i, 1);
    y = dev_grid.pos(1, j, 1);
    z = dev_grid.pos(2, k, 1);

    // Calculate Ex
    Ddx = get_gamma_d11(dev_params.a, x, y, z) * Dx[ijk] + get_gamma_d12(dev_params.a, x, y, z) * Dy[ijk]
          + get_gamma_d13(dev_params.a, x, y, z) * Dz[ijk];
    Ex[ijk] = get_alpha(dev_params.a, x, y, z) * Ddx + get_sqrt_gamma(dev_params.a, x, y, z) *
              (get_beta_u2(dev_params.a, x, y, z) * Bz[ijk] - get_beta_u3(dev_params.a, x, y, z) * By[ijk]);

    // Calculate Ey
    Ddy = get_gamma_d12(dev_params.a, x, y, z) * Dx[ijk] + get_gamma_d22(dev_params.a, x, y, z) * Dy[ijk]
          + get_gamma_d23(dev_params.a, x, y, z) * Dz[ijk];
    Ey[ijk] = get_alpha(dev_params.a, x, y, z) * Ddy + get_sqrt_gamma(dev_params.a, x, y, z) *
              (get_beta_u3(dev_params.a, x, y, z) * Bx[ijk] - get_beta_u1(dev_params.a, x, y, z) * Bz[ijk]);

    // Calculate Ez
    Ddz = get_gamma_d13(dev_params.a, x, y, z) * Dx[ijk] + get_gamma_d23(dev_params.a, x, y, z) * Dy[ijk]
          + get_gamma_d33(dev_params.a, x, y, z) * Dz[ijk];
    Ez[ijk] = get_alpha(dev_params.a, x, y, z) * Ddz + get_sqrt_gamma(dev_params.a, x, y, z) *
              (get_beta_u1(dev_params.a, x, y, z) * By[ijk] - get_beta_u2(dev_params.a, x, y, z) * Bx[ijk]);
  }
}

__global__ void
kernel_compute_H_gr_thread(const Scalar *Dx, const Scalar *Dy, const Scalar *Dz,
                          const Scalar *Bx, const Scalar *By, const Scalar *Bz,
                          Scalar *Hx, Scalar *Hy, Scalar *Hz, int shift) {
  size_t ijk;
  Scalar Bdx, Bdy, Bdz;
  Scalar x, y, z;

  int i = threadIdx.x + blockIdx.x * blockDim.x + dev_grid.guard[0] - shift;
  int j = threadIdx.y + blockIdx.y * blockDim.y + dev_grid.guard[1] - shift;
  int k = threadIdx.z + blockIdx.z * blockDim.z + dev_grid.guard[2] - shift;
  if (i < dev_grid.dims[0] - dev_grid.guard[0] + shift &&
      j < dev_grid.dims[1] - dev_grid.guard[1] + shift &&
      k < dev_grid.dims[2] - dev_grid.guard[2] + shift) {
    ijk = i + j * dev_grid.dims[0] +
          k * dev_grid.dims[0] * dev_grid.dims[1];
    x = dev_grid.pos(0, i, 1);
    y = dev_grid.pos(1, j, 1);
    z = dev_grid.pos(2, k, 1);
    // Calculate Hx
    Bdx = get_gamma_d11(dev_params.a, x, y, z) * Bx[ijk] + get_gamma_d12(dev_params.a, x, y, z) * By[ijk]
          + get_gamma_d13(dev_params.a, x, y, z) * Bz[ijk];
    Hx[ijk] = get_alpha(dev_params.a, x, y, z) * Bdx - get_sqrt_gamma(dev_params.a, x, y, z) *
              (get_beta_u2(dev_params.a, x, y, z) * Dz[ijk] - get_beta_u3(dev_params.a, x, y, z) * Dy[ijk]);

    // Calculate Hy
    Bdy = get_gamma_d12(dev_params.a, x, y, z) * Bx[ijk] + get_gamma_d22(dev_params.a, x, y, z) * By[ijk]
          + get_gamma_d23(dev_params.a, x, y, z) * Bz[ijk];
    Hy[ijk] = get_alpha(dev_params.a, x, y, z) * Bdy - get_sqrt_gamma(dev_params.a, x, y, z) *
              (get_beta_u3(dev_params.a, x, y, z) * Dx[ijk] - get_beta_u1(dev_params.a, x, y, z) * Dz[ijk]);

    // Calculate Hz
    Bdz = get_gamma_d13(dev_params.a, x, y, z) * Bx[ijk] + get_gamma_d23(dev_params.a, x, y, z) * By[ijk]
          + get_gamma_d33(dev_params.a, x, y, z) * Bz[ijk];
    Hz[ijk] = get_alpha(dev_params.a, x, y, z) * Bdz - get_sqrt_gamma(dev_params.a, x, y, z) *
              (get_beta_u1(dev_params.a, x, y, z) * Dy[ijk] - get_beta_u2(dev_params.a, x, y, z) * Dx[ijk]);
  }
}

__global__ void
kernel_rk_step1_thread(const Scalar *Ex, const Scalar *Ey,
                       const Scalar *Ez, const Scalar *Hx,
                       const Scalar *Hy, const Scalar *Hz,
                       const Scalar Dx, const Scalar Dy,
                       const Scalar Dz, const const Scalar *Bx,
                       const Scalar *By, const Scalar *Bz, Scalar *dDx,
                       Scalar *dDy, Scalar *dDz, Scalar *dBx,
                       Scalar *dBy, Scalar *dBz, const Scalar *P,
                       Scalar *dP, int shift, Scalar As) {
  size_t ijk;
  int i =
      threadIdx.x + blockIdx.x * blockDim.x + dev_grid.guard[0] - shift;
  int j =
      threadIdx.y + blockIdx.y * blockDim.y + dev_grid.guard[1] - shift;
  int k =
      threadIdx.z + blockIdx.z * blockDim.z + dev_grid.guard[2] - shift;
  if (i < dev_grid.dims[0] - dev_grid.guard[0] + shift &&
      j < dev_grid.dims[1] - dev_grid.guard[1] + shift &&
      k < dev_grid.dims[2] - dev_grid.guard[2] + shift) {
    ijk = i + j * dev_grid.dims[0] +
          k * dev_grid.dims[0] * dev_grid.dims[1];

    Scalar x = dev_grid.pos(0, i, 1);
    Scalar y = dev_grid.pos(1, j, 1);
    Scalar z = dev_grid.pos(2, k, 1);
    Scalar gmsqrt = get_sqrt_gamma(dev_params.a, x, y, z);
    Scalar alpha = get_alpha(dev_params.a, x, y, z);

    Scalar rotHx = (dfdy(Hz, ijk) - dfdz(Hy, ijk)) / gmsqrt;
    Scalar rotHy = (dfdz(Hx, ijk) - dfdx(Hz, ijk)) / gmsqrt;
    Scalar rotHz = (dfdx(Hy, ijk) - dfdy(Hx, ijk)) / gmsqrt;
    Scalar rotEx = (dfdy(Ez, ijk) - dfdz(Ey, ijk)) / gmsqrt;
    Scalar rotEy = (dfdz(Ex, ijk) - dfdx(Ez, ijk)) / gmsqrt;
    Scalar rotEz = (dfdx(Ey, ijk) - dfdy(Ex, ijk)) / gmsqrt;

    Scalar divD = div4(Dx, Dy, Dz, ijk, x, y, z);
    Scalar divB = div4(Bx, By, Bz, ijk, x, y, z);

    Scalar Bdx = get_gamma_d11(dev_params.a, x, y, z) * Bx[ijk] +
                 get_gamma_d12(dev_params.a, x, y, z) * By[ijk] +
                 get_gamma_d13(dev_params.a, x, y, z) * Bz[ijk];
    Scalar Bdy = get_gamma_d12(dev_params.a, x, y, z) * Bx[ijk] +
                 get_gamma_d22(dev_params.a, x, y, z) * By[ijk] +
                 get_gamma_d23(dev_params.a, x, y, z) * Bz[ijk];
    Scalar Bdz = get_gamma_d13(dev_params.a, x, y, z) * Bx[ijk] +
                 get_gamma_d23(dev_params.a, x, y, z) * By[ijk] +
                 get_gamma_d33(dev_params.a, x, y, z) * Bz[ijk];
    Scalar Ddx = get_gamma_d11(dev_params.a, x, y, z) * Dx[ijk] +
                 get_gamma_d12(dev_params.a, x, y, z) * Dy[ijk] +
                 get_gamma_d13(dev_params.a, x, y, z) * Dz[ijk];
    Scalar Ddy = get_gamma_d12(dev_params.a, x, y, z) * Dx[ijk] +
                 get_gamma_d22(dev_params.a, x, y, z) * Dy[ijk] +
                 get_gamma_d23(dev_params.a, x, y, z) * Dz[ijk];
    Scalar Ddz = get_gamma_d13(dev_params.a, x, y, z) * Dx[ijk] +
                 get_gamma_d23(dev_params.a, x, y, z) * Dy[ijk] +
                 get_gamma_d33(dev_params.a, x, y, z) * Dz[ijk];
    Scalar B2 = Bx[ijk] * Bdx + By[ijk] * Bdy + Bz[ijk] * Bdz;
    if (B2 < TINY) B2 = TINY;

    Scalar Jp = (Bdx * rotHx + Bdy * rotHy + Bdz * rotHz) -
                (Ddx * rotEx + dDy * rotEy + Ddz * rotEz);
    Scalar Jx = (divD * (Ey[ijk] * Bdz - Ez[ijk] * Bdy) / gmsqrt +
                 Jp * Bx[ijk]) /
                B2;
    Scalar Jy = (divD * (Ez[ijk] * Bdx - Ex[ijk] * Bdz) / gmsqrt +
                 Jp * By[ijk]) /
                B2;
    Scalar Jz = (divD * (Ex[ijk] * Bdy - Ey[ijk] * Bdx) / gmsqrt +
                 Jp * Bz[ijk]) /
                B2;

    Scalar Pxd = dfdx(P, ijk);
    Scalar Pyd = dfdy(P, ijk);
    Scalar Pzd = dfdz(P, ijk);
    Scalar Pxu = get_gamma_u11(dev_params.a, x, y, z) * Pxd +
                 get_gamma_u12(dev_params.a, x, y, z) * Pyd +
                 get_gamma_u13(dev_params.a, x, y, z) * Pzd;
    Scalar Pyu = get_gamma_u12(dev_params.a, x, y, z) * Pxd +
                 get_gamma_u22(dev_params.a, x, y, z) * Pyd +
                 get_gamma_u23(dev_params.a, x, y, z) * Pzd;
    Scalar Pzu = get_gamma_u13(dev_params.a, x, y, z) * Pxd +
                 get_gamma_u23(dev_params.a, x, y, z) * Pyd +
                 get_gamma_u33(dev_params.a, x, y, z) * Pzd;

    dBx[ijk] =
        As * dBx[ijk] +
        dev_params.dt * (-rotEx - alpha * Pxu +
                         get_beta_u1(dev_params.a, x, y, z) * divB);
    dBy[ijk] =
        As * dBy[ijk] +
        dev_params.dt * (-rotEy - alpha * Pyu +
                         get_beta_u2(dev_params.a, x, y, z) * divB);
    dBz[ijk] =
        As * dBz[ijk] +
        dev_params.dt * (-rotEz - alpha * Pzu +
                         get_beta_u3(dev_params.a, x, y, z) * divB);

    dDx[ijk] = As * dDx[ijk] + dev_params.dt * (rotBx - Jx);
    dDy[ijk] = As * dDy[ijk] + dev_params.dt * (rotBy - Jy);
    dDz[ijk] = As * dDz[ijk] + dev_params.dt * (rotBz - Jz);

    dP[ijk] =
        As * dP[ijk] +
        dev_params.dt *
            (-alpha * divB + get_beta_u1(dev_params.a, x, y, z) * Pxd +
             get_beta_u2(dev_params.a, x, y, z) * Pyd +
             get_beta_u3(dev_params.a, x, y, z) * Pzd -
             alpha * P[ijk] / dev_params.tau);
  }
}
}  // namespace Coffee