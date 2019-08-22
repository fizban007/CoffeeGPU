#include "cuda/constant_mem.h"
#include "cuda/constant_mem_func.h"
#include "cuda/cuda_utility.h"
#include "cuda/cuda_control.h"
#include "field_solver_gr.h"
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

__global__ void
kernel_compute_rho_gr_thread(const Scalar *Dx, const Scalar *Dy, const Scalar *Dz,
                          Scalar *rho, int shift) {
  size_t ijk;
  int i = threadIdx.x + blockIdx.x * blockDim.x + dev_grid.guard[0] - shift;
  int j = threadIdx.y + blockIdx.y * blockDim.y + dev_grid.guard[1] - shift;
  int k = threadIdx.z + blockIdx.z * blockDim.z + dev_grid.guard[2] - shift;
  if (i < dev_grid.dims[0] - dev_grid.guard[0] + shift &&
      j < dev_grid.dims[1] - dev_grid.guard[1] + shift &&
      k < dev_grid.dims[2] - dev_grid.guard[2] + shift) {
    ijk = i + j * dev_grid.dims[0] +
          k * dev_grid.dims[0] * dev_grid.dims[1];
    Scalar x = dev_grid.pos(0, i, 1);
    Scalar y = dev_grid.pos(1, j, 1);
    Scalar z = dev_grid.pos(2, k, 1);
    rho[ijk] = (dev_grid.inv_delta[0] * (Dx[ijk] * get_sqrt_gamma(dev_params.a, x + dev_grid.delta[0] / 2.0, y, z) 
               - Dx[ijk - 1] * get_sqrt_gamma(dev_params.a, x - dev_grid.delta[0] / 2.0, y, z)) +
               dev_grid.inv_delta[1] * (Dy[ijk] * get_sqrt_gamma(dev_params.a, x, y + dev_grid.delta[1] / 2.0, z)
               - Dy[ijk - dev_grid.dims[0]] * get_sqrt_gamma(dev_params.a, x, y - dev_grid.delta[1] / 2.0, z)) +
               dev_grid.inv_delta[2] * (Dz[ijk] * get_sqrt_gamma(dev_params.a, x, y, z + dev_grid.delta[2] / 2.0)
               - Dz[ijk - dev_grid.dims[0] * dev_grid.dims[1]]
               * get_sqrt_gamma(dev_params.a, x, y, z - dev_grid.delta[2] / 2.0)))
               / get_sqrt_gamma(dev_params.a, x, y, z);
  }
}

__global__ void
kernel_compute_E_gr_thread(const Scalar *Dx, const Scalar *Dy, const Scalar *Dz,
                          const Scalar *Bx, const Scalar *By, const Scalar *Bz,
                          Scalar *Ex, Scalar *Ey, Scalar *Ez, int shift) {
  size_t ijk;
  Scalar intDx, intDy, intDz, intBx, intBy, intBz;
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
    // Calculate Ex
    x = dev_grid.pos(0, i, 0);
    y = dev_grid.pos(1, j, 1);
    z = dev_grid.pos(2, k, 1);
    intDy = interpolate(Dy, ijk, Stagger(0b101), Stagger(0b110),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intDz = interpolate(Dz, ijk, Stagger(0b011), Stagger(0b110),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intBy = interpolate(By, ijk, Stagger(0b010), Stagger(0b110),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intBz = interpolate(Bz, ijk, Stagger(0b100), Stagger(0b110),
                       dev_grid.dims[0], dev_grid.dims[1]);
    Ddx = get_gamma_d11(dev_params.a, x, y, z) * Dx[ijk] + get_gamma_d12(dev_params.a, x, y, z) * intDy
          + get_gamma_d13(dev_params.a, x, y, z) * intDz;
    Ex[ijk] = get_alpha(dev_params.a, x, y, z) * Ddx + get_sqrt_gamma(dev_params.a, x, y, z) *
              (get_beta_u2(dev_params.a, x, y, z) * intBz - get_beta_u3(dev_params.a, x, y, z) * intBy);

    // Calculate Ey
    x = dev_grid.pos(0, i, 1);
    y = dev_grid.pos(1, j, 0);
    z = dev_grid.pos(2, k, 1);
    intDx = interpolate(Dx, ijk, Stagger(0b110), Stagger(0b101),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intDz = interpolate(Dz, ijk, Stagger(0b011), Stagger(0b101),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intBx = interpolate(Bx, ijk, Stagger(0b001), Stagger(0b101),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intBz = interpolate(Bz, ijk, Stagger(0b100), Stagger(0b101),
                       dev_grid.dims[0], dev_grid.dims[1]);
    Ddy = get_gamma_d12(dev_params.a, x, y, z) * intDx + get_gamma_d22(dev_params.a, x, y, z) * Dy[ijk]
          + get_gamma_d23(dev_params.a, x, y, z) * intDz;
    Ey[ijk] = get_alpha(dev_params.a, x, y, z) * Ddy + get_sqrt_gamma(dev_params.a, x, y, z) *
              (get_beta_u3(dev_params.a, x, y, z) * intBx - get_beta_u1(dev_params.a, x, y, z) * intBz);

    // Calculate Ez
    x = dev_grid.pos(0, i, 1);
    y = dev_grid.pos(1, j, 1);
    z = dev_grid.pos(2, k, 0);
    intDx = interpolate(Dx, ijk, Stagger(0b110), Stagger(0b011),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intDy = interpolate(Dy, ijk, Stagger(0b101), Stagger(0b011),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intBx = interpolate(Bx, ijk, Stagger(0b001), Stagger(0b011),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intBy = interpolate(By, ijk, Stagger(0b010), Stagger(0b011),
                       dev_grid.dims[0], dev_grid.dims[1]);
    Ddz = get_gamma_d13(dev_params.a, x, y, z) * intDx + get_gamma_d23(dev_params.a, x, y, z) * intDy
          + get_gamma_d33(dev_params.a, x, y, z) * Dz[ijk];
    Ez[ijk] = get_alpha(dev_params.a, x, y, z) * Ddz + get_sqrt_gamma(dev_params.a, x, y, z) *
              (get_beta_u1(dev_params.a, x, y, z) * intBy - get_beta_u2(dev_params.a, x, y, z) * intBx);
  }
}


__global__ void
kernel_compute_H_gr_thread(const Scalar *Dx, const Scalar *Dy, const Scalar *Dz,
                          const Scalar *Bx, const Scalar *By, const Scalar *Bz,
                          Scalar *B0x, Scalar *B0y, Scalar *B0z,
                          Scalar *Hx, Scalar *Hy, Scalar *Hz, int shift) {
  size_t ijk;
  Scalar intDx, intDy, intDz, intBx, intBy, intBz;
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
    // We use B0 to store lower B
    // Calculate Hx
    x = dev_grid.pos(0, i, 1);
    y = dev_grid.pos(1, j, 0);
    z = dev_grid.pos(2, k, 0);
    intDy = interpolate(Dy, ijk, Stagger(0b101), Stagger(0b001),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intDz = interpolate(Dz, ijk, Stagger(0b011), Stagger(0b001),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intBy = interpolate(By, ijk, Stagger(0b010), Stagger(0b001),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intBz = interpolate(Bz, ijk, Stagger(0b100), Stagger(0b001),
                       dev_grid.dims[0], dev_grid.dims[1]);
    Bdx = get_gamma_d11(dev_params.a, x, y, z) * Bx[ijk] + get_gamma_d12(dev_params.a, x, y, z) * intBy
          + get_gamma_d13(dev_params.a, x, y, z) * intBz;
    Hx[ijk] = get_alpha(dev_params.a, x, y, z) * Bdx - get_sqrt_gamma(dev_params.a, x, y, z) *
              (get_beta_u2(dev_params.a, x, y, z) * intDz - get_beta_u3(dev_params.a, x, y, z) * intDy);
    B0x[ijk] = Bdx;

    // Calculate Hy
    x = dev_grid.pos(0, i, 0);
    y = dev_grid.pos(1, j, 1);
    z = dev_grid.pos(2, k, 0);
    intDx = interpolate(Dx, ijk, Stagger(0b110), Stagger(0b010),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intDz = interpolate(Dz, ijk, Stagger(0b011), Stagger(0b010),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intBx = interpolate(Bx, ijk, Stagger(0b001), Stagger(0b010),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intBz = interpolate(Bz, ijk, Stagger(0b100), Stagger(0b010),
                       dev_grid.dims[0], dev_grid.dims[1]);
    Bdy = get_gamma_d12(dev_params.a, x, y, z) * intBx + get_gamma_d22(dev_params.a, x, y, z) * By[ijk]
          + get_gamma_d23(dev_params.a, x, y, z) * intBz;
    Hy[ijk] = get_alpha(dev_params.a, x, y, z) * Bdy - get_sqrt_gamma(dev_params.a, x, y, z) *
              (get_beta_u3(dev_params.a, x, y, z) * intDx - get_beta_u1(dev_params.a, x, y, z) * intDz);
    B0y[ijk] = Bdy;

    // Calculate Hz
    x = dev_grid.pos(0, i, 0);
    y = dev_grid.pos(1, j, 0);
    z = dev_grid.pos(2, k, 1);
    intDx = interpolate(Dx, ijk, Stagger(0b110), Stagger(0b100),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intDy = interpolate(Dy, ijk, Stagger(0b101), Stagger(0b100),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intBx = interpolate(Bx, ijk, Stagger(0b001), Stagger(0b100),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intBy = interpolate(By, ijk, Stagger(0b010), Stagger(0b100),
                       dev_grid.dims[0], dev_grid.dims[1]);
    Bdz = get_gamma_d13(dev_params.a, x, y, z) * intBx + get_gamma_d23(dev_params.a, x, y, z) * intBy
          + get_gamma_d33(dev_params.a, x, y, z) * Bz[ijk];
    Hz[ijk] = get_alpha(dev_params.a, x, y, z) * Bdz - get_sqrt_gamma(dev_params.a, x, y, z) *
              (get_beta_u1(dev_params.a, x, y, z) * intDy - get_beta_u2(dev_params.a, x, y, z) * intDx);
    B0z[ijk] = Bdz;
  }
}

__global__ void
kernel_rk_push_gr_thread(const Scalar *Ex, const Scalar *Ey, const Scalar *Ez,
                      const Scalar *Hx, const Scalar *Hy, const Scalar *Hz,
                      const Scalar *Bx, const Scalar *By, const Scalar *Bz,
                      const Scalar *B0x, const Scalar *B0y, const Scalar *B0z, 
                      Scalar *dDx, Scalar *dDy, Scalar *dDz, 
                      Scalar *dBx, Scalar *dBy, Scalar *dBz,
                      Scalar *rho, int shift) {
  Scalar CCx = dev_params.dt * dev_grid.inv_delta[0];
  Scalar CCy = dev_params.dt * dev_grid.inv_delta[1];
  Scalar CCz = dev_params.dt * dev_grid.inv_delta[2];
  Scalar intEx, intEy, intEz, intBx, intBy, intBz, intrho;
  Scalar intBdx, intBdy, intBdz;
  Scalar jx, jy, jz;
  Scalar x, y, z;
  size_t ijk, iP1jk, iM1jk, ijP1k, ijM1k, ijkP1, ijkM1;

  int i = threadIdx.x + blockIdx.x * blockDim.x + dev_grid.guard[0] - shift;
  int j = threadIdx.y + blockIdx.y * blockDim.y + dev_grid.guard[1] - shift;
  int k = threadIdx.z + blockIdx.z * blockDim.z + dev_grid.guard[2] - shift;
  if (i < dev_grid.dims[0] - dev_grid.guard[0] + shift &&
      j < dev_grid.dims[1] - dev_grid.guard[1] + shift &&
      k < dev_grid.dims[2] - dev_grid.guard[2] + shift) {
    ijk = i + j * dev_grid.dims[0] +
          k * dev_grid.dims[0] * dev_grid.dims[1];
    iP1jk = ijk + 1;
    iM1jk = ijk - 1;
    ijP1k = ijk + dev_grid.dims[0];
    ijM1k = ijk - dev_grid.dims[0];
    ijkP1 = ijk + dev_grid.dims[0] * dev_grid.dims[1];
    ijkM1 = ijk - dev_grid.dims[0] * dev_grid.dims[1];
    // push B-field
    x = dev_grid.pos(0, i, 1);
    y = dev_grid.pos(1, j, 0);
    z = dev_grid.pos(2, k, 0);
    dBx[ijk] = (CCz * (Ey[ijkP1] - Ey[ijk]) - CCy * (Ez[ijP1k] - Ez[ijk])) / get_sqrt_gamma(dev_params.a, x, y, z);
    x = dev_grid.pos(0, i, 0);
    y = dev_grid.pos(1, j, 1);
    z = dev_grid.pos(2, k, 0);
    dBy[ijk] = (CCx * (Ez[iP1jk] - Ez[ijk]) - CCz * (Ex[ijkP1] - Ex[ijk])) / get_sqrt_gamma(dev_params.a, x, y, z);
    x = dev_grid.pos(0, i, 0);
    y = dev_grid.pos(1, j, 0);
    z = dev_grid.pos(2, k, 1);
    dBz[ijk] = (CCy * (Ex[ijP1k] - Ex[ijk]) - CCx * (Ey[iP1jk] - Ey[ijk])) / get_sqrt_gamma(dev_params.a, x, y, z);
    // push D-field
    x = dev_grid.pos(0, i, 0);
    y = dev_grid.pos(1, j, 1);
    z = dev_grid.pos(2, k, 1);
    dDx[ijk] = (CCz * (Hy[ijkM1] - Hy[ijk]) - CCy *(Hz[ijM1k] - Hz[ijk])) / get_sqrt_gamma(dev_params.a, x, y, z);
    x = dev_grid.pos(0, i, 1);
    y = dev_grid.pos(1, j, 0);
    z = dev_grid.pos(2, k, 1);
    dDy[ijk] = (CCx * (Hz[iM1jk] - Hz[ijk]) - CCz * (Hx[ijkM1] - Hx[ijk])) / get_sqrt_gamma(dev_params.a, x, y, z);
    x = dev_grid.pos(0, i, 1);
    y = dev_grid.pos(1, j, 1);
    z = dev_grid.pos(2, k, 0);
    dDz[ijk] = (CCy * (Hx[ijM1k] - Hx[ijk]) - CCx * (Hy[iM1jk] - Hy[ijk])) / get_sqrt_gamma(dev_params.a, x, y, z);
    // if (i == 10 && j == 10 && k == 10)
      // printf("%d, %d, %d\n", dev_grid.dims[0], dev_grid.dims[1], dev_grid.dims[2]);
      // printf("%f, %f, %f\n", dEx[ijk], dEy[ijk], dEz[ijk]);
      // printf("%lu, %lu, %lu\n", ijkM1, ijM1k, iM1jk);
    
    if (dev_params.calc_current)
    {
      // computing currents
      // Note that lower B is stored in B0
      //   `j_x`:
      x = dev_grid.pos(0, i, 0);
      y = dev_grid.pos(1, j, 1);
      z = dev_grid.pos(2, k, 1);
      intrho = interpolate(rho, ijk, Stagger(0b111), Stagger(0b110),
                           dev_grid.dims[0], dev_grid.dims[1]);
      intEx = interpolate(Ex, ijk, Stagger(0b110), Stagger(0b110),
                         dev_grid.dims[0], dev_grid.dims[1]);
      intEy = interpolate(Ey, ijk, Stagger(0b101), Stagger(0b110),
                         dev_grid.dims[0], dev_grid.dims[1]);
      intEz = interpolate(Ez, ijk, Stagger(0b011), Stagger(0b110),
                         dev_grid.dims[0], dev_grid.dims[1]);
      intBx = interpolate(Bx, ijk, Stagger(0b001), Stagger(0b110),
                         dev_grid.dims[0], dev_grid.dims[1]);
      intBy = interpolate(By, ijk, Stagger(0b010), Stagger(0b110),
                         dev_grid.dims[0], dev_grid.dims[1]);
      intBz = interpolate(Bz, ijk, Stagger(0b100), Stagger(0b110),
                         dev_grid.dims[0], dev_grid.dims[1]);
      intBdx = interpolate(B0x, ijk, Stagger(0b001), Stagger(0b110),
                         dev_grid.dims[0], dev_grid.dims[1]);
      intBdy = interpolate(B0y, ijk, Stagger(0b010), Stagger(0b110),
                         dev_grid.dims[0], dev_grid.dims[1]);
      intBdz = interpolate(B0z, ijk, Stagger(0b100), Stagger(0b110),
                         dev_grid.dims[0], dev_grid.dims[1]);
      jx = dev_params.dt * intrho * (intEy * intBdz - intBdy * intEz) / get_sqrt_gamma(dev_params.a, x, y, z) /
           (intBx * intBdx + intBy * intBdy + intBz * intBdz + TINY);
      //   `j_y`:
      x = dev_grid.pos(0, i, 1);
      y = dev_grid.pos(1, j, 0);
      z = dev_grid.pos(2, k, 1);
      intrho = interpolate(rho, ijk, Stagger(0b111), Stagger(0b101),
                           dev_grid.dims[0], dev_grid.dims[1]);
      intEx = interpolate(Ex, ijk, Stagger(0b110), Stagger(0b101),
                         dev_grid.dims[0], dev_grid.dims[1]);
      intEy = interpolate(Ey, ijk, Stagger(0b101), Stagger(0b101),
                         dev_grid.dims[0], dev_grid.dims[1]);
      intEz = interpolate(Ez, ijk, Stagger(0b011), Stagger(0b101),
                         dev_grid.dims[0], dev_grid.dims[1]);
      intBx = interpolate(Bx, ijk, Stagger(0b001), Stagger(0b101),
                         dev_grid.dims[0], dev_grid.dims[1]);
      intBy = interpolate(By, ijk, Stagger(0b010), Stagger(0b101),
                         dev_grid.dims[0], dev_grid.dims[1]);
      intBz = interpolate(Bz, ijk, Stagger(0b100), Stagger(0b101),
                         dev_grid.dims[0], dev_grid.dims[1]);
      intBdx = interpolate(B0x, ijk, Stagger(0b001), Stagger(0b101),
                         dev_grid.dims[0], dev_grid.dims[1]);
      intBdy = interpolate(B0y, ijk, Stagger(0b010), Stagger(0b101),
                         dev_grid.dims[0], dev_grid.dims[1]);
      intBdz = interpolate(B0z, ijk, Stagger(0b100), Stagger(0b101),
                         dev_grid.dims[0], dev_grid.dims[1]);
      jy = dev_params.dt * intrho * (intEz * intBdx - intEx * intBdz) / get_sqrt_gamma(dev_params.a, x, y, z) /
           (intBx * intBdx + intBy * intBdy + intBz * intBdz + TINY);
      //   `j_z`:
      x = dev_grid.pos(0, i, 1);
      y = dev_grid.pos(1, j, 1);
      z = dev_grid.pos(2, k, 0);
      intrho = interpolate(rho, ijk, Stagger(0b111), Stagger(0b011),
                           dev_grid.dims[0], dev_grid.dims[1]);
      intEx = interpolate(Ex, ijk, Stagger(0b110), Stagger(0b011),
                         dev_grid.dims[0], dev_grid.dims[1]);
      intEy = interpolate(Ey, ijk, Stagger(0b101), Stagger(0b011),
                         dev_grid.dims[0], dev_grid.dims[1]);
      intEz = interpolate(Ez, ijk, Stagger(0b011), Stagger(0b011),
                         dev_grid.dims[0], dev_grid.dims[1]);
      intBx = interpolate(Bx, ijk, Stagger(0b001), Stagger(0b011),
                         dev_grid.dims[0], dev_grid.dims[1]);
      intBy = interpolate(By, ijk, Stagger(0b010), Stagger(0b011),
                         dev_grid.dims[0], dev_grid.dims[1]);
      intBz = interpolate(Bz, ijk, Stagger(0b100), Stagger(0b011),
                         dev_grid.dims[0], dev_grid.dims[1]);
      intBdx = interpolate(B0x, ijk, Stagger(0b001), Stagger(0b011),
                         dev_grid.dims[0], dev_grid.dims[1]);
      intBdy = interpolate(B0y, ijk, Stagger(0b010), Stagger(0b011),
                         dev_grid.dims[0], dev_grid.dims[1]);
      intBdz = interpolate(B0z, ijk, Stagger(0b100), Stagger(0b011),
                         dev_grid.dims[0], dev_grid.dims[1]);
      jz = dev_params.dt * intrho * (intEx * intBdy - intBdx * intEy) / get_sqrt_gamma(dev_params.a, x, y, z) /
           (intBx * intBdx + intBy * intBdy + intBz * intBdz + TINY);

      dDx[ijk] -= jx;
      dDy[ijk] -= jy;
      dDz[ijk] -= jz;
    }
    
  }
}



__global__ void
kernel_rk_update_gr_thread(Scalar *Dx, Scalar *Dy, Scalar *Dz, Scalar *Bx,
                        Scalar *By, Scalar *Bz, const Scalar *Dnx,
                        const Scalar *Dny, const Scalar *Dnz,
                        const Scalar *Bnx, const Scalar *Bny,
                        const Scalar *Bnz, Scalar *dDx, Scalar *dDy,
                        Scalar *dDz, const Scalar *dBx, const Scalar *dBy,
                        const Scalar *dBz, Scalar rk_c1, Scalar rk_c2,
                        Scalar rk_c3, int shift) {
  size_t ijk;
  int i = threadIdx.x + blockIdx.x * blockDim.x + dev_grid.guard[0] - shift;
  int j = threadIdx.y + blockIdx.y * blockDim.y + dev_grid.guard[1] - shift;
  int k = threadIdx.z + blockIdx.z * blockDim.z + dev_grid.guard[2] - shift;
  if (i < dev_grid.dims[0] - dev_grid.guard[0] + shift &&
      j < dev_grid.dims[1] - dev_grid.guard[1] + shift &&
      k < dev_grid.dims[2] - dev_grid.guard[2] + shift) {
    ijk = i + j * dev_grid.dims[0] +
          k * dev_grid.dims[0] * dev_grid.dims[1];
    // update D-field
    Dx[ijk] = rk_c1 * Dnx[ijk] + rk_c2 * Dx[ijk] + rk_c3 * dDx[ijk];
    Dy[ijk] = rk_c1 * Dny[ijk] + rk_c2 * Dy[ijk] + rk_c3 * dDy[ijk];
    Dz[ijk] = rk_c1 * Dnz[ijk] + rk_c2 * Dz[ijk] + rk_c3 * dDz[ijk];
    dDx[ijk] = Dx[ijk];
    dDy[ijk] = Dy[ijk];
    dDz[ijk] = Dz[ijk];
    // update B-field
    Bx[ijk] = rk_c1 * Bnx[ijk] + rk_c2 * Bx[ijk] + rk_c3 * dBx[ijk];
    By[ijk] = rk_c1 * Bny[ijk] + rk_c2 * By[ijk] + rk_c3 * dBy[ijk];
    Bz[ijk] = rk_c1 * Bnz[ijk] + rk_c2 * Bz[ijk] + rk_c3 * dBz[ijk];
  }
}



__global__ void
kernel_clean_epar_gr_thread(const Scalar *Dx, const Scalar *Dy, const Scalar *Dz,
                         const Scalar *Bx, const Scalar *By, const Scalar *Bz,
                         Scalar *dDx, Scalar *dDy, Scalar *dDz, int shift) {
  Scalar intDx, intDy, intDz, intBx, intBy, intBz, intBdx, intBdy, intBdz;
  Scalar x, y, z;
  size_t ijk;
  int i = threadIdx.x + blockIdx.x * blockDim.x + dev_grid.guard[0] - shift;
  int j = threadIdx.y + blockIdx.y * blockDim.y + dev_grid.guard[1] - shift;
  int k = threadIdx.z + blockIdx.z * blockDim.z + dev_grid.guard[2] - shift;
  if (i < dev_grid.dims[0] - dev_grid.guard[0] + shift &&
      j < dev_grid.dims[1] - dev_grid.guard[1] + shift &&
      k < dev_grid.dims[2] - dev_grid.guard[2] + shift) {
    ijk = i + j * dev_grid.dims[0] +
          k * dev_grid.dims[0] * dev_grid.dims[1];
    // x:
    x = dev_grid.pos(0, i, 0);
    y = dev_grid.pos(1, j, 1);
    z = dev_grid.pos(2, k, 1);
    intDy = interpolate(Dy, ijk, Stagger(0b101), Stagger(0b110),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intDz = interpolate(Dz, ijk, Stagger(0b011), Stagger(0b110),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intBx = interpolate(Bx, ijk, Stagger(0b001), Stagger(0b110),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intBy = interpolate(By, ijk, Stagger(0b010), Stagger(0b110),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intBz = interpolate(Bz, ijk, Stagger(0b100), Stagger(0b110),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intBdx = get_gamma_d11(dev_params.a, x, y, z) * intBx +
             get_gamma_d12(dev_params.a, x, y, z) * intBy +
             get_gamma_d13(dev_params.a, x, y, z) * intBz;
    intBdy = get_gamma_d12(dev_params.a, x, y, z) * intBx +
             get_gamma_d22(dev_params.a, x, y, z) * intBy +
             get_gamma_d23(dev_params.a, x, y, z) * intBz;
    intBdz = get_gamma_d13(dev_params.a, x, y, z) * intBx +
             get_gamma_d23(dev_params.a, x, y, z) * intBy +
             get_gamma_d33(dev_params.a, x, y, z) * intBz;
    dDx[ijk] = Dx[ijk] -
               (Dx[ijk] * intBdx + intDy * intBdy + intDz * intBdz) *
                   intBx /
                   (intBx * intBdx + intBy * intBdy + intBz * intBdz + TINY);

    // y:
    x = dev_grid.pos(0, i, 1);
    y = dev_grid.pos(1, j, 0);
    z = dev_grid.pos(2, k, 1);
    intDx = interpolate(Dx, ijk, Stagger(0b110), Stagger(0b101),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intDz = interpolate(Dz, ijk, Stagger(0b011), Stagger(0b101),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intBx = interpolate(Bx, ijk, Stagger(0b001), Stagger(0b101),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intBy = interpolate(By, ijk, Stagger(0b010), Stagger(0b101),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intBz = interpolate(Bz, ijk, Stagger(0b100), Stagger(0b101),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intBdx = get_gamma_d11(dev_params.a, x, y, z) * intBx +
             get_gamma_d12(dev_params.a, x, y, z) * intBy +
             get_gamma_d13(dev_params.a, x, y, z) * intBz;
    intBdy = get_gamma_d12(dev_params.a, x, y, z) * intBx +
             get_gamma_d22(dev_params.a, x, y, z) * intBy +
             get_gamma_d23(dev_params.a, x, y, z) * intBz;
    intBdz = get_gamma_d13(dev_params.a, x, y, z) * intBx +
             get_gamma_d23(dev_params.a, x, y, z) * intBy +
             get_gamma_d33(dev_params.a, x, y, z) * intBz;
    dDy[ijk] = Dy[ijk] -
               (intDx * intBdx + Dy[ijk] * intBdy + intDz * intBdz) *
                   intBy /
                   (intBx * intBdx + intBy * intBdy + intBz * intBdz + TINY);

    // z:
    x = dev_grid.pos(0, i, 1);
    y = dev_grid.pos(1, j, 1);
    z = dev_grid.pos(2, k, 0);
    intDx = interpolate(Dx, ijk, Stagger(0b110), Stagger(0b011),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intDy = interpolate(Dy, ijk, Stagger(0b101), Stagger(0b011),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intBx = interpolate(Bx, ijk, Stagger(0b001), Stagger(0b011),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intBy = interpolate(By, ijk, Stagger(0b010), Stagger(0b011),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intBz = interpolate(Bz, ijk, Stagger(0b100), Stagger(0b011),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intBdx = get_gamma_d11(dev_params.a, x, y, z) * intBx +
             get_gamma_d12(dev_params.a, x, y, z) * intBy +
             get_gamma_d13(dev_params.a, x, y, z) * intBz;
    intBdy = get_gamma_d12(dev_params.a, x, y, z) * intBx +
             get_gamma_d22(dev_params.a, x, y, z) * intBy +
             get_gamma_d23(dev_params.a, x, y, z) * intBz;
    intBdz = get_gamma_d13(dev_params.a, x, y, z) * intBx +
             get_gamma_d23(dev_params.a, x, y, z) * intBy +
             get_gamma_d33(dev_params.a, x, y, z) * intBz;
    dDz[ijk] = Dz[ijk] -
               (intDx * intBdx + intDy * intBdy + Dz[ijk] * intBdz) *
                   intBz /
                   (intBx * intBdx + intBy * intBdy + intBz * intBdz + TINY);
  }
}



__global__ void
kernel_check_eGTb_gr_thread(const Scalar *dDx, const Scalar *dDy,
                         const Scalar *dDz, Scalar *Dx, Scalar *Dy, Scalar *Dz,
                         const Scalar *Bx, const Scalar *By,
                         const Scalar *Bz, int shift) {
  Scalar intDx, intDy, intDz, intBx, intBy, intBz, emag, bmag, temp;
  Scalar intDdx, intDdy, intDdz, intBdx, intBdy, intBdz, x, y, z;
  size_t ijk;
  int i = threadIdx.x + blockIdx.x * blockDim.x + dev_grid.guard[0] - shift;
  int j = threadIdx.y + blockIdx.y * blockDim.y + dev_grid.guard[1] - shift;
  int k = threadIdx.z + blockIdx.z * blockDim.z + dev_grid.guard[2] - shift;
  if (i < dev_grid.dims[0] - dev_grid.guard[0] + shift &&
      j < dev_grid.dims[1] - dev_grid.guard[1] + shift &&
      k < dev_grid.dims[2] - dev_grid.guard[2] + shift) {
    ijk = i + j * dev_grid.dims[0] +
          k * dev_grid.dims[0] * dev_grid.dims[1];
    // x:
    x = dev_grid.pos(0, i, 0);
    y = dev_grid.pos(1, j, 1);
    z = dev_grid.pos(2, k, 1);
    intDx = dDx[ijk];
    intDy = interpolate(dDy, ijk, Stagger(0b101), Stagger(0b110),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intDz = interpolate(dDz, ijk, Stagger(0b011), Stagger(0b110),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intDdx = get_gamma_d11(dev_params.a, x, y, z) * intDx +
             get_gamma_d12(dev_params.a, x, y, z) * intDy +
             get_gamma_d13(dev_params.a, x, y, z) * intDz;
    intDdy = get_gamma_d12(dev_params.a, x, y, z) * intDx +
             get_gamma_d22(dev_params.a, x, y, z) * intDy +
             get_gamma_d23(dev_params.a, x, y, z) * intDz;
    intDdz = get_gamma_d13(dev_params.a, x, y, z) * intDx +
             get_gamma_d23(dev_params.a, x, y, z) * intDy +
             get_gamma_d33(dev_params.a, x, y, z) * intDz;
    emag = intDx * intDdx + intDy * intDdy + intDz * intDdz + TINY;
    intBx = interpolate(Bx, ijk, Stagger(0b001), Stagger(0b110),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intBy = interpolate(By, ijk, Stagger(0b010), Stagger(0b110),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intBz = interpolate(Bz, ijk, Stagger(0b100), Stagger(0b110),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intBdx = get_gamma_d11(dev_params.a, x, y, z) * intBx +
             get_gamma_d12(dev_params.a, x, y, z) * intBy +
             get_gamma_d13(dev_params.a, x, y, z) * intBz;
    intBdy = get_gamma_d12(dev_params.a, x, y, z) * intBx +
             get_gamma_d22(dev_params.a, x, y, z) * intBy +
             get_gamma_d23(dev_params.a, x, y, z) * intBz;
    intBdz = get_gamma_d13(dev_params.a, x, y, z) * intBx +
             get_gamma_d23(dev_params.a, x, y, z) * intBy +
             get_gamma_d33(dev_params.a, x, y, z) * intBz;
    bmag = intBx * intBdx + intBy * intBdy + intBz * intBdz + TINY;
    if (emag > bmag) {
      temp = sqrt(bmag / emag);
    } else {
      temp = 1.0;
    }
    Dx[ijk] = temp * dDx[ijk];

    // y:
    x = dev_grid.pos(0, i, 1);
    y = dev_grid.pos(1, j, 0);
    z = dev_grid.pos(2, k, 1);
    intDx = interpolate(dDx, ijk, Stagger(0b110), Stagger(0b101),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intDy = dDy[ijk];
    intDz = interpolate(dDz, ijk, Stagger(0b011), Stagger(0b101),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intDdx = get_gamma_d11(dev_params.a, x, y, z) * intDx +
             get_gamma_d12(dev_params.a, x, y, z) * intDy +
             get_gamma_d13(dev_params.a, x, y, z) * intDz;
    intDdy = get_gamma_d12(dev_params.a, x, y, z) * intDx +
             get_gamma_d22(dev_params.a, x, y, z) * intDy +
             get_gamma_d23(dev_params.a, x, y, z) * intDz;
    intDdz = get_gamma_d13(dev_params.a, x, y, z) * intDx +
             get_gamma_d23(dev_params.a, x, y, z) * intDy +
             get_gamma_d33(dev_params.a, x, y, z) * intDz;
    emag = intDx * intDdx + intDy * intDdy + intDz * intDdz + TINY;
    intBx = interpolate(Bx, ijk, Stagger(0b001), Stagger(0b101),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intBy = interpolate(By, ijk, Stagger(0b010), Stagger(0b101),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intBz = interpolate(Bz, ijk, Stagger(0b100), Stagger(0b101),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intBdx = get_gamma_d11(dev_params.a, x, y, z) * intBx +
             get_gamma_d12(dev_params.a, x, y, z) * intBy +
             get_gamma_d13(dev_params.a, x, y, z) * intBz;
    intBdy = get_gamma_d12(dev_params.a, x, y, z) * intBx +
             get_gamma_d22(dev_params.a, x, y, z) * intBy +
             get_gamma_d23(dev_params.a, x, y, z) * intBz;
    intBdz = get_gamma_d13(dev_params.a, x, y, z) * intBx +
             get_gamma_d23(dev_params.a, x, y, z) * intBy +
             get_gamma_d33(dev_params.a, x, y, z) * intBz;
    bmag = intBx * intBdx + intBy * intBdy + intBz * intBdz + TINY;
    if (emag > bmag) {
      temp = sqrt(bmag / emag);
    } else {
      temp = 1.0;
    }
    Dy[ijk] = temp * dDy[ijk];

    // z:
    x = dev_grid.pos(0, i, 1);
    y = dev_grid.pos(1, j, 1);
    z = dev_grid.pos(2, k, 0);
    intDx = interpolate(dDx, ijk, Stagger(0b110), Stagger(0b011),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intDy = interpolate(dDy, ijk, Stagger(0b101), Stagger(0b011),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intDz = dDz[ijk];
    intDdx = get_gamma_d11(dev_params.a, x, y, z) * intDx +
             get_gamma_d12(dev_params.a, x, y, z) * intDy +
             get_gamma_d13(dev_params.a, x, y, z) * intDz;
    intDdy = get_gamma_d12(dev_params.a, x, y, z) * intDx +
             get_gamma_d22(dev_params.a, x, y, z) * intDy +
             get_gamma_d23(dev_params.a, x, y, z) * intDz;
    intDdz = get_gamma_d13(dev_params.a, x, y, z) * intDx +
             get_gamma_d23(dev_params.a, x, y, z) * intDy +
             get_gamma_d33(dev_params.a, x, y, z) * intDz;
    emag = intDx * intDdx + intDy * intDdy + intDz * intDdz + TINY;
    intBx = interpolate(Bx, ijk, Stagger(0b001), Stagger(0b011),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intBy = interpolate(By, ijk, Stagger(0b010), Stagger(0b011),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intBz = interpolate(Bz, ijk, Stagger(0b100), Stagger(0b011),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intBdx = get_gamma_d11(dev_params.a, x, y, z) * intBx +
             get_gamma_d12(dev_params.a, x, y, z) * intBy +
             get_gamma_d13(dev_params.a, x, y, z) * intBz;
    intBdy = get_gamma_d12(dev_params.a, x, y, z) * intBx +
             get_gamma_d22(dev_params.a, x, y, z) * intBy +
             get_gamma_d23(dev_params.a, x, y, z) * intBz;
    intBdz = get_gamma_d13(dev_params.a, x, y, z) * intBx +
             get_gamma_d23(dev_params.a, x, y, z) * intBy +
             get_gamma_d33(dev_params.a, x, y, z) * intBz;
    bmag = intBx * intBdx + intBy * intBdy + intBz * intBdz + TINY;
    if (emag > bmag) {
      temp = sqrt(bmag / emag);
    } else {
      temp = 1.0;
    }
    Dz[ijk] = temp * dDz[ijk];
  }
}

HOST_DEVICE Scalar sigma(Scalar a, Scalar x, Scalar y, Scalar z, Scalar r0, Scalar d, Scalar sig0) {
  Scalar r = get_r(a, x, y, z);
  return sig0 * cube((r0 - r) / d);
}

HOST_DEVICE Scalar pmlsigma(Scalar x, Scalar xl, Scalar xh, Scalar pmlscale, Scalar sig0) {
  if (x > xh) return sig0 * pow((x - xh) / pmlscale, 3.0);
  else if (x < xl) return sig0 * pow((xl - x) / pmlscale, 3.0);
  else return 0.0;
}


__global__ void
kernel_absorbing_boundary_thread(const Scalar *Dnx, const Scalar *Dny, const Scalar *Dnz,
                      const Scalar *Bnx, const Scalar *Bny, const Scalar *Bnz,
                      Scalar *Dx, Scalar *Dy, Scalar *Dz, 
                      Scalar *Bx, Scalar *By, Scalar *Bz,
                      int shift) {
  Scalar x, y, z;
  size_t ijk;
  Scalar rH = 1.0 + sqrt(1.0 - square(dev_params.a));
  Scalar r1 = 0.8 * rH;
  Scalar r2 = 0.01 * rH;
  Scalar dd = 0.2 * rH;
  Scalar sig, sigx, sigy, sigz;
  Scalar sig0 = dev_params.sigpml;
  Scalar dx = dev_grid.delta[0] / 2.0;
  Scalar dy = dev_grid.delta[1] / 2.0;
  Scalar dz = dev_grid.delta[2] / 2.0;

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

    // Inside event horizon
    Scalar r = get_r(dev_params.a, x, y, z);
    if (r < r1) {
      // Dx
      sig = sigma(dev_params.a, x + dx, y, z, r1, dd, sig0) * dev_params.dt;
      if (sig > TINY) Dx[ijk] = exp(- sig) * Dnx[ijk] + (1.0 - exp(- sig)) / sig * (Dx[ijk] - Dnx[ijk]); 
      // if (sig > 0) Dx[ijk] = exp(- sig) * Dx[ijk]; 
      // Dy
      sig = sigma(dev_params.a, x, y + dy, z, r1, dd, sig0) * dev_params.dt;
      if (sig > TINY) Dy[ijk] = exp(- sig) * Dny[ijk] + (1.0 - exp(- sig)) / sig * (Dy[ijk] - Dny[ijk]); 
      // if (sig > 0) Dy[ijk] = exp(- sig) * Dy[ijk]; 
      // Dz
      sig = sigma(dev_params.a, x, y, z + dz, r1, dd, sig0) * dev_params.dt;
      if (sig > TINY) Dz[ijk] = exp(- sig) * Dnz[ijk] + (1.0 - exp(- sig)) / sig * (Dz[ijk] - Dnz[ijk]);
      // if (sig > 0) Dz[ijk] = exp(- sig) * Dz[ijk];
      // Bx
      sig = sigma(dev_params.a, x, y + dy, z + dz, r1, dd, sig0) * dev_params.dt;
      if (sig > TINY) Bx[ijk] = exp(- sig) * Bnx[ijk] + (1.0 - exp(- sig)) / sig * (Bx[ijk] - Bnx[ijk]);
      // if (sig > 0) Bx[ijk] = exp(- sig) * Bx[ijk]; 
      // By
      sig = sigma(dev_params.a, x + dx, y, z + dz, r1, dd, sig0) * dev_params.dt;
      if (sig > TINY) By[ijk] = exp(- sig) * Bny[ijk] + (1.0 - exp(- sig)) / sig * (By[ijk] - Bny[ijk]);
      // if (sig > 0) By[ijk] = exp(- sig) * By[ijk];
      // Bz
      sig = sigma(dev_params.a, x + dx, y + dy, z, r1, dd, sig0) * dev_params.dt;
      if (sig > TINY) Bz[ijk] = exp(- sig) * Bnz[ijk] + (1.0 - exp(- sig)) / sig * (Bz[ijk] - Bnz[ijk]);
      // if (sig > 0) Bz[ijk] = exp(- sig) * Bz[ijk];   
    }
    if (r < r2) {
      Dx[ijk] = 0;
      Dy[ijk] = 0;
      Dz[ijk] = 0;
      Bx[ijk] = 0;
      By[ijk] = 0;
      Bz[ijk] = 0;
    }

    // Outer boundary
    Scalar xh = dev_params.lower[0] + dev_params.size[0] - dev_params.pml[0] * dev_grid.delta[0];
    Scalar xl = dev_params.lower[0] + dev_params.pml[0] * dev_grid.delta[0];
    Scalar yh = dev_params.lower[1] + dev_params.size[1] - dev_params.pml[1] * dev_grid.delta[1];
    Scalar yl = dev_params.lower[1] + dev_params.pml[1] * dev_grid.delta[1];
    Scalar zh = dev_params.lower[2] + dev_params.size[2] - dev_params.pml[2] * dev_grid.delta[2];
    Scalar zl = dev_params.lower[2] + dev_params.pml[2] * dev_grid.delta[2];
    if (x > xh || x < xl || y > yh || y < yl || z > zh || z < zl) {
      sigx = pmlsigma(x, xl, xh, dev_params.pmllen * dev_grid.delta[0], dev_params.sigpml);
      sigy = pmlsigma(y, yl, yh, dev_params.pmllen * dev_grid.delta[0], dev_params.sigpml);
      sigz = pmlsigma(z, zl, zh, dev_params.pmllen * dev_grid.delta[0], dev_params.sigpml);
      sig = sigx + sigy + sigz;
      if (sig > TINY) {
        Dx[ijk] = exp(-sig) * Dnx[ijk] + (1.0 - exp(-sig)) / sig * (Dx[ijk] - Dnx[ijk]);
        Dy[ijk] = exp(-sig) * Dny[ijk] + (1.0 - exp(-sig)) / sig * (Dy[ijk] - Dny[ijk]);
        Dz[ijk] = exp(-sig) * Dnz[ijk] + (1.0 - exp(-sig)) / sig * (Dz[ijk] - Dnz[ijk]); 
        Bx[ijk] = exp(-sig) * Bnx[ijk] + (1.0 - exp(-sig)) / sig * (Bx[ijk] - Bnx[ijk]);
        By[ijk] = exp(-sig) * Bny[ijk] + (1.0 - exp(-sig)) / sig * (By[ijk] - Bny[ijk]);
        Bz[ijk] = exp(-sig) * Bnz[ijk] + (1.0 - exp(-sig)) / sig * (Bz[ijk] - Bnz[ijk]); 
      }
    }
    if (x <= dev_params.lower[0] || x >= dev_params.lower[0] + dev_params.size[0]
      || y <= dev_params.lower[1] || y >= dev_params.lower[1] + dev_params.size[1]
      || z <= dev_params.lower[2] || z >= dev_params.lower[2] + dev_params.size[2]) {
      Dx[ijk] = Dnx[ijk];
      Dy[ijk] = Dny[ijk];
      Dz[ijk] = Dnz[ijk];
      Bx[ijk] = Bnx[ijk];
      By[ijk] = Bny[ijk];
      Bz[ijk] = Bnz[ijk];
    }
  }
}


field_solver_gr::field_solver_gr(sim_data &mydata, sim_environment& env) : m_data(mydata), m_env(env) {
  // Note that m_data.E contain D upper components
  // I think we need at least 6 guard cells
  Dn = vector_field<Scalar>(m_data.env.grid());
  dD = vector_field<Scalar>(m_data.env.grid());
  Ed = vector_field<Scalar>(m_data.env.grid());
  Dn.copy_stagger(m_data.E);
  dD.copy_stagger(m_data.E);
  Ed.copy_stagger(m_data.E);
  Dn.initialize();
  dD.initialize();
  Ed.initialize();

  Bn = vector_field<Scalar>(m_data.env.grid());
  dB = vector_field<Scalar>(m_data.env.grid());
  Hd = vector_field<Scalar>(m_data.env.grid());
  Bn.copy_stagger(m_data.B);
  dB.copy_stagger(m_data.B);
  Hd.copy_stagger(m_data.B);
  Bn.initialize();
  dB.initialize();
  Hd.initialize();

  rho = multi_array<Scalar>(m_data.env.grid().extent());
  rho.assign_dev(0.0);

  blockGroupSize = dim3((m_data.env.grid().reduced_dim(0) + m_env.params().shift_ghost * 2 + blockSize.x - 1) / blockSize.x,
                        (m_data.env.grid().reduced_dim(1) + m_env.params().shift_ghost * 2 + blockSize.y - 1) / blockSize.y,
                        (m_data.env.grid().reduced_dim(2) + m_env.params().shift_ghost * 2 + blockSize.z - 1) / blockSize.z);
  std::cout << blockSize.x << ", " << blockSize.y << ", " << blockSize.z << std::endl;
  std::cout << blockGroupSize.x << ", " << blockGroupSize.y << ", " << blockGroupSize.z << std::endl;
}

field_solver_gr::~field_solver_gr() {}

void
field_solver_gr::evolve_fields_gr() {
  RANGE_PUSH("Compute", CLR_GREEN);
  copy_fields_gr();

  // substep #1:
  compute_E_gr();
  compute_H_gr();
  rk_push_gr();
  rk_update_gr(1.0, 0.0, 1.0);
  if (m_env.params().check_egb) {
    check_eGTb_gr();
    std::cout << "substep 1, check_eGTb done." << std::endl;
  }
  CudaSafeCall(cudaDeviceSynchronize());
  RANGE_POP;
  m_env.send_guard_cells(m_data);

  // substep #2:
  RANGE_PUSH("Compute", CLR_GREEN);
  compute_E_gr();
  compute_H_gr();
  rk_push_gr();
  rk_update_gr(0.75, 0.25, 0.25);
  if (m_env.params().check_egb) {
    check_eGTb_gr();
    std::cout << "substep 2, check_eGTb done." << std::endl;
  }
  CudaSafeCall(cudaDeviceSynchronize());
  RANGE_POP;
  m_env.send_guard_cells(m_data);

  // substep #3:
  RANGE_PUSH("Compute", CLR_GREEN);
  compute_E_gr();
  compute_H_gr();
  rk_push_gr();
  rk_update_gr(1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0);
  if (m_env.params().clean_ep) {
    clean_epar_gr();
    std::cout << "substep 3, clean_epar done." << std::endl;
  }
  if (m_env.params().check_egb) {
    check_eGTb_gr();
    std::cout << "substep 3, check_eGTb done." << std::endl;
  }
  absorbing_boundary();
  CudaSafeCall(cudaDeviceSynchronize());
  RANGE_POP;

  m_env.send_guard_cells(m_data);
}

void
field_solver_gr::copy_fields_gr() {
  // `En = E, Bn = B`:
  Dn.copy_from(m_data.E);
  Bn.copy_from(m_data.B);
  dD.initialize();
  dB.initialize();
}

void
field_solver_gr::rk_push_gr() {
  // `rho = div E`
  // kernel_compute_rho<<<gridSize, blockSize>>>(
  kernel_compute_rho_gr_thread<<<blockGroupSize, blockSize>>>(
      m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
      rho.dev_ptr(), m_env.params().shift_ghost);
  CudaCheckError();
  // `dE = curl B - curl B0 - j, dB = -curl E`
  // kernel_rk_push<<<g, blockSize>>>(
  kernel_rk_push_gr_thread<<<blockGroupSize, blockSize>>>(
      Ed.dev_ptr(0), Ed.dev_ptr(1), Ed.dev_ptr(2),
      Hd.dev_ptr(0), Hd.dev_ptr(1), Hd.dev_ptr(2),
      m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2),
      m_data.B0.dev_ptr(0), m_data.B0.dev_ptr(1), m_data.B0.dev_ptr(2),
      dD.dev_ptr(0), dD.dev_ptr(1), dD.dev_ptr(2), dB.dev_ptr(0),
      dB.dev_ptr(1), dB.dev_ptr(2), rho.dev_ptr(), m_env.params().shift_ghost);
  CudaCheckError();
}

void
field_solver_gr::rk_update_gr(Scalar rk_c1, Scalar rk_c2, Scalar rk_c3) {
  // `E = c1 En + c2 E + c3 dE, B = c1 Bn + c2 B + c3 dB`
  // kernel_rk_update<<<dim3(8, 16, 16), dim3(64, 4, 4)>>>(
  kernel_rk_update_gr_thread<<<blockGroupSize, blockSize>>>(
      m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
      m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2),
      Dn.dev_ptr(0), Dn.dev_ptr(1), Dn.dev_ptr(2), Bn.dev_ptr(0),
      Bn.dev_ptr(1), Bn.dev_ptr(2), dD.dev_ptr(0), dD.dev_ptr(1),
      dD.dev_ptr(2), dB.dev_ptr(0), dB.dev_ptr(1), dB.dev_ptr(2), rk_c1,
      rk_c2, rk_c3, m_env.params().shift_ghost);
  CudaCheckError();
}

void
field_solver_gr::compute_E_gr() {
  kernel_compute_E_gr_thread<<<blockGroupSize, blockSize>>>(
    m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
    m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2),
    Ed.dev_ptr(0), Ed.dev_ptr(1), Ed.dev_ptr(2), m_env.params().shift_ghost);
  CudaCheckError();
}

void
field_solver_gr::compute_H_gr() {
  kernel_compute_H_gr_thread<<<blockGroupSize, blockSize>>>(
    m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
    m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2),
    m_data.B0.dev_ptr(0), m_data.B0.dev_ptr(1), m_data.B0.dev_ptr(2),
    Hd.dev_ptr(0), Hd.dev_ptr(1), Hd.dev_ptr(2), m_env.params().shift_ghost);
  CudaCheckError();
}


void
field_solver_gr::clean_epar_gr() {
  // clean `E || B`
  // kernel_clean_epar<<<gridSize, blockSize>>>(
  kernel_clean_epar_gr_thread<<<blockGroupSize, blockSize>>>(
      m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
      m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2),
      dD.dev_ptr(0), dD.dev_ptr(1), dD.dev_ptr(2), m_env.params().shift_ghost);
  CudaCheckError();
}

void
field_solver_gr::check_eGTb_gr() {
  // renormalizing `E > B`
  // kernel_check_eGTb<<<dim3(8, 16, 16), dim3(32, 4, 4)>>>(
  kernel_check_eGTb_gr_thread<<<blockGroupSize, blockSize>>>(
      dD.dev_ptr(0), dD.dev_ptr(1), dD.dev_ptr(2), m_data.E.dev_ptr(0),
      m_data.E.dev_ptr(1), m_data.E.dev_ptr(2), m_data.B.dev_ptr(0),
      m_data.B.dev_ptr(1), m_data.B.dev_ptr(2), m_env.params().shift_ghost);
  CudaCheckError();
}

void
field_solver_gr::absorbing_boundary() {
  kernel_absorbing_boundary_thread<<<blockGroupSize, blockSize>>>(
      Dn.dev_ptr(0), Dn.dev_ptr(1), Dn.dev_ptr(2), Bn.dev_ptr(0),
      Bn.dev_ptr(1), Bn.dev_ptr(2), m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), 
      m_data.E.dev_ptr(2), m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), 
      m_data.B.dev_ptr(2), m_env.params().shift_ghost);
  CudaCheckError();
}

}  // namespace Coffee
