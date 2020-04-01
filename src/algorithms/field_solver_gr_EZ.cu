#include "algorithms/finite_diff.h"
#include "boundary.h"
#include "cuda/constant_mem.h"
#include "cuda/constant_mem_func.h"
#include "cuda/cuda_control.h"
#include "cuda/cuda_utility.h"
#include "field_solver_gr_EZ.h"
#include "interpolation.h"
#include "metric_cks.h"
#include "utils/nvproftool.h"
#include "utils/timer.h"

#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 2
#define BLOCK_SIZE_Z 2

#define FFE_DISSIPATION_ORDER 6

namespace Coffee {

using namespace CKS;

// static dim3 gridSize(8, 16, 16);
static dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);

static dim3 blockGroupSize;

__device__ __forceinline__ Scalar
dfdx(const Scalar *f, int ijk) {
  return df1(f, ijk, 1, dev_grid.inv_delta[0]);
}

__device__ __forceinline__ Scalar
dfdy(const Scalar *f, int ijk) {
  return df1(f, ijk, dev_grid.dims[0], dev_grid.inv_delta[1]);
}

__device__ __forceinline__ Scalar
dfdz(const Scalar *f, int ijk) {
  return df1(f, ijk, dev_grid.dims[0] * dev_grid.dims[1],
             dev_grid.inv_delta[2]);
}

__device__ Scalar
div4(const Scalar *fx, const Scalar *fy, const Scalar *fz, int ijk,
     Scalar x, Scalar y, Scalar z) {
  Scalar tmpx =
      (fx[ijk - 2] * get_sqrt_gamma(dev_params.a,
                                    x - 2.0 * dev_grid.delta[0], y, z) -
       8.0 * fx[ijk - 1] *
           get_sqrt_gamma(dev_params.a, x - 1.0 * dev_grid.delta[0], y,
                          z) +
       8.0 * fx[ijk + 1] *
           get_sqrt_gamma(dev_params.a, x + 1.0 * dev_grid.delta[0], y,
                          z) -
       fx[ijk + 2] * get_sqrt_gamma(dev_params.a,
                                    x + 2.0 * dev_grid.delta[0], y,
                                    z)) /
      12.0 / dev_grid.delta[0];
  int s = dev_grid.dims[0];
  Scalar tmpy = (fy[ijk - 2 * s] *
                     get_sqrt_gamma(dev_params.a, x,
                                    y - 2.0 * dev_grid.delta[1], z) -
                 8.0 * fy[ijk - 1 * s] *
                     get_sqrt_gamma(dev_params.a, x,
                                    y - 1.0 * dev_grid.delta[1], z) +
                 8.0 * fy[ijk + 1 * s] *
                     get_sqrt_gamma(dev_params.a, x,
                                    y + 1.0 * dev_grid.delta[1], z) -
                 fy[ijk + 2 * s] *
                     get_sqrt_gamma(dev_params.a, x,
                                    y + 2.0 * dev_grid.delta[1], z)) /
                12.0 / dev_grid.delta[1];
  s = dev_grid.dims[0] * dev_grid.dims[1];
  Scalar tmpz =
      (fz[ijk - 2 * s] * get_sqrt_gamma(dev_params.a, x, y,
                                        z - 2.0 * dev_grid.delta[2]) -
       8.0 * fz[ijk - 1 * s] *
           get_sqrt_gamma(dev_params.a, x, y,
                          z - 1.0 * dev_grid.delta[2]) +
       8.0 * fz[ijk + 1 * s] *
           get_sqrt_gamma(dev_params.a, x, y,
                          z + 1.0 * dev_grid.delta[2]) -
       fz[ijk + 2 * s] * get_sqrt_gamma(dev_params.a, x, y,
                                        z + 2.0 * dev_grid.delta[2])) /
      12.0 / dev_grid.delta[2];
  return (tmpx + tmpy + tmpz) / get_sqrt_gamma(dev_params.a, x, y, z);
}

__global__ void
kernel_compute_E_gr(const Scalar *Dx, const Scalar *Dy,
                    const Scalar *Dz, const Scalar *Bx,
                    const Scalar *By, const Scalar *Bz, Scalar *Ex,
                    Scalar *Ey, Scalar *Ez, int shift) {
  size_t ijk;
  Scalar Ddx, Ddy, Ddz;
  Scalar x, y, z;

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int k = threadIdx.z + blockIdx.z * blockDim.z;
  if (i < dev_grid.dims[0] && j < dev_grid.dims[1] &&
      k < dev_grid.dims[2]) {
    ijk = i + j * dev_grid.dims[0] +
          k * dev_grid.dims[0] * dev_grid.dims[1];

    x = dev_grid.pos(0, i, 1);
    y = dev_grid.pos(1, j, 1);
    z = dev_grid.pos(2, k, 1);

    // Calculate Ex
    Ddx = get_gamma_d11(dev_params.a, x, y, z) * Dx[ijk] +
          get_gamma_d12(dev_params.a, x, y, z) * Dy[ijk] +
          get_gamma_d13(dev_params.a, x, y, z) * Dz[ijk];
    Ex[ijk] = get_alpha(dev_params.a, x, y, z) * Ddx +
              get_sqrt_gamma(dev_params.a, x, y, z) *
                  (get_beta_u2(dev_params.a, x, y, z) * Bz[ijk] -
                   get_beta_u3(dev_params.a, x, y, z) * By[ijk]);

    // Calculate Ey
    Ddy = get_gamma_d12(dev_params.a, x, y, z) * Dx[ijk] +
          get_gamma_d22(dev_params.a, x, y, z) * Dy[ijk] +
          get_gamma_d23(dev_params.a, x, y, z) * Dz[ijk];
    Ey[ijk] = get_alpha(dev_params.a, x, y, z) * Ddy +
              get_sqrt_gamma(dev_params.a, x, y, z) *
                  (get_beta_u3(dev_params.a, x, y, z) * Bx[ijk] -
                   get_beta_u1(dev_params.a, x, y, z) * Bz[ijk]);

    // Calculate Ez
    Ddz = get_gamma_d13(dev_params.a, x, y, z) * Dx[ijk] +
          get_gamma_d23(dev_params.a, x, y, z) * Dy[ijk] +
          get_gamma_d33(dev_params.a, x, y, z) * Dz[ijk];
    Ez[ijk] = get_alpha(dev_params.a, x, y, z) * Ddz +
              get_sqrt_gamma(dev_params.a, x, y, z) *
                  (get_beta_u1(dev_params.a, x, y, z) * By[ijk] -
                   get_beta_u2(dev_params.a, x, y, z) * Bx[ijk]);
  }
}

__global__ void
kernel_compute_H_gr(const Scalar *Dx, const Scalar *Dy,
                    const Scalar *Dz, const Scalar *Bx,
                    const Scalar *By, const Scalar *Bz, Scalar *Hx,
                    Scalar *Hy, Scalar *Hz, int shift) {
  size_t ijk;
  Scalar Bdx, Bdy, Bdz;
  Scalar x, y, z;

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int k = threadIdx.z + blockIdx.z * blockDim.z;
  if (i < dev_grid.dims[0] && j < dev_grid.dims[1] &&
      k < dev_grid.dims[2]) {
    ijk = i + j * dev_grid.dims[0] +
          k * dev_grid.dims[0] * dev_grid.dims[1];
    x = dev_grid.pos(0, i, 1);
    y = dev_grid.pos(1, j, 1);
    z = dev_grid.pos(2, k, 1);
    // Calculate Hx
    Bdx = get_gamma_d11(dev_params.a, x, y, z) * Bx[ijk] +
          get_gamma_d12(dev_params.a, x, y, z) * By[ijk] +
          get_gamma_d13(dev_params.a, x, y, z) * Bz[ijk];
    Hx[ijk] = get_alpha(dev_params.a, x, y, z) * Bdx -
              get_sqrt_gamma(dev_params.a, x, y, z) *
                  (get_beta_u2(dev_params.a, x, y, z) * Dz[ijk] -
                   get_beta_u3(dev_params.a, x, y, z) * Dy[ijk]);

    // Calculate Hy
    Bdy = get_gamma_d12(dev_params.a, x, y, z) * Bx[ijk] +
          get_gamma_d22(dev_params.a, x, y, z) * By[ijk] +
          get_gamma_d23(dev_params.a, x, y, z) * Bz[ijk];
    Hy[ijk] = get_alpha(dev_params.a, x, y, z) * Bdy -
              get_sqrt_gamma(dev_params.a, x, y, z) *
                  (get_beta_u3(dev_params.a, x, y, z) * Dx[ijk] -
                   get_beta_u1(dev_params.a, x, y, z) * Dz[ijk]);

    // Calculate Hz
    Bdz = get_gamma_d13(dev_params.a, x, y, z) * Bx[ijk] +
          get_gamma_d23(dev_params.a, x, y, z) * By[ijk] +
          get_gamma_d33(dev_params.a, x, y, z) * Bz[ijk];
    Hz[ijk] = get_alpha(dev_params.a, x, y, z) * Bdz -
              get_sqrt_gamma(dev_params.a, x, y, z) *
                  (get_beta_u1(dev_params.a, x, y, z) * Dy[ijk] -
                   get_beta_u2(dev_params.a, x, y, z) * Dx[ijk]);
  }
}

__device__ void
j_ext(Scalar x, Scalar y, Scalar z, Scalar *jnew) {
  Scalar r = get_r(dev_params.a, x, y, z);
  jnew[0] = 0.0;
  jnew[1] = 0.0;
  jnew[2] = 0.0;
  if (std::abs(z) < dev_grid.delta[2] * (3.0 + 1.0 / 4.0)) {
    Scalar tmp = (r - dev_params.rin) * 2.0 * M_PI / dev_params.rj;
    if (tmp < 2.0 * M_PI && tmp > 0) {
      Scalar iphi = dev_params.b0 * sin(tmp) /
                    pow(r / dev_params.rin, dev_params.al);
      jnew[0] = -y / r * iphi;
      jnew[1] = x / r * iphi;
    }
  }
}

__global__ void
kernel_rk_step1_gr(const Scalar *Ex, const Scalar *Ey, const Scalar *Ez,
                   const Scalar *Hx, const Scalar *Hy, const Scalar *Hz,
                   const Scalar *Dx, const Scalar *Dy, const Scalar *Dz,
                   const Scalar *Bx, const Scalar *By, const Scalar *Bz,
                   Scalar *dDx, Scalar *dDy, Scalar *dDz, Scalar *dBx,
                   Scalar *dBy, Scalar *dBz, Scalar *jx, Scalar *jy,
                   Scalar *jz, Scalar *DivB, Scalar *DivE,
                   const Scalar *P, Scalar *dP, int shift, Scalar As) {
  size_t ijk;
  Scalar Jx, Jy, Jz, Jp, jd[3] = {0.0, 0.0, 0.0};
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
    Scalar r = get_r(dev_params.a, x, y, z);
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

    if (dev_params.calc_current) {
      Scalar B2 = Bx[ijk] * Bdx + By[ijk] * Bdy + Bz[ijk] * Bdz;
      if (B2 < TINY) B2 = TINY;

      Jp = (Bdx * rotHx + Bdy * rotHy + Bdz * rotHz) -
           (Ddx * rotEx + Ddy * rotEy + Ddz * rotEz);
      Jx = (divD * (Ey[ijk] * Bdz - Ez[ijk] * Bdy) / gmsqrt +
            Jp * Bx[ijk]) /
           B2;
      Jy = (divD * (Ez[ijk] * Bdx - Ex[ijk] * Bdz) / gmsqrt +
            Jp * By[ijk]) /
           B2;
      Jz = (divD * (Ex[ijk] * Bdy - Ey[ijk] * Bdx) / gmsqrt +
            Jp * Bz[ijk]) /
           B2;
    } else {
      Jx = 0.0;
      Jy = 0.0;
      Jz = 0.0;
    }

    if (dev_params.ext_current) {
      Scalar x = dev_grid.pos(0, i, 1);
      Scalar y = dev_grid.pos(1, j, 1);
      Scalar z = dev_grid.pos(2, k, 1);
      j_ext(x, y, z, jd);
      Jx += jd[0];
      Jy += jd[1];
      Jz += jd[2];
    }

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

    Scalar r0 = 5.0;
    Scalar del = 1.0;
    Scalar shape = 0.5 * (1.0 - tanh((r - r0) / del));

    if (dev_params.divB_clean) {
      // dBx[ijk] =
      //     As * dBx[ijk] +
      //     dev_params.dt * (- rotEx - alpha * Pxu +
      //                      get_beta_u1(dev_params.a, x, y, z) * divB);
      // dBy[ijk] =
      //     As * dBy[ijk] +
      //     dev_params.dt * (- rotEy - alpha * Pyu +
      //                      get_beta_u2(dev_params.a, x, y, z) * divB);
      // dBz[ijk] =
      //     As * dBz[ijk] +
      //     dev_params.dt * (- rotEz - alpha * Pzu +
      //                      get_beta_u3(dev_params.a, x, y, z) * divB);
      dBx[ijk] =
          As * dBx[ijk] +
          dev_params.dt *
              (-rotEx + (-alpha * Pxu +
                         get_beta_u1(dev_params.a, x, y, z) * divB) *
                            shape);
      dBy[ijk] =
          As * dBy[ijk] +
          dev_params.dt *
              (-rotEy + (-alpha * Pyu +
                         get_beta_u2(dev_params.a, x, y, z) * divB) *
                            shape);
      dBz[ijk] =
          As * dBz[ijk] +
          dev_params.dt *
              (-rotEz + (-alpha * Pzu +
                         get_beta_u3(dev_params.a, x, y, z) * divB) *
                            shape);
    } else {
      dBx[ijk] = As * dBx[ijk] - dev_params.dt * rotEx;
      dBy[ijk] = As * dBy[ijk] - dev_params.dt * rotEy;
      dBz[ijk] = As * dBz[ijk] - dev_params.dt * rotEz;
    }

    dDx[ijk] = As * dDx[ijk] + dev_params.dt * (rotHx - Jx);
    dDy[ijk] = As * dDy[ijk] + dev_params.dt * (rotHy - Jy);
    dDz[ijk] = As * dDz[ijk] + dev_params.dt * (rotHz - Jz);

    dP[ijk] =
        As * dP[ijk] +
        dev_params.dt *
            (-alpha * divB + get_beta_u1(dev_params.a, x, y, z) * Pxd +
             get_beta_u2(dev_params.a, x, y, z) * Pyd +
             get_beta_u3(dev_params.a, x, y, z) * Pzd -
             alpha * P[ijk] / dev_params.tau);

    jx[ijk] = Jx;
    jy[ijk] = Jy;
    jz[ijk] = Jz;
    DivB[ijk] = divB;
    DivE[ijk] = divD;

    // Inside the damping layer
    //    Scalar xh =
    //        dev_params.lower[0] + dev_params.size[0] -
    //        (dev_params.pml[0] + dev_params.guard[0]) *
    //        dev_grid.delta[0];
    //    Scalar xl =
    //        dev_params.lower[0] +
    //        (dev_params.pml[0] + dev_params.guard[0]) *
    //        dev_grid.delta[0];
    //    Scalar yh =
    //        dev_params.lower[1] + dev_params.size[1] -
    //        (dev_params.pml[1] + dev_params.guard[1]) *
    //        dev_grid.delta[1];
    //    Scalar yl =
    //        dev_params.lower[1] +
    //        (dev_params.pml[1] + dev_params.guard[1]) *
    //        dev_grid.delta[1];
    //    Scalar zh =
    //        dev_params.lower[2] + dev_params.size[2] -
    //        (dev_params.pml[2] + dev_params.guard[2]) *
    //        dev_grid.delta[2];
    //    Scalar zl =
    //        dev_params.lower[2] +
    //        (dev_params.pml[2] + dev_params.guard[2]) *
    //        dev_grid.delta[2];
    //    if (x > xh || x < xl || y > yh || y < yl || z > zh || z < zl)
    //    {
    //      dBx[ijk] = As * dBx[ijk] - dev_params.dt * rotEx;
    //      dBy[ijk] = As * dBy[ijk] - dev_params.dt * rotEy;
    //      dBz[ijk] = As * dBz[ijk] - dev_params.dt * rotEz;
    //      dP[ijk] = 0.0;
    //    }
  }
}

__global__ void
kernel_rk_step2_gr(Scalar *Dx, Scalar *Dy, Scalar *Dz, Scalar *Bx,
                   Scalar *By, Scalar *Bz, const Scalar *dDx,
                   const Scalar *dDy, const Scalar *dDz,
                   const Scalar *dBx, const Scalar *dBy,
                   const Scalar *dBz, Scalar *P, const Scalar *dP,
                   int shift, Scalar Bs) {
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

    Dx[ijk] = Dx[ijk] + Bs * dDx[ijk];
    Dy[ijk] = Dy[ijk] + Bs * dDy[ijk];
    Dz[ijk] = Dz[ijk] + Bs * dDz[ijk];

    Bx[ijk] = Bx[ijk] + Bs * dBx[ijk];
    By[ijk] = By[ijk] + Bs * dBy[ijk];
    Bz[ijk] = Bz[ijk] + Bs * dBz[ijk];

    P[ijk] = P[ijk] + Bs * dP[ijk];

    // Inside the damping layer
    //    Scalar x = dev_grid.pos(0, i, 1);
    //    Scalar y = dev_grid.pos(1, j, 1);
    //    Scalar z = dev_grid.pos(2, k, 1);
    //    Scalar xh =
    //        dev_params.lower[0] + dev_params.size[0] -
    //        (dev_params.pml[0] + dev_params.guard[0]) *
    //        dev_grid.delta[0];
    //    Scalar xl =
    //        dev_params.lower[0] +
    //        (dev_params.pml[0] + dev_params.guard[0]) *
    //        dev_grid.delta[0];
    //    Scalar yh =
    //        dev_params.lower[1] + dev_params.size[1] -
    //        (dev_params.pml[1] + dev_params.guard[1]) *
    //        dev_grid.delta[1];
    //    Scalar yl =
    //        dev_params.lower[1] +
    //        (dev_params.pml[1] + dev_params.guard[1]) *
    //        dev_grid.delta[1];
    //    Scalar zh =
    //        dev_params.lower[2] + dev_params.size[2] -
    //        (dev_params.pml[2] + dev_params.guard[2]) *
    //        dev_grid.delta[2];
    //    Scalar zl =
    //        dev_params.lower[2] +
    //        (dev_params.pml[2] + dev_params.guard[2]) *
    //        dev_grid.delta[2];
    //    if (x > xh || x < xl || y > yh || y < yl || z > zh || z < zl)
    //    {
    //      P[ijk] = 0.0;
    //    }
  }
}

__global__ void
kernel_clean_epar_gr(Scalar *Dx, Scalar *Dy, Scalar *Dz,
                     const Scalar *Bx, const Scalar *By,
                     const Scalar *Bz, int shift) {
  Scalar Bdx, Bdy, Bdz, B2, DB;
  Scalar x, y, z;
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

    x = dev_grid.pos(0, i, 1);
    y = dev_grid.pos(1, j, 1);
    z = dev_grid.pos(2, k, 1);

    Bdx = get_gamma_d11(dev_params.a, x, y, z) * Bx[ijk] +
          get_gamma_d12(dev_params.a, x, y, z) * By[ijk] +
          get_gamma_d13(dev_params.a, x, y, z) * Bz[ijk];
    Bdy = get_gamma_d12(dev_params.a, x, y, z) * Bx[ijk] +
          get_gamma_d22(dev_params.a, x, y, z) * By[ijk] +
          get_gamma_d23(dev_params.a, x, y, z) * Bz[ijk];
    Bdz = get_gamma_d13(dev_params.a, x, y, z) * Bx[ijk] +
          get_gamma_d23(dev_params.a, x, y, z) * By[ijk] +
          get_gamma_d33(dev_params.a, x, y, z) * Bz[ijk];
    B2 = Bx[ijk] * Bdx + By[ijk] * Bdy + Bz[ijk] * Bdz;
    if (B2 < TINY) B2 = TINY;
    DB = Dx[ijk] * Bdx + Dy[ijk] * Bdy + Dz[ijk] * Bdz;

    Dx[ijk] = Dx[ijk] - DB * Bx[ijk] / B2;
    Dy[ijk] = Dy[ijk] - DB * By[ijk] / B2;
    Dz[ijk] = Dz[ijk] - DB * Bz[ijk] / B2;
  }
}

__global__ void
kernel_check_eGTb_gr(Scalar *Dx, Scalar *Dy, Scalar *Dz,
                     const Scalar *Bx, const Scalar *By,
                     const Scalar *Bz, int shift) {
  size_t ijk;
  Scalar temp;
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

    Scalar Ddx = get_gamma_d11(dev_params.a, x, y, z) * Dx[ijk] +
                 get_gamma_d12(dev_params.a, x, y, z) * Dy[ijk] +
                 get_gamma_d13(dev_params.a, x, y, z) * Dz[ijk];
    Scalar Ddy = get_gamma_d12(dev_params.a, x, y, z) * Dx[ijk] +
                 get_gamma_d22(dev_params.a, x, y, z) * Dy[ijk] +
                 get_gamma_d23(dev_params.a, x, y, z) * Dz[ijk];
    Scalar Ddz = get_gamma_d13(dev_params.a, x, y, z) * Dx[ijk] +
                 get_gamma_d23(dev_params.a, x, y, z) * Dy[ijk] +
                 get_gamma_d33(dev_params.a, x, y, z) * Dz[ijk];
    Scalar D2 = Dx[ijk] * Ddx + Dy[ijk] * Ddy + Dz[ijk] * Ddz;
    if (D2 < TINY) D2 = TINY;

    Scalar Bdx = get_gamma_d11(dev_params.a, x, y, z) * Bx[ijk] +
                 get_gamma_d12(dev_params.a, x, y, z) * By[ijk] +
                 get_gamma_d13(dev_params.a, x, y, z) * Bz[ijk];
    Scalar Bdy = get_gamma_d12(dev_params.a, x, y, z) * Bx[ijk] +
                 get_gamma_d22(dev_params.a, x, y, z) * By[ijk] +
                 get_gamma_d23(dev_params.a, x, y, z) * Bz[ijk];
    Scalar Bdz = get_gamma_d13(dev_params.a, x, y, z) * Bx[ijk] +
                 get_gamma_d23(dev_params.a, x, y, z) * By[ijk] +
                 get_gamma_d33(dev_params.a, x, y, z) * Bz[ijk];
    Scalar B2 = Bx[ijk] * Bdx + By[ijk] * Bdy + Bz[ijk] * Bdz;
    if (D2 > B2) {
      temp = std::sqrt(B2 / D2);
    } else {
      temp = 1.0;
    }
    Dx[ijk] = temp * Dx[ijk];
    Dy[ijk] = temp * Dy[ijk];
    Dz[ijk] = temp * Dz[ijk];
  }
}

__global__ void
kernel_KO_step1_gr(Scalar *Ex, Scalar *Ey, Scalar *Ez, Scalar *Bx,
                   Scalar *By, Scalar *Bz, Scalar *Ex_tmp,
                   Scalar *Ey_tmp, Scalar *Ez_tmp, Scalar *Bx_tmp,
                   Scalar *By_tmp, Scalar *Bz_tmp, Scalar *P,
                   Scalar *P_tmp, int shift) {
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

    Ex_tmp[ijk] = KO(Ex, ijk, dev_grid);
    Ey_tmp[ijk] = KO(Ey, ijk, dev_grid);
    Ez_tmp[ijk] = KO(Ez, ijk, dev_grid);

    Bx_tmp[ijk] = KO(Bx, ijk, dev_grid);
    By_tmp[ijk] = KO(By, ijk, dev_grid);
    Bz_tmp[ijk] = KO(Bz, ijk, dev_grid);

    P_tmp[ijk] = KO(P, ijk, dev_grid);
  }
}

__global__ void
kernel_KO_step2_gr(Scalar *Ex, Scalar *Ey, Scalar *Ez, Scalar *Bx,
                   Scalar *By, Scalar *Bz, Scalar *Ex_tmp,
                   Scalar *Ey_tmp, Scalar *Ez_tmp, Scalar *Bx_tmp,
                   Scalar *By_tmp, Scalar *Bz_tmp, Scalar *P,
                   Scalar *P_tmp, int shift) {
  Scalar KO_const = 0.0;

  switch (FFE_DISSIPATION_ORDER) {
    case 4:
      KO_const = -1. / 16;
      break;
    case 6:
      KO_const = -1. / 64;
      break;
  }

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

    Ex[ijk] -= dev_params.KOeps * KO_const * Ex_tmp[ijk];
    Ey[ijk] -= dev_params.KOeps * KO_const * Ey_tmp[ijk];
    Ez[ijk] -= dev_params.KOeps * KO_const * Ez_tmp[ijk];

    Bx[ijk] -= dev_params.KOeps * KO_const * Bx_tmp[ijk];
    By[ijk] -= dev_params.KOeps * KO_const * By_tmp[ijk];
    Bz[ijk] -= dev_params.KOeps * KO_const * Bz_tmp[ijk];

    P[ijk] -= dev_params.KOeps * KO_const * P_tmp[ijk];
  }
}

HOST_DEVICE Scalar
innersigma(Scalar a, Scalar x, Scalar y, Scalar z, Scalar r0, Scalar d,
           Scalar sig0) {
  Scalar r = get_r(a, x, y, z);
  return sig0 * cube((r0 - r) / d);
}

__global__ void
kernel_absorbing_inner(const Scalar *Dnx, const Scalar *Dny,
                       const Scalar *Dnz, const Scalar *Bnx,
                       const Scalar *Bny, const Scalar *Bnz, Scalar *Dx,
                       Scalar *Dy, Scalar *Dz, Scalar *Bx, Scalar *By,
                       Scalar *Bz, int shift) {
  Scalar x, y, z;
  size_t ijk;
  Scalar rH = 1.0 + sqrt(1.0 - square(dev_params.a));
  Scalar r1 = 0.7 * rH;
  Scalar r2 = 0.01 * rH;
  Scalar dd = 0.2 * rH;
  Scalar sig, sigx, sigy, sigz;
  Scalar sig0 = dev_params.sigpml;

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
    x = dev_grid.pos(0, i, 1);
    y = dev_grid.pos(1, j, 1);
    z = dev_grid.pos(2, k, 1);

    Scalar r = get_r(dev_params.a, x, y, z);
    if (r < r1) {
      // Dx
      sig = innersigma(dev_params.a, x, y, z, r1, dd, sig0) *
            dev_params.dt;
      if (sig > TINY)
        Dx[ijk] = exp(-sig) * Dnx[ijk] +
                  (1.0 - exp(-sig)) / sig * (Dx[ijk] - Dnx[ijk]);
      // if (sig > 0) Dx[ijk] = exp(- sig) * Dx[ijk];
      // Dy
      sig = innersigma(dev_params.a, x, y, z, r1, dd, sig0) *
            dev_params.dt;
      if (sig > TINY)
        Dy[ijk] = exp(-sig) * Dny[ijk] +
                  (1.0 - exp(-sig)) / sig * (Dy[ijk] - Dny[ijk]);
      // if (sig > 0) Dy[ijk] = exp(- sig) * Dy[ijk];
      // Dz
      sig = innersigma(dev_params.a, x, y, z, r1, dd, sig0) *
            dev_params.dt;
      if (sig > TINY)
        Dz[ijk] = exp(-sig) * Dnz[ijk] +
                  (1.0 - exp(-sig)) / sig * (Dz[ijk] - Dnz[ijk]);
      // if (sig > 0) Dz[ijk] = exp(- sig) * Dz[ijk];
      // Bx
      sig = innersigma(dev_params.a, x, y, z, r1, dd, sig0) *
            dev_params.dt;
      if (sig > TINY)
        Bx[ijk] = exp(-sig) * Bnx[ijk] +
                  (1.0 - exp(-sig)) / sig * (Bx[ijk] - Bnx[ijk]);
      // if (sig > 0) Bx[ijk] = exp(- sig) * Bx[ijk];
      // By
      sig = innersigma(dev_params.a, x, y, z, r1, dd, sig0) *
            dev_params.dt;
      if (sig > TINY)
        By[ijk] = exp(-sig) * Bny[ijk] +
                  (1.0 - exp(-sig)) / sig * (By[ijk] - Bny[ijk]);
      // if (sig > 0) By[ijk] = exp(- sig) * By[ijk];
      // Bz
      sig = innersigma(dev_params.a, x, y, z, r1, dd, sig0) *
            dev_params.dt;
      if (sig > TINY)
        Bz[ijk] = exp(-sig) * Bnz[ijk] +
                  (1.0 - exp(-sig)) / sig * (Bz[ijk] - Bnz[ijk]);
      // if (sig > 0) Bz[ijk] = exp(- sig) * Bz[ijk];
    }
    if (r < r2) {
      Dx[ijk] = 0.0;
      Dy[ijk] = 0.0;
      Dz[ijk] = 0.0;
      Bx[ijk] = 0.0;
      By[ijk] = 0.0;
      Bz[ijk] = 0.0;
    }
  }
}

__global__ void
kernel_outgoing_z(Scalar *Dx, Scalar *Dy, Scalar *Dz, Scalar *Bx,
                  Scalar *By, Scalar *Bz, Scalar *P, int shift) {
  size_t ijk;
  int i =
      threadIdx.x + blockIdx.x * blockDim.x + dev_grid.guard[0] - shift;
  int j =
      threadIdx.y + blockIdx.y * blockDim.y + dev_grid.guard[1] - shift;
  if (i < dev_grid.dims[0] - dev_grid.guard[0] + shift &&
      j < dev_grid.dims[1] - dev_grid.guard[1] + shift) {
    int k = dev_grid.guard[2];
    ijk = i + j * dev_grid.dims[0] +
          k * dev_grid.dims[0] * dev_grid.dims[1];
    Scalar x = dev_grid.pos(0, i, 1);
    Scalar y = dev_grid.pos(1, j, 1);
    Scalar z = dev_grid.pos(2, k, 1);
    int s = dev_grid.dims[0] * dev_grid.dims[1];
    if (std::abs(z - dev_params.lower[2]) < dev_grid.delta[2] / 2.0) {
      for (int l = 1; l <= dev_params.guard[2]; ++l) {
        Dx[ijk - l * s] = Dx[ijk];
        Dy[ijk - l * s] = Dy[ijk];
        Dz[ijk - l * s] = Dz[ijk];
        Bx[ijk - l * s] = Bx[ijk];
        By[ijk - l * s] = By[ijk];
        Bz[ijk - l * s] = Bz[ijk];
        P[ijk - l * s] = P[ijk];
      }
    }
    k = dev_grid.dims[2] - dev_grid.guard[2];
    ijk = i + j * dev_grid.dims[0] +
          k * dev_grid.dims[0] * dev_grid.dims[1];
    x = dev_grid.pos(0, i, 1);
    y = dev_grid.pos(1, j, 1);
    z = dev_grid.pos(2, k, 1);
    if (std::abs(z - dev_params.lower[2] + dev_params.size[2]) <
        dev_grid.delta[2] / 2.0) {
      for (int l = 1; l <= dev_params.guard[2]; ++l) {
        Dx[ijk + l * s] = Dx[ijk];
        Dy[ijk + l * s] = Dy[ijk];
        Dz[ijk + l * s] = Dz[ijk];
        Bx[ijk + l * s] = Bx[ijk];
        By[ijk + l * s] = By[ijk];
        Bz[ijk + l * s] = Bz[ijk];
        P[ijk + l * s] = P[ijk];
      }
    }
  }
}

__device__ Scalar
omegad_gr(Scalar r, Scalar rmax) {
  Scalar del = 0.2;
  Scalar shape = 0.5 * (1.0 - tanh((r - rmax) / del));
  return 1.0 /
                 (dev_params.a + sqrt(cube(r))) * shape;
}

__global__ void
kernel_boundary_disk_conductor_gr(Scalar *Dx, Scalar *Dy, Scalar *Dz,
                                  Scalar *Bx, Scalar *By, Scalar *Bz,
                                  Scalar *P, Scalar t, int shift) {
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
    Scalar r = get_r(dev_params.a, x, y, z);
    int s = dev_grid.dims[0] * dev_grid.dims[1];
    Scalar xh = dev_params.lower[0] + dev_params.size[0] -
                dev_params.pml[0] * dev_grid.delta[0];
    Scalar xl =
        dev_params.lower[0] + dev_params.pml[3] * dev_grid.delta[0];
    Scalar yh = dev_params.lower[1] + dev_params.size[1] -
                dev_params.pml[1] * dev_grid.delta[1];
    Scalar yl =
        dev_params.lower[1] + dev_params.pml[4] * dev_grid.delta[1];
    Scalar rmax = get_r(dev_params.a, xh, 0.0, 0.0);
    Scalar alpha = get_alpha(dev_params.a, x, y, z);
    Scalar gmsqrt = get_sqrt_gamma(dev_params.a, x, y, z);

    Scalar Ddx, Ddy, Ddz, Dux, Duy, Duz;

    if (std::abs(z) < dev_grid.delta[2] * (3.0 + 1.0 / 4.0)) {
      if (r < rmax) {
        Scalar w = omegad_gr(r, rmax);
        Scalar vx =
            (get_beta_u1(dev_params.a, x, y, z) - w * y) / alpha;
        Scalar vy =
            (get_beta_u2(dev_params.a, x, y, z) + w * x) / alpha;
        Ddx = -gmsqrt * vy * Bz[ijk];
        Ddy = gmsqrt * vx * Bz[ijk];
        Ddz = gmsqrt * (-vx * By[ijk] + vy * Bx[ijk]);
        Dux = get_gamma_u11(dev_params.a, x, y, z) * Ddx +
              get_gamma_u12(dev_params.a, x, y, z) * Ddy +
              get_gamma_u13(dev_params.a, x, y, z) * Ddz;
        Duy = get_gamma_u12(dev_params.a, x, y, z) * Ddx +
              get_gamma_u22(dev_params.a, x, y, z) * Ddy +
              get_gamma_u23(dev_params.a, x, y, z) * Ddz;
        Duz = get_gamma_u13(dev_params.a, x, y, z) * Ddx +
              get_gamma_u23(dev_params.a, x, y, z) * Ddy +
              get_gamma_u33(dev_params.a, x, y, z) * Ddz;
        if (std::abs(z) < dev_grid.delta[2] * (2.0 + 1.0 / 4.0)) {
          Dx[ijk] = Dux;
          Dy[ijk] = Duy;
          Dz[ijk] = Duz;
        } else {
          Dx[ijk] = Dux;
          Dy[ijk] = Duy;
        }
      }
    }
  }
}

void
field_solver_gr_EZ::rk_step(Scalar As, Scalar Bs) {
  kernel_rk_step1_gr<<<blockGroupSize, blockSize>>>(
      Ed.dev_ptr(0), Ed.dev_ptr(1), Ed.dev_ptr(2), Hd.dev_ptr(0),
      Hd.dev_ptr(1), Hd.dev_ptr(2), m_data.E.dev_ptr(0),
      m_data.E.dev_ptr(1), m_data.E.dev_ptr(2), m_data.B.dev_ptr(0),
      m_data.B.dev_ptr(1), m_data.B.dev_ptr(2), dD.dev_ptr(0),
      dD.dev_ptr(1), dD.dev_ptr(2), dB.dev_ptr(0), dB.dev_ptr(1),
      dB.dev_ptr(2), m_data.B0.dev_ptr(0), m_data.B0.dev_ptr(1),
      m_data.B0.dev_ptr(2), m_data.divB.dev_ptr(),
      m_data.divE.dev_ptr(), m_data.P.dev_ptr(), dP.dev_ptr(),
      m_env.params().shift_ghost, As);
  CudaCheckError();
  kernel_rk_step2_gr<<<blockGroupSize, blockSize>>>(
      m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
      m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2),
      dD.dev_ptr(0), dD.dev_ptr(1), dD.dev_ptr(2), dB.dev_ptr(0),
      dB.dev_ptr(1), dB.dev_ptr(2), m_data.P.dev_ptr(), dP.dev_ptr(),
      m_env.params().shift_ghost, Bs);
  CudaCheckError();
}

void
field_solver_gr_EZ::Kreiss_Oliger() {
  kernel_KO_step1_gr<<<blockGroupSize, blockSize>>>(
      m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
      m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2),
      Dtmp.dev_ptr(0), Dtmp.dev_ptr(1), Dtmp.dev_ptr(2),
      Btmp.dev_ptr(0), Btmp.dev_ptr(1), Btmp.dev_ptr(2),
      m_data.P.dev_ptr(), Ptmp.dev_ptr(), m_env.params().shift_ghost);
  CudaCheckError();
  kernel_KO_step2_gr<<<blockGroupSize, blockSize>>>(
      m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
      m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2),
      Dtmp.dev_ptr(0), Dtmp.dev_ptr(1), Dtmp.dev_ptr(2),
      Btmp.dev_ptr(0), Btmp.dev_ptr(1), Btmp.dev_ptr(2),
      m_data.P.dev_ptr(), Ptmp.dev_ptr(), m_env.params().shift_ghost);
  CudaCheckError();
}

void
field_solver_gr_EZ::clean_epar() {
  kernel_clean_epar_gr<<<blockGroupSize, blockSize>>>(
      m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
      m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2),
      m_env.params().shift_ghost);
  CudaCheckError();
}

void
field_solver_gr_EZ::check_eGTb() {
  kernel_check_eGTb_gr<<<blockGroupSize, blockSize>>>(
      m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
      m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2),
      m_env.params().shift_ghost);
  CudaCheckError();
}

void
field_solver_gr_EZ::boundary_absorbing() {
  //  kernel_boundary_absorbing1_thread<<<blockGroupSize, blockSize>>>(
  //      Dtmp.dev_ptr(0), Dtmp.dev_ptr(1), Dtmp.dev_ptr(2),
  //      Btmp.dev_ptr(0), Btmp.dev_ptr(1), Btmp.dev_ptr(2),
  //      m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
  //      m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2),
  //      Ptmp.dev_ptr(), P.dev_ptr(), m_env.params().shift_ghost);
  kernel_boundary_absorbing_thread<<<blockGroupSize, blockSize>>>(
      Dtmp.dev_ptr(0), Dtmp.dev_ptr(1), Dtmp.dev_ptr(2),
      Btmp.dev_ptr(0), Btmp.dev_ptr(1), Btmp.dev_ptr(2),
      m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
      m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2),
      m_env.params().shift_ghost);
  CudaCheckError();
  kernel_absorbing_inner<<<blockGroupSize, blockSize>>>(
      Dtmp.dev_ptr(0), Dtmp.dev_ptr(1), Dtmp.dev_ptr(2),
      Btmp.dev_ptr(0), Btmp.dev_ptr(1), Btmp.dev_ptr(2),
      m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
      m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2),
      m_env.params().shift_ghost);
  CudaCheckError();
}

void
field_solver_gr_EZ::get_Ed() {
  dim3 blockGroupSize1 =
      dim3((m_data.env.grid().dims[0] + blockSize.x - 1) / blockSize.x,
           (m_data.env.grid().dims[1] + blockSize.y - 1) / blockSize.y,
           (m_data.env.grid().dims[2] + blockSize.z - 1) / blockSize.z);
  kernel_compute_E_gr<<<blockGroupSize1, blockSize>>>(
      m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
      m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2),
      Ed.dev_ptr(0), Ed.dev_ptr(1), Ed.dev_ptr(2),
      m_env.params().shift_ghost);
  CudaCheckError();
}

void
field_solver_gr_EZ::get_Hd() {
  dim3 blockGroupSize1 =
      dim3((m_data.env.grid().dims[0] + blockSize.x - 1) / blockSize.x,
           (m_data.env.grid().dims[1] + blockSize.y - 1) / blockSize.y,
           (m_data.env.grid().dims[2] + blockSize.z - 1) / blockSize.z);
  kernel_compute_H_gr<<<blockGroupSize1, blockSize>>>(
      m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
      m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2),
      Hd.dev_ptr(0), Hd.dev_ptr(1), Hd.dev_ptr(2),
      m_env.params().shift_ghost);
  CudaCheckError();
}

void
field_solver_gr_EZ::boundary_disk(Scalar t) {
  if (m_env.params().calc_current) {
    kernel_boundary_disk_conductor_gr<<<blockGroupSize, blockSize>>>(
      m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
      m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2),
      m_data.P.dev_ptr(), t, m_env.params().shift_ghost);
    CudaCheckError();
  }
}

void
field_solver_gr_EZ::evolve_fields(Scalar time) {
  Scalar As[5] = {0, -0.4178904745, -1.192151694643, -1.697784692471,
                  -1.514183444257};
  Scalar Bs[5] = {0.1496590219993, 0.3792103129999, 0.8229550293869,
                  0.6994504559488, 0.1530572479681};
  Scalar cs[5] = {0, 0.1496590219993, 0.3704009573644, 0.6222557631345,
                  0.9582821306784};

  Dtmp.copy_from(m_data.E);
  Btmp.copy_from(m_data.B);
  Ptmp.copy_from(m_data.P);

  for (int i = 0; i < 5; ++i) {
    timer::stamp();
    get_Ed();
    get_Hd();
    rk_step(As[i], Bs[i]);
    CudaSafeCall(cudaDeviceSynchronize());
    if (m_env.rank() == 0)
      timer::show_duration_since_stamp("rk_step", "ms");

    timer::stamp();
    if (m_env.params().clean_ep) clean_epar();
    if (m_env.params().check_egb) check_eGTb();

    if (m_env.params().disk)
      boundary_disk(time + cs[i] * m_env.params().dt);
    if (i == 4) boundary_absorbing();

    CudaSafeCall(cudaDeviceSynchronize());
    if (m_env.rank() == 0)
      timer::show_duration_since_stamp("clean/check/boundary", "ms");

    timer::stamp();
    m_env.send_guard_cells(m_data);
    CudaSafeCall(cudaDeviceSynchronize());
    if (m_env.rank() == 0)
      timer::show_duration_since_stamp("communication", "ms");
  }

  timer::stamp();
  Kreiss_Oliger();
  if (m_env.params().clean_ep) clean_epar();
  if (m_env.params().check_egb) check_eGTb();
  if (m_env.params().disk) boundary_disk(time + m_env.params().dt);
  CudaSafeCall(cudaDeviceSynchronize());
  m_env.send_guard_cells(m_data);
  if (m_env.rank() == 0)
    timer::show_duration_since_stamp("Kreiss Oliger", "ms");
}

field_solver_gr_EZ::field_solver_gr_EZ(sim_data &mydata,
                                       sim_environment &env)
    : m_data(mydata), m_env(env) {
  dD = vector_field<Scalar>(m_data.env.grid());
  dD.copy_stagger(m_data.E);
  dD.initialize();

  Dtmp = vector_field<Scalar>(m_data.env.grid());
  Dtmp.copy_stagger(m_data.E);
  Dtmp.copy_from(m_data.E);

  Ed = vector_field<Scalar>(m_data.env.grid());
  Ed.copy_stagger(m_data.E);
  Ed.initialize();

  dB = vector_field<Scalar>(m_data.env.grid());
  dB.copy_stagger(m_data.B);
  dB.initialize();

  Hd = vector_field<Scalar>(m_data.env.grid());
  Hd.copy_stagger(m_data.B);
  Hd.initialize();

  Btmp = vector_field<Scalar>(m_data.env.grid());
  Btmp.copy_stagger(m_data.B);
  Btmp.copy_from(m_data.B);

  dP = multi_array<Scalar>(m_data.env.grid().extent());
  dP.assign_dev(0.0);
  Ptmp = multi_array<Scalar>(m_data.env.grid().extent());
  Ptmp.assign_dev(0.0);

  blockGroupSize =
      dim3((m_data.env.grid().reduced_dim(0) +
            m_env.params().shift_ghost * 2 + blockSize.x - 1) /
               blockSize.x,
           (m_data.env.grid().reduced_dim(1) +
            m_env.params().shift_ghost * 2 + blockSize.y - 1) /
               blockSize.y,
           (m_data.env.grid().reduced_dim(2) +
            m_env.params().shift_ghost * 2 + blockSize.z - 1) /
               blockSize.z);
  std::cout << blockSize.x << ", " << blockSize.y << ", " << blockSize.z
            << std::endl;
  std::cout << blockGroupSize.x << ", " << blockGroupSize.y << ", "
            << blockGroupSize.z << std::endl;
}

field_solver_gr_EZ::~field_solver_gr_EZ() {}

}  // namespace Coffee
