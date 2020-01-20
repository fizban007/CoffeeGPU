#include "cuda/constant_mem.h"
#include "cuda/constant_mem_func.h"
#include "cuda/cuda_utility.h"
#include "field_solver_EZ_spherical.h"
#include "pulsar.h"
#include "utils/timer.h"

// 2D axisymmetric code in spherical coordinates. Original x, y, z
// correspond to x = log r, theta, phi.
// The field components in data are the the upper components.

#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 2
#define BLOCK_SIZE_Z 1

#define FFE_DISSIPATION_ORDER 6

namespace Coffee {

static dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);

static dim3 blockGroupSize;

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
cdiff1x4(const Scalar *f, int ijk, Scalar cm2, Scalar cm1, Scalar cp1, Scalar cp2) {
  return (f[ijk - 2] * cm2 - 8 * f[ijk - 1] *cm1 + 8 * f[ijk + 1] * cp1 - f[ijk + 2] * cp2) /
         12.0;
}

__device__ inline Scalar
cdiff1y4(const Scalar *f, int ijk, Scalar cm2, Scalar cm1, Scalar cp1, Scalar cp2) {
  int s = dev_grid.dims[0];
  return (f[ijk - 2 * s] * cm2 - 8 * f[ijk - 1 * s] * cm1 + 8 * f[ijk + 1 * s] * cp1 -
          f[ijk + 2 * s] * cp2) /
         12.0;
}

__device__ inline Scalar
cdiff1z4(const Scalar *f, int ijk, Scalar cm2, Scalar cm1, Scalar cp1, Scalar cp2) {
  int s = dev_grid.dims[0] * dev_grid.dims[1];
  return (f[ijk - 2 * s] * cm2 - 8 * f[ijk - 1 * s] * cm1 + 8 * f[ijk + 1 * s] * cp1 -
          f[ijk + 2 * s] * cp2) /
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

__device__ inline Scalar
cdfdx(const Scalar *f, int ijk, Scalar cm2, Scalar cm1, Scalar cp1, Scalar cp2) {
  return cdiff1x4(f, ijk, cm2, cm2, cp1, cp2) / dev_grid.delta[0];
}

__device__ inline Scalar
cdfdy(const Scalar *f, int ijk, Scalar cm2, Scalar cm1, Scalar cp1, Scalar cp2) {
  return cdiff1y4(f, ijk, cm2, cm2, cp1, cp2) / dev_grid.delta[1];
}

__device__ inline Scalar
cdfdz(const Scalar *f, int ijk, Scalar cm2, Scalar cm1, Scalar cp1, Scalar cp2) {
  return cdiff1z4(f, ijk, cm2, cm2, cp1, cp2) / dev_grid.delta[2];
}

__device__ inline Scalar
KO(const Scalar *f, int ijk) {
  if (FFE_DISSIPATION_ORDER == 4)
    return diff4x2(f, ijk) + diff4y2(f, ijk) + diff4z2(f, ijk);
  if (FFE_DISSIPATION_ORDER == 6)
    return diff6x2(f, ijk) + diff6y2(f, ijk) + diff6z2(f, ijk);
}

HD_INLINE Scalar
get_gamma_d11(Scalar x, Scalar y, Scalar z) {
  Scalar r = exp(x);
  return r * r;
}

HD_INLINE Scalar
get_gamma_d22(Scalar x, Scalar y, Scalar z) {
  Scalar r = exp(x);
  return r * r;
}

HD_INLINE Scalar
get_gamma_d33(Scalar x, Scalar y, Scalar z) {
  Scalar r = exp(x);
  return square(r * sin(y));
}

HD_INLINE Scalar
get_gamma(Scalar x, Scalar y, Scalar z) {
  return get_gamma_d11(x, y, z) * get_gamma_d22(x, y, z) *
         get_gamma_d33(x, y, z);
}

HD_INLINE Scalar
get_sqrt_gamma(Scalar x, Scalar y, Scalar z) {
  return std::sqrt(get_gamma(x, y, z));
}

__device__ Scalar
div4(const Scalar *fx, const Scalar *fy, const Scalar *fz, int ijk,
     Scalar x, Scalar y, Scalar z) {
  Scalar tmpx =
      (fx[ijk - 2] * get_sqrt_gamma(x - 2.0 * dev_grid.delta[0], y, z) -
       8.0 * fx[ijk - 1] *
           get_sqrt_gamma(x - 1.0 * dev_grid.delta[0], y, z) +
       8.0 * fx[ijk + 1] *
           get_sqrt_gamma(x + 1.0 * dev_grid.delta[0], y, z) -
       fx[ijk + 2] *
           get_sqrt_gamma(x + 2.0 * dev_grid.delta[0], y, z)) /
      12.0 / dev_grid.delta[0];
  int s = dev_grid.dims[0];
  Scalar tmpy =
      (fy[ijk - 2 * s] *
           get_sqrt_gamma(x, y - 2.0 * dev_grid.delta[1], z) -
       8.0 * fy[ijk - 1 * s] *
           get_sqrt_gamma(x, y - 1.0 * dev_grid.delta[1], z) +
       8.0 * fy[ijk + 1 * s] *
           get_sqrt_gamma(x, y + 1.0 * dev_grid.delta[1], z) -
       fy[ijk + 2 * s] *
           get_sqrt_gamma(x, y + 2.0 * dev_grid.delta[1], z)) /
      12.0 / dev_grid.delta[1];
  s = dev_grid.dims[0] * dev_grid.dims[1];
  Scalar tmpz =
      (fz[ijk - 2 * s] *
           get_sqrt_gamma(x, y, z - 2.0 * dev_grid.delta[2]) -
       8.0 * fz[ijk - 1 * s] *
           get_sqrt_gamma(x, y, z - 1.0 * dev_grid.delta[2]) +
       8.0 * fz[ijk + 1 * s] *
           get_sqrt_gamma(x, y, z + 1.0 * dev_grid.delta[2]) -
       fz[ijk + 2 * s] *
           get_sqrt_gamma(x, y, z + 2.0 * dev_grid.delta[2])) /
      12.0 / dev_grid.delta[2];
  return (tmpx + tmpy + tmpz) / get_sqrt_gamma(x, y, z);
}


__global__ void
kernel_compute_ElBl(const Scalar *Ex, const Scalar *Ey,
                    const Scalar *Ez, const Scalar *Bx,
                    const Scalar *By, const Scalar *Bz, Scalar *Elx,
                    Scalar *Ely, Scalar *Elz, Scalar *Blx, Scalar *Bly,
                    Scalar *Blz, int shift) {
  int i =
      threadIdx.x + blockIdx.x * blockDim.x + dev_grid.guard[0] - shift;
  int j =
      threadIdx.y + blockIdx.y * blockDim.y + dev_grid.guard[1] - shift;
  int k =
      threadIdx.z + blockIdx.z * blockDim.z + dev_grid.guard[2] - shift;
  if (i < dev_grid.dims[0] - dev_grid.guard[0] + shift &&
      j < dev_grid.dims[1] - dev_grid.guard[1] + shift &&
      k < dev_grid.dims[2] - dev_grid.guard[2] + shift) {
    size_t ijk = i + j * dev_grid.dims[0] +
                 k * dev_grid.dims[0] * dev_grid.dims[1];

    Scalar x = dev_grid.pos(0, i, 1);
    Scalar y = dev_grid.pos(1, j, 1);
    Scalar z = dev_grid.pos(2, k, 1);

    Elx[ijk] = get_gamma_d11(x, y, z) * Ex[ijk];
    Ely[ijk] = get_gamma_d22(x, y, z) * Ey[ijk];
    Elz[ijk] = get_gamma_d33(x, y, z) * Ez[ijk];

    Blx[ijk] = get_gamma_d11(x, y, z) * Bx[ijk];
    Bly[ijk] = get_gamma_d22(x, y, z) * By[ijk];
    Blz[ijk] = get_gamma_d33(x, y, z) * Bz[ijk];
  }
}

__global__ void
kernel_rk_step1(const Scalar *Elx, const Scalar *Ely, const Scalar *Elz,
                const Scalar *Blx, const Scalar *Bly, const Scalar *Blz,
                const Scalar *Ex, const Scalar *Ey, const Scalar *Ez,
                const Scalar *Bx, const Scalar *By, const Scalar *Bz,
                Scalar *dEx, Scalar *dEy, Scalar *dEz, Scalar *dBx,
                Scalar *dBy, Scalar *dBz, Scalar *jx, Scalar *jy,
                Scalar *jz, Scalar *DivB, Scalar *DivE, const Scalar *P,
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

    Scalar gmsqrt = get_sqrt_gamma(x, y, z);

    Scalar rotBx = (dfdy(Blz, ijk) - dfdz(Bly, ijk)) / gmsqrt;
    Scalar rotBy = (dfdz(Blx, ijk) - dfdx(Blz, ijk)) / gmsqrt;
    Scalar rotBz = (dfdx(Bly, ijk) - dfdy(Blx, ijk)) / gmsqrt;
    Scalar rotEx = (dfdy(Elz, ijk) - dfdz(Ely, ijk)) / gmsqrt;
    Scalar rotEy = (dfdz(Elx, ijk) - dfdx(Elz, ijk)) / gmsqrt;
    Scalar rotEz = (dfdx(Ely, ijk) - dfdy(Elx, ijk)) / gmsqrt;

    Scalar divE = div4(Ex, Ey, Ez, ijk, x, y, z);
    Scalar divB = div4(Bx, By, Bz, ijk, x, y, z);

    Scalar B2 =
        Bx[ijk] * Blx[ijk] + By[ijk] * Bly[ijk] + Bz[ijk] * Blz[ijk];
    if (B2 < TINY) B2 = TINY;

    Scalar Jp = (Blx[ijk] * rotBx + Bly[ijk] * rotBy + Blz[ijk] * rotBz) -
                (Elx[ijk] * rotEx + Ely[ijk] * rotEy + Elz[ijk] * rotEz);
    Scalar Jx = (divE * (Ely[ijk] * Blz[ijk] - Elz[ijk] * Bly[ijk]) / gmsqrt +
                 Jp * Bx[ijk]) /
                B2;
    Scalar Jy = (divE * (Elz[ijk] * Blx[ijk] - Elx[ijk] * Blz[ijk]) / gmsqrt +
                 Jp * By[ijk]) /
                B2;
    Scalar Jz = (divE * (Ex[ijk] * By[ijk] - Ey[ijk] * Bx[ijk]) / gmsqrt +
                 Jp * Bz[ijk]) /
                B2;
    Scalar Px = dfdx(P, ijk) / get_gamma_d11(x, y, z);
    Scalar Py = dfdy(P, ijk) / get_gamma_d22(x, y, z);
    Scalar Pz = dfdz(P, ijk) / get_gamma_d33(x, y, z);
    // Scalar Px = 0.0;
    // Scalar Py = 0.0;
    // Scalar Pz = 0.0;

    // dP[ijk] = As * dP[ijk] - dev_params.dt * (dev_params.ch2 * divB +
    //                                           P[ijk] / dev_params.tau);

    // Inside the damping layer
    // Scalar x = dev_grid.pos(0, i, 1);
    // Scalar y = dev_grid.pos(1, j, 1);
    // Scalar z = dev_grid.pos(2, k, 1);
    // Scalar xh = dev_params.lower[0] + dev_params.size[0] -
    //             dev_params.pml[0] * dev_grid.delta[0];
    // Scalar xl =
    //     dev_params.lower[0] + dev_params.pml[0] * dev_grid.delta[0];
    // Scalar yh = dev_params.lower[1] + dev_params.size[1] -
    //             dev_params.pml[1] * dev_grid.delta[1];
    // Scalar yl =
    //     dev_params.lower[1] + dev_params.pml[1] * dev_grid.delta[1];
    // Scalar zh = dev_params.lower[2] + dev_params.size[2] -
    //             dev_params.pml[2] * dev_grid.delta[2];
    // Scalar zl =
    //     dev_params.lower[2] + dev_params.pml[2] * dev_grid.delta[2];
    // if (x > xh || x < xl || y > yh || y < yl || z > zh || z < zl) {
    //   Px = 0.0;
    //   Py = 0.0;
    //   Pz = 0.0;
    //   dP[ijk] = 0.0;
    // }

    dBx[ijk] = As * dBx[ijk] - dev_params.dt * (rotEx + Px);
    dBy[ijk] = As * dBy[ijk] - dev_params.dt * (rotEy + Py);
    dBz[ijk] = As * dBz[ijk] - dev_params.dt * (rotEz + Pz);

    dEx[ijk] = As * dEx[ijk] + dev_params.dt * (rotBx - Jx);
    dEy[ijk] = As * dEy[ijk] + dev_params.dt * (rotBy - Jy);
    dEz[ijk] = As * dEz[ijk] + dev_params.dt * (rotBz - Jz);

    dP[ijk] = As * dP[ijk] - dev_params.dt * (dev_params.ch2 * divB +
                                              P[ijk] / dev_params.tau);
    jx[ijk] = Jx;
    jy[ijk] = Jy;
    jz[ijk] = Jz;
    DivB[ijk] = divB;
    DivE[ijk] = divE;
  }
}

__global__ void
kernel_rk_step2(Scalar *Ex, Scalar *Ey, Scalar *Ez, Scalar *Bx,
                Scalar *By, Scalar *Bz, const Scalar *dEx,
                const Scalar *dEy, const Scalar *dEz, const Scalar *dBx,
                const Scalar *dBy, const Scalar *dBz, Scalar *P, const Scalar *dP,
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

    Ex[ijk] = Ex[ijk] + Bs * dEx[ijk];
    Ey[ijk] = Ey[ijk] + Bs * dEy[ijk];
    Ez[ijk] = Ez[ijk] + Bs * dEz[ijk];

    Bx[ijk] = Bx[ijk] + Bs * dBx[ijk];
    By[ijk] = By[ijk] + Bs * dBy[ijk];
    Bz[ijk] = Bz[ijk] + Bs * dBz[ijk];

    P[ijk] = P[ijk] + Bs * dP[ijk];

    // Inside the damping layer
    // Scalar x = dev_grid.pos(0, i, 1);
    // Scalar y = dev_grid.pos(1, j, 1);
    // Scalar z = dev_grid.pos(2, k, 1);
    // Scalar xh = dev_params.lower[0] + dev_params.size[0] -
    //             dev_params.pml[0] * dev_grid.delta[0];
    // Scalar xl =
    //     dev_params.lower[0] + dev_params.pml[0] * dev_grid.delta[0];
    // Scalar yh = dev_params.lower[1] + dev_params.size[1] -
    //             dev_params.pml[1] * dev_grid.delta[1];
    // Scalar yl =
    //     dev_params.lower[1] + dev_params.pml[1] * dev_grid.delta[1];
    // Scalar zh = dev_params.lower[2] + dev_params.size[2] -
    //             dev_params.pml[2] * dev_grid.delta[2];
    // Scalar zl =
    //     dev_params.lower[2] + dev_params.pml[2] * dev_grid.delta[2];
    // if (x > xh || x < xl || y > yh || y < yl || z > zh || z < zl) {
    //   P[ijk] = 0.0;
    // }

  }
}

}  // namespace Coffee