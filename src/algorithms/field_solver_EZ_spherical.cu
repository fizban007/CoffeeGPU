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

// Finite difference formulae

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


__device__ inline Scalar
KO(const Scalar *f, int ijk) {
  if (FFE_DISSIPATION_ORDER == 4)
    return diff4x2(f, ijk) + diff4y2(f, ijk) + diff4z2(f, ijk);
  if (FFE_DISSIPATION_ORDER == 6)
    return diff6x2(f, ijk) + diff6y2(f, ijk) + diff6z2(f, ijk);
}

// metric, where coordinate transformation can be included
HD_INLINE Scalar
get_r(Scalar x, Scalar y, Scalar z) {
  return exp(x);
}

HD_INLINE Scalar
get_th(Scalar x, Scalar y, Scalar z) {
  return y;
}

HD_INLINE Scalar
get_gamma_d11(Scalar x, Scalar y, Scalar z) {
  Scalar r = get_r(x, y, z);
  return r * r;
}

HD_INLINE Scalar
get_gamma_d22(Scalar x, Scalar y, Scalar z) {
  Scalar r = get_r(x, y, z);
  return r * r;
}

HD_INLINE Scalar
get_gamma_d33(Scalar x, Scalar y, Scalar z) {
  Scalar r = get_r(x, y, z);
  Scalar th = get_th(x, y, z);
  return square(r * sin(th));
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
kernel_rk_step1_sph(const Scalar *Elx, const Scalar *Ely, const Scalar *Elz,
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
kernel_rk_step2_sph(Scalar *Ex, Scalar *Ey, Scalar *Ez, Scalar *Bx,
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

__global__ void
kernel_Epar_sph(Scalar *Ex, Scalar *Ey, Scalar *Ez, const Scalar *Bx,
                const Scalar *By, const Scalar *Bz, int shift) {
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

    Scalar B2 = get_gamma_d11(x, y, z) * Bx[ijk] * Bx[ijk] +
                get_gamma_d22(x, y, z) * By[ijk] * By[ijk] +
                get_gamma_d33(x, y, z) * Bz[ijk] * Bz[ijk];
    if (B2 < TINY) B2 = TINY;
    Scalar EB = get_gamma_d11(x, y, z) * Ex[ijk] * Bx[ijk] +
                get_gamma_d22(x, y, z) * Ey[ijk] * By[ijk] +
                get_gamma_d33(x, y, z) * Ez[ijk] * Bz[ijk];

    Ex[ijk] = Ex[ijk] - EB / B2 * Bx[ijk];
    Ey[ijk] = Ey[ijk] - EB / B2 * By[ijk];
    Ez[ijk] = Ez[ijk] - EB / B2 * Bz[ijk];
  }
}

__global__ void
kernel_EgtB_sph(Scalar *Ex, Scalar *Ey, Scalar *Ez, const Scalar *Bx,
                const Scalar *By, const Scalar *Bz, int shift) {
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

    Scalar B2 = get_gamma_d11(x, y, z) * Bx[ijk] * Bx[ijk] +
                get_gamma_d22(x, y, z) * By[ijk] * By[ijk] +
                get_gamma_d33(x, y, z) * Bz[ijk] * Bz[ijk];
    if (B2 < TINY) B2 = TINY;
    Scalar E2 = get_gamma_d11(x, y, z) * Ex[ijk] * Ex[ijk] +
                get_gamma_d22(x, y, z) * Ey[ijk] * Ey[ijk] +
                get_gamma_d33(x, y, z) * Ez[ijk] * Ez[ijk];

    if (E2 > B2) {
      Scalar s = sqrt(B2 / E2);
      Ex[ijk] *= s;
      Ey[ijk] *= s;
      Ez[ijk] *= s;
    }
  }
}

__global__ void
kernel_KO_step1_sph(Scalar *Ex, Scalar *Ey, Scalar *Ez, Scalar *Bx,
                Scalar *By, Scalar *Bz, Scalar *Ex_tmp, Scalar *Ey_tmp,
                Scalar *Ez_tmp, Scalar *Bx_tmp, Scalar *By_tmp,
                Scalar *Bz_tmp, Scalar *P, Scalar *P_tmp, int shift) {
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

    Ex_tmp[ijk] = KO(Ex, ijk);
    Ey_tmp[ijk] = KO(Ey, ijk);
    Ez_tmp[ijk] = KO(Ez, ijk);

    Bx_tmp[ijk] = KO(Bx, ijk);
    By_tmp[ijk] = KO(By, ijk);
    Bz_tmp[ijk] = KO(Bz, ijk);

    P_tmp[ijk] = KO(P, ijk);
  }
}

__global__ void
kernel_KO_step2_sph(Scalar *Ex, Scalar *Ey, Scalar *Ez, Scalar *Bx,
                Scalar *By, Scalar *Bz, Scalar *Ex_tmp, Scalar *Ey_tmp,
                Scalar *Ez_tmp, Scalar *Bx_tmp, Scalar *By_tmp,
                Scalar *Bz_tmp, Scalar *P, Scalar *P_tmp, int shift) {
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

__device__ Scalar
wpert(Scalar t, Scalar r, Scalar th) {
  Scalar th1 = acos(std::sqrt(1.0 - 1.0 / dev_params.rpert1));
  Scalar th2 = acos(std::sqrt(1.0 - 1.0 / dev_params.rpert2));
  if (th1 > th2) {
    Scalar tmp = th1;
    th1 = th2;
    th2 = tmp;
  } 
  Scalar mu = (th1 + th2) / 2.0;
  Scalar s = (mu - th1) / 3.0;
  if (t >= dev_params.tp_start && t <= dev_params.tp_end && th >= th1 &&
      th <= th2)
    return dev_params.dw0 * exp(- 0.5 * square((th - mu) / s)) *
           sin((t - dev_params.tp_start) * 2.0 * M_PI /
               (dev_params.tp_end - dev_params.tp_start)) *
           0.5 *
           (1.0 + tanh((r - 0.5 * dev_params.radius) /
                       (0.05 * dev_params.radius)));
  else
    return 0;
}

__global__ void
kernel_boundary_pulsar_sph(Scalar *Ex, Scalar *Ey, Scalar *Ez,
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
    Scalar r = get_r(x, y, z);
    Scalar th = get_th(x, y, z);
    Scalar w = dev_params.omega + wpert(t, r, th);

    if (r <= dev_params.radius) {
      Scalar bxn = dev_params.b0 * dipole_sph_2d(r, th, 0);
      Scalar byn = dev_params.b0 * dipole_sph_2d(r, th, 1);
      Scalar bzn = dev_params.b0 * dipole_sph_2d(r, th, 2);
      Bx[ijk] = bxn / std::sqrt(get_gamma_d11(x, y, z));
      By[ijk] = byn / std::sqrt(get_gamma_d22(x, y, z));
      Bz[ijk] = bzn / std::sqrt(get_gamma_d33(x, y, z));
      Scalar v3n = w * r * sin(th);
      Scalar exn = v3n * byn;
      Scalar eyn = -v3n * bxn;
      Ex[ijk] = exn / std::sqrt(get_gamma_d11(x, y, z));
      Ey[ijk] = eyn / std::sqrt(get_gamma_d22(x, y, z));
      Ez[ijk] = 0.0;
    }
  }
}

}  // namespace Coffee