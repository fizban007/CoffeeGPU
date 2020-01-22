#include "algorithms/finite_diff.h"
#include "cuda/constant_mem.h"
#include "cuda/constant_mem_func.h"
#include "cuda/cuda_utility.h"
#include "field_solver_EZ_spherical.h"
#include "pulsar.h"
#include "utils/timer.h"
#include "metric_sph.h"

// 2D axisymmetric code in spherical coordinates. Original x, y, z
// correspond to x = log r, theta, phi.
// The field components in data are the the upper components.

#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 4
#define BLOCK_SIZE_Z 1

#define FFE_DISSIPATION_ORDER 6

namespace Coffee {

using namespace SPH;

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
div4_sph(const Scalar *fx, const Scalar *fy, const Scalar *fz, int ijk,
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
kernel_rk_step1_sph(const Scalar *Elx, const Scalar *Ely,
                    const Scalar *Elz, const Scalar *Blx,
                    const Scalar *Bly, const Scalar *Blz,
                    const Scalar *Ex, const Scalar *Ey,
                    const Scalar *Ez, const Scalar *Bx,
                    const Scalar *By, const Scalar *Bz, Scalar *dEx,
                    Scalar *dEy, Scalar *dEz, Scalar *dBx, Scalar *dBy,
                    Scalar *dBz, Scalar *jx, Scalar *jy, Scalar *jz,
                    Scalar *DivB, Scalar *DivE, const Scalar *P,
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

    Scalar divE = div4_sph(Ex, Ey, Ez, ijk, x, y, z);
    Scalar divB = div4_sph(Bx, By, Bz, ijk, x, y, z);

    Scalar B2 =
        Bx[ijk] * Blx[ijk] + By[ijk] * Bly[ijk] + Bz[ijk] * Blz[ijk];
    if (B2 < TINY) B2 = TINY;

    Scalar Jp =
        (Blx[ijk] * rotBx + Bly[ijk] * rotBy + Blz[ijk] * rotBz) -
        (Elx[ijk] * rotEx + Ely[ijk] * rotEy + Elz[ijk] * rotEz);
    Scalar Jx =
        (divE * (Ely[ijk] * Blz[ijk] - Elz[ijk] * Bly[ijk]) / gmsqrt +
         Jp * Bx[ijk]) /
        B2;
    Scalar Jy =
        (divE * (Elz[ijk] * Blx[ijk] - Elx[ijk] * Blz[ijk]) / gmsqrt +
         Jp * By[ijk]) /
        B2;
    Scalar Jz =
        (divE * (Ex[ijk] * By[ijk] - Ey[ijk] * Bx[ijk]) / gmsqrt +
         Jp * Bz[ijk]) /
        B2;
    // Scalar Px = dfdx(P, ijk) / get_gamma_d11(x, y, z);
    // Scalar Py = dfdy(P, ijk) / get_gamma_d22(x, y, z);
    // Scalar Pz = dfdz(P, ijk) / get_gamma_d33(x, y, z);
    Scalar Px = 0.0;
    Scalar Py = 0.0;
    Scalar Pz = 0.0;

    // dP[ijk] = As * dP[ijk] - dev_params.dt * (dev_params.ch2 * divB +
    //                                           P[ijk] /
    //                                           dev_params.tau);

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
                    const Scalar *dEy, const Scalar *dEz,
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

    Scalar x = dev_grid.pos(0, i, 1);
    Scalar y = dev_grid.pos(1, j, 1);
    Scalar z = dev_grid.pos(2, k, 1);

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

    Scalar x = dev_grid.pos(0, i, 1);
    Scalar y = dev_grid.pos(1, j, 1);
    Scalar z = dev_grid.pos(2, k, 1);

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
kernel_KO_step2_sph(Scalar *Ex, Scalar *Ey, Scalar *Ez, Scalar *Bx,
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

__device__ Scalar
wpert_sph(Scalar t, Scalar r, Scalar th) {
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
    return dev_params.dw0 * exp(-0.5 * square((th - mu) / s)) *
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
    Scalar w = dev_params.omega + wpert_sph(t, r, th);

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

__global__ void
kernel_boundary_axis_sph(Scalar *Ex, Scalar *Ey, Scalar *Ez, Scalar *Bx,
                         Scalar *By, Scalar *Bz, Scalar *P, int shift) {
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
    int s = dev_grid.dims[0];
    if (std::abs(th) < dev_grid.delta[1] / 4.0) {
      Ex[ijk] = Ex[ijk + s];
      Ey[ijk] = 0.0;
      Ez[ijk] = 0.0;
      Bx[ijk] = Bx[ijk + s];
      By[ijk] = 0.0;
      Bz[ijk] = 0.0;
      P[ijk] = P[ijk + s];
      for (int l = 1; l <= 3; ++l) {
        Ex[ijk - l * s] = Ex[ijk + l * s];
        Ey[ijk - l * s] = -Ey[ijk + l * s];
        Ez[ijk - l * s] = -Ez[ijk + l * s];
        Bx[ijk - l * s] = Bx[ijk + l * s];
        By[ijk - l * s] = -By[ijk + l * s];
        Bz[ijk - l * s] = -Bz[ijk + l * s];
      }
    }
    if (std::abs(th - M_PI) < dev_grid.delta[1] / 4.0) {
      Ex[ijk] = Ex[ijk - s];
      Ey[ijk] = 0.0;
      Ez[ijk] = 0.0;
      Bx[ijk] = Bx[ijk - s];
      By[ijk] = 0.0;
      Bz[ijk] = 0.0;
      P[ijk] = P[ijk - s];
      for (int l = 1; l <= 3; ++l) {
        Ex[ijk + l * s] = Ex[ijk - l * s];
        Ey[ijk + l * s] = -Ey[ijk - l * s];
        Ez[ijk + l * s] = -Ez[ijk - l * s];
        Bx[ijk + l * s] = Bx[ijk - l * s];
        By[ijk + l * s] = -By[ijk - l * s];
        Bz[ijk + l * s] = -Bz[ijk - l * s];
      }
    }
  }
}

HOST_DEVICE Scalar
pmlsigma_sph(Scalar x, Scalar xh, Scalar pmlscale, Scalar sig0) {
  if (x > xh)
    return sig0 * cube((x - xh) / pmlscale);
  else
    return 0.0;
}

__global__ void
kernel_boundary_absorbing_sph(const Scalar *enx, const Scalar *eny,
                              const Scalar *enz, const Scalar *bnx,
                              const Scalar *bny, const Scalar *bnz,
                              Scalar *ex, Scalar *ey, Scalar *ez,
                              Scalar *bx, Scalar *by, Scalar *bz,
                              int shift) {
  Scalar x, y, z;
  Scalar sig = 0.0;
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
    Scalar r = get_r(x, y, z);
    Scalar xh = dev_params.lower[0] + dev_params.size[0] -
                dev_params.pml[0] * dev_grid.delta[0];
    Scalar rh = get_r(xh, y, z);
    if (x > xh) {
      sig = pmlsigma_sph(r, rh, dev_params.pmllen * 1.0,
                         dev_params.sigpml);
      if (sig > TINY) {
        ex[ijk] = exp(-sig) * enx[ijk] +
                  (1.0 - exp(-sig)) / sig * (ex[ijk] - enx[ijk]);
        ey[ijk] = exp(-sig) * eny[ijk] +
                  (1.0 - exp(-sig)) / sig * (ey[ijk] - eny[ijk]);
        ez[ijk] = exp(-sig) * enz[ijk] +
                  (1.0 - exp(-sig)) / sig * (ez[ijk] - enz[ijk]);
        bx[ijk] = exp(-sig) * bnx[ijk] +
                  (1.0 - exp(-sig)) / sig * (bx[ijk] - bnx[ijk]);
        by[ijk] = exp(-sig) * bny[ijk] +
                  (1.0 - exp(-sig)) / sig * (by[ijk] - bny[ijk]);
        bz[ijk] = exp(-sig) * bnz[ijk] +
                  (1.0 - exp(-sig)) / sig * (bz[ijk] - bnz[ijk]);
      }
    }
  }
}

field_solver_EZ_spherical::field_solver_EZ_spherical(
    sim_data &mydata, sim_environment &env)
    : m_data(mydata), m_env(env) {
  dE = vector_field<Scalar>(m_data.env.grid());
  dE.copy_stagger(m_data.E);
  dE.initialize();

  Etmp = vector_field<Scalar>(m_data.env.grid());
  Etmp.copy_stagger(m_data.E);
  Etmp.copy_from(m_data.E);

  El = vector_field<Scalar>(m_data.env.grid());
  El.copy_stagger(m_data.E);
  El.initialize();

  dB = vector_field<Scalar>(m_data.env.grid());
  dB.copy_stagger(m_data.B);
  dB.initialize();

  Btmp = vector_field<Scalar>(m_data.env.grid());
  Btmp.copy_stagger(m_data.B);
  Btmp.copy_from(m_data.B);

  Bl = vector_field<Scalar>(m_data.env.grid());
  Bl.copy_stagger(m_data.B);
  Bl.initialize();

  // P = multi_array<Scalar>(m_data.env.grid().extent());
  // P.assign_dev(0.0);
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

field_solver_EZ_spherical::~field_solver_EZ_spherical() {}

void
field_solver_EZ_spherical::get_ElBl() {
  kernel_compute_ElBl<<<blockGroupSize, blockSize>>>(
      m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
      m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2),
      El.dev_ptr(0), El.dev_ptr(1), El.dev_ptr(2), Bl.dev_ptr(0),
      Bl.dev_ptr(1), Bl.dev_ptr(2), m_env.params().shift_ghost);
  CudaCheckError();
}

void
field_solver_EZ_spherical::rk_step(Scalar As, Scalar Bs) {
  kernel_rk_step1_sph<<<blockGroupSize, blockSize>>>(
      El.dev_ptr(0), El.dev_ptr(1), El.dev_ptr(2), Bl.dev_ptr(0),
      Bl.dev_ptr(1), Bl.dev_ptr(2), m_data.E.dev_ptr(0),
      m_data.E.dev_ptr(1), m_data.E.dev_ptr(2), m_data.B.dev_ptr(0),
      m_data.B.dev_ptr(1), m_data.B.dev_ptr(2), dE.dev_ptr(0),
      dE.dev_ptr(1), dE.dev_ptr(2), dB.dev_ptr(0), dB.dev_ptr(1),
      dB.dev_ptr(2), m_data.B0.dev_ptr(0), m_data.B0.dev_ptr(1),
      m_data.B0.dev_ptr(2), m_data.divB.dev_ptr(),
      m_data.divE.dev_ptr(), m_data.P.dev_ptr(), dP.dev_ptr(),
      m_env.params().shift_ghost, As);
  CudaCheckError();
  kernel_rk_step2_sph<<<blockGroupSize, blockSize>>>(
      m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
      m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2),
      dE.dev_ptr(0), dE.dev_ptr(1), dE.dev_ptr(2), dB.dev_ptr(0),
      dB.dev_ptr(1), dB.dev_ptr(2), m_data.P.dev_ptr(), dP.dev_ptr(),
      m_env.params().shift_ghost, Bs);
  CudaCheckError();
}

void
field_solver_EZ_spherical::Kreiss_Oliger() {
  kernel_KO_step1_sph<<<blockGroupSize, blockSize>>>(
      m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
      m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2),
      Etmp.dev_ptr(0), Etmp.dev_ptr(1), Etmp.dev_ptr(2),
      Btmp.dev_ptr(0), Btmp.dev_ptr(1), Btmp.dev_ptr(2),
      m_data.P.dev_ptr(), Ptmp.dev_ptr(), m_env.params().shift_ghost);
  CudaCheckError();
  kernel_KO_step2_sph<<<blockGroupSize, blockSize>>>(
      m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
      m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2),
      Etmp.dev_ptr(0), Etmp.dev_ptr(1), Etmp.dev_ptr(2),
      Btmp.dev_ptr(0), Btmp.dev_ptr(1), Btmp.dev_ptr(2),
      m_data.P.dev_ptr(), Ptmp.dev_ptr(), m_env.params().shift_ghost);
  CudaCheckError();
}

void
field_solver_EZ_spherical::check_eGTb() {
  kernel_EgtB_sph<<<blockGroupSize, blockSize>>>(
      m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
      m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2),
      m_env.params().shift_ghost);
  CudaCheckError();
}

void
field_solver_EZ_spherical::clean_epar() {
  kernel_Epar_sph<<<blockGroupSize, blockSize>>>(
      m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
      m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2),
      m_env.params().shift_ghost);
  CudaCheckError();
}

void
field_solver_EZ_spherical::boundary_pulsar(Scalar t) {
  kernel_boundary_pulsar_sph<<<blockGroupSize, blockSize>>>(
      m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
      m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2),
      m_data.P.dev_ptr(), t, m_env.params().shift_ghost);
  CudaCheckError();
}

void
field_solver_EZ_spherical::boundary_axis() {
  kernel_boundary_axis_sph<<<blockGroupSize, blockSize>>>(
      m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
      m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2),
      m_data.P.dev_ptr(), m_env.params().shift_ghost);
  CudaCheckError();
}

void
field_solver_EZ_spherical::boundary_absorbing() {
  kernel_boundary_absorbing_sph<<<blockGroupSize, blockSize>>>(
      Etmp.dev_ptr(0), Etmp.dev_ptr(1), Etmp.dev_ptr(2),
      Btmp.dev_ptr(0), Btmp.dev_ptr(1), Btmp.dev_ptr(2),
      m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
      m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2),
      m_env.params().shift_ghost);
  CudaCheckError();
}

void
field_solver_EZ_spherical::evolve_fields(Scalar time) {
  Scalar As[5] = {0, -0.4178904745, -1.192151694643, -1.697784692471,
                  -1.514183444257};
  Scalar Bs[5] = {0.1496590219993, 0.3792103129999, 0.8229550293869,
                  0.6994504559488, 0.1530572479681};
  Scalar cs[5] = {0, 0.1496590219993, 0.3704009573644, 0.6222557631345,
                  0.9582821306784};

  boundary_axis();
  Etmp.copy_from(m_data.E);
  Btmp.copy_from(m_data.B);
  Ptmp.copy_from(m_data.P);

  for (int i = 0; i < 5; ++i) {
    timer::stamp();
    get_ElBl();
    rk_step(As[i], Bs[i]);
    CudaSafeCall(cudaDeviceSynchronize());
    if (m_env.rank() == 0)
      timer::show_duration_since_stamp("rk_step", "ms");

    timer::stamp();
    if (m_env.params().clean_ep) clean_epar();
    if (m_env.params().check_egb) check_eGTb();

    boundary_pulsar(time + cs[i] * m_env.params().dt);
    if (i == 4) boundary_absorbing();
    boundary_axis();

    CudaSafeCall(cudaDeviceSynchronize());
    if (m_env.rank() == 0)
      timer::show_duration_since_stamp("clean/check/boundary", "ms");

    timer::stamp();
    m_env.send_guard_cells(m_data);
    // m_env.send_guard_cell_array(P);
    CudaSafeCall(cudaDeviceSynchronize());
    if (m_env.rank() == 0)
      timer::show_duration_since_stamp("communication", "ms");
  }

  timer::stamp();
  Kreiss_Oliger();
  if (m_env.params().clean_ep) clean_epar();
  if (m_env.params().check_egb) check_eGTb();
  boundary_pulsar(time + m_env.params().dt);
  boundary_axis();
  CudaSafeCall(cudaDeviceSynchronize());
  m_env.send_guard_cells(m_data);
  // m_env.send_guard_cell_array(P);
  if (m_env.rank() == 0)
    timer::show_duration_since_stamp("Kreiss Oliger", "ms");
}

Scalar
field_solver_EZ_spherical::total_energy(vector_field<Scalar> &f) {
  f.sync_to_host();
  Scalar Wtmp = 0.0, W = 0.0;
  Scalar xh = m_env.params().lower[0] + m_env.params().size[0] -
              m_env.params().pml[0] * m_env.grid().delta[0];
  Scalar xl = m_env.params().lower[0];
  Scalar yh = m_env.params().lower[1] + m_env.params().size[1] -
              m_env.params().pml[1] * m_env.grid().delta[1];
  Scalar yl = m_env.params().lower[1] +
              m_env.params().pml[1] * m_env.grid().delta[1];
  Scalar zh = m_env.params().lower[2] + m_env.params().size[2] -
              m_env.params().pml[2] * m_env.grid().delta[2];
  Scalar zl = m_env.params().lower[2] +
              m_env.params().pml[2] * m_env.grid().delta[2];
  for (int k = m_env.grid().guard[2];
       k < m_env.grid().dims[2] - m_env.grid().guard[2]; ++k) {
    for (int j = m_env.grid().guard[1];
         j < m_env.grid().dims[1] - m_env.grid().guard[1]; ++j) {
      for (int i = m_env.grid().guard[0];
           i < m_env.grid().dims[0] - m_env.grid().guard[0]; ++i) {
        int ijk = i + j * m_env.grid().dims[0] +
                  k * m_env.grid().dims[0] * m_env.grid().dims[1];
        Scalar x = m_env.grid().pos(0, i, 1);
        Scalar y = m_env.grid().pos(1, j, 1);
        Scalar z = m_env.grid().pos(2, k, 1);
        Scalar r = get_r(x, y, z);
        if (r >= m_env.params().radius && x < xh && x > xl && y < yh &&
            y > yl && z < zh && z > zl) {
          Wtmp += (get_gamma_d11(x, y, z) * f.data(0)[ijk] *
                       f.data(0)[ijk] +
                   get_gamma_d22(x, y, z) * f.data(1)[ijk] *
                       f.data(1)[ijk] +
                   get_gamma_d33(x, y, z) * f.data(2)[ijk] *
                       f.data(2)[ijk]) *
                  get_sqrt_gamma(x, y, z) * m_env.grid().delta[0] *
                  m_env.grid().delta[1] * m_env.grid().delta[2];
        }
      }
    }
  }
  MPI_Reduce(&Wtmp, &W, 1, m_env.scalar_type(), MPI_SUM, 0,
             m_env.world());
  return W;
}

}  // namespace Coffee
