#include "cuda/constant_mem.h"
#include "cuda/constant_mem_func.h"
#include "cuda/cuda_utility.h"
#include "boundary.h"
#include "field_solver_EZ.h"
#include "utils/timer.h"
#include "pulsar.h"

#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 2
#define BLOCK_SIZE_Z 2

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

__global__ void
kernel_rk_step1(const Scalar *Ex, const Scalar *Ey,
                       const Scalar *Ez, const Scalar *Bx,
                       const Scalar *By, const Scalar *Bz, Scalar *dEx,
                       Scalar *dEy, Scalar *dEz, Scalar *dBx,
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

    Scalar rotBx = dfdy(Bz, ijk) - dfdz(By, ijk);
    Scalar rotBy = dfdz(Bx, ijk) - dfdx(Bz, ijk);
    Scalar rotBz = dfdx(By, ijk) - dfdy(Bx, ijk);
    Scalar rotEx = dfdy(Ez, ijk) - dfdz(Ey, ijk);
    Scalar rotEy = dfdz(Ex, ijk) - dfdx(Ez, ijk);
    Scalar rotEz = dfdx(Ey, ijk) - dfdy(Ex, ijk);

    Scalar divE = dfdx(Ex, ijk) + dfdy(Ey, ijk) + dfdz(Ez, ijk);
    Scalar divB = dfdx(Bx, ijk) + dfdy(By, ijk) + dfdz(Bz, ijk);

    Scalar B2 =
        Bx[ijk] * Bx[ijk] + By[ijk] * By[ijk] + Bz[ijk] * Bz[ijk];
    if (B2 < TINY) B2 = TINY;

    Scalar Jp = (Bx[ijk] * rotBx + By[ijk] * rotBy + Bz[ijk] * rotBz) -
                (Ex[ijk] * rotEx + Ey[ijk] * rotEy + Ez[ijk] * rotEz);
    Scalar Jx = (divE * (Ey[ijk] * Bz[ijk] - Ez[ijk] * By[ijk]) +
                 Jp * Bx[ijk]) /
                B2;
    Scalar Jy = (divE * (Ez[ijk] * Bx[ijk] - Ex[ijk] * Bz[ijk]) +
                 Jp * By[ijk]) /
                B2;
    Scalar Jz = (divE * (Ex[ijk] * By[ijk] - Ey[ijk] * Bx[ijk]) +
                 Jp * Bz[ijk]) /
                B2;

    dBx[ijk] = As * dBx[ijk] - dev_params.dt * (rotEx + dfdx(P, ijk));
    dBy[ijk] = As * dBy[ijk] - dev_params.dt * (rotEy + dfdy(P, ijk));
    dBz[ijk] = As * dBz[ijk] - dev_params.dt * (rotEz + dfdz(P, ijk));

    dEx[ijk] = As * dEx[ijk] + dev_params.dt * (rotBx - Jx);
    dEy[ijk] = As * dEy[ijk] + dev_params.dt * (rotBy - Jy);
    dEz[ijk] = As * dEz[ijk] + dev_params.dt * (rotBz - Jz);

    dP[ijk] = As * dP[ijk] - dev_params.dt * (dev_params.ch2 * divB +
                                              P[ijk] / dev_params.tau);
    // Inside the damping layer
    Scalar xh = dev_params.lower[0] + dev_params.size[0] -
                dev_params.pml[0] * dev_grid.delta[0];
    Scalar xl =
        dev_params.lower[0] + dev_params.pml[0] * dev_grid.delta[0];
    Scalar yh = dev_params.lower[1] + dev_params.size[1] -
                dev_params.pml[1] * dev_grid.delta[1];
    Scalar yl =
        dev_params.lower[1] + dev_params.pml[1] * dev_grid.delta[1];
    Scalar zh = dev_params.lower[2] + dev_params.size[2] -
                dev_params.pml[2] * dev_grid.delta[2];
    Scalar zl =
        dev_params.lower[2] + dev_params.pml[2] * dev_grid.delta[2];
    if (x > xh || x < xl || y > yh || y < yl || z > zh || z < zl) {
      dP[ijk] = 0.0;
    }
  }
}

__global__ void
kernel_rk_step2(Scalar *Ex, Scalar *Ey, Scalar *Ez, Scalar *Bx,
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
    Scalar xh = dev_params.lower[0] + dev_params.size[0] -
                dev_params.pml[0] * dev_grid.delta[0];
    Scalar xl =
        dev_params.lower[0] + dev_params.pml[0] * dev_grid.delta[0];
    Scalar yh = dev_params.lower[1] + dev_params.size[1] -
                dev_params.pml[1] * dev_grid.delta[1];
    Scalar yl =
        dev_params.lower[1] + dev_params.pml[1] * dev_grid.delta[1];
    Scalar zh = dev_params.lower[2] + dev_params.size[2] -
                dev_params.pml[2] * dev_grid.delta[2];
    Scalar zl =
        dev_params.lower[2] + dev_params.pml[2] * dev_grid.delta[2];
    if (x > xh || x < xl || y > yh || y < yl || z > zh || z < zl) {
      P[ijk] = 0.0;
    }
  }
}

__global__ void
kernel_Epar(Scalar *Ex, Scalar *Ey, Scalar *Ez, const Scalar *Bx,
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

    Scalar B2 =
        Bx[ijk] * Bx[ijk] + By[ijk] * By[ijk] + Bz[ijk] * Bz[ijk];
    if (B2 < TINY) B2 = TINY;
    Scalar EB =
        Ex[ijk] * Bx[ijk] + Ey[ijk] * By[ijk] + Ez[ijk] * Bz[ijk];

    Ex[ijk] = Ex[ijk] - EB / B2 * Bx[ijk];
    Ey[ijk] = Ey[ijk] - EB / B2 * By[ijk];
    Ez[ijk] = Ez[ijk] - EB / B2 * Bz[ijk];
  }
}

__global__ void
kernel_EgtB(Scalar *Ex, Scalar *Ey, Scalar *Ez, const Scalar *Bx,
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

    Scalar B2 =
        Bx[ijk] * Bx[ijk] + By[ijk] * By[ijk] + Bz[ijk] * Bz[ijk];
    if (B2 < TINY) B2 = TINY;
    Scalar E2 =
        Ex[ijk] * Ex[ijk] + Ey[ijk] * Ey[ijk] + Ez[ijk] * Ez[ijk];

    if (E2 > B2) {
      Scalar s = sqrt(B2 / E2);
      Ex[ijk] *= s;
      Ey[ijk] *= s;
      Ez[ijk] *= s;
    }
  }
}

__global__ void
kernel_KO_step1(Scalar *Ex, Scalar *Ey, Scalar *Ez, Scalar *Bx,
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
kernel_KO_step2(Scalar *Ex, Scalar *Ey, Scalar *Ez, Scalar *Bx,
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

__global__ void
kernel_boundary_pulsar(Scalar *Ex, Scalar *Ey, Scalar *Ez,
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
    Scalar r2 = x * x + y * y + z * z;
    if (r2 < TINY) r2 = TINY;
    Scalar r = std::sqrt(r2);
    Scalar rl = 2.0 * dev_params.radius;
    Scalar ri = 0.5 * dev_params.radius;
    // Scalar scale = 1.0 * dev_grid.delta[0];
    Scalar scaleEpar = 0.5 * dev_grid.delta[0];
    Scalar scaleEperp = 0.25 * dev_grid.delta[0];
    Scalar scaleBperp = scaleEpar;
    Scalar scaleBpar = scaleBperp;
    Scalar d1 = 4 * dev_grid.delta[0];
    Scalar d0 = 0;
    Scalar phase = dev_params.omega * t;
    Scalar Bxnew, Bynew, Bznew, Exnew, Eynew, Eznew;
    if (r < rl) {
      // Scalar bxn = dev_params.b0 * cube(dev_params.radius) *
      //              dipole_x(x, y, z, dev_params.alpha, phase);
      // Scalar byn = dev_params.b0 * cube(dev_params.radius) *
      //              dipole_y(x, y, z, dev_params.alpha, phase);
      // Scalar bzn = dev_params.b0 * cube(dev_params.radius) *
      //              dipole_z(x, y, z, dev_params.alpha, phase);
      Scalar bxn =
          dev_params.b0 *
          quadru_dipole(x, y, z, dev_params.p1, dev_params.p2,
                        dev_params.p3, dev_params.q11, dev_params.q12,
                        dev_params.q13, dev_params.q22, dev_params.q23,
                        dev_params.q_offset_x, dev_params.q_offset_y,
                        dev_params.q_offset_z, phase, 0);
      Scalar byn =
          dev_params.b0 *
          quadru_dipole(x, y, z, dev_params.p1, dev_params.p2,
                        dev_params.p3, dev_params.q11, dev_params.q12,
                        dev_params.q13, dev_params.q22, dev_params.q23,
                        dev_params.q_offset_x, dev_params.q_offset_y,
                        dev_params.q_offset_z, phase, 1);
      Scalar bzn =
          dev_params.b0 *
          quadru_dipole(x, y, z, dev_params.p1, dev_params.p2,
                        dev_params.p3, dev_params.q11, dev_params.q12,
                        dev_params.q13, dev_params.q22, dev_params.q23,
                        dev_params.q_offset_x, dev_params.q_offset_y,
                        dev_params.q_offset_z, phase, 2);
      Scalar s = shape(r, dev_params.radius - d1, scaleBperp);
      Bxnew =
          (bxn * x + byn * y + bzn * z) * x / r2 * s +
          (Bx[ijk] * x + By[ijk] * y + Bz[ijk] * z) * x / r2 * (1 - s);
      Bynew =
          (bxn * x + byn * y + bzn * z) * y / r2 * s +
          (Bx[ijk] * x + By[ijk] * y + Bz[ijk] * z) * y / r2 * (1 - s);
      Bznew =
          (bxn * x + byn * y + bzn * z) * z / r2 * s +
          (Bx[ijk] * x + By[ijk] * y + Bz[ijk] * z) * z / r2 * (1 - s);
      s = shape(r, dev_params.radius - d1, scaleBpar);
      Bxnew += (bxn - (bxn * x + byn * y + bzn * z) * x / r2) * s +
               (Bx[ijk] -
                (Bx[ijk] * x + By[ijk] * y + Bz[ijk] * z) * x / r2) *
                   (1 - s);
      Bynew += (byn - (bxn * x + byn * y + bzn * z) * y / r2) * s +
               (By[ijk] -
                (Bx[ijk] * x + By[ijk] * y + Bz[ijk] * z) * y / r2) *
                   (1 - s);
      Bznew += (bzn - (bxn * x + byn * y + bzn * z) * z / r2) * s +
               (Bz[ijk] -
                (Bx[ijk] * x + By[ijk] * y + Bz[ijk] * z) * z / r2) *
                   (1 - s);

      Scalar w = dev_params.omega;
      // Scalar w = dev_params.omega + wpert(t, z);
      Scalar vx = - w * y;
      Scalar vy = w * x;
      Scalar exn = - vy * Bz[ijk];
      Scalar eyn = vx * Bz[ijk];
      Scalar ezn = - vx * By[ijk] + vy * Bx[ijk];
      s = shape(r, dev_params.radius - d0, scaleEperp);
      Exnew =
          (exn * x + eyn * y + ezn * z) * x / r2 * s +
          (Ex[ijk] * x + Ey[ijk] * y + Ez[ijk] * z) * x / r2 * (1 - s);
      Eynew =
          (exn * x + eyn * y + ezn * z) * y / r2 * s +
          (Ex[ijk] * x + Ey[ijk] * y + Ez[ijk] * z) * y / r2 * (1 - s);
      Eznew =
          (exn * x + eyn * y + ezn * z) * z / r2 * s +
          (Ex[ijk] * x + Ey[ijk] * y + Ez[ijk] * z) * z / r2 * (1 - s);
      s = shape(r, dev_params.radius - d0, scaleEpar);
      Exnew += (exn - (exn * x + eyn * y + ezn * z) * x / r2) * s +
               (Ex[ijk] -
                (Ex[ijk] * x + Ey[ijk] * y + Ez[ijk] * z) * x / r2) *
                   (1 - s);
      Eynew += (eyn - (exn * x + eyn * y + ezn * z) * y / r2) * s +
               (Ey[ijk] -
                (Ex[ijk] * x + Ey[ijk] * y + Ez[ijk] * z) * y / r2) *
                   (1 - s);
      Eznew += (ezn - (exn * x + eyn * y + ezn * z) * z / r2) * s +
               (Ez[ijk] -
                (Ex[ijk] * x + Ey[ijk] * y + Ez[ijk] * z) * z / r2) *
                   (1 - s);
      Bx[ijk] = Bxnew;
      By[ijk] = Bynew;
      Bz[ijk] = Bznew;
      Ex[ijk] = Exnew;
      Ey[ijk] = Eynew;
      Ez[ijk] = Eznew;
      if (r < ri) {
        Bx[ijk] = bxn;
        By[ijk] = byn;
        Bz[ijk] = bzn;
        Ex[ijk] = exn;
        Ey[ijk] = eyn;
        Ez[ijk] = ezn;
      }
    }
  }
}

void
field_solver_EZ::rk_step(Scalar As, Scalar Bs) {
  kernel_rk_step1<<<blockGroupSize, blockSize>>>(
      m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
      m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2),
      dE.dev_ptr(0), dE.dev_ptr(1), dE.dev_ptr(2), dB.dev_ptr(0),
      dB.dev_ptr(1), dB.dev_ptr(2), P.dev_ptr(), dP.dev_ptr(),
      m_env.params().shift_ghost, As);
  CudaCheckError();
  kernel_rk_step2<<<blockGroupSize, blockSize>>>(
      m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
      m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2),
      dE.dev_ptr(0), dE.dev_ptr(1), dE.dev_ptr(2), dB.dev_ptr(0),
      dB.dev_ptr(1), dB.dev_ptr(2), P.dev_ptr(), dP.dev_ptr(),
      m_env.params().shift_ghost, Bs);
  CudaCheckError();
}

void
field_solver_EZ::Kreiss_Oliger() {
  kernel_KO_step1<<<blockGroupSize, blockSize>>>(
      m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
      m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2),
      Etmp.dev_ptr(0), Etmp.dev_ptr(1), Etmp.dev_ptr(2),
      Btmp.dev_ptr(0), Btmp.dev_ptr(1), Btmp.dev_ptr(2), P.dev_ptr(),
      Ptmp.dev_ptr(), m_env.params().shift_ghost);
  CudaCheckError();
  kernel_KO_step2<<<blockGroupSize, blockSize>>>(
      m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
      m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2),
      Etmp.dev_ptr(0), Etmp.dev_ptr(1), Etmp.dev_ptr(2),
      Btmp.dev_ptr(0), Btmp.dev_ptr(1), Btmp.dev_ptr(2), P.dev_ptr(),
      Ptmp.dev_ptr(), m_env.params().shift_ghost);
  CudaCheckError();
}

void
field_solver_EZ::clean_epar() {
  kernel_Epar<<<blockGroupSize, blockSize>>>(
      m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
      m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2),
      m_env.params().shift_ghost);
  CudaCheckError();
}

void
field_solver_EZ::check_eGTb() {
  kernel_EgtB<<<blockGroupSize, blockSize>>>(
      m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
      m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2),
      m_env.params().shift_ghost);
  CudaCheckError();
}

void
field_solver_EZ::boundary_pulsar(Scalar t) {
  kernel_boundary_pulsar<<<blockGroupSize, blockSize>>>(
      m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
      m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2),
      P.dev_ptr(), t, m_env.params().shift_ghost);
  CudaCheckError();
}

void
field_solver_EZ::boundary_absorbing() {
  kernel_boundary_absorbing_thread<<<blockGroupSize, blockSize>>>(
      Etmp.dev_ptr(0), Etmp.dev_ptr(1), Etmp.dev_ptr(2),
      Btmp.dev_ptr(0), Btmp.dev_ptr(1), Btmp.dev_ptr(2),
      m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
      m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2),
      m_env.params().shift_ghost);
  CudaCheckError();
}


void
field_solver_EZ::evolve_fields(Scalar time) {
  Scalar As[5] = {0, -0.4178904745, -1.192151694643, -1.697784692471,
                  -1.514183444257};
  Scalar Bs[5] = {0.1496590219993, 0.3792103129999, 0.8229550293869,
                  0.6994504559488, 0.1530572479681};
  Scalar cs[5] = {0, 0.1496590219993, 0.3704009573644, 0.6222557631345,
                  0.9582821306784};

  Etmp.copy_from(m_data.E);
  Btmp.copy_from(m_data.B);

  for (int i = 0; i < 5; ++i) {
    timer::stamp();
    rk_step(As[i], Bs[i]);
    CudaSafeCall(cudaDeviceSynchronize());
    if (m_env.rank() == 0)
      timer::show_duration_since_stamp("rk_step", "ms");

    timer::stamp();
    if (m_env.params().clean_ep) clean_epar();
    if (m_env.params().check_egb) check_eGTb();

    boundary_pulsar(time + cs[i] * m_env.params().dt);
    if (i == 4) boundary_absorbing();

    CudaSafeCall(cudaDeviceSynchronize());
    if (m_env.rank() == 0)
      timer::show_duration_since_stamp("clean/check/boundary", "ms");

    timer::stamp();
    m_env.send_guard_cells(m_data);
    m_env.send_guard_cell_array(P);
    CudaSafeCall(cudaDeviceSynchronize());
    if (m_env.rank() == 0)
      timer::show_duration_since_stamp("communication", "ms");
  }

  timer::stamp();
  Kreiss_Oliger();
  if (m_env.params().clean_ep) clean_epar();
  if (m_env.params().check_egb) check_eGTb();
  boundary_pulsar(time + m_env.params().dt);
  CudaSafeCall(cudaDeviceSynchronize());
  m_env.send_guard_cells(m_data);
  m_env.send_guard_cell_array(P);
  if (m_env.rank() == 0)
    timer::show_duration_since_stamp("Kreiss Oliger", "ms");
}

field_solver_EZ::field_solver_EZ(sim_data &mydata, sim_environment &env)
    : m_data(mydata), m_env(env) {
  dE = vector_field<Scalar>(m_data.env.grid());
  Etmp = vector_field<Scalar>(m_data.env.grid());
  dE.copy_stagger(m_data.E);
  Etmp.copy_stagger(m_data.E);
  dE.initialize();
  Etmp.copy_from(m_data.E);

  dB = vector_field<Scalar>(m_data.env.grid());
  Btmp = vector_field<Scalar>(m_data.env.grid());
  dB.copy_stagger(m_data.B);
  Btmp.copy_stagger(m_data.B);
  dB.initialize();
  Btmp.copy_from(m_data.B);

  P = multi_array<Scalar>(m_data.env.grid().extent());
  P.assign_dev(0.0);
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

field_solver_EZ::~field_solver_EZ() {}

Scalar
field_solver_EZ::total_energy(vector_field<Scalar> &f) {
  f.sync_to_host();
  Scalar Wtmp = 0.0, W = 0.0;
  Scalar xh = m_env.params().lower[0] + m_env.params().size[0] -
              m_env.params().pml[0] * m_env.grid().delta[0];
  Scalar xl = m_env.params().lower[0] +
              m_env.params().pml[0] * m_env.grid().delta[0];
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
        Scalar r = std::sqrt(x * x + y * y + z * z);
        if (r >= m_env.params().radius && x < xh && x > xl && y < yh &&
            y > yl && z < zh && z > zl) {
          Wtmp += f.data(0)[ijk] * f.data(0)[ijk] +
                  f.data(1)[ijk] * f.data(1)[ijk] +
                  f.data(2)[ijk] * f.data(2)[ijk];
        }
      }
    }
  }
  MPI_Reduce(&Wtmp, &W, 1, m_env.scalar_type(), MPI_SUM, 0,
             m_env.world());
  return W;
}

}  // namespace Coffee
