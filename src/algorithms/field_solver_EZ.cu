#include "algorithms/finite_diff_simd.h"
#include "boundary.h"
#include "cuda/constant_mem.h"
#include "cuda/constant_mem_func.h"
#include "cuda/cuda_utility.h"
#include "field_solver_EZ.h"
#include "pulsar.h"
#include "utils/timer.h"

#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 2
#define BLOCK_SIZE_Z 2

namespace Coffee {

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

static dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);

static dim3 blockGroupSize;

__device__ void
j_ext(Scalar x, Scalar y, Scalar z, Scalar *jnew) {
  Scalar r = sqrt(x * x + y * y);
  jnew[0] = 0.0;
  jnew[1] = 0.0;
  jnew[2] = 0.0;
  if (std::abs(z) < dev_grid.delta[2] / 4.0) {
    Scalar tmp = (r - dev_params.radius) * 2.0 * M_PI / dev_params.rj;
    if (tmp < 2.0 * M_PI && tmp > 0) {
      Scalar iphi = dev_params.b0 * sin(tmp) / pow(r / dev_params.radius, dev_params.al);
      jnew[0] = - y / r * iphi;
      jnew[1] = x / r * iphi;
    }
  }
}

__global__ void
kernel_rk_step1(const Scalar *Ex, const Scalar *Ey, const Scalar *Ez,
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

    Scalar Jx, Jy, Jz, jd[3] = {0.0, 0.0, 0.0};

    if (dev_params.calc_current) {
      if (dev_params.use_edotb_damping) {
        Scalar E2 =
            Ex[ijk] * Ex[ijk] + Ey[ijk] * Ey[ijk] + Ez[ijk] * Ez[ijk];
        Scalar chi2 = B2 - E2;
        Scalar EdotB =
            Ex[ijk] * Bx[ijk] + Ey[ijk] * By[ijk] + Ez[ijk] * Bz[ijk];
        Scalar E02 =
            0.5 * (sqrt(chi2 * chi2 + 4.0 * EdotB * EdotB) - chi2);

        Scalar Jp =
            (Bx[ijk] * rotBx + By[ijk] * rotBy + Bz[ijk] * rotBz) -
            (Ex[ijk] * rotEx + Ey[ijk] * rotEy + Ez[ijk] * rotEz) +
            dev_params.damp_gamma * EdotB / dev_params.dt;
        Jx = divE * (Ey[ijk] * Bz[ijk] - Ez[ijk] * By[ijk]) / (E02 + B2) +
             Jp * Bx[ijk] / B2;
        Jy = divE * (Ez[ijk] * Bx[ijk] - Ex[ijk] * Bz[ijk]) / (E02 + B2) +
             Jp * By[ijk] / B2;
        Jz = divE * (Ex[ijk] * By[ijk] - Ey[ijk] * Bx[ijk]) / (E02 + B2) +
             Jp * Bz[ijk] / B2;
      } else {
        Scalar Jp =
            (Bx[ijk] * rotBx + By[ijk] * rotBy + Bz[ijk] * rotBz) -
            (Ex[ijk] * rotEx + Ey[ijk] * rotEy + Ez[ijk] * rotEz);
        Jx = (divE * (Ey[ijk] * Bz[ijk] - Ez[ijk] * By[ijk]) +
              Jp * Bx[ijk]) /
             B2;
        Jy = (divE * (Ez[ijk] * Bx[ijk] - Ex[ijk] * Bz[ijk]) +
              Jp * By[ijk]) /
             B2;
        Jz = (divE * (Ex[ijk] * By[ijk] - Ey[ijk] * Bx[ijk]) +
              Jp * Bz[ijk]) /
             B2;
      }
    }
    else {
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

    Scalar Px, Py, Pz;
    if (dev_params.divB_clean) {
      Px = dfdx(P, ijk);
      Py = dfdy(P, ijk);
      Pz = dfdz(P, ijk);
    }
    else {
      Px = 0.0;
      Py = 0.0;
      Pz = 0.0;
    }

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
kernel_rk_step2(Scalar *Ex, Scalar *Ey, Scalar *Ez, Scalar *Bx,
                Scalar *By, Scalar *Bz, const Scalar *dEx,
                const Scalar *dEy, const Scalar *dEz, const Scalar *dBx,
                const Scalar *dBy, const Scalar *dBz, Scalar *P,
                const Scalar *dP, int shift, Scalar Bs) {
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

    Ex_tmp[ijk] = KO(Ex, ijk, dev_grid);
    Ey_tmp[ijk] = KO(Ey, ijk, dev_grid);
    Ez_tmp[ijk] = KO(Ez, ijk, dev_grid);

    Bx_tmp[ijk] = KO(Bx, ijk, dev_grid);
    By_tmp[ijk] = KO(By, ijk, dev_grid);
    Bz_tmp[ijk] = KO(Bz, ijk, dev_grid);

    P_tmp[ijk] = KO(P, ijk, dev_grid);

    // // Exclude the damping layer
    // Scalar x = dev_grid.pos(0, i, 1);
    // Scalar y = dev_grid.pos(1, j, 1);
    // Scalar z = dev_grid.pos(2, k, 1);
    // Scalar xh =
    //     dev_params.lower[0] + dev_params.size[0] -
    //     (dev_params.pml[0] + dev_params.guard[0]) *
    //     dev_grid.delta[0];
    // Scalar xl =
    //     dev_params.lower[0] +
    //     (dev_params.pml[0] + dev_params.guard[0]) *
    //     dev_grid.delta[0];
    // Scalar yh =
    //     dev_params.lower[1] + dev_params.size[1] -
    //     (dev_params.pml[1] + dev_params.guard[1]) *
    //     dev_grid.delta[1];
    // Scalar yl =
    //     dev_params.lower[1] +
    //     (dev_params.pml[1] + dev_params.guard[1]) *
    //     dev_grid.delta[1];
    // Scalar zh =
    //     dev_params.lower[2] + dev_params.size[2] -
    //     (dev_params.pml[2] + dev_params.guard[2]) *
    //     dev_grid.delta[2];
    // Scalar zl =
    //     dev_params.lower[2] +
    //     (dev_params.pml[2] + dev_params.guard[2]) *
    //     dev_grid.delta[2];
    // if (x < xh && x > xl && y < yh && y > yl && z < zh && z > zl) {
    //   Ex_tmp[ijk] = KO(Ex, ijk);
    //   Ey_tmp[ijk] = KO(Ey, ijk);
    //   Ez_tmp[ijk] = KO(Ez, ijk);

    //   Bx_tmp[ijk] = KO(Bx, ijk);
    //   By_tmp[ijk] = KO(By, ijk);
    //   Bz_tmp[ijk] = KO(Bz, ijk);

    //   P_tmp[ijk] = KO(P, ijk);
    // }
    // else {
    //   Ex_tmp[ijk] = 0;
    //   Ey_tmp[ijk] = 0;
    //   Ez_tmp[ijk] = 0;

    //   Bx_tmp[ijk] = 0;
    //   By_tmp[ijk] = 0;
    //   Bz_tmp[ijk] = 0;

    //   P_tmp[ijk] = 0;
    // }
  }
}

__global__ void
kernel_KO_step2(Scalar *Ex, Scalar *Ey, Scalar *Ez, Scalar *Bx,
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

__global__ void
kernel_boundary_pulsar(Scalar *Ex, Scalar *Ey, Scalar *Ez, Scalar *Bx,
                       Scalar *By, Scalar *Bz, Scalar *P, Scalar t,
                       int shift) {
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
      Bx[ijk] = Bxnew;
      By[ijk] = Bynew;
      Bz[ijk] = Bznew;

      Scalar w = dev_params.omega;
      // Scalar w = dev_params.omega + wpert(t, z);
      Scalar vx = -w * y;
      Scalar vy = w * x;
      Scalar exn = -vy * Bz[ijk];
      Scalar eyn = vx * Bz[ijk];
      Scalar ezn = -vx * By[ijk] + vy * Bx[ijk];
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
      // Bx[ijk] = Bxnew;
      // By[ijk] = Bynew;
      // Bz[ijk] = Bznew;
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

__global__ void
kernel_boundary_disk_vacuum(Scalar *Ex, Scalar *Ey, Scalar *Ez, Scalar *Bx,
                       Scalar *By, Scalar *Bz, Scalar *P, Scalar t,
                       int shift) {
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
    int s = dev_grid.dims[0] * dev_grid.dims[1];

    if (std::abs(z) < dev_grid.delta[2] / 4.0) {
      for (int l = 1; l <= 3; l++) {
        Bx[ijk - l * s] = -Bx[ijk + l * s];
        By[ijk - l * s] = -By[ijk + l * s];
        Bz[ijk - l * s] = Bz[ijk + l * s];
        Ex[ijk - l * s] = Ex[ijk + l * s];
        Ey[ijk - l * s] = Ey[ijk + l * s];
        Ez[ijk - l * s] = -Ez[ijk + l * s];
      }
    }
  }
}


__device__ Scalar omegad(Scalar R) {
  if (R < dev_params.radius)
    return dev_params.omegad0;
  else
    return dev_params.omegad0 / sqrt(cube(R / dev_params.radius));
}

__global__ void
kernel_boundary_disk_conductor(Scalar *Ex, Scalar *Ey, Scalar *Ez, Scalar *Bx,
                       Scalar *By, Scalar *Bz, Scalar *P, Scalar t,
                       int shift) {
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

    if (std::abs(z) < dev_grid.delta[2] / 4.0) {
      Scalar R = sqrt(x * x + y * y);
      Scalar w = omegad(R);
      Scalar vx = - w * y;
      Scalar vy = w * x;
      Ex[ijk] = - vy * Bz[ijk];
      Ey[ijk] = vx * Bz[ijk];
      for (int l = 1; l <= 3; l++) {
        Bx[ijk - l * s] = -Bx[ijk + l * s];
        By[ijk - l * s] = -By[ijk + l * s];
        Bz[ijk - l * s] = Bz[ijk + l * s];
        Ex[ijk - l * s] = Ex[ijk + l * s];
        Ey[ijk - l * s] = Ey[ijk + l * s];
        Ez[ijk - l * s] = -Ez[ijk + l * s];
      }
    }
  }
}

__global__ void
kernel_compute_skymap(Scalar *skymap, int Nth, int Nph,
                      const Scalar *Jx, const Scalar *Jy,
                      const Scalar *Jz, const Scalar *Bx,
                      const Scalar *By, const Scalar *Bz,
                      const Scalar *Ex, const Scalar *Ey,
                      const Scalar *Ez, float Rmin, float Rmax,
                      float min_lambda, int ppc) {
  // Loop over the cells
  size_t ijk;
  int i = threadIdx.x + blockIdx.x * blockDim.x + dev_grid.guard[0];
  int j = threadIdx.y + blockIdx.y * blockDim.y + dev_grid.guard[1];
  int k = threadIdx.z + blockIdx.z * blockDim.z + dev_grid.guard[2];
  if (i < dev_grid.dims[0] - dev_grid.guard[0] &&
      j < dev_grid.dims[1] - dev_grid.guard[1] &&
      k < dev_grid.dims[2] - dev_grid.guard[2]) {
    ijk = i + j * dev_grid.dims[0] +
          k * dev_grid.dims[0] * dev_grid.dims[1];
    Scalar jx = Jx[ijk];
    Scalar jy = Jy[ijk];
    Scalar jz = Jz[ijk];
    Scalar bx = Bx[ijk];
    Scalar by = By[ijk];
    Scalar bz = Bz[ijk];

    Scalar x = dev_grid.pos(0, i, true);
    Scalar y = dev_grid.pos(1, j, true);
    Scalar z = dev_grid.pos(2, k, true);
    Scalar r = sqrt(x * x + y * y + z * z);

    Scalar b2 = bx * bx + by * by + bz * bz;
    Scalar lambda = (jx * bx + jy * by + jz * bz) / b2;

    if (lambda > min_lambda && r > Rmin && r < Rmax) {
      Scalar ex = Ex[ijk];
      Scalar ey = Ey[ijk];
      Scalar ez = Ez[ijk];
      // Generate ppc particles in the cell and compute their emission
      for (int n = 0; n < ppc; n++) {
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
      dB.dev_ptr(1), dB.dev_ptr(2), m_data.B0.dev_ptr(0),
      m_data.B0.dev_ptr(1), m_data.B0.dev_ptr(2), m_data.divB.dev_ptr(),
      m_data.divE.dev_ptr(), m_data.P.dev_ptr(), dP.dev_ptr(),
      m_env.params().shift_ghost, As);
  CudaCheckError();
  kernel_rk_step2<<<blockGroupSize, blockSize>>>(
      m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
      m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2),
      dE.dev_ptr(0), dE.dev_ptr(1), dE.dev_ptr(2), dB.dev_ptr(0),
      dB.dev_ptr(1), dB.dev_ptr(2), m_data.P.dev_ptr(), dP.dev_ptr(),
      m_env.params().shift_ghost, Bs);
  CudaCheckError();
}

void
field_solver_EZ::Kreiss_Oliger() {
  kernel_KO_step1<<<blockGroupSize, blockSize>>>(
      m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
      m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2),
      Etmp.dev_ptr(0), Etmp.dev_ptr(1), Etmp.dev_ptr(2),
      Btmp.dev_ptr(0), Btmp.dev_ptr(1), Btmp.dev_ptr(2),
      m_data.P.dev_ptr(), Ptmp.dev_ptr(), m_env.params().shift_ghost);
  CudaCheckError();
  kernel_KO_step2<<<blockGroupSize, blockSize>>>(
      m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
      m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2),
      Etmp.dev_ptr(0), Etmp.dev_ptr(1), Etmp.dev_ptr(2),
      Btmp.dev_ptr(0), Btmp.dev_ptr(1), Btmp.dev_ptr(2),
      m_data.P.dev_ptr(), Ptmp.dev_ptr(), m_env.params().shift_ghost);
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
      m_data.P.dev_ptr(), t, m_env.params().shift_ghost);
  CudaCheckError();
}

void
field_solver_EZ::boundary_disk(Scalar t) {
  if (!m_env.params().calc_current)
    kernel_boundary_disk_vacuum<<<blockGroupSize, blockSize>>>(
      m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
      m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2),
      m_data.P.dev_ptr(), t, m_env.params().shift_ghost);
  else
    kernel_boundary_disk_conductor<<<blockGroupSize, blockSize>>>(
      m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
      m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2),
      m_data.P.dev_ptr(), t, m_env.params().shift_ghost);
  CudaCheckError();
}

void
field_solver_EZ::boundary_absorbing() {
  // kernel_boundary_absorbing_thread<<<blockGroupSize, blockSize>>>(
  //     Etmp.dev_ptr(0), Etmp.dev_ptr(1), Etmp.dev_ptr(2),
  //     Btmp.dev_ptr(0), Btmp.dev_ptr(1), Btmp.dev_ptr(2),
  //     m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
  //     m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2),
  //     m_env.params().shift_ghost);
  kernel_boundary_absorbing1_thread<<<blockGroupSize, blockSize>>>(
      Etmp.dev_ptr(0), Etmp.dev_ptr(1), Etmp.dev_ptr(2),
      Btmp.dev_ptr(0), Btmp.dev_ptr(1), Btmp.dev_ptr(2),
      m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
      m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2),
      Ptmp.dev_ptr(), m_data.P.dev_ptr(), m_env.params().shift_ghost);
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
  Ptmp.copy_from(m_data.P);

  for (int i = 0; i < 5; ++i) {
    timer::stamp();
    rk_step(As[i], Bs[i]);
    CudaSafeCall(cudaDeviceSynchronize());
    if (m_env.rank() == 0)
      timer::show_duration_since_stamp("rk_step", "ms");

    timer::stamp();
    if (m_env.params().clean_ep) clean_epar();
    if (m_env.params().check_egb) check_eGTb();

    if (m_env.params().pulsar) boundary_pulsar(time + cs[i] * m_env.params().dt);
    if (m_env.params().disk) boundary_disk(time + cs[i] * m_env.params().dt);
    if (i == 4) boundary_absorbing();

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
  if (m_env.params().pulsar) boundary_pulsar(time + m_env.params().dt);
  if (m_env.params().disk) boundary_disk(time + m_env.params().dt);
  CudaSafeCall(cudaDeviceSynchronize());
  m_env.send_guard_cells(m_data);
  // m_env.send_guard_cell_array(P);
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

  // P = multi_array<Scalar>(m_data.env.grid().extent());
  // P.assign_dev(0.0);
  dP = multi_array<Scalar>(m_data.env.grid().extent());
  dP.assign_dev(0.0);
  Ptmp = multi_array<Scalar>(m_data.env.grid().extent());
  Ptmp.assign_dev(0.0);

  skymap = multi_array<Scalar>(env.params().skymap_Nth,
                               env.params().skymap_Nph);
  skymap.assign_dev(0.0);
  skymap.sync_to_host();

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
              m_env.params().pml[3] * m_env.grid().delta[0];
  Scalar yh = m_env.params().lower[1] + m_env.params().size[1] -
              m_env.params().pml[1] * m_env.grid().delta[1];
  Scalar yl = m_env.params().lower[4] +
              m_env.params().pml[1] * m_env.grid().delta[1];
  Scalar zh = m_env.params().lower[2] + m_env.params().size[2] -
              m_env.params().pml[2] * m_env.grid().delta[2];
  Scalar zl = m_env.params().lower[5] +
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
        if ((!(m_env.params().pulsar && r < m_env.params().radius)) && x < xh &&
            x > xl && y < yh && y > yl && z < zh && z > zl) {
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
