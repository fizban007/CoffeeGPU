#include "cuda/constant_mem.h"
#include "cuda/constant_mem_func.h"
#include "cuda/cuda_utility.h"
#include "field_solver_EZ_cylindrical.h"
#include "pulsar.h"
#include "utils/timer.h"

// 2D axisymmetric code. Original x, y, z correspond to R, z, phi.

#define BLOCK_SIZE_R 32
#define BLOCK_SIZE_Z 2
#define BLOCK_SIZE_F 1

#define TINY 1e-7

#define FFE_DISSIPATION_ORDER 6

namespace Coffee {

static dim3 blockSize(BLOCK_SIZE_R, BLOCK_SIZE_Z, BLOCK_SIZE_F);

static dim3 blockGroupSize;

__device__ inline Scalar
diff1R4(const Scalar *f, int ijk) {
  return (f[ijk - 2] - 8 * f[ijk - 1] + 8 * f[ijk + 1] - f[ijk + 2]) /
         12.0;
}

__device__ inline Scalar
diff1aR4(const Scalar *f, int ijk) {
  return - 25.0 / 12.0 * f[ijk] + 4.0 * f[ijk + 1] - 3.0 * f[ijk + 2] +
         4.0 / 3.0 * f[ijk + 3] - 1.0 / 4.0 * f[ijk + 4];
}

__device__ inline Scalar
diff1RR4(const Scalar *f, const Scalar R0, int ijk) {
  Scalar dR = dev_grid.delta[0];
  return (f[ijk - 2] * (R0 - 2 * dR) - 8 * f[ijk - 1] * (R0 - dR) +
          8 * f[ijk + 1] * (R0 + dR) - f[ijk + 2] * (R0 + 2 * dR)) /
         12.0;
}

__device__ inline Scalar
diff1aRR4(const Scalar *f, const Scalar R0, int ijk) {
  Scalar dR = dev_grid.delta[0];
  return -25.0 / 12.0 * f[ijk] * R0 + 4.0 * f[ijk + 1] * (R0 + dR) -
         3.0 * f[ijk + 2] * (R0 + 2 * dR) +
         4.0 / 3.0 * f[ijk + 3] * (R0 + 3 * dR) -
         1.0 / 4.0 * f[ijk + 4] * (R0 + 4 * dR);
}

__device__ inline Scalar
diff1z4(const Scalar *f, int ijk) {
  int s = dev_grid.dims[0];
  return (f[ijk - 2 * s] - 8 * f[ijk - 1 * s] + 8 * f[ijk + 1 * s] -
          f[ijk + 2 * s]) /
         12.0;
}

__device__ inline Scalar
diff4R2(const Scalar *f, int ijk) {
  return (f[ijk - 2] - 4 * f[ijk - 1] + 6 * f[ijk] - 4 * f[ijk + 1] +
          f[ijk + 2]);
}

__device__ inline Scalar
diff4aR2(const Scalar *f, int ijk) {
  return 3.0 * f[ijk] - 14.0 * f[ijk + 1] + 26.0 * f[ijk + 2] -
         24.0 * f[ijk + 3] + 11.0 * f[ijk + 4] - 2.0 * f[ijk + 5];
}

__device__ inline Scalar
diff4z2(const Scalar *f, int ijk) {
  int s = dev_grid.dims[0];
  return (f[ijk - 2 * s] - 4 * f[ijk - 1 * s] + 6 * f[ijk] -
          4 * f[ijk + 1 * s] + f[ijk + 2 * s]);
}

__device__ inline Scalar
diff6R2(const Scalar *f, int ijk) {
  return (f[ijk - 3] - 6 * f[ijk - 2] + 15 * f[ijk - 1] - 20 * f[ijk] +
          15 * f[ijk + 1] - 6 * f[ijk + 2] + f[ijk + 3]);
}

__device__ inline Scalar
diff6aR2(const Scalar *f, int ijk) {
  return 4.0 * f[ijk] - 27.0 * f[ijk + 1] +
         78.0 * f[ijk + 2] - 125.0 * f[ijk + 3] +
         120.0 * f[ijk + 4] - 69.0 * f[ijk + 5] +
         22.0 * f[ijk + 6] - 3.0 * f[ijk + 7];
}

__device__ inline Scalar
diff6z2(const Scalar *f, int ijk) {
  int s = dev_grid.dims[0];
  return (f[ijk - 3 * s] - 6 * f[ijk - 2 * s] + 15 * f[ijk - 1 * s] -
          20 * f[ijk] + 15 * f[ijk + 1 * s] - 6 * f[ijk + 2 * s] +
          f[ijk + 3 * s]);
}

__device__ inline Scalar
dfdR(const Scalar *f, const Scalar R0, int ijk) {
  // if (R0 - 2 * dev_grid.delta[0] < 0) return diff1aR4(f, ijk) / dev_grid.delta[0];
  // else return diff1R4(f, ijk) / dev_grid.delta[0];
  return diff1R4(f, ijk) / dev_grid.delta[0];
}

__device__ inline Scalar
dfRdR(const Scalar *f, const Scalar R0, int ijk) {
  return diff1RR4(f, R0, ijk) / dev_grid.delta[0];
}

__device__ inline Scalar
dfdz(const Scalar *f, int ijk) {
  return diff1z4(f, ijk) / dev_grid.delta[1];
}

// __device__ inline Scalar
// KO(const Scalar *f, const Scalar R0, int ijk) {
//   if (FFE_DISSIPATION_ORDER == 4) {
//     if (R0 - 2 * dev_grid.delta[0] < 0) return diff4aR2(f, ijk) + diff4z2(f, ijk);
//     else return diff4R2(f, ijk) + diff4z2(f, ijk);
//   }
//   if (FFE_DISSIPATION_ORDER == 6) {
//     if (R0 - 2 * dev_grid.delta[0] < 0) return diff6aR2(f, ijk) + diff6z2(f, ijk);
//     else return diff6R2(f, ijk) + diff6z2(f, ijk);
//   }
// }

__device__ inline Scalar
KO(const Scalar *f, const Scalar R0, int ijk) {
  if (FFE_DISSIPATION_ORDER == 4) {
    return diff4R2(f, ijk) + diff4z2(f, ijk);
  }
  if (FFE_DISSIPATION_ORDER == 6) {
    return diff6R2(f, ijk) + diff6z2(f, ijk);
  }
}

__global__ void
kernel_rk_step1_thread(const Scalar *ER, const Scalar *Ez,
                       const Scalar *Ef, const Scalar *BR,
                       const Scalar *Bz, const Scalar *Bf, Scalar *dER,
                       Scalar *dEz, Scalar *dEf, Scalar *dBR,
                       Scalar *dBz, Scalar *dBf, const Scalar *P,
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

    Scalar R = dev_grid.pos(0, i, 1);

    if (std::abs(R) < TINY) R = TINY;

    Scalar rotBR = -dfdz(Bf, ijk);
    Scalar rotBz = dfRdR(Bf, R, ijk) / R;
    Scalar rotBf = dfdz(BR, ijk) - dfdR(Bz, R, ijk);
    Scalar rotER = -dfdz(Ef, ijk);
    Scalar rotEz = dfRdR(Ef, R, ijk) / R;
    Scalar rotEf = dfdz(ER, ijk) - dfdR(Ez, R, ijk);

    Scalar divE = dfRdR(ER, R, ijk) / R + dfdz(Ez, ijk);
    Scalar divB = dfRdR(BR, R, ijk) / R + dfdz(Bz, ijk);

    Scalar B2 =
        BR[ijk] * BR[ijk] + Bz[ijk] * Bz[ijk] + Bf[ijk] * Bf[ijk];
    if (B2 < TINY) B2 = TINY;

    Scalar Jp = (BR[ijk] * rotBR + Bz[ijk] * rotBz + Bf[ijk] * rotBf) -
                (ER[ijk] * rotER + Ez[ijk] * rotEz + Ef[ijk] * rotEf);
    Scalar JR = (divE * (Ef[ijk] * Bz[ijk] - Ez[ijk] * Bf[ijk]) +
                 Jp * BR[ijk]) /
                B2;
    Scalar Jf = (divE * (Ez[ijk] * BR[ijk] - ER[ijk] * Bz[ijk]) +
                 Jp * Bf[ijk]) /
                B2;
    Scalar Jz = (divE * (ER[ijk] * Bf[ijk] - Ef[ijk] * BR[ijk]) +
                 Jp * Bz[ijk]) /
                B2;

    dBR[ijk] = As * dBR[ijk] - dev_params.dt * (rotER + dfdR(P, R, ijk));
    dBf[ijk] = As * dBf[ijk] - dev_params.dt * (rotEf + 0.0);
    dBz[ijk] = As * dBz[ijk] - dev_params.dt * (rotEz + dfdz(P, ijk));

    dER[ijk] = As * dER[ijk] + dev_params.dt * (rotBR - JR);
    dEf[ijk] = As * dEf[ijk] + dev_params.dt * (rotBf - Jf);
    dEz[ijk] = As * dEz[ijk] + dev_params.dt * (rotBz - Jz);

    dP[ijk] = As * dP[ijk] - dev_params.dt * (dev_params.ch2 * divB +
                                              P[ijk] / dev_params.tau);
  }
}

__global__ void
kernel_rk_step2_thread(Scalar *ER, Scalar *Ez, Scalar *Ef, Scalar *BR,
                       Scalar *Bz, Scalar *Bf, const Scalar *dER,
                       const Scalar *dEz, const Scalar *dEf,
                       const Scalar *dBR, const Scalar *dBz,
                       const Scalar *dBf, const Scalar *dP, Scalar *P,
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

    ER[ijk] = ER[ijk] + Bs * dER[ijk];
    Ef[ijk] = Ef[ijk] + Bs * dEf[ijk];
    Ez[ijk] = Ez[ijk] + Bs * dEz[ijk];

    BR[ijk] = BR[ijk] + Bs * dBR[ijk];
    Bf[ijk] = Bf[ijk] + Bs * dBf[ijk];
    Bz[ijk] = Bz[ijk] + Bs * dBz[ijk];

    P[ijk] = P[ijk] + Bs * dP[ijk];
  }
}

__global__ void
kernel_Epar_thread(Scalar *Ex, Scalar *Ey, Scalar *Ez, const Scalar *Bx,
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
kernel_EgtB_thread(Scalar *Ex, Scalar *Ey, Scalar *Ez, const Scalar *Bx,
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
      Scalar s = std::sqrt(B2 / E2);
      Ex[ijk] *= s;
      Ey[ijk] *= s;
      Ez[ijk] *= s;
    }
  }
}

__global__ void
kernel_KO_step1_thread(Scalar *Ex, Scalar *Ey, Scalar *Ez, Scalar *Bx,
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
    Scalar R = dev_grid.pos(0, i, 1);

    Ex_tmp[ijk] = KO(Ex, R, ijk);
    Ey_tmp[ijk] = KO(Ey, R, ijk);
    Ez_tmp[ijk] = KO(Ez, R, ijk);

    Bx_tmp[ijk] = KO(Bx, R, ijk);
    By_tmp[ijk] = KO(By, R, ijk);
    Bz_tmp[ijk] = KO(Bz, R, ijk);

    P_tmp[ijk] = KO(P, R, ijk);
  }
}

__global__ void
kernel_KO_step2_thread(Scalar *Ex, Scalar *Ey, Scalar *Ez, Scalar *Bx,
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
kernel_boundary_axis_thread(Scalar *ER, Scalar *Ez, Scalar *Ef,
                            Scalar *BR, Scalar *Bz, Scalar *Bf,
                            Scalar *P, int shift) {
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
    Scalar R = dev_grid.pos(0, i, 1);
    if (std::abs(R) < dev_grid.delta[0] / 4.0) {
      ER[ijk] = 0.0;
      Ez[ijk] = Ez[ijk + 1];
      Ef[ijk] = 0.0;
      BR[ijk] = 0.0;
      Bz[ijk] = Bz[ijk + 1];
      Bf[ijk] = 0.0;
      P[ijk] = P[ijk + 1];
      for (int l = 1; l <= 3; ++l) {
        ER[ijk - l] = -ER[ijk + l];
        Ez[ijk - l] = Ez[ijk + l];
        Ef[ijk - l] = -Ef[ijk + l];
        BR[ijk - l] = -BR[ijk + l];
        Bz[ijk - l] = Bz[ijk + l];
        Bf[ijk - l] = -Bf[ijk + l];
        P[ijk - l] = P[ijk + l];
      }
    }
  }
}

__device__ Scalar
wpert(Scalar t, Scalar z) {
  Scalar z1 = 0.0;
  Scalar z2 = dev_params.radius * std::sqrt(1.0 - 1.0 / dev_params.rpert);
  if (t >= dev_params.tp_start && t <= dev_params.tp_end && z >= z1 &&
      z <= z2)
    return dev_params.dw0 * sin((z - z1) * M_PI / (z2 - z1)) *
           sin((t - dev_params.tp_start) * 2.0 * M_PI /
               (dev_params.tp_end - dev_params.tp_start));
  else return 0;
}

__global__ void
kernel_boundary_pulsar_thread(Scalar *ER, Scalar *Ez, Scalar *Ef,
                              Scalar *BR, Scalar *Bz, Scalar *Bf,
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
    Scalar R = dev_grid.pos(0, i, 1);
    Scalar z = dev_grid.pos(1, j, 1);
    Scalar r = std::sqrt(R * R + z * z);
    if (r < TINY) r = TINY;
    Scalar rl = 2.0 * dev_params.radius;
    // Scalar scale = 1.0 * dev_grid.delta[0];
    Scalar scaleEpar = 0.5 * dev_grid.delta[0];
    Scalar scaleEperp = 0.25 * dev_grid.delta[0];
    Scalar scaleBperp = scaleEpar;
    Scalar scaleBpar = scaleBperp;
    Scalar d1 = 4 * dev_grid.delta[0];
    Scalar d0 = 0;
    Scalar BRnew, Bznew, Bfnew, ERnew, Eznew, Efnew;
    if (r < rl) {
      Scalar bRn = dev_params.b0 * cube(dev_params.radius) *
                   dipole_x(R, 0, z, 0.0, 0.0);
      Scalar bzn = dev_params.b0 * cube(dev_params.radius) *
                   dipole_z(R, 0, z, 0.0, 0.0);
      Scalar bfn = 0.0;
      Scalar s = shape(r, dev_params.radius - d1, scaleBperp);
      BRnew = (bRn * R + bzn * z) * R / (r * r) * s +
              (BR[ijk] * R + Bz[ijk] * z) * R / (r * r) * (1 - s);
      Bznew = (bRn * R + bzn * z) * z / (r * r) * s +
              (BR[ijk] * R + Bz[ijk] * z) * z / (r * r) * (1 - s);
      s = shape(r, dev_params.radius - d1, scaleBpar);
      BRnew += (bRn - (bRn * R + bzn * z) * R / (r * r)) * s +
               (BR[ijk] - (BR[ijk] * R + Bz[ijk] * z) * R / (r * r)) *
                   (1 - s);
      Bznew += (bzn - (bRn * R + bzn * z) * z / (r * r)) * s +
               (Bz[ijk] - (BR[ijk] * R + Bz[ijk] * z) * z / (r * r)) *
                   (1 - s);
      Bfnew = bfn * s + Bf[ijk] * (1 - s);

      // Scalar w = dev_params.omega;
      Scalar w = wpert(t, z);
      Scalar eRn = - w * R * Bz[ijk];
      Scalar ezn = w * R * BR[ijk];
      Scalar efn = 0.0;
      s = shape(r, dev_params.radius - d0, scaleEperp);
      ERnew = (eRn * R + ezn * z) * R / (r * r) * s +
              (ER[ijk] * R + Ez[ijk] * z) * R / (r * r) * (1 - s);
      Eznew = (eRn * R + ezn * z) * z / (r * r) * s +
              (ER[ijk] * R + Ez[ijk] * z) * z / (r * r) * (1 - s);
      s = shape(r, dev_params.radius - d0, scaleEpar);
      ERnew += (eRn - (eRn * R + ezn * z) * R / (r * r)) * s +
               (ER[ijk] - (ER[ijk] * R + Ez[ijk] * z) * R / (r * r)) *
                   (1 - s);
      Eznew += (ezn - (eRn * R + ezn * z) * z / (r * r)) * s +
               (Ez[ijk] - (ER[ijk] * R + Ez[ijk] * z) * z / (r * r)) *
                   (1 - s);
      Efnew = efn * s + Ef[ijk] * (1 - s);
      BR[ijk] = BRnew;
      Bz[ijk] = Bznew;
      Bf[ijk] = Bfnew;
      ER[ijk] = ERnew;
      Ez[ijk] = Eznew;
      Ef[ijk] = Efnew;
    }
  }
}

HOST_DEVICE Scalar
pmlsigma(Scalar x, Scalar xl, Scalar xh, Scalar pmlscale, Scalar sig0) {
  if (x > xh)
    return sig0 * pow((x - xh) / pmlscale, 3.0);
  else if (x < xl)
    return sig0 * pow((xl - x) / pmlscale, 3.0);
  else
    return 0.0;
}

__global__ void
kernel_boundary_absorbing_thread(const Scalar *enx, const Scalar *eny,
                                 const Scalar *enz, const Scalar *bnx,
                                 const Scalar *bny, const Scalar *bnz,
                                 Scalar *ex, Scalar *ey, Scalar *ez,
                                 Scalar *bx, Scalar *by, Scalar *bz,
                                 int shift) {
  Scalar x, y, z;
  Scalar sigx = 0.0, sigy = 0.0, sigz = 0.0, sig = 0.0;
  size_t ijk;
  Scalar dx = dev_grid.delta[0] / 2.0;
  Scalar dy = dev_grid.delta[1] / 2.0;
  Scalar dz = dev_grid.delta[2] / 2.0;
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
    Scalar xh = dev_params.lower[0] + dev_params.size[0] -
                dev_params.pml[0] * dev_grid.delta[0];
    Scalar xl = - xh;
    //     dev_params.lower[0] + dev_params.pml[0] * dev_grid.delta[0];
    Scalar yh = dev_params.lower[1] + dev_params.size[1] -
                dev_params.pml[1] * dev_grid.delta[1];
    Scalar yl =
        dev_params.lower[1] + dev_params.pml[1] * dev_grid.delta[1];
    // Scalar zh = dev_params.lower[2] + dev_params.size[2] -
    //             dev_params.pml[2] * dev_grid.delta[2];
    // Scalar zl =
    //     dev_params.lower[2] + dev_params.pml[2] * dev_grid.delta[2];
    // if (x > xh || x < xl || y > yh || y < yl || z > zh || z < zl) {
      if (x > xh || y < yl || y > yh) {
      sigx = pmlsigma(x, xl, xh, dev_params.pmllen * dev_grid.delta[0],
                      dev_params.sigpml);
      sigy = pmlsigma(y, yl, yh, dev_params.pmllen * dev_grid.delta[0],
                      dev_params.sigpml);
      // sigz = pmlsigma(z, zl, zh, dev_params.pmllen * dev_grid.delta[0],
      //                 dev_params.sigpml);
      // sig = sigx + sigy + sigz;
      sig = sigx + sigy;
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

void
field_solver_EZ_cylindrical::rk_step(Scalar As, Scalar Bs) {
  kernel_rk_step1_thread<<<blockGroupSize, blockSize>>>(
      m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
      m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2),
      dE.dev_ptr(0), dE.dev_ptr(1), dE.dev_ptr(2), dB.dev_ptr(0),
      dB.dev_ptr(1), dB.dev_ptr(2), P.dev_ptr(), dP.dev_ptr(),
      m_env.params().shift_ghost, As);
  CudaCheckError();
  kernel_rk_step2_thread<<<blockGroupSize, blockSize>>>(
      m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
      m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2),
      dE.dev_ptr(0), dE.dev_ptr(1), dE.dev_ptr(2), dB.dev_ptr(0),
      dB.dev_ptr(1), dB.dev_ptr(2), P.dev_ptr(), dP.dev_ptr(),
      m_env.params().shift_ghost, Bs);
  CudaCheckError();
}

void
field_solver_EZ_cylindrical::Kreiss_Oliger() {
  kernel_KO_step1_thread<<<blockGroupSize, blockSize>>>(
      m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
      m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2),
      Etmp.dev_ptr(0), Etmp.dev_ptr(1), Etmp.dev_ptr(2),
      Btmp.dev_ptr(0), Btmp.dev_ptr(1), Btmp.dev_ptr(2), P.dev_ptr(),
      Ptmp.dev_ptr(), m_env.params().shift_ghost);
  CudaCheckError();
  kernel_KO_step2_thread<<<blockGroupSize, blockSize>>>(
      m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
      m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2),
      Etmp.dev_ptr(0), Etmp.dev_ptr(1), Etmp.dev_ptr(2),
      Btmp.dev_ptr(0), Btmp.dev_ptr(1), Btmp.dev_ptr(2), P.dev_ptr(),
      Ptmp.dev_ptr(), m_env.params().shift_ghost);
  CudaCheckError();
}

void
field_solver_EZ_cylindrical::clean_epar() {
  kernel_Epar_thread<<<blockGroupSize, blockSize>>>(
      m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
      m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2),
      m_env.params().shift_ghost);
  CudaCheckError();
}

void
field_solver_EZ_cylindrical::check_eGTb() {
  kernel_EgtB_thread<<<blockGroupSize, blockSize>>>(
      m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
      m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2),
      m_env.params().shift_ghost);
  CudaCheckError();
}

void
field_solver_EZ_cylindrical::boundary_axis() {
  kernel_boundary_axis_thread<<<blockGroupSize, blockSize>>>(
      m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
      m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2),
      P.dev_ptr(), m_env.params().shift_ghost);
  CudaCheckError();
}

void
field_solver_EZ_cylindrical::boundary_pulsar(Scalar t) {
  kernel_boundary_pulsar_thread<<<blockGroupSize, blockSize>>>(
      m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
      m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2),
      P.dev_ptr(), t, m_env.params().shift_ghost);
  CudaCheckError();
}

void
field_solver_EZ_cylindrical::boundary_absorbing() {
  kernel_boundary_absorbing_thread<<<blockGroupSize, blockSize>>>(
      Etmp.dev_ptr(0), Etmp.dev_ptr(1), Etmp.dev_ptr(2),
      Btmp.dev_ptr(0), Btmp.dev_ptr(1), Btmp.dev_ptr(2),
      m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
      m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2),
      m_env.params().shift_ghost);
  CudaCheckError();
}

void
field_solver_EZ_cylindrical::evolve_fields(Scalar time) {
  Scalar As[5] = {0, -0.4178904745, -1.192151694643, -1.697784692471,
                  -1.514183444257};
  Scalar Bs[5] = {0.1496590219993, 0.3792103129999, 0.8229550293869,
                  0.6994504559488, 0.1530572479681};
  Scalar cs[5] = {0.1496590219993, 0.3704009573644, 0.6222557631345,
                  0.9582821306784, 1.0};

  Etmp.copy_from(m_data.E);
  Btmp.copy_from(m_data.B);

  for (int i = 0; i < 5; ++i) {
    rk_step(As[i], Bs[i]);

    if (m_env.params().clean_ep) clean_epar();
    if (m_env.params().check_egb) check_eGTb();

    boundary_axis();
    boundary_pulsar(time + cs[i] * m_env.params().dt);
    if (i == 4) boundary_absorbing();

    CudaSafeCall(cudaDeviceSynchronize());
    m_env.send_guard_cells(m_data);
    m_env.send_guard_cell_array(P);
  }

  Kreiss_Oliger();
  if (m_env.params().clean_ep) clean_epar();
  if (m_env.params().check_egb) check_eGTb();
  boundary_axis();
  CudaSafeCall(cudaDeviceSynchronize());
  m_env.send_guard_cells(m_data);
  m_env.send_guard_cell_array(P);
}

field_solver_EZ_cylindrical::field_solver_EZ_cylindrical(
    sim_data &mydata, sim_environment &env)
    : m_data(mydata), m_env(env) {
  dE = vector_field<Scalar>(m_data.env.grid());
  Etmp = vector_field<Scalar>(m_data.env.grid());
  dE.copy_stagger(m_data.E);
  Etmp.copy_stagger(m_data.E);
  dE.initialize();
  Etmp.initialize();

  dB = vector_field<Scalar>(m_data.env.grid());
  Btmp = vector_field<Scalar>(m_data.env.grid());
  dB.copy_stagger(m_data.B);
  Btmp.copy_stagger(m_data.B);
  dB.initialize();
  Btmp.initialize();

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

field_solver_EZ_cylindrical::~field_solver_EZ_cylindrical() {}

Scalar
field_solver_EZ_cylindrical::total_energy(vector_field<Scalar> &f) {
  f.sync_to_host();
  Scalar Wtmp = 0.0, W = 0.0;
  for (int k = m_env.grid().guard[2] + m_env.params().pml[2];
       k < m_env.grid().dims[2] - m_env.grid().guard[2] -
               m_env.params().pml[2];
       ++k) {
    for (int j = m_env.grid().guard[1] + m_env.params().pml[1];
         j < m_env.grid().dims[1] - m_env.grid().guard[1] -
                 m_env.params().pml[1];
         ++j) {
      for (int i = m_env.grid().guard[0];
           i < m_env.grid().dims[0] - m_env.grid().guard[0] -
                   m_env.params().pml[0];
           ++i) {
        int ijk = i + j * m_env.grid().dims[0] +
                  k * m_env.grid().dims[0] * m_env.grid().dims[1];
        Scalar R = m_env.grid().pos(0, i, 1);
        Wtmp += (f.data(0)[ijk] * f.data(0)[ijk] +
                 f.data(1)[ijk] * f.data(1)[ijk] +
                 f.data(2)[ijk] * f.data(2)[ijk]) *
                R;
      }
    }
  }
  MPI_Reduce(&Wtmp, &W, 1, m_env.scalar_type(), MPI_SUM, 0,
             m_env.world());
  return W;
}

}  // namespace Coffee
