#include "cuda/constant_mem.h"
#include "cuda/constant_mem_func.h"
#include "cuda/cuda_utility.h"
#include "field_solver_resistive.h"
#include "interpolation.h"
#include "utils/timer.h"
#include "utils/nvproftool.h"
#include <cmath>
#include <iomanip>

#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>

#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 4
#define BLOCK_SIZE_Z 4


#define TINY 1e-7

// using namespace H5;
using namespace HighFive;

namespace Coffee {

// static dim3 gridSize(8, 16, 16);
static dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);

static dim3 blockGroupSize;

template <typename T> 
HD_INLINE int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

template <typename T>
HD_INLINE T square(T x) { return x * x; }

__global__ void
kernel_compute_rho_rsstv(const Scalar *ex, const Scalar *ey, const Scalar *ez,
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
    rho[ijk] = dev_grid.inv_delta[0] * (ex[ijk] - ex[ijk - 1]) +
               dev_grid.inv_delta[1] * (ey[ijk] - ey[ijk - dev_grid.dims[0]]) +
               dev_grid.inv_delta[2] * (ez[ijk] - ez[ijk - dev_grid.dims[0] * dev_grid.dims[1]]);
  }
}



__global__ void
kernel_rk_push_noj_rsstv(const Scalar *ex, const Scalar *ey, const Scalar *ez,
                      const Scalar *bx, const Scalar *by, const Scalar *bz,
                      const Scalar *bx0, const Scalar *by0,
                      const Scalar *bz0, Scalar *dex, Scalar *dey,
                      Scalar *dez, Scalar *dbx, Scalar *dby, Scalar *dbz,
                      Scalar *rho, int shift) {
  Scalar CCx = dev_params.dt * dev_grid.inv_delta[0];
  Scalar CCy = dev_params.dt * dev_grid.inv_delta[1];
  Scalar CCz = dev_params.dt * dev_grid.inv_delta[2];
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
    dbx[ijk] = CCz * (ey[ijkP1] - ey[ijk]) - CCy * (ez[ijP1k] - ez[ijk]);
    dby[ijk] = CCx * (ez[iP1jk] - ez[ijk]) - CCz * (ex[ijkP1] - ex[ijk]);
    dbz[ijk] = CCy * (ex[ijP1k] - ex[ijk]) - CCx * (ey[iP1jk] - ey[ijk]);
    // push E-field
    dex[ijk] = (CCz * (by[ijkM1] - by[ijk]) - CCy * (bz[ijM1k] - bz[ijk])) -
                      (CCz * (by0[ijkM1] - bz0[ijk]) - CCy * (bz0[ijM1k] - bz0[ijk]));
    dey[ijk] = (CCx * (bz[iM1jk] - bz[ijk]) - CCz * (bx[ijkM1] - bx[ijk])) -
                      (CCx * (bz0[iM1jk] - bz0[ijk]) - CCz * (bx0[ijkM1] - bx0[ijk]));
    dez[ijk] = (CCy * (bx[ijM1k] - bx[ijk]) - CCx * (by[iM1jk] - by[ijk])) -
                      (CCy * (bx0[ijM1k] - bx0[ijk]) - CCx * (by0[iM1jk] - by0[ijk]));
    
  }
}

__global__ void
kernel_rk_push_ffjperp_rsstv(const Scalar *ex, const Scalar *ey, const Scalar *ez,
                      const Scalar *bx, const Scalar *by, const Scalar *bz,
                      Scalar *dex, Scalar *dey, Scalar *dez, 
                      Scalar *dbx, Scalar *dby, Scalar *dbz,
                      Scalar *rho, int shift) {
  Scalar intex, intey, intez, intbx, intby, intbz, intrho;
  Scalar jx, jy, jz;
  size_t ijk;

  int i = threadIdx.x + blockIdx.x * blockDim.x + dev_grid.guard[0] - shift;
  int j = threadIdx.y + blockIdx.y * blockDim.y + dev_grid.guard[1] - shift;
  int k = threadIdx.z + blockIdx.z * blockDim.z + dev_grid.guard[2] - shift;
  if (i < dev_grid.dims[0] - dev_grid.guard[0] + shift &&
      j < dev_grid.dims[1] - dev_grid.guard[1] + shift &&
      k < dev_grid.dims[2] - dev_grid.guard[2] + shift) {
    ijk = i + j * dev_grid.dims[0] +
          k * dev_grid.dims[0] * dev_grid.dims[1];
    // computing currents
    //   `j_x`:
    intrho = interpolate(rho, ijk, Stagger(0b111), Stagger(0b110),
                         dev_grid.dims[0], dev_grid.dims[1]);
    intex = interpolate(ex, ijk, Stagger(0b110), Stagger(0b110),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intey = interpolate(ey, ijk, Stagger(0b101), Stagger(0b110),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intez = interpolate(ez, ijk, Stagger(0b011), Stagger(0b110),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intbx = interpolate(bx, ijk, Stagger(0b001), Stagger(0b110),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intby = interpolate(by, ijk, Stagger(0b010), Stagger(0b110),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intbz = interpolate(bz, ijk, Stagger(0b100), Stagger(0b110),
                       dev_grid.dims[0], dev_grid.dims[1]);
    jx = dev_params.dt * intrho * (intey * intbz - intby * intez) /
         (intbx * intbx + intby * intby + intbz * intbz + TINY);
    //   `j_y`:
    intrho = interpolate(rho, ijk, Stagger(0b111), Stagger(0b101),
                         dev_grid.dims[0], dev_grid.dims[1]);
    intex = interpolate(ex, ijk, Stagger(0b110), Stagger(0b101),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intey = interpolate(ey, ijk, Stagger(0b101), Stagger(0b101),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intez = interpolate(ez, ijk, Stagger(0b011), Stagger(0b101),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intbx = interpolate(bx, ijk, Stagger(0b001), Stagger(0b101),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intby = interpolate(by, ijk, Stagger(0b010), Stagger(0b101),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intbz = interpolate(bz, ijk, Stagger(0b100), Stagger(0b101),
                       dev_grid.dims[0], dev_grid.dims[1]);
    jy = dev_params.dt * intrho * (intez * intbx - intex * intbz) /
         (intbx * intbx + intby * intby + intbz * intbz + TINY);
    //   `j_z`:
    intrho = interpolate(rho, ijk, Stagger(0b111), Stagger(0b011),
                         dev_grid.dims[0], dev_grid.dims[1]);
    intex = interpolate(ex, ijk, Stagger(0b110), Stagger(0b011),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intey = interpolate(ey, ijk, Stagger(0b101), Stagger(0b011),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intez = interpolate(ez, ijk, Stagger(0b011), Stagger(0b011),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intbx = interpolate(bx, ijk, Stagger(0b001), Stagger(0b011),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intby = interpolate(by, ijk, Stagger(0b010), Stagger(0b011),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intbz = interpolate(bz, ijk, Stagger(0b100), Stagger(0b011),
                       dev_grid.dims[0], dev_grid.dims[1]);
    jz = dev_params.dt * intrho * (intex * intby - intbx * intey) /
         (intbx * intbx + intby * intby + intbz * intbz + TINY);

    dex[ijk] -= jx;
    dey[ijk] -= jy;
    dez[ijk] -= jz;
  }
}


HOST_DEVICE Scalar iphi(Scalar r, Scalar ri, Scalar r0, Scalar mag, Scalar alpha) {
  Scalar tmp = (r - ri) * 2.0 * M_PI / r0 + M_PI / 2.0;
  if (tmp < 2.0 * M_PI && tmp > 0) 
    return mag * sin(tmp) / pow(r / ri, alpha);
  else 
    return 0.0;
}

__global__ void
kernel_rk_push_jvacuum_rsstv(Scalar *dex, Scalar *dey, Scalar *dez, int shift) {
  Scalar x, y, z, rd;
  Scalar jx, jy;
  size_t ijk;

  int i = threadIdx.x + blockIdx.x * blockDim.x + dev_grid.guard[0] - shift;
  int j = threadIdx.y + blockIdx.y * blockDim.y + dev_grid.guard[1] - shift;
  int k = threadIdx.z + blockIdx.z * blockDim.z + dev_grid.guard[2] - shift;
  if (i < dev_grid.dims[0] - dev_grid.guard[0] + shift &&
      j < dev_grid.dims[1] - dev_grid.guard[1] + shift &&
      k < dev_grid.dims[2] - dev_grid.guard[2] + shift) {
    ijk = i + j * dev_grid.dims[0] +
          k * dev_grid.dims[0] * dev_grid.dims[1];
    // computing currents
    z = dev_grid.pos(2, k, 1);
    if (std::abs(z) < dev_grid.delta[2] / 4.0) {
      // jx
      x = dev_grid.pos(0, i, 0);
      y = dev_grid.pos(1, j, 1);
      rd = sqrt(x * x + y * y);
      jx = - y / rd * iphi(rd, dev_params.r2, dev_params.wid, dev_params.j0, dev_params.alpha);
      dex[ijk] -= dev_params.dt * jx;
      // jy
      x = dev_grid.pos(0, i, 1);
      y = dev_grid.pos(1, j, 0);
      rd = sqrt(x * x + y * y);
      jy = x / rd * iphi(rd, dev_params.r2, dev_params.wid, dev_params.j0, dev_params.alpha);
      dey[ijk] -= dev_params.dt * jy;
    }
  }
}

__global__ void
kernel_rk_push_rjperp_rsstv(const Scalar *ex, const Scalar *ey, const Scalar *ez,
                      const Scalar *bx, const Scalar *by, const Scalar *bz,
                      Scalar *dex, Scalar *dey, Scalar *dez, 
                      Scalar *dbx, Scalar *dby, Scalar *dbz,
                      Scalar *rho, int shift) {
  Scalar intex, intey, intez, intbx, intby, intbz, intrho;
  Scalar jxperp, jyperp, jzperp;
  Scalar Bsq, Esq, edotb, B0sq, E00, B00, E0sq;
  size_t ijk;

  int i = threadIdx.x + blockIdx.x * blockDim.x + dev_grid.guard[0] - shift;
  int j = threadIdx.y + blockIdx.y * blockDim.y + dev_grid.guard[1] - shift;
  int k = threadIdx.z + blockIdx.z * blockDim.z + dev_grid.guard[2] - shift;
  if (i < dev_grid.dims[0] - dev_grid.guard[0] + shift &&
      j < dev_grid.dims[1] - dev_grid.guard[1] + shift &&
      k < dev_grid.dims[2] - dev_grid.guard[2] + shift) {
    ijk = i + j * dev_grid.dims[0] +
          k * dev_grid.dims[0] * dev_grid.dims[1];

    // jxperp
    intrho = interpolate(rho, ijk, Stagger(0b111), Stagger(0b110),
                         dev_grid.dims[0], dev_grid.dims[1]);
    intex = interpolate(ex, ijk, Stagger(0b110), Stagger(0b110),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intey = interpolate(ey, ijk, Stagger(0b101), Stagger(0b110),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intez = interpolate(ez, ijk, Stagger(0b011), Stagger(0b110),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intbx = interpolate(bx, ijk, Stagger(0b001), Stagger(0b110),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intby = interpolate(by, ijk, Stagger(0b010), Stagger(0b110),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intbz = interpolate(bz, ijk, Stagger(0b100), Stagger(0b110),
                       dev_grid.dims[0], dev_grid.dims[1]);

    Bsq = intbx * intbx + intby * intby + intbz * intbz;
    Esq = intex * intex + intey * intey + intez * intez;
    edotb = intex * intbx + intey * intby + intez * intbz;
    B0sq = std::abs(0.5 * ((Bsq - Esq) + sqrt(square(Bsq - Esq) + 4.0 * square(edotb))));
    E00 = sqrt(std::abs(B0sq - Bsq + Esq));
    B00 = sgn (edotb) * sqrt(B0sq);
    E0sq = E00 * E00;
    jxperp = dev_params.dt / (Bsq + E0sq + TINY) * intrho * (intey * intbz - intez * intby);

    //jyperp
    intrho = interpolate(rho, ijk, Stagger(0b111), Stagger(0b101),
                         dev_grid.dims[0], dev_grid.dims[1]);
    intex = interpolate(ex, ijk, Stagger(0b110), Stagger(0b101),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intey = interpolate(ey, ijk, Stagger(0b101), Stagger(0b101),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intez = interpolate(ez, ijk, Stagger(0b011), Stagger(0b101),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intbx = interpolate(bx, ijk, Stagger(0b001), Stagger(0b101),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intby = interpolate(by, ijk, Stagger(0b010), Stagger(0b101),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intbz = interpolate(bz, ijk, Stagger(0b100), Stagger(0b101),
                       dev_grid.dims[0], dev_grid.dims[1]);

    Bsq = intbx * intbx + intby * intby + intbz * intbz;
    Esq = intex * intex + intey * intey + intez * intez;
    edotb = intex * intbx + intey * intby + intez * intbz;
    B0sq = std::abs(0.5 * ((Bsq - Esq) + sqrt(square(Bsq - Esq) + 4.0 * square(edotb))));
    E00 = sqrt(std::abs(B0sq - Bsq + Esq));
    B00 = sgn (edotb) * sqrt(B0sq);
    E0sq = E00 * E00;
    jyperp = dev_params.dt / (Bsq + E0sq + TINY) * intrho * (intez * intbx - intex * intbz);

    //jzperp
    intrho = interpolate(rho, ijk, Stagger(0b111), Stagger(0b011),
                         dev_grid.dims[0], dev_grid.dims[1]);
    intex = interpolate(ex, ijk, Stagger(0b110), Stagger(0b011),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intey = interpolate(ey, ijk, Stagger(0b101), Stagger(0b011),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intez = interpolate(ez, ijk, Stagger(0b011), Stagger(0b011),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intbx = interpolate(bx, ijk, Stagger(0b001), Stagger(0b011),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intby = interpolate(by, ijk, Stagger(0b010), Stagger(0b011),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intbz = interpolate(bz, ijk, Stagger(0b100), Stagger(0b011),
                       dev_grid.dims[0], dev_grid.dims[1]);

    Bsq = intbx * intbx + intby * intby + intbz * intbz;
    Esq = intex * intex + intey * intey + intez * intez;
    edotb = intex * intbx + intey * intby + intez * intbz;
    B0sq = std::abs(0.5 * ((Bsq - Esq) + sqrt(square(Bsq - Esq) + 4.0 * square(edotb))));
    E00 = sqrt(std::abs(B0sq - Bsq + Esq));
    B00 = sgn (edotb) * sqrt(B0sq);
    E0sq = E00 * E00;
    jzperp = dev_params.dt / (Bsq + E0sq + TINY) * intrho * (intex * intby - intey * intbx);

    dex[ijk] -= jxperp;
    dey[ijk] -= jyperp;
    dez[ijk] -= jzperp;
  }
}


__global__ void
kernel_rk_update_rsstv(Scalar *ex, Scalar *ey, Scalar *ez, Scalar *bx,
                        Scalar *by, Scalar *bz, const Scalar *enx,
                        const Scalar *eny, const Scalar *enz,
                        const Scalar *bnx, const Scalar *bny,
                        const Scalar *bnz, Scalar *dex, Scalar *dey,
                        Scalar *dez, const Scalar *dbx, const Scalar *dby,
                        const Scalar *dbz, Scalar rk_c1, Scalar rk_c2,
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
    // update E-field
    ex[ijk] = rk_c1 * enx[ijk] + rk_c2 * ex[ijk] + rk_c3 * dex[ijk];
    ey[ijk] = rk_c1 * eny[ijk] + rk_c2 * ey[ijk] + rk_c3 * dey[ijk];
    ez[ijk] = rk_c1 * enz[ijk] + rk_c2 * ez[ijk] + rk_c3 * dez[ijk];
    dex[ijk] = ex[ijk];
    dey[ijk] = ey[ijk];
    dez[ijk] = ez[ijk];
    // update B-field
    bx[ijk] = rk_c1 * bnx[ijk] + rk_c2 * bx[ijk] + rk_c3 * dbx[ijk];
    by[ijk] = rk_c1 * bny[ijk] + rk_c2 * by[ijk] + rk_c3 * dby[ijk];
    bz[ijk] = rk_c1 * bnz[ijk] + rk_c2 * bz[ijk] + rk_c3 * dbz[ijk];
  }
}

__global__ void
kernel_rk_update_rjparsub_rsstv(Scalar *ex, Scalar *ey, Scalar *ez, Scalar *bx,
                        Scalar *by, Scalar *bz, Scalar *dex, Scalar *dey,
                        Scalar *dez, Scalar *rho, Scalar rk_c3, int shift) {
  Scalar intex, intey, intez, intbx, intby, intbz, intrho;
  Scalar jxpar, jypar, jzpar;
  Scalar Bsq, Esq, edotb, B0sq, E00, B00, E0sq;
  size_t ijk;
  int i = threadIdx.x + blockIdx.x * blockDim.x + dev_grid.guard[0] - shift;
  int j = threadIdx.y + blockIdx.y * blockDim.y + dev_grid.guard[1] - shift;
  int k = threadIdx.z + blockIdx.z * blockDim.z + dev_grid.guard[2] - shift;
  if (i < dev_grid.dims[0] - dev_grid.guard[0] + shift &&
      j < dev_grid.dims[1] - dev_grid.guard[1] + shift &&
      k < dev_grid.dims[2] - dev_grid.guard[2] + shift) {
    ijk = i + j * dev_grid.dims[0] +
          k * dev_grid.dims[0] * dev_grid.dims[1];

    // jxpar
    intrho = interpolate(rho, ijk, Stagger(0b111), Stagger(0b110),
                         dev_grid.dims[0], dev_grid.dims[1]);
    intex = interpolate(ex, ijk, Stagger(0b110), Stagger(0b110),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intey = interpolate(ey, ijk, Stagger(0b101), Stagger(0b110),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intez = interpolate(ez, ijk, Stagger(0b011), Stagger(0b110),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intbx = interpolate(bx, ijk, Stagger(0b001), Stagger(0b110),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intby = interpolate(by, ijk, Stagger(0b010), Stagger(0b110),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intbz = interpolate(bz, ijk, Stagger(0b100), Stagger(0b110),
                       dev_grid.dims[0], dev_grid.dims[1]);

    Bsq = intbx * intbx + intby * intby + intbz * intbz;
    Esq = intex * intex + intey * intey + intez * intez;
    edotb = intex * intbx + intey * intby + intez * intbz;
    B0sq = std::abs(0.5 * ((Bsq - Esq) + sqrt(square(Bsq - Esq) + 4.0 * square(edotb))));
    E00 = sqrt(std::abs(B0sq - Bsq + Esq));
    B00 = sgn (edotb) * sqrt(B0sq);
    E0sq = E00 * E00;

    jxpar = dev_params.dt / (Bsq + E0sq + TINY) * 
          sqrt((Bsq + E0sq) / (B0sq + E0sq + TINY) * dev_params.sigsq * E0sq) *
          (B00 * intbx + E00 * intex);

    //jypar
    intrho = interpolate(rho, ijk, Stagger(0b111), Stagger(0b101),
                         dev_grid.dims[0], dev_grid.dims[1]);
    intex = interpolate(ex, ijk, Stagger(0b110), Stagger(0b101),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intey = interpolate(ey, ijk, Stagger(0b101), Stagger(0b101),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intez = interpolate(ez, ijk, Stagger(0b011), Stagger(0b101),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intbx = interpolate(bx, ijk, Stagger(0b001), Stagger(0b101),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intby = interpolate(by, ijk, Stagger(0b010), Stagger(0b101),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intbz = interpolate(bz, ijk, Stagger(0b100), Stagger(0b101),
                       dev_grid.dims[0], dev_grid.dims[1]);

    Bsq = intbx * intbx + intby * intby + intbz * intbz;
    Esq = intex * intex + intey * intey + intez * intez;
    edotb = intex * intbx + intey * intby + intez * intbz;
    B0sq = std::abs(0.5 * ((Bsq - Esq) + sqrt(square(Bsq - Esq) + 4.0 * square(edotb))));
    E00 = sqrt(std::abs(B0sq - Bsq + Esq));
    B00 = sgn (edotb) * sqrt(B0sq);
    E0sq = E00 * E00;

    jypar = dev_params.dt / (Bsq + E0sq + TINY) * 
          sqrt((Bsq + E0sq) / (B0sq + E0sq + TINY) * dev_params.sigsq * E0sq) *
          (B00 * intby + E00 * intey);

    //jzpar
    intrho = interpolate(rho, ijk, Stagger(0b111), Stagger(0b011),
                         dev_grid.dims[0], dev_grid.dims[1]);
    intex = interpolate(ex, ijk, Stagger(0b110), Stagger(0b011),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intey = interpolate(ey, ijk, Stagger(0b101), Stagger(0b011),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intez = interpolate(ez, ijk, Stagger(0b011), Stagger(0b011),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intbx = interpolate(bx, ijk, Stagger(0b001), Stagger(0b011),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intby = interpolate(by, ijk, Stagger(0b010), Stagger(0b011),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intbz = interpolate(bz, ijk, Stagger(0b100), Stagger(0b011),
                       dev_grid.dims[0], dev_grid.dims[1]);

    Bsq = intbx * intbx + intby * intby + intbz * intbz;
    Esq = intex * intex + intey * intey + intez * intez;
    edotb = intex * intbx + intey * intby + intez * intbz;
    B0sq = std::abs(0.5 * ((Bsq - Esq) + sqrt(square(Bsq - Esq) + 4.0 * square(edotb))));
    E00 = sqrt(std::abs(B0sq - Bsq + Esq));
    B00 = sgn (edotb) * sqrt(B0sq);
    E0sq = E00 * E00;

    jzpar = dev_params.dt / (Bsq + E0sq + TINY) * 
          sqrt((Bsq + E0sq) / (B0sq + E0sq + TINY) * dev_params.sigsq * E0sq) *
          (B00 * intbz + E00 * intez);

    dex[ijk] = ex[ijk] - jxpar / dev_params.subsamp * rk_c3;
    dey[ijk] = ey[ijk] - jypar / dev_params.subsamp * rk_c3;
    dez[ijk] = ez[ijk] - jzpar / dev_params.subsamp * rk_c3;
  }
}



__global__ void
kernel_clean_epar_rsstv(const Scalar *ex, const Scalar *ey, const Scalar *ez,
                         const Scalar *bx, const Scalar *by, const Scalar *bz,
                         Scalar *dex, Scalar *dey, Scalar *dez, int shift) {
  Scalar intex, intey, intez, intbx, intby, intbz;
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
    intex = interpolate(ex, ijk, Stagger(0b110), Stagger(0b110),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intey = interpolate(ey, ijk, Stagger(0b101), Stagger(0b110),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intez = interpolate(ez, ijk, Stagger(0b011), Stagger(0b110),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intbx = interpolate(bx, ijk, Stagger(0b001), Stagger(0b110),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intby = interpolate(by, ijk, Stagger(0b010), Stagger(0b110),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intbz = interpolate(bz, ijk, Stagger(0b100), Stagger(0b110),
                        dev_grid.dims[0], dev_grid.dims[1]);
    dex[ijk] = ex[ijk] -
               (intex * intbx + intey * intby + intez * intbz) *
                   intbx /
                   (intbx * intbx + intby * intby + intbz * intbz + TINY);

    // y:
    intex = interpolate(ex, ijk, Stagger(0b110), Stagger(0b101),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intey = interpolate(ey, ijk, Stagger(0b101), Stagger(0b101),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intez = interpolate(ez, ijk, Stagger(0b011), Stagger(0b101),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intbx = interpolate(bx, ijk, Stagger(0b001), Stagger(0b101),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intby = interpolate(by, ijk, Stagger(0b010), Stagger(0b101),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intbz = interpolate(bz, ijk, Stagger(0b100), Stagger(0b101),
                        dev_grid.dims[0], dev_grid.dims[1]);
    dey[ijk] = ey[ijk] -
               (intex * intbx + intey * intby + intez * intbz) *
                   intby /
                   (intbx * intbx + intby * intby + intbz * intbz + TINY);

    // z:
    intex = interpolate(ex, ijk, Stagger(0b110), Stagger(0b011),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intey = interpolate(ey, ijk, Stagger(0b101), Stagger(0b011),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intez = interpolate(ez, ijk, Stagger(0b011), Stagger(0b011),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intbx = interpolate(bx, ijk, Stagger(0b001), Stagger(0b011),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intby = interpolate(by, ijk, Stagger(0b010), Stagger(0b011),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intbz = interpolate(bz, ijk, Stagger(0b100), Stagger(0b011),
                        dev_grid.dims[0], dev_grid.dims[1]);
    dez[ijk] = ez[ijk] -
               (intex * intbx + intey * intby + intez * intbz) *
                   intbz /
                   (intbx * intbx + intby * intby + intbz * intbz + TINY);
  }
}



__global__ void
kernel_check_eGTb_rsstv(const Scalar *dex, const Scalar *dey,
                         const Scalar *dez, Scalar *ex, Scalar *ey, Scalar *ez,
                         const Scalar *bx, const Scalar *by,
                         const Scalar *bz, int shift) {
  Scalar intex, intey, intez, intbx, intby, intbz, emag, bmag, temp;
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
    intex = interpolate(ex, ijk, Stagger(0b110), Stagger(0b110),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intey = interpolate(ey, ijk, Stagger(0b101), Stagger(0b110),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intez = interpolate(ez, ijk, Stagger(0b011), Stagger(0b110),
                        dev_grid.dims[0], dev_grid.dims[1]);
    emag = intex * intex + intey * intey + intez * intez + TINY;
    intbx = interpolate(bx, ijk, Stagger(0b001), Stagger(0b110),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intby = interpolate(by, ijk, Stagger(0b010), Stagger(0b110),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intbz = interpolate(bz, ijk, Stagger(0b100), Stagger(0b110),
                        dev_grid.dims[0], dev_grid.dims[1]);
    bmag = intbx * intbx + intby * intby + intbz * intbz + TINY;
    if (emag > bmag) {
      temp = sqrt(bmag / emag);
    } else {
      temp = 1.0;
    }
    ex[ijk] = temp * dex[ijk];

    // y:
    intex = interpolate(ex, ijk, Stagger(0b110), Stagger(0b101),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intey = interpolate(ey, ijk, Stagger(0b101), Stagger(0b101),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intez = interpolate(ez, ijk, Stagger(0b011), Stagger(0b101),
                        dev_grid.dims[0], dev_grid.dims[1]);
    emag = intex * intex + intey * intey + intez * intez + TINY;
    intbx = interpolate(bx, ijk, Stagger(0b001), Stagger(0b101),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intby = interpolate(by, ijk, Stagger(0b010), Stagger(0b101),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intbz = interpolate(bz, ijk, Stagger(0b100), Stagger(0b101),
                        dev_grid.dims[0], dev_grid.dims[1]);
    bmag = intbx * intbx + intby * intby + intbz * intbz + TINY;
    if (emag > bmag) {
      temp = sqrt(bmag / emag);
    } else {
      temp = 1.0;
    }
    ey[ijk] = temp * dey[ijk];

    // z:
    intex = interpolate(ex, ijk, Stagger(0b110), Stagger(0b011),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intey = interpolate(ey, ijk, Stagger(0b101), Stagger(0b011),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intez = interpolate(ez, ijk, Stagger(0b011), Stagger(0b011),
                        dev_grid.dims[0], dev_grid.dims[1]);
    emag = intex * intex + intey * intey + intez * intez + TINY;
    intbx = interpolate(bx, ijk, Stagger(0b001), Stagger(0b011),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intby = interpolate(by, ijk, Stagger(0b010), Stagger(0b011),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intbz = interpolate(bz, ijk, Stagger(0b100), Stagger(0b011),
                        dev_grid.dims[0], dev_grid.dims[1]);
    bmag = intbx * intbx + intby * intby + intbz * intbz + TINY;
    if (emag > bmag) {
      temp = sqrt(bmag / emag);
    } else {
      temp = 1.0;
    }
    ez[ijk] = temp * dez[ijk];
  }
}


HOST_DEVICE Scalar pmlsigma(Scalar x, Scalar xl, Scalar xh, Scalar pmlscale, Scalar sig0) {
  if (x > xh) return sig0 * pow((x - xh) / pmlscale, 3.0);
  else if (x < xl) return sig0 * pow((xl - x) / pmlscale, 3.0);
  else return 0.0;
}

__global__ void
kernel_absorbing_boundary_rsstv(const Scalar *enx, const Scalar *eny,
                         const Scalar *enz, const Scalar *bnx, const Scalar *bny,
                         const Scalar *bnz, Scalar *ex, Scalar *ey, Scalar *ez,
                         Scalar *bx, Scalar *by, Scalar *bz, int shift) {
  Scalar x, y, z;
  Scalar sigx, sigy, sigz, sig;
  size_t ijk;
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
        ex[ijk] = exp(-sig) * enx[ijk] + (1.0 - exp(-sig)) / sig * (ex[ijk] - enx[ijk]);
        ey[ijk] = exp(-sig) * eny[ijk] + (1.0 - exp(-sig)) / sig * (ey[ijk] - eny[ijk]);
        ez[ijk] = exp(-sig) * enz[ijk] + (1.0 - exp(-sig)) / sig * (ez[ijk] - enz[ijk]); 
        bx[ijk] = exp(-sig) * bnx[ijk] + (1.0 - exp(-sig)) / sig * (bx[ijk] - bnx[ijk]);
        by[ijk] = exp(-sig) * bny[ijk] + (1.0 - exp(-sig)) / sig * (by[ijk] - bny[ijk]);
        bz[ijk] = exp(-sig) * bnz[ijk] + (1.0 - exp(-sig)) / sig * (bz[ijk] - bnz[ijk]); 
      }
    }
  }
}

HOST_DEVICE Scalar shape(Scalar x, Scalar y, Scalar z, Scalar r0) {
  Scalar r = sqrt(x * x + y * y + z * z);
  return 0.5 * (1.0 - tanh(r - r0));
} 

__global__ void
kernel_disk_boundary_rsstv(Scalar *ex, Scalar *ey, Scalar *ez,
                         Scalar *bx, Scalar *by, Scalar *bz, int shift) {
  Scalar x, y, z, rd;
  Scalar rmax = (dev_grid.dims[0] / 2 - dev_params.pml[0] - 20) * dev_grid.delta[0];
  Scalar wscale = 4.0 * dev_grid.delta[0];
  Scalar ddr = fmin((dev_params.r1 + dev_params.r2) / 2.0 - dev_params.r1, 5.0 * dev_grid.delta[0]);
  Scalar scaleEpar = 0.5 * dev_grid.delta[0];
  Scalar omz, vx, vy;
  Scalar bzn, exn, eyn, s;
  Scalar cosph, sinph, gm, intbx, intby, intbz, intez;

  size_t ijk;
  int i = threadIdx.x + blockIdx.x * blockDim.x + dev_grid.guard[0] - shift;
  int j = threadIdx.y + blockIdx.y * blockDim.y + dev_grid.guard[1] - shift;
  int k = threadIdx.z + blockIdx.z * blockDim.z + dev_grid.guard[2] - shift;
  if (i < dev_grid.dims[0] - dev_grid.guard[0] + shift &&
      j < dev_grid.dims[1] - dev_grid.guard[1] + shift &&
      k < dev_grid.dims[2] - dev_grid.guard[2] + shift) {
    ijk = i + j * dev_grid.dims[0] +
          k * dev_grid.dims[0] * dev_grid.dims[1];
    z = dev_grid.pos(2, k, 1);
    if (std::abs(z) < dev_grid.delta[2] / 4.0) {

      // Set Ex
      x = dev_grid.pos(0, i, 0);
      y = dev_grid.pos(1, j, 1);
      rd = sqrt(x * x + y * y);

      if (rd > dev_params.r2 - ddr) {
        omz = dev_params.omegad0 * pow(dev_params.r2 / rd, 3.0 / 2.0)
              * shape(x / wscale, y / wscale, z / wscale, rmax / wscale);
        vy = omz * x;
        bzn = interpolate(bz, ijk, Stagger(0b100), Stagger(0b110),
                        dev_grid.dims[0], dev_grid.dims[1]);
        exn = - vy * bzn;
        s = shape (x / scaleEpar, y / scaleEpar, z / scaleEpar, dev_params.r2 / scaleEpar);
        ex[ijk] = ex[ijk] * s + exn * (1.0 - s);
      }
      else if (rd < dev_params.r1 + ddr) {
        cosph = x / rd;
        sinph = y / rd;
        omz = dev_params.omega0;
        vx = - omz * y;
        vy = omz * x;
        gm = 1.0 / sqrt(1.0 - (vx * vx + vy * vy));
        intbx = interpolate(bx, ijk, Stagger(0b001), Stagger(0b010),
                          dev_grid.dims[0], dev_grid.dims[1]);
        intby = by[ijk];
        intbz = interpolate(bz, ijk, Stagger(0b100), Stagger(0b110),
                          dev_grid.dims[0], dev_grid.dims[1]);
        intez = interpolate(ez, ijk, Stagger(0b011), Stagger(0b010),
                          dev_grid.dims[0], dev_grid.dims[1]);
        exn = - dev_params.eta * cosph * (cosph * intby - sinph * intbx) / gm - omz * x * intbz
              - dev_params.eta * gm * (sinph * (cosph * intbx + sinph * intby) - omz * y * intez);
        s = shape(x / scaleEpar, y / scaleEpar, z / scaleEpar, dev_params.r1 / scaleEpar);
        ex[ijk] = exn * s + ex[ijk] * (1.0 - s);
      }

      // Set Ey
      x = dev_grid.pos(0, i, 1);
      y = dev_grid.pos(1, j, 0);
      rd = sqrt(x * x + y * y);

      if (rd > dev_params.r2 - ddr) {
        omz = dev_params.omegad0 * pow(dev_params.r2 / rd, 3.0 / 2.0)
              * shape(x / wscale, y / wscale, z / wscale, rmax / wscale);
        vx = - omz * y;
        bzn = interpolate(bz, ijk, Stagger(0b100), Stagger(0b101),
                        dev_grid.dims[0], dev_grid.dims[1]);
        eyn = vx * bzn;
        s = shape (x / scaleEpar, y / scaleEpar, z / scaleEpar, dev_params.r2 / scaleEpar);
        ey[ijk] = ey[ijk] * s + eyn * (1.0 - s);
      }
      else if (rd < dev_params.r1 + ddr) {
        cosph = x / rd;
        sinph = y / rd;
        omz = dev_params.omega0;
        vx = - omz * y;
        vy = omz * x;
        gm = 1.0 / sqrt(1.0 - (vx * vx + vy * vy));
        intbx = bx[ijk];
        intby = interpolate(bx, ijk, Stagger(0b010), Stagger(0b001),
                          dev_grid.dims[0], dev_grid.dims[1]);
        intbz = interpolate(bz, ijk, Stagger(0b100), Stagger(0b101),
                          dev_grid.dims[0], dev_grid.dims[1]);
        intez = interpolate(ez, ijk, Stagger(0b011), Stagger(0b001),
                          dev_grid.dims[0], dev_grid.dims[1]);
        eyn = - dev_params.eta * sinph * (cosph * intby - sinph * intbx) / gm - omz * y * intbz
              - dev_params.eta * gm * (cosph * (cosph * intbx + sinph * intby) - omz * x * intez);
        s = shape(x / scaleEpar, y / scaleEpar, z / scaleEpar, dev_params.r1 / scaleEpar);
        ey[ijk] = eyn * s + ey[ijk] * (1.0 - s);
      }
    }
  }
}


__global__ void
kernel_emissivity_rsstv(const Scalar *ex, const Scalar *ey, const Scalar *ez, 
                        const Scalar *bx, const Scalar *by, const Scalar *bz, 
                        Scalar* em[], int shift) {
  Scalar intex, intey, intez, intbx, intby, intbz;
  Scalar P0, vx, vy, vz, beta, gm, x, y, z, r, th, cth, sth, mu;
  Scalar Bsq, Esq, edotb, B0sq, E00, B00, E0sq;
  size_t ijk;
  int i = threadIdx.x + blockIdx.x * blockDim.x + dev_grid.guard[0] - shift;
  int j = threadIdx.y + blockIdx.y * blockDim.y + dev_grid.guard[1] - shift;
  int k = threadIdx.z + blockIdx.z * blockDim.z + dev_grid.guard[2] - shift;
  if (i < dev_grid.dims[0] - dev_grid.guard[0] + shift &&
      j < dev_grid.dims[1] - dev_grid.guard[1] + shift &&
      k < dev_grid.dims[2] - dev_grid.guard[2] + shift) {
    ijk = i + j * dev_grid.dims[0] +
          k * dev_grid.dims[0] * dev_grid.dims[1];

    intex = interpolate(ex, ijk, Stagger(0b110), Stagger(0b111),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intey = interpolate(ey, ijk, Stagger(0b101), Stagger(0b111),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intez = interpolate(ez, ijk, Stagger(0b011), Stagger(0b111),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intbx = interpolate(bx, ijk, Stagger(0b001), Stagger(0b111),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intby = interpolate(by, ijk, Stagger(0b010), Stagger(0b111),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intbz = interpolate(bz, ijk, Stagger(0b100), Stagger(0b111),
                       dev_grid.dims[0], dev_grid.dims[1]);

    Bsq = intbx * intbx + intby * intby + intbz * intbz;
    Esq = intex * intex + intey * intey + intez * intez;
    edotb = intex * intbx + intey * intby + intez * intbz;
    B0sq = std::abs(0.5 * ((Bsq - Esq) + sqrt(square(Bsq - Esq) + 4.0 * square(edotb))));
    E00 = sqrt(std::abs(B0sq - Bsq + Esq));
    B00 = sgn (edotb) * sqrt(B0sq);
    E0sq = E00 * E00;
    P0 = sqrt(dev_params.sigsq) * E0sq;
    vx = 1.0 / (Bsq + E0sq + TINY) * (intey * intbz - intez * intby);
    vy = 1.0 / (Bsq + E0sq + TINY) * (intez * intbx - intex * intbz);
    vz = 1.0 / (Bsq + E0sq + TINY) * (intex * intby - intey * intbx);
    beta = sqrt(vx * vx + vy * vy + vz * vz);
    if (beta>1) beta = 1.0 - TINY;
    gm = 1.0 / sqrt(1.0 - beta * beta);
    for (int ith = 0; ith < 4 ; ++ith) {
      th = ith * M_PI /6.0;
      cth = cos(th);
      sth = sin(th);
      if (beta < TINY) mu = 0;
      else mu = (sth * vx + cth * vz) / beta;
      em[ith][ijk]=1.0/(pow(gm, 4) * pow(1.0 - beta * mu, 4)) * P0;
    }
  }
}

field_solver_resistive::field_solver_resistive(sim_data &mydata, sim_environment& env) : m_data(mydata), m_env(env) {
  En = vector_field<Scalar>(m_data.env.grid());
  dE = vector_field<Scalar>(m_data.env.grid());
  En.copy_stagger(m_data.E);
  dE.copy_stagger(m_data.E);
  En.initialize();
  dE.initialize();

  Bn = vector_field<Scalar>(m_data.env.grid());
  dB = vector_field<Scalar>(m_data.env.grid());
  Bn.copy_stagger(m_data.B);
  dB.copy_stagger(m_data.B);
  Bn.initialize();
  dB.initialize();

  rho = multi_array<Scalar>(m_data.env.grid().extent());
  rho.assign_dev(0.0);

  blockGroupSize = dim3((m_data.env.grid().reduced_dim(0) + m_env.params().shift_ghost * 2 + blockSize.x - 1) / blockSize.x,
                        (m_data.env.grid().reduced_dim(1) + m_env.params().shift_ghost * 2 + blockSize.y - 1) / blockSize.y,
                        (m_data.env.grid().reduced_dim(2) + m_env.params().shift_ghost * 2 + blockSize.z - 1) / blockSize.z);
  std::cout << blockSize.x << ", " << blockSize.y << ", " << blockSize.z << std::endl;
  std::cout << blockGroupSize.x << ", " << blockGroupSize.y << ", " << blockGroupSize.z << std::endl;
}

field_solver_resistive::~field_solver_resistive() {}

void
field_solver_resistive::evolve_fields() {
  RANGE_PUSH("Compute", CLR_GREEN);
  copy_fields();

  // substep #1:
  rk_push_noj();
  if (m_env.params().vacuum) rk_push_jvacuum();
  else if (m_env.params().resistive) rk_push_rjperp();
  else rk_push_ffjperp();
  rk_update(1.0, 0.0, 1.0);
  if (m_env.params().resistive && !m_env.params().vacuum) {
    for (int i = 0; i < m_env.params().subsamp; ++i)
    {
      disk_boundary();
      rk_update_rjparsub(1.0);
    }
  }
  if (!m_env.params().vacuum && !m_env.params().resistive) check_eGTb();
  if (!m_env.params().vacuum) disk_boundary();
  CudaSafeCall(cudaDeviceSynchronize());
  RANGE_POP;
  m_env.send_guard_cells(m_data);

  // substep #2:
  RANGE_PUSH("Compute", CLR_GREEN);
  rk_push_noj();
  if (m_env.params().vacuum) rk_push_jvacuum();
  else if (m_env.params().resistive) rk_push_rjperp();
  else rk_push_ffjperp();
  rk_update(0.75, 0.25, 0.25);
  if (m_env.params().resistive && !m_env.params().vacuum) {
    for (int i = 0; i < m_env.params().subsamp; ++i)
    {
      disk_boundary();
      rk_update_rjparsub(0.25);
    }
  }
  if (!m_env.params().vacuum && !m_env.params().resistive) check_eGTb();
  if (!m_env.params().vacuum) disk_boundary();
  CudaSafeCall(cudaDeviceSynchronize());
  RANGE_POP;
  m_env.send_guard_cells(m_data);

  // substep #3:
  RANGE_PUSH("Compute", CLR_GREEN);
  rk_push_noj();
  if (m_env.params().vacuum) rk_push_jvacuum();
  else if (m_env.params().resistive) rk_push_rjperp();
  else rk_push_ffjperp();
  rk_update(1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0);
  if (m_env.params().resistive && !m_env.params().vacuum) {
    for (int i = 0; i < m_env.params().subsamp; ++i)
    {
      disk_boundary();
      rk_update_rjparsub(2.0 / 3.0);
    }
  }
  if (!m_env.params().vacuum && !m_env.params().resistive) {
    clean_epar();
    check_eGTb();
  }
  if (!m_env.params().vacuum) disk_boundary();
  absorbing_boundary();
  CudaSafeCall(cudaDeviceSynchronize());
  RANGE_POP;

  m_env.send_guard_cells(m_data);
}

void
field_solver_resistive::copy_fields() {
  // `En = E, Bn = B`:
  En.copy_from(m_data.E);
  Bn.copy_from(m_data.B);
  dE.initialize();
  dB.initialize();
}

void
field_solver_resistive::rk_push_noj() {
  // `dE = curl B - curl B0 - j, dB = -curl E`
  // kernel_rk_push<<<g, blockSize>>>(
  kernel_rk_push_noj_rsstv<<<blockGroupSize, blockSize>>>(
      m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
      m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2),
      m_data.B0.dev_ptr(0), m_data.B0.dev_ptr(1), m_data.B0.dev_ptr(2),
      dE.dev_ptr(0), dE.dev_ptr(1), dE.dev_ptr(2), dB.dev_ptr(0),
      dB.dev_ptr(1), dB.dev_ptr(2), rho.dev_ptr(), m_env.params().shift_ghost);
  CudaCheckError();
}

// Original force-free algorithm, current part
void
field_solver_resistive::rk_push_ffjperp() {
  // `rho = div E`
  // kernel_compute_rho<<<gridSize, blockSize>>>(
  kernel_compute_rho_rsstv<<<blockGroupSize, blockSize>>>(
      m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
      rho.dev_ptr(), m_env.params().shift_ghost);
  CudaCheckError();
  // `dE = curl B - curl B0 - j, dB = -curl E`
  // kernel_rk_push<<<g, blockSize>>>(
  kernel_rk_push_ffjperp_rsstv<<<blockGroupSize, blockSize>>>(
      m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
      m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2),
      dE.dev_ptr(0), dE.dev_ptr(1), dE.dev_ptr(2), dB.dev_ptr(0),
      dB.dev_ptr(1), dB.dev_ptr(2), rho.dev_ptr(), m_env.params().shift_ghost);
  CudaCheckError();
}

// Vacuum push, with current only in the accretion disk
void
field_solver_resistive::rk_push_jvacuum() {
  kernel_rk_push_jvacuum_rsstv<<<blockGroupSize, blockSize>>>(
      dE.dev_ptr(0), dE.dev_ptr(1), dE.dev_ptr(2), m_env.params().shift_ghost);
  CudaCheckError();
}

// Resistive formalism, perpendicular current part
void
field_solver_resistive::rk_push_rjperp() {
  // `rho = div E`
  // kernel_compute_rho<<<gridSize, blockSize>>>(
  kernel_compute_rho_rsstv<<<blockGroupSize, blockSize>>>(
      m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
      rho.dev_ptr(), m_env.params().shift_ghost);
  CudaCheckError();
  // `dE = curl B - curl B0 - j, dB = -curl E`
  // kernel_rk_push<<<g, blockSize>>>(
  kernel_rk_push_rjperp_rsstv<<<blockGroupSize, blockSize>>>(
      m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
      m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2),
      dE.dev_ptr(0), dE.dev_ptr(1), dE.dev_ptr(2), dB.dev_ptr(0),
      dB.dev_ptr(1), dB.dev_ptr(2), rho.dev_ptr(), m_env.params().shift_ghost);
  CudaCheckError();
}

void
field_solver_resistive::rk_update(Scalar rk_c1, Scalar rk_c2, Scalar rk_c3) {
  // `E = c1 En + c2 E + c3 dE, B = c1 Bn + c2 B + c3 dB`
  // kernel_rk_update<<<dim3(8, 16, 16), dim3(64, 4, 4)>>>(
  kernel_rk_update_rsstv<<<blockGroupSize, blockSize>>>(
      m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
      m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2),
      En.dev_ptr(0), En.dev_ptr(1), En.dev_ptr(2), Bn.dev_ptr(0),
      Bn.dev_ptr(1), Bn.dev_ptr(2), dE.dev_ptr(0), dE.dev_ptr(1),
      dE.dev_ptr(2), dB.dev_ptr(0), dB.dev_ptr(1), dB.dev_ptr(2), rk_c1,
      rk_c2, rk_c3, m_env.params().shift_ghost);
  CudaCheckError();
}

void
field_solver_resistive::rk_update_rjparsub(Scalar rk_c3) {
  // first conpute rho
  kernel_compute_rho_rsstv<<<blockGroupSize, blockSize>>>(
      m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
      rho.dev_ptr(), m_env.params().shift_ghost);
  CudaCheckError();

  // Update; results stored in dE
  kernel_rk_update_rjparsub_rsstv<<<blockGroupSize, blockSize>>>(
      m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
      m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2),
      dE.dev_ptr(0), dE.dev_ptr(1),dE.dev_ptr(2), rho.dev_ptr(),
      rk_c3, m_env.params().shift_ghost);
  CudaCheckError();

  // Copy results back to E, B
  m_data.E.copy_from(dE);
}

void
field_solver_resistive::clean_epar() {
  // clean `E || B`
  // kernel_clean_epar<<<gridSize, blockSize>>>(
  kernel_clean_epar_rsstv<<<blockGroupSize, blockSize>>>(
      m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
      m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2),
      dE.dev_ptr(0), dE.dev_ptr(1), dE.dev_ptr(2), m_env.params().shift_ghost);
  CudaCheckError();
}

void
field_solver_resistive::check_eGTb() {
  // renormalizing `E > B`
  // kernel_check_eGTb<<<dim3(8, 16, 16), dim3(32, 4, 4)>>>(
  kernel_check_eGTb_rsstv<<<blockGroupSize, blockSize>>>(
      dE.dev_ptr(0), dE.dev_ptr(1), dE.dev_ptr(2), m_data.E.dev_ptr(0),
      m_data.E.dev_ptr(1), m_data.E.dev_ptr(2), m_data.B.dev_ptr(0),
      m_data.B.dev_ptr(1), m_data.B.dev_ptr(2), m_env.params().shift_ghost);
  CudaCheckError();
}

void
field_solver_resistive::absorbing_boundary() {
  kernel_absorbing_boundary_rsstv<<<blockGroupSize, blockSize>>>(
    En.dev_ptr(0), En.dev_ptr(1), En.dev_ptr(2), Bn.dev_ptr(0),
    Bn.dev_ptr(1), Bn.dev_ptr(2), m_data.E.dev_ptr(0), 
    m_data.E.dev_ptr(1), m_data.E.dev_ptr(2), m_data.B.dev_ptr(0), 
    m_data.B.dev_ptr(1), m_data.B.dev_ptr(2), m_env.params().shift_ghost);
  CudaCheckError();
}

void
field_solver_resistive::disk_boundary() {
  kernel_disk_boundary_rsstv<<<blockGroupSize, blockSize>>>(
    m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2), 
    m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2), m_env.params().shift_ghost);
  CudaCheckError();
}

void 
field_solver_resistive::light_curve(uint32_t step) {
  Scalar *em[4];
  em[0] = En.dev_ptr(0);
  em[1] = En.dev_ptr(1);
  em[2] = En.dev_ptr(2);
  em[3] = rho.dev_ptr();
  kernel_emissivity_rsstv<<<blockGroupSize, blockSize>>>(
    m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2), 
    m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2), 
    em, m_env.params().shift_ghost);
  CudaCheckError();
  En.sync_to_host();
  rho.sync_to_host();
  em[0] = En.host_ptr(0);
  em[1] = En.host_ptr(1);
  em[2] = En.host_ptr(2);
  em[3] = rho.host_ptr();

  int l0 = m_env.params().size[2] * sqrt(3.0) / 2.0 / m_env.params().dt;
  int len = int((m_env.params().max_steps + l0 * 2) / m_env.params().lc_interval + 2);
  // std::vector<Scalar> lc(len * 12, 0), lc0(len * 12, 0);
  if (lc.size() != len * 12) lc.resize(len * 12, 0);
  if (lc0.size() != len * 12) lc0.resize(len * 12, 0);
  for (int k = m_env.grid().guard[2]; k < m_env.grid().dims[2] - m_env.grid().guard[2]; ++k) {
    for (int j = m_env.grid().guard[1]; j < m_env.grid().dims[1] - m_env.grid().guard[1]; ++j) {
      for (int i = m_env.grid().guard[0]; i < m_env.grid().dims[0] - m_env.grid().guard[0]; ++i) {
        int ijk = i + j * m_env.grid().dims[0] +
          k * m_env.grid().dims[0] * m_env.grid().dims[1];
        Scalar x = m_env.grid().pos(0, i, 1);
        Scalar y = m_env.grid().pos(1, j, 1);
        Scalar z = m_env.grid().pos(2, k, 1);
        Scalar r = sqrt(x * x + y * y + z * z);
        for (int ith = 0; ith < 4; ++ith) {
          Scalar th = ith * M_PI / 6.0;
          Scalar cth = cos(th);
          Scalar sth = sin(th);
          Scalar cosd = (x * sth + y * cth) / r;
          Scalar dd = r * cosd;
          int il = int(floor(l0 - dd / m_env.params().dt) + 1 + step);
          for (int ih = 0; ih < 3; ++ih) {
            if (z > ih) lc[il + ih * len + ith * len * 3] += em[ith][ijk];
          } // ih
        } // ith
      } // i
    } // j
  } // k

  MPI_Reduce(lc.data(), lc0.data(), len * 12, m_env.scalar_type(), MPI_SUM, 0, m_env.world());

  if (m_env.rank() == 0) {
    std::stringstream ss;
    ss << std::setw(5) << std::setfill('0')
      << step / m_env.params().lc_interval;
    std::string num = ss.str();
    File file(std::string("./Data/lc") + num + std::string(".h5"), 
      File::ReadWrite | File::Create | File::Truncate);
    DataSet dataset = file.createDataSet<Scalar>("/lc",  DataSpace::From(lc0));
    dataset.write(lc0);
  }
}

}  // namespace Coffee
