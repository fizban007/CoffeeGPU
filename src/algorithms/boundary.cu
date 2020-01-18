#include "boundary.h"
#include "pulsar.h"
#include "cuda/constant_mem.h"

namespace Coffee {

HOST_DEVICE Scalar
pmlsigma(Scalar x, Scalar xl, Scalar xh, Scalar pmlscale, Scalar sig0) {
  if (x > xh)
    return sig0 * cube((x - xh) / pmlscale);
  else if (x < xl)
    return sig0 * cube((xl - x) / pmlscale);
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
    // if (x > xh || y < yl || y > yh) {
      sigx = pmlsigma(x, xl, xh, dev_params.pmllen * dev_grid.delta[0],
                      dev_params.sigpml);
      sigy = pmlsigma(y, yl, yh, dev_params.pmllen * dev_grid.delta[0],
                      dev_params.sigpml);
      sigz = pmlsigma(z, zl, zh, dev_params.pmllen * dev_grid.delta[0],
                      dev_params.sigpml);
      sig = sigx + sigy + sigz;
      // sig = sigx + sigy;
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

__global__ void
kernel_boundary_absorbing1_thread(const Scalar *enx, const Scalar *eny,
                                 const Scalar *enz, const Scalar *bnx,
                                 const Scalar *bny, const Scalar *bnz,
                                 Scalar *ex, Scalar *ey, Scalar *ez,
                                 Scalar *bx, Scalar *by, Scalar *bz,
                                 Scalar *Pn, Scalar *P, int shift) {
  Scalar x, y, z;
  Scalar sigx = 0.0, sigy = 0.0, sigz = 0.0, sig = 0.0;
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
    // if (x > xh || y < yl || y > yh) {
      sigx = pmlsigma(x, xl, xh, dev_params.pmllen * dev_grid.delta[0],
                      dev_params.sigpml);
      sigy = pmlsigma(y, yl, yh, dev_params.pmllen * dev_grid.delta[0],
                      dev_params.sigpml);
      sigz = pmlsigma(z, zl, zh, dev_params.pmllen * dev_grid.delta[0],
                      dev_params.sigpml);
      sig = sigx + sigy + sigz;
      // sig = sigx + sigy;
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
        P[ijk] = exp(-sig) * Pn[ijk] +
                  (1.0 - exp(-sig)) / sig * (P[ijk] - Pn[ijk]);
      }
    }
  }
}
}