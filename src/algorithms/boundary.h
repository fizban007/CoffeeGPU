#ifndef _BOUNDARY_H_
#define _BOUNDARY_H_

#include "cuda/cuda_control.h"
#include "data/typedefs.h"

namespace Coffee {

HOST_DEVICE Scalar pmlsigma(Scalar x, Scalar xl, Scalar xh,
                            Scalar pmlscale, Scalar sig0);

__global__ void kernel_boundary_absorbing_thread(
    const Scalar *enx, const Scalar *eny, const Scalar *enz,
    const Scalar *bnx, const Scalar *bny, const Scalar *bnz, Scalar *ex,
    Scalar *ey, Scalar *ez, Scalar *bx, Scalar *by, Scalar *bz,
    int shift);

__global__ void kernel_boundary_absorbing_EZ_thread(
    const Scalar *enx, const Scalar *eny, const Scalar *enz,
    const Scalar *bnx, const Scalar *bny, const Scalar *bnz,
    const Scalar *bgx, const Scalar *bgy, const Scalar *bgz, Scalar *ex,
    Scalar *ey, Scalar *ez, Scalar *bx, Scalar *by, Scalar *bz,
    Scalar *Pn, Scalar *P, int shift);

__global__ void kernel_outgoing_z(Scalar *Dx, Scalar *Dy, Scalar *Dz,
                                  Scalar *Bx, Scalar *By, Scalar *Bz,
                                  Scalar *P, int shift);

}  // namespace Coffee

#endif  // _BOUNDARY_H_
