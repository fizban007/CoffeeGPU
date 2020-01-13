#ifndef _BOUNDARY_H_
#define _BOUNDARY_H_

#include "data/typedefs.h"
#include "cuda/cuda_control.h"

namespace Coffee {

HOST_DEVICE Scalar
pmlsigma(Scalar x, Scalar xl, Scalar xh, Scalar pmlscale, Scalar sig0);

__global__ void
kernel_boundary_absorbing_thread(const Scalar *enx, const Scalar *eny,
                                 const Scalar *enz, const Scalar *bnx,
                                 const Scalar *bny, const Scalar *bnz,
                                 Scalar *ex, Scalar *ey, Scalar *ez,
                                 Scalar *bx, Scalar *by, Scalar *bz,
                                 int shift);
_global__ void
kernel_boundary_absorbing1_thread(const Scalar *enx, const Scalar *eny,
                                 const Scalar *enz, const Scalar *bnx,
                                 const Scalar *bny, const Scalar *bnz,
                                 Scalar *ex, Scalar *ey, Scalar *ez,
                                 Scalar *bx, Scalar *by, Scalar *bz,
                                 Scalar *Pn, Scalar *P, int shift);

}

#endif  // _BOUNDARY_H_
