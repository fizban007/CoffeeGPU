#ifndef _DAMPING_BOUNDARY_H_
#define _DAMPING_BOUNDARY_H_

#include "cuda/cuda_control.h"
#include "data/fields.h"
#include "data/grid.h"
#include "data/multi_array.h"
#include "data/typedefs.h"
#include "sim_params.h"
#include "utils/util_functions.h"

namespace Coffee {

HD_INLINE Scalar
pmlsigma(Scalar x, Scalar xl, Scalar xh, Scalar pmlscale, Scalar sig0) {
  if (x > xh)
    return sig0 * cube((x - xh) / pmlscale);
  else if (x < xl)
    return sig0 * cube((xl - x) / pmlscale);
  else
    return 0.0;
}

void damping_boundary(const vector_field<Scalar>& En,
                      const vector_field<Scalar>& Bn,
                      const vector_field<Scalar>& Bbg,
                      vector_field<Scalar>& E, vector_field<Scalar>& B,
                      multi_array<Scalar>& Pn, multi_array<Scalar>& P,
                      int shift, const Grid& grid,
                      const sim_params& params);

}  // namespace Coffee

#endif  // _DAMPING_BOUNDARY_H_
