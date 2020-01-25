#ifndef _FINITE_DIFF_H_
#define _FINITE_DIFF_H_

#include "cuda/cuda_control.h"
#include "data/grid.h"

namespace Coffee {

#define FFE_DISSIPATION_ORDER 6

// central derivative

HD_INLINE Scalar
diff1_4(const Scalar *f, int ijk, int s) {
  return (f[ijk - 2 * s] - 8.0 * f[ijk - s] + 8.0 * f[ijk + s] -
          f[ijk + 2 * s]) /
         12.0;
}

HD_INLINE Scalar
diff4_2(const Scalar *f, int ijk, int s) {
  return (f[ijk - 2 * s] - 4.0 * f[ijk - s] + 6.0 * f[ijk] -
          4.0 * f[ijk + s] + f[ijk + 2 * s]);
}

HD_INLINE Scalar
diff6_2(const Scalar *f, int ijk, int s) {
  return (f[ijk - 3 * s] - 6.0 * f[ijk - 2 * s] + 15.0 * f[ijk - s] -
          20.0 * f[ijk] + 15.0 * f[ijk + s] - 6.0 * f[ijk + 2 * s] +
          f[ijk + 3 * s]);
}

// forward one-sided derivative
HD_INLINE Scalar
diff1_4f(const Scalar *f, int ijk, int s) {
  return -25.0 / 12.0 * f[ijk] + 4.0 * f[ijk + s] -
         3.0 * f[ijk + 2 * s] + 4.0 / 3.0 * f[ijk + 3 * s] -
         1.0 / 4.0 * f[ijk + 4 * s];
}

HD_INLINE Scalar
diff4_2f(const Scalar *f, int ijk, int s) {
  return 3.0 * f[ijk] - 14.0 * f[ijk + s] + 26.0 * f[ijk + 2 * s] -
         24.0 * f[ijk + 3 * s] + 11.0 * f[ijk + 4 * s] -
         2.0 * f[ijk + 5 * s];
}

HD_INLINE Scalar
diff6_2f(const Scalar *f, int ijk, int s) {
  return 4.0 * f[ijk] - 27.0 * f[ijk + s] + 78.0 * f[ijk + 2 * s] -
         125.0 * f[ijk + 3 * s] + 120.0 * f[ijk + 4 * s] -
         69.0 * f[ijk + 5 * s] + 22.0 * f[ijk + 6 * s] -
         3.0 * f[ijk + 7 * s];
}

// backward one-sided derivative
HD_INLINE Scalar
diff1_4b(const Scalar *f, int ijk, int s) {
  return 25.0 / 12.0 * f[ijk] - 4.0 * f[ijk - s] +
         3.0 * f[ijk - 2 * s] - 4.0 / 3.0 * f[ijk - 3 * s] +
         1.0 / 4.0 * f[ijk - 4 * s];
}

HD_INLINE Scalar
diff4_2b(const Scalar *f, int ijk, int s) {
  return 3.0 * f[ijk] - 14.0 * f[ijk - s] + 26.0 * f[ijk - 2 * s] -
         24.0 * f[ijk - 3 * s] + 11.0 * f[ijk - 4 * s] -
         2.0 * f[ijk - 5 * s];
}

HD_INLINE Scalar
diff6_2b(const Scalar *f, int ijk, int s) {
  return 4.0 * f[ijk] - 27.0 * f[ijk - s] + 78.0 * f[ijk - 2 * s] -
         125.0 * f[ijk - 3 * s] + 120.0 * f[ijk - 4 * s] -
         69.0 * f[ijk - 5 * s] + 22.0 * f[ijk - 6 * s] -
         3.0 * f[ijk - 7 * s];
}

HD_INLINE Scalar
df1(const Scalar *f, int ijk, int s, Scalar inv_delta) {
  return diff1_4(f, ijk, s) * inv_delta;
}

HD_INLINE Scalar
KO(const Scalar *f, int ijk, const Grid &grid) {
  if (FFE_DISSIPATION_ORDER == 4)
    return diff4_2(f, ijk, 1) + diff4_2(f, ijk, grid.dims[0]) +
           diff4_2(f, ijk, grid.dims[0] * grid.dims[1]);
  else if (FFE_DISSIPATION_ORDER == 6)
    return diff6_2(f, ijk, 1) + diff6_2(f, ijk, grid.dims[0]) +
           diff6_2(f, ijk, grid.dims[0] * grid.dims[1]);
}

}  // namespace Coffee

#endif  // _FINITE_DIFF_H_
