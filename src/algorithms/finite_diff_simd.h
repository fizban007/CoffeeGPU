#ifndef _FINITE_DIFF_SIMD_H_
#define _FINITE_DIFF_SIMD_H_

#include "cuda/cuda_control.h"
#include "data/grid.h"
#include "utils/simd.h"
#include "vectorclass.h"

namespace Coffee {

#define FFE_DISSIPATION_ORDER 6

#ifdef USE_SIMD
using namespace simd;

inline Vec_f_t diff1_4_simd(const Scalar *f, int ijk, int s) {
  Vec_f_t fm2;
  fm2.load(f + ijk - 2 * s);
  Vec_f_t fm1;
  fm1.load(f + ijk - s);
  Vec_f_t fp1;
  fp1.load(f + ijk + s);
  Vec_f_t fp2;
  fp2.load(f + ijk + 2 * s);
  return (fm2 - 8.0 * fm1 + 8.0 * fp1 - fp2) / 12.0;
}

inline Vec_f_t diff4_2_simd(const Scalar *f, int ijk, int s) {
  Vec_f_t fm2;
  fm2.load(f + ijk - 2 * s);
  Vec_f_t fm1;
  fm1.load(f + ijk - s);
  Vec_f_t f0;
  f0.load(f + ijk);
  Vec_f_t fp1;
  fp1.load(f + ijk + s);
  Vec_f_t fp2;
  fp2.load(f + ijk + 2 * s);
  return (fm2 - 4.0 * fm1 + 6.0 * f0 - 4.0 * fp1 + fp2);
}

inline Vec_f_t diff6_2_simd(const Scalar *f, int ijk, int s) {
  Vec_f_t fm3;
  fm3.load(f + ijk - 3 * s);
  Vec_f_t fm2;
  fm2.load(f + ijk - 2 * s);
  Vec_f_t fm1;
  fm1.load(f + ijk - s);
  Vec_f_t f0;
  f0.load(f + ijk);
  Vec_f_t fp1;
  fp1.load(f + ijk + s);
  Vec_f_t fp2;
  fp2.load(f + ijk + 2 * s);
  Vec_f_t fp3;
  fp3.load(f + ijk + 3 * s);
  return (fm3 - 6.0 * fm2 + 15.0 * fm1 - 20.0 * f0 + 15.0 * fp1 - 6.0 * fp2 +
          fp3);
}

inline Vec_f_t df1_simd(const Scalar *f, int ijk, int s, Scalar inv_delta) {
  return diff1_4_simd(f, ijk, s) * inv_delta;
}

inline Vec_f_t KO_simd(const Scalar *f, int ijk, const Grid &grid) {
  if (FFE_DISSIPATION_ORDER == 4)
    return diff4_2(f, ijk, 1) + diff4_2(f, ijk, grid.dims[0]) +
           diff4_2(f, ijk, grid.dims[0] * grid.dims[1]);
  else if (FFE_DISSIPATION_ORDER == 6)
    return diff6_2(f, ijk, 1) + diff6_2(f, ijk, grid.dims[0]) +
           diff6_2(f, ijk, grid.dims[0] * grid.dims[1]);
}

#endif

HD_INLINE Scalar
diff1_4(const Scalar *f, int ijk, int s) {
  return (f[ijk - 2 * s] - 8.0 * f[ijk - s] + 8.0 * f[ijk + s] -
          f[ijk + 2 * s]) /
         12.0;
}

HD_INLINE Scalar
df1(const Scalar *f, int ijk, int s, Scalar inv_delta) {
  return diff1_4(f, ijk, s) * inv_delta;
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

HD_INLINE Scalar
KO(const Scalar *f, int ijk, const Grid &grid) {
  if (FFE_DISSIPATION_ORDER == 4)
    return diff4_2(f, ijk, 1) + diff4_2(f, ijk, grid.dims[0]) +
           diff4_2(f, ijk, grid.dims[0] * grid.dims[1]);
  else if (FFE_DISSIPATION_ORDER == 6)
    return diff6_2(f, ijk, 1) + diff6_2(f, ijk, grid.dims[0]) +
           diff6_2(f, ijk, grid.dims[0] * grid.dims[1]);
}

} // namespace Coffee

#endif // _FINITE_DIFF_SIMD_H_
