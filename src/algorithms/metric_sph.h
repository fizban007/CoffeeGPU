#ifndef _METRIC_SPH_H_
#define _METRIC_SPH_H_

#include <cmath>

#include "cuda/cuda_control.h"
#include "data/grid.h"
#include "data/typedefs.h"
#include "utils/simd.h"
#include "utils/util_functions.h"
#include "vectormath_exp.h"
#include "vectormath_trig.h"

namespace Coffee {

#ifdef USE_SIMD
using simd::Vec_f_t;

inline Vec_f_t
pos_simd(const Grid& g, int i, int n, int stagger) {
  Vec_f_t result = simd::vec_inc * g.delta[i] + g.pos(i, n, stagger);
  return result;
}

inline Vec_f_t
pos_simd(const Grid& g, int i, int n, bool stagger) {
  return pos_simd(g, i, n, (int)stagger);
}
#endif

namespace SPH {

#ifdef USE_SIMD
inline Vec_f_t
get_x_simd(const Vec_f_t& r) {
  return log(r);
}

inline Vec_f_t
get_r_simd(const Vec_f_t& x, const Vec_f_t& y, const Vec_f_t& z) {
  return exp(x);
}

inline Vec_f_t
get_th_simd(const Vec_f_t& x, const Vec_f_t& y, const Vec_f_t& z) {
  return y;
}

inline Vec_f_t
get_gamma_d11_simd(const Vec_f_t& x, const Vec_f_t& y,
                   const Vec_f_t& z) {
  auto r = get_r_simd(x, y, z);
  return r * r;
}

inline Vec_f_t
get_gamma_d22_simd(const Vec_f_t& x, const Vec_f_t& y,
                   const Vec_f_t& z) {
  auto r = get_r_simd(x, y, z);
  return r * r;
}

inline Vec_f_t
get_gamma_d33_simd(const Vec_f_t& x, const Vec_f_t& y,
                   const Vec_f_t& z) {
  auto r = get_r_simd(x, y, z);
  auto th = get_th_simd(x, y, z);
  return square(r * sin(th));
}

inline Vec_f_t
get_gamma_simd(const Vec_f_t& x, const Vec_f_t& y, const Vec_f_t& z) {
  return get_gamma_d11_simd(x, y, z) * get_gamma_d22_simd(x, y, z) *
         get_gamma_d33_simd(x, y, z);
}

inline Vec_f_t
get_sqrt_gamma_simd(const Vec_f_t& x, const Vec_f_t& y,
                    const Vec_f_t& z) {
  return sqrt(get_gamma_simd(x, y, z));
}

#endif

HD_INLINE Scalar
get_x(Scalar r) {
  return log(r);
}

HD_INLINE Scalar
get_r(Scalar x, Scalar y, Scalar z) {
  return exp(x);
}

HD_INLINE Scalar
get_th(Scalar x, Scalar y, Scalar z) {
  return y;
}

HD_INLINE Scalar
get_gamma_d11(Scalar x, Scalar y, Scalar z) {
  Scalar r = get_r(x, y, z);
  return r * r;
}

HD_INLINE Scalar
get_gamma_d22(Scalar x, Scalar y, Scalar z) {
  Scalar r = get_r(x, y, z);
  return r * r;
}

HD_INLINE Scalar
get_gamma_d33(Scalar x, Scalar y, Scalar z) {
  Scalar r = get_r(x, y, z);
  Scalar th = get_th(x, y, z);
  return square(r * sin(th));
}

HD_INLINE Scalar
get_gamma(Scalar x, Scalar y, Scalar z) {
  return get_gamma_d11(x, y, z) * get_gamma_d22(x, y, z) *
         get_gamma_d33(x, y, z);
}

HD_INLINE Scalar
get_sqrt_gamma(Scalar x, Scalar y, Scalar z) {
  return std::sqrt(get_gamma(x, y, z));
}

}  // namespace SPH

}  // namespace Coffee

#endif  // _METRIC_SPH_H_
