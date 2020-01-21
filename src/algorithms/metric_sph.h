#ifndef _METRIC_SPH_H_
#define _METRIC_SPH_H_

#include "cuda/cuda_control.h"
#include "data/typedefs.h"
#include "utils/util_functions.h"

namespace Coffee {

namespace SPH {

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

}

}

#endif // _METRIC_SPH_H_