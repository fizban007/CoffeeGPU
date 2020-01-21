#ifndef _METRIC_SPH_H_
#define _METRIC_SPH_H_

#include "cuda/cuda_control.h"
#include "data/typedefs.h"
#include "utils/util_functions.h"

namespace Coffee {

namespace SPH {

HD_INLINE Scalar get_r(Scalar x, Scalar y, Scalar z);
HD_INLINE Scalar get_th(Scalar x, Scalar y, Scalar z);
HD_INLINE Scalar get_gamma_d11(Scalar x, Scalar y, Scalar z);
HD_INLINE Scalar get_gamma_d22(Scalar x, Scalar y, Scalar z);
HD_INLINE Scalar get_gamma_d33(Scalar x, Scalar y, Scalar z);
HD_INLINE Scalar get_gamma(Scalar x, Scalar y, Scalar z);
HD_INLINE Scalar get_sqrt_gamma(Scalar x, Scalar y, Scalar z);

}

}

#endif // _METRIC_SPH_H_