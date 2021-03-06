#ifndef _METRIC_CKS_H_
#define _METRIC_CKS_H_

#include "cuda/cuda_control.h"
#include "data/typedefs.h"
#include "utils/util_functions.h"

namespace Coffee {

namespace CKS {

HOST_DEVICE Scalar get_R2(Scalar x, Scalar y, Scalar z);
HOST_DEVICE Scalar get_r(Scalar a, Scalar x, Scalar y, Scalar z);

HOST_DEVICE Scalar get_g();

HOST_DEVICE Scalar get_beta_d1(Scalar a, Scalar x, Scalar y, Scalar z);
HOST_DEVICE Scalar get_beta_d2(Scalar a, Scalar x, Scalar y, Scalar z);
HOST_DEVICE Scalar get_beta_d3(Scalar a, Scalar x, Scalar y, Scalar z);

HOST_DEVICE Scalar get_beta_u1(Scalar a, Scalar x, Scalar y, Scalar z);
HOST_DEVICE Scalar get_beta_u2(Scalar a, Scalar x, Scalar y, Scalar z);
HOST_DEVICE Scalar get_beta_u3(Scalar a, Scalar x, Scalar y, Scalar z);

HOST_DEVICE Scalar get_gamma(Scalar a, Scalar x, Scalar y, Scalar z);
HOST_DEVICE Scalar get_sqrt_gamma(Scalar a, Scalar x, Scalar y, Scalar z);

HOST_DEVICE Scalar get_alpha(Scalar a, Scalar x, Scalar y, Scalar z);

HOST_DEVICE Scalar get_gamma_d11(Scalar a, Scalar x, Scalar y, Scalar z);
HOST_DEVICE Scalar get_gamma_d12(Scalar a, Scalar x, Scalar y, Scalar z);
HOST_DEVICE Scalar get_gamma_d13(Scalar a, Scalar x, Scalar y, Scalar z);
HOST_DEVICE Scalar get_gamma_d22(Scalar a, Scalar x, Scalar y, Scalar z);
HOST_DEVICE Scalar get_gamma_d23(Scalar a, Scalar x, Scalar y, Scalar z);
HOST_DEVICE Scalar get_gamma_d33(Scalar a, Scalar x, Scalar y, Scalar z);

HOST_DEVICE Scalar get_gamma_u11(Scalar a, Scalar x, Scalar y, Scalar z);
HOST_DEVICE Scalar get_gamma_u12(Scalar a, Scalar x, Scalar y, Scalar z);
HOST_DEVICE Scalar get_gamma_u13(Scalar a, Scalar x, Scalar y, Scalar z);
HOST_DEVICE Scalar get_gamma_u22(Scalar a, Scalar x, Scalar y, Scalar z);
HOST_DEVICE Scalar get_gamma_u23(Scalar a, Scalar x, Scalar y, Scalar z);
HOST_DEVICE Scalar get_gamma_u33(Scalar a, Scalar x, Scalar y, Scalar z);

} //namespace CKS 

} // namespace Coffee

#endif // _METRIC_CKS_H_
