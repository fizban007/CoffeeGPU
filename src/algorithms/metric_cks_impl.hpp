#ifndef _METRIC_CKS_IMPL_H_
#define _METRIC_CKS_IMPL_H_

#include "algorithms/metric_cks.h"
#include "data/typedefs.h"
#include <cmath>

namespace Coffee {

namespace CKS {

HOST_DEVICE Scalar
get_R2(Scalar x, Scalar y, Scalar z) {
  return x * x + y * y + z * z;
}

HOST_DEVICE Scalar
get_r(Scalar a, Scalar x, Scalar y, Scalar z) {
  Scalar R2 = get_R2(x, y, z);
  return sqrt(
      (R2 - a * a +
       sqrt(max(square(R2 - a * a) + 4.0f * square(a * z), TINY))) /
      2.0f);
}

HOST_DEVICE Scalar
get_g() {
  return -1.0;
}

HOST_DEVICE Scalar
get_beta_d1(Scalar a, Scalar x, Scalar y, Scalar z) {
  Scalar r = get_r(a, x, y, z);
  return 2.0f * r * r * r * (r * x + a * y) / max(a * a + r * r, TINY) /
         max(r * r * r * r + a * a * z * z, TINY);
}

HOST_DEVICE Scalar
get_beta_d2(Scalar a, Scalar x, Scalar y, Scalar z) {
  Scalar r = get_r(a, x, y, z);
  return 2.0f * r * r * r * (-a * x + r * y) / max(a * a + r * r, TINY) /
         max(r * r * r * r + a * a * z * z, TINY);
}

HOST_DEVICE Scalar
get_beta_d3(Scalar a, Scalar x, Scalar y, Scalar z) {
  Scalar r = get_r(a, x, y, z);
  return 2.0f * r * r * z / max(r * r * r * r + a * a * z * z, TINY);
}

HOST_DEVICE Scalar
get_beta_u1(Scalar a, Scalar x, Scalar y, Scalar z) {
  Scalar r = get_r(a, x, y, z);
  return 2.0f * r * r * r * (r * x + a * y) / max(a * a + r * r, TINY) /
         max(2.0f * r * r * r + r * r * r * r + a * a * z * z, TINY);
}

HOST_DEVICE Scalar
get_beta_u2(Scalar a, Scalar x, Scalar y, Scalar z) {
  Scalar r = get_r(a, x, y, z);
  return 2 * r * r * r * (-a * x + r * y) / max(a * a + r * r, TINY) /
         max(2.0f * r * r * r + r * r * r * r + a * a * z * z, TINY);
}

HOST_DEVICE Scalar
get_beta_u3(Scalar a, Scalar x, Scalar y, Scalar z) {
  Scalar r = get_r(a, x, y, z);
  return 2.0f * r * r * z /
         max(2.0f * r * r * r + r * r * r * r + a * a * z * z, TINY);
}

HOST_DEVICE Scalar
get_gamma(Scalar a, Scalar x, Scalar y, Scalar z) {
  Scalar r = get_r(a, x, y, z);
  return 1.0f + 2.0f * r * r * r / max(r * r * r * r + a * a * z * z, TINY);
}

HOST_DEVICE Scalar
get_sqrt_gamma(Scalar a, Scalar x, Scalar y, Scalar z) {
  return sqrt(get_gamma(a, x, y, z));
}

HOST_DEVICE Scalar
get_alpha(Scalar a, Scalar x, Scalar y, Scalar z) {
  Scalar r = get_r(a, x, y, z);
  return sqrt(
      1.0 /
      (1.0f + 2.0f * r * r * r / max(r * r * r * r + a * a * z * z, TINY)));
}

HOST_DEVICE Scalar
get_gamma_d11(Scalar a, Scalar x, Scalar y, Scalar z) {
  Scalar r = get_r(a, x, y, z);
  return 1.0 + 2.0 * r * r * r * square(r * x + a * y) /
                   max(square(a * a + r * r), TINY) /
                   max(r * r * r * r + a * a * z * z, TINY);
}

HOST_DEVICE Scalar
get_gamma_d12(Scalar a, Scalar x, Scalar y, Scalar z) {
  Scalar r = get_r(a, x, y, z);
  return 2.0 * r * r * r * (r * x + a * y) * (-a * x + r * y) /
         max(square(a * a + r * r), TINY) /
         max(r * r * r * r + a * a * z * z, TINY);
}

HOST_DEVICE Scalar
get_gamma_d13(Scalar a, Scalar x, Scalar y, Scalar z) {
  Scalar r = get_r(a, x, y, z);
  return 2.0 * r * r * (r * x + a * y) * z / max(a * a + r * r, TINY) /
         max(r * r * r * r + a * a * z * z, TINY);
}

HOST_DEVICE Scalar
get_gamma_d22(Scalar a, Scalar x, Scalar y, Scalar z) {
  Scalar r = get_r(a, x, y, z);
  return 1.0 + 2.0 * r * r * r * square(a * x - r * y) /
                   max(square(a * a + r * r), TINY) /
                   max(r * r * r * r + a * a * z * z, TINY);
}

HOST_DEVICE Scalar
get_gamma_d23(Scalar a, Scalar x, Scalar y, Scalar z) {
  Scalar r = get_r(a, x, y, z);
  return 2.0 * r * r * (-a * x + r * y) * z / max(a * a + r * r, TINY) /
         max(r * r * r * r + a * a * z * z, TINY);
}

HOST_DEVICE Scalar
get_gamma_d33(Scalar a, Scalar x, Scalar y, Scalar z) {
  Scalar r = get_r(a, x, y, z);
  return 1.0 + 2.0 * r * z * z / max(r * r * r * r + a * a * z * z, TINY);
}

HOST_DEVICE Scalar
get_gamma_u11(Scalar a, Scalar x, Scalar y, Scalar z) {
  Scalar r = get_r(a, x, y, z);
  return 1.0 -
         2.0 * r * r * r * square(r * x + a * y) /
             max(square(a * a + r * r), TINY) /
             max(2.0f * r * r * r + r * r * r * r + a * a * z * z, TINY);
}

HOST_DEVICE Scalar
get_gamma_u12(Scalar a, Scalar x, Scalar y, Scalar z) {
  Scalar r = get_r(a, x, y, z);
  return -2.0 * r * r * r * (r * x + a * y) * (-a * x + r * y) /
         max(square(a * a + r * r), TINY) /
         max(2.0f * r * r * r + r * r * r * r + a * a * z * z, TINY);
}

HOST_DEVICE Scalar
get_gamma_u13(Scalar a, Scalar x, Scalar y, Scalar z) {
  Scalar r = get_r(a, x, y, z);
  return -2.0 * r * r * (r * x + a * y) * z / max(a * a + r * r, TINY) /
         max(2.0f * r * r * r + r * r * r * r + a * a * z * z, TINY);
}

HOST_DEVICE Scalar
get_gamma_u22(Scalar a, Scalar x, Scalar y, Scalar z) {
  Scalar r = get_r(a, x, y, z);
  return 1.0 -
         2.0 * r * r * r * square(a * x - r * y) /
             max(square(a * a + r * r), TINY) /
             max(2.0f * r * r * r + r * r * r * r + a * a * z * z, TINY);
}

HOST_DEVICE Scalar
get_gamma_u23(Scalar a, Scalar x, Scalar y, Scalar z) {
  Scalar r = get_r(a, x, y, z);
  return 2.0 * r * r * (a * x - r * y) * z / max(a * a + r * r, TINY) /
         max(2.0f * r * r * r + r * r * r * r + a * a * z * z, TINY);
}

HOST_DEVICE Scalar
get_gamma_u33(Scalar a, Scalar x, Scalar y, Scalar z) {
  Scalar r = get_r(a, x, y, z);
  return 1.0 -
         2.0 * r * z * z /
             max(2.0f * r * r * r + r * r * r * r + a * a * z * z, TINY);
}

} // namespace CKS 

} // namespace Coffee

#endif // _METRIC_CKS_IMPL_H_
