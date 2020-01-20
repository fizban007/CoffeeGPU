#ifndef _PULSAR_IMPL_H_
#define _PULSAR_IMPL_H_

#include "algorithms/pulsar.h"
#include "data/typedefs.h"
#include "math.h"

namespace Coffee {

HOST_DEVICE Scalar
dipole_x(Scalar x, Scalar y, Scalar z, Scalar alpha, Scalar phase) {
  Scalar r = std::sqrt(x * x + y * y + z * z);
  if (std::abs(r) < TINY) r = TINY;
  Scalar mux = sin(alpha) * cos(phase);
  Scalar muy = sin(alpha) * sin(phase);
  Scalar muz = cos(alpha);
  Scalar xn = x / r;
  Scalar yn = y / r;
  Scalar zn = z / r;
  Scalar mun = mux * xn + muy * yn + muz * zn;
  return (3.0 * xn * mun - mux) / (cube(r) + TINY);
}

HOST_DEVICE Scalar
dipole_y(Scalar x, Scalar y, Scalar z, Scalar alpha, Scalar phase) {
  Scalar r = std::sqrt(x * x + y * y + z * z);
  if (std::abs(r) < TINY) r = TINY;
  Scalar mux = sin(alpha) * cos(phase);
  Scalar muy = sin(alpha) * sin(phase);
  Scalar muz = cos(alpha);
  Scalar xn = x / r;
  Scalar yn = y / r;
  Scalar zn = z / r;
  Scalar mun = mux * xn + muy * yn + muz * zn;
  return (3.0 * yn * mun - muy) / (cube(r) + TINY);
}

HOST_DEVICE Scalar
dipole_z(Scalar x, Scalar y, Scalar z, Scalar alpha, Scalar phase) {
  Scalar r = std::sqrt(x * x + y * y + z * z);
  if (std::abs(r) < TINY) r = TINY;
  Scalar mux = sin(alpha) * cos(phase);
  Scalar muy = sin(alpha) * sin(phase);
  Scalar muz = cos(alpha);
  Scalar xn = x / r;
  Scalar yn = y / r;
  Scalar zn = z / r;
  Scalar mun = mux * xn + muy * yn + muz * zn;
  return (3.0 * zn * mun - muz) / (cube(r) + TINY);
}

HOST_DEVICE Scalar
shape(Scalar r, Scalar r0, Scalar del) {
  return 0.5 * (1.0 - tanh((r - r0) / del));
}

HOST_DEVICE Scalar
dipole2(Scalar x, Scalar y, Scalar z, Scalar p1, Scalar p2, Scalar p3,
        Scalar phase, int n) {
  Scalar cosph = cos(phase);
  Scalar sinph = sin(phase);
  Scalar p1t = p1 * cosph - p2 * sinph;
  Scalar p2t = p1 * sinph + p2 * cosph;
  Scalar p3t = p3;
  Scalar r2 = x * x + y * y + z * z;
  if (n == 0)
    return (3.0 * x * (p1t * x + p2t * y + p3t * z) / (r2 + TINY) -
            p1t) /
           (std::sqrt(cube(r2)) + TINY);
  else if (n == 1)
    return (3.0 * y * (p1t * x + p2t * y + p3t * z) / (r2 + TINY) -
            p2t) /
           (std::sqrt(cube(r2)) + TINY);
  else if (n == 2)
    return (3.0 * z * (p1t * x + p2t * y + p3t * z) / (r2 + TINY) -
            p3t) /
           (std::sqrt(cube(r2)) + TINY);
  else
    return 0;
}

HOST_DEVICE Scalar
quadrupole(Scalar x, Scalar y, Scalar z, Scalar q11, Scalar q12,
           Scalar q13, Scalar q22, Scalar q23, Scalar q_offset_x,
           Scalar q_offset_y, Scalar q_offset_z, Scalar phase, int n) {
  Scalar cosph = cos(phase);
  Scalar sinph = sin(phase);
  Scalar q11t = q11 * square(cosph) - 2.0 * q12 * cosph * sinph +
                q22 * square(sinph);
  Scalar q12t = q12 * cos(2.0 * phase) + (q11 - q22) * cosph * sinph;
  Scalar q22t = q22 * square(cosph) + q11 * square(sinph) +
                2.0 * q12 * cosph * sinph;
  Scalar q13t = q13 * cosph - q23 * sinph;
  Scalar q23t = q23 * cosph + q13 * sinph;
  Scalar q33t = -q11 - q22;
  Scalar q_offset_x1 = q_offset_x * cosph - q_offset_y * sinph;
  Scalar q_offset_y1 = q_offset_x * sinph + q_offset_y * cosph;
  Scalar x1 = x - q_offset_x1;
  Scalar y1 = y - q_offset_y1;
  Scalar z1 = z - q_offset_z;
  Scalar r2 = x1 * x1 + y1 * y1 + z1 * z1;
  Scalar r7 = cube(r2) * std::sqrt(r2);
  Scalar xqx = q11t * x1 * x1 + 2.0 * q12t * x1 * y1 + q22t * y1 * y1 +
               2.0 * (q13t * x1 + q23t * y1) * z1 + q33t * z1 * z1;
  if (n == 0)
    return (-2.0 * (q11t * x1 + q12t * y1 + q13t * z1) * r2 +
            5.0 * x1 * xqx) /
           (r7 + TINY);
  else if (n == 1)
    return (-2.0 * (q12t * x1 + q22t * y1 + q23t * z1) * r2 +
            5.0 * y1 * xqx) /
           (r7 + TINY);
  else if (n == 2)
    return (-2.0 * (q13t * x1 + q23t * y1 + q33t * z1) * r2 +
            5.0 * z1 * xqx) /
           (r7 + TINY);
  else
    return 0;
}

HOST_DEVICE Scalar
quadru_dipole(Scalar x, Scalar y, Scalar z, Scalar p1, Scalar p2,
              Scalar p3, Scalar q11, Scalar q12, Scalar q13, Scalar q22,
              Scalar q23, Scalar q_offset_x, Scalar q_offset_y,
              Scalar q_offset_z, Scalar phase, int n) {
  return dipole2(x, y, z, p1, p2, p3, phase, n) +
         quadrupole(x, y, z, q11, q12, q13, q22, q23, q_offset_x,
                    q_offset_y, q_offset_z, phase, n);
}

}  // namespace Coffee

#endif  // _PULSAR_IMPL_H_
