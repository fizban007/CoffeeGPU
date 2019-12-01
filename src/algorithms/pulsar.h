#include "data/typedefs.h"

#define DELTA 1e-10

namespace Coffee {

template <typename T>
HD_INLINE T square(T x) { return x * x; }

template <typename T>
HD_INLINE T cube(T x) { return x * x * x; }

HOST_DEVICE Scalar dipole_x(Scalar x, Scalar y, Scalar z, Scalar alpha, Scalar phase) {
  Scalar r = std::sqrt(x * x + y * y + z * z);
  if (std::abs(r) < DELTA) r = DELTA;
  Scalar mux = sin(alpha) * cos (phase);
  Scalar muy = sin(alpha) * sin(phase);
  Scalar muz = cos(alpha);
  Scalar xn = x / r;
  Scalar yn = y / r;
  Scalar zn = z / r;
  Scalar mun = mux * xn + muy * yn + muz *zn;
  return (3.0 * xn * mun - mux) / cube(r);
}

HOST_DEVICE Scalar dipole_y(Scalar x, Scalar y, Scalar z, Scalar alpha, Scalar phase) {
  Scalar r = std::sqrt(x * x + y * y + z * z);
  if (std::abs(r) < DELTA) r = DELTA;
  Scalar mux = sin(alpha) * cos (phase);
  Scalar muy = sin(alpha) * sin(phase);
  Scalar muz = cos(alpha);
  Scalar xn = x / r;
  Scalar yn = y / r;
  Scalar zn = z / r;
  Scalar mun = mux * xn + muy * yn + muz *zn;
  return (3.0 * yn * mun - muy) / cube(r);
}

HOST_DEVICE Scalar dipole_z(Scalar x, Scalar y, Scalar z, Scalar alpha, Scalar phase) {
  Scalar r = std::sqrt(x * x + y * y + z * z);
  if (std::abs(r) < DELTA) r = DELTA;
  Scalar mux = sin(alpha) * cos (phase);
  Scalar muy = sin(alpha) * sin(phase);
  Scalar muz = cos(alpha);
  Scalar xn = x / r;
  Scalar yn = y / r;
  Scalar zn = z / r;
  Scalar mun = mux * xn + muy * yn + muz *zn;
  return (3.0 * zn * mun - muz) / cube(r);
}

HD_INLINE Scalar shape(Scalar r, Scalar r0, Scalar del) {
	return 0.5 * (1.0 - tanh((r - r0) / del));
}

}