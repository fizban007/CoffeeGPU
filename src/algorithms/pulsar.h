#include "data/typedefs.h"
#include "cuda/cuda_control.h"

#define DELTA 1e-7

namespace Coffee {

template <typename T>
HD_INLINE T square(T x) { return x * x; }

template <typename T>
HD_INLINE T cube(T x) { return x * x * x; }

HOST_DEVICE Scalar dipole_x(Scalar x, Scalar y, Scalar z, Scalar alpha, Scalar phase); 

HOST_DEVICE Scalar dipole_y(Scalar x, Scalar y, Scalar z, Scalar alpha, Scalar phase); 

HOST_DEVICE Scalar dipole_z(Scalar x, Scalar y, Scalar z, Scalar alpha, Scalar phase); 

HOST_DEVICE Scalar shape(Scalar r, Scalar r0, Scalar del); 

HOST_DEVICE Scalar
dipole2(Scalar x, Scalar y, Scalar z, Scalar p1, Scalar p2, Scalar p3,
         Scalar phase, int n);

HOST_DEVICE Scalar
quadrupole(Scalar x, Scalar y, Scalar z, Scalar q11, Scalar q12,
           Scalar q13, Scalar q22, Scalar q23, Scalar q_offset_x,
           Scalar q_offset_y, Scalar q_offset_z, Scalar phase, int n);
}
