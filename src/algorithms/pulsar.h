#include "data/typedefs.h"
#include "cuda/cuda_control.h"

#define DELTA 1e-10

namespace Coffee {

template <typename T>
HD_INLINE T square(T x) { return x * x; }

template <typename T>
HD_INLINE T cube(T x) { return x * x * x; }

HOST_DEVICE Scalar dipole_x(Scalar x, Scalar y, Scalar z, Scalar alpha, Scalar phase); 

HOST_DEVICE Scalar dipole_y(Scalar x, Scalar y, Scalar z, Scalar alpha, Scalar phase); 

HOST_DEVICE Scalar dipole_z(Scalar x, Scalar y, Scalar z, Scalar alpha, Scalar phase); 

HOST_DEVICE Scalar shape(Scalar r, Scalar r0, Scalar del); 

}
