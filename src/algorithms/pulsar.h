#ifndef _PULSAR_H_
#define _PULSAR_H_

#include "cuda/cuda_control.h"
#include "data/typedefs.h"
#include "utils/util_functions.h"

namespace Coffee {

HOST_DEVICE Scalar dipole_x(Scalar x, Scalar y, Scalar z, Scalar alpha,
                            Scalar phase);

HOST_DEVICE Scalar dipole_y(Scalar x, Scalar y, Scalar z, Scalar alpha,
                            Scalar phase);

HOST_DEVICE Scalar dipole_z(Scalar x, Scalar y, Scalar z, Scalar alpha,
                            Scalar phase);

HOST_DEVICE Scalar shape(Scalar r, Scalar r0, Scalar del);

HOST_DEVICE Scalar dipole2(Scalar x, Scalar y, Scalar z, Scalar p1,
                           Scalar p2, Scalar p3, Scalar phase, int n);

HOST_DEVICE Scalar quadrupole(Scalar x, Scalar y, Scalar z, Scalar q11,
                              Scalar q12, Scalar q13, Scalar q22,
                              Scalar q23, Scalar q_offset_x,
                              Scalar q_offset_y, Scalar q_offset_z,
                              Scalar phase, int n);

HOST_DEVICE Scalar quadru_dipole(Scalar x, Scalar y, Scalar z,
                                 Scalar p1, Scalar p2, Scalar p3,
                                 Scalar q11, Scalar q12, Scalar q13,
                                 Scalar q22, Scalar q23,
                                 Scalar q_offset_x, Scalar q_offset_y,
                                 Scalar q_offset_z, Scalar phase,
                                 int n);

HOST_DEVICE Scalar dipole_sph_2d(Scalar r, Scalar th, int n);

}  // namespace Coffee

#endif  // _PULSAR_H_

