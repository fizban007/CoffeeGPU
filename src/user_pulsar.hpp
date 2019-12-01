#ifndef _USER_INIT_H_
#define _USER_INIT_H_
#include "pulsar.h"

// Axisymmetric pulsar, using field_solver_EZ_cylindrical

Stagger st_e[3] = {Stagger(0b111), Stagger(0b111), Stagger(0b111)};
Stagger st_b[3] = {Stagger(0b111), Stagger(0b111), Stagger(0b111)};
data.E.set_stagger(st_e);
data.B.set_stagger(st_b);

data.B.initialize(0, [&](Scalar R, Scalar z, Scalar phi) {
  // Put your initial condition for Bx here
  return env.params().b0 * cube(env.params().radius) *
         dipole_x(R, 0, z, 0, 0);
});

data.B.initialize(1, [&](Scalar R, Scalar z, Scalar phi) {
  // Put your initial condition for Bx here
  return env.params().b0 * cube(env.params().radius) *
         dipole_z(R, 0, z, 0, 0);
});

data.B.initialize(2, [&](Scalar R, Scalar z, Scalar phi) {
  // Put your initial condition for Bx here
  return 0.0;
});

data.E.initialize();
data.B0.initialize();

#endif  // _USER_INIT_H_
