#ifndef _USER_INIT_H_
#define _USER_INIT_H_

// Axisymmetric pulsar, using field_solver_EZ_cylindrical

Stagger st_e[3] = {Stagger(0b111), Stagger(0b111), Stagger(0b111)};
Stagger st_b[3] = {Stagger(0b111), Stagger(0b111), Stagger(0b111)};
data.E.set_stagger(st_e);
data.B.set_stagger(st_b);

data.B.initialize(0, [&](Scalar x, Scalar y, Scalar z) {
  // Put your initial condition for Bx here
  return env.params().b0 * cube(env.params().radius) *
         dipole_x(x, y, z, env.params().alpha, 0);
});

data.B.initialize(1, [&](Scalar x, Scalar y, Scalar z) {
  // Put your initial condition for Bx here
  return env.params().b0 * cube(env.params().radius) *
         dipole_y(x, y, z, env.params().alpha, 0);
});

data.B.initialize(2, [&](Scalar x, Scalar y, Scalar z) {
  // Put your initial condition for Bx here
  return env.params().b0 * cube(env.params().radius) *
         dipole_z(x, y, z, env.params().alpha, 0);
});


data.E.initialize();
data.B0.initialize();

#endif  // _USER_INIT_H_
