#ifndef _USER_INIT_H_
#define _USER_INIT_H_

// 3D pulsar, using field_solver_EZ_cylindrical

Stagger st_e[3] = {Stagger(0b111), Stagger(0b111), Stagger(0b111)};
Stagger st_b[3] = {Stagger(0b111), Stagger(0b111), Stagger(0b111)};
data.E.set_stagger(st_e);
data.B.set_stagger(st_b);
data.B0.set_stagger(st_b);

for (int i = 0; i < 3; ++i) {
  data.B.initialize(i, [&](Scalar x, Scalar y, Scalar z) {
    // Put your initial condition for Bx here
    // return env.params().b0 * cube(env.params().radius) *
    //        dipole_x(x, y, z, env.params().alpha, 0);
    return env.params().b0 *
           quadru_dipole(
               x, y, z, env.params().p1, env.params().p2,
               env.params().p3, env.params().q11, env.params().q12,
               env.params().q13, env.params().q22, env.params().q23,
               env.params().q_offset_x, env.params().q_offset_y,
               env.params().q_offset_z, 0, i);
  });
}

data.E.initialize();
data.B0.initialize();
data.P.initialize();

#endif  // _USER_INIT_H_
