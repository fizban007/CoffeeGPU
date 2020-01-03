#ifndef _USER_INIT_H_
#define _USER_INIT_H_

// 3d pulsar, using field_solver


for (int i = 0; i < 3; ++i) {
  data.B.initialize(i, [&](Scalar x, Scalar y, Scalar z) {
    // Put your initial condition for Bx here

    return env.params().b0 *
           (dipole2(x, y, z, env.params().p1, env.params().p2,
                    env.params().p3, 0, i) +
            quadrupole(x, y, z, env.params().q11, env.params().q12,
                       env.params().q13, env.params().q22,
                       env.params().q23, env.params().q_offset_x,
                       env.params().q_offset_y, env.params().q_offset_z,
                       0, i));
    // return env.params().b0 *
    //        dipole2(x, y, z, env.params().p1, env.params().p2,
    //                 env.params().p3, 0, i);
  });
}

data.E.initialize();
data.B0.initialize();

#endif  // _USER_INIT_H_
