#ifndef _USER_INIT_H_
#define _USER_INIT_H_


vector_field<Scalar>& A = solver.Dn;
multi_array<Scalar>& A0 = solver.rho;
Scalar a0 = env.params().a;
Scalar x, y, z;
Scalar intBx, intBy, intBz, intEdx, intEdy, intEdz;
Scalar Eux, Euy, Euz, Bdx, Bdy, Bdz;
size_t ijk; 

A.initialize(0, [&](Scalar x, Scalar y, Scalar z) {
  Scalar r = get_r(a0, x, y, z);
  return (a0 * z * z * (r * x - a0 * y) - r * r * r * (a0 * x + r * y))
      / (r * r + a0 * a0) / (r * r * r * r + a0 * a0 * z * z) / 2.0
      * (r * r + a0 * a0 - 2.0 * a0 * a0 * r * (r * r + z * z)
        / (r * r * r * r + a0 * a0 * z * z));
});

A.initialize(1, [&](Scalar x, Scalar y, Scalar z) {
  Scalar r = get_r(a0, x, y, z);
  return (a0 * z * z * (a0 * x + r * y) + r * r * r * (r * x - a0 * y))
      / (r * r + a0 * a0) / (r * r * r * r + a0 * a0 * z * z) / 2.0
      * (r * r + a0 * a0 - 2.0 * a0 * a0 * r * (r * r + z * z)
        / (r * r * r * r + a0 * a0 * z * z));
});

A.initialize(2, [&](Scalar x, Scalar y, Scalar z) {
  Scalar r = get_r(a0, x, y, z);
  return - a0 * z / r * (r * r - z * z)
      / (r * r * r * r + a0 * a0 * z * z) / 2.0
      * (r * r + a0 * a0 - 2.0 * a0 * a0 * r * (r * r + z * z)
        / (r * r * r * r + a0 * a0 * z * z));
});

for (int k = 0; k < env.grid().dims[2]; ++k) {
  for (int j = 0; j < env.grid().dims[1]; ++j) {
    for (int i = 0; i < env.grid().dims[0]; ++i) {
      x = env.grid().pos(0, i, 1);
      y = env.grid().pos(1, j, 1);
      z = env.grid().pos(2, k, 1);
      Scalar r = get_r(a0, x, y, z);
      A0(i, j, k) = a0 * r * (r * r + z * z)
                    / (r * r * r * r + a0 * a0 * z * z); 
    }
  }
}

for (int k = 0; k < env.grid().dims[2] - 1; ++k) {
  for (int j = 0; j < env.grid().dims[1] - 1; ++j) {
    for (int i = 0; i < env.grid().dims[0] - 1; ++i) {
      x = env.grid().pos(0, i, 1);
      y = env.grid().pos(1, j, 1);
      z = env.grid().pos(2, k, 1);

      data.B(0, i, j, k) = ((A(2, i, j + 1, k) - A(2, i, j, k)) 
                            * env.grid().inv_delta[1]
                            - (A(1, i, j, k + 1) - A(1, i, j, k)) 
                            * env.grid().inv_delta[2])
                            / get_sqrt_gamma(a0, x, y + env.grid().delta[1] / 2.0, 
                              z + env.grid().delta[2] / 2.0);
      data.B(1, i, j, k) = ((A(0, i, j, k + 1) - A(0, i, j, k)) 
                            * env.grid().inv_delta[2]
                            - (A(2, i + 1, j, k) - A(2, i, j, k)) 
                            * env.grid().inv_delta[0])
                            / get_sqrt_gamma(a0, x + env.grid().delta[0] / 2.0, 
                              y, z + env.grid().delta[2] / 2.0);
      data.B(2, i, j, k) = ((A(1, i + 1, j, k) - A(1, i, j, k)) 
                            * env.grid().inv_delta[0] 
                            - (A(0, i, j + 1, k) - A(0, i, j, k)) 
                            * env.grid().inv_delta[1])
                            / get_sqrt_gamma(a0, x + env.grid().delta[0] / 2.0, 
                              y + env.grid().delta[1] / 2.0, z);

      solver.Ed(0, i, j, k) = (A0(i + 1, j, k) - A0(i, j, k))
                            * env.grid().inv_delta[0];
      solver.Ed(1, i, j, k) = (A0(i, j + 1, k) - A0(i, j, k))
                            * env.grid().inv_delta[1];
      solver.Ed(2, i, j, k) = (A0(i, j, k + 1) - A0(i, j, k))
                            * env.grid().inv_delta[2];
    }
  }
}

for (int k = 1; k < env.grid().dims[2] - 1; ++k) {
  for (int j = 1; j < env.grid().dims[1] - 1; ++j) {
    for (int i = 1; i < env.grid().dims[0] - 1; ++i) {
      ijk = i + j * env.grid().dims[0] +
                k * env.grid().dims[0] * env.grid().dims[1];
      x = env.grid().pos(0, i, 0);
      y = env.grid().pos(1, j, 1);
      z = env.grid().pos(2, k, 1);
      intEdx = solver.Ed(0, i, j, k);
      intEdy = interpolate(solver.Ed.host_ptr(1), ijk, Stagger(0b101), Stagger(0b110),
                            env.grid().dims[0], env.grid().dims[1]);
      intEdz = interpolate(solver.Ed.host_ptr(2), ijk, Stagger(0b011), Stagger(0b110),
                            env.grid().dims[0], env.grid().dims[1]);
      intBx = interpolate(data.B.host_ptr(0), ijk, Stagger(0b001), Stagger(0b110),
                            env.grid().dims[0], env.grid().dims[1]);
      intBy = interpolate(data.B.host_ptr(1), ijk, Stagger(0b010), Stagger(0b110),
                            env.grid().dims[0], env.grid().dims[1]);
      intBz = interpolate(data.B.host_ptr(2), ijk, Stagger(0b100), Stagger(0b110),
                            env.grid().dims[0], env.grid().dims[1]);
      Eux = get_gamma_u11(a0, x, y, z) * intEdx + get_gamma_u12(a0, x, y, z) * intEdy
            + get_gamma_u13(a0, x, y, z) * intEdz;
      Bdy = get_gamma_d12(a0, x, y, z) * intBx + get_gamma_d22(a0, x, y, z) * intBy
            + get_gamma_d23(a0, x, y, z) * intBz;
      Bdz = get_gamma_d13(a0, x, y, z) * intBx + get_gamma_d23(a0, x, y, z) * intBy
            + get_gamma_d33(a0, x, y, z) * intBz;
      data.E(0, i, j, k) = (Eux - (get_beta_d2(a0, x, y, z) * Bdz - get_beta_d3(a0, x, y, z) * Bdy) 
        / get_sqrt_gamma(a0, x, y, z)) / get_alpha(a0, x, y, z);

      x = env.grid().pos(0, i, 1);
      y = env.grid().pos(1, j, 0);
      z = env.grid().pos(2, k, 1);
      intEdx = interpolate(solver.Ed.host_ptr(0), ijk, Stagger(0b110), Stagger(0b101),
                            env.grid().dims[0], env.grid().dims[1]);
      intEdy = solver.Ed(1, i, j, k);
      intEdz = interpolate(solver.Ed.host_ptr(2), ijk, Stagger(0b011), Stagger(0b101),
                            env.grid().dims[0], env.grid().dims[1]);
      intBx = interpolate(data.B.host_ptr(0), ijk, Stagger(0b001), Stagger(0b101),
                            env.grid().dims[0], env.grid().dims[1]);
      intBy = interpolate(data.B.host_ptr(1), ijk, Stagger(0b010), Stagger(0b101),
                            env.grid().dims[0], env.grid().dims[1]);
      intBz = interpolate(data.B.host_ptr(2), ijk, Stagger(0b100), Stagger(0b101),
                            env.grid().dims[0], env.grid().dims[1]);
      Euy = get_gamma_u12(a0, x, y, z) * intEdx + get_gamma_u22(a0, x, y, z) * intEdy
            + get_gamma_u23(a0, x, y, z) * intEdz;
      Bdx = get_gamma_d11(a0, x, y, z) * intBx + get_gamma_d12(a0, x, y, z) * intBy
            + get_gamma_d13(a0, x, y, z) * intBz;
      Bdz = get_gamma_d13(a0, x, y, z) * intBx + get_gamma_d23(a0, x, y, z) * intBy
            + get_gamma_d33(a0, x, y, z) * intBz;
      data.E(1, i, j, k) = (Euy - (get_beta_d3(a0, x, y, z) * Bdx - get_beta_d1(a0, x, y, z) * Bdz) 
        / get_sqrt_gamma(a0, x, y, z)) / get_alpha(a0, x, y, z);

      x = env.grid().pos(0, i, 1);
      y = env.grid().pos(1, j, 1);
      z = env.grid().pos(2, k, 0);
      intEdx = interpolate(solver.Ed.host_ptr(0), ijk, Stagger(0b110), Stagger(0b011),
                            env.grid().dims[0], env.grid().dims[1]);
      intEdy = interpolate(solver.Ed.host_ptr(1), ijk, Stagger(0b101), Stagger(0b011),
                            env.grid().dims[0], env.grid().dims[1]);
      intEdz = solver.Ed(2, i, j, k);
      intBx = interpolate(data.B.host_ptr(0), ijk, Stagger(0b001), Stagger(0b011),
                            env.grid().dims[0], env.grid().dims[1]);
      intBy = interpolate(data.B.host_ptr(1), ijk, Stagger(0b010), Stagger(0b011),
                            env.grid().dims[0], env.grid().dims[1]);
      intBz = interpolate(data.B.host_ptr(2), ijk, Stagger(0b100), Stagger(0b011),
                            env.grid().dims[0], env.grid().dims[1]);
      Euz = get_gamma_u13(a0, x, y, z) * intEdx + get_gamma_u23(a0, x, y, z) * intEdy
            + get_gamma_u33(a0, x, y, z) * intEdz;
      Bdx = get_gamma_d11(a0, x, y, z) * intBx + get_gamma_d12(a0, x, y, z) * intBy
            + get_gamma_d13(a0, x, y, z) * intBz;
      Bdy = get_gamma_d12(a0, x, y, z) * intBx + get_gamma_d22(a0, x, y, z) * intBy
            + get_gamma_d23(a0, x, y, z) * intBz;
      data.E(2, i, j, k) = (Euz - (get_beta_d1(a0, x, y, z) * Bdy - get_beta_d2(a0, x, y, z) * Bdx) 
        / get_sqrt_gamma(a0, x, y, z)) / get_alpha(a0, x, y, z);
    }
  }
}


data.B.sync_to_device();

// data.E.initialize();
data.E.sync_to_device();

data.B0.initialize();
data.B0.sync_to_device();

#endif  // _USER_INIT_H_
