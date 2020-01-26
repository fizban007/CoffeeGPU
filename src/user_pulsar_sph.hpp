#ifndef _USER_INIT_H_
#define _USER_INIT_H_

// Axisymmetric pulsar, using field_solver_EZ_spherical

Stagger st_e[3] = {Stagger(0b111), Stagger(0b111), Stagger(0b111)};
Stagger st_b[3] = {Stagger(0b111), Stagger(0b111), Stagger(0b111)};
data.E.set_stagger(st_e);
data.B.set_stagger(st_b);

data.B.initialize(0, [&](Scalar x, Scalar y, Scalar z) {
  // Put your initial condition for Bx here
  Scalar r = get_r(x, y, z);
  Scalar th = get_th(x, y, z);
  Scalar g11 = std::sqrt(get_gamma_d11(x, y, z));
  if (g11 < TINY) g11 = TINY;
  return env.params().b0 * dipole_sph_2d(r, th, 0) / g11;
});

data.B.initialize(1, [&](Scalar x, Scalar y, Scalar z) {
  // Put your initial condition for Bx here
  Scalar r = get_r(x, y, z);
  Scalar th = get_th(x, y, z);
  Scalar g22 = std::sqrt(get_gamma_d22(x, y, z));
  if (g22 < TINY) g22 = TINY;
  return env.params().b0 * dipole_sph_2d(r, th, 1) / g22;
});

data.B.initialize(2, [&](Scalar x, Scalar y, Scalar z) {
  // Put your initial condition for Bx here
  Scalar r = get_r(x, y, z);
  Scalar th = get_th(x, y, z);
  Scalar g33 = std::sqrt(get_gamma_d33(x, y, z));
  if (g33 < TINY) g33 = TINY;
  return env.params().b0 * dipole_sph_2d(r, th, 2) / g33;
});

data.E.initialize();
data.B0.initialize();
data.P.assign(0.0);
data.P.sync_to_device();

#endif  // _USER_INIT_H_
