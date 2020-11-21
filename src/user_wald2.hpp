#ifndef _USER_INIT_H_
#define _USER_INIT_H_

// Scalar a0 = 0.0;
Scalar a0 = env.params().a;

data.B.initialize(0, [&](Scalar x, Scalar y, Scalar z) { return 0.0; });

data.B.initialize(1, [&](Scalar x, Scalar y, Scalar z) { return 0.0; });

data.B.initialize(2, [&](Scalar x, Scalar y, Scalar z) {
  return 1.0 / get_sqrt_gamma(a0, x, y, z);
});

data.B.sync_to_device();

// data.E.initialize();

data.E.initialize(0, [&](Scalar x, Scalar y, Scalar z) {
  Scalar Bdz = get_gamma_d33(a0, x, y, z) / get_sqrt_gamma(a0, x, y, z);
  Scalar Bdy = get_gamma_d23(a0, x, y, z) / get_sqrt_gamma(a0, x, y, z);
  return 1.0 / get_alpha(a0, x, y, z) / get_sqrt_gamma(a0, x, y, z) *
         (get_beta_d2(a0, x, y, z) * Bdz - get_beta_d3(a0, x, y, z) * Bdy);
});

data.E.initialize(1, [&](Scalar x, Scalar y, Scalar z) {
  Scalar Bdz = get_gamma_d33(a0, x, y, z) / get_sqrt_gamma(a0, x, y, z);
  Scalar Bdx = get_gamma_d13(a0, x, y, z) / get_sqrt_gamma(a0, x, y, z);
  return 1.0 / get_alpha(a0, x, y, z) / get_sqrt_gamma(a0, x, y, z) *
         (get_beta_d3(a0, x, y, z) * Bdx - get_beta_d1(a0, x, y, z) * Bdz);
});

data.E.initialize(2, [&](Scalar x, Scalar y, Scalar z) {
  Scalar Bdx = get_gamma_d13(a0, x, y, z) / get_sqrt_gamma(a0, x, y, z);
  Scalar Bdy = get_gamma_d23(a0, x, y, z) / get_sqrt_gamma(a0, x, y, z);
  return 1.0 / get_alpha(a0, x, y, z) / get_sqrt_gamma(a0, x, y, z) *
         (get_beta_d1(a0, x, y, z) * Bdy - get_beta_d2(a0, x, y, z) * Bdx);
});

data.E.sync_to_device();

data.B0.initialize();
data.B0.sync_to_device();

#endif  // _USER_INIT_H_