#ifndef _USER_INIT_H_
#define _USER_INIT_H_

Scalar c = 1;
// Scalar bx0 = 1.5;
// Scalar by0 = 2.8;
// Scalar bz0 = 3.2;
// Scalar kx = 5.0 * 2.0 * M_PI / env.params().size[0];
// Scalar ky = 8.0 * 2.0 * M_PI / env.params().size[1];
// Scalar kz = 2.0 * 2.0 * M_PI / env.params().size[2];
Scalar bx0 = 1.0;
Scalar by0 = 0.0;
Scalar bz0 = 0.0;
Scalar kx = 5.0 * 2.0 * M_PI / env.params().size[0];
Scalar ky = 0.0;
Scalar kz = 1.0 * 2.0 * M_PI / env.params().size[2];
Scalar bn = sqrt(bx0 * bx0 + by0 * by0 + bz0 * bz0);
Scalar w = (kx * bx0 + ky * by0 + kz * bz0) / bn * c;
Scalar xi0 = 0.1;
Scalar nx0 = ky * bz0 - kz * by0;
Scalar ny0 = kz * bx0 - kx * bz0;
Scalar nz0 = kx * by0 - ky * bx0;
Scalar enx = ny0 * bz0 - nz0 * by0;
Scalar eny = nz0 * bx0 - nx0 * bz0;
Scalar enz = nx0 * by0 - ny0 * bx0;
Scalar norm = sqrt(enx * enx + eny * eny + enz * enz);
enx = enx / norm;
eny = eny / norm;
enz = enz / norm;
Scalar bnx = ky * enz - kz * eny;
Scalar bny = kz * enx - kx * enz;
Scalar bnz = kx * eny - ky * enx;

#ifdef EZ
Stagger st_e[3] = {Stagger(0b111), Stagger(0b111), Stagger(0b111)};
Stagger st_b[3] = {Stagger(0b111), Stagger(0b111), Stagger(0b111)};
data.E.set_stagger(st_e);
data.B.set_stagger(st_b);
#endif

data.E.initialize(0, [&](Scalar x, Scalar y, Scalar z) {
  // Put your initial condition for Ex here
  Scalar phi = kx * x + ky * y + kz * z;
  return - xi0 * std::sin(phi) * enx / c;
});

data.E.initialize(1, [&](Scalar x, Scalar y, Scalar z) {
  // Put your initial condition for Ey here
  Scalar phi = kx * x + ky * y + kz * z;
  return - xi0 * std::sin(phi) * eny / c;
});

data.E.initialize(2, [&](Scalar x, Scalar y, Scalar z) {
  // Put your initial condition for Ez here
  Scalar phi = kx * x + ky * y + kz * z;
  return - xi0 * std::sin(phi) * enz / c;
});

#ifdef EZ

data.B.initialize(0, [&](Scalar x, Scalar y, Scalar z) {
  // Put your initial condition for Ex here
  Scalar phi = kx * x + ky * y + kz * z;
  return bx0 - xi0 * std::sin(phi) * bnx / w;
});

data.B.initialize(1, [&](Scalar x, Scalar y, Scalar z) {
  // Put your initial condition for Ex here
  Scalar phi = kx * x + ky * y + kz * z;
  return by0 - xi0 * std::sin(phi) * bny / w;
});

data.B.initialize(2, [&](Scalar x, Scalar y, Scalar z) {
  // Put your initial condition for Ex here
  Scalar phi = kx * x + ky * y + kz * z;
  return bz0 - xi0 * std::sin(phi) * bnz / w;
});

#else
vector_field<Scalar> dA(env.grid());
dA.copy_stagger(data.E);

dA.initialize(0, [&](Scalar x, Scalar y, Scalar z) {
  Scalar phi = kx * x + ky * y + kz * z;
  return xi0 * std::cos(phi) * enx / w;
});

dA.initialize(1, [&](Scalar x, Scalar y, Scalar z) {
  Scalar phi = kx * x + ky * y + kz * z;
  return xi0 * std::cos(phi) * eny / w;
});

dA.initialize(2, [&](Scalar x, Scalar y, Scalar z) {
  Scalar phi = kx * x + ky * y + kz * z;
  return xi0 * std::cos(phi) * enz / w;
});

for (int k = 0; k < env.grid().dims[2] - 1; ++k) {
  for (int j = 0; j < env.grid().dims[1] - 1; ++j) {
    for (int i = 0; i < env.grid().dims[0] - 1; ++i) {
      data.B(0, i, j, k) = bx0 + dA(2, i, j + 1, k) - dA(2, i, j, k) -
                           dA(1, i, j, k + 1) + dA(1, i, j, k);
      data.B(1, i, j, k) = by0 + dA(0, i, j, k + 1) - dA(0, i, j, k) -
                           dA(2, i + 1, j, k) + dA(2, i, j, k);
      data.B(2, i, j, k) = bz0 + dA(1, i + 1, j, k) - dA(1, i, j, k) -
                           dA(0, i, j + 1, k) + dA(0, i, j, k);
    }
  }
}

data.B.sync_to_device();
#endif

data.B0.initialize();

#endif  // _USER_INIT_H_
