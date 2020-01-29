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
Scalar xi0 = 0.2;
Scalar nx0 = ky * bz0 - kz * by0;
Scalar ny0 = kz * bx0 - kx * bz0;
Scalar nz0 = kx * by0 - ky * bx0;
Scalar norm = sqrt(nx0 * nx0 + ny0 * ny0 + nz0 * nz0);
nx0 = nx0 / norm;
ny0 = ny0 / norm;
nz0 = nz0 / norm;
Scalar enx = ny0 * bz0 - nz0 * by0;
Scalar eny = nz0 * bx0 - nx0 * bz0;
Scalar enz = nx0 * by0 - ny0 * bx0;

vector_field<Scalar> dA(env.grid());
dA.copy_stagger(data.E);

data.E.initialize(0, [&](Scalar x, Scalar y, Scalar z) {
  // Put your initial condition for Ex here
  Scalar phi = kx * x + ky * y + kz * z;
  return -w * xi0 * std::sin(phi) * enx / c;
});

data.E.initialize(1, [&](Scalar x, Scalar y, Scalar z) {
  // Put your initial condition for Ey here
  Scalar phi = kx * x + ky * y + kz * z;
  return -w * xi0 * std::sin(phi) * eny / c;
});

data.E.initialize(2, [&](Scalar x, Scalar y, Scalar z) {
  // Put your initial condition for Ez here
  Scalar phi = kx * x + ky * y + kz * z;
  return -w * xi0 * std::sin(phi) * enz / c;
});

dA.initialize(0, [&](Scalar x, Scalar y, Scalar z) {
  Scalar phi = kx * x + ky * y + kz * z;
  return xi0 * std::cos(phi) * enx;
});

dA.initialize(1, [&](Scalar x, Scalar y, Scalar z) {
  Scalar phi = kx * x + ky * y + kz * z;
  return xi0 * std::cos(phi) * eny;
});

dA.initialize(2, [&](Scalar x, Scalar y, Scalar z) {
  Scalar phi = kx * x + ky * y + kz * z;
  return xi0 * std::cos(phi) * enz;
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
// data.B.copy_from(data.B0);
data.B0.initialize();

#endif  // _USER_INIT_H_
