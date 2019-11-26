#ifndef _USER_INIT_H_
#define _USER_INIT_H_

Scalar c = 1;
Scalar bx0 = 1.5;
Scalar by0 = 2.8;
Scalar bz0 = 3.2;
Scalar kx = 5.0 * 2.0 * M_PI / env.params().size[0];
Scalar ky = 8.0 * 2.0 * M_PI / env.params().size[1];
Scalar kz = 2.0 * 2.0 * M_PI / env.params().size[2];
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
Scalar bnx = ky * enz - kz * eny;
Scalar bny = kz * enx - kx * enz;
Scalar bnz = kx * eny - ky * enx;

Stagger st_e[3] = {Stagger(0b111), Stagger(0b111), Stagger(0b111)};
Stagger st_b[3] = {Stagger(0b111), Stagger(0b111), Stagger(0b111)};
data.E.set_stagger(st_e);
data.B.set_stagger(st_b);

data.E.initialize(0, [&](Scalar x, Scalar y, Scalar z) {
  // Put your initial condition for Ex here
  Scalar phi = kx * x + ky * y + kz * z;
  return - w * xi0 * std::sin(phi) * enx / c;
});

data.E.initialize(1, [&](Scalar x, Scalar y, Scalar z) {
  // Put your initial condition for Ey here
  Scalar phi = kx * x + ky * y + kz * z;
  return - w * xi0 * std::sin(phi) * eny / c;
});

data.E.initialize(2, [&](Scalar x, Scalar y, Scalar z) {
  // Put your initial condition for Ez here
  Scalar phi = kx * x + ky * y + kz * z;
  return - w * xi0 * std::sin(phi) * enz / c;
});

data.B.initialize(0, [&](Scalar x, Scalar y, Scalar z) {
  // Put your initial condition for Ex here
  Scalar phi = kx * x + ky * y + kz * z;
  return - xi0 * std::sin(phi) * bnx;
});

data.B.initialize(1, [&](Scalar x, Scalar y, Scalar z) {
  // Put your initial condition for Ex here
  Scalar phi = kx * x + ky * y + kz * z;
  return - xi0 * std::sin(phi) * bny;
});

data.B.initialize(2, [&](Scalar x, Scalar y, Scalar z) {
  // Put your initial condition for Ex here
  Scalar phi = kx * x + ky * y + kz * z;
  return - xi0 * std::sin(phi) * bnz;
});


data.B0.initialize();

#endif  // _USER_INIT_H_
