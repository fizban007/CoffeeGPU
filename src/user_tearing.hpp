#ifndef _USER_INIT_H_
#define _USER_INIT_H_

Scalar c = 1;
Scalar B0 = 1;
Scalar a = 0.1;
Scalar k = M_PI;
Scalar amp = 1.0e-4;

#ifdef EZ
Stagger st_e[3] = {Stagger(0b111), Stagger(0b111), Stagger(0b111)};
Stagger st_b[3] = {Stagger(0b111), Stagger(0b111), Stagger(0b111)};
data.E.set_stagger(st_e);
data.B.set_stagger(st_b);
#endif

data.E.initialize();

data.P.assign(0.0);
data.P.sync_to_device();

#ifdef EZ

data.B.initialize(0, [&](Scalar x, Scalar y, Scalar z) {
  // Put your initial condition for Bx here
  return B0 * tanh(z / a) +
         amp / (a * k) * B0 * sin(k * x) * tanh(z / a) / cosh(z / a);
});

data.B.initialize(1, [&](Scalar x, Scalar y, Scalar z) {
  // Put your initial condition for By here
  return B0 / cosh(z / a);
});

data.B.initialize(2, [&](Scalar x, Scalar y, Scalar z) {
  // Put your initial condition for Bz here
  return amp * B0 * cos(k * x) / cosh(z / a);
});

data.B0.initialize(0, [&](Scalar x, Scalar y, Scalar z) {
  // Put your initial condition for Bx here
  return B0 * tanh(z / a);
});

data.B0.initialize(1, [&](Scalar x, Scalar y, Scalar z) {
  // Put your initial condition for By here
  return B0 / cosh(z / a);
});

data.B0.initialize(2, [&](Scalar x, Scalar y, Scalar z) {
  // Put your initial condition for Bz here
  return 0.0;
});

#else
vector_field<Scalar> dA(env.grid());
dA.copy_stagger(data.E);

dA.initialize(0, [&](Scalar x, Scalar y, Scalar z) {
  return 2.0 * a * atan(tanh(z / 2.0 / a));
});

dA.initialize(1, [&](Scalar x, Scalar y, Scalar z) {
  return -a * log(cosh(z / a)) +
         amp / k * B0 * sin(k * x) / cosh(z / a);
});

dA.initialize(2, [&](Scalar x, Scalar y, Scalar z) { return 0.0; });

for (int k = 0; k < env.grid().dims[2] - 1; ++k) {
  for (int j = 0; j < env.grid().dims[1] - 1; ++j) {
    for (int i = 0; i < env.grid().dims[0] - 1; ++i) {
      data.B(0, i, j, k) = (dA(2, i, j + 1, k) - dA(2, i, j, k)) *
                               env.grid().inv_delta[1] -
                           (dA(1, i, j, k + 1) - dA(1, i, j, k)) *
                               env.grid().inv_delta[2];
      data.B(1, i, j, k) = (dA(0, i, j, k + 1) - dA(0, i, j, k)) *
                               env.grid().inv_delta[2] -
                           (dA(2, i + 1, j, k) - dA(2, i, j, k)) *
                               env.grid().inv_delta[0];
      data.B(2, i, j, k) = (dA(1, i + 1, j, k) - dA(1, i, j, k)) *
                               env.grid().inv_delta[0] -
                           (dA(0, i, j + 1, k) - dA(0, i, j, k)) *
                               env.grid().inv_delta[1];
    }
  }
}

data.B0.initialize();

#endif

data.B.sync_to_device();
data.B0.sync_to_device();

#endif  // _USER_INIT_H_
