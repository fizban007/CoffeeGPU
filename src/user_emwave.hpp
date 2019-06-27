#ifndef _USER_INIT_H_
#define _USER_INIT_H_

Scalar kx, ky, ex_norm, ey_norm, exy_norm;
kx = env.params().kn[0] * 2.0 * M_PI / env.params().size[0];
ky = env.params().kn[1] * 2.0 * M_PI / env.params().size[1];
if (ky != 0) {
  ex_norm = 1.0;
  ey_norm = -kx / ky;
} else {
  ey_norm = 1.0;
  ex_norm = -ky / kx;
}
exy_norm = std::sqrt(ex_norm * ex_norm + ey_norm * ey_norm);
ex_norm = ex_norm / exy_norm;
ey_norm = ey_norm / exy_norm;

data.E.initialize(0, [&](Scalar x, Scalar y, Scalar z) {
  // Put your initial condition for Ex here
  return ex_norm * std::sin(kx * x + ky * y);
});

data.E.initialize(1, [&](Scalar x, Scalar y, Scalar z) {
  // Put your initial condition for Ey here
  return ey_norm * std::sin(kx * x + ky * y);
});

data.E.initialize(2, [&](Scalar x, Scalar y, Scalar z) {
  // Put your initial condition for Ez here
  return 0.0;
});

data.B.initialize(0, [&](Scalar x, Scalar y, Scalar z) {
  // Put your initial condition for Bx here
  return 0.0;
});

data.B.initialize(1, [&](Scalar x, Scalar y, Scalar z) {
  // Put your initial condition for By here
  return 0.0;
});

data.B.initialize(2, [&](Scalar x, Scalar y, Scalar z) {
  // Put your initial condition for Bz here
  return std::sin(kx * x + ky * y);
});

// data.B.copy_from(data.B0);
data.B0.initialize();

#endif  // _USER_INIT_H_
