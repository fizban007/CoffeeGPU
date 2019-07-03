#ifndef _USER_INIT_H_
#define _USER_INIT_H_

// Scalar getMonopole(Scalar x, Scalar y, Scalar z,
//                    Scalar xc, Scalar yc, Scalar zc,
//                    int pol, int component) {
//
//   if (component == 0) {
//     return pol * rx / (r * r * r);
//   } else if (component == 1) {
//     return pol * ry / (r * r * r);
//   } else if (component == 2) {
//     return pol * rz / (r * r * r);
//   } else {
//     return -1.0;
//   }
// }

Scalar xc = env.params().size[0] * 0.5;
Scalar yc1 = env.params().size[1] * 0.5 - env.params().a0 * env.params().L0;
Scalar yc2 = env.params().size[1] * 0.5 + env.params().a0 * env.params().L0;
Scalar zc = - env.params().h0 * env.params().L0;

data.E.initialize(0, [](Scalar x, Scalar y, Scalar z) {
  // Put your initial condition for Ex here
  return 0.0;
});

data.E.initialize(1, [](Scalar x, Scalar y, Scalar z) {
  // Put your initial condition for Ey here
  return 0.0;
});

data.E.initialize(2, [](Scalar x, Scalar y, Scalar z) {
  // Put your initial condition for Ez here
  return 0.0;
});

data.B0.initialize(0, [](Scalar x, Scalar y, Scalar z) {
  // Put your initial condition for Bx here
  Scalar rx = (x - xc);
  Scalar ry1 = (y - yc1);
  Scalar ry2 = (y - yc2);
  Scalar rz = (z - zc);
  Scalar r1 = (rx * rx + ry1 * ry1 + rz * rz);
  Scalar r2 = (rx * rx + ry2 * ry2 + rz * rz);

  return rx / (r1 * r1 * r1) - rx / (r2 * r2 * r2);
});

data.B0.initialize(1, [](Scalar x, Scalar y, Scalar z) {
  // Put your initial condition for By here
  Scalar rx = (x - xc);
  Scalar ry1 = (y - yc1);
  Scalar ry2 = (y - yc2);
  Scalar rz = (z - zc);
  Scalar r1 = (rx * rx + ry1 * ry1 + rz * rz);
  Scalar r2 = (rx * rx + ry2 * ry2 + rz * rz);

  return ry1 / (r1 * r1 * r1) - ry2 / (r2 * r2 * r2);
});

data.B0.initialize(2, [](Scalar x, Scalar y, Scalar z) {
  // Put your initial condition for Bz here
  Scalar rx = (x - xc);
  Scalar ry1 = (y - yc1);
  Scalar ry2 = (y - yc2);
  Scalar rz = (z - zc);
  Scalar r1 = (rx * rx + ry1 * ry1 + rz * rz);
  Scalar r2 = (rx * rx + ry2 * ry2 + rz * rz);

  return rz / (r1 * r1 * r1) - rz / (r2 * r2 * r2);
});

data.B.copy_from(data.B0);

#endif  // _USER_INIT_H_
