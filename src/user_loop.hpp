#ifndef _USER_INIT_H_
#define _USER_INIT_H_

Scalar getMonopole(Scalar x, Scalar y, Scalar z,
                   Scalar xc, Scalar yc, Scalar zc,
                   int pol, int component) {
  Scalar rx = (x - xc);
  Scalar ry = (y - yc);
  Scalar rz = (z - zc);
  Scalar r = (rx * rx + ry * ry + rz * rz);
  if (component == 0) {
    return pol * rx / (r * r * r);
  } else if (component == 1) {
    return pol * ry / (r * r * r);
  } else if (component == 2) {
    return pol * rz / (r * r * r);
  } else {
    return -1.0;
  }
}

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
  Scalar xc = env.params().size[0] * 0.5;
  Scalar yc1 = env.params().size[1] * 0.5 - env.params().a0 * env.params().L0;
  Scalar yc2 = env.params().size[1] * 0.5 + env.params().a0 * env.params().L0;
  Scalar zc = - env.params().h0 * env.params().L0;

  Scalar b1 = getMonopole(x, y, z,
                          xc, yc1, zc, 1, 0);
  Scalar b2 = getMonopole(x, y, z,
                          xc, yc2, zc, -1, 0);
  return b1 + b2;
});

data.B0.initialize(1, [](Scalar x, Scalar y, Scalar z) {
  // Put your initial condition for By here
  Scalar xc = env.params().size[0] * 0.5;
  Scalar yc1 = env.params().size[1] * 0.5 - env.params().a0 * env.params().L0;
  Scalar yc2 = env.params().size[1] * 0.5 + env.params().a0 * env.params().L0;
  Scalar zc = - env.params().h0 * env.params().L0;

  Scalar b1 = getMonopole(x, y, z,
                          xc, yc1, zc, 1, 1);
  Scalar b2 = getMonopole(x, y, z,
                          xc, yc2, zc, -1, 1);
  return b1 + b2;
});

data.B0.initialize(2, [](Scalar x, Scalar y, Scalar z) {
  // Put your initial condition for Bz here
  Scalar xc = env.params().size[0] * 0.5;
  Scalar yc1 = env.params().size[1] * 0.5 - env.params().a0 * env.params().L0;
  Scalar yc2 = env.params().size[1] * 0.5 + env.params().a0 * env.params().L0;
  Scalar zc = - env.params().h0 * env.params().L0;

  Scalar b1 = getMonopole(x, y, z,
                          xc, yc1, zc, 1, 2);
  Scalar b2 = getMonopole(x, y, z,
                          xc, yc2, zc, -1, 2);
  return b1 + b2;
});

data.B.copy_from(data.B0);

#endif  // _USER_INIT_H_
