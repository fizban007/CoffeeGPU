#ifndef _USER_INIT_H_
#define _USER_INIT_H_

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

data.B.initialize(0, [](Scalar x, Scalar y, Scalar z) {
  // Put your initial condition for Bx here
  return 0.0;
});

data.B.initialize(1, [](Scalar x, Scalar y, Scalar z) {
  // Put your initial condition for By here
  return 0.0;
});

data.B.initialize(2, [](Scalar x, Scalar y, Scalar z) {
  // Put your initial condition for Bz here
  return 0.0;
});

#endif  // _USER_INIT_H_
