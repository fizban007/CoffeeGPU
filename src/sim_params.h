#ifndef _SIM_PARAMS_H_
#define _SIM_PARAMS_H_

#include "data/typedefs.h"

namespace Coffee {

struct sim_params {
  // Simulation parameters
  Scalar dt = 0.01;
  int max_steps = 10000;
  int data_interval = 100;
  bool periodic_boundary[3] = {false};

  // Grid parameters
  int N[3] = {1};
  int guard[3] = {0};
  Scalar lower[3] = {0.0};
  Scalar size[3] = {1.0};
};

}

#endif  // _SIM_PARAMS_H_
