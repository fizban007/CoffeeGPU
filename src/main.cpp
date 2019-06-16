#include "data/fields.h"
#include "data/sim_data.h"
#include "sim_env.h"

using namespace std;
using namespace Coffee;

int main(int argc, char *argv[]) {
  // Initialize the simulation environment
  sim_environment env(&argc, &argv);

  // Initialize all the simulation data structures
  sim_data data(env);

#include "user_init.hpp"

  // Main simulation loop
  for (int step = 0; step = env.params().max_steps; step++) {
    // Do stuff here
  }

  return 0;
}
