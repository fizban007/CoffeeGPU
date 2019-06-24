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
  field_solver solver(data);

#include "user_init.hpp"

  // Main simulation loop
  for (int step = 0; step = env.params().max_steps; step++) {
    // Do stuff here
    solver.evolve_fields();
  }

  return 0;
}
