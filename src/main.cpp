#include "data/fields.h"
#include "data/sim_data.h"
#include "sim_env.h"
#include "utils/data_exporter.h"
#include "algorithms/field_solver.h"
#include "utils/timer.h"

using namespace std;
using namespace Coffee;

int main(int argc, char *argv[]) {
  // Initialize the simulation environment
  sim_environment env(&argc, &argv);

  // Initialize all the simulation data structures
  sim_data data(env);
  field_solver solver(data, env);

// #include "user_init.hpp"
  #include "user_emwave.hpp"

  uint32_t step = 0;
  data_exporter exporter(env, step);

  // std::cout << "Attempting to write output" << std::endl;
  // exporter.write_output(data, step, 0.0);
  // exporter.sync();

  // Main simulation loop
  for (step = 0; step <= env.params().max_steps; step++) {
    // Do stuff here
    timer::stamp();
    solver.evolve_fields();
    timer::show_duration_since_stamp("evolve field", "ms");
  }

  return 0;
}
