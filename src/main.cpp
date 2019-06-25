#include "algorithms/field_solver.h"
#include "data/fields.h"
#include "data/sim_data.h"
#include "sim_env.h"
#include "utils/data_exporter.h"
#include "utils/timer.h"

using namespace std;
using namespace Coffee;

int
main(int argc, char* argv[]) {
  // Initialize the simulation environment
  sim_environment env(&argc, &argv);

  // Initialize all the simulation data structures
  sim_data data(env);
  field_solver solver(data);

  // #include "user_init.hpp"

  data.E.initialize(
      0, [&env](Scalar x, Scalar y, Scalar z) { return env.rank(); });

  uint32_t step = 0;
  data_exporter exporter(env, step);

  // env.send_guard_cell_x(data, -1);

  // data.E.sync_to_host(0);
  // if (env.rank() == 2) {
  //   const Grid& g = env.grid();
  //   for (int j = 0; j < g.dims[1]; j++) {
  //     for (int i = 0; i < g.dims[0]; i++) {
  //       cout << data.E(0, i, j, 0) << " ";
  //     }
  //     cout << endl;
  //   }
  // }

  // std::cout << "Attempting to write output" << std::endl;
  // exporter.write_output(data, step, 0.0);
  // exporter.sync();

  // Main simulation loop
  for (step = 0; step <= env.params().max_steps; step++) {
    // Do stuff here
    timer::stamp("evolve");
    solver.evolve_fields();
    timer::show_duration_since_stamp("evolving fields", "ms",
    "evolve");
  }

  return 0;
}
