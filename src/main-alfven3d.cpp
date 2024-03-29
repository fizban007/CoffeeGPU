#include "algorithms/field_solver.h"
#include "algorithms/field_solver_EZ.h"
#include "data/fields.h"
#include "data/sim_data.h"
#include "sim_env.h"
#include "utils/data_exporter.h"
#include "utils/timer.h"
#include <fstream>

// #include "algorithms/metric.h"
// #include "algorithms/interpolation.h"
#include "algorithms/pulsar.h"

using namespace std;
using namespace Coffee;

// #define ENG
#define EZ

int
main(int argc, char *argv[]) {
  timer::stamp("begin");
  // Initialize the simulation environment
  sim_environment env(&argc, &argv);

  int errorcode = 10;
  if (env.params().problem != 2) {
    std::cout << "This executable solves the Alfven wave from pulsar "
                 "problem. Please set the 'problem' parameter to 2."
              << std::endl;
    MPI_Abort(env.cart(), errorcode);
  }

  // Initialize all the simulation data structures
  sim_data data(env);

#ifdef EZ
#include "user_pulsar3d_EZ.hpp"
#else
#include "user_pulsar3d.hpp"
#endif

#ifdef EZ
  field_solver_EZ solver(data, env);
#else
  field_solver solver(data, env);
#endif

  uint32_t step = 0;
  data_exporter exporter(env, step);

  // std::cout << "Attempting to write output" << std::endl;
  // exporter.write_output(data, step, 0.0);
  // exporter.sync();

#ifdef ENG
  ofstream efile;
#endif

  // Main simulation loop
  Scalar time = step * env.params().dt;
  uint32_t prev_snapshot = -1;
  if (env.is_restart()) {
    // cout << "Restarting from snapshot file " << env.restart_file() <<
    // "\n";
    // exporter.load_snapshot(env.restart_file(), data, step,
    // time);
    cout << "Restarting from snapshot file(s) \n";
    exporter.load_snapshot_multiple(data, step, time);
    prev_snapshot = step;
  }

  for (; step <= env.params().max_steps; step++) {
    if (step % env.params().snapshot_interval == 0 &&
        step != prev_snapshot) {
      timer::stamp("restart");
      // exporter.save_snapshot(
      //     exporter.output_directory() + "snapshot.h5", data, step,
      //     time);
      exporter.save_snapshot_multiple(data, step, time);
      if (env.rank() == 0)
        timer::show_duration_since_stamp("writing a restart file", "s",
                                         "restart");
    }
    if (env.rank() == 0) std::cout << "step = " << step << std::endl;
    // Do stuff here
    if (step % env.params().data_interval == 0 &&
        step != prev_snapshot) {
      timer::stamp("output");
      exporter.write_output(data, step, 0.0);
      if (env.rank() == 0)
        timer::show_duration_since_stamp("output", "ms", "output");

#ifdef ENG
      Scalar Wb = solver.total_energy(data.B);
      Scalar We = solver.total_energy(data.E);
      if (env.rank() == 0) {
        efile.open("Data/energy.txt", ios::out | ios::app);
        efile << Wb << " " << We << std::endl;
        efile.close();
      }
#endif
    }

    if ((env.params().slice_x || env.params().slice_y ||
         env.params().slice_z || env.params().slice_xy) &&
        (step % env.params().slice_interval == 0)) {
      timer::stamp("slice output");
      exporter.write_slice_output(data, step, 0.0);
      if (env.rank() == 0)
        timer::show_duration_since_stamp("slice output", "ms",
                                         "slice output");
    }

    timer::stamp("step");
    // solver.evolve_fields_gr();
    solver.evolve_fields(time);
    if (env.rank() == 0)
      timer::show_duration_since_stamp("evolve field", "ms", "step");
    time += env.params().dt;
  }

  timer::show_duration_since_stamp("the whole program", "s", "begin");

  return 0;
}
