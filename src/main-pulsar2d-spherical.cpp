#include "algorithms/field_solver_EZ_spherical.h"
#include "data/fields.h"
#include "data/sim_data.h"
#include "sim_env.h"
#include "utils/data_exporter.h"
#include "utils/timer.h"
#include <fstream>
#include <iomanip>

#include "algorithms/metric_sph.h"
#include "algorithms/pulsar.h"

using namespace std;
using namespace Coffee;
using namespace SPH;

#define ENG

int
main(int argc, char *argv[]) {
  timer::stamp("begin");
  // Initialize the simulation environment
  sim_environment env(&argc, &argv);

  // Initialize all the simulation data structures
  sim_data data(env);
  field_solver_EZ_spherical solver(data, env);

#include "user_pulsar_sph.hpp"

  uint32_t step = 0;
  data_exporter exporter(env, step);

  // std::cout << "Attempting to write output" << std::endl;
  // exporter.write_output(data, step, 0.0);
  // exporter.sync();

#ifdef ENG
  ofstream efile;
  efile.open("Data/energy.txt", ios::out | ios::app);
#endif

  // Main simulation loop
  Scalar time = step * env.params().dt;
  if (env.is_restart()) {
    cout << "Restarting from snapshot file " << env.restart_file() << "\n";
    exporter.load_snapshot(env.restart_file(), data, step, time);
  }

  for (; step <= env.params().max_steps; step++) {
    if (step % env.params().snapshot_interval == 0) {
      timer::stamp("restart");
      exporter.save_snapshot(
          exporter.output_directory() + "snapshot.h5", data, step,
          time);
      if (env.rank() == 0)
        timer::show_duration_since_stamp("writing a restart file", "s",
                                         "restart");
    }
    std::cout << "step = " << step << std::endl;
    // Do stuff here
    if (step % env.params().data_interval == 0) {
      timer::stamp("output");
      exporter.write_output(data, step, 0.0);
      if (env.rank() == 0)
        timer::show_duration_since_stamp("output", "ms", "output");

#ifdef ENG
      Scalar Wb = solver.total_energy(data.B);
      Scalar We = solver.total_energy(data.E);
      if (env.rank() == 0) {
        efile << std::setprecision(8) << Wb << " " << We << std::endl;
      }
#endif
    }
    timer::stamp("step");
    // solver.evolve_fields_gr();
    solver.evolve_fields(time);
    if (env.rank() == 0)
      timer::show_duration_since_stamp("evolve field", "ms", "step");
    time += env.params().dt;
  }

#ifdef ENG
  efile.close();
#endif

  timer::show_duration_since_stamp("the whole program", "s", "begin");

  return 0;
}
