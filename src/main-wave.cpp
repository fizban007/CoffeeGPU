#include "data/fields.h"
#include "data/sim_data.h"
#include "sim_env.h"
#include "utils/data_exporter.h"
#include "algorithms/field_solver_EZ.h"
#include "algorithms/field_solver.h"
#include "utils/timer.h"
#include <fstream>

// #include "algorithms/metric.h"
// #include "algorithms/interpolation.h"
// #include "algorithms/pulsar.h"

using namespace std;
using namespace Coffee;

#define ENG
#define EZ

int main(int argc, char *argv[]) {
  timer::stamp("begin");
  // Initialize the simulation environment
  sim_environment env(&argc, &argv);

  int errorcode = 10;
  if (env.params().problem != 0) {
    std::cout << "This executable solves the alfven wave in periodic box problem. Please "
                 "set the 'problem' parameter to 0."
              << std::endl;
    MPI_Abort(env.cart(), errorcode);
  }

  // Initialize all the simulation data structures
  sim_data data(env);
  // field_solver_gr solver(data, env);
#ifdef EZ
  field_solver_EZ solver(data, env);
#else
  field_solver solver(data, env);
#endif

  #include "user_alfven.hpp"

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
  Scalar time = 0.0;
  for (step = 0; step <= env.params().max_steps; step++) {
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
          efile << Wb << " " << We << std::endl;
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
