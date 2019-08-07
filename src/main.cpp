#include "data/fields.h"
#include "data/sim_data.h"
#include "sim_env.h"
#include "utils/data_exporter.h"
#include "algorithms/field_solver_resistive.h"
#include "utils/timer.h"

#include "algorithms/metric.h"
#include "algorithms/interpolation.h"

#include <iomanip>

#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>
// using namespace H5;
using namespace HighFive;

using namespace std;
using namespace Coffee;

int main(int argc, char *argv[]) {
  timer::stamp("begin");
  // Initialize the simulation environment
  sim_environment env(&argc, &argv);

  // Initialize all the simulation data structures
  sim_data data(env);
  field_solver_resistive solver(data, env);

  #include "user_init.hpp"
  // #include "user_emwave.hpp"
  // #include "user_alfven.hpp"

  uint32_t step = 0;
  data_exporter exporter(env, step);

  // std::cout << "Attempting to write output" << std::endl;
  // exporter.write_output(data, step, 0.0);
  // exporter.sync();

  // Main simulation loop
  env.params().vacuum = true;
  for (step = 0; step <= env.params().max_steps; step++) {
    std::cout << "step = " << step << std::endl;
    // Do stuff here
    if (step % env.params().data_interval == 0) {
      exporter.write_output(data, step, 0.0);
    }
    if (env.params().lc_interval > 0 && step > env.params().vacstep && step % env.params().lc_interval == 0) {
      solver.light_curve(step);
      if (env.rank() == 0) {
        std::stringstream ss;
        ss << std::setw(5) << std::setfill('0')
          << step / env.params().lc_interval;
        std::string num = ss.str();
        File file(std::string("./Data/lc") + num + std::string(".h5"), 
          File::ReadWrite | File::Create | File::Truncate);
        DataSet dataset = file.createDataSet<Scalar>("/lc",  DataSpace::From(solver.lc0));
        dataset.write(solver.lc0);
      }
    }
    timer::stamp("step");
    if (step == env.params().vacstep) env.params().vacuum = false;
    solver.evolve_fields();
    timer::show_duration_since_stamp("evolve field", "ms", "step");
  }

  timer::show_duration_since_stamp("the whole program", "s", "begin");

  return 0;
}
