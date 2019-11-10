#ifndef _DATA_EXPORTER_H_
#define _DATA_EXPORTER_H_

#include "data/multi_array.h"
#include "data/typedefs.h"
#include <boost/multi_array.hpp>
#include <fstream>
#include <memory>
#include <thread>
#include <vector>
#include "hdf5.h"

namespace H5 {
class H5File;
}

namespace HighFive {
class File;
}

namespace Coffee {

struct sim_data;
class sim_environment;

class data_exporter {
 public:
  data_exporter(sim_environment& env, uint32_t& timestep);
  virtual ~data_exporter();

  void write_output(sim_data& data, uint32_t timestep, double time);

  void write_field_output(sim_data& data, uint32_t timestep,
                          double time);

  void sync();

 protected:
  // template <typename Func>
  // void add_grid_output(sim_data& data, const std::string& name, Func f,
  //                      // HighFive::File& file);
  //                      hid_t file_id);

  void add_grid_output(multi_array<Scalar>& array, const std::string& name,
                       Stagger stagger,
                       // HighFive::File& file);
                       hid_t file_id);

  // std::unique_ptr<Grid> grid;
  sim_environment& m_env;
  std::string
      outputDirectory;  //!< Sets the directory of all the data files
  // std::string filePrefix;  //!< Sets the common prefix of the data
  // files

  std::ofstream xmf;  //!< This is the accompanying xmf file describing
                      //!< the hdf structure

  multi_array<float> tmp_grid_data;  //!< This stores the temporary
                                     //!< downsampled data for output
  boost::multi_array<float, 3> m_output;

  std::unique_ptr<std::thread> m_thread;
};

}  // namespace Coffee

#endif  // _DATA_EXPORTER_H_
