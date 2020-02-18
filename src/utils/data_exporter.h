#ifndef _DATA_EXPORTER_H_
#define _DATA_EXPORTER_H_

#include "data/multi_array.h"
#include "data/typedefs.h"
#include "hdf_wrapper.h"
#include <fstream>
#include <memory>
#include <thread>
#include <vector>

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

  void save_snapshot(const std::string& filename, sim_data& data,
                     uint32_t step, Scalar time);
  void load_snapshot(const std::string& filename, sim_data& data,
                     uint32_t& step, Scalar& time);
  // void write_multi_array(multi_array<Scalar>& array, const
  // std::string& name,
  //                        const Extent& total_ext, const Index&
  //                        offset, hid_t file_id);
  void write_multi_array(const multi_array<Scalar>& array,
                         const std::string& name, H5File& file);

  const std::string& output_directory() const {
    return outputDirectory;
  }

 protected:
  // template <typename Func>
  // void add_grid_output(sim_data& data, const std::string& name, Func
  // f,
  //                      // HighFive::File& file);
  //                      hid_t file_id);

  void add_grid_output(multi_array<Scalar>& array,
                       const std::string& name, Stagger stagger,
                       // HighFive::File& file);
                       H5File& file);

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
  std::unique_ptr<std::thread> m_thread;
};

}  // namespace Coffee

#endif  // _DATA_EXPORTER_H_
