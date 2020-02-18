#ifndef _HDF_WRAPPER_H_
#define _HDF_WRAPPER_H_

#include "data/multi_array.h"
#include "data/vec3.h"
#include "hdf5.h"
#include <string>

namespace Coffee {

enum class H5OpenMode {
  read_write,
  read_only,
  rw_parallel,
  read_parallel
};

enum class H5CreateMode { trunc, excl, trunc_parallel, excl_parallel };

class H5File {
 private:
  hid_t m_file_id = 0;
  bool m_is_open = false;
  bool m_is_parallel = false;

 public:
  H5File();
  H5File(hid_t file_id);
  H5File(const std::string& filename,
         H5OpenMode mode = H5OpenMode::read_only);
  H5File(const H5File& other) = delete;
  H5File(H5File&& other);
  ~H5File();

  H5File& operator=(const H5File&) = delete;
  H5File& operator=(H5File&& other);

  void open(const std::string& filename,
            H5OpenMode mode = H5OpenMode::read_only);
  void close();

  template <typename T>
  void write(const multi_array<T>& array, const std::string& name);
  template <typename T>
  void write(T value, const std::string& name);
  template <typename T>
  void write_parallel(const multi_array<T>& array,
                      const Extent& ext_total, const Index& idx_dst,
                      const Extent& ext, const Index& idx_src,
                      const std::string& name);

  template <typename T>
  multi_array<T> read(const std::string& name);
  template <typename T>
  T read_scalar(const std::string& name);

  template <typename T>
  void read_subset(multi_array<T>& array, const std::string& name,
                   const Index& idx_src, const Extent& ext,
                   const Index& idx_dst);

  void set_parallel(bool p) { m_is_parallel = p; }
};

H5File hdf_create(const std::string& filename,
                  H5CreateMode mode = H5CreateMode::trunc);

template <typename T>
hid_t h5datatype();

}  // namespace Coffee

#include "hdf_wrapper_impl.hpp"

#endif  // _HDF_WRAPPER_H_
