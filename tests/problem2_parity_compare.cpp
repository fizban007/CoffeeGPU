#include "data/typedefs.h"
#include "data/vec3.h"
#include "utils/hdf_wrapper_impl.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace Coffee;

namespace {

struct ComparisonResult {
  std::string dataset;
  std::size_t mismatch_count = 0;
  double max_abs_diff = 0.0;
  double max_rel_diff = 0.0;
  Index worst_index;
  Scalar cpu_value = 0.0;
  Scalar gpu_value = 0.0;
};

double
default_abs_tol() {
#ifdef USE_DOUBLE
  return 1.0e-11;
#else
  return 5.0e-5;
#endif
}

double
default_rel_tol() {
#ifdef USE_DOUBLE
  return 1.0e-10;
#else
  return 1.0e-4;
#endif
}

bool
within_tol(Scalar a, Scalar b, double abs_tol, double rel_tol,
           double& abs_diff, double& rel_diff) {
  abs_diff = std::abs(static_cast<double>(a) - static_cast<double>(b));
  double scale =
      std::max({std::abs(static_cast<double>(a)),
                std::abs(static_cast<double>(b)), 1.0});
  rel_diff = abs_diff / scale;
  return abs_diff <= abs_tol || rel_diff <= rel_tol;
}

ComparisonResult
compare_dataset(const std::filesystem::path& cpu_path,
                const std::filesystem::path& gpu_path,
                const std::string& dataset_name, double abs_tol,
                double rel_tol) {
  H5File cpu_file(cpu_path.string(), H5OpenMode::read_only);
  H5File gpu_file(gpu_path.string(), H5OpenMode::read_only);

  auto cpu = cpu_file.read<Scalar>(dataset_name);
  auto gpu = gpu_file.read<Scalar>(dataset_name);
  cpu_file.close();
  gpu_file.close();

  auto ext_cpu = cpu.extent();
  auto ext_gpu = gpu.extent();
  if (ext_cpu.x != ext_gpu.x || ext_cpu.y != ext_gpu.y ||
      ext_cpu.z != ext_gpu.z) {
    std::ostringstream msg;
    msg << "Extent mismatch for " << dataset_name << ": cpu=("
        << ext_cpu.x << ", " << ext_cpu.y << ", " << ext_cpu.z
        << "), gpu=(" << ext_gpu.x << ", " << ext_gpu.y << ", "
        << ext_gpu.z << ")";
    throw std::runtime_error(msg.str());
  }

  ComparisonResult result;
  result.dataset = dataset_name;
  for (int k = 0; k < ext_cpu.z; ++k) {
    for (int j = 0; j < ext_cpu.y; ++j) {
      for (int i = 0; i < ext_cpu.x; ++i) {
        double abs_diff = 0.0;
        double rel_diff = 0.0;
        Scalar a = cpu(i, j, k);
        Scalar b = gpu(i, j, k);
        if (!within_tol(a, b, abs_tol, rel_tol, abs_diff, rel_diff)) {
          ++result.mismatch_count;
        }
        if (abs_diff > result.max_abs_diff) {
          result.max_abs_diff = abs_diff;
          result.max_rel_diff = rel_diff;
          result.worst_index = Index(i, j, k);
          result.cpu_value = a;
          result.gpu_value = b;
        }
      }
    }
  }
  return result;
}

void
compare_scalar_metadata(const std::filesystem::path& cpu_path,
                        const std::filesystem::path& gpu_path,
                        double abs_tol, double rel_tol) {
  H5File cpu_file(cpu_path.string(), H5OpenMode::read_only);
  H5File gpu_file(gpu_path.string(), H5OpenMode::read_only);

  auto cpu_step = cpu_file.read_scalar<uint32_t>("step");
  auto gpu_step = gpu_file.read_scalar<uint32_t>("step");
  if (cpu_step != gpu_step) {
    throw std::runtime_error("Snapshot step metadata differs");
  }

  auto cpu_time = cpu_file.read_scalar<Scalar>("time");
  auto gpu_time = gpu_file.read_scalar<Scalar>("time");
  double abs_diff = 0.0;
  double rel_diff = 0.0;
  if (!within_tol(cpu_time, gpu_time, abs_tol, rel_tol, abs_diff,
                  rel_diff)) {
    throw std::runtime_error("Snapshot time metadata differs");
  }

  cpu_file.close();
  gpu_file.close();
}

}  // namespace

int
main(int argc, char** argv) {
  if (argc < 3 || argc > 5) {
    std::cerr
        << "Usage: problem2_parity_compare <cpu_data_dir> <gpu_data_dir>"
        << " [abs_tol] [rel_tol]\n";
    return 2;
  }

  std::filesystem::path cpu_dir(argv[1]);
  std::filesystem::path gpu_dir(argv[2]);
  double abs_tol = argc >= 4 ? std::stod(argv[3]) : default_abs_tol();
  double rel_tol = argc >= 5 ? std::stod(argv[4]) : default_rel_tol();

  const std::vector<std::string> components = {"Ex", "Ey", "Ez",
                                               "Bx", "By", "Bz"};

  try {
    compare_scalar_metadata(cpu_dir / "snapshot_Ex.h5",
                            gpu_dir / "snapshot_Ex.h5", abs_tol,
                            rel_tol);

    bool failed = false;
    for (const auto& component : components) {
      auto filename = std::string("snapshot_") + component + ".h5";
      auto result =
          compare_dataset(cpu_dir / filename, gpu_dir / filename,
                          component, abs_tol, rel_tol);

      std::cout << component << ": max_abs_diff=" << result.max_abs_diff
                << ", max_rel_diff=" << result.max_rel_diff
                << ", mismatches=" << result.mismatch_count << "\n";

      if (result.mismatch_count > 0) {
        failed = true;
        std::cout << "  worst index=(" << result.worst_index[0] << ", "
                  << result.worst_index[1] << ", "
                  << result.worst_index[2] << "), cpu="
                  << result.cpu_value << ", gpu=" << result.gpu_value
                  << "\n";
      }
    }

    if (failed) {
      return 1;
    }
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << "\n";
    return 1;
  }

  std::cout << "CPU and GPU problem 2 snapshots match within tolerance."
            << std::endl;
  return 0;
}
