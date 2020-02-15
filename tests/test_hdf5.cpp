#include "catch.hpp"
#include "data/multi_array.h"
#include "utils/hdf_wrapper_impl.hpp"
#include <iostream>

using namespace Coffee;

TEST_CASE("Saving a multi_array to hdf5", "[hdf5]") {
  // int N = 10;
  multi_array<float> a(8, 9, 10);
  a.assign(3.0f);

  auto file = hdf_create("test_output.h5", H5CreateMode::trunc);
  file.write(a, "a");
  file.close();
}

TEST_CASE("Read a multi_array from hdf5", "[hdf5]") {
  H5File file("test_output.h5");

  auto array = file.read<float>("a");

  auto ext = array.extent();
  REQUIRE(ext.x == 8);
  REQUIRE(ext.y == 9);
  REQUIRE(ext.z == 10);
  REQUIRE(array(1, 2, 3) == 3.0f);
  file.close();
}

TEST_CASE("Writing a single number", "[hdf5]") {
  H5File file("test_output.h5", H5OpenMode::read_write);
  file.write(42.0, "const");
  file.close();
}

TEST_CASE("Reading a single number", "[hdf5]") {
  H5File file("test_output.h5", H5OpenMode::read_only);
  auto a = file.read_scalar<float>("const");
  REQUIRE(a == 42.0f);
  file.close();
}
