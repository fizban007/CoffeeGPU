#include "catch.hpp"

__global__ void
test_kernel() {
  printf("hello\n");
}

TEST_CASE("Launching kernel", "[kernel]") {
  test_kernel<<<1, 1>>>();
  cudaDeviceSynchronize();
}
