#include "catch.hpp"
#include "data/multi_array.h"

using namespace Coffee;

TEST_CASE("Assign a single value to a multi_array", "[multi_array]") {
  int N = 100;
  multi_array<float> a(N, N, N);
  a.assign_dev(3.0f);
  a.sync_to_host();

  for (int k = 0; k < N; k++) {
    for (int j = 0; j < N; j++) {
      for (int i = 0; i < N; i++) {
        CHECK(a(i, j, k) == 3.0f);
      }
    }
  }
}

TEST_CASE("Checking that index is computed correctly", "[multi_array]") {
  int N = 50;
  multi_array<float> a(N, N, N);

  for (int k = 0; k < N; k++) {
    for (int j = 0; j < N; j++) {
      for (int i = 0; i < N; i++) {
        a(i, j, k) = i + j * a.width() + k * a.width() * a.height();
      }
    }
  }

  for (int n = 0; n < a.size(); n++) {
    CHECK(a[n] == n);
  }
}
