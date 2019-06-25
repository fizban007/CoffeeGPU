#include "algorithms/interpolation.h"
#include "catch.hpp"
#include "data/fields.h"
#include "data/vec3.h"

using namespace Coffee;

TEST_CASE("Trying out simple interpolation", "[interp]") {
  multi_array<float> f(10, 10, 10);
  for (int k = 0; k < 10; ++k) {
    for (int j = 0; j < 10; ++j) {
      for (int i = 0; i < 10; ++i) {
        f(i, j, k) = i * j * k;
      }
    }
  }

  CHECK(interpolate(f.host_ptr(), Index(5, 5, 6), Stagger(0b011),
                    Stagger(0b111), 10, 10) == Approx(5.0 * 5.0 * 5.5));

  CHECK(interpolate(f.host_ptr(), Index(2, 3, 4), Stagger(0b011),
                    Stagger(0b100), 10, 10) == Approx(2.5 * 3.5 * 3.5));

  CHECK(interpolate(f.host_ptr(), Index(4, 2, 6), Stagger(0b010),
                    Stagger(0b101), 10, 10) == Approx(3.5 * 2.5 * 5.5));

  CHECK(interpolate(f.host_ptr(), Index(7, 5, 3), Stagger(0b010),
                    Stagger(0b011), 10, 10) == Approx(6.5 * 5.0 * 3.0));
}
