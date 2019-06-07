#include "data/stagger.h"
#include "catch.hpp"

using namespace Coffee;

TEST_CASE("Initializing stagger in different ways", "[Stagger]") {
  Stagger st(0b001);

  CHECK(st[0] == 1);
  CHECK(st[1] == 0);
  CHECK(st[2] == 0);

  Stagger st2(st);

  CHECK(st2[0] == 1);
  CHECK(st2[1] == 0);
  CHECK(st2[2] == 0);
}

TEST_CASE("Testing bit manipulations", "[Stagger]") {
  Stagger st(0b010);

  st.set_bit(2, 1);

  CHECK(st[0] == 0);
  CHECK(st[1] == 1);
  CHECK(st[2] == 1);

  st.flip(1);

  CHECK(st[1] == 0);

  Stagger cp = st.complement();

  CHECK(cp[0] == 1);
  CHECK(cp[1] == 1);
  CHECK(cp[2] == 0);
}
