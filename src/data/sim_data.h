#ifndef _SIM_DATA_H_
#define _SIM_DATA_H_

#include "fields.h"
#include "grid.h"

namespace Coffee {

class sim_data {
 public:
  vector_field<Scalar> E, B;
  Grid grid;
};

}  // namespace Coffee

#endif  // _SIM_DATA_H_
