#include "field_solver_EZ.h"

#include <omp.h>

#include "algorithms/damping_boundary.h"
#include "algorithms/finite_diff_simd.h"
#include "algorithms/pulsar.h"
#include "algorithms/metric_sph.h"
#include "utils/simd.h"
#include "utils/timer.h"

namespace Coffee {

using namespace simd;

const Grid *l_grid;


}  // namespace Coffee
