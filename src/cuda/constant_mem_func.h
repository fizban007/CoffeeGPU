#ifndef _CONSTANT_MEM_FUNC_H_
#define _CONSTANT_MEM_FUNC_H_

namespace Coffee {

// Copy a given parameter struct to constant memory. This can be used to
// update the parameters on device even after initializing the
// simulation
void init_dev_params(const sim_params& params);

// Copy a given grid configuration to constant memory.
void init_dev_grid(const Grid& g);

}  // namespace Coffee

#endif  // _CONSTANT_MEM_FUNC_H_
