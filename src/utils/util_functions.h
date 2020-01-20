#ifndef _UTIL_FUNCTIONS_H_
#define _UTIL_FUNCTIONS_H_

#include "cuda/cuda_control.h"

namespace Coffee {

template <typename T>
HD_INLINE T
square(T x) {
  return x * x;
}

template <typename T>
HD_INLINE T
cube(T x) {
  return x * x * x;
}

}  // namespace Coffee

#endif  // _UTIL_FUNCTIONS_H_
