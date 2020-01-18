#ifndef _TYPEDEFS_H_
#define _TYPEDEFS_H_

namespace Coffee {

#ifndef USE_DOUBLE
typedef float Scalar;
#define TINY 1e-7
#else
typedef double Scalar;
#define TINY 1e-11
#endif

}

#endif  // _TYPEDEFS_H_
