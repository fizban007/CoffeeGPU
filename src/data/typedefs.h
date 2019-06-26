#ifndef _TYPEDEFS_H_
#define _TYPEDEFS_H_

namespace Coffee {

#ifndef USE_DOUBLE
typedef float Scalar;
#else
typedef double Scalar;
#endif

#define M_PI (3.141592653589793)

}

#endif  // _TYPEDEFS_H_
