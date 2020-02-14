#ifndef _SIMD_H_
#define _SIMD_H_

// #include <immintrin.h>
#include "data/typedefs.h"
#define MAX_VECTOR_SIZE 256
#include "vectorclass.h"

namespace Coffee {

namespace simd {

#if !defined(USE_DOUBLE) && \
    (defined(__AVX512F__) || defined(__AVX512__))
#define USE_SIMD
typedef Vec16ui Vec_ui_t;
typedef Vec16i Vec_i_t;
typedef Vec16ib Vec_ib_t;
typedef Vec16f Vec_f_t;
constexpr int vec_width = 16;
#elif defined(USE_DOUBLE) && \
    (defined(__AVX512F__) || defined(__AVX512__))
#define USE_SIMD
typedef Vec8uq Vec_ui_t;
typedef Vec8q Vec_i_t;
typedef Vec8qb Vec_ib_t;
typedef Vec8d Vec_f_t;
constexpr int vec_width = 8;
#elif !defined(USE_DOUBLE) && defined(__AVX2__)
#define USE_SIMD
typedef Vec8ui Vec_idx_t;
typedef Vec8ui Vec_ui_t;
typedef Vec8i Vec_i_t;
typedef Vec8ib Vec_ib_t;
typedef Vec8f Vec_f_t;
constexpr int vec_width = 8;
#elif defined(USE_DOUBLE) && defined(__AVX2__)
#define USE_SIMD
typedef Vec8ui Vec_idx_t;
typedef Vec4uq Vec_ui_t;
typedef Vec4q Vec_i_t;
typedef Vec4qb Vec_ib_t;
typedef Vec4d Vec_f_t;
constexpr int vec_width = 4;
#else
#undef USE_SIMD
typedef uint32_t Vec_idx_t;
typedef uint32_t Vec_ui_t;
typedef int Vec_i_t;
typedef bool Vec_ib_t;
typedef float Vec_f_t;
constexpr int vec_width = 1;
#endif

}  // namespace simd

}  // namespace Coffee

#endif  // _SIMD_H_
