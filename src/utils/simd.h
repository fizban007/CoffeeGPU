#ifndef _SIMD_H_
#define _SIMD_H_

// #include <immintrin.h>
#include "data/typedefs.h"

namespace Coffee {

namespace simd {

#if !defined(USE_DOUBLE) && \
    (defined(__AVX512F__) || defined(__AVX512__))
#define MAX_VECTOR_SIZE 512
#include "vectorclass.h"
#define USE_SIMD
#pragma message "using AVX512 with float"
typedef Vec16ui Vec_ui_t;
typedef Vec16i Vec_i_t;
typedef Vec16ib Vec_ib_t;
typedef Vec16f Vec_f_t;
const Vec_f_t vec_inc = Vec16f(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f,
                               8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f);
constexpr int vec_width = 16;
#elif defined(USE_DOUBLE) && \
    (defined(__AVX512F__) || defined(__AVX512__))
#define MAX_VECTOR_SIZE 512
#include "vectorclass.h"
#define USE_SIMD
#pragma message "using AVX512 with double"
typedef Vec8uq Vec_ui_t;
typedef Vec8q Vec_i_t;
typedef Vec8qb Vec_ib_t;
typedef Vec8d Vec_f_t;
const Vec_f_t vec_inc = Vec8d(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0);
constexpr int vec_width = 8;
#elif !defined(USE_DOUBLE) && defined(__AVX2__)
#define MAX_VECTOR_SIZE 512
#include "vectorclass.h"
#define USE_SIMD
#pragma message "using AVX2 with float"
typedef Vec8ui Vec_idx_t;
typedef Vec8ui Vec_ui_t;
typedef Vec8i Vec_i_t;
typedef Vec8ib Vec_ib_t;
typedef Vec8f Vec_f_t;
const Vec_f_t vec_inc = Vec8f(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0);
constexpr int vec_width = 8;
#elif defined(USE_DOUBLE) && defined(__AVX2__)
#define MAX_VECTOR_SIZE 512
#include "vectorclass.h"
#define USE_SIMD
#pragma message "using AVX2 with double"
typedef Vec8ui Vec_idx_t;
typedef Vec4uq Vec_ui_t;
typedef Vec4q Vec_i_t;
typedef Vec4qb Vec_ib_t;
typedef Vec4d Vec_f_t;
const Vec_f_t vec_inc = Vec4d(0.0, 1.0, 2.0, 3.0);
constexpr int vec_width = 4;
#else
#undef USE_SIMD
typedef uint32_t Vec_idx_t;
typedef uint32_t Vec_ui_t;
typedef int Vec_i_t;
typedef bool Vec_ib_t;
typedef float Vec_f_t;
const Vec_i_t vec_int = 0;
constexpr int vec_width = 1;
#endif

}  // namespace simd

}  // namespace Coffee

#endif  // _SIMD_H_
