#ifndef _CUDACONTROL_H_
#define _CUDACONTROL_H_

#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#define HD_INLINE __host__ __device__ __forceinline__
// #define WITH_CUDA_ENABLED
#else
#define HOST_DEVICE
#define HD_INLINE inline
// #ifdef WITH_CUDA_ENABLED
//   #undef WITH_CUDA_ENABLED
// #endif
#endif

#endif  // _CUDACONTROL_H_
