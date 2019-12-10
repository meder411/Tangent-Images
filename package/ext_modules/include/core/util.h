#ifndef CORE_UTIL_H_
#define CORE_UTIL_H_

#include <math.h>
#include <omp.h>
#include <cstdio>

// For GCC
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif

// From
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
// With help from https://stackoverflow.com/a/39287554/3427580
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
static __inline__ __device__ double atomicAdd(double *address, double val) {
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old             = *address_as_ull, assumed;
  if (val == 0.0) { return __longlong_as_double(old); }
  do {
    assumed = old;
    old     = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}
#endif

template <typename T>
__host__ __device__ inline void atomic_add(T *address, T val) {
#ifdef __CUDACC__  // CUDA versions of atomic add
  atomicAdd(address, val);
#else  // C++ version of atomic add
#pragma omp atomic
  *address += val;
#endif
}

template <typename T>
__host__ __device__ inline const T fnegmod(const T lval, const T rval) {
  return fmod(fmod(lval, rval) + rval, rval);
}

__host__ __device__ inline int64_t negmod(const int64_t lval,
                                          const int64_t rval) {
  return ((lval % rval) + rval) % rval;
}

#endif