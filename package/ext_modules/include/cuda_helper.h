#ifndef CUDA_HELPER_H_
#define CUDA_HELPER_H_

#define CUDA_NUM_THREADS 512
#define CUDA_CHECK(err)                                   \
  if (cudaSuccess != err) {                               \
    fprintf(stderr, "CUDA kernel failed: %s (%s:%d)\n",   \
            cudaGetErrorString(err), __FILE__, __LINE__); \
    std::exit(-1);                                        \
  }

#define CHECK_CUDA(x) \
  AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CPU(x) \
  AT_ASSERTM(!x.type().is_cuda(), #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) \
  AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

#endif