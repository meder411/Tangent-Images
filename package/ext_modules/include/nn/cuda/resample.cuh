#ifndef RESAMPLE_CUH_
#define RESAMPLE_CUH_

#include <torch/extension.h>

#include "cuda_helper.h"
#include "nn/common/resample.h"

namespace tangent_images {
namespace nn {
namespace cuda {

template <typename T>
__global__ void ResampleToMap2DKernel(
    const int64_t n, const T *__restrict__ data_in_ptr,
    const T *__restrict__ sample_map_ptr, const int64_t channels,
    const int64_t in_height, const int64_t in_width, const int64_t out_height,
    const int64_t out_width, const int64_t interpolation,
    T *__restrict__ data_out_ptr) {
  int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n) { return; }

  common::ResampleToMap2D(index, data_in_ptr, sample_map_ptr, channels,
                          in_height, in_width, out_height, out_width,
                          interpolation, data_out_ptr);
}

void ResampleToMap2DLauncher(torch::Tensor data_in,
                             torch::Tensor sample_map,  // IH, IW, 2
                             const int64_t channels, const int64_t in_height,
                             const int64_t in_width, const int64_t out_height,
                             const int64_t out_width,
                             const int64_t interpolation,
                             torch::Tensor data_out) {
  const int64_t num_kernels = channels * in_height * in_width;
  const dim3 blocks((num_kernels + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);

  AT_DISPATCH_FLOATING_TYPES(
      data_in.scalar_type(), "ResampleToMap2DKernel", ([&] {
        ResampleToMap2DKernel<scalar_t><<<blocks, CUDA_NUM_THREADS>>>(
            num_kernels, data_in.data<scalar_t>(),
            sample_map.data<scalar_t>(), channels, in_height, in_width,
            out_height, out_width, interpolation,
            data_out.data<scalar_t>());
      }));
  CUDA_CHECK(cudaGetLastError())
}

template <typename T>
__global__ void ResampleFromMap2DKernel(
    const int64_t n, const T *__restrict__ data_out_ptr,
    const T *__restrict__ sample_map_ptr,  // IH, IW, 2
    const int64_t channels, const int64_t in_height, const int64_t in_width,
    const int64_t out_height, const int64_t out_width,
    const int64_t interpolation, T *__restrict__ data_in_ptr) {
  int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n) { return; }

  common::ResampleFromMap2D(index, data_out_ptr, sample_map_ptr, channels,
                            in_height, in_width, out_height, out_width,
                            interpolation, data_in_ptr);
}

void ResampleFromMap2DLauncher(
    torch::Tensor data_out, torch::Tensor sample_map, const int64_t channels,
    const int64_t in_height, const int64_t in_width, const int64_t out_height,
    const int64_t out_width, const int64_t interpolation,
    torch::Tensor data_in) {
  const int64_t num_kernels = channels * in_height * in_width;
  const dim3 blocks((num_kernels + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);

  AT_DISPATCH_FLOATING_TYPES(
      data_in.scalar_type(), "ResampleFromMap2DKernel", ([&] {
        ResampleFromMap2DKernel<scalar_t><<<blocks, CUDA_NUM_THREADS>>>(
            num_kernels, data_out.data<scalar_t>(),
            sample_map.data<scalar_t>(), channels, in_height, in_width,
            out_height, out_width, interpolation,
            data_in.data<scalar_t>());
      }));
  CUDA_CHECK(cudaGetLastError())
}

// ----------------------------------------------
// ----------------------------------------------

template <typename T>
__global__ void ResampleToMap2DWeightedKernel(
    const int64_t n, const T *__restrict__ data_in_ptr,
    const T *__restrict__ sample_map_ptr,
    const T *__restrict__ interp_weights_ptr, const int64_t channels,
    const int64_t in_height, const int64_t in_width, const int64_t out_height,
    const int64_t out_width, const int64_t interpolation,
    const int64_t num_interp_pts, T *__restrict__ data_out_ptr) {
  int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n) { return; }

  common::ResampleToMap2DWeighted(index, data_in_ptr, sample_map_ptr,
                                  interp_weights_ptr, channels, in_height,
                                  in_width, out_height, out_width,
                                  interpolation, num_interp_pts, data_out_ptr);
}

void ResampleToMap2DWeightedLauncher(
    torch::Tensor data_in,
    torch::Tensor sample_map,      // IH, IW, P, 2
    torch::Tensor interp_weights,  // IH, IW, P
    const int64_t channels, const int64_t in_height, const int64_t in_width,
    const int64_t out_height, const int64_t out_width,
    const int64_t interpolation, const int64_t num_interp_pts,
    torch::Tensor data_out) {
  const int64_t num_kernels = channels * in_height * in_width;
  const dim3 blocks((num_kernels + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);

  AT_DISPATCH_FLOATING_TYPES(
      data_in.scalar_type(), "ResampleToMap2DWeightedKernel", ([&] {
        ResampleToMap2DWeightedKernel<scalar_t><<<blocks, CUDA_NUM_THREADS>>>(
            num_kernels, data_in.data<scalar_t>(),
            sample_map.data<scalar_t>(),
            interp_weights.data<scalar_t>(), channels, in_height, in_width,
            out_height, out_width, interpolation, num_interp_pts,
            data_out.data<scalar_t>());
      }));
  CUDA_CHECK(cudaGetLastError())
}

template <typename T>
__global__ void ResampleFromMap2DWeightedKernel(
    const int64_t n, const T *__restrict__ data_out_ptr,
    const T *__restrict__ sample_map_ptr,      // IH, IW, P, 2
    const T *__restrict__ interp_weights_ptr,  // IH, IW, P
    const int64_t channels, const int64_t in_height, const int64_t in_width,
    const int64_t out_height, const int64_t out_width,
    const int64_t interpolation, const int64_t num_interp_pts,
    T *__restrict__ data_in_ptr) {
  int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n) { return; }

  common::ResampleFromMap2DWeighted(
      index, data_out_ptr, sample_map_ptr, interp_weights_ptr, channels,
      in_height, in_width, out_height, out_width, interpolation,
      num_interp_pts, data_in_ptr);
}

void ResampleFromMap2DWeightedLauncher(
    torch::Tensor data_out, torch::Tensor sample_map,
    torch::Tensor interp_weights, const int64_t channels,
    const int64_t in_height, const int64_t in_width, const int64_t out_height,
    const int64_t out_width, const int64_t interpolation,
    const int64_t num_interp_pts, torch::Tensor data_in) {
  const int64_t num_kernels = channels * in_height * in_width;
  const dim3 blocks((num_kernels + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);

  AT_DISPATCH_FLOATING_TYPES(
      data_in.scalar_type(), "ResampleFromMap2DWeightedKernel", ([&] {
        ResampleFromMap2DWeightedKernel<scalar_t>
            <<<blocks, CUDA_NUM_THREADS>>>(
                num_kernels, data_out.data<scalar_t>(),
                sample_map.data<scalar_t>(),
                interp_weights.data<scalar_t>(), channels, in_height,
                in_width, out_height, out_width, interpolation, num_interp_pts,
                data_in.data<scalar_t>());
      }));
  CUDA_CHECK(cudaGetLastError())
}

// ----------------------------------------------
// ----------------------------------------------

template <typename T>
__global__ void ResampleToMap2DVotingKernel(
    const int64_t n, const T *__restrict__ data_in_ptr,
    const T *__restrict__ sample_map_ptr, const int64_t channels,
    const int64_t in_height, const int64_t in_width, const int64_t out_height,
    const int64_t out_width, const int64_t numCandidates,
    T *__restrict__ data_out_ptr) {
  int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n) { return; }

  common::ResampleToMap2DVoting(index, data_in_ptr, sample_map_ptr, channels,
                                in_height, in_width, out_height, out_width,
                                numCandidates, data_out_ptr);
}

void ResampleToMap2DVotingLauncher(
    torch::Tensor data_in,
    torch::Tensor sample_map,  // IH, IW, P, 2
    const int64_t channels, const int64_t in_height, const int64_t in_width,
    const int64_t out_height, const int64_t out_width,
    const int64_t numCandidates, torch::Tensor data_out) {
  const int64_t num_kernels = channels * in_height * in_width;
  const dim3 blocks((num_kernels + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);

  ResampleToMap2DVotingKernel<<<blocks, CUDA_NUM_THREADS>>>(
      num_kernels, data_in.data<int64_t>(), sample_map.data<int64_t>(),
      channels, in_height, in_width, out_height, out_width, numCandidates,
      data_out.data<int64_t>());
  CUDA_CHECK(cudaGetLastError())
}

// ----------------------------------------------
// ----------------------------------------------

template <typename T>
__global__ void ResampleFromUVMapsKernel(
    const int64_t n, const T *__restrict__ data_out_ptr,
    const int64_t *__restrict__ quad_idx_ptr,  // IH, IW
    const T *__restrict__ tex_uv_ptr,          // IH, IW, 2
    const int64_t channels, const int64_t num_textures,
    const int64_t tex_height, const int64_t tex_width, const int64_t in_height,
    const int64_t in_width, const int64_t interpolation,
    T *__restrict__ data_in_ptr) {
  int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n) { return; }

  common::ResampleFromUVMaps(index, data_out_ptr, quad_idx_ptr, tex_uv_ptr,
                             channels, num_textures, tex_height, tex_width,
                             in_height, in_width, interpolation, data_in_ptr);
}

void ResampleFromUVMapsLauncher(
    torch::Tensor data_out, torch::Tensor quad_idx, torch::Tensor tex_uv,
    const int64_t channels, const int64_t num_textures,
    const int64_t tex_height, const int64_t tex_width, const int64_t in_height,
    const int64_t in_width, const int64_t interpolation,
    torch::Tensor data_in) {
  const int64_t num_kernels = channels * in_height * in_width;
  const dim3 blocks((num_kernels + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);

  AT_DISPATCH_FLOATING_TYPES(
      data_out.scalar_type(), "ResampleFromUVMapsKernel", ([&] {
        ResampleFromUVMapsKernel<scalar_t><<<blocks, CUDA_NUM_THREADS>>>(
            num_kernels, data_out.data<scalar_t>(),
            quad_idx.data<int64_t>(), tex_uv.data<scalar_t>(),
            channels, num_textures, tex_height, tex_width, in_height, in_width,
            interpolation, data_in.data<scalar_t>());
      }));
  CUDA_CHECK(cudaGetLastError())
}

// ----------------------------------------------
// ----------------------------------------------

template <typename T>
__global__ void ResampleToUVMapsKernel(
    const int64_t n, const T *__restrict__ data_in_ptr,
    const int64_t *__restrict__ quad_idx_ptr,  // IH, IW
    const T *__restrict__ tex_uv_ptr,          // IH, IW, 2
    const int64_t channels, const int64_t num_textures,
    const int64_t tex_height, const int64_t tex_width, const int64_t in_height,
    const int64_t in_width, const int64_t interpolation,
    T *__restrict__ data_out_ptr) {
  int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n) { return; }

  common::ResampleToUVMaps(index, data_in_ptr, quad_idx_ptr, tex_uv_ptr,
                           channels, num_textures, tex_height, tex_width,
                           in_height, in_width, interpolation, data_out_ptr);
}

void ResampleToUVMapsLauncher(torch::Tensor data_in, torch::Tensor quad_idx,
                              torch::Tensor tex_uv, const int64_t channels,
                              const int64_t num_textures,
                              const int64_t tex_height,
                              const int64_t tex_width, const int64_t in_height,
                              const int64_t in_width,
                              const int64_t interpolation,
                              torch::Tensor data_out) {
  const int64_t num_kernels = channels * in_height * in_width;
  const dim3 blocks((num_kernels + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);

  AT_DISPATCH_FLOATING_TYPES(
      data_in.scalar_type(), "ResampleToUVMapsKernel", ([&] {
        ResampleToUVMapsKernel<scalar_t><<<blocks, CUDA_NUM_THREADS>>>(
            num_kernels, data_in.data<scalar_t>(),
            quad_idx.data<int64_t>(), tex_uv.data<scalar_t>(),
            channels, num_textures, tex_height, tex_width, in_height, in_width,
            interpolation, data_out.data<scalar_t>());
      }));
  CUDA_CHECK(cudaGetLastError())
}

}  // namespace cuda
}  // namespace nn
}  // namespace tangent_images
#endif