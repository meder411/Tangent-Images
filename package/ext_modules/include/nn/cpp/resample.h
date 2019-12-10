#ifndef RESAMPLE_H_
#define RESAMPLE_H_

#include <omp.h>
#include <torch/extension.h>

#include "nn/common/resample.h"

namespace tangent_images {
namespace nn {
namespace cpu {

template <typename T>
void ResampleToMap2D(const int64_t num_kernels, torch::Tensor data_in,
                     torch::Tensor sample_map, const int64_t channels,
                     const int64_t in_height, const int64_t in_width,
                     const int64_t out_height, const int64_t out_width,
                     const int64_t interpolation, torch::Tensor data_out) {
  const T *data_in_ptr    = data_in.data<T>();
  const T *sample_map_ptr = sample_map.data<T>();
  T *data_out_ptr         = data_out.data<T>();
  int64_t index;
#pragma omp parallel for shared(data_in_ptr, sample_map_ptr, \
                                data_out_ptr) private(index) schedule(static)
  for (index = 0; index < num_kernels; index++) {
    common::ResampleToMap2D(index, data_in_ptr, sample_map_ptr, channels,
                            in_height, in_width, out_height, out_width,
                            interpolation, data_out_ptr);
  }
}

template <typename T>
void ResampleFromMap2D(const int64_t num_kernels, torch::Tensor data_out,
                       torch::Tensor sample_map, const int64_t channels,
                       const int64_t in_height, const int64_t in_width,
                       const int64_t out_height, const int64_t out_width,
                       const int64_t interpolation, torch::Tensor data_in) {
  const T *data_out_ptr   = data_out.data<T>();
  const T *sample_map_ptr = sample_map.data<T>();
  T *data_in_ptr          = data_in.data<T>();
  int64_t index;
#pragma omp parallel for shared(data_in_ptr, sample_map_ptr, \
                                data_out_ptr) private(index) schedule(static)
  for (index = 0; index < num_kernels; index++) {
    common::ResampleFromMap2D(index, data_out_ptr, sample_map_ptr, channels,
                              in_height, in_width, out_height, out_width,
                              interpolation, data_in_ptr);
  }
}

// --------------------------------------------
// --------------------------------------------

template <typename T>
void ResampleToMap2DWeighted(
    const int64_t num_kernels, torch::Tensor data_in, torch::Tensor sample_map,
    torch::Tensor interp_weights, const int64_t channels,
    const int64_t in_height, const int64_t in_width, const int64_t out_height,
    const int64_t out_width, const int64_t interpolation,
    const int64_t num_interp_pts, torch::Tensor data_out) {
  const T *data_in_ptr        = data_in.data<T>();
  const T *sample_map_ptr     = sample_map.data<T>();
  const T *interp_weights_ptr = interp_weights.data<T>();
  T *data_out_ptr             = data_out.data<T>();
  int64_t index;
#pragma omp parallel for shared(data_in_ptr, sample_map_ptr, \
                                interp_weights_ptr,          \
                                data_out_ptr) private(index) schedule(static)
  for (index = 0; index < num_kernels; index++) {
    common::ResampleToMap2DWeighted(
        index, data_in_ptr, sample_map_ptr, interp_weights_ptr, channels,
        in_height, in_width, out_height, out_width, interpolation,
        num_interp_pts, data_out_ptr);
  }
}

template <typename T>
void ResampleFromMap2DWeighted(
    const int64_t num_kernels, torch::Tensor data_out,
    torch::Tensor sample_map, torch::Tensor interp_weights,
    const int64_t channels, const int64_t in_height, const int64_t in_width,
    const int64_t out_height, const int64_t out_width,
    const int64_t interpolation, const int64_t num_interp_pts,
    torch::Tensor data_in) {
  const T *data_out_ptr       = data_out.data<T>();
  const T *sample_map_ptr     = sample_map.data<T>();
  const T *interp_weights_ptr = interp_weights.data<T>();
  T *data_in_ptr              = data_in.data<T>();
  int64_t index;
#pragma omp parallel for shared(data_in_ptr, sample_map_ptr, \
                                interp_weights_ptr,          \
                                data_out_ptr) private(index) schedule(static)
  for (index = 0; index < num_kernels; index++) {
    common::ResampleFromMap2DWeighted(
        index, data_out_ptr, sample_map_ptr, interp_weights_ptr, channels,
        in_height, in_width, out_height, out_width, interpolation,
        num_interp_pts, data_in_ptr);
  }
}

// --------------------------------------------
// --------------------------------------------

template <typename T>
void ResampleToMap2DVoting(const int64_t num_kernels, torch::Tensor data_in,
                           torch::Tensor sample_map, const int64_t channels,
                           const int64_t in_height, const int64_t in_width,
                           const int64_t out_height, const int64_t out_width,
                           const int64_t numCandidates,
                           torch::Tensor data_out) {
  const int64_t *data_in_ptr    = data_in.data<int64_t>();
  const int64_t *sample_map_ptr = sample_map.data<int64_t>();
  int64_t *data_out_ptr         = data_out.data<int64_t>();
  int64_t index;
#pragma omp parallel for shared(data_in_ptr, sample_map_ptr, \
                                data_out_ptr) private(index) schedule(static)
  for (index = 0; index < num_kernels; index++) {
    common::ResampleToMap2DVoting(index, data_in_ptr, sample_map_ptr, channels,
                                  in_height, in_width, out_height, out_width,
                                  numCandidates, data_out_ptr);
  }
}

template <typename T>
void ResampleFromUVMaps(const int64_t num_kernels, torch::Tensor data_out,
                        torch::Tensor quad_idx, torch::Tensor tex_uv,
                        const int64_t channels, const int64_t num_textures,
                        const int64_t tex_height, const int64_t tex_width,
                        const int64_t in_height, const int64_t in_width,
                        const int64_t interpolation, torch::Tensor data_in) {
  const T *data_out_ptr       = data_out.data<T>();
  const int64_t *quad_idx_ptr = quad_idx.data<int64_t>();
  const T *tex_uv_ptr         = tex_uv.data<T>();
  T *data_in_ptr              = data_in.data<T>();
  int64_t index;
#pragma omp parallel for shared(data_in_ptr, quad_idx_ptr, tex_uv_ptr, \
                                data_out_ptr) private(index) schedule(static)
  for (index = 0; index < num_kernels; index++) {
    common::ResampleFromUVMaps(index, data_out_ptr, quad_idx_ptr, tex_uv_ptr,
                               channels, num_textures, tex_height, tex_width,
                               in_height, in_width, interpolation,
                               data_in_ptr);
  }
}

template <typename T>
void ResampleToUVMaps(const int64_t num_kernels, torch::Tensor data_in,
                      torch::Tensor quad_idx, torch::Tensor tex_uv,
                      const int64_t channels, const int64_t num_textures,
                      const int64_t tex_height, const int64_t tex_width,
                      const int64_t in_height, const int64_t in_width,
                      const int64_t interpolation, torch::Tensor data_out) {
  const T *data_in_ptr        = data_in.data<T>();
  const int64_t *quad_idx_ptr = quad_idx.data<int64_t>();
  const T *tex_uv_ptr         = tex_uv.data<T>();
  T *data_out_ptr             = data_out.data<T>();
  int64_t index;
#pragma omp parallel for shared(data_out_ptr, quad_idx_ptr, tex_uv_ptr, \
                                data_in_ptr) private(index) schedule(static)
  for (index = 0; index < num_kernels; index++) {
    common::ResampleToUVMaps(index, data_in_ptr, quad_idx_ptr, tex_uv_ptr,
                             channels, num_textures, tex_height, tex_width,
                             in_height, in_width, interpolation, data_out_ptr);
  }
}

}  // namespace cpu
}  // namespace nn
}  // namespace tangent_images
#endif