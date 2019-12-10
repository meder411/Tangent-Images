#ifndef COMMON_RESAMPLE_CUH_
#define COMMON_RESAMPLE_CUH_

#include "core/resample.h"

namespace tangent_images {
namespace nn {
namespace common {

template <typename T>
__host__ __device__ void ResampleToMap2D(
    const int64_t index, const T *data_in_ptr, const T *sample_map_ptr,
    const int64_t channels, const int64_t in_height, const int64_t in_width,
    const int64_t out_height, const int64_t out_width,
    const int64_t interpolation, T *data_out_ptr) {
  // Locations
  const int64_t x_in = index % in_width;
  const int64_t y_in = (index / in_width) % in_height;
  const int64_t c    = (index / in_width / in_height) % channels;

  // Output image location given by the map
  const int64_t sample_map_idx = 2 * (y_in * in_width + x_in);

  // Data to uninterpolate
  const T data = data_in_ptr[index];

  core::ResampleToMap2D(data, sample_map_ptr + sample_map_idx, out_height,
                        out_width, interpolation,
                        data_out_ptr + c * out_height * out_width);
}

template <typename T>
__host__ __device__ void ResampleFromMap2D(
    const int64_t index, const T *data_out_ptr,
    const T *sample_map_ptr,  // IH, IW, 2
    const int64_t channels, const int64_t in_height, const int64_t in_width,
    const int64_t out_height, const int64_t out_width,
    const int64_t interpolation, T *data_in_ptr) {
  // Locations
  const int64_t x_in = index % in_width;
  const int64_t y_in = (index / in_width) % in_height;
  const int64_t c    = (index / in_width / in_height) % channels;

  // Output image location given by the output map
  const int64_t sample_map_idx = 2 * (y_in * in_width + x_in);

  data_in_ptr[index] = core::ResampleFromMap2D(
      data_out_ptr + c * out_height * out_width,
      sample_map_ptr + sample_map_idx, out_height, out_width, interpolation);
}

template <typename T>
__host__ __device__ void ResampleToMap2DWeighted(
    const int64_t index, const T *data_in_ptr,
    const T *sample_map_ptr,      // IH, IW, P, 2
    const T *interp_weights_ptr,  // IH, IW, P
    const int64_t channels, const int64_t in_height, const int64_t in_width,
    const int64_t out_height, const int64_t out_width,
    const int64_t interpolation, const int64_t num_interp_pts,
    T *data_out_ptr) {
  // Locations
  const int64_t x_in = index % in_width;
  const int64_t y_in = (index / in_width) % in_height;
  const int64_t c    = (index / in_width / in_height) % channels;

  // Output image location given by the map
  const int64_t sample_map_idx =
      y_in * in_width * num_interp_pts * 2 + x_in * num_interp_pts * 2;
  const int64_t interp_weights_idx =
      y_in * in_width * num_interp_pts + x_in * num_interp_pts;

  // Data to uninterpolate
  const T data = data_in_ptr[index];

  core::ResampleToMap2DWeighted(
      data, sample_map_ptr + sample_map_idx,
      interp_weights_ptr + interp_weights_idx, num_interp_pts, interpolation,
      out_height, out_width, data_out_ptr + c * out_height * out_width);
}

template <typename T>
__host__ __device__ void ResampleFromMap2DWeighted(
    const int64_t index, const T *data_out_ptr, const T *sample_map_ptr,
    const T *interp_weights_ptr, const int64_t channels,
    const int64_t in_height, const int64_t in_width, const int64_t out_height,
    const int64_t out_width, const int64_t interpolation,
    const int64_t num_interp_pts, T *data_in_ptr) {
  // Locations
  const int64_t x_in = index % in_width;
  const int64_t y_in = (index / in_width) % in_height;
  const int64_t c    = (index / in_width / in_height) % channels;

  // Output image location given by the output map
  const int64_t sample_map_idx =
      y_in * in_width * num_interp_pts * 2 + x_in * num_interp_pts * 2;
  const int64_t interp_weights_idx =
      y_in * in_width * num_interp_pts + x_in * num_interp_pts;

  data_in_ptr[index] = core::ResampleFromMap2DWeighted(
      data_out_ptr + c * out_height * out_width,
      sample_map_ptr + sample_map_idx, interp_weights_ptr + interp_weights_idx,
      num_interp_pts, interpolation, out_height, out_width);
}

template <typename T = int64_t, typename enabled = typename std::enable_if<
                                    std::is_same<T, int64_t>::value>::type>
__host__ __device__ void ResampleToMap2DVoting(
    const int64_t index, const T *data_in_ptr,
    const T *sample_map_ptr,  // IH, IW, 2
    const int64_t channels, const int64_t in_height, const int64_t in_width,
    const int64_t out_height, const int64_t out_width,
    const int64_t num_candidates, T *data_out_ptr) {
  // Locations
  const int64_t x_in = index % in_width;
  const int64_t y_in = (index / in_width) % in_height;
  const int64_t c    = (index / in_width / in_height) % channels;

  // Output image location given by the map
  const int64_t sample_map_idx = y_in * in_width * 2 + x_in * 2;
  const int64_t data           = data_in_ptr[index];

  core::ResampleToMap2DVoting(data, sample_map_ptr + sample_map_idx,
                              num_candidates, out_height, out_width,
                              data_out_ptr + c * out_height * out_width);
}

template <typename T>
__host__ __device__ void ResampleFromUVMaps(
    const int64_t index, const T *data_out_ptr,
    const int64_t *quad_idx_ptr,  // IH, IW
    const T *tex_uv_ptr,          // IH, IW, 2 (pixel coords, not normalized)
    const int64_t channels, const int64_t num_textures,
    const int64_t tex_height, const int64_t tex_width, const int64_t in_height,
    const int64_t in_width, const int64_t interpolation, T *data_in_ptr) {
  // Locations
  const int64_t x_in     = index % in_width;
  const int64_t y_in     = (index / in_width) % in_height;
  const int64_t c        = (index / in_width / in_height) % channels;
  const int64_t quad_idx = quad_idx_ptr[y_in * in_width + x_in];

  // Isolate patch
  const T *cur_texture_ptr = data_out_ptr +
                             c * num_textures * tex_height * tex_width +
                             quad_idx * tex_height * tex_width;

  // Resample with interpolation
  data_in_ptr[index] = core::ResampleFromMap2D(
      cur_texture_ptr, tex_uv_ptr + y_in * in_width * 2 + x_in * 2, tex_height,
      tex_width, interpolation);
}

template <typename T>
__host__ __device__ void ResampleToUVMaps(
    const int64_t index, const T *data_in_ptr,
    const int64_t *quad_idx_ptr,  // IH, IW
    const T *tex_uv_ptr,          // IH, IW, 2 (pixel coords, not normalized)
    const int64_t channels, const int64_t num_textures,
    const int64_t tex_height, const int64_t tex_width, const int64_t in_height,
    const int64_t in_width, const int64_t interpolation, T *data_out_ptr) {
  // Locations
  const int64_t x_in     = index % in_width;
  const int64_t y_in     = (index / in_width) % in_height;
  const int64_t c        = (index / in_width / in_height) % channels;
  const int64_t quad_idx = quad_idx_ptr[y_in * in_width + x_in];

  // Isolate patch
  T *cur_texture_ptr = data_out_ptr +
                       c * num_textures * tex_height * tex_width +
                       quad_idx * tex_height * tex_width;

  // Data to uninterpolate
  const T data = data_in_ptr[index];

  core::ResampleToMap2D(data, tex_uv_ptr + y_in * in_width * 2 + x_in * 2,
                        tex_height, tex_width, interpolation, cur_texture_ptr);
}

}  // namespace common
}  // namespace nn
}  // namespace tangent_images
#endif