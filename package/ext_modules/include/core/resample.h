#ifndef CORE_RESAMPLE_H_
#define CORE_RESAMPLE_H_

#include "core/interpolation.h"
#include "core/util.h"

namespace tangent_images {
namespace core {

template <typename T>
__host__ __device__ void ResampleToMap1D(const T data, const T *sample_map_ptr,
                                         const int64_t num_elements,
                                         T *data_out_ptr) {
  // Find the (fractional) location in the input image from which the mapped
  // convolution sampled at a given index of the kernel
  const T sample = *sample_map_ptr;

  // Perform the uninterpolation (includes boundary checks)
  Extrapolate1DNearest(data, sample, num_elements, data_out_ptr);
}

template <typename T>
__host__ __device__ const T ResampleFromMap1D(const T *data_ptr,
                                              const T *sample_map_ptr,
                                              const int64_t num_elements) {
  // Location given by the sample map
  const T sample = *sample_map_ptr;

  // Interpolate the image (handles boundary checks)
  return Interpolate1DNearest(sample, num_elements, data_ptr);
}

template <typename T>
__host__ __device__ void ResampleToMap2D(const T data, const T *sample_map_ptr,
                                         const int64_t height_im,
                                         const int64_t width_im,
                                         const int64_t interpolation,
                                         T *data_out_ptr) {
  // Find the (fractional) location in the input image from which the mapped
  // convolution sampled at a given index of the kernel
  const T x = *sample_map_ptr;
  const T y = *(sample_map_ptr + 1);

  // Perform the uninterpolation (includes boundary checks)
  switch (interpolation) {
    default:
    case 0:
      Extrapolate2DNearest(data, x, y, height_im, width_im, data_out_ptr);
      break;

    case 1:
      Extrapolate2DBilinear(data, x, y, height_im, width_im, data_out_ptr);
      break;

    case 2:
      Extrapolate2DBispherical(data, x, y, height_im, width_im, data_out_ptr);
      break;
  }
}

template <typename T>
__host__ __device__ const T ResampleFromMap2D(const T *data_ptr,
                                              const T *sample_map_ptr,
                                              const int64_t height_im,
                                              const int64_t width_im,
                                              const int64_t interpolation) {
  // Output image location given by the output map
  const T x = *sample_map_ptr;
  const T y = *(sample_map_ptr + 1);

  // Interpolate the image (handles boundary checks)
  T interpolated = 0;
  switch (interpolation) {
    default:
    case 0:
      interpolated = Interpolate2DNearest(x, y, height_im, width_im, data_ptr);
      break;

    case 1:
      interpolated =
          Interpolate2DBilinear(x, y, height_im, width_im, data_ptr);
      break;

    case 2:
      interpolated =
          Interpolate2DBispherical(x, y, height_im, width_im, data_ptr);
      break;
  }
  return interpolated;
}

template <typename T>
__host__ __device__ void ResampleToMap1DWeighted(
    const T data,
    const T *sample_map_ptr,      // * x K x P x 1
    const T *interp_weights_ptr,  // * x K x P
    const int64_t num_interp_pts, const int64_t num_elements,
    T *data_out_ptr) {
  // Go through each point
  for (int i = 0; i < num_interp_pts; i++) {
    // Weight
    const T w = *(interp_weights_ptr + i);

    ResampleToMap1D(w * data, sample_map_ptr + i, num_elements, data_out_ptr);
  }
}

template <typename T>
__host__ __device__ const T ResampleFromMap1DWeighted(
    const T *data_ptr, const T *sample_map_ptr, const T *interp_weights_ptr,
    const int64_t sample_map_idx, const int64_t interp_weights_idx,
    const int64_t num_interp_pts, const int64_t num_elements) {
  T data = T(0);

  // Go through each point
  for (int i = 0; i < num_interp_pts; i++) {
    // Weight
    const T w = *(interp_weights_ptr + i);

    // Accumulate the weighted data
    data += w * ResampleFromMap1D(data_ptr, sample_map_ptr + i, num_elements);
  }

  return data;
}

template <typename T>
__host__ __device__ void ResampleToMap2DWeighted(
    const T data,
    const T *sample_map_ptr,      // OH x OW x K x P x 2
    const T *interp_weights_ptr,  // OH x OW x K x P
    const int64_t num_interp_pts, const int64_t interpolation,
    const int64_t height_im, const int64_t width_im, T *data_out_ptr) {
  // Go through each point
  for (int i = 0; i < num_interp_pts; i++) {
    // Weight
    const T w = *(interp_weights_ptr + i);

    ResampleToMap2D(w * data, sample_map_ptr + 2 * i, height_im, width_im,
                    interpolation, data_out_ptr);
  }
}

template <typename T>
__host__ __device__ const T ResampleFromMap2DWeighted(
    const T *data_ptr, const T *sample_map_ptr, const T *interp_weights_ptr,
    const int64_t num_interp_pts, const int64_t interpolation,
    const int64_t height_im, const int64_t width_im) {
  T data = T(0);

  // Go through each point
  for (int i = 0; i < num_interp_pts; i++) {
    // Weight
    const T w = *(interp_weights_ptr + i);

    // Accumulate weighted data
    data += w * ResampleFromMap2D(data_ptr, sample_map_ptr + 2 * i, height_im,
                                  width_im, interpolation);
  }

  return data;
}

template <typename T = int64_t, typename enabled = typename std::enable_if<
                                    std::is_same<T, int64_t>::value>::type>
__host__ __device__ void ResampleToMap2DVoting(
    const T data,
    const T *sample_map_ptr,  // OH x OW x K x 2
    const int64_t numCandidates, const int64_t height_im,
    const int64_t width_im, T *data_out_ptr) {
  Vote2D(data, *sample_map_ptr, *(sample_map_ptr + 1), height_im, width_im,
         numCandidates, data_out_ptr);
}
}  // namespace core
}  // namespace tangent_images
#endif