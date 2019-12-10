#ifndef CORE_INTERPOLATION_H_
#define CORE_INTERPOLATION_H_

#include <torch/extension.h>

#include <math.h>
#include <omp.h>

#include "core/util.h"

#include <cstdio>

namespace tangent_images {
namespace core {

template <typename T>
__host__ __device__ inline const T BilinearInterpolation(
    const T x, const T y, const int64_t x1, const int64_t x2, const int64_t y1,
    const int64_t y2, const T x1y1_data, const T x2y1_data, const T x1y2_data,
    const T x2y2_data) {
  // Perform the interpolation
  const T A    = y2 - y;
  const T B    = x2 - x;
  const T C    = x - x1;
  const T D    = y - y1;
  const T x1y1 = A * B * x1y1_data;
  const T x2y1 = A * C * x2y1_data;
  const T x1y2 = D * B * x1y2_data;
  const T x2y2 = D * C * x2y2_data;

  // Return the interpolated value
  return x1y1 + x2y1 + x1y2 + x2y2;
}

template <typename T>
__host__ __device__ inline void BilinearExtrapolation(
    const T value, const T x, const T y, const int64_t x1, const int64_t x2,
    const int64_t y1, const int64_t y2, T *x1y1_data_ptr,
    const bool x1y1_valid_ptr, T *x2y1_data_ptr, const bool x2y1_valid_ptr,
    T *x1y2_data_ptr, const bool x1y2_valid_ptr, T *x2y2_data_ptr,
    const bool x2y2_valid_ptr) {
  // Compute derivatives with respect to each source of interpolation
  // Atomic adds are necessary as we are potentially writing concurrently to
  // each source of interpolation
  const T A = y2 - y;
  const T B = x2 - x;
  const T C = x - x1;
  const T D = y - y1;
  if (x1y1_valid_ptr) { atomic_add(x1y1_data_ptr, A * B * value); }
  if (x2y1_valid_ptr) { atomic_add(x2y1_data_ptr, A * C * value); }
  if (x1y2_valid_ptr) { atomic_add(x1y2_data_ptr, D * B * value); }
  if (x2y2_valid_ptr) { atomic_add(x2y2_data_ptr, D * C * value); }
}

template <typename T>
__host__ __device__ inline const T Interpolate1DNearest(
    const T sample, const int64_t num_elements, const T *data_ptr) {
  // Find the nearest integer value
  int64_t sample_int = llround(double(sample));

  // Check for validity of pixels
  const bool sample_int_valid =
      sample_int >= 0 && sample_int <= num_elements - 1;

  // Get input data if pixel is valid
  const T data = sample_int_valid ? data_ptr[sample_int] : T(0);
  return data;
}

template <typename T>
__host__ __device__ inline void Extrapolate1DNearest(
    const T data, const T sample, const int64_t num_elements, T *target_ptr) {
  // Find the nearest integer value
  int64_t sample_int = llround(double(sample));

  // Check for validity of pixels
  const bool sample_int_valid =
      sample_int >= 0 && sample_int <= num_elements - 1;

  // "Uninterpolate" the data value
  if (sample_int_valid) { atomic_add(target_ptr + sample_int, data); }
}

template <typename T>
__host__ __device__ inline const T Interpolate2DNearest(const T x, const T y,
                                                        const int64_t height,
                                                        const int64_t width,
                                                        const T *data_ptr) {
  // Find the nearest integer value
  int64_t x_int = llround(double(x));
  int64_t y_int = llround(double(y));

  // Check for validity of pixels
  const bool x_int_valid = x_int >= 0 && x_int <= width - 1;
  const bool y_int_valid = y_int >= 0 && y_int <= height - 1;

  // Get input data if pixel is valid
  const T data =
      x_int_valid && y_int_valid ? data_ptr[y_int * width + x_int] : T(0);

  return data;
}

template <typename T>
__host__ __device__ inline void Extrapolate2DNearest(const T data, const T x,
                                                     const T y,
                                                     const int64_t height,
                                                     const int64_t width,
                                                     T *target_ptr) {
  // Find the nearest integer value
  int64_t x_int = llround(double(x));
  int64_t y_int = llround(double(y));

  // Check for validity of pixels
  const bool x_int_valid = x_int >= 0 && x_int <= width - 1;
  const bool y_int_valid = y_int >= 0 && y_int <= height - 1;

  // "Uninterpolate" the data value
  if (x_int_valid && y_int_valid) {
    atomic_add(target_ptr + y_int * width + x_int, data);
  }
}

template <typename T>
__host__ __device__ inline const T Interpolate2DBilinear(const T x, const T y,
                                                         const int64_t height,
                                                         const int64_t width,
                                                         const T *data_ptr) {
  // We assume pixels are always spaced 1 unit apart
  const int64_t x1 = std::floor(x);
  const int64_t y1 = std::floor(y);
  const int64_t x2 = x1 + 1;
  const int64_t y2 = y1 + 1;

  // Check for validity of pixels
  const bool x1_valid = x1 >= 0 && x1 <= width - 1;
  const bool x2_valid = x2 >= 0 && x2 <= width - 1;
  const bool y1_valid = y1 >= 0 && y1 <= height - 1;
  const bool y2_valid = y2 >= 0 && y2 <= height - 1;

  // Get input data from valid pixels
  const T x1y1_data =
      (x1_valid && y1_valid) ? data_ptr[y1 * width + x1] : T(0);
  const T x2y1_data =
      (x2_valid && y1_valid) ? data_ptr[y1 * width + x2] : T(0);
  const T x1y2_data =
      (x1_valid && y2_valid) ? data_ptr[y2 * width + x1] : T(0);
  const T x2y2_data =
      (x2_valid && y2_valid) ? data_ptr[y2 * width + x2] : T(0);

  // Perform the interpolation of the input data
  return BilinearInterpolation(x, y, x1, x2, y1, y2, x1y1_data, x2y1_data,
                               x1y2_data, x2y2_data);
}

template <typename T>
__host__ __device__ inline void Extrapolate2DBilinear(const T data, const T x,
                                                      const T y,
                                                      const int64_t height,
                                                      const int64_t width,
                                                      T *target_ptr) {
  // We assume pixels are always spaced 1 unit apart
  const int64_t x1 = std::floor(x);
  const int64_t y1 = std::floor(y);
  const int64_t x2 = x1 + 1;
  const int64_t y2 = y1 + 1;

  // Check for validity of pixels
  const bool x1_valid = x1 >= 0 && x1 <= width - 1;
  const bool x2_valid = x2 >= 0 && x2 <= width - 1;
  const bool y1_valid = y1 >= 0 && y1 <= height - 1;
  const bool y2_valid = y2 >= 0 && y2 <= height - 1;

  // "Uninterpolate" the data value
  BilinearExtrapolation(data, x, y, x1, x2, y1, y2,
                        target_ptr + y1 * width + x1, x1_valid && y1_valid,
                        target_ptr + y1 * width + x2, x2_valid && y1_valid,
                        target_ptr + y2 * width + x1, x1_valid && y2_valid,
                        target_ptr + y2 * width + x2, x2_valid && y2_valid);
}

template <typename T>
__host__ __device__ inline const T Interpolate2DBispherical(
    const T x, const T y, const int64_t height, const int64_t width,
    const T *data_ptr) {
  // We assume pixels are always spaced 1 unit apart
  const int64_t x1 = std::floor(x);
  const int64_t y1 = std::floor(y);
  const int64_t x2 = x1 + 1;
  const int64_t y2 = y1 + 1;

  // Let the x-axis wrap
  const int64_t x1_wrap = negmod(x1, width);
  const int64_t x2_wrap = negmod(x2, width);

  // Check for validity of pixels
  // All x values are valid
  const bool y1_valid = y1 >= 0 && y1 <= height - 1;
  const bool y2_valid = y2 >= 0 && y2 <= height - 1;

  // Get input data from valid pixels
  const T x1y1_data = y1_valid ? data_ptr[y1 * width + x1_wrap] : T(0);
  const T x2y1_data = y1_valid ? data_ptr[y1 * width + x2_wrap] : T(0);
  const T x1y2_data = y2_valid ? data_ptr[y2 * width + x1_wrap] : T(0);
  const T x2y2_data = y2_valid ? data_ptr[y2 * width + x2_wrap] : T(0);

  // Perform the interpolation of the input data
  return BilinearInterpolation(x, y, x1, x2, y1, y2, x1y1_data, x2y1_data,
                               x1y2_data, x2y2_data);
}

template <typename T>
__host__ __device__ inline void Extrapolate2DBispherical(const T data,
                                                         const T x, const T y,
                                                         const int64_t height,
                                                         const int64_t width,
                                                         T *target_ptr) {
  // We assume pixels are always spaced 1 unit apart
  const int64_t x1 = std::floor(x);
  const int64_t y1 = std::floor(y);
  const int64_t x2 = x1 + 1;
  const int64_t y2 = y1 + 1;

  // Let the x-axis wrap
  const int64_t x1_wrap = negmod(x1, width);
  const int64_t x2_wrap = negmod(x2, width);

  // Check for validity of pixels
  // All x values are valid
  const bool y1_valid = y1 >= 0 && y1 <= height - 1;
  const bool y2_valid = y2 >= 0 && y2 <= height - 1;

  // "Uninterpolate" the data value
  BilinearExtrapolation(data, x, y, x1, x2, y1, y2,
                        target_ptr + y1 * width + x1_wrap, y1_valid,
                        target_ptr + y1 * width + x2_wrap, y1_valid,
                        target_ptr + y2 * width + x1_wrap, y2_valid,
                        target_ptr + y2 * width + x2_wrap, y2_valid);
}

__host__ __device__ inline void Vote2D(const int64_t data, const int64_t x,
                                       const int64_t y, const int64_t height,
                                       const int64_t width,
                                       const int64_t numCandidates,
                                       int64_t *target_ptr) {
  // Check for validity of pixels
  const bool x_valid = x >= 0 && x <= width - 1;
  const bool y_valid = y >= 0 && y <= height - 1;

  // If the pixel location is valid, vote
  if (x_valid && y_valid) {
    atomic_add(
        reinterpret_cast<unsigned long long int *>(
            target_ptr + y * width * numCandidates + x * numCandidates + data),
        static_cast<unsigned long long int>(1));
  }
}

}  // namespace core
}  // namespace tangent_images
#endif
