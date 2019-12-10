#ifndef CORE_CONVERSIONS_H_
#define CORE_CONVERSIONS_H_

#include <math.h>
#include <algorithm>

#include "core/util.h"

namespace tangent_images {
namespace core {

// Returns lat/lon both in range [-pi, pi]
template <typename T>
__host__ __device__ void XYZToSpherical(const T x, const T y, const T z,
                                        T &lon, T &lat) {
  lat = atan2(-y, sqrt(x * x + z * z));
  lon = atan2(x, z);
}

template <typename T>
__host__ __device__ void SphericalToXYZ(const T lon, const T lat, T &x, T &y,
                                        T &z) {
  x = cos(lat) * sin(lon);
  y = -sin(lat);
  z = cos(lat) * cos(lon);
}

template <typename T>
__host__ __device__ const T XToLongitude(const int64_t x,
                                         const int64_t width) {
  return 2 * M_PI * (T(x) - T(width) / 2) / T(width);
}

template <typename T>
__host__ __device__ const T YToLatitude(const int64_t y,
                                        const int64_t height) {
  return M_PI * (T(y) - (T(height) - 1) / 2) / (T(height) - 1);
}

// Outputs in range [0,1]
template <typename T>
__host__ __device__ void ConvertXYZToEquirectangular(const int64_t height,
                                                     const int64_t width,
                                                     const T x, const T y,
                                                     const T z, T &u, T &v) {
  // Convert XYZ to lon/lat
  T lon, lat;
  XYZToSpherical(x, y, z, lon, lat);
  // Convert lat/lon to locations on an equirectangular image
#ifdef __CUDACC__
  const int64_t size = max(height, width);
#else
  const int64_t size = std::max(height, width);
#endif

  lon /= (2 * M_PI);
  lat /= (-2 * M_PI);
  u = lon * T(size) + T(width) / T(2);
  v = lat * T(size) + T(height) / T(2);
}

// From https://en.wikipedia.org/wiki/Cube_mapping
// Minor change is to flip the origin to the top-left of each cube
// Also change the index to [-z, -x, +z, +x, +y, -y]
template <typename T>
__host__ __device__ void ConvertXYZToCubeMap(const T x, const T y, const T z,
                                             int64_t &index, T &u, T &v) {
  const T abs_x = fabs(x);
  const T abs_y = fabs(y);
  const T abs_z = fabs(z);

  const bool is_x_positive = x > 0;
  const bool is_y_positive = y > 0;
  const bool is_z_positive = z > 0;

  T max_axis, uc, vc;

  // POSITIVE X
  if (is_x_positive && abs_x >= abs_y && abs_x >= abs_z) {
    // u (0 to 1) goes from +z to -z
    // v (0 to 1) goes from +y to -y
    max_axis = abs_x;
    uc       = -z;
    vc       = -y;
    index    = 3;
  }
  // NEGATIVE X
  if (!is_x_positive && abs_x >= abs_y && abs_x >= abs_z) {
    // u (0 to 1) goes from -z to +z
    // v (0 to 1) goes from +y to -y
    max_axis = abs_x;
    uc       = z;
    vc       = -y;
    index    = 1;
  }
  // POSITIVE Y
  if (is_y_positive && abs_y >= abs_x && abs_y >= abs_z) {
    // u (0 to 1) goes from -x to +x
    // v (0 to 1) goes from -z to +z
    max_axis = abs_y;
    uc       = x;
    vc       = z;
    index    = 4;
  }
  // NEGATIVE Y
  if (!is_y_positive && abs_y >= abs_x && abs_y >= abs_z) {
    // u (0 to 1) goes from -x to +x
    // v (0 to 1) goes from +z to -z
    max_axis = abs_y;
    uc       = x;
    vc       = -z;
    index    = 5;
  }
  // POSITIVE Z
  if (is_z_positive && abs_z >= abs_x && abs_z >= abs_y) {
    // u (0 to 1) goes from -x to +x
    // v (0 to 1) goes from +y to -y
    max_axis = abs_z;
    uc       = x;
    vc       = -y;
    index    = 2;
  }
  // NEGATIVE Z
  if (!is_z_positive && abs_z >= abs_x && abs_z >= abs_y) {
    // u (0 to 1) goes from +x to -x
    // v (0 to 1) goes from +y to -y
    max_axis = abs_z;
    uc       = -x;
    vc       = -y;
    index    = 0;
  }

  // Convert range from -1 to 1 to 0 to 1
  u = T(0.5) * (uc / max_axis + T(1));
  v = T(0.5) * (vc / max_axis + T(1));
}

// Assumes that pixel centers are (u + 0.5, v + 0.5) and uv in [0,1]
// (Before calling this, do uv = (uv + 0.5) / cube_dim)
// Index order assumes to [-z, -x, +z, +x, +y, -y]
// uv(0,0) is top left
template <typename T>
__host__ __device__ void ConvertCubeMapToXYZ(const T u, const T v,
                                             const int64_t &index, T &x, T &y,
                                             T &z) {
  // Convert from [0,1] range to [-1,1]
  T uc = 2 * u - 1;
  T vc = 2 * v - 1;

  switch (index) {
    // NEGATIVE Z
    case 0:
      x = -u;
      y = -v;
      z = T(-1);
      break;
    // NEGATIVE X
    case 1:
      x = T(-1);
      y = -v;
      z = u;
      break;
    // POSITIVE Z
    case 2:
      x = u;
      y = -v;
      z = T(1);
      break;
    // POSITIVE X
    case 3:
      x = T(1);
      y = -v;
      z = -u;
      break;
    // POSITIVE Y
    case 4:
      x = u;
      ;
      y = T(1);
      z = v;
      break;
    // NEGATIVE Y
    case 5:
      x = u;
      y = T(-1);
      z = -v;
      break;
  }
}

}  // namespace core
}  // namespace tangent_images
#endif