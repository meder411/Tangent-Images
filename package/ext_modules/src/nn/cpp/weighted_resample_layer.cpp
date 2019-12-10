#include "nn/layers/weighted_resample_layer.h"
#include "nn/cpp/resample.h"

namespace tangent_images {
namespace nn {
namespace cpu {

torch::Tensor WeightedResampleToMap(torch::Tensor input,
                                    torch::Tensor sample_map,
                                    torch::Tensor interp_weights,
                                    int outputHeight, int outputWidth,
                                    int interpolation) {
  // Useful dimensions to have
  const int64_t batchSize      = input.size(0);
  const int64_t channels       = input.size(1);
  const int64_t inputHeight    = input.size(2);
  const int64_t inputWidth     = input.size(3);
  const int64_t num_interp_pts = interp_weights.size(2);

  // Initialize output and index mask
  torch::Tensor output = torch::zeros(
      {batchSize, channels, outputHeight, outputWidth}, input.options());

  // Call the CUDA kernel once per batch
  for (int b = 0; b < batchSize; b++) {
    if (input.dtype() == torch::kDouble) {
      ResampleToMap2DWeighted<double>(
          channels * inputHeight * inputWidth, input[b], sample_map,
          interp_weights, channels, inputHeight, inputWidth, outputHeight,
          outputWidth, interpolation, num_interp_pts, output[b]);
    } else if (input.dtype() == torch::kFloat) {
      ResampleToMap2DWeighted<float>(
          channels * inputHeight * inputWidth, input[b], sample_map,
          interp_weights, channels, inputHeight, inputWidth, outputHeight,
          outputWidth, interpolation, num_interp_pts, output[b]);
    }
  }

  return output;
}

torch::Tensor WeightedResampleFromMap(torch::Tensor grad_output,
                                      torch::Tensor sample_map,
                                      torch::Tensor interp_weights,
                                      int interpolation) {
  // Useful dimensions to have
  const int64_t batchSize      = grad_output.size(0);
  const int64_t channels       = grad_output.size(1);
  const int64_t inputHeight    = sample_map.size(0);
  const int64_t inputWidth     = sample_map.size(1);
  const int64_t outputHeight   = grad_output.size(2);
  const int64_t outputWidth    = grad_output.size(3);
  const int64_t num_interp_pts = interp_weights.size(2);

  // Initialize output and index mask
  torch::Tensor input = torch::zeros(
      {batchSize, channels, inputHeight, inputWidth}, grad_output.options());

  // Call the CUDA kernel once per batch
  for (int b = 0; b < batchSize; b++) {
    if (grad_output.dtype() == torch::kDouble) {
      ResampleFromMap2DWeighted<double>(
          channels * inputHeight * inputWidth, grad_output[b], sample_map,
          interp_weights, channels, inputHeight, inputWidth, outputHeight,
          outputWidth, interpolation, num_interp_pts, input[b]);
    } else if (grad_output.dtype() == torch::kFloat) {
      ResampleFromMap2DWeighted<float>(
          channels * inputHeight * inputWidth, grad_output[b], sample_map,
          interp_weights, channels, inputHeight, inputWidth, outputHeight,
          outputWidth, interpolation, num_interp_pts, input[b]);
    }
  }

  return input;
}

}  // namespace cpu
}  // namespace nn
}  // namespace tangent_images