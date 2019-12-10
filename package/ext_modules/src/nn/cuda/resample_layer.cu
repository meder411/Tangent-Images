#include <torch/extension.h>

#include "nn/cuda/resample.cuh"

namespace tangent_images {
namespace nn {
namespace cuda {

torch::Tensor ResampleToMap(torch::Tensor input, torch::Tensor output_map,
                            int outputHeight, int outputWidth,
                            int interpolation) {
  // Useful dimensions to have
  const int64_t batchSize   = input.size(0);
  const int64_t channels    = input.size(1);
  const int64_t inputHeight = input.size(2);
  const int64_t inputWidth  = input.size(3);

  // Initialize output and index mask
  torch::Tensor output = torch::zeros(
      {batchSize, channels, outputHeight, outputWidth}, input.options());

  // Call the CUDA kernel once per batch
  for (int b = 0; b < batchSize; b++) {
    ResampleToMap2DLauncher(input[b], output_map, channels, inputHeight,
                            inputWidth, outputHeight, outputWidth,
                            interpolation, output[b]);
  }

  return output;
}

torch::Tensor ResampleFromMap(torch::Tensor grad_output,
                              torch::Tensor output_map, int interpolation) {
  // Useful dimensions to have
  const int64_t batchSize    = grad_output.size(0);
  const int64_t channels     = grad_output.size(1);
  const int64_t inputHeight  = output_map.size(0);
  const int64_t inputWidth   = output_map.size(1);
  const int64_t outputHeight = grad_output.size(2);
  const int64_t outputWidth  = grad_output.size(3);

  // Initialize output and index mask
  torch::Tensor input = torch::zeros(
      {batchSize, channels, inputHeight, inputWidth}, grad_output.options());

  // Call the CUDA kernel once per batch
  for (int b = 0; b < batchSize; b++) {
    ResampleFromMap2DLauncher(grad_output[b], output_map, channels,
                              inputHeight, inputWidth, outputHeight,
                              outputWidth, interpolation, input[b]);
  }

  return input;
}

}  // namespace cuda
}  // namespace nn
}  // namespace tangent_images