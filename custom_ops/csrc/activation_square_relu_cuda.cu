#include "registration.h"
#include <cuda.h>
#include <cuda_runtime.h>

namespace custom_ops {
namespace activation {

__global__ void square_relu_forward_kernel(
    const float* input,
    float* output,
    int64_t size
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        output[idx] = x > 0 ? x * x : 0.0f;
    }
}

__global__ void square_relu_backward_kernel(
    const float* grad_output,
    const float* input,
    float* grad_input,
    int64_t size
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        grad_input[idx] = x > 0 ? 2.0f * x * grad_output[idx] : 0.0f;
    }
}

at::Tensor square_relu_forward_cuda(const at::Tensor& input) {
    TORCH_CHECK(input.device().is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(input.dtype() == at::kFloat, "Input must be float32");
    
    auto result = torch::empty_like(input);
    const int64_t size = input.numel();
    
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    square_relu_forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        result.data_ptr<float>(),
        size
    );
    
    cudaError_t error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(error));
    
    return result;
}

at::Tensor square_relu_backward_cuda(const at::Tensor& grad_output, const at::Tensor& input) {
    TORCH_CHECK(grad_output.device().is_cuda(), "grad_output tensor must be on CUDA");
    TORCH_CHECK(input.device().is_cuda(), "input tensor must be on CUDA");
    TORCH_CHECK(grad_output.sizes() == input.sizes(), "grad_output and input must have the same size");
    TORCH_CHECK(grad_output.dtype() == at::kFloat, "grad_output must be float32");
    TORCH_CHECK(input.dtype() == at::kFloat, "input must be float32");
    
    auto grad_input = torch::empty_like(input);
    const int64_t size = input.numel();
    
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    square_relu_backward_kernel<<<blocks, threads>>>(
        grad_output.data_ptr<float>(),
        input.data_ptr<float>(),
        grad_input.data_ptr<float>(),
        size
    );
    
    cudaError_t error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(error));
    
    return grad_input;
}

}} // namespace custom_ops::activation