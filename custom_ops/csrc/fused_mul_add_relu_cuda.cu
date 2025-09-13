#include "registration.h"
#include <cuda.h>
#include <cuda_runtime.h>

namespace custom_ops {
namespace fused {

__global__ void mul_add_relu_kernel(
    const float* x,
    const float* weight,
    const float* bias,
    float* output,
    int64_t size
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float result = x[idx] * weight[idx] + bias[idx];
        output[idx] = fmaxf(0.0f, result);  // ReLU
    }
}

at::Tensor mul_add_relu_cuda(const at::Tensor& x, const at::Tensor& weight, const at::Tensor& bias) {
    TORCH_CHECK(x.device().is_cuda(), "Tensor 'x' must be on CUDA");
    TORCH_CHECK(weight.device().is_cuda(), "Tensor 'weight' must be on CUDA");
    TORCH_CHECK(bias.device().is_cuda(), "Tensor 'bias' must be on CUDA");
    TORCH_CHECK(x.sizes() == weight.sizes(), "Tensors 'x' and 'weight' must have the same size");
    TORCH_CHECK(x.sizes() == bias.sizes(), "Tensors 'x' and 'bias' must have the same size");
    TORCH_CHECK(x.dtype() == at::kFloat, "All tensors must be float32");
    TORCH_CHECK(weight.dtype() == at::kFloat, "All tensors must be float32");
    TORCH_CHECK(bias.dtype() == at::kFloat, "All tensors must be float32");
    
    auto output = torch::empty_like(x);
    const int64_t size = x.numel();
    
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    mul_add_relu_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        size
    );
    
    cudaError_t error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(error));
    
    return output;
}

}} // namespace custom_ops::fused