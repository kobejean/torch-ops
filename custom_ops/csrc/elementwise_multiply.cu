#include "registration.h"
#include <cuda.h>
#include <cuda_runtime.h>

namespace custom_ops {
namespace elementwise {

__global__ void multiply_kernel(
    const float* a, 
    const float* b, 
    float* result, 
    int64_t size
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] * b[idx];
    }
}

at::Tensor multiply_cuda(const at::Tensor& a, const at::Tensor& b) {
    utils::check_cuda_tensors({a, b});
    utils::check_same_size(a, b);
    TORCH_CHECK(a.dtype() == at::kFloat, "Tensors must be float32");
    TORCH_CHECK(b.dtype() == at::kFloat, "Tensors must be float32");
    
    auto result = torch::empty_like(a);
    const int64_t size = a.numel();
    
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    multiply_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        result.data_ptr<float>(),
        size
    );
    
    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(error));
    
    return result;
}

}} // namespace custom_ops::elementwise