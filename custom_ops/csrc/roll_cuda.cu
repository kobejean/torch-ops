#include "registration.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

namespace custom_ops {
namespace tensor {

template <typename scalar_t>
__global__ void roll_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t num_elements,
    const int effective_dims,
    const int* __restrict__ sizes,
    const int* __restrict__ strides,
    const int* __restrict__ shifts
) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_elements) return;
    
    // Compute offset between input and output positions
    int64_t offset = 0;
    
    for (int d = 0; d < effective_dims; ++d) {
        const int dim_idx = (idx / strides[d]) % sizes[d];
        const int shifted_idx = (dim_idx + shifts[d]) % sizes[d];
        offset += (shifted_idx - dim_idx) * strides[d];
    }
    
    output[idx + offset] = input[idx];
}

at::Tensor roll_cuda(
    const at::Tensor& input,
    at::IntArrayRef shifts,
    at::IntArrayRef dims
) {
    utils::check_cuda_tensors({input});
    
    // Handle empty tensor
    if (input.numel() == 0) {
        return input.clone();
    }
    
    // Validate dimensions
    const int64_t ndim = input.ndimension();
    TORCH_CHECK(shifts.size() == dims.size(),
                "shifts and dims must have the same length");
    
    // Create a map of dimension to shift amount
    std::vector<int> shift_amounts(ndim, 0);
    for (size_t i = 0; i < dims.size(); ++i) {
        int64_t dim = dims[i];
        // Handle negative dims
        if (dim < 0) {
            dim += ndim;
        }
        TORCH_CHECK(dim >= 0 && dim < ndim,
                    "Dimension out of range (expected to be in range of [",
                    -ndim, ", ", ndim - 1, "], but got ", dims[i], ")");
        
        // Accumulate shifts for the same dimension
        shift_amounts[dim] += shifts[i];
    }
    
    // Convert shifts to positive values and filter out zero shifts
    auto sizes = input.sizes();
    auto strides = input.strides();
    
    std::vector<int> effective_sizes;
    std::vector<int> effective_strides;
    std::vector<int> effective_shifts;
    
    for (int d = 0; d < ndim; ++d) {
        if (sizes[d] > 0) {
            // Convert negative shifts to positive equivalents
            int normalized_shift = ((shift_amounts[d] % sizes[d]) + sizes[d]) % sizes[d];
            
            // Only include dimensions with non-zero shifts
            if (normalized_shift != 0) {
                effective_sizes.push_back(sizes[d]);
                effective_strides.push_back(strides[d]);
                effective_shifts.push_back(normalized_shift);
            }
        }
    }
    
    const int effective_dims = effective_sizes.size();
    
    // Early exit if no dimensions need shifting
    if (effective_dims == 0) {
        return input.clone();
    }
    
    // Allocate device memory for effective dimension info
    int* d_sizes;
    int* d_strides;
    int* d_shifts;
    
    cudaMalloc(&d_sizes, effective_dims * sizeof(int));
    cudaMalloc(&d_strides, effective_dims * sizeof(int));
    cudaMalloc(&d_shifts, effective_dims * sizeof(int));
    
    cudaMemcpy(d_sizes, effective_sizes.data(), effective_dims * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_strides, effective_strides.data(), effective_dims * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_shifts, effective_shifts.data(), effective_dims * sizeof(int), cudaMemcpyHostToDevice);
    
    // Create output tensor
    auto output = torch::empty_like(input);
    const int64_t num_elements = input.numel();
    
    // Launch kernel
    const int threads = 256;
    const int blocks = (num_elements + threads - 1) / threads;
    
    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, input.scalar_type(), "roll_cuda", [&] {
        roll_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            num_elements,
            effective_dims,
            d_sizes,
            d_strides,
            d_shifts
        );
    });
    
    // Clean up
    cudaFree(d_sizes);
    cudaFree(d_strides);
    cudaFree(d_shifts);
    
    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(error));
    
    return output;
}

}} // namespace custom_ops::tensor