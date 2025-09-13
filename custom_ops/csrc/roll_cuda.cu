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
    const int64_t n_elem,
    const int n_dims,
    const int64_t* __restrict__ sizes,
    const int64_t* __restrict__ strides,
    const int64_t* __restrict__ shifts
) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n_elem) return;
    
    // Compute offset between input and output positions
    int64_t offset = 0;
    
    for (int d = 0; d < n_dims; ++d) {
        const int64_t dim_idx = (idx / strides[d]) % sizes[d];
        const int64_t shifted_idx = (dim_idx + shifts[d]) % sizes[d];
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
    std::vector<int64_t> shift_amounts(ndim, 0);
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
    
    std::vector<int64_t> eff_sizes;
    std::vector<int64_t> eff_strides;
    std::vector<int64_t> eff_shifts;
    
    for (int64_t d = 0; d < ndim; ++d) {
        if (sizes[d] > 0) {
            // Convert negative shifts to positive equivalents
            int64_t normalized_shift = ((shift_amounts[d] % sizes[d]) + sizes[d]) % sizes[d];
            
            // Only include dimensions with non-zero shifts
            if (normalized_shift != 0) {
                eff_sizes.push_back(sizes[d]);
                eff_strides.push_back(strides[d]);
                eff_shifts.push_back(normalized_shift);
            }
        }
    }
    
    const int eff_dims = eff_sizes.size();
    
    // Early exit if no dimensions need shifting
    if (eff_dims == 0) {
        return input.clone();
    }
    
    // Allocate device memory for effective dimension info
    int64_t* d_sizes;
    int64_t* d_strides;
    int64_t* d_shifts;
    
    cudaMalloc(&d_sizes, eff_dims * sizeof(int64_t));
    cudaMalloc(&d_strides, eff_dims * sizeof(int64_t));
    cudaMalloc(&d_shifts, eff_dims * sizeof(int64_t));
    
    cudaMemcpy(d_sizes, eff_sizes.data(), eff_dims * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_strides, eff_strides.data(), eff_dims * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_shifts, eff_shifts.data(), eff_dims * sizeof(int64_t), cudaMemcpyHostToDevice);
    
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
            eff_dims,
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