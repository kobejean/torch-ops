#include "registration.h"

namespace custom_ops {
namespace tensor {

at::Tensor roll_cpu(
    const at::Tensor& input, 
    at::IntArrayRef shifts, 
    at::IntArrayRef dims
) {
    TORCH_CHECK(input.dtype() == at::kFloat || input.dtype() == at::kDouble || 
                input.dtype() == at::kInt || input.dtype() == at::kLong,
                "Roll operation supports float32, float64, int32, and int64 tensors");
    TORCH_INTERNAL_ASSERT(input.device().type() == at::DeviceType::CPU);
    
    // Handle empty tensor
    if (input.numel() == 0) {
        return input.clone();
    }
    
    // Validate dims
    const int64_t ndim = input.ndimension();
    TORCH_CHECK(shifts.size() == dims.size(), 
                "shifts and dims must have the same length");
    
    for (size_t i = 0; i < dims.size(); ++i) {
        int64_t dim = dims[i];
        // Handle negative dims
        if (dim < 0) {
            dim += ndim;
        }
        TORCH_CHECK(dim >= 0 && dim < ndim, 
                    "Dimension out of range (expected to be in range of [", 
                    -ndim, ", ", ndim - 1, "], but got ", dims[i], ")");
    }
    
    // Use PyTorch's optimized CPU implementation
    return at::roll(input, shifts, dims);
}

}} // namespace custom_ops::tensor