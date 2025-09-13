#include "registration.h"

namespace custom_ops {
namespace elementwise {

at::Tensor multiply_cpu(const at::Tensor& a, const at::Tensor& b) {
    TORCH_CHECK(a.sizes() == b.sizes(), "Tensors must have the same size");
    TORCH_CHECK(a.dtype() == at::kFloat, "Tensors must be float32");
    TORCH_CHECK(b.dtype() == at::kFloat, "Tensors must be float32");
    TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CPU);
    
    // Use PyTorch's optimized CPU implementation
    return a * b;
}

}} // namespace custom_ops::elementwise