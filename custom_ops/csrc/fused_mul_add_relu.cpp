#include "registration.h"

namespace custom_ops {
namespace fused {

at::Tensor mul_add_relu_cpu(const at::Tensor& x, const at::Tensor& weight, const at::Tensor& bias) {
    TORCH_CHECK(x.device() == weight.device(), "All tensors must be on the same device");
    TORCH_CHECK(x.device() == bias.device(), "All tensors must be on the same device");
    TORCH_CHECK(x.sizes() == weight.sizes(), "Tensors 'x' and 'weight' must have the same size");
    TORCH_CHECK(x.sizes() == bias.sizes(), "Tensors 'x' and 'bias' must have the same size");
    TORCH_CHECK(x.dtype() == at::kFloat, "All tensors must be float32");
    TORCH_CHECK(weight.dtype() == at::kFloat, "All tensors must be float32");
    TORCH_CHECK(bias.dtype() == at::kFloat, "All tensors must be float32");
    
    // Fused: relu(x * weight + bias)
    return torch::relu(x * weight + bias);
}

}} // namespace custom_ops::fused