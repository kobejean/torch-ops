#include "registration.h"

namespace custom_ops {
namespace fused {

at::Tensor mul_add_relu_cpu(const at::Tensor& x, const at::Tensor& weight, const at::Tensor& bias) {
    utils::check_same_device({x, weight, bias});
    utils::check_same_size(x, weight);
    utils::check_same_size(x, bias);
    TORCH_CHECK(x.dtype() == at::kFloat, "All tensors must be float32");
    TORCH_CHECK(weight.dtype() == at::kFloat, "All tensors must be float32");
    TORCH_CHECK(bias.dtype() == at::kFloat, "All tensors must be float32");
    
    // Fused: relu(x * weight + bias)
    return torch::relu(x * weight + bias);
}

}} // namespace custom_ops::fused