#include "registration.h"

namespace custom_ops {
namespace activation {

at::Tensor square_relu_forward_cpu(const at::Tensor& input) {
    TORCH_CHECK(input.dtype() == at::kFloat, "Input must be float32");
    TORCH_INTERNAL_ASSERT(input.device().type() == at::DeviceType::CPU);
    
    // f(x) = x^2 if x > 0, else 0
    return torch::where(input > 0, input * input, torch::zeros_like(input));
}

at::Tensor square_relu_backward_cpu(const at::Tensor& grad_output, const at::Tensor& input) {
    TORCH_CHECK(grad_output.sizes() == input.sizes(), "grad_output and input must have the same size");
    TORCH_CHECK(grad_output.dtype() == at::kFloat, "grad_output must be float32");
    TORCH_CHECK(input.dtype() == at::kFloat, "input must be float32");
    TORCH_INTERNAL_ASSERT(grad_output.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(input.device().type() == at::DeviceType::CPU);
    
    // df/dx = 2x if x > 0, else 0
    return torch::where(input > 0, 2 * input * grad_output, torch::zeros_like(input));
}

}} // namespace custom_ops::activation