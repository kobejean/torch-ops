#include "registration.h"
#include <pybind11/pybind11.h>

namespace custom_ops {
namespace utils {

void check_cuda_tensors(const std::vector<at::Tensor>& tensors) {
    for (const auto& tensor : tensors) {
        TORCH_CHECK(tensor.device().is_cuda(), "All tensors must be on CUDA");
    }
}

void check_same_device(const std::vector<at::Tensor>& tensors) {
    if (tensors.empty()) return;
    
    auto device = tensors[0].device();
    for (const auto& tensor : tensors) {
        TORCH_CHECK(tensor.device() == device, "All tensors must be on the same device");
    }
}

void check_same_size(const at::Tensor& a, const at::Tensor& b) {
    TORCH_CHECK(a.sizes() == b.sizes(), "Tensors must have the same size");
}

} // namespace utils
} // namespace custom_ops

// Register all operators
TORCH_LIBRARY(custom_ops, m) {
    // Elementwise operations
    m.def("elementwise_multiply(Tensor a, Tensor b) -> Tensor");
    
    // Activation functions
    m.def("activation_square_relu_forward(Tensor input) -> Tensor");
    m.def("activation_square_relu_backward(Tensor grad_output, Tensor input) -> Tensor");
    
    // Fused operations
    m.def("fused_mul_add_relu(Tensor x, Tensor weight, Tensor bias) -> Tensor");
}

// Register CPU implementations
TORCH_LIBRARY_IMPL(custom_ops, CPU, m) {
    m.impl("elementwise_multiply", &custom_ops::elementwise::multiply_cpu);
    m.impl("activation_square_relu_forward", &custom_ops::activation::square_relu_forward_cpu);
    m.impl("activation_square_relu_backward", &custom_ops::activation::square_relu_backward_cpu);
    m.impl("fused_mul_add_relu", &custom_ops::fused::mul_add_relu_cpu);
}

// Register CUDA implementations
TORCH_LIBRARY_IMPL(custom_ops, CUDA, m) {
    m.impl("elementwise_multiply", &custom_ops::elementwise::multiply_cuda);
    m.impl("activation_square_relu_forward", &custom_ops::activation::square_relu_forward_cuda);
    m.impl("activation_square_relu_backward", &custom_ops::activation::square_relu_backward_cuda);
    m.impl("fused_mul_add_relu", &custom_ops::fused::mul_add_relu_cuda);
}

// Python module creation
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Empty - operators registered via TORCH_LIBRARY
}