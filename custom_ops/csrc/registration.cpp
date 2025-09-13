#include "registration.h"
#include <pybind11/pybind11.h>

// Register all operators
TORCH_LIBRARY(custom_ops, m) {
    // Elementwise operations
    m.def("elementwise_multiply(Tensor a, Tensor b) -> Tensor");
    
    // Activation functions
    m.def("activation_square_relu_forward(Tensor input) -> Tensor");
    m.def("activation_square_relu_backward(Tensor grad_output, Tensor input) -> Tensor");
    
    // Fused operations
    m.def("fused_mul_add_relu(Tensor x, Tensor weight, Tensor bias) -> Tensor");
    
    // Tensor operations
    m.def("tensor_roll(Tensor input, int[] shifts, int[] dims) -> Tensor");
}

// Register CPU implementations
TORCH_LIBRARY_IMPL(custom_ops, CPU, m) {
    m.impl("elementwise_multiply", &custom_ops::elementwise::multiply_cpu);
    m.impl("activation_square_relu_forward", &custom_ops::activation::square_relu_forward_cpu);
    m.impl("activation_square_relu_backward", &custom_ops::activation::square_relu_backward_cpu);
    m.impl("fused_mul_add_relu", &custom_ops::fused::mul_add_relu_cpu);
    m.impl("tensor_roll", &custom_ops::tensor::roll_cpu);
}

// Register CUDA implementations
TORCH_LIBRARY_IMPL(custom_ops, CUDA, m) {
    m.impl("elementwise_multiply", &custom_ops::elementwise::multiply_cuda);
    m.impl("activation_square_relu_forward", &custom_ops::activation::square_relu_forward_cuda);
    m.impl("activation_square_relu_backward", &custom_ops::activation::square_relu_backward_cuda);
    m.impl("fused_mul_add_relu", &custom_ops::fused::mul_add_relu_cuda);
    m.impl("tensor_roll", &custom_ops::tensor::roll_cuda);
}

// Python module creation
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Empty - operators registered via TORCH_LIBRARY
}