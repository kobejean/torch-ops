#pragma once
#include <torch/extension.h>
#include <ATen/ATen.h>

// Namespace for all our operations
namespace custom_ops {

// Elementwise operations
namespace elementwise {
    at::Tensor multiply_cpu(const at::Tensor& a, const at::Tensor& b);
    at::Tensor multiply_cuda(const at::Tensor& a, const at::Tensor& b);
}

// Activation functions  
namespace activation {
    at::Tensor square_relu_forward_cpu(const at::Tensor& input);
    at::Tensor square_relu_forward_cuda(const at::Tensor& input);
    at::Tensor square_relu_backward_cpu(const at::Tensor& grad_output, const at::Tensor& input);
    at::Tensor square_relu_backward_cuda(const at::Tensor& grad_output, const at::Tensor& input);
}

// Fused operations
namespace fused {
    at::Tensor mul_add_relu_cpu(const at::Tensor& x, const at::Tensor& weight, const at::Tensor& bias);
    at::Tensor mul_add_relu_cuda(const at::Tensor& x, const at::Tensor& weight, const at::Tensor& bias);
}

// Tensor operations
namespace tensor {
    at::Tensor roll_cpu(const at::Tensor& input, at::IntArrayRef shifts, at::IntArrayRef dims);
    at::Tensor roll_cuda(const at::Tensor& input, at::IntArrayRef shifts, at::IntArrayRef dims);
}

} // namespace custom_ops