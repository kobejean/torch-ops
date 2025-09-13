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

// Utility functions
namespace utils {
    void check_cuda_tensors(const std::vector<at::Tensor>& tensors);
    void check_same_device(const std::vector<at::Tensor>& tensors);
    void check_same_size(const at::Tensor& a, const at::Tensor& b);
}

} // namespace custom_ops