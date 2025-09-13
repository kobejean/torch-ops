# Custom PyTorch CUDA Operations

A collection of optimized CUDA kernels for PyTorch, featuring custom implementations of tensor operations with performance improvements over standard PyTorch implementations.

## Featured: Custom Roll Operation

The primary focus of this project is a custom CUDA implementation of `torch.roll` that achieves significant performance improvements on large tensors.

### Key Features

- **CUDA-optimized kernel** with constant memory for dimension metadata
- **Perfect GPU utilization** - 100% theoretical occupancy on Tesla T4
- **Multi-dimensional rolling** support for up to 8 dimensions
- **Branch-free execution** for optimal GPU performance

### Performance Results

Benchmarked on Tesla T4 GPU with comprehensive testing:

| Tensor Size | Custom (ms) | PyTorch (ms) | Speedup |
|-------------|-------------|--------------|---------|
| 500 × 500   | 0.064       | 0.037        | 0.57×   |
| 1000 × 1000 | 0.139       | 0.132        | 0.95×   |
| 5000 × 5000 | 2.588       | 3.021        | **1.17×** |
| 10000 × 10000 | 5.521     | 7.123        | **1.29×** |

**Key insights:**
- Achieves **1.29× speedup** on large tensors (100M elements)
- Maintains **perfect correctness** with zero numerical difference vs PyTorch
- Reaches **41.2 GB/s memory bandwidth** utilization
- **100% theoretical GPU occupancy** demonstrates optimal kernel configuration

### Technical Implementation

The kernel leverages several CUDA optimization techniques:

- **Constant memory** for fast access to dimension metadata
- **256 threads per block** configuration for optimal occupancy
- **Optimized memory access** patterns where possible
- **Efficient dimension filtering** to skip zero shifts

### Quick Test

```bash
# Clone and test
git clone https://github.com/kobejean/torch-ops.git
cd torch-ops
pip install -e .

# Run in Python
import torch
import custom_ops

x = torch.randn(1000, 1000, device='cuda')
result = custom_ops.roll(x, [100, 200], [0, 1])

# Verify correctness
expected = torch.roll(x, [100, 200], [0, 1])
print(f"Results match: {torch.allclose(result, expected)}")
```

### Colab Demo

Try the full performance analysis in Google Colab (Make sure to set the runtime type to an appropriate GPU setting):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kobejean/torch-ops/blob/main/test_roll_colab.ipynb)

The notebook includes comprehensive benchmarking, occupancy analysis, and profiling following GPU programming best practices.

## Other Operations

This repository also includes implementations of:

- **Square ReLU activation** - f(x) = x² for x > 0, else 0
- **Fused multiply-add-ReLU** - Optimized relu(x * weight + bias)

All operations support both CPU and CUDA backends with proper PyTorch integration.

## Requirements

- PyTorch >= 1.12
- CUDA toolkit
- C++14 compatible compiler

## Installation

```bash
pip install -e .
```

Built and tested with PyTorch's extension system using pybind11 and TORCH_LIBRARY registration.