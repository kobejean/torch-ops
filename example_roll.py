#!/usr/bin/env python3
"""
Example usage of the custom roll operation.

This demonstrates how to use the custom torch.roll implementation
with both CPU and CUDA tensors.
"""

import torch
import torch.ops.custom_ops as custom_ops

def test_roll_operation():
    """Test the custom roll operation with various inputs."""
    
    # Test with CPU tensor
    print("Testing CPU roll operation:")
    x_cpu = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    print(f"Original tensor:\n{x_cpu}")
    
    # Roll along dimension 0 (rows)
    rolled_cpu = custom_ops.tensor_roll(x_cpu, [1], [0])
    print(f"Rolled by 1 along dim 0:\n{rolled_cpu}")
    
    # Roll along dimension 1 (columns)
    rolled_cpu = custom_ops.tensor_roll(x_cpu, [2], [1])
    print(f"Rolled by 2 along dim 1:\n{rolled_cpu}")
    
    # Roll along multiple dimensions
    rolled_cpu = custom_ops.tensor_roll(x_cpu, [1, 2], [0, 1])
    print(f"Rolled by [1, 2] along dims [0, 1]:\n{rolled_cpu}")
    
    # Compare with PyTorch's built-in roll
    pytorch_roll = torch.roll(x_cpu, [1, 2], [0, 1])
    print(f"PyTorch roll (for comparison):\n{pytorch_roll}")
    print(f"Results match: {torch.allclose(rolled_cpu, pytorch_roll)}")
    
    # Test with CUDA if available
    if torch.cuda.is_available():
        print("\nTesting CUDA roll operation:")
        x_cuda = x_cpu.cuda()
        rolled_cuda = custom_ops.tensor_roll(x_cuda, [1, 2], [0, 1])
        print(f"CUDA rolled result:\n{rolled_cuda}")
        
        # Compare with PyTorch's CUDA roll
        pytorch_cuda_roll = torch.roll(x_cuda, [1, 2], [0, 1])
        print(f"Results match PyTorch CUDA: {torch.allclose(rolled_cuda, pytorch_cuda_roll)}")
    else:
        print("\nCUDA not available, skipping CUDA tests")

def benchmark_roll():
    """Simple benchmark comparing custom roll vs PyTorch roll."""
    import time
    
    # Create larger tensor for benchmarking
    size = (1000, 1000)
    x = torch.randn(size, dtype=torch.float32)
    shifts = [100, 200]
    dims = [0, 1]
    
    # Warm up
    for _ in range(10):
        _ = custom_ops.tensor_roll(x, shifts, dims)
        _ = torch.roll(x, shifts, dims)
    
    # Benchmark custom implementation
    start = time.time()
    for _ in range(100):
        result_custom = custom_ops.tensor_roll(x, shifts, dims)
    custom_time = time.time() - start
    
    # Benchmark PyTorch implementation
    start = time.time()
    for _ in range(100):
        result_pytorch = torch.roll(x, shifts, dims)
    pytorch_time = time.time() - start
    
    print(f"\nBenchmark results (100 iterations on {size} tensor):")
    print(f"Custom roll:   {custom_time:.4f}s")
    print(f"PyTorch roll:  {pytorch_time:.4f}s")
    print(f"Speedup:       {pytorch_time/custom_time:.2f}x")
    print(f"Results match: {torch.allclose(result_custom, result_pytorch)}")

if __name__ == "__main__":
    test_roll_operation()
    benchmark_roll()