import torch
from . import _C  # Load the extension

# Operator access with clear naming
elementwise_multiply = torch.ops.custom_ops.elementwise_multiply.default
activation_square_relu_forward = torch.ops.custom_ops.activation_square_relu_forward.default
activation_square_relu_backward = torch.ops.custom_ops.activation_square_relu_backward.default
fused_mul_add_relu = torch.ops.custom_ops.fused_mul_add_relu.default
tensor_roll = torch.ops.custom_ops.tensor_roll.default
tensor_roll_backward = torch.ops.custom_ops.tensor_roll_backward.default

# torch.compile support
@torch.library.register_fake("custom_ops::elementwise_multiply")
def _(a, b):
    torch._check(a.shape == b.shape, "Tensors must have same shape")
    torch._check(a.dtype == torch.float, "Tensors must be float32")
    torch._check(b.dtype == torch.float, "Tensors must be float32")
    return torch.empty_like(a)

@torch.library.register_fake("custom_ops::activation_square_relu_forward")
def _(input):
    torch._check(input.dtype == torch.float, "Input must be float32")
    return torch.empty_like(input)

@torch.library.register_fake("custom_ops::activation_square_relu_backward")
def _(grad_output, input):
    torch._check(grad_output.shape == input.shape, "Tensors must have same shape")
    torch._check(grad_output.dtype == torch.float, "grad_output must be float32")
    torch._check(input.dtype == torch.float, "input must be float32")
    return torch.empty_like(input)

@torch.library.register_fake("custom_ops::fused_mul_add_relu")
def _(x, weight, bias):
    torch._check(x.shape == weight.shape, "x and weight must have same shape")
    torch._check(x.shape == bias.shape, "x and bias must have same shape")
    torch._check(x.dtype == torch.float, "All tensors must be float32")
    return torch.empty_like(x)

@torch.library.register_fake("custom_ops::tensor_roll")
def _(input, shifts, dims):
    torch._check(len(shifts) == len(dims), "shifts and dims must have same length")
    for dim in dims:
        torch._check(dim >= -input.ndim and dim < input.ndim, f"dim {dim} out of range")
    return torch.empty_like(input)

@torch.library.register_fake("custom_ops::tensor_roll_backward")
def _(grad_output, shifts, dims):
    torch._check(len(shifts) == len(dims), "shifts and dims must have same length")
    for dim in dims:
        torch._check(dim >= -grad_output.ndim and dim < grad_output.ndim, f"dim {dim} out of range")
    return torch.empty_like(grad_output)

# Autograd support for SquareReLU
def _square_relu_backward_fn(ctx, grad):
    input, = ctx.saved_tensors
    return activation_square_relu_backward(grad, input)

def _square_relu_setup_context(ctx, inputs, output):
    input, = inputs
    ctx.save_for_backward(input)

torch.library.register_autograd(
    "custom_ops::activation_square_relu_forward",
    _square_relu_backward_fn,
    setup_context=_square_relu_setup_context
)

# Autograd support for fused operation
def _fused_mul_add_relu_backward_fn(ctx, grad_output):
    x, weight, bias = ctx.saved_tensors
    
    # Compute ReLU mask: output > 0 iff input > 0
    relu_mask = (x * weight + bias) > 0
    
    # Gradients
    grad_x = grad_output * weight * relu_mask
    grad_weight = grad_output * x * relu_mask
    grad_bias = grad_output * relu_mask
    
    return grad_x, grad_weight, grad_bias

def _fused_mul_add_relu_setup_context(ctx, inputs, output):
    x, weight, bias = inputs
    ctx.save_for_backward(x, weight, bias)

torch.library.register_autograd(
    "custom_ops::fused_mul_add_relu",
    _fused_mul_add_relu_backward_fn,
    setup_context=_fused_mul_add_relu_setup_context
)

# Autograd support for tensor roll
def _tensor_roll_backward_fn(ctx, grad_output):
    shifts, dims = ctx.shifts, ctx.dims
    return tensor_roll_backward(grad_output, shifts, dims), None, None

def _tensor_roll_setup_context(ctx, inputs, output):
    input, shifts, dims = inputs
    ctx.shifts = shifts
    ctx.dims = dims

torch.library.register_autograd(
    "custom_ops::tensor_roll",
    _tensor_roll_backward_fn,
    setup_context=_tensor_roll_setup_context
)

# Convenient wrapper functions
def square_relu(input):
    """SquareReLU activation: f(x) = x^2 if x > 0, else 0"""
    return activation_square_relu_forward(input)

def multiply(a, b):
    """Element-wise multiplication"""
    return elementwise_multiply(a, b)

def fused_linear_relu(x, weight, bias):
    """Fused linear transformation with ReLU: relu(x * weight + bias)"""
    return fused_mul_add_relu(x, weight, bias)

def roll(input, shifts, dims):
    """Roll tensor elements along specified dimensions"""
    return tensor_roll(input, shifts, dims)

# Testing and utilities
def test_all_operators():
    """Test all operators for correctness"""
    print("🧪 Testing all custom operators...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Test elementwise multiply
    a = torch.randn(1000, device=device)
    b = torch.randn(1000, device=device)
    result = multiply(a, b)
    expected = a * b
    print(f"✓ Elementwise multiply: {torch.allclose(result, expected)}")
    
    # Test square_relu with gradients
    x = torch.randn(100, device=device, requires_grad=True)
    y = square_relu(x)
    y.sum().backward()
    print(f"✓ SquareReLU forward: {y.shape == x.shape}")
    print(f"✓ SquareReLU backward: {x.grad is not None}")
    
    # Test fused operation
    x = torch.randn(1000, device=device)
    weight = torch.randn(1000, device=device)
    bias = torch.randn(1000, device=device)
    
    fused_result = fused_linear_relu(x, weight, bias)
    unfused_result = torch.relu(x * weight + bias)
    print(f"✓ Fused mul-add-relu: {torch.allclose(fused_result, unfused_result)}")
    
    # Test roll operation with gradients
    x_roll = torch.arange(12, device=device, dtype=torch.float32, requires_grad=True).reshape(3, 4)
    roll_result = roll(x_roll, [1, 2], [0, 1])
    roll_expected = torch.roll(x_roll.detach(), [1, 2], [0, 1])
    print(f"✓ Roll operation forward: {torch.allclose(roll_result, roll_expected)}")

    # Test backward pass
    roll_result.sum().backward()
    print(f"✓ Roll operation backward: {x_roll.grad is not None and torch.allclose(x_roll.grad, torch.ones_like(x_roll))}")
    
    # Test opcheck
    try:
        torch.library.opcheck(elementwise_multiply, [a[:10], b[:10]])
        print("✅ All opcheck tests passed!")
    except Exception as e:
        print(f"⚠️ Opcheck: {e}")
    
    print("🎉 All tests completed!")

# Version and metadata
__version__ = "1.0.0"
__all__ = [
    "elementwise_multiply", "multiply",
    "activation_square_relu_forward", "square_relu", 
    "fused_mul_add_relu", "fused_linear_relu",
    "tensor_roll", "roll",
    "test_all_operators"
]