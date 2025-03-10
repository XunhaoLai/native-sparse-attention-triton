import torch
from typing import Any 
from native_sparse_attention.ops.torch.compress_key_value import linear_compress as linear_compress_torch
from native_sparse_attention.ops.triton.linear_compress import linear_compress as linear_compress_triton

def test_linear_compress(
    batch_size: int = 4,
    num_heads: int = 8,
    head_dim: int = 64,
    max_seqlen: int = 512,
    kernel_sizes: list = [16, 32],
    kernel_strides: list = [8, 16],
    use_pe: bool = True,
    dtype: torch.dtype = torch.float32,
    device: str = "cuda"
):
    """
    Test both PyTorch and Triton implementations of linear_compress for equivalence,
    including forward and backward passes.
    
    Args:
        batch_size: Number of sequences in the batch
        num_heads: Number of attention heads
        head_dim: Dimension of each attention head
        max_seqlen: Maximum sequence length
        kernel_sizes: List of kernel sizes to test
        kernel_strides: List of kernel strides to test
        use_pe: Whether to test with positional encoding
        dtype: Data type for tensors
        device: Device to run the test on
    """
    torch.manual_seed(42)
    
    # Generate random sequence lengths for each batch
    seqlens = torch.randint(
        low=kernel_sizes[0],  # minimum length should be at least kernel_size
        high=max_seqlen + 1,
        size=(batch_size,),
        device=device
    )
    cu_seqlens = torch.cat([
        torch.tensor([0], device=device, dtype=torch.int32),
        torch.cumsum(seqlens, dim=0).to(torch.int32)
    ])
    
    total_len = cu_seqlens[-1].item()
    
    for kernel_size, kernel_stride in zip(kernel_sizes, kernel_strides):
        print(f"\nTesting kernel_size={kernel_size}, kernel_stride={kernel_stride}")
        
        # Create input tensors with requires_grad=True
        x_torch = torch.zeros(
            (total_len, num_heads, head_dim),
            dtype=dtype,
            device=device,
        ).uniform_(-1, 1)
        x_torch.requires_grad_(True)

        x_triton = x_torch.clone().detach().requires_grad_(True)
        
        w_torch = torch.ones(
            (num_heads, kernel_size * head_dim, head_dim),
            dtype=dtype,
            device=device,
        ) / kernel_size
        w_torch.requires_grad_(True)

        w_triton = w_torch.clone().detach().requires_grad_(True)
        
        pe_torch = None
        pe_triton = None
        if use_pe:
            pe_torch = torch.randn(
                (num_heads, kernel_size, head_dim),
                dtype=dtype,
                device=device,
                requires_grad=True
            )
            pe_triton = pe_torch.clone().detach().requires_grad_(True)
        
        # Run forward passes
        y_torch, y_cu_seqlens_torch = linear_compress_torch(
            x=x_torch,
            w=w_torch,
            cu_seqlens=cu_seqlens,
            kernel_size=kernel_size,
            kernel_stride=kernel_stride,
            pe=pe_torch
        )
        
        y_triton, y_cu_seqlens_triton = linear_compress_triton(
            x=x_triton,
            w=w_triton,
            cu_seqlens=cu_seqlens,
            kernel_size=kernel_size,
            kernel_stride=kernel_stride,
            pe=pe_triton
        )
        
        # Check forward pass numerical equivalence
        atol, rtol = 1e-2, 1e-2
        values_match = torch.allclose(y_torch, y_triton, atol=atol, rtol=rtol)
        print(f"Forward pass - Output values match (atol={atol}, rtol={rtol}): {values_match}")
        if not values_match:
            max_diff = (y_torch - y_triton).abs().max().item()
            print(f"Forward pass - Maximum difference: {max_diff}")
            print("\nSample values (first batch, first head):")
            print("Torch:", y_torch[0, 0, :5])
            print("Triton:", y_triton[0, 0, :5])
        
        # Create random output gradients for backward pass
        grad_output = torch.randn_like(y_torch)
        
        # Run backward passes
        y_torch.backward(grad_output)
        y_triton.backward(grad_output)
        
        # Check gradient equivalence
        print("\nTesting backward pass:")
        
        # Check x gradients
        x_grads_match = torch.allclose(
            x_torch.grad,
            x_triton.grad,
            atol=atol,
            rtol=rtol
        )
        print(f"x gradients match (atol={atol}, rtol={rtol}): {x_grads_match}")
        if not x_grads_match:
            max_diff = (x_torch.grad - x_triton.grad).abs().max().item()
            print(f"x gradients - Maximum difference: {max_diff}")
            print("\nSample x gradients (first batch, first head):")
            print("Torch:", x_torch.grad[0, 0, :5])
            print("Triton:", x_triton.grad[0, 0, :5])
        
        # Check w gradients
        w_grads_match = torch.allclose(
            w_torch.grad,
            w_triton.grad,
            atol=atol,
            rtol=rtol
        )
        print(f"w gradients match (atol={atol}, rtol={rtol}): {w_grads_match}")
        if not w_grads_match:
            max_diff = (w_torch.grad - w_triton.grad).abs().max().item()
            print(f"w gradients - Maximum difference: {max_diff}")
            print("\nSample w gradients (first head):")
            print("Torch:", w_torch.grad[0, :5, 0])
            print("Triton:", w_triton.grad[0, :5, 0])
        
        # Check pe gradients if used
        if use_pe:
            pe_grads_match = torch.allclose(
                pe_torch.grad,
                pe_triton.grad,
                atol=atol,
                rtol=rtol
            )
            print(f"pe gradients match (atol={atol}, rtol={rtol}): {pe_grads_match}")
            if not pe_grads_match:
                max_diff = (pe_torch.grad - pe_triton.grad).abs().max().item()
                print(f"pe gradients - Maximum difference: {max_diff}")
                print("\nSample pe gradients (first head):")
                print("Torch:", pe_torch.grad[0, :5, 0])
                print("Triton:", pe_triton.grad[0, :5, 0])
        
        # Clean up gradients for next iteration
        x_torch.grad = None
        x_triton.grad = None
        w_torch.grad = None
        w_triton.grad = None
        if use_pe:
            pe_torch.grad = None
            pe_triton.grad = None


if __name__ == "__main__":
    # Run tests
    test_linear_compress(
        batch_size=4,
        num_heads=8,
        head_dim=32,
        max_seqlen=512,
        kernel_sizes=[32, 64],
        kernel_strides=[16, 32],
        use_pe=False,
        dtype=torch.float16,
        device="cuda"
    )