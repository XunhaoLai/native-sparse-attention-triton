import torch
from typing import Any 
from einops import rearrange, einsum
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
    dtype: torch.dtype = torch.float16,
    device: str = "cuda"
):
    """
    Test both PyTorch and Triton implementations of linear_compress for equivalence.
    
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
    
    # Create input tensors
    x = torch.randn(
        (total_len, num_heads, head_dim),
        dtype=dtype,
        device=device,
        requires_grad=False
    )
    
    for kernel_size, kernel_stride in zip(kernel_sizes, kernel_strides):
        print(f"\nTesting kernel_size={kernel_size}, kernel_stride={kernel_stride}")
        
        # Create weight tensor
        w = torch.randn(
            (num_heads, kernel_size * head_dim, head_dim),
            dtype=dtype,
            device=device,
            requires_grad=False
        )
        
        # Create positional encoding if needed
        pe = None
        if use_pe:
            pe = torch.randn(
                (num_heads, kernel_size, head_dim),
                dtype=dtype,
                device=device,
                requires_grad=False
            )
        
        # Run PyTorch implementation
        y_torch, y_cu_seqlens_torch = linear_compress_torch(  # Assuming your PyTorch implementation is named this
            x=x,
            w=w,
            cu_seqlens=cu_seqlens,
            kernel_size=kernel_size,
            kernel_stride=kernel_stride,
            pe=pe
        )
        
        # Run Triton implementation
        y_triton, y_cu_seqlens_triton = linear_compress_triton(  # Assuming your Triton implementation is named this
            x=x,
            w=w,
            cu_seqlens=cu_seqlens,
            kernel_size=kernel_size,
            kernel_stride=kernel_stride,
            pe=pe
        )
        
        
        # Check numerical equivalence
        atol, rtol = 1e-3, 1e-1
        values_match = torch.allclose(y_torch, y_triton, atol=atol, rtol=rtol)
        print(f"Output values match (atol={atol}, rtol={rtol}): {values_match}")
        if not values_match:
            max_diff = (y_torch - y_triton).abs().max().item()
            print(f"Maximum difference: {max_diff}")
            
            # Print some sample values for debugging
            print("\nSample values (first batch, first head):")
            print("Torch:", y_torch[0, 0, :5])
            print("Triton:", y_triton[0, 0, :5])



if __name__ == "__main__":
    # Run tests
    test_linear_compress(
        batch_size=4,
        num_heads=8,
        head_dim=64,
        max_seqlen=512,
        kernel_sizes=[16, 32],
        kernel_strides=[8, 16],
        use_pe=False,
        dtype=torch.float16,
        device="cuda"
    )