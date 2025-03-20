# Triton Functions for Native Sparse Attention

This folder provides efficient Triton-based implementations of components for Native Sparse Attention. This README introduces the available functions, explains how to set them up, and offers guidance on their usage.

---

## Overview of Functions

The functions are organized into two main categories:

1. **Compression Methods**: Techniques for compressing key and value tensors.
2. **Attention Mechanisms**: Methods for computing attention between queries and compressed key/value tensors, including top-k sparse attention.

---

## Function Descriptions

### Compression Methods

These functions compress key and value tensors using a sliding window approach. Within each window, `kernel_size` tokens are compressed into a single token, with a stride of `kernel_stride`. All compression functions share similar input parameters and return formats.

**Parameters:**
- `x`: Input tensor (`total_len, num_heads, head_dim`)
- `w`: Weight tensor (shape varies by compression method)
- `cu_seqlens`: Cumulative sequence lengths (`batch_size + 1`)
- `kernel_size`: Size of the compression window
- `kernel_stride`: Stride of the compression window
- `pe`: Optional positional embedding (`num_heads, kernel_size, head_dim`)

**Returns:**
- Compressed tensor (`total_compress_len, num_heads, head_dim`)
- Cumulative sequence lengths (`com_cu_seqlens`) for the compressed tensor

#### `weightedpool_compress`
Compresses the input tensor using weighted pooling, applying a weighted sum over each block:  
$\hat{k} = w_1 k_1 + \dots + w_m k_m$  
- **Weight shape**: `(num_heads, kernel_size)`

#### `avgpool_compress`
Compresses the input tensor using average pooling:  
$\hat{k} = (k_1 + \dots + k_m) / m$  
- **Weight**: Must be `None`

#### `linear_compress`
Compresses the input tensor via linear projection, mapping each block to a single vector using learned weights:  
$\hat{k} = \text{cat}(k_1, \dots, k_m) W$  
- **Weight shape**: `(num_heads, kernel_size * head_dim, head_dim)`

---

### Attention Mechanisms

These functions compute attention using either full or sparse mechanisms.

#### `flash_attention_varlen`
A variable-length implementation of flash attention, similar to `flash_attn_varlen_func` from the `flash_attn` package.

**Parameters:**
- `q`, `k`, `v`: Query, key, and value tensors (`total_len, num_heads, head_dim`)
- `cu_seqlens_q`, `cu_seqlens_k`: Cumulative sequence lengths for queries and keys
- `max_seqlen_q`, `max_seqlen_k`: Maximum sequence lengths in the batch
- `causal`: Apply causal masking (default: `False`)
- `sm_scale`: Softmax scale (default: `1 / sqrt(head_dim)`)

**Returns:**
- Attention output tensor (`total_q_len, num_q_heads, head_dim`)

#### `compressed_attention`
Computes attention between a query and compressed key/value tensors, identifying the top-k blocks for sparse attention.

**Parameters:**
- `q`: Query tensor (`total_len, num_heads, head_dim`)
- `k`, `v`: Compressed key and value tensors (`total_compress_len, num_heads, head_dim`)
- `kernel_size`, `kernel_stride`: Compression parameters
- `block_size`: Size of blocks for sparse attention
- `topk`: Number of top blocks to select
- `cu_seqlens_q`, `cu_seqlens_k`: Cumulative sequence lengths for query and compressed key/value
- `max_seqlen_q`, `max_seqlen_k`: Maximum sequence lengths for query and compressed key/value
- `sm_scale`: Softmax scale (default: `1 / sqrt(head_dim)`)
- `init_blocks`: Number of initial blocks forced to be selected (default: `1`)
- `local_blocks`: Number of local blocks forced to be selected (default: `2`)

**Returns:**
- Tuple containing:
  - Attention output tensor
  - Top-k block indices

#### `topk_sparse_attention`
Performs sparse attention using precomputed top-k block indices. If a query attends to fewer than `topk` key/value blocks, the `topk_idx` should be padded with `-1` on the right.

**Parameters:**
- `q`, `k`, `v`: Query, key, and value tensors (`total_len, num_heads, head_dim`)
- `topk_idx`: Precomputed top-k indices (`num_kv_heads, total_len, topk`)
- `block_size`: Block size for sparse attention (recommended: `64` or `128`)
- `cu_seqlens`: Cumulative sequence lengths
- `softmax_scale`: Softmax scale (default: `1 / sqrt(head_dim)`)

**Returns:**
- Attention output tensor (`total_len, num_q_heads, head_dim`)

---

## Usage Example

Below is a typical workflow demonstrating how to combine these sparse attention functions:

```python
import torch
from native_sparse_attention.ops import linear_compress, compressed_attention, topk_sparse_attention

# Example input setup
num_q_heads = 64
num_kv_heads = 4
head_dim = 128
cu_seqlens = torch.tensor([0, 1024, 8192, 16384], dtype=torch.int32).cuda()

# Query, key, and value tensors
query = torch.randn(16384, num_q_heads, head_dim, dtype=torch.bfloat16).cuda()
key = torch.randn(16384, num_kv_heads, head_dim, dtype=torch.bfloat16).cuda()
value = torch.randn(16384, num_kv_heads, head_dim, dtype=torch.bfloat16).cuda()

# Compression weights and positional embeddings
kernel_size = 32
kernel_stride = 16
wk = torch.randn(num_kv_heads, kernel_size * head_dim, head_dim, dtype=torch.bfloat16).cuda()
wv = torch.randn_like(wk)
pe = torch.randn(num_kv_heads, kernel_size, head_dim, dtype=torch.bfloat16).cuda()

# Parameters for top-k sparse attention
block_size = 64
topk = 16

# 1. Compress key and value tensors
compressed_key, compressed_cu_seqlens = linear_compress(
    key, wk, cu_seqlens, kernel_size, kernel_stride, pe
)
compressed_value, _ = linear_compress(
    value, wv, cu_seqlens, kernel_size, kernel_stride, None
)

# 2. Compute attention with compressed key/value and get top-k indices
compressed_attn_output, topk_idx = compressed_attention(
    query,
    compressed_key,
    compressed_value,
    kernel_size,
    kernel_stride,
    block_size,
    topk,
    cu_seqlens,
    compressed_cu_seqlens,
    init_blocks=1,
    local_blocks=2,
)

# 3. Perform top-k sparse attention
sparse_attn_output = topk_sparse_attention(
    query,
    key,
    value,
    topk_idx,
    block_size,
    cu_seqlens,
)

# 4. Combine attention outputs (e.g., average)
attn_output = (compressed_attn_output + sparse_attn_output) / 2
```

For a complete implementation of the Native Sparse Attention module, see `native_sparse_attention/module/native_sparse_attention.py`.
