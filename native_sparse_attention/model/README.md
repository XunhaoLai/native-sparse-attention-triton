# Guide for the ToyNSALlama Model

The `ToyNSALlama` model is a custom implementation of a Llama-like transformer architecture featuring a Native Sparse Attention (NSA) module. This guide explains how to integrate the NSA module into your own model.

## Overview

The `ToyNSALlama` model consists of:
- **Configuration**: Defined by `ToyNSALlamaConfig` (model structure parameters) and `InferenceConfig` (inference-specific parameters).
- **Components**: An embedding layer, multiple NativeSparseAttention modules, Feed-Forward Network (FFN) modules, normalization layers, and a language model head.

## Step-by-Step Instructions

### 1. Import Necessary Modules
```python
import torch
import torch.nn as nn
from native_sparse_attention.model import ToyNSALlama, ToyNSALlamaConfig, InferenceConfig
```

### 2. Define Configurations
Create instances of `ToyNSALlamaConfig` and `InferenceConfig` to set model and inference parameters.

#### Model Configuration
The model configuration aligns with the Transformers Llama model configuration. Adjust the following parameters to control the NSA module’s sparsity:
- `compress_type`: Compression method for keys/values. Supported options: `avgpool`, `weightedpool`, `linear`.
- `kernel_size` & `kernel_stride`: `kernel_size` determines how many tokens are compressed into one; `kernel_stride` sets the sliding window stride (must be divisible by `kernel_size`).
- `block_size`: Block size for sparse attention (recommended: 64 or 128).
- `topk`, `init_blocks`, `local_blocks`: `topk` specifies the number of blocks selected in sparse attention; `init_blocks` and `local_blocks` define the number of initial and local blocks that must be selected.
- `window_size`: Size of the sliding window for attention.

Example:
```python
config = ToyNSALlamaConfig(
    hidden_size=4096,
    intermediate_size=14336,
    num_hidden_layers=8,
    num_attention_heads=32,
    num_key_value_heads=2,
    head_dim=128,
    vocab_size=128288,
    max_position_embeddings=131072,
    rope_theta=500000.0,
    rope_scaling={
        "factor": 8.0,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3",
    },
    compress_type="weightedpool",
    kernel_size=32,
    kernel_stride=16,
    block_size=64,
    topk=8,
    init_blocks=1,
    local_blocks=2,
    window_size=512,
)
```

#### Inference Configuration
This configuration applies during inference, initializing the Key-Value (KV) Cache based on these settings. The full KV cache size is calculated as `max_batch_size × max_length × num_kv_heads × num_layers × 2 × 2` bytes. Currently, only greedy decoding is supported as an example.

Example:
```python
inference_config = InferenceConfig(
    max_batch_size=4,
    max_length=8192,
    max_new_tokens=128,
)
```

### 3. Initialize the Model
Instantiate the model and move it to the GPU with the appropriate data type (currently, only `bfloat16` is supported).

```python
model = ToyNSALlama(config, inference_config).cuda().to(torch.bfloat16)
```

### 4. Forward & Generate
The model supports two methods:
- **`forward`**: Accepts `input_ids` and `cu_seqlens`, returning final logits after the language model head. Use this for training or evaluation.
- **`generate`**: Accepts `input_ids` and `cu_seqlens`, generating output tokens via greedy sampling. This demonstrates KV cache usage for token generation (pre-filling and decoding).

Example:
```python
# Example input
batch_size = 4
seqlens = torch.randint(0, 4096, (batch_size,), dtype=torch.int32, device="cuda")
cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device="cuda")
cu_seqlens[1:] = torch.cumsum(seqlens, dim=0)
input_ids = torch.randint(0, 128288, (cu_seqlens[-1],), dtype=torch.int64, device="cuda")
print(f"\nEXAMPLE INPUT:\ncu_seqlens: {cu_seqlens}\ninput_ids: {input_ids.shape}\n")

# Example forward
logits = model(input_ids, cu_seqlens)
print(f"\nEXAMPLE OUTPUT:\nlogits: {logits.shape}\n")

# Example generate
output_tokens = model.generate(input_ids, cu_seqlens)
print(f"\nEXAMPLE GENERATE:\noutput_tokens: {output_tokens}\n")
```

## Toy Llama Model with Self-Attention
A simpler toy model with the Llama structure is available in `native_sparse_attention/model/toy_llama.py`. Compare `ToyLlama` and `ToyNSALlama` to see how to adapt a self-attention model into an NSA-based model.

The primary difference lies in replacing the `SelfAttention` module with the `NativeSparseAttention` module, along with updates to the KV cache and inference function. These changes are straightforward and easy to implement.
