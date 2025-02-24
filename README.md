# Native Sparse Attention Triton

This repository implements the sparse attention mechanism introduced in the paper [Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention](https://arxiv.org/abs/2502.11089) and provides an efficient training implementation based on [Triton](https://github.com/triton-lang/triton).

## Requirements
Ensure the following dependencies are installed:
- PyTorch >= 2.1.0
- triton >= 3.0.0
- einops >= 0.7.0
- flash_attn >= 2.6.3

## Usage

### Notes
1. PyTorch implementations (`ops.torch`) are intended for debugging only.
2. For production use, prefer Triton operators (`ops.triton`).
3. All implementations are based on the varlen approach similiar to flash_attn_func_varlen. Please concatenate the inputs of a batch before use.
4. Only support attention head dimension less than 128 for now.

### Functions

The `ops` module has implemented several functions required for native sparse attention. Please refer to the function docstrings for usage.

You can import those functions from the `ops` module:

```python
from ops import (
    conv_compress,
    compressed_attention,
    topk_sparse_attention
)

# 1. key value compression
compressed_key, compressed_cu_seqlens = conv_compress(
    key, 
    w, 
    cu_seqlens, 
    kernel_size, 
    kernel_stride, 
    pe
)
compressed_value, _ = conv_compress(
    value, 
    w, 
    cu_seqlens, 
    kernel_size, 
    kernel_stride, 
    None
)

# 2. attention between query and compressed key value
compressed_attn_output, topk_idx = compressed_attention_torch(
    q,
    compressed_key,
    compressed_value,
    kernel_size,
    kernel_stride,
    block_size,
    topk,
    cu_seqlens,
    compressed_cu_seqlens,
    max_seqlen,
    compressed_max_seqlen,
)

# 3. topk sparse attention
sparse_attn_output = topk_sparse_attention(
    query, 
    key, 
    value, 
    topk_idx, 
    block_size, 
    cu_seqlens
)

```

### Module

The `modules` directory also provides implementations based on `torch.nn.module` for easy integration into models.

```python
from modules import NativeSparseAttention, RopeConfig

NSA_Layer = NativeSparseAttentionNoRoPE(
    hidden_size=4096,
    num_q_heads=64,
    num_kv_heads=4,
    head_dim=128,
    kernel_size=32,
    kernel_stride=16,
    block_size=64,
    topk=4,
    init_blocks=1,
    local_blocks=2,
    window_size=512,
    rope_config=RopeConfig(
        max_position_embeddings=32768,
        head_dim=128,
        rope_theta=500000,
        rope_scaling={
            "factor": 4.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3",
        },
    ),
)
```

## Testing

Some test scripts are available in the `test` folder and can be run directly for unit testing. For example:

```bash
python test/test_topk_sparse_attention.py
python test/test_nsa_w_rope.py
```

## Contributing
Contributions are welcome! Please open an issue to discuss major changes.

## Contact

For any questions or feedback, please feel free to contact laixunhao@pku.edu.cn.

## Citations

```bibtex
@inproceedings{Yuan2025NativeSA,
    title   = {Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention},
    author  = {Jingyang Yuan and Huazuo Gao and Damai Dai and Junyu Luo and Liang Zhao and Zhengyan Zhang and Zhenda Xie and Y. X. Wei and Lean Wang and Zhiping Xiao and Yuqing Wang and Chong Ruan and Ming Zhang and Wenfeng Liang and Wangding Zeng},
    year    = {2025},
    url     = {https://api.semanticscholar.org/CorpusID:276408911}
}
```