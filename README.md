<div align="center">

# Native Sparse Attention Triton

</div>

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
import torch
from native_sparse_attention.ops import linear_compress, compressed_attention, topk_sparse_attention

# input example
num_q_heads = 64
num_kv_heads = 4
head_dim = 128
kernel_size = 32
kernel_stride = 16
block_size = 64
topk = 16
cu_seqlens = torch.Tensor([0, 1024, 8192, 16384]).to(torch.int32).cuda()
query = torch.randn(16384, num_q_heads, head_dim).to(torch.bfloat16).cuda()
key = torch.randn(16384, num_kv_heads, head_dim).to(torch.bfloat16).cuda()
value = torch.randn(16384, num_kv_heads, head_dim).to(torch.bfloat16).cuda()

# weight example
w = (
    torch.randn(num_kv_heads, kernel_size * head_dim, head_dim)
    .to(torch.bfloat16)
    .cuda()
)
pe = torch.randn(num_kv_heads, kernel_size, head_dim).to(torch.bfloat16).cuda()

# 1. key value compression
compressed_key, compressed_cu_seqlens = linear_compress(
    key, w, cu_seqlens, kernel_size, kernel_stride, pe
)
compressed_value, _ = linear_compress(
    value, w, cu_seqlens, kernel_size, kernel_stride, None
)

# 2. attention between query and compressed key value
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

# 3. topk sparse attention
sparse_attn_output = topk_sparse_attention(
    query,
    key,
    value,
    topk_idx,
    block_size,
    cu_seqlens,
)
```

### Module

The `modules` directory also provides implementations based on `torch.nn.module` for easy integration into models.

```python
from native_sparse_attention.modules import NativeSparseAttention, RopeConfig

NSA_Layer = NativeSparseAttention(
    hidden_size=4096,
    num_q_heads=64,
    num_kv_heads=4,
    head_dim=128,
    kernel_size=32,
    kernel_stride=16,
    block_size=64,
    topk=8,
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

### Benchmarks

Here are the speed benchmarks conducted on a single NVIDIA A100 GPU for the `topk_sparse_attention` function: 

```sh
** forward with block size 64 **:
         N      Flash  Triton-Flash  Triton-Top8  Triton-Top16
0   2048.0   0.247232      0.365920     0.273440      0.456608
1   4096.0   0.760672      1.226640     0.521120      0.874240
2   8192.0   2.706912      4.451872     1.009248      1.710320
3  16384.0  10.247040     16.811008     1.997696      3.388352
4  32768.0  39.650383     65.532028     3.978768      6.744656

** backward with block size 64 **:
         N       Flash  Triton-Flash  Triton-Top8  Triton-Top16
0   2048.0    0.741216      1.752576     1.270288      1.646496
1   4096.0    2.264448      6.235072     1.861792      2.622128
2   8192.0    7.818064     24.082623     3.102800      4.736192
3  16384.0   29.267200     97.103806     5.942896      9.158496
4  32768.0  114.074753    384.978882    11.771808     18.923136
```

Here comes another speed benchmark result for testing `compressed_attention` function:
```sh
** forward speed for compressed attention (kernel 32 stride 16) **:
         N      Flash  Triton-Flash  Compressed  Compressed-wo-Score
0   2048.0   0.245216      0.333344    0.702752             0.081600
1   4096.0   0.760288      1.129856    0.792752             0.182816
2   8192.0   2.745728      4.138080    1.719584             0.470144
3  16384.0  10.412096     15.776016    5.314288             1.443296
4  32768.0  40.484608     61.347775   17.406656             4.967680
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
