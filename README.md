<div align="center">

# Native Sparse Attention Triton

</div>

This repository implements the sparse attention mechanism introduced in the paper [Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention](https://arxiv.org/abs/2502.11089) and provides an efficient training implementation based on [Triton](https://github.com/triton-lang/triton).

🎉 We now support both training and inference for Native Sparse Attention (variable-length version, including prefilling, decoding, and KV cache management). We have provided a toy model at `model.ToyNSALlama`, which supports `forward` function for training and `generate` function for inference. Welcome to try it out!

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

### Install

You can install `native_sparse_attention` using pip:

```shell
pip install git+https://github.com/XunhaoLai/native-sparse-attention-triton.git
```

### Functions

The `ops` module has implemented several functions required for native sparse attention. For detailed usage instructions, please see [this link](https://github.com/XunhaoLai/native-sparse-attention-triton/tree/main/native_sparse_attention/ops#readme).

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
    compress_type="linear",
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

### Model

We offer two simplified LLaMA models in the `model` directory, featuring self-attention and native sparse attention. For more details on how to use these models, please refer to [this link](https://github.com/XunhaoLai/native-sparse-attention-triton/tree/main/native_sparse_attention/model#readme).


```python
from native_sparse_attention.model import ToyNSALlamaConfig, InferenceConfig, ToyNSALlama

config = ToyNSALlamaConfig(
    hidden_size=4096,
    intermediate_size=14336,
    num_hidden_layers=8,
    num_attention_heads=32,
    num_key_value_heads=2,
    head_dim=128,
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
inference_config = InferenceConfig(
    max_batch_size=4,
    max_length=8192,
    max_new_tokens=128,
)
model = ToyNSALlama(config, inference_config).cuda().bfloat16()
```

## Testing

Some test scripts are available in the `test` folder and can be run directly for unit testing. For example:

```bash
python test/test_topk_sparse_attention.py
python test/test_nsa_module.py
python test/test_nsa_model.py
```

### Benchmarks

Here are the speed benchmarks conducted on a single NVIDIA A100 GPU or H100 GPU for the `topk_sparse_attention` function: 

A100 GPU speed benchmarks:
```sh
** forward with block size 64 **:
          N       Flash  Triton-Flash  Triton-Top8  Triton-Top16
0    2048.0     0.414144      0.635648     0.633440      1.009184
1    4096.0     1.400304      2.267552     1.179808      1.916736
2    8192.0     5.223776      8.528160     2.266816      3.723168
3   16384.0    20.225697     32.745537     4.468128      7.359168
4   32768.0    79.587715    128.951065     8.517440     14.142848
5   65536.0   321.240479    511.652100    17.249599     30.991360
6  131072.0  1349.810425   2063.245605    36.400482     67.884544

** backward with block size 64 **:
          N        Flash  Triton-Flash  Triton-Top8  Triton-Top16
0    2048.0     1.315440      2.348560     1.941568      2.691040
1    4096.0     4.271584      8.553184     3.647744      5.032160
2    8192.0    15.323984     32.665440     5.650144      9.066112
3   16384.0    58.753281    127.675964    11.160832     17.113279
4   32768.0   227.770462    504.572693    21.723392     34.715614
5   65536.0   899.181274   2059.718506    44.517181     76.309441
6  131072.0  3587.918701   8530.726562   105.344734    182.970169
```

H100 GPU benchmarks:
```sh
** forward with block size 64 **:
          N       Flash  Triton-Flash  Triton-Top8  Triton-Top16
0    2048.0    0.259552      0.293888     0.584544      0.917664
1    4096.0    0.846848      1.029904     1.094976      1.745136
2    8192.0    3.043744      3.843392     2.128256      3.396880
3   16384.0   11.743568     14.791360     4.190528      6.704192
4   32768.0   45.968513     57.532478     7.614496     12.417440
5   65536.0  187.234375    228.093948    14.840048     24.511856
6  131072.0  810.890381    914.693970    29.470400     48.990192

** backward with block size 64 **:
          N        Flash  Triton-Flash  Triton-Top8  Triton-Top16
0    2048.0     0.798976      1.096096     1.117312      1.380016
1    4096.0     2.545680      3.826336     1.669760      2.214880
2    8192.0     9.029760     14.411633     2.772096      3.947456
3   16384.0    34.144016     58.945698     5.201344      7.538912
4   32768.0   135.718369    233.369247     9.968864     15.154192
5   65536.0   541.053894    929.337646    21.089870     33.818878
6  131072.0  2139.974854   3785.540527    54.918144     93.750717
```

Here comes another speed benchmark result for testing `compressed_attention` function on a single NVIDIA A100 GPU or H100 GPU:

A100 GPU speed benchmarks:
```sh
** forward with kernel 32 and stride 16 **:
          N       Flash  Triton-Flash  Compressed  Compressed-wo-Score
0    2048.0     0.413664      0.635488    0.655024             0.170816
1    4096.0     1.396416      2.247648    1.132304             0.377152
2    8192.0     5.234656      8.526400    2.879200             0.977952
3   16384.0    19.988865     32.755199    9.426448             2.943024
4   32768.0    79.419907    128.955170   30.284096             9.901120
5   65536.0   321.590210    511.615509  112.260544            36.001602
6  131072.0  1346.996338   2069.837891  423.099518           136.820038

** backward with kernel 32 and stride 16 **:
          N        Flash  Triton-Flash  Compressed
0    2048.0     1.322560      2.352000    0.486784
1    4096.0     4.270832      8.552608    0.971392
2    8192.0    15.515680     32.671329    2.603744
3   16384.0    59.345055    128.377472    8.499456
4   32768.0   230.626144    506.581238   30.064833
5   65536.0   919.260498   2068.642578  113.466560
6  131072.0  3646.603760   8498.374023  439.623444
```

H100 GPU speed benchmarks:
```sh
** forward with kernel 32 and stride 16 **:
          N       Flash  Triton-Flash  Compressed  Compressed-wo-Score
0    2048.0    0.259488      0.297152    0.485920             0.103232
1    4096.0    0.847376      1.030400    0.710208             0.217760
2    8192.0    3.044016      3.875840    1.607360             0.516016
3   16384.0   11.823104     14.829360    4.970272             1.440288
4   32768.0   46.204750     57.527809   15.004992             4.584736
5   65536.0  187.324249    227.909958   53.009087            16.134224
6  131072.0  810.707214    910.106873  191.245728            60.154270

** backward with kernel 32 and stride 16 **:
          N        Flash  Triton-Flash  Compressed
0    2048.0     0.797728      1.090640    0.283104
1    4096.0     2.547088      3.834592    0.550464
2    8192.0     9.021520     14.421088    1.249184
3   16384.0    34.159508     58.793377    3.743440
4   32768.0   136.844070    233.447708   12.640032
5   65536.0   537.559814    929.360229   46.054817
6  131072.0  2135.629883   3782.351562  175.587296
```

All the speed benchmarks above were tested with 64 query heads, 4 key/value heads, and a head dimension of 128.

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
