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
0    2048.0    0.247792      0.350080     0.358848      0.549056
1    4096.0    0.761984      1.189184     0.631840      1.003584
2    8192.0    2.759744      4.376128     1.178512      1.912064
3   16384.0   10.384128     16.663263     2.275392      3.737120
4   32768.0   40.459824     65.101570     4.255344      7.108576
5   65536.0  162.824539    258.214478     8.304832     14.227424
6  131072.0  679.648376   1042.490845    17.316799     31.072479

** backward with block size 64 **:
          N        Flash  Triton-Flash  Triton-Top8  Triton-Top16
0    2048.0     0.733824      1.209280     1.336128      1.766240
1    4096.0     2.251104      4.325680     1.912032      2.675632
2    8192.0     7.898272     16.339600     3.182112      4.745376
3   16384.0    29.623199     63.897568     5.925872      9.035536
4   32768.0   119.139870    253.116516    11.397137     18.289024
5   65536.0   461.959900   1009.080872    23.698000     39.007843
6  131072.0  1834.623779   4359.882812    61.211681    105.494911
```

H100 GPU benchmarks:
```sh
** forward with block size 64 **:
          N       Flash  Triton-Flash  Triton-Top8  Triton-Top16
0    2048.0    0.159200      0.165728     0.325360      0.491968
1    4096.0    0.476032      0.559616     0.583520      0.908928
2    8192.0    1.617632      2.013248     1.095936      1.732896
3   16384.0    5.968144      7.722304     2.116128      3.390864
4   32768.0   23.162865     29.995071     3.831808      6.267936
5   65536.0   96.067169    117.449501     7.424608     12.260143
6  131072.0  409.013336    467.084351    14.549168     24.249104

** backward with block size 64 **:
          N        Flash  Triton-Flash  Triton-Top8  Triton-Top16
0    2048.0     0.455232      0.549344     0.881184      0.982032
1    4096.0     1.370976      1.899680     1.188448      1.390112
2    8192.0     4.660448      7.084544     1.754016      2.264096
3   16384.0    17.109247     27.394367     3.124768      4.203744
4   32768.0    66.424797    107.965630     5.839104      8.373632
5   65536.0   271.946869    466.106720    12.282047     18.128288
6  131072.0  1064.834229   1873.626465    36.218735     59.914242
```

Here comes another speed benchmark result for testing `compressed_attention` function on a single NVIDIA A100 GPU or H100 GPU:

A100 GPU speed benchmarks:
```sh
** forward with kernel 32 and stride 16 **:
          N       Flash  Triton-Flash  Compressed  Compressed-wo-Score
0    2048.0    0.247776      0.349792    0.836448             0.093056
1    4096.0    0.763008      1.190032    0.950176             0.205088
2    8192.0    2.755104      4.393216    1.887968             0.510272
3   16384.0   10.367872     16.659361    5.605536             1.501696
4   32768.0   40.275154     65.112129   18.290815             5.011456
5   65536.0  162.694016    257.520844   67.459999            18.133345
6  131072.0  681.280579   1042.662964  256.283569            68.728767

** backward with kernel 32 and stride 16 **:
          N        Flash  Triton-Flash  Compressed
0    2048.0     0.732208      1.206528    0.375584
1    4096.0     2.266496      4.327280    0.744544
2    8192.0     7.982176     16.375648    1.560896
3   16384.0    29.987295     63.860863    4.504832
4   32768.0   117.116608    253.300415   15.235985
5   65536.0   458.011871   1009.457336   56.966946
6  131072.0  1850.674683   4345.405273  224.101349
```

H100 GPU speed benchmarks:
```sh
** forward with kernel 32 and stride 16 **:
          N       Flash  Triton-Flash  Compressed  Compressed-wo-Score
0    2048.0    0.160192      0.168032    0.612640             0.055552
1    4096.0    0.477952      0.554848    0.676272             0.117600
2    8192.0    1.620480      2.016064    1.157312             0.272480
3   16384.0    5.943648      7.568608    3.143920             0.731776
4   32768.0   23.243376     29.755199    9.849648             2.311520
5   65536.0  101.783905    117.467422   35.026608             8.185743
6  131072.0  403.563995    466.646332  129.695496            30.363457

** backward with kernel 32 and stride 16 **:
          N        Flash  Triton-Flash  Compressed
0    2048.0     0.459872      0.550368    0.213248
1    4096.0     1.374704      1.907616    0.406656
2    8192.0     4.670624      7.099840    0.826160
3   16384.0    17.159361     27.210720    1.916064
4   32768.0    66.499390    107.536163    6.418336
5   65536.0   269.522186    466.093445   23.238575
6  131072.0  1070.837158   1876.903320   88.790909
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
