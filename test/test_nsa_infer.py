# Copyright 2025 Xunhao Lai & Jianqiao Lu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
from native_sparse_attention.ops import linear_compress, weightedpool_compress
from native_sparse_attention.module import NSACache, RotaryEmbedding, RopeConfig
from native_sparse_attention.infer import nsa_infer


if __name__ == "__main__":
    torch.manual_seed(42)

    num_heads = 4
    head_dim = 128
    kernel_size = 32
    kernel_stride = 16
    block_size = 64
    window_size = 512
    topk = 16
    init_blocks = 1
    local_blocks = 2

    # init seqlens
    seqlens = torch.tensor([12, 576, 12000]).to(torch.int32).cuda()
    batch_size = seqlens.shape[0]
    cu_seqlens = torch.zeros(seqlens.shape[0] + 1, dtype=torch.int32, device="cuda")
    cu_seqlens[1:] = seqlens.cumsum(0)
    step = 0

    # init cache and weight and rope
    cache = NSACache(4, 16384, num_heads, head_dim, 32, 16, 512, torch.bfloat16, "cuda")
    compress_weight = [
        torch.ones(num_heads, kernel_size * head_dim, head_dim).cuda().bfloat16()
        / (kernel_size * head_dim),
        torch.ones(num_heads, kernel_size).cuda().bfloat16() / kernel_size,
    ]
    compress_func = [linear_compress, weightedpool_compress]
    rope = RotaryEmbedding(
        RopeConfig(
            max_position_embeddings=131072,
            head_dim=128,
            rope_theta=500000,
            rope_scaling={
                "factor": 8.0,
                "high_freq_factor": 4.0,
                "low_freq_factor": 1.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3",
            },
        )
    )

    # test prefill
    q = torch.randn(cu_seqlens[-1], num_heads * 16, head_dim).cuda().bfloat16()
    k = torch.randn(cu_seqlens[-1], num_heads, head_dim).cuda().bfloat16()
    v = torch.randn_like(k)
    g = torch.rand(cu_seqlens[-1], num_heads * 16, 3).cuda().bfloat16()
    o = nsa_infer(
        cu_seqlens,
        step,
        q,
        k,
        v,
        g,
        rope,
        cache,
        compress_weight,
        compress_func,
        None,
        kernel_size,
        kernel_stride,
        block_size,
        topk,
        init_blocks,
        local_blocks,
        window_size,
    )
    print(o.shape, o.norm())

    # test decode
    q = torch.randn(cu_seqlens.shape[0] - 1, num_heads * 16, head_dim).cuda().bfloat16()
    k = torch.randn(cu_seqlens.shape[0] - 1, num_heads, head_dim).cuda().bfloat16()
    v = torch.randn_like(k)
    g = torch.rand(cu_seqlens.shape[0] - 1, num_heads * 16, 3).cuda().bfloat16()
    step = 1
    o = nsa_infer(
        cu_seqlens,
        step,
        q,
        k,
        v,
        g,
        rope,
        cache,
        compress_weight,
        compress_func,
        None,
        kernel_size,
        kernel_stride,
        block_size,
        topk,
        init_blocks,
        local_blocks,
        window_size,
    )
    print(o.shape, o.norm())
