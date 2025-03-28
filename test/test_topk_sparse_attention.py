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
# See the License for the specific
import torch
import triton
import math
from native_sparse_attention.ops.torch.topk_sparse_attention import (
    topk_sparse_attention_torch,
)
from native_sparse_attention.ops.triton.topk_sparse_attention import (
    topk_sparse_attention,
    _topk_sparse_attention_fwd,
    _topk_sparse_attention_bwd,
)
from native_sparse_attention.ops.triton.flash_attention import (
    _flash_attention_fwd,
    _flash_attention_bwd,
)
from flash_attn.flash_attn_interface import (
    _flash_attn_varlen_forward,
    _flash_attn_varlen_backward,
)


def generate_topk_idx_example(
    seqlens: torch.Tensor,
    block_size_k: int,
    topk: int,
    num_heads: int,
    block_size_q: int = 1,
) -> torch.Tensor:
    """Generate topk idx example for test.

    Args:
        seqlens (torch.Tensor): shape [batch_size + 1], similar to cu_seqlens in flash_attn_func_varlen.
        block_size_q (int): query block size
        block_size_k (int): key value block size
        topk (int): selected topk
        num_heads (int): number of key value heads

    Returns:
        torch.Tensor: shape [num_heads, total_seqlen, topk], topk key value block idx for each query. -1 means padding.
    """
    batch_size = seqlens.shape[0]
    num_blocks = torch.ceil(seqlens / block_size_k).to(torch.int32)
    topk_idx_all_heads = []
    cu_seqlens = torch.nn.functional.pad(seqlens.cumsum(0), pad=(1, 0), value=0)
    for _ in range(num_heads):
        topk_idx = [
            torch.randn(seqlens[i], num_blocks[i], device="cuda")
            .topk(min(topk, num_blocks[i]), dim=-1)
            .indices.to(torch.int32)
            for i in range(batch_size)
        ]
        topk_idx = [
            torch.nn.functional.pad(
                topk_idx[i], (0, topk - topk_idx[i].shape[-1]), value=topk
            )
            for i in range(batch_size)
        ]
        topk_idx = torch.cat(topk_idx, dim=0)
        topk_idx = torch.sort(topk_idx, dim=1).values
        topk_idx[:, 0] = 0
        q_idx = torch.cat(
            [torch.arange(seqlens[i], device="cuda") for i in range(batch_size)], dim=0
        )
        topk_idx[topk_idx > (q_idx // block_size_k)[:, None]] = -1  # -1 means padding
        topk_idx = torch.cat(
            [
                topk_idx[cu_seqlens[i] : cu_seqlens[i + 1]][0::block_size_q]
                for i in range(batch_size)
            ],
            dim=0,
        )
        topk_idx_all_heads.append(topk_idx)
    topk_idx = torch.stack(topk_idx_all_heads, dim=0)
    return topk_idx


if __name__ == "__main__":
    torch.manual_seed(42)
    batch_size = 3
    seqlens = torch.LongTensor([1000, 2000, 4096]).int().cuda()
    cu_seqlens = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device="cuda"),
            torch.cumsum(seqlens, dim=0),
        ],
        dim=0,
    ).to(torch.int32)
    max_seqlen = seqlens.max().item()
    q = (
        torch.empty(cu_seqlens[-1], 64, 96, device="cuda")
        .uniform_(-1, 1)
        .to(torch.float16)
    )
    k = (
        torch.empty(cu_seqlens[-1], 8, 96, device="cuda")
        .uniform_(-1, 1)
        .to(torch.float16)
    )
    v = (
        torch.empty(cu_seqlens[-1], 8, 96, device="cuda")
        .uniform_(-1, 1)
        .to(torch.float16)
    )
    q.requires_grad = True
    k.requires_grad = True
    v.requires_grad = True
    block_size = 64
    topk = 5
    topk_idx = generate_topk_idx_example(seqlens, block_size, topk, 8)

    o = topk_sparse_attention_torch(q, k, v, topk_idx, block_size, cu_seqlens)

    randn = torch.randn_like(o)
    loss = (o * randn).sum()
    loss.backward()

    torch.manual_seed(42)
    q1 = q.clone().detach().requires_grad_()
    k1 = k.clone().detach().requires_grad_()
    v1 = v.clone().detach().requires_grad_()
    topk_idx1 = topk_idx.clone().detach()
    cu_seqlens1 = cu_seqlens.clone().detach()

    o1 = topk_sparse_attention(q1, k1, v1, topk_idx, block_size, cu_seqlens)

    randn2 = randn.clone().detach()
    loss2 = (o1 * randn2).sum()
    loss2.backward()

    print("Same Output:", torch.allclose(o, o1, atol=0.01, rtol=0.01))
    print("Max Error:", (o - o1).abs().max().item())
    print()
    print("Same Query Gradient:", torch.allclose(q.grad, q1.grad, atol=0.01, rtol=0.01))
    print("Max Query Gradient Error:", (q.grad - q1.grad).abs().max().item())
    print()
    print("Same Key Gradient:", torch.allclose(k.grad, k1.grad, atol=0.01, rtol=0.01))
    print("Max Key Gradient Error:", (k.grad - k1.grad).abs().max().item())
    print()
    print("Same Value Gradient:", torch.allclose(v.grad, v1.grad, atol=0.01, rtol=0.01))
    print("Max Value Gradient Error:", (v.grad - v1.grad).abs().max().item())
    print()

    # benchmark
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["N"],
            x_vals=[1024 * 2**i for i in range(1, 8)],
            line_arg="provider",
            line_vals=["flash", "triton-flash", "triton-top8", "triton-top16"],
            line_names=[
                "Flash",
                "Triton-Flash",
                "Triton-Top8",
                "Triton-Top16",
            ],
            styles=[("green", "-"), ("green", "--"), ("blue", "-"), ("blue", "--")],
            ylabel="ms",
            plot_name="** forward with block size 64 **",
            args={"H": 64, "D": 128, "K": 64},
        )
    )
    def benchmark(N, H, D, K, provider):
        q = torch.randn((N, H, D), device="cuda", dtype=torch.bfloat16)
        k = torch.randn((N, H // 16, D), device="cuda", dtype=torch.bfloat16)
        v = torch.randn((N, H // 16, D), device="cuda", dtype=torch.bfloat16)
        cu_seqlens = torch.tensor([0, N], device="cuda", dtype=torch.int32)
        sm_scale = 1 / math.sqrt(D)

        top8_idx = generate_topk_idx_example(cu_seqlens[1:], K, 8, H // 16)
        top16_idx = generate_topk_idx_example(cu_seqlens[1:], K, 16, H // 16)

        quantiles = [0.5, 0.2, 0.8]
        if provider == "flash":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: _flash_attn_varlen_forward(
                    q,
                    k,
                    v,
                    cu_seqlens,
                    cu_seqlens,
                    N,
                    N,
                    dropout_p=0.0,
                    causal=True,
                    softmax_scale=sm_scale,
                ),
                quantiles=quantiles,
            )
        if provider == "triton-flash":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: _flash_attention_fwd(
                    q, k, v, cu_seqlens, cu_seqlens, N, N, True, sm_scale
                ),
                quantiles=quantiles,
            )
        if provider == "triton-top8":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: _topk_sparse_attention_fwd(
                    q, k, v, top8_idx, K, cu_seqlens, cu_seqlens, N, N, sm_scale
                ),
                quantiles=quantiles,
            )
        if provider == "triton-top16":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: _topk_sparse_attention_fwd(
                    q, k, v, top16_idx, K, cu_seqlens, cu_seqlens, N, N, sm_scale
                ),
                quantiles=quantiles,
            )
        return ms, min_ms, max_ms

    benchmark.run(show_plots=True, print_data=True)

    # benchmark
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["N"],
            x_vals=[1024 * 2**i for i in range(1, 8)],
            line_arg="provider",
            line_vals=["flash", "triton-flash", "triton-top8", "triton-top16"],
            line_names=[
                "Flash",
                "Triton-Flash",
                "Triton-Top8",
                "Triton-Top16",
            ],
            styles=[("green", "-"), ("green", "--"), ("blue", "-"), ("blue", "--")],
            ylabel="ms",
            plot_name="** backward with block size 64 **",
            args={"H": 64, "D": 128, "K": 64},
        )
    )
    def benchmark(N, H, D, K, provider):
        q = torch.randn((N, H, D), device="cuda", dtype=torch.bfloat16)
        k = torch.randn((N, H // 16, D), device="cuda", dtype=torch.bfloat16)
        v = torch.randn((N, H // 16, D), device="cuda", dtype=torch.bfloat16)
        o = torch.randn((N, H, D), device="cuda", dtype=torch.bfloat16)
        do = torch.randn((N, H, D), device="cuda", dtype=torch.bfloat16)
        lse = torch.randn((H, N), device="cuda", dtype=torch.float32)
        sm_scale = 1 / math.sqrt(D)
        cu_seqlens = torch.tensor([0, N], device="cuda", dtype=torch.int32)
        top8_idx = generate_topk_idx_example(cu_seqlens[1:], K, 8, H // 16)
        top16_idx = generate_topk_idx_example(cu_seqlens[1:], K, 16, H // 16)
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        quantiles = [0.5, 0.2, 0.8]
        if provider == "flash":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: _flash_attn_varlen_backward(
                    do,
                    q,
                    k,
                    v,
                    o,
                    lse.transpose(0, 1),
                    dq,
                    dk,
                    dv,
                    cu_seqlens,
                    cu_seqlens,
                    N,
                    N,
                    dropout_p=0.0,
                    causal=True,
                    softmax_scale=sm_scale,
                    window_size=(-1, -1),
                    softcap=0.0,
                    alibi_slopes=None,
                    deterministic=False,
                ),
                quantiles=quantiles,
            )
        if provider == "triton-flash":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: _flash_attention_bwd(
                    o, do, lse, q, k, v, cu_seqlens, cu_seqlens, N, N, True, sm_scale
                ),
                quantiles=quantiles,
            )
        if provider == "triton-top8":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: _topk_sparse_attention_bwd(
                    o,
                    do,
                    lse,
                    q,
                    k,
                    v,
                    top8_idx,
                    K,
                    cu_seqlens,
                    cu_seqlens,
                    N,
                    N,
                    sm_scale,
                ),
                quantiles=quantiles,
            )
        if provider == "triton-top16":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: _topk_sparse_attention_bwd(
                    o,
                    do,
                    lse,
                    q,
                    k,
                    v,
                    top16_idx,
                    K,
                    cu_seqlens,
                    cu_seqlens,
                    N,
                    N,
                    sm_scale,
                ),
                quantiles=quantiles,
            )
        return ms, min_ms, max_ms

    benchmark.run(show_plots=True, print_data=True)
