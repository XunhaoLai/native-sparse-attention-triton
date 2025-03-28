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
from native_sparse_attention.ops.torch.compressed_attention import (
    compressed_attention_torch,
)
from native_sparse_attention.ops.triton.compressed_attention import (
    compressed_attention,
    _compressed_attention_bwd,
)
from native_sparse_attention.ops import avgpool_compress
from native_sparse_attention.ops.triton.flash_attention import (
    flash_attention_varlen,
    _flash_attention_bwd,
)
from flash_attn import flash_attn_varlen_func
from flash_attn.flash_attn_interface import _flash_attn_varlen_backward


if __name__ == "__main__":
    torch.manual_seed(42)
    num_heads = 32
    head_dim = 96
    kernel_size = 32
    kernel_stride = 16
    block_size = 64
    topk = 16
    seqlens = torch.LongTensor([1000, 4000, 8192]).int().cuda()
    cu_seqlens = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device="cuda"),
            torch.cumsum(seqlens, dim=0),
        ],
        dim=0,
    ).to(torch.int32)
    max_seqlen = seqlens.max().item()
    q = (
        torch.empty(cu_seqlens[-1], num_heads, head_dim, device="cuda")
        .uniform_(-1, 1)
        .to(torch.float16)
    )
    k = (
        torch.empty(cu_seqlens[-1], num_heads // 4, head_dim, device="cuda")
        .uniform_(-1, 1)
        .to(torch.float16)
    )
    v = (
        torch.empty(cu_seqlens[-1], num_heads // 4, head_dim, device="cuda")
        .uniform_(-1, 1)
        .to(torch.float16)
    )
    q.requires_grad = True
    k.requires_grad = True
    v.requires_grad = True

    ck, ck_cu_seqlens = avgpool_compress(
        k, None, cu_seqlens, kernel_size, kernel_stride
    )

    ck = torch.empty_like(ck).uniform_(-1, 1)
    cv = torch.empty_like(ck).uniform_(-1, 1)
    ck.requires_grad = True
    cv.requires_grad = True

    ck_seqlens = ck_cu_seqlens[1:] - ck_cu_seqlens[:-1]
    ck_max_seqlen = ck_seqlens.max().item()

    o, topk_idx = compressed_attention_torch(
        q,
        ck,
        cv,
        kernel_size,
        kernel_stride,
        block_size,
        topk,
        cu_seqlens,
        ck_cu_seqlens,
        max_seqlen,
        ck_max_seqlen,
    )

    randn = torch.randn_like(o)
    loss = (o * randn).sum()
    loss.backward()

    torch.manual_seed(42)

    q1 = q.detach().clone().requires_grad_()
    ck1 = ck.detach().clone().requires_grad_()
    cv1 = cv.detach().clone().requires_grad_()

    o1, topk_idx1 = compressed_attention(
        q1,
        ck1,
        cv1,
        kernel_size,
        kernel_stride,
        block_size,
        topk,
        cu_seqlens,
        ck_cu_seqlens,
        max_seqlen,
        ck_max_seqlen,
    )
    randn1 = randn.clone().detach()
    loss1 = (o1 * randn1).sum()
    loss1.backward()

    print("Same Output:", torch.allclose(o, o1, atol=0.01, rtol=0.01))
    print("Max Error:", (o - o1).abs().max().item())
    print()
    print("Same Query Gradient:", torch.allclose(q.grad, q1.grad, atol=0.01, rtol=0.01))
    print("Max Query Gradient Error:", (q.grad - q1.grad).abs().max().item())
    print()
    print("Same Key Gradient:", torch.allclose(ck.grad, ck1.grad, atol=0.01, rtol=0.01))
    print("Max Key Gradient Error:", (ck.grad - ck1.grad).abs().max().item())
    print()
    print(
        "Same Value Gradient:", torch.allclose(cv.grad, cv1.grad, atol=0.01, rtol=0.01)
    )
    print("Max Value Gradient Error:", (cv.grad - cv1.grad).abs().max().item())
    print()

    # There are some discrepancies in the topk indices (about 3%). These might be due to bugs and will be addressed later.
    all_num = 0
    err_num = 0
    for h in range(topk_idx.shape[0]):
        for i in range(topk_idx.shape[1]):
            s = set(topk_idx[h, i][topk_idx[h, i] >= 0].tolist())
            s1 = set(topk_idx1[h, i][topk_idx1[h, i] >= 0].tolist())
            all_num += len(s)
            err_num += len(s) - len(s1 & s)
    print("Topk Idx Error Rate:", err_num / all_num)

    # benchmark
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["N"],
            x_vals=[1024 * 2**i for i in range(1, 8)],
            line_arg="provider",
            line_vals=[
                "flash",
                "triton-flash",
                "triton-compressed",
                "triton-compressed-wo-score",
            ],
            line_names=[
                "Flash",
                "Triton-Flash",
                "Compressed",
                "Compressed-wo-Score",
            ],
            styles=[("green", "-"), ("green", "--"), ("blue", "-"), ("blue", "--")],
            ylabel="ms",
            plot_name="** forward speed for compressed attention (kernel 32 stride 16) **",
            args={"H": 64, "D": 128},
        )
    )
    def benchmark(N, H, D, provider):
        q = torch.randn((N, H, D), device="cuda", dtype=torch.bfloat16)
        k = torch.randn((N, H // 16, D), device="cuda", dtype=torch.bfloat16)
        v = torch.randn((N, H // 16, D), device="cuda", dtype=torch.bfloat16)
        cu_seqlens = torch.tensor([0, N], device="cuda", dtype=torch.int32)
        sm_scale = 1 / math.sqrt(D)
        com_k, com_cu_seqlens = avgpool_compress(k, None, cu_seqlens, 32, 16, None)
        com_v, com_cu_seqlens = avgpool_compress(v, None, cu_seqlens, 32, 16, None)
        M = (com_cu_seqlens[1:] - com_cu_seqlens[:-1]).max().item()

        quantiles = [0.5, 0.2, 0.8]
        if provider == "flash":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: flash_attn_varlen_func(
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
                lambda: flash_attention_varlen(
                    q, k, v, cu_seqlens, cu_seqlens, N, N, True, sm_scale
                ),
                quantiles=quantiles,
            )
        if provider == "triton-compressed":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: compressed_attention(
                    q,
                    com_k,
                    com_v,
                    32,
                    16,
                    64,
                    16,
                    cu_seqlens,
                    com_cu_seqlens,
                    N,
                    M,
                    sm_scale,
                ),
                quantiles=quantiles,
            )
        if provider == "triton-compressed-wo-score":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: compressed_attention(
                    q,
                    com_k,
                    com_v,
                    32,
                    16,
                    64,
                    -1,
                    cu_seqlens,
                    com_cu_seqlens,
                    N,
                    M,
                    sm_scale,
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
            line_vals=[
                "flash",
                "triton-flash",
                "triton-compressed",
            ],
            line_names=[
                "Flash",
                "Triton-Flash",
                "Compressed",
            ],
            styles=[("green", "-"), ("green", "--"), ("blue", "-"), ("blue", "--")],
            ylabel="ms",
            plot_name="** backward speed for compressed attention (kernel 32 stride 16) **",
            args={"H": 64, "D": 128},
        )
    )
    def benchmark(N, H, D, provider):
        q = torch.randn((N, H, D), device="cuda", dtype=torch.bfloat16)
        k = torch.randn((N, H // 16, D), device="cuda", dtype=torch.bfloat16)
        v = torch.randn((N, H // 16, D), device="cuda", dtype=torch.bfloat16)
        o = torch.randn((N, H, D), device="cuda", dtype=torch.bfloat16)
        do = torch.randn((N, H, D), device="cuda", dtype=torch.bfloat16)
        lse = torch.randn((H, N), device="cuda", dtype=torch.float32)
        sm_scale = 1 / math.sqrt(D)
        cu_seqlens = torch.tensor([0, N], device="cuda", dtype=torch.int32)
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        com_k, com_cu_seqlens = avgpool_compress(k, None, cu_seqlens, 32, 16, None)
        com_v, com_cu_seqlens = avgpool_compress(v, None, cu_seqlens, 32, 16, None)
        M = (com_cu_seqlens[1:] - com_cu_seqlens[:-1]).max().item()

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
        if provider == "triton-compressed":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: _compressed_attention_bwd(
                    o,
                    do,
                    lse,
                    q,
                    com_k,
                    com_v,
                    32,
                    16,
                    cu_seqlens,
                    com_cu_seqlens,
                    N,
                    M,
                    sm_scale,
                ),
                quantiles=quantiles,
            )
        return ms, min_ms, max_ms

    benchmark.run(show_plots=True, print_data=True)
