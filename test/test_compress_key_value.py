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
from native_sparse_attention.ops import linear_compress


if __name__ == "__main__":
    torch.manual_seed(42)
    num_heads = 4
    head_dim = 192
    kernel_size = 32
    kernel_stride = 16
    seqlens = torch.LongTensor([1000, 2000, 4096]).int().cuda()
    cu_seqlens = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device="cuda"),
            torch.cumsum(seqlens, dim=0),
        ],
        dim=0,
    ).to(torch.int32)

    x = (
        torch.zeros(cu_seqlens[-1], num_heads, head_dim)
        .uniform_(-1, 1)
        .cuda()
        .bfloat16()
        .requires_grad_()
    )
    w = (
        torch.zeros(num_heads, kernel_size * head_dim, head_dim)
        .uniform_(-1, 1)
        .cuda()
        .bfloat16()
        .requires_grad_()
    )
    pe = (
        torch.zeros(num_heads, kernel_size, head_dim)
        .uniform_(-1, 1)
        .cuda()
        .bfloat16()
        .requires_grad_()
    )

    y, y_cu_seqlens = linear_compress(x, w, cu_seqlens, kernel_size, kernel_stride, pe)

    loss = (y * torch.randn_like(y)).mean()
    loss.backward()

    print(y.shape, y_cu_seqlens)
    print(y.norm(), x.grad.norm())
    print(
        w.grad.norm() if w.grad is not None else None,
        pe.grad.norm() if pe.grad is not None else None,
    )

    # benchmark
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["N"],
            x_vals=[1024 * 2**i for i in range(1, 6)],
            line_arg="provider",
            line_vals=["batch1", "batch8", "batch32"],
            line_names=["batch1", "batch8", "batch32"],
            styles=[("green", "-"), ("blue", "-"), ("blue", "--")],
            ylabel="ms",
            plot_name="** forward **",
            args={"H": 4, "D": 128},
        )
    )
    def benchmark(N, H, D, provider):
        K, S = 32, 16
        x = torch.zeros(N, H, D, device="cuda", dtype=torch.bfloat16).uniform_(-1, 1)
        w = torch.zeros(H, K * D, D, device="cuda", dtype=torch.bfloat16).uniform_(
            -1, 1
        )
        pe = torch.zeros(H, K, D, device="cuda", dtype=torch.bfloat16).uniform_(-1, 1)
        cu_seqlens_b1 = torch.LongTensor([0, N]).int().cuda()
        cu_seqlens_b8 = (
            torch.LongTensor([N // 8 if i > 0 else 0 for i in range(9)]).int().cuda()
        )
        cu_seqlens_b32 = (
            torch.LongTensor([N // 32 if i > 0 else 0 for i in range(33)]).int().cuda()
        )
        cu_seqlens_b1 = cu_seqlens_b1.cumsum(0).to(torch.int32)
        cu_seqlens_b8 = cu_seqlens_b8.cumsum(0).to(torch.int32)
        cu_seqlens_b32 = cu_seqlens_b32.cumsum(0).to(torch.int32)

        quantiles = [0.5, 0.2, 0.8]
        if provider == "batch1":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: linear_compress(x, w, cu_seqlens_b1, K, S, pe),
                quantiles=quantiles,
            )
        if provider == "batch8":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: linear_compress(x, w, cu_seqlens_b8, K, S, pe),
                quantiles=quantiles,
            )
        if provider == "batch32":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: linear_compress(x, w, cu_seqlens_b32, K, S, pe),
                quantiles=quantiles,
            )
        return ms, min_ms, max_ms

    benchmark.run(show_plots=True, print_data=True)
