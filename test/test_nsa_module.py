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
from native_sparse_attention.module import (
    SelfAttention,
    NativeSparseAttention,
    RopeConfig,
)


if __name__ == "__main__":
    torch.manual_seed(42)
    NSA = (
        NativeSparseAttention(
            compress_type="avgpool",
            hidden_size=8192,
            num_q_heads=64,
            num_kv_heads=4,
            head_dim=128,
            kernel_size=32,
            kernel_stride=16,
            block_size=64,
            topk=16,
            init_blocks=1,
            local_blocks=2,
            window_size=512,
            rope_config=RopeConfig(
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
            ),
        )
        .cuda()
        .to(torch.bfloat16)
    )
    print("======= Init Moduel: Native Sparse Attention =======\n")
    for name, param in NSA.named_parameters():
        print(f"NSA Parameters, {name}, shape: {param.shape}\n")

    # random input
    seqlens = torch.LongTensor([4000, 8192, 16384]).int().cuda()
    cu_seqlens = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device="cuda"),
            torch.cumsum(seqlens, dim=0),
        ],
        dim=0,
    ).to(torch.int32)
    x = torch.zeros(cu_seqlens[-1], 8192, device="cuda", dtype=torch.bfloat16).uniform_(
        -1, 1
    )

    # forward test
    print("======= NSA Forward & Backward Test =======\n")
    y = NSA(x, cu_seqlens)
    print(f"Forward, output shape: {y.shape}, output norm: {y.norm()}\n")

    # backward test
    loss = (y * torch.randn_like(y)).sum(-1).mean()
    loss.backward()
    for name, param in NSA.named_parameters():
        print(
            f"Backward, {name}, grad shape: {param.grad.shape}, grad norm: {param.grad.norm()}\n"
        )

    # speed benchmark
    SelfAttn = (
        SelfAttention(
            hidden_size=8192,
            num_q_heads=64,
            num_kv_heads=4,
            head_dim=128,
            rope_config=RopeConfig(
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
            ),
        )
        .cuda()
        .to(torch.bfloat16)
    )

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["N"],
            x_vals=[1024 * 2**i for i in range(1, 8)],
            line_arg="provider",
            line_vals=["Self-Attention", "Native-Sparse-Attention"],
            line_names=["Self-Attention", "Native-Sparse-Attention"],
            styles=[("green", "-"), ("blue", "-")],
            ylabel="ms",
            plot_name="** NSA forward speed benchmark **",
            args={},
        )
    )
    def benchmark(N, provider):
        x = torch.randn(N, 8192, device="cuda", dtype=torch.bfloat16)
        cu_seqlens = torch.tensor([0, N], device="cuda", dtype=torch.int32)
        quantiles = [0.5, 0.2, 0.8]
        with torch.no_grad():
            if provider == "Self-Attention":
                ms, min_ms, max_ms = triton.testing.do_bench(
                    lambda: SelfAttn(x, cu_seqlens),
                    quantiles=quantiles,
                )
            if provider == "Native-Sparse-Attention":
                ms, min_ms, max_ms = triton.testing.do_bench(
                    lambda: NSA(x, cu_seqlens),
                    quantiles=quantiles,
                )
        return ms, min_ms, max_ms

    benchmark.run(show_plots=True, print_data=True)

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["N"],
            x_vals=[1024 * 2**i for i in range(1, 8)],
            line_arg="provider",
            line_vals=["Self-Attention", "Native-Sparse-Attention"],
            line_names=["Self-Attention", "Native-Sparse-Attention"],
            styles=[("green", "-"), ("blue", "-")],
            ylabel="ms",
            plot_name="** NSA backward speed benchmark **",
            args={},
        )
    )
    def benchmark(N, provider):
        x = torch.randn(N, 8192, device="cuda", dtype=torch.bfloat16)
        cu_seqlens = torch.tensor([0, N], device="cuda", dtype=torch.int32)
        quantiles = [0.5, 0.2, 0.8]
        if provider == "Self-Attention":
            loss = SelfAttn(x.clone().detach().requires_grad_(), cu_seqlens).mean()
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: loss.backward(retain_graph=True),
                quantiles=quantiles,
            )
        elif provider == "Native-Sparse-Attention":
            loss = NSA(x.clone().detach().requires_grad_(), cu_seqlens).mean()
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: loss.backward(retain_graph=True),
                quantiles=quantiles,
            )
        return ms, min_ms, max_ms

    benchmark.run(show_plots=True, print_data=True)
