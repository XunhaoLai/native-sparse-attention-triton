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
from native_sparse_attention.ops.triton.flash_attention import (
    flash_attention_varlen,
    _flash_attention_fwd,
    _flash_attention_bwd,
)
from flash_attn import flash_attn_varlen_func
from flash_attn.flash_attn_interface import (
    _flash_attn_varlen_forward,
    _flash_attn_varlen_backward,
)
from xperf_gpt_custom.plugins.linear_attn.linear_attn.ops.gated_delta_rule.chunk import (
    chunk_gated_delta_rule,
)


def mixheadlinear(
    q,
    k,
    v,
    cu_seqlens,
    N,
    sm_scale,
    H_linear,
    g,
    beta,
):
    gqa = q.shape[1] // k.shape[1]
    _flash_attention_fwd(
        q[:, :-H_linear, ...],
        k[:, :-H_linear // gqa, ...],
        v[:, :-H_linear // gqa, ...],
        cu_seqlens,
        cu_seqlens,
        N,
        N,
        True,
        sm_scale,
    ),
    chunk_gated_delta_rule(
        q.unsqueeze(0)[:, :, -H_linear:, ...],
        k.unsqueeze(0)[:, :, -H_linear // gqa:, ...],
        v.unsqueeze(0)[:, :, -H_linear // gqa:, ...],
        g=g[..., -H_linear:],
        beta=beta[..., -H_linear:],
        initial_state=None,
        output_final_state=True,
        head_first=False,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
    ),


if __name__ == "__main__":
    # for causal in [False, True]:
    #     torch.manual_seed(42)

    #     # flash attention
    #     q = torch.randn(
    #         1000, 32, 128, dtype=torch.float16, device="cuda", requires_grad=True
    #     )
    #     k = torch.randn(
    #         1000, 16, 128, dtype=torch.float16, device="cuda", requires_grad=True
    #     )
    #     v = torch.randn(
    #         1000, 16, 128, dtype=torch.float16, device="cuda", requires_grad=True
    #     )
    #     cu_seqlens_q = torch.Tensor([0, 100, 384, 1000]).cuda().to(torch.int32)
    #     cu_seqlens_k = torch.Tensor([0, 100, 384, 1000]).cuda().to(torch.int32)
    #     max_seqlen_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max()
    #     max_seqlen_k = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).max()
    #     o = flash_attn_varlen_func(
    #         q,
    #         k,
    #         v,
    #         cu_seqlens_q,
    #         cu_seqlens_k,
    #         max_seqlen_q,
    #         max_seqlen_k,
    #         causal=causal,
    #     )
    #     randn = torch.randn_like(o)
    #     loss = (o * randn).sum()
    #     loss.backward()

    #     # triton flash attention
    #     torch.manual_seed(42)
    #     q1 = q.clone().detach().requires_grad_()
    #     k1 = k.clone().detach().requires_grad_()
    #     v1 = v.clone().detach().requires_grad_()
    #     cu_seqlens_q1 = cu_seqlens_q.clone().detach()
    #     cu_seqlens_k1 = cu_seqlens_k.clone().detach()
    #     max_seqlen_q1 = (cu_seqlens_q1[1:] - cu_seqlens_q1[:-1]).max()
    #     max_seqlen_k1 = (cu_seqlens_k1[1:] - cu_seqlens_k1[:-1]).max()
    #     o1 = flash_attention_varlen(
    #         q1,
    #         k1,
    #         v1,
    #         cu_seqlens_q1,
    #         cu_seqlens_k1,
    #         max_seqlen_q1,
    #         max_seqlen_k1,
    #         causal=causal,
    #     )
    #     randn2 = randn.clone().detach()
    #     loss2 = (o1 * randn2).sum()
    #     loss2.backward()

    # # diff
    # print(
    #     f"=== Flash Attention Backward Test ({'causal' if causal else 'full'}) ==="
    # )
    # print("Same Output:", torch.allclose(o, o1, atol=0.01, rtol=0.01))
    # print("Max Error:", (o - o1).abs().max().item())
    # print()
    # print(
    #     "Same Query Gradient:",
    #     torch.allclose(q.grad, q1.grad, atol=0.01, rtol=0.01),
    # )
    # print("Max Query Gradient Error:", (q.grad - q1.grad).abs().max().item())
    # print()
    # print(
    #     "Same Key Gradient:", torch.allclose(k.grad, k1.grad, atol=0.01, rtol=0.01)
    # )
    # print("Max Key Gradient Error:", (k.grad - k1.grad).abs().max().item())
    # print()
    # print(
    #     "Same Value Gradient:",
    #     torch.allclose(v.grad, v1.grad, atol=0.01, rtol=0.01),
    # )
    # print("Max Value Gradient Error:", (v.grad - v1.grad).abs().max().item())
    # print()

    # benchmark
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["N"],
            x_vals=[1024 * 2**i for i in range(1, 10)],
            line_arg="provider",
            line_vals=[
                "flash",
                "triton-flash",
                "triton-linear",
                "triton-mixheadlinear",
            ],
            line_names=[
                "Flash(CUDA)",
                "Triton-Flash",
                "Triton-Linear",
                "Triton-MixLinear(3:1)",
            ],
            styles=[("green", "-"), ("green", "--"), ("blue", "--"), ("red", "--")],
            ylabel="ms",
            plot_name="** forward **",
            args={"H": 24, "D": 128},
        )
    )
    def benchmark(N, H, D, provider):
        q = torch.randn((N, H, D), device="cuda", dtype=torch.bfloat16)
        k = torch.randn((N, H // 6, D), device="cuda", dtype=torch.bfloat16)
        v = torch.randn((N, H // 6, D), device="cuda", dtype=torch.bfloat16)
        g = torch.randn(1, N, H, device="cuda", dtype=torch.bfloat16)
        beta = torch.randn(1, N, H, device="cuda", dtype=torch.bfloat16)
        H_linear = H // 4 * 3
        cu_seqlens = torch.tensor([0, N], device="cuda", dtype=torch.int32)
        sm_scale = 1 / math.sqrt(D)

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
                lambda: _flash_attention_fwd(
                    q, k, v, cu_seqlens, cu_seqlens, N, N, True, sm_scale
                ),
                quantiles=quantiles,
            )
        if provider == "triton-linear":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: chunk_gated_delta_rule(
                    q.unsqueeze(0),
                    k.unsqueeze(0),
                    v.unsqueeze(0),
                    g=g,
                    beta=beta,
                    initial_state=None,
                    output_final_state=True,
                    head_first=False,
                    cu_seqlens=cu_seqlens,
                    use_qk_l2norm_in_kernel=True,
                ),
                quantiles=quantiles,
            )

        if provider == "triton-mixheadlinear":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: mixheadlinear(
                    q,
                    k,
                    v,
                    cu_seqlens,
                    N,
                    sm_scale,
                    H_linear,
                    g,
                    beta,
                ),
                quantiles=quantiles,
            )
        return ms, min_ms, max_ms

    benchmark.run(show_plots=True, print_data=True)

    # # benchmark
    # @triton.testing.perf_report(
    #     triton.testing.Benchmark(
    #         x_names=["N"],
    #         x_vals=[1024 * 2**i for i in range(1, 6)],
    #         line_arg="provider",
    #         line_vals=["flash", "triton-flash"],
    #         line_names=[
    #             "Flash",
    #             "Triton-Flash",
    #         ],
    #         styles=[("green", "-"), ("green", "--")],
    #         ylabel="ms",
    #         plot_name="** backward **",
    #         args={"H": 32, "D": 128},
    #     )
    # )
    # def benchmark(N, H, D, provider):
    #     q = torch.randn((N, H, D), device="cuda", dtype=torch.bfloat16)
    #     k = torch.randn((N, H // 16, D), device="cuda", dtype=torch.bfloat16)
    #     v = torch.randn((N, H // 16, D), device="cuda", dtype=torch.bfloat16)
    #     o = torch.randn((N, H, D), device="cuda", dtype=torch.bfloat16)
    #     do = torch.randn((N, H, D), device="cuda", dtype=torch.bfloat16)
    #     lse = torch.randn((N, H), device="cuda", dtype=torch.bfloat16)
    #     sm_scale = 1 / math.sqrt(D)
    #     cu_seqlens = torch.tensor([0, N], device="cuda", dtype=torch.int32)
    #     dq = torch.zeros_like(q)
    #     dk = torch.zeros_like(k)
    #     dv = torch.zeros_like(v)

    #     quantiles = [0.5, 0.2, 0.8]
    #     if provider == "flash":
    #         ms, min_ms, max_ms = triton.testing.do_bench(
    #             lambda: _flash_attn_varlen_backward(
    #                 do,
    #                 q,
    #                 k,
    #                 v,
    #                 o,
    #                 lse.transpose(0, 1),
    #                 dq,
    #                 dk,
    #                 dv,
    #                 cu_seqlens,
    #                 cu_seqlens,
    #                 N,
    #                 N,
    #                 dropout_p=0.0,
    #                 causal=True,
    #                 softmax_scale=sm_scale,
    #                 window_size=(-1, -1),
    #                 softcap=0.0,
    #                 alibi_slopes=None,
    #                 deterministic=False,
    #             ),
    #             quantiles=quantiles,
    #         )
    #     if provider == "triton-flash":
    #         ms, min_ms, max_ms = triton.testing.do_bench(
    #             lambda: _flash_attention_bwd(
    #                 o, do, lse, q, k, v, cu_seqlens, cu_seqlens, N, N, True, sm_scale
    #             ),
    #             quantiles=quantiles,
    #         )
    #     return ms, min_ms, max_ms

    # benchmark.run(show_plots=True, print_data=True)
