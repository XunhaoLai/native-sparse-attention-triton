# Copyright 2026 Xunhao Lai & Jianqiao Lu.
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
"""Backward regression tests for flash_attention_varlen against a torch
reference, covering the configurations the flash_attn-based script cannot:
gqa_interleave with non-square head ratios, and split qk/v head dimensions."""
import torch

from native_sparse_attention.ops import flash_attention_varlen


def torch_ref_varlen(q, k, v, cu_seqlens, causal, gqa_interleave):
    total_q, num_q_heads, qk_head_dim = q.shape
    num_kv_heads, v_head_dim = k.shape[1], v.shape[2]
    share = num_q_heads // num_kv_heads
    sm_scale = 1.0 / (qk_head_dim**0.5)
    out = torch.zeros(total_q, num_q_heads, v_head_dim, device=q.device, dtype=torch.float32)
    for b in range(len(cu_seqlens) - 1):
        s, e = cu_seqlens[b].item(), cu_seqlens[b + 1].item()
        seq_len = e - s
        for h in range(num_q_heads):
            kh = h % num_kv_heads if gqa_interleave else h // share
            scores = q[s:e, h].float() @ k[s:e, kh].float().t() * sm_scale
            if causal:
                mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool), 1)
                scores = scores.masked_fill(mask, float("-inf"))
            out[s:e, h] = torch.softmax(scores, dim=-1) @ v[s:e, kh].float()
    return out


def run_case(num_q_heads, num_kv_heads, qk_head_dim, v_head_dim, gqa_interleave, causal=True):
    torch.manual_seed(42)
    device = "cuda"
    cu_seqlens = torch.tensor([0, 256, 512], device=device, dtype=torch.int32)
    total = 512
    q = torch.randn(total, num_q_heads, qk_head_dim, device=device, dtype=torch.float16, requires_grad=True)
    k = torch.randn(total, num_kv_heads, qk_head_dim, device=device, dtype=torch.float16, requires_grad=True)
    v = torch.randn(total, num_kv_heads, v_head_dim, device=device, dtype=torch.float16, requires_grad=True)
    do = torch.randn(total, num_q_heads, v_head_dim, device=device, dtype=torch.float16)

    out = flash_attention_varlen(q, k, v, cu_seqlens, cu_seqlens, 256, 256, causal=causal, gqa_interleave=gqa_interleave)
    out.backward(do)

    qr = q.detach().clone().requires_grad_(True)
    kr = k.detach().clone().requires_grad_(True)
    vr = v.detach().clone().requires_grad_(True)
    ref = torch_ref_varlen(qr, kr, vr, cu_seqlens, causal, gqa_interleave)
    ref.backward(do.float())

    for name, got, want in [("out", out, ref), ("dq", q.grad, qr.grad), ("dk", k.grad, kr.grad), ("dv", v.grad, vr.grad)]:
        max_err = (got.float() - want.float()).abs().max().item()
        assert max_err < 5e-2, (
            f"{name} max err {max_err:.4f} (Hq={num_q_heads}, Hk={num_kv_heads}, "
            f"qk_dim={qk_head_dim}, v_dim={v_head_dim}, interleave={gqa_interleave})"
        )


def test_backward_gqa_interleave_non_square_ratios():
    # backward_dkdv's interleave branch must split the flat head index by
    # num_kv_heads (as forward_kernel and backward_dq do); splitting by the
    # share count is only correct when the two happen to be equal.
    run_case(8, 2, 128, 128, gqa_interleave=True)   # share > kv heads
    run_case(2, 1, 128, 128, gqa_interleave=True)   # MQA
    run_case(4, 2, 128, 128, gqa_interleave=True)   # square control


def test_backward_split_head_dims():
    # backward_dq's v and dO block pointers must declare v_head_dim, not
    # qk_head_dim: smaller declares drop real value channels from dp, larger
    # ones read past the row end whenever v_head_dim is not a power of two.
    run_case(4, 4, 64, 128, gqa_interleave=False)
    run_case(4, 4, 128, 96, gqa_interleave=False)
    run_case(4, 4, 128, 128, gqa_interleave=False)  # equal-dim control


if __name__ == "__main__":
    test_backward_gqa_interleave_non_square_ratios()
    test_backward_split_head_dims()
    print("all backward regression tests passed")
