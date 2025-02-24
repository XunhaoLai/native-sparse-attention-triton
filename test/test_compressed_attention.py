# Copyright 2025 Xunhao Lai.
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
from ops.torch.compressed_attention import compressed_attention_torch
from ops.triton.compressed_attention import compressed_attention
from ops.torch.compress_key_value import conv_compress


if __name__ == "__main__":
    torch.manual_seed(42)
    num_heads = 32
    head_dim = 128
    kernel_size = 32
    kernel_stride = 16
    block_size = 64
    topk = 4
    seqlens = torch.LongTensor([128, 1000, 4000]).int().cuda()
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
    w = (
        torch.empty(num_heads // 4 * head_dim, head_dim, kernel_size, device="cuda")
        .uniform_(-1, 1)
        .to(torch.float16)
    )
    q.requires_grad = True
    k.requires_grad = True
    v.requires_grad = True
    w.requires_grad = True

    ck, ck_cu_seqlens = conv_compress(k, w, cu_seqlens, kernel_size, kernel_stride)

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
