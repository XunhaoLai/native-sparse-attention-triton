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

import math
import torch
import triton
import triton.language as tl
from typing import Optional


@triton.jit
def forward_kernel(
    q_ptr,  # Q: b x h x d
    k_ptr,  # K: b x n x kh x d
    v_ptr,  # V: b x n x kh x d
    t_ptr,  # topk_idx: kh x b x k
    o_ptr,  # O: b x h x d
    # seqlens
    seqlens,
    # shape
    NUM_SHARE_Q_HEADS,
    HEAD_DIM,
    TOPK,
    # sm_scale
    sm_scale,
    # stride
    stride_qb,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_kn,
    stride_kh,
    stride_kd,
    stride_vb,
    stride_vn,
    stride_vh,
    stride_vd,
    stride_th,
    stride_tb,
    stride_tk,
    stride_ob,
    stride_oh,
    stride_od,
    # META parameters
    BLOCK_SIZE_K: tl.constexpr,  # k block size
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
):
    qk_scale = sm_scale * 1.44269504
    # get batch id and head id
    pid_b = tl.program_id(0)
    pid_kh = tl.program_id(1)
    # get kv_len
    kv_len = tl.load(seqlens + pid_b)
    # init topk idx pointer
    off_t = tl.arange(0, BLOCK_SIZE_T)
    t_ptr = t_ptr + pid_b * stride_tb + pid_kh * stride_th
    topk_idx = tl.load(t_ptr + off_t * stride_tk, mask=off_t < TOPK, other=-1)
    real_topk = tl.sum(
        tl.where((topk_idx >= 0) & (topk_idx <= (kv_len - 1) // BLOCK_SIZE_K), 1, 0),
        axis=0,
    )
    # init qkv pointer
    q_ptrs = tl.make_block_ptr(
        base=q_ptr + pid_b * stride_qb + pid_kh * NUM_SHARE_Q_HEADS * stride_qh,
        shape=(NUM_SHARE_Q_HEADS, HEAD_DIM),
        strides=(stride_qh, stride_qd),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
        order=(1, 0),
    )
    k_ptrs = tl.make_block_ptr(
        base=k_ptr + pid_b * stride_kb + pid_kh * stride_kh,
        shape=(HEAD_DIM, kv_len),
        strides=(stride_kd, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_K),
        order=(0, 1),
    )
    v_ptrs = tl.make_block_ptr(
        base=v_ptr + pid_b * stride_vb + pid_kh * stride_vh,
        shape=(kv_len, HEAD_DIM),
        strides=(stride_vn, stride_vd),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
        order=(1, 0),
    )
    # load q
    q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")
    # init statistics
    off_k = tl.arange(0, BLOCK_SIZE_K)
    m_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
    lse_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
    acc_o = tl.full((BLOCK_SIZE_H, BLOCK_SIZE_D), 0, dtype=tl.float32)
    # sparse attention
    for i in range(real_topk):
        # get current block start index
        c = tl.load(t_ptr).to(tl.int32) * BLOCK_SIZE_K
        t_ptr = t_ptr + stride_tk
        # load k
        k = tl.load(
            tl.advance(k_ptrs, (0, c)), boundary_check=(1, 0), padding_option="zero"
        )
        # compute qk
        qk = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_K), dtype=tl.float32)
        qk += tl.where((kv_len > c + off_k)[None, :], 0, float("-inf"))
        # [BLOCK_SIZE_H, HEAD_DIM] @ [HEAD_DIM, BLOCK_SIZE_K] -> [BLOCK_SIZE_H, BLOCK_SIZE_K]
        qk += tl.dot(q, k) * qk_scale
        # compute m_ij and l_ij
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        # scale acc_o
        acc_o_scale = tl.exp2(m_i - m_ij)
        acc_o = acc_o * acc_o_scale[:, None]
        # load v and update acc_o
        v = tl.load(
            tl.advance(v_ptrs, (c, 0)), boundary_check=(0, 1), padding_option="zero"
        )
        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)
        # update statistics
        m_i = m_ij
        lse_i = m_ij + tl.math.log2(tl.exp2(lse_i - m_ij) + l_ij)
    # final scale
    acc_o = acc_o * tl.exp2(m_i - lse_i)[:, None]
    # save output
    o_ptrs = tl.make_block_ptr(
        base=o_ptr + pid_b * stride_ob + pid_kh * NUM_SHARE_Q_HEADS * stride_oh,
        shape=(NUM_SHARE_Q_HEADS, HEAD_DIM),
        strides=(stride_oh, stride_od),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
        order=(1, 0),
    )
    tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1))


def topk_sparse_attention_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    topk_idx: torch.Tensor,
    block_size: int,
    seqlens: torch.Tensor,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """_summary_

    Args:
        q (torch.Tensor): shape [batch_size, num_q_heads, head_dim]
        k (torch.Tensor): shape [batch_size, kv_len, num_kv_heads, head_dim]
        v (torch.Tensor): shape [batch_size, kv_len, num_kv_heads, head_dim]
        topk_idx (torch.Tensor): topk block idx for each query, shape [num_kv_heads, batch_size, topk]. -1 means padding.
        block_size (int): key value block size.
        seqlens (torch.Tensor): max kv length for each sequence
        softmax_scale (Optional[float], optional): Defaults to None, means 1/sqrt(head_dim).

    Returns:
        torch.Tensor: sparse attention output
    """
    # dtype check
    assert q.dtype == torch.bfloat16 or q.dtype == torch.float16
    assert k.dtype == q.dtype and v.dtype == q.dtype
    assert seqlens.dtype == torch.int32
    # shape
    batch_size, num_q_heads, head_dim = q.shape
    _, k_len, num_k_heads, head_dim = k.shape
    _, v_len, num_v_heads, head_dim = v.shape
    assert k_len == v_len and batch_size == seqlens.shape[0]
    assert num_k_heads == topk_idx.shape[0] and batch_size == topk_idx.shape[1]
    topk = topk_idx.shape[-1]
    # gqa
    assert num_k_heads == num_v_heads
    assert num_q_heads % num_k_heads == 0
    num_share_q_heads = num_q_heads // num_k_heads
    # sm scale
    if sm_scale is None:
        sm_scale = 1 / math.sqrt(head_dim)
    # output tensor
    o = torch.zeros_like(q)
    # launch kernel
    grid = (batch_size, num_k_heads)
    num_warps = 4 if head_dim <= 64 else 8
    num_stages = 3
    BLOCK_SIZE_K = triton.next_power_of_2(block_size)
    BLOCK_SIZE_D = triton.next_power_of_2(head_dim)
    BLOCK_SIZE_H = max(16, triton.next_power_of_2(num_share_q_heads))
    BLOCK_SIZE_T = triton.next_power_of_2(topk)
    forward_kernel[grid](
        q,
        k,
        v,
        topk_idx,
        o,
        seqlens,
        num_share_q_heads,
        head_dim,
        topk,
        sm_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_T=BLOCK_SIZE_T,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return o


def torch_topk_sparse_attention_decode(
    q: torch.Tensor,  # [batch_size, num_q_heads, head_dim]
    k: torch.Tensor,  # [batch_size, kv_len, num_k_heads, head_dim]
    v: torch.Tensor,  # [batch_size, kv_len, num_k_heads, head_dim]
    topk_idx: torch.Tensor,  # [num_k_heads, batch_size, topk]
    block_size: int,
    seqlens: torch.Tensor,  # [batch_size, ]
    sm_scale: Optional[float] = None,
):
    # dtype check
    assert q.dtype == torch.bfloat16 or q.dtype == torch.float16
    assert k.dtype == q.dtype and v.dtype == q.dtype
    assert seqlens.dtype == torch.int32
    # shape
    batch_size, num_q_heads, head_dim = q.shape
    _, k_len, num_k_heads, head_dim = k.shape
    _, v_len, num_v_heads, head_dim = v.shape
    assert k_len == v_len and batch_size == seqlens.shape[0]
    assert num_k_heads == topk_idx.shape[0] and batch_size == topk_idx.shape[1]
    topk = topk_idx.shape[-1]
    # gqa
    assert num_k_heads == num_v_heads
    assert num_q_heads % num_k_heads == 0
    num_share_q_heads = num_q_heads // num_k_heads
    # sm scale
    if sm_scale is None:
        sm_scale = 1 / math.sqrt(head_dim)
    # mask
    mask = torch.zeros(
        (batch_size, num_k_heads, k_len), dtype=torch.bool, device=q.device
    )
    for b in range(batch_size):
        for h in range(num_k_heads):
            for t in range(topk):
                if topk_idx[h, b, t] != -1:
                    mask[
                        b,
                        h,
                        topk_idx[h, b, t]
                        * block_size : (topk_idx[h, b, t] + 1)
                        * block_size,
                    ] = True
    mask = mask & (
        (seqlens - 1)[:, None, None] >= torch.arange(k_len).cuda()[None, None, :]
    )
    mask = mask.repeat_interleave(num_share_q_heads, 1)
    # attention
    attn = (
        torch.einsum(
            "bqhd,bkhd->bhqk", q.unsqueeze(1), k.repeat_interleave(num_share_q_heads, 2)
        )
        * sm_scale
    )
    attn = attn.masked_fill(~mask.unsqueeze(2), -torch.inf)
    attn = torch.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
    out = torch.einsum(
        "bhqk,bkhd->bqhd", attn, v.repeat_interleave(num_share_q_heads, 2)
    ).squeeze(1)
    return out


def generate_topk_idx_example(
    seqlens: torch.Tensor, block_size: int, topk: int, num_heads: int
) -> torch.Tensor:
    batch_size = seqlens.shape[0]
    num_blocks = torch.ceil(seqlens / block_size).to(torch.int32)
    topk_idx_all_heads = []
    for _ in range(num_heads):
        topk_idx = [
            torch.randn(1, num_blocks[i], device="cuda")
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
        q_idx = seqlens - 1
        topk_idx[topk_idx > (q_idx // block_size)[:, None]] = -1  # -1 means padding
        topk_idx_all_heads.append(topk_idx)
    topk_idx = torch.stack(topk_idx_all_heads, dim=0)
    return topk_idx


if __name__ == "__main__":
    torch.manual_seed(42)
    topk = 16
    block_size = 64
    batch_size = 76
    max_length = 8192
    seqlens = torch.arange(batch_size, dtype=torch.int32).cuda() * 128 + 1
    seqlens[seqlens > max_length] = max_length
    seqlens = seqlens[torch.randn_like(seqlens, dtype=torch.float32).argsort(-1)]
    q = (
        torch.empty(batch_size, 32, 128, device="cuda")
        .uniform_(-1, 1)
        .to(torch.float16)
    )
    k = (
        torch.empty(batch_size, max_length, 4, 128, device="cuda")
        .uniform_(-1, 1)
        .to(torch.float16)
    )
    v = (
        torch.empty(batch_size, max_length, 4, 128, device="cuda")
        .uniform_(-1, 1)
        .to(torch.float16)
    )
    topk_idx = generate_topk_idx_example(seqlens, block_size, topk, 4)

    o1 = torch_topk_sparse_attention_decode(q, k, v, topk_idx, block_size, seqlens)
    o2 = topk_sparse_attention_decode(q, k, v, topk_idx, block_size, seqlens)

    print(torch.allclose(o1, o2, atol=1e-3, rtol=1e-3))
    print((o1 - o2).abs().max())
