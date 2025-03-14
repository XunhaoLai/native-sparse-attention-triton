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
def decode_kernel(
    q_ptr,  # Q: b x h x d
    k_ptr,  # K: b x n x h x d
    v_ptr,  # V: b x n x h x d
    o_ptr,  # O: b x h x d
    seqlens,
    # shape
    BATCH_SIZE,
    NUM_SHARE_Q_HEADS,
    HEAD_DIM,
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
    stride_ob,
    stride_oh,
    stride_od,
    # META parameters
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    qk_scale = sm_scale * 1.44269504
    # get batch id and head id
    pid_h = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_kh = pid_h // NUM_SHARE_Q_HEADS
    # get q k start and len after rmpad
    off_b = tl.arange(0, BLOCK_SIZE_B)
    kv_len = tl.load(
        seqlens + pid_b * BLOCK_SIZE_B + off_b,
        mask=pid_b * BLOCK_SIZE_B + off_b < BATCH_SIZE,
        other=0,
    )
    max_kv_len = tl.max(kv_len)
    # init qkv pointer
    q_ptrs = tl.make_block_ptr(
        base=q_ptr + pid_h * stride_qh,
        shape=(BATCH_SIZE, HEAD_DIM),
        strides=(stride_qb, stride_qd),
        offsets=(pid_b * BLOCK_SIZE_B, 0),
        block_shape=(BLOCK_SIZE_B, BLOCK_SIZE_D),
        order=(1, 0),
    )
    k_ptrs = tl.make_block_ptr(
        base=k_ptr + pid_kh * stride_kh,
        shape=(BATCH_SIZE, max_kv_len, HEAD_DIM),
        strides=(stride_kb, stride_kn, stride_kd),
        offsets=(pid_b * BLOCK_SIZE_B, 0, 0),
        block_shape=(BLOCK_SIZE_B, BLOCK_SIZE_K, BLOCK_SIZE_D),
        order=(2, 1, 0),
    )
    v_ptrs = tl.make_block_ptr(
        base=v_ptr + pid_kh * stride_vh,
        shape=(BATCH_SIZE, max_kv_len, HEAD_DIM),
        strides=(stride_vb, stride_vn, stride_vd),
        offsets=(pid_b * BLOCK_SIZE_B, 0, 0),
        block_shape=(BLOCK_SIZE_B, BLOCK_SIZE_K, BLOCK_SIZE_D),
        order=(2, 1, 0),
    )
    # load q
    q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")
    # init statistics
    off_k = tl.arange(0, BLOCK_SIZE_K)
    m_i = tl.full((BLOCK_SIZE_B,), float("-inf"), dtype=tl.float32)
    lse_i = tl.full((BLOCK_SIZE_B,), float("-inf"), dtype=tl.float32)
    acc_o = tl.full((BLOCK_SIZE_B, BLOCK_SIZE_D), 0, dtype=tl.float32)
    # full attention or causal attention
    for i in range(0, max_kv_len, BLOCK_SIZE_K):
        i = tl.multiple_of(i, BLOCK_SIZE_K)
        # load k
        k = tl.load(k_ptrs, boundary_check=(0, 1, 2), padding_option="zero")
        # compute qk
        qk = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_K), dtype=tl.float32)
        qk += tl.where(off_k[None, :] + i < kv_len[:, None], 0, float("-inf"))
        # [B, D], [B, K, D] -> [B, K]
        qk += tl.sum(q[:, None, :] * k, axis=2) * qk_scale
        # compute m_ij and l_ij
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.math.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        # scale acc_o
        acc_o_scale = tl.math.exp2(m_i - m_ij)
        acc_o = acc_o * acc_o_scale[:, None]
        # load v and update acc_o
        v = tl.load(v_ptrs, boundary_check=(0, 1, 2), padding_option="zero")
        p = p.to(v.dtype)
        # [B, K], [B, K, D] -> [B, D]
        acc_o += tl.sum(p[:, :, None] * v, axis=1)
        # update statistics
        m_i = m_ij
        lse_i = m_ij + tl.math.log2(tl.math.exp2(lse_i - m_ij) + l_ij)
        # update ptrs
        k_ptrs = tl.advance(k_ptrs, (0, BLOCK_SIZE_K, 0))
        v_ptrs = tl.advance(v_ptrs, (0, BLOCK_SIZE_K, 0))
    # final scale
    acc_o = acc_o * tl.math.exp2(m_i - lse_i)[:, None]
    # save output
    o_ptrs = tl.make_block_ptr(
        base=o_ptr + pid_h * stride_oh,
        shape=(BATCH_SIZE, HEAD_DIM),
        strides=(stride_ob, stride_od),
        offsets=(pid_b * BLOCK_SIZE_B, 0),
        block_shape=(BLOCK_SIZE_B, BLOCK_SIZE_D),
        order=(1, 0),
    )
    tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1))


def flash_attention_decode(
    q: torch.Tensor,  # [batch_size, num_heads, head_dim]
    k: torch.Tensor,  # [batch_size, max_len, num_heads, head_dim]
    v: torch.Tensor,
    seqlens: torch.Tensor,  # [batch_size, ]
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """flash attention for decode.

    Args:
        q (torch.Tensor): query, shape [batch_size, num_q_heads, head_dim]
        k (torch.Tensor): key, shape [batch_size, kv_len, num_kv_heads, head_dim]
        v (torch.Tensor): value, shape [batch_size, kv_len, num_kv_heads, head_dim]
        seqlens (torch.Tensor): kv length for each sequence
        sm_scale (Optional[float]): softmax scale, default to 1/sqrt(head_dim)

    Returns:
        torch.Tensor: attention output
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
    num_warps = 4 if head_dim <= 64 else 8
    num_stages = 3
    # there is a bug for triton 3.0.0 if BLOCK_SIZE_B > 16
    BLOCK_SIZE_B = min(16, triton.next_power_of_2(batch_size))
    BLOCK_SIZE_K = 128
    BLOCK_SIZE_D = triton.next_power_of_2(head_dim)
    grid = (num_q_heads, triton.cdiv(batch_size, BLOCK_SIZE_B))
    decode_kernel[grid](
        q,
        k,
        v,
        o,
        seqlens,
        batch_size,
        num_share_q_heads,
        head_dim,
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
        o.stride(0),
        o.stride(1),
        o.stride(2),
        BLOCK_SIZE_B=BLOCK_SIZE_B,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return o


def torch_attention_decode(
    q: torch.Tensor,  # [batch_size, num_heads, head_dim]
    k: torch.Tensor,  # [batch_size, max_len, num_heads, head_dim]
    v: torch.Tensor,
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
    # gqa
    assert num_k_heads == num_v_heads
    assert num_q_heads % num_k_heads == 0
    num_share_q_heads = num_q_heads // num_k_heads
    # sm scale
    if sm_scale is None:
        sm_scale = 1 / math.sqrt(head_dim)
    # attention
    attn = (
        torch.einsum(
            "bqhd,bkhd->bhqk",
            q.unsqueeze(1),
            k.repeat_interleave(num_share_q_heads, dim=2),
        )
        * sm_scale
    )
    mask = torch.arange(k_len, device=q.device)[None, :] < seqlens[:, None]
    attn = attn.masked_fill(~mask[:, None, None, :], -torch.inf)
    attn = torch.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
    out = torch.einsum(
        "bhqk,bkhd->bqhd", attn, v.repeat_interleave(num_share_q_heads, dim=2)
    ).squeeze(1)
    return out


if __name__ == "__main__":
    torch.manual_seed(42)
    batch_size = 76
    max_length = 8192
    seqlens = torch.arange(batch_size, dtype=torch.int32).cuda() * 128 + 1
    seqlens[seqlens > max_length] = max_length
    seqlens = seqlens[torch.randn_like(seqlens, dtype=torch.float32).argsort(-1)]
    q = (
        torch.empty(batch_size, 32, 128, device="cuda")
        .uniform_(-1, 1)
        .to(torch.bfloat16)
    )
    k = (
        torch.empty(batch_size, max_length, 4, 128, device="cuda")
        .uniform_(-1, 1)
        .to(torch.bfloat16)
    )
    v = (
        torch.empty(batch_size, max_length, 4, 128, device="cuda")
        .uniform_(-1, 1)
        .to(torch.bfloat16)
    )

    o1 = torch_attention_decode(q, k, v, seqlens)
    o2 = flash_attention_decode(q, k, v, seqlens)

    print(torch.allclose(o1, o2, atol=1e-2, rtol=1e-2))
