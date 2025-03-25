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
import math
from typing import Optional


def topk_sparse_attention_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    topk_idx: torch.Tensor,
    block_size_k: int,
    cu_seqlens: torch.Tensor,
    softmax_scale: Optional[float] = None,
    block_size_q: int = 1,
) -> torch.Tensor:
    """Simple topk sparse attention varlen version implemented in torch. Extremly slow, only for debugging.

    Args:
        q (torch.Tensor): shape [total_len, num_q_heads, head_dim]
        k (torch.Tensor): shape [total_len, num_kv_heads, head_dim]
        v (torch.Tensor): shape [total_len, num_kv_heads, head_dim]
        topk_idx (torch.Tensor): topk block idx for each query, shape [num_kv_heads, total_len, topk]. -1 means padding.
        block_size_q (int): query block size.
        block_size_k (int): key value block size.
        cu_seqlens (torch.Tensor): shape [batch_size + 1], similar to cu_seqlens in flash_attn_func_varlen.
        softmax_scale (Optional[float], optional): Defaults to None, means 1/sqrt(head_dim).

    Returns:
        torch.Tensor: attention output, shape [total_len, num_q_heads, head_dim]
    """
    total_seqlen, num_q_heads, head_dim = q.shape
    total_seqlen, num_kv_heads, head_dim = k.shape
    num_share_q_heads = num_q_heads // num_kv_heads
    batch_size = cu_seqlens.shape[0] - 1
    topk = topk_idx.shape[-1]
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    seqblocks_q = torch.ceil(seqlens / block_size_q).to(torch.int32)
    cu_seqblocks_q = torch.nn.functional.pad(seqblocks_q.cumsum(0), (1, 0), value=0)
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    # get mask
    mask = torch.zeros(
        (num_kv_heads, total_seqlen, total_seqlen), dtype=torch.bool, device=q.device
    )
    for i in range(batch_size):
        num_q_blocks = math.ceil(seqlens[i] / block_size_q)
        num_kv_blocks = math.ceil(seqlens[i] / block_size_k)
        for h in range(num_kv_heads):
            temp_mask = torch.zeros(
                num_q_blocks, num_kv_blocks, dtype=torch.bool, device=q.device
            )
            temp_idx = topk_idx[h, cu_seqblocks_q[i] : cu_seqblocks_q[i + 1]].clone()
            temp_idx[temp_idx < 0] = 0
            temp_mask[torch.arange(num_q_blocks).to(q.device)[:, None], temp_idx] = True
            temp_mask = torch.repeat_interleave(temp_mask, block_size_q, dim=0)
            temp_mask = torch.repeat_interleave(temp_mask, block_size_k, dim=1)
            temp_mask = temp_mask[: seqlens[i], : seqlens[i]]
            mask[
                h, cu_seqlens[i] : cu_seqlens[i + 1], cu_seqlens[i] : cu_seqlens[i + 1]
            ] = temp_mask
    mask = torch.tril(mask).repeat_interleave(num_share_q_heads, 0)
    # qk attn
    qk = (
        torch.einsum("qhd,khd->hqk", q, k.repeat_interleave(num_share_q_heads, 1))
        * softmax_scale
    )
    qk = torch.masked_fill(qk, ~mask, -torch.inf)
    qk = torch.softmax(qk, dim=-1, dtype=torch.float32).to(q.dtype)
    o = torch.einsum("hqk,khd->qhd", qk, v.repeat_interleave(num_share_q_heads, 1))
    return o
