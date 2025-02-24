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
    block_size: int,
    cu_seqlens: torch.Tensor,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """Simple topk sparse attention varlen version implemented in torch. Extremly slow, only for debugging.

    Args:
        q (torch.Tensor): shape [total_len, num_q_heads, head_dim]
        k (torch.Tensor): shape [total_len, num_kv_heads, head_dim]
        v (torch.Tensor): shape [total_len, num_kv_heads, head_dim]
        topk_idx (torch.Tensor): topk block idx for each query, shape [num_kv_heads, total_len, topk]. -1 means padding.
        block_size (int): key value block size.
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
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    # causal mask for topk idx
    q_idx = torch.cat(
        [torch.arange(seqlens[i], device="cuda") for i in range(batch_size)], dim=0
    )
    topk_idx[topk_idx > (q_idx // block_size)[None, :, None]] = -1
    # get mask
    mask = torch.zeros(
        (num_kv_heads, total_seqlen, total_seqlen), dtype=torch.bool, device=q.device
    )
    for i in range(batch_size):
        start = cu_seqlens[i]
        for h in range(num_kv_heads):
            for j in range(seqlens[i]):
                for t in range(topk):
                    if topk_idx[h, start + j, t] != -1:
                        mask[
                            h,
                            start + j,
                            start
                            + topk_idx[h, start + j, t] * block_size : start
                            + (topk_idx[h, start + j, t] + 1) * block_size,
                        ] = True
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
