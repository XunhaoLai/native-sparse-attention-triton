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
from typing import Tuple
from collections import Counter
from einops import rearrange


def transform_score(
    score: torch.Tensor,
    kernel_size: int,
    kernel_stride: int,
    block_size: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    init_blocks: int = 1,
    local_blocks: int = 2,
) -> torch.Tensor:
    num_k_heads, total_query_len, _ = score.shape
    pad_len = kernel_size // kernel_stride - 1
    score = torch.nn.functional.pad(score, (pad_len, pad_len), value=0)
    max_blocks = math.ceil(max_seqlen_q / block_size)
    full_blocks = max_seqlen_q // block_size
    block_score = torch.zeros(
        num_k_heads,
        total_query_len,
        max_blocks,
        dtype=torch.float32,
        device=score.device,
    )
    offs = (
        torch.arange(kernel_size // kernel_stride)[:, None]
        + torch.arange(block_size // kernel_stride)[None, :]
    ).view(-1)
    offs = dict(Counter(offs.tolist()))
    for k, v in offs.items():
        block_score[..., :full_blocks] += (
            v * score[..., k :: block_size // kernel_stride][..., :full_blocks]
        )
    # set init block and local block score
    batch_size = cu_seqlens_q.shape[0] - 1
    q_idx = torch.cat(
        [
            torch.arange(cu_seqlens_q[i + 1] - cu_seqlens_q[i], device=score.device)
            for i in range(batch_size)
        ],
        dim=0,
    )
    q_idx = q_idx // block_size
    b_idx = torch.arange(max_blocks, device=score.device)
    block_score[..., :init_blocks] = torch.inf
    local_mask = (q_idx[:, None] >= b_idx[None, :]) & (
        q_idx[:, None] < b_idx[None, :] + local_blocks
    )
    local_mask = local_mask.unsqueeze(0).expand(num_k_heads, -1, -1)
    block_score[local_mask] = torch.inf
    return block_score


def compressed_attention_torch(
    q: torch.Tensor,  # [total_query_len, num_q_heads, head_dim]
    k: torch.Tensor,  # [total_key_len, num_k_heads, head_dim]
    v: torch.Tensor,  # [total_key_len, num_k_heads, head_dim]
    kernel_size: int,
    kernel_stride: int,
    block_size: int,
    topk: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    sm_scale: float = None,
    init_blocks: int = 1,
    local_blocks: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Attention between query and compressed key and value. Implemented with torch, only for debug.

    Args:
        q (torch.Tensor): shape [total_q_len, num_q_heads, head_dim]
        k (torch.Tensor): shape [total_kv_len, num_kv_heads, head_dim]
        v (torch.Tensor): shape [total_kv_len, num_kv_heads, head_dim]
        kernel_size (int): kernel size in compress_key_value
        kernel_stride (int): stride of compress_key_value
        block_size (int): key value block size for topk sparse attention.
        topk (int): number of blocks for each query.
        cu_seqlens_q (torch.Tensor): shape [batch_size + 1], similar to cu_seqlens_q in flash_attn_func_varlen.
        cu_seqlens_k (torch.Tensor): shape [batch_size + 1], similar to cu_seqlens_k in flash_attn_func_varlen.
        max_seqlen_q (int): max q len of the batch.
        max_seqlen_k (int): max k len of the batch.
        sm_scale (float, optional): softmax scale. Defaults to None, means 1/sqrt(head_dim).
        init_blocks (int, optional): Number of init blocks for each query. Defaults to 1.
        local_blocks (int, optional): Number of local blocks for each query. Defaults to 2.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: attention output and topk_idx used in topk_sparse_attention
    """
    assert block_size % kernel_size == 0 and kernel_size % kernel_stride == 0
    total_query_len, num_q_heads, head_dim = q.shape
    total_key_len, num_k_heads, _ = k.shape
    num_share_q_heads = num_q_heads // num_k_heads
    batch_size = cu_seqlens_q.shape[0] - 1
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)
    # get mask
    mask = torch.zeros(
        (total_query_len, total_key_len), dtype=torch.bool, device=q.device
    )
    for b in range(batch_size):
        q_len, k_len = (
            cu_seqlens_q[b + 1] - cu_seqlens_q[b],
            cu_seqlens_k[b + 1] - cu_seqlens_k[b],
        )
        k_max_ids = (
            torch.arange(k_len, device=q.device) * kernel_stride + kernel_size - 1
        )
        q_ids = torch.arange(q_len, device=q.device)
        mask[
            cu_seqlens_q[b] : cu_seqlens_q[b + 1], cu_seqlens_k[b] : cu_seqlens_k[b + 1]
        ] = (q_ids[:, None] >= k_max_ids[None, :])
    # attention
    qk = (
        torch.einsum("qhd,khd->hqk", q, k.repeat_interleave(num_share_q_heads, 1))
        * sm_scale
    )
    qk = qk.masked_fill_(~mask[None, ...], -torch.inf)
    # query from beginning of the sequence can't attend to any compressed key
    qk = qk.softmax(dim=-1, dtype=torch.float32)
    qk = qk.nan_to_num(0)
    attn_output = torch.einsum(
        "hqk,khd->qhd", qk.to(v.dtype), v.repeat_interleave(num_share_q_heads, 1)
    )
    with torch.no_grad():
        # get avg score over gqa heads
        # qk shape [num_k_heads, total_q_len, total_k_len]
        score = torch.zeros(
            num_k_heads,
            cu_seqlens_q[-1],
            max_seqlen_k,
            dtype=torch.float32,
            device=q.device,
        )
        qk = rearrange(qk, "(h g) q k -> h g q k", h=num_k_heads).sum(1)
        for b in range(batch_size):
            score[
                :,
                cu_seqlens_q[b] : cu_seqlens_q[b + 1],
                : cu_seqlens_k[b + 1] - cu_seqlens_k[b],
            ] = qk[
                :,
                cu_seqlens_q[b] : cu_seqlens_q[b + 1],
                cu_seqlens_k[b] : cu_seqlens_k[b + 1],
            ]
        # transform score to block-wise score
        score = transform_score(
            score,
            kernel_size,
            kernel_stride,
            block_size,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            init_blocks,
            local_blocks,
        )
        # get topk
        batch_size = cu_seqlens_q.shape[0] - 1
        q_idx = torch.cat(
            [
                torch.arange(cu_seqlens_q[i + 1] - cu_seqlens_q[i], device=q.device)
                for i in range(batch_size)
            ],
            dim=0,
        )
        q_idx = q_idx // block_size
        topk = min(topk, score.shape[-1])
        topk_idx = score.topk(topk, dim=-1).indices.sort(-1).values
        topk_idx[topk_idx > q_idx[None, :, None]] = -1
        topk_idx = topk_idx.to(torch.int32)
    return attn_output, topk_idx
