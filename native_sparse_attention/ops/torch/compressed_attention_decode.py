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
from typing import Tuple, Optional
from collections import Counter
from einops import rearrange


def transform_score(
    score: torch.Tensor,
    seqlens: torch.Tensor,
    kernel_size: int,
    kernel_stride: int,
    block_size: int,
    init_blocks: int = 1,
    local_blocks: int = 2,
) -> torch.Tensor:
    num_k_heads, batch_size, kv_len = score.shape
    pad_len = kernel_size // kernel_stride - 1
    score = torch.nn.functional.pad(score, (pad_len, pad_len), value=0)
    max_seqlen = seqlens.max().item()
    max_blocks = math.ceil(max_seqlen / block_size)
    full_blocks = max_seqlen // block_size
    block_score = torch.zeros(
        num_k_heads,
        batch_size,
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
    q_idx = (seqlens - 1) // block_size
    b_idx = torch.arange(max_blocks, device=score.device)
    block_score[..., :init_blocks] = torch.inf
    local_mask = (q_idx[:, None] >= b_idx[None, :]) & (
        q_idx[:, None] < b_idx[None, :] + local_blocks
    )
    local_mask = local_mask.unsqueeze(0).expand(num_k_heads, -1, -1)
    block_score[local_mask] = torch.inf
    block_score = block_score.nan_to_num(0, torch.inf, -torch.inf)
    return block_score


def compressed_attention_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seqlens: torch.Tensor,
    compress_seqlens: torch.Tensor,
    kernel_size: int,
    kernel_stride: int,
    block_size: int,
    topk: int,
    init_blocks: int = 1,
    local_blocks: int = 2,
    sm_scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """_summary_

    Args:
        q (torch.Tensor): shape [batch_size, num_q_heads, head_dim]
        k (torch.Tensor): shape [batch_size, kv_len, num_kv_heads, head_dim]
        v (torch.Tensor): shape [batch_size, kv_len, num_kv_heads, head_dim]
        seqlens (torch.Tensor): original kv length for each sequence
        compress_seqlens (torch.Tensor): kv length for each sequence after compression
        kernel_size (int): kernel size in compress_key_value
        kernel_stride (int): stride of compress_key_value
        block_size (int): key value block size for topk sparse attention.
        topk (int): number of blocks for each query.
        init_blocks (int, optional): Number of init blocks for each query. Defaults to 1.
        local_blocks (int, optional): Number of local blocks for each query. Defaults to 2.
        sm_scale (float, optional): softmax scale. Defaults to None, means 1/sqrt(head_dim).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: attention output and topk_idx used in topk_sparse_attention_decode
    """
    assert block_size % kernel_size == 0 and kernel_size % kernel_stride == 0
    batch_size, num_q_heads, head_dim = q.shape
    batch_size, kv_len, num_k_heads, _ = k.shape
    num_share_q_heads = num_q_heads // num_k_heads
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)
    # input is too short to have a valid block
    if kv_len == 0:
        return torch.zeros_like(q), torch.zeros(
            num_k_heads, batch_size, 1, device=q.device, dtype=torch.int32
        )
    # get mask
    mask = (
        compress_seqlens[:, None]
        > torch.arange(
            kv_len, device=compress_seqlens.device, dtype=compress_seqlens.dtype
        )[None, :]
    )
    # attention
    qk = (
        torch.einsum(
            "bihgd, bjhgd -> bhgij",
            rearrange(q, "b (h g) d -> b 1 h g d", g=num_share_q_heads),
            rearrange(k, "b j h d -> b j h 1 d"),
        )
        * sm_scale
    )
    qk = qk.masked_fill_(~mask[:, None, None, None, :], -torch.inf)
    qk = qk.softmax(dim=-1, dtype=torch.float32)
    qk = qk.nan_to_num_(0)  # qk is nan when seqlen == 0
    attn_output = torch.einsum(
        "bhgij, bjhgd -> bihgd",
        qk.to(v.dtype),
        rearrange(v, "b k h d -> b k h 1 d"),
    )
    attn_output = rearrange(attn_output, "b 1 h g d -> b (h g) d")

    # get score
    score = rearrange(qk.sum(2).squeeze(2), "b h j -> h b j")
    # transform score to block-wise score
    score = transform_score(
        score,
        seqlens,
        kernel_size,
        kernel_stride,
        block_size,
        init_blocks,
        local_blocks,
    )
    # get topk
    q_idx = (seqlens - 1) // block_size
    topk = min(topk, score.shape[-1])
    topk_idx = score.topk(topk, dim=-1).indices.sort(-1).values
    topk_idx[topk_idx > q_idx[None, :, None]] = -1
    topk_idx = topk_idx.to(torch.int32)
    return attn_output, topk_idx
