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
from typing import Tuple, Callable, Optional
from flash_attn import flash_attn_varlen_func
from native_sparse_attention.ops import (
    flash_attention_decode,
    compressed_attention,
    compressed_attention_decode,
    topk_sparse_attention,
    topk_sparse_attention_decode,
)
from native_sparse_attention.ops.triton.utils import get_compressed_seqlens


def compress_infer(
    cu_seqlens: torch.Tensor,
    step: int,
    key: torch.Tensor,
    value: torch.Tensor,
    cache,
    weight: Tuple[torch.Tensor, torch.Tensor],
    compress_func: Tuple[Callable, Callable],
    intra_block_pe: Optional[torch.Tensor],
    kernel_size: int,
    kernel_stride: int,
):
    if step == 0:
        key, compress_cu_seqlens = compress_func[0](
            key,
            weight[0],
            cu_seqlens,
            kernel_size,
            kernel_stride,
            intra_block_pe,
        )
        value, _ = compress_func[1](
            value,
            weight[1],
            cu_seqlens,
            kernel_size,
            kernel_stride,
        )
    else:
        batch_size = cu_seqlens.shape[0] - 1
        aux_cu_seqlens = (
            torch.arange(batch_size + 1, dtype=torch.int32).to(cu_seqlens.device)
            * kernel_size
        )
        key, _ = compress_func[0](
            cache.before_compress_kv_cache[0, :batch_size].view(
                batch_size * kernel_size, cache.num_kv_heads, cache.head_dim
            ),
            weight[0],
            aux_cu_seqlens,
            kernel_size,
            kernel_stride,
            intra_block_pe,
        )
        value, _ = compress_func[1](
            cache.before_compress_kv_cache[1, :batch_size].view(
                batch_size * kernel_size, cache.num_kv_heads, cache.head_dim
            ),
            weight[1],
            aux_cu_seqlens,
            kernel_size,
            kernel_stride,
        )
        # return actual compress_cu_seqlens before this token
        compress_cu_seqlens = torch.zeros(
            batch_size + 1, dtype=torch.int32, device=key.device
        )
        compress_cu_seqlens[1:] = torch.cumsum(
            cache.compress_kv_len[:batch_size], dim=0
        )
    return key, value, compress_cu_seqlens


def compressed_attention_infer(
    cu_seqlens,
    step,
    query,
    key,
    value,
    cache,
    kernel_size,
    kernel_stride,
    topk,
    block_size,
    init_blocks,
    local_blocks,
):
    if step == 0:
        seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
        compress_seqlens, compress_cu_seqlens = get_compressed_seqlens(
            cu_seqlens, kernel_size, kernel_stride
        )
        attn_output, topk_idx = compressed_attention(
            query,
            key,
            value,
            kernel_size,
            kernel_stride,
            block_size,
            topk,
            cu_seqlens,
            compress_cu_seqlens,
            seqlens.max().item(),
            compress_seqlens.max().item(),
            None,
            init_blocks,
            local_blocks,
        )
    else:
        batch_size = cu_seqlens.shape[0] - 1
        seqlens = cu_seqlens[1:] - cu_seqlens[:-1] + step
        attn_output, topk_idx = compressed_attention_decode(
            query,
            cache.compress_kv_cache[
                0, :batch_size, : cache.compress_kv_len[:batch_size].max()
            ],
            cache.compress_kv_cache[
                1, :batch_size, : cache.compress_kv_len[:batch_size].max()
            ],
            seqlens,
            cache.compress_kv_len[:batch_size],
            kernel_size,
            kernel_stride,
            block_size,
            topk,
            init_blocks,
            local_blocks,
        )
    return attn_output, topk_idx


def topk_sparse_attention_infer(
    cu_seqlens,
    step,
    query,
    key,
    value,
    cache,
    topk_idx,
    block_size,
):
    if step == 0:
        attn_output = topk_sparse_attention(
            query, key, value, topk_idx, block_size, cu_seqlens
        )
    else:
        batch_size = cu_seqlens.shape[0] - 1
        attn_output = topk_sparse_attention_decode(
            query,
            cache.sparse_kv_cache[0, :batch_size],
            cache.sparse_kv_cache[1, :batch_size],
            topk_idx,
            block_size,
            cache.sparse_kv_len[:batch_size],
        )
    return attn_output


def sliding_window_attention_infer(
    cu_seqlens, step, query, key, value, cache, window_size
):
    if step == 0:
        seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
        attn_output = flash_attn_varlen_func(
            query,
            key,
            value,
            cu_seqlens,
            cu_seqlens,
            seqlens.max().item(),
            seqlens.max().item(),
            causal=True,
            window_size=(window_size, -1),
        )
    else:
        batch_size = cu_seqlens.shape[0] - 1
        attn_output = flash_attention_decode(
            query,
            cache.sliding_kv_cache[0, :batch_size],
            cache.sliding_kv_cache[1, :batch_size],
            torch.minimum(
                cache.sliding_kv_len,
                torch.zeros_like(cache.sliding_kv_len) + window_size,
            )[:batch_size],
        )
    return attn_output
