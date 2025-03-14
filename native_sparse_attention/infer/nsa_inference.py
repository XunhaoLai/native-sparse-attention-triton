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
from typing import Tuple, Callable
from native_sparse_attention.module import RotaryEmbedding, NSACache
from native_sparse_attention.infer.inference_func import (
    compress_infer,
    compressed_attention_infer,
    topk_sparse_attention_infer,
    sliding_window_attention_infer,
)


def nsa_infer(
    cu_seqlens: torch.Tensor,
    step: int,
    # qkv for three parts
    query: torch.Tensor,
    key: torch.Tensor,  # prefill: [total_len, num_heads, head_dim], decode: [batch_size, num_heads, head_dim]
    value: torch.Tensor,
    gate_value: torch.Tensor,  # prefill: [total_len, num_heads, 3], decode: [batch_size, num_heads, 3]
    # rope and kv cache
    rope: RotaryEmbedding,
    cache: NSACache,
    # weight for nsa compress
    compress_weight: Tuple[
        torch.Tensor, torch.Tensor
    ],  # compress weight for key and value
    compress_func: Tuple[Callable, Callable],  # compress function for key and value
    # nsa parameters
    kernel_size: int,
    kernel_stride: int,
    block_size: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    window_size: int,
):
    # prepare for compress
    cache.prepare_compress(cu_seqlens, step, key, value)
    # compressed key and value before rope
    compress_key, compress_value, compress_cu_seqlens = compress_infer(
        cu_seqlens,
        step,
        key,
        value,
        cache,
        compress_weight,
        compress_func,
        kernel_size,
        kernel_stride,
    )
    # do rope
    query = rope(query, cu_seqlens, step)
    if step == 0:
        compress_key = rope(
            compress_key, compress_cu_seqlens, step, stride=cache.kernel_stride
        )
    else:
        compress_key = rope(
            compress_key, compress_cu_seqlens, 1, stride=cache.kernel_stride
        )
    key = rope(key, cu_seqlens, step)
    # update kv cache
    cache.update_kv(
        cu_seqlens,
        step,
        compress_key,
        compress_value,
        key,
        value,
        key,
        value,
    )
    # compressed attention
    compress_attn_output, topk_idx = compressed_attention_infer(
        cu_seqlens,
        step,
        query,
        compress_key,
        compress_value,
        cache,
        kernel_size,
        kernel_stride,
        topk,
        block_size,
        init_blocks,
        local_blocks,
    )
    # topk sparse attention
    sparse_attn_output = topk_sparse_attention_infer(
        cu_seqlens,
        step,
        query,
        key,
        value,
        cache,
        topk_idx,
        block_size,
    )
    # sliding window attention
    sliding_attn_output = sliding_window_attention_infer(
        cu_seqlens, step, query, key, value, cache, window_size
    )
    # combine 3 attn output
    attn_output = (
        gate_value[..., 0, None] * compress_attn_output
        + gate_value[..., 1, None] * sparse_attn_output
        + gate_value[..., 2, None] * sliding_attn_output
    )
    return attn_output
