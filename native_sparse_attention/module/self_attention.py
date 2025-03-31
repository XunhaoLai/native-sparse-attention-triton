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
from flash_attn import flash_attn_varlen_func
from einops import rearrange
from native_sparse_attention.module.rope import RopeConfig, RotaryEmbedding
from native_sparse_attention.module.kv_cache import KVCache
from native_sparse_attention.ops import flash_attention_decode


class SelfAttention(torch.nn.Module):
    """self attention module

    Args:
        hidden_size (int): hidden dimension
        num_q_heads (int): number of query heads
        num_kv_heads (int): number of key/value heads, must be divisible by num_q_heads
        head_dim (int): head dim
        rope_config (RopeConfig): config for rotary embedding, see native_sparse_attention.module.rope.RopeConfig for details
    """

    def __init__(
        self,
        hidden_size: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rope_config: RopeConfig,
        rope_device: str = "cuda",
    ):
        super().__init__()
        # configs
        self.hidden_size = hidden_size
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rope_config = rope_config
        assert self.head_dim == self.rope_config.head_dim

        # qkv proj and o proj
        self.proj_q = torch.nn.Linear(
            self.hidden_size, self.num_q_heads * self.head_dim, bias=False
        )
        self.proj_k = torch.nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.proj_v = torch.nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.proj_o = torch.nn.Linear(
            self.num_q_heads * self.head_dim, self.hidden_size, bias=False
        )
        # rope
        self.rope = RotaryEmbedding(self.rope_config, device=rope_device)

        # init parameters
        self.init_params()

    def init_params(self):
        for p in self.parameters():
            torch.nn.init.xavier_uniform_(p)

    def forward(
        self,
        x: torch.Tensor,  # shape: [total_len, hidden_size]
        cu_seqlens: torch.Tensor,  # shape: [batch_size + 1]
    ):
        # dtype and shape check
        assert x.dtype == torch.bfloat16 or x.dtype == torch.float16
        assert x.shape[-1] == self.hidden_size
        cu_seqlens = cu_seqlens.to(torch.int32)
        seqlens = cu_seqlens[1:] - cu_seqlens[:-1]

        # qkv proj
        q = self.proj_q(x).view(-1, self.num_q_heads, self.head_dim)
        k = self.proj_k(x).view(-1, self.num_kv_heads, self.head_dim)
        v = self.proj_v(x).view(-1, self.num_kv_heads, self.head_dim)

        # do rope for query and compressed key
        q = self.rope(q, cu_seqlens)
        k = self.rope(k, cu_seqlens)

        # self attention
        attn_output = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            seqlens.max().item(),
            seqlens.max().item(),
            causal=True,
        )

        # rearrange and output proj
        attn_output = rearrange(attn_output, "n h d -> n (h d)")
        attn_output = self.proj_o(attn_output)

        return attn_output

    @torch.no_grad()
    def inference(
        self,
        x: torch.Tensor,  # shape: [total_len, hidden_size]
        cu_seqlens: torch.Tensor,  # shape: [batch_size + 1]
        step: int,
        cache: KVCache,
    ):
        # dtype and shape check
        assert x.dtype == torch.bfloat16 or x.dtype == torch.float16
        assert x.shape[-1] == self.hidden_size
        cu_seqlens = cu_seqlens.to(torch.int32)
        seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
        assert step >= 0
        if step == 0:
            assert x.shape[0] == cu_seqlens[-1]
        else:
            assert x.shape[0] == cu_seqlens.shape[0] - 1
        batch_size = cu_seqlens.shape[0] - 1
        # qkv proj
        q = self.proj_q(x).view(-1, self.num_q_heads, self.head_dim)
        k = self.proj_k(x).view(-1, self.num_kv_heads, self.head_dim)
        v = self.proj_v(x).view(-1, self.num_kv_heads, self.head_dim)
        # do rope for query and compressed key
        q = self.rope(q, cu_seqlens, step)
        k = self.rope(k, cu_seqlens, step)
        # reset and update kv cache
        if step == 0:
            cache.reset()
        cache.update_kv(cu_seqlens, step, k, v)
        # self attention
        if step == 0:
            cu_seqlens_q = cu_seqlens_k = cu_seqlens
            max_seqlen_in_batch_q = max_seqlen_in_batch_k = seqlens.max().item()
            output = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                causal=True,
            )
        else:
            output = flash_attention_decode(
                q,
                cache.kv_cache[0, :batch_size],
                cache.kv_cache[1, :batch_size],
                cache.kv_len[:batch_size],
            )
        # rearrange and output proj
        output = rearrange(output, "n h d -> n (h d)")
        output = self.proj_o(output)
        return output
