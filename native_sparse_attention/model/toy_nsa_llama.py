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
from typing import Optional
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from native_sparse_attention.module import NativeSparseAttention, RopeConfig, NSACache


@dataclass
class ToyNSALlamaConfig:
    # embedding config
    vocab_size: int = 128288
    max_position_embeddings: int = 131072
    # model config
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 2
    head_dim: int = 128
    # rope config
    rope_theta: float = 500000.0
    rope_scaling: dict = field(
        default_factory=lambda: {
            "factor": 8.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3",
        }
    )
    # nsa config
    compress_type: str = "weightedpool"
    kernel_size: int = 32
    kernel_stride: int = 16
    block_size: int = 64
    topk: int = 16
    init_blocks: int = 1
    local_blocks: int = 2
    window_size: int = 512


@dataclass
class InferenceConfig:
    max_batch_size: int = 32
    max_length: int = 8192
    max_new_tokens: int = 128


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class FFN(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class ToyNSALlamaLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        compress_type: str,
        kernel_size: int,
        kernel_stride: int,
        block_size: int,
        topk: int,
        init_blocks: int,
        local_blocks: int,
        window_size: int,
        rope_config: RopeConfig,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.compress_type = compress_type
        self.kernel_size = kernel_size
        self.kernel_stride = kernel_stride
        self.block_size = block_size
        self.topk = topk
        self.init_blocks = init_blocks
        self.local_blocks = local_blocks
        self.window_size = window_size
        self.rope_config = rope_config
        self.attn_norm = RMSNorm(self.hidden_size)
        self.nsa = NativeSparseAttention(
            hidden_size=self.hidden_size,
            num_q_heads=self.num_q_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            compress_type=self.compress_type,
            kernel_size=self.kernel_size,
            kernel_stride=self.kernel_stride,
            block_size=self.block_size,
            topk=self.topk,
            init_blocks=self.init_blocks,
            local_blocks=self.local_blocks,
            window_size=self.window_size,
            rope_config=rope_config,
        )
        self.ffn_norm = RMSNorm(self.hidden_size)
        self.ffn = FFN(
            hidden_size=self.hidden_size, intermediate_size=self.intermediate_size
        )

    def forward(self, x, cu_seqlens):
        x = x + self.nsa(self.attn_norm(x), cu_seqlens)
        x = x + self.ffn(self.ffn_norm(x))
        return x

    @torch.no_grad()
    def inference(self, x, cu_seqlens, step, kv_cache):
        x = x + self.nsa.inference(self.attn_norm(x), cu_seqlens, step, kv_cache)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class ToyNSALlama(nn.Module):
    def __init__(
        self,
        config: ToyNSALlamaConfig,
        inference_config: Optional[InferenceConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.rope_config = RopeConfig(
            head_dim=self.config.head_dim,
            rope_theta=self.config.rope_theta,
            rope_scaling=self.config.rope_scaling,
        )
        self.layers = nn.ModuleList(
            [
                ToyNSALlamaLayer(
                    hidden_size=self.config.hidden_size,
                    intermediate_size=self.config.intermediate_size,
                    num_q_heads=self.config.num_attention_heads,
                    num_kv_heads=self.config.num_key_value_heads,
                    head_dim=self.config.head_dim,
                    compress_type=self.config.compress_type,
                    kernel_size=self.config.kernel_size,
                    kernel_stride=self.config.kernel_stride,
                    block_size=self.config.block_size,
                    topk=self.config.topk,
                    init_blocks=self.config.init_blocks,
                    local_blocks=self.config.local_blocks,
                    window_size=self.config.window_size,
                    rope_config=RopeConfig(
                        self.config.max_position_embeddings,
                        self.config.head_dim,
                        self.config.rope_theta,
                        self.config.rope_scaling,
                    ),
                )
                for _ in range(self.config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(self.config.hidden_size)
        self.lm_head = nn.Linear(
            self.config.hidden_size, self.config.vocab_size, bias=False
        )

        # inference config and kv cache
        self.inference_config = inference_config
        self.kv_cache = None

    def forward(
        self,
        input_ids: torch.LongTensor,  # shape: [batch_size, max_length]
        cu_seqlens: torch.LongTensor,  # shape: [batch_size + 1, ]
    ):
        # embedding
        x = self.embedding(input_ids).to(torch.bfloat16)
        # layers
        for layer in self.layers:
            x = layer(x, cu_seqlens)
        # final norm
        x = self.norm(x)
        # lanugauge head
        x = self.lm_head(x).to(torch.float32)  # [total_len, vocab_size]
        return x

    @torch.no_grad()
    def inference(
        self,
        input_ids: torch.LongTensor,  # prefill shape: [total_length, ]; decode shape: [batch_size, ]
        cu_seqlens: torch.LongTensor,  # shape: [batch_size + 1, ]
        step: int,
    ):
        # set kv cache if self.kv_cache is None
        if self.kv_cache is None:
            self.kv_cache = [
                NSACache(
                    max_batch_size=self.inference_config.max_batch_size,
                    max_length=self.inference_config.max_length,
                    num_kv_heads=self.config.num_key_value_heads,
                    head_dim=self.config.head_dim,
                    kernel_size=self.config.kernel_size,
                    kernel_stride=self.config.kernel_stride,
                    window_size=self.config.window_size,
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                for _ in range(self.config.num_hidden_layers)
            ]
        # embedding
        x = self.embedding(input_ids).to(torch.bfloat16)
        # layers
        for i, layer in enumerate(self.layers):
            x = layer.inference(x, cu_seqlens, step, self.kv_cache[i])
        # final norm
        x = self.norm(x)
        # lanugauge head
        if step == 0:
            x = x[cu_seqlens[1:] - 1, :]
        x = self.lm_head(x).to(torch.float32)  # [total_len, vocab_size]
        return x

    def generate(
        self,
        input_ids: torch.LongTensor,
        cu_seqlens: torch.LongTensor,
        max_new_tokens: int = -1,
    ):
        output_tokens = []
        if max_new_tokens <= 0:
            max_new_tokens = self.inference_config.max_new_tokens
        for step in range(max_new_tokens):
            logits = self.inference(
                input_ids, cu_seqlens, step
            )  # shape: [batch_size, vocab_size]
            next_token = torch.argmax(logits, dim=-1)  # shape: [batch_size, ]
            input_ids = next_token
            output_tokens.append(next_token)
        output_tokens = torch.stack(
            output_tokens, dim=1
        )  # shape: [batch_size, max_new_tokens]
        return output_tokens


if __name__ == "__main__":
    torch.manual_seed(42)
    # initialize model
    config = ToyNSALlamaConfig(
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=8,
        num_attention_heads=32,
        num_key_value_heads=2,
        head_dim=128,
        rope_theta=500000.0,
        rope_scaling={
            "factor": 8.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3",
        },
        compress_type="weightedpool",
        kernel_size=32,
        kernel_stride=16,
        block_size=64,
        topk=8,
        init_blocks=1,
        local_blocks=2,
        window_size=512,
    )
    inference_config = InferenceConfig(
        max_batch_size=4,
        max_length=8192,
        max_new_tokens=128,
    )
    model = ToyNSALlama(config, inference_config).cuda().bfloat16()
    print(f"\nMODEL CONFIG:\n{config}\n")
    print(f"\nINFERENCE CONFIG:\n{inference_config}\n")
    print(f"\nMODEL:\n{model}\n")

    # example input
    batch_size = 4
    seqlens = torch.randint(0, 4096, (batch_size,), dtype=torch.int32, device="cuda")
    cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device="cuda")
    cu_seqlens[1:] = torch.cumsum(seqlens, dim=0)
    input_ids = torch.randint(
        0, 128288, (cu_seqlens[-1],), dtype=torch.int64, device="cuda"
    )
    print(f"\nEXAMPLE INPUT:\ncu_seqlens: {cu_seqlens}\ninput_ids: {input_ids.shape}\n")

    # example output
    logits = model(input_ids, cu_seqlens)
    print(f"\nEXAMPLE OUTPUT:\nlogits: {logits.shape}\n")

    # example generate
    output_tokens = model.generate(input_ids, cu_seqlens, 64)
    print(f"\nEXAMPLE GENERATE:\noutput_tokens: {output_tokens}\n")
