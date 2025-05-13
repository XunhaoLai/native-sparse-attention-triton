# Copyright 2025 Xunhao Lai & Jianqiao Lu.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
from dataclasses import dataclass, field
from torch import nn
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS


# default to llama3.1 rope config
@dataclass
class RopeConfig:
    """Config for RotaryEmbedding, similar to transformers llama."""

    max_position_embeddings: int = 131072
    head_dim: int = 128
    rope_theta: float = 500000
    rope_scaling: dict = field(
        default_factory=lambda: {
            "factor": 8.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3",
        }
    )
    # useless, just for compatibility, please use head_dim instead
    hidden_size: int = 1
    num_attention_heads: int = 1

    def __post_init__(self):
        self.num_attention_heads = 1
        self.hidden_size = self.head_dim


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# copy and modify from modify from hugigngface transformers
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
class RotaryEmbedding(nn.Module):
    """Rotary embedding

    Args:
        config (RopeConfig): config for rotary embedding, see native_sparse_attention.module.rope.RopeConfig for details
        device (str): default to 'cuda'
    """

    cos = None
    sin = None

    def __init__(
        self, config: RopeConfig, device=torch.device(torch.cuda.current_device())
    ):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get(
                "rope_type", config.rope_scaling.get("type")
            )
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len
            )
            self.register_buffer(
                "inv_freq", inv_freq, persistent=False
            )  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if (
            seq_len < self.original_max_seq_len
            and self.max_seq_len_cached > self.original_max_seq_len
        ):  # reset
            # This .to() is needed if the model has been moved to a device after being initialized (because
            # the buffer is automatically moved, but not the original copy)
            self.original_inv_freq = self.original_inv_freq.to(device)
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def generate_cos_sin(self, x: torch.Tensor, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = (
            device_type
            if isinstance(device_type, str) and device_type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            # # donot use this if use flash_attn
            # emb = torch.cat((freqs, freqs), dim=-1)
            emb = freqs
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = (cos * self.attention_scaling).to(dtype=x.dtype).squeeze(0)
        sin = (sin * self.attention_scaling).to(dtype=x.dtype).squeeze(0)

        # save cos sin
        RotaryEmbedding.cos = torch.cat([cos, cos], dim=-1)
        RotaryEmbedding.sin = torch.cat([sin, sin], dim=-1)

        return RotaryEmbedding.cos, RotaryEmbedding.sin

    @torch.no_grad()
    def generate_pos_embs(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        seqlens: torch.Tensor,
        step: int = 0,
        stride: int = 1,
    ):
        if (
            RotaryEmbedding.cos is None
            or seqlens.max() + step > RotaryEmbedding.cos.shape[0]
        ):
            self.generate_cos_sin(
                x, torch.arange(seqlens.max() + step).to(x.device)[None, :]
            )

        cos_embs = []
        sin_embs = []
        bsz = len(cu_seqlens) - 1

        for i in range(bsz):
            if step == 0:  # prefilling
                r = cu_seqlens[i + 1] - cu_seqlens[i]
                cos_emb, sin_emb = (
                    RotaryEmbedding.cos[: r * stride : stride],
                    RotaryEmbedding.sin[: r * stride : stride],
                )
            elif step > 0:  # decoding
                r = cu_seqlens[i + 1] - cu_seqlens[i] + step - 1
                cos_emb, sin_emb = (
                    RotaryEmbedding.cos[r * stride : r * stride + 1],
                    RotaryEmbedding.sin[r * stride : r * stride + 1],
                )
            cos_embs.append(cos_emb)
            sin_embs.append(sin_emb)

        cos_embs = torch.cat(cos_embs, dim=0)
        sin_embs = torch.cat(sin_embs, dim=0)
        return cos_embs, sin_embs

    def forward(self, x, cu_seqlens, step=0, stride=1):
        seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
        cos_embs, sin_embs = self.generate_pos_embs(
            x,
            cu_seqlens,
            seqlens,
            step=step,
            stride=stride,
        )
        N, H, D = x.shape[0], x.shape[-2], x.shape[-1]  # H: number of heads
        x = x * cos_embs.view(N, 1, D) + rotate_half(x) * sin_embs.view(N, 1, D)
        return x
