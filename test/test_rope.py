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
from native_sparse_attention.module import RopeConfig, RotaryEmbedding

if __name__ == "__main__":
    rope_config = RopeConfig(
        max_position_embeddings=131072,
        head_dim=128,
        rope_theta=500000,
        rope_scaling={
            "factor": 8.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3",
        },
    )
    rope = RotaryEmbedding(rope_config, "cuda")

    # random input
    torch.manual_seed(42)
    seqlens = torch.LongTensor([1000, 2000, 4096]).int().cuda()
    cu_seqlens = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device="cuda"),
            torch.cumsum(seqlens, dim=0),
        ],
        dim=0,
    ).to(torch.int32)
    x = torch.zeros(
        cu_seqlens[-1], 32, 128, device="cuda", dtype=torch.bfloat16
    ).uniform_(-1, 1)
    y = rope(x, cu_seqlens)
