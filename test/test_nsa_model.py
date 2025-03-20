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
from native_sparse_attention.model import (
    ToyNSALlamaConfig,
    InferenceConfig,
    ToyNSALlama,
)


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
