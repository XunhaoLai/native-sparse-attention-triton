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

# compress method
from native_sparse_attention.ops.triton.weighted_pool import (
    weightedpool_compress,
    avgpool_compress,
)
from native_sparse_attention.ops.triton.linear_compress import linear_compress

# prefill attention
from native_sparse_attention.ops.triton.flash_attention import flash_attention_varlen
from native_sparse_attention.ops.triton.compressed_attention import compressed_attention
from native_sparse_attention.ops.triton.topk_sparse_attention import (
    topk_sparse_attention,
)

# decode attention
from native_sparse_attention.ops.triton.flash_attention_decode import (
    flash_attention_decode,
)
from native_sparse_attention.ops.torch.compressed_attention_decode import (
    compressed_attention_decode,
)
from native_sparse_attention.ops.triton.topk_sparse_attention_decode import (
    topk_sparse_attention_decode,
)

__all__ = [
    # compress method
    "avgpool_compress",
    "weightedpool_compress",
    "linear_compress",
    # prefill attention, trainable
    "flash_attention_varlen",
    "compressed_attention",
    "topk_sparse_attention",
    # decode attention, no grad
    "flash_attention_decode",
    "compressed_attention_decode",
    "topk_sparse_attention_decode",
]
