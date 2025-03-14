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
from native_sparse_attention.module.kv_cache import NSACache


if __name__ == "__main__":
    from native_sparse_attention.ops import avgpool_compress

    torch.manual_seed(42)

    num_heads = 4
    head_dim = 128
    seqlens = torch.tensor([12, 576, 12000]).to(torch.int32).cuda()
    batch_size = seqlens.shape[0]
    cu_seqlens = torch.zeros(seqlens.shape[0] + 1, dtype=torch.int32, device="cuda")
    cu_seqlens[1:] = seqlens.cumsum(0)

    # init cache
    cache = NSACache(4, 16384, num_heads, head_dim, 32, 16, 512, torch.bfloat16, "cuda")

    # test prefill
    step = 0
    k = torch.randn(cu_seqlens[-1], num_heads, head_dim).cuda().bfloat16()
    v = torch.randn_like(k)
    ck, _ = avgpool_compress(k, None, cu_seqlens, 32, 16, None)
    cv, _ = avgpool_compress(v, None, cu_seqlens, 32, 16, None)
    cache.prepare_compress(cu_seqlens, step, k, v)
    cache.update_kv(cu_seqlens, step, ck, cv, k, v, k, v)

    # test decode
    step = 1
    k = torch.randn(batch_size, num_heads, head_dim).cuda().bfloat16()
    v = torch.randn_like(k)
    ck = torch.randn_like(k)
    cv = torch.randn_like(v)
    cache.prepare_compress(cu_seqlens, step, k, v)
    cache.update_kv(cu_seqlens, step, ck, cv, k, v, k, v)
