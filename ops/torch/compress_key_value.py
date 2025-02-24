# Copyright 2025 Xunhao Lai.
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
from typing import Optional, Tuple
from einops import rearrange


def conv_compress(
    x: torch.Tensor,
    w: torch.Tensor,
    cu_seqlens,
    kernel_size: int,
    kernel_stride: int,
    pe: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compress key and value tensor with kernel_size and kernel_stride.

    Args:
        x (torch.Tensor): key_states or value_states, shape (total_len, num_heads, head_dim)
        w (torch.Tensor): weight of conv1d, shape (num_heads * head_dim, head_dim, kernel_size)
        cu_seqlens (_type_): shape [batch_size + 1], similar to cu_seqlens_q in flash_attn_func_varlen.
        kernel_size (int): kernel_size, each (kernel_size, head_dim) blocks will be compressed to (1, head_dim)
        kernel_stride (int): kernel_stride for conv1d
        pe (Optional[torch.Tensor], optional): intra-block positional embedding with shape (num_heads, kernel_size, head_dim). Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: compressed states and corresponding cu_seqlens.
    """
    # dtype check
    assert x.dtype == torch.float16 or x.dtype == torch.bfloat16
    assert x.dtype == w.dtype
    assert x.dtype == pe.dtype if pe is not None else True
    assert cu_seqlens.dtype == torch.int32

    # shape check
    total_len, num_heads, head_dim = x.shape
    batch_size = cu_seqlens.shape[0] - 1
    assert num_heads * head_dim == w.shape[0]
    assert w.shape[1] == head_dim
    assert w.shape[2] == kernel_size
    assert kernel_size % kernel_stride == 0
    assert kernel_size in {16, 32, 64, 128}

    # compute seqlens after compression
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    y_seqlens = torch.floor((seqlens - kernel_size) / kernel_stride).to(torch.int32) + 1
    y_cu_seqlens = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device="cuda"),
            torch.cumsum(y_seqlens, dim=0),
        ],
        dim=0,
    ).to(torch.int32)

    # pad and rearrange x
    x = rearrange(x, "n h d -> n (h d)")
    splited_x = torch.split(x, seqlens.tolist(), 0)
    x = torch.nn.utils.rnn.pad_sequence(splited_x, batch_first=True)
    x = rearrange(x, "b n d -> b d n")
    # conv1d
    y = torch.nn.functional.conv1d(x, w, stride=kernel_stride, groups=num_heads)
    y = rearrange(y, "b (h d) n -> b n h d", h=num_heads)
    # only keep useful part
    y = torch.cat([y[i, : y_seqlens[i]] for i in range(batch_size)], dim=0)

    # position embedding as a bias
    if pe is not None:
        bias = torch.nn.functional.conv1d(
            rearrange(pe, "h n d -> (h d) n"),
            w,
            stride=kernel_stride,
            groups=num_heads,
        )
        bias = rearrange(bias, "(h d) 1 -> 1 h d", h=num_heads)
        y = y + bias
    return y, y_cu_seqlens
