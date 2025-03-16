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
from typing import Optional, Tuple, Any
import torch
import tilelang
import tilelang.language as T
from einops import rearrange, einsum
from native_sparse_attention.ops.triton.utils import is_hopper_gpu


IS_HOPPER_GPU = is_hopper_gpu()

def linear_compress_func(batch_size, UX, UY, max_len_y, kernel_size, kernel_stride, head_num, headd_dim, headD_dim):
    x_shape = [UX, head_num, headd_dim]
    w_shape = [head_num, kernel_size, headD_dim]
    y_shape = [UY, head_num, headD_dim]
    block_Y = 16  # Adjusted to be a multiple of 16
    block_d = 64
    block_k = 32
    block_D = 16  # Already a multiple of 16
    num_stages = 0
    threads = 3

    dtype = "float16"
    accum_dtype = "float"

    def kernel_func(block_Y, block_D, num_stages, threads):

        @T.prim_func
        def main(
                X_unpad: T.Buffer(x_shape, dtype),
                W_unpad: T.Buffer(w_shape, dtype),
                cu_seqlens: T.Buffer([batch_size + 1], "int32"),
                cu_seqlens_y: T.Buffer([batch_size + 1], "int32"),
                Output_unpad: T.Buffer(y_shape, dtype),
        ):
            with T.Kernel(
                T.ceildiv(max_len_y, block_Y) * T.ceildiv(headD_dim, block_D), head_num, batch_size, 
                threads=threads) as (byD, bh, bb):
                by = byD // T.ceildiv(headD_dim, block_D)
                bD = byD % T.ceildiv(headD_dim, block_D)


                x_shared = T.alloc_shared([block_Y, block_k, block_d], dtype, "shared")
                w_shared = T.alloc_shared([block_k, block_d, block_D], dtype, "shared")
                
                y = T.alloc_fragment([block_Y, block_D], accum_dtype)


                batch_idx = bb
                x_start = cu_seqlens[batch_idx]
                x_end = cu_seqlens[batch_idx+ 1]
                x_len = x_end - x_start

                y_start = cu_seqlens_y[batch_idx]
                y_end = cu_seqlens_y[batch_idx+ 1]
                y_len = y_end - y_start


                x_pos = x_start + by * block_Y * kernel_stride
                for iy, ik, id in T.Parallel(block_Y, block_k, block_d):
                    x_loc = x_start + iy * kernel_stride + ik
                    if x_loc < x_len :
                        x_shared[iy, ik, id] = X_unpad[x_loc, bh, id]
                    else:
                        x_shared[iy, ik, id] = 0
                
                for ik, id, iD in T.Parallel(block_k, block_d, block_D):
                    if (id + bD * block_D) < headD_dim:
                        w_shared[ik, id, iD] = W_unpad[ik, id, iD]
                    else:
                        w_shared[ik, id, iD] = 0
                
                T.fill(y, 0)

                x_shared_block_k = T.alloc_shared([block_Y, block_d], dtype, "shared")
                w_shared_block_k = T.alloc_shared([block_d, block_D], dtype, "shared")

                # for ibk in T.Pipelined(block_k, num_stages=num_stages):
                #     for iiby, iibd in T.Parallel(block_Y, block_d):
                #         x_shared_block_k[iiby, iibd] = x_shared[iiby, ibk, iibd]
                #     for iibd, iibD in T.Parallel(block_d, block_D):
                #         w_shared_block_k[iibd, iibD] = w_shared[ibk, iibd, iibD]
                #     T.gemm(
                #         x_shared_block_k,
                #         w_shared_block_k,
                #         y,
                #         policy=T.GemmWarpPolicy.FullRow)
                
                # y_start_D = bD * block_D
                # y_start_SEQ = y_start + by * block_Y
                # for iy, iD in T.Parallel(block_Y, block_D):
                #     if (iy + by * block_Y) < y_len :
                #         Output_unpad[y_start_SEQ + iy, bh, iD] = y[iy, iD]
                #     else:
                #         Output_unpad[y_start_SEQ + iy, bh, iD] = 0

        return main
    
    return kernel_func(block_Y, block_D, num_stages, threads)





class LinearCompress(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        w: torch.Tensor,
        cu_seqlens: torch.Tensor,
        kernel_size: int,
        kernel_stride: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress key and value tensor with kernel_size and kernel_stride. Similar to conv_compress.

        Args:
            x (torch.Tensor): key_states or value_states, shape (total_len, num_heads, head_dim)
            w (torch.Tensor): weight for each head, shape (num_heads, kernel_size * head_dim, head_dim)
            cu_seqlens (torch.Tensor): shape [batch_size + 1], similar to cu_seqlens_q in flash_attn_func_varlen
            kernel_size (int): kernel_size, each (kernel_size, head_dim) blocks will be compressed to (1, head_dim)
            kernel_stride (int): stride for each compress kernel

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: compressed states and corresponding cu_seqlens.
        """
        # dtype check
        assert x.dtype == torch.float16 or x.dtype == torch.bfloat16
        assert x.dtype == w.dtype
        assert cu_seqlens.dtype == torch.int32

        # shape check
        total_len, num_heads, head_dim = x.shape
        batch_size = cu_seqlens.shape[0] - 1
        assert w.shape[0] == num_heads
        assert w.shape[1] == kernel_size * head_dim
        assert w.shape[2] == head_dim
        assert kernel_size % kernel_stride == 0
        assert kernel_size in {16, 32, 64, 128}
        assert head_dim % 8 == 0

        # compute seqlens after compression
        seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
        y_seqlens = (
            torch.floor((seqlens - kernel_size) / kernel_stride).to(torch.int32) + 1
        )
        # corner case: if sequence_length < kernel_size, no compression for this sequence
        y_seqlens[seqlens < kernel_size] = 0
        y_cu_seqlens = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device="cuda"),
                torch.cumsum(y_seqlens, dim=0),
            ],
            dim=0,
        ).to(torch.int32)

        y = torch.zeros(
            y_cu_seqlens[-1], num_heads, head_dim, dtype=x.dtype, device=x.device
        )

        block_kernel_size = 16
        block_head_dim = 8 
        block_headD_dim = 32
        block_output_seq_size = 64
        w = w.reshape(num_heads, kernel_size, head_dim, head_dim).contiguous()

        program = linear_compress_func(batch_size, cu_seqlens[-1].item(), y_cu_seqlens[-1].item(), y_seqlens.max(0)[0].item(), kernel_size, kernel_stride, num_heads, head_dim, head_dim)
        kernel = tilelang.compile(program, out_idx=-1, execution_backend="cython")
        print(kernel.get_kernel_source())

        y = kernel(x, w, cu_seqlens, y_cu_seqlens)

        # save for backward
        ctx.save_for_backward(x, w, cu_seqlens, y_seqlens, y_cu_seqlens)
        # save value
        ctx.kernel_size = kernel_size
        ctx.kernel_stride = kernel_stride
        ctx.block_kernel_size = block_kernel_size
        ctx.block_headd_dim = block_head_dim
        ctx.block_headD_dim = block_headD_dim
        ctx.block_output_seq_size = block_output_seq_size
        return y, y_cu_seqlens

    @staticmethod
    def backward(ctx, dy: torch.Tensor, *args) -> Any:
        x, w, cu_seqlens, y_seqlens, y_cu_seqlens = ctx.saved_tensors
        kernel_size = ctx.kernel_size
        kernel_stride = ctx.kernel_stride
        block_kernel_size = ctx.block_kernel_size
        block_head_dim = ctx.block_headd_dim
        block_headD_dim = ctx.block_headD_dim
        block_output_seq_size = ctx.block_output_seq_size

        total_len, num_heads, head_dim = x.shape
        batch_size = cu_seqlens.shape[0] - 1

        dx = torch.zeros(
            cu_seqlens[-1], num_heads, head_dim, dtype=torch.float32, device=x.device
        )

        dw = torch.zeros(
            num_heads,
            kernel_size,
            head_dim,
            head_dim,
            dtype=torch.float32,
            device=x.device,
        )



        return (
            dx.to(x.dtype),
            rearrange(dw.to(x.dtype), "n k d D -> n (k d) D"),
            None,
            None,
            None,
        )


def linear_compress(
    x: torch.Tensor,
    w: torch.Tensor,
    cu_seqlens: torch.Tensor,
    kernel_size: int,
    kernel_stride: int,
    pe: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compress key and value tensor with kernel_size and kernel_stride. Similar to conv_compress.

    Args:
        x (torch.Tensor): key_states or value_states, shape (total_len, num_heads, head_dim)
        w (torch.Tensor): weight for each head, shape (num_heads, kernel_size * head_dim, head_dim)
        cu_seqlens (_type_): shape [batch_size + 1], similar to cu_seqlens_q in flash_attn_func_varlen.
        kernel_size (int): kernel_size, each (kernel_size, head_dim) blocks will be compressed to (1, head_dim)
        kernel_stride (int): stride for each compress kernel
        pe (Optional[torch.Tensor], optional): intra-block positional embedding with shape (num_heads, kernel_size, head_dim). Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: compressed states and corresponding cu_seqlens.
    """
    y, y_cu_seqlens = LinearCompress.apply(x, w, cu_seqlens, kernel_size, kernel_stride)
    # position embedding as a bias
    if pe is not None:
        assert pe.dtype == x.dtype and pe.device == x.device
        pe = rearrange(pe, "h k d -> h (k d)")
        bias = einsum(pe, w, "h D, h D d -> h d")
        y = y + bias.unsqueeze(0)
    return y, y_cu_seqlens
