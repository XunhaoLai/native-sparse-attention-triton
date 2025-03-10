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
import triton
import torch
import triton.language as tl
from einops import rearrange, einsum


@triton.jit
def linear_compress_fwd_paralleld_kernel(
    X,  # input pointer [total_len, num_heads, head_dim]
    Y,  # output pointer [total_compressed_len, num_heads, head_dim]
    W,  # weight matrix pointer [num_heads, kernel_size, head_dim, head_dim]
    cu_seqlens_x,  # cumulative sequence lengths before compression
    cu_seqlens_y,  # cumulative sequence lengths after compression
    stride_xn,  # stride for X's sequence dimension
    stride_xh,  # stride for X's num head dimension
    stride_xd,  # stride for X's head_dim dimension
    stride_wh,  # stride for W's num head dimension
    stride_wk,  # stride for W's kernel size  dimension
    stride_wd,  # stride for W's initial head dim dimension
    stride_wD,  # stride for W's final head dim dimension
    stride_yn,  # stride for Y's sequence dimension
    stride_yh,  # stride for Y's num head dimension
    stride_yd,  # stride for Y's head_dim dimension
    NUM_HEADS: tl.constexpr,  # total num heads
    KERNEL_SIZE: tl.constexpr,  # kernel size when calculate the output
    KERNEL_STRIDE: tl.constexpr,  # kernel stride when calculate the output
    HEADd_DIM: tl.constexpr,  # initial head dimension size
    HEADD_DIM: tl.constexpr,  # final head dimension size
    BLOCK_KERNEL_SIZE: tl.constexpr,  # Loaded kernel size when calculate the output
    BLOCK_HEADd_DIM: tl.constexpr,  # Loaded  orignal head dimension size
    BLOCK_HEADD_DIM: tl.constexpr,  # loaded final head dimension size
):
    pid_bh = tl.program_id(0)
    pid_b = pid_bh // NUM_HEADS
    pid_h = pid_bh % NUM_HEADS
    pid_k = tl.program_id(1)
    pid_D = tl.program_id(2)

    x_start = tl.load(cu_seqlens_x + pid_b)
    x_end = tl.load(cu_seqlens_x + pid_b + 1)
    x_len = x_end - x_start

    y_start = tl.load(cu_seqlens_y + pid_b)
    y_end = tl.load(cu_seqlens_y + pid_b + 1)
    y_len = y_end - y_start
    if pid_k >= y_len:
        return

    x_ptrs = tl.make_block_ptr(
        base=X + x_start * stride_xn + pid_h * stride_xh,
        shape=(x_len, HEADd_DIM),
        strides=(stride_xn, stride_xd),
        offsets=(pid_k * KERNEL_STRIDE, 0),
        block_shape=(BLOCK_KERNEL_SIZE, BLOCK_HEADd_DIM),
        order=(1, 0),
    )

    w_ptrs = tl.make_block_ptr(
        base=W + pid_h * stride_wh,
        shape=(KERNEL_SIZE, HEADd_DIM, HEADD_DIM),
        strides=(stride_wk, stride_wd, stride_wD),
        offsets=(0, 0, pid_D * BLOCK_HEADD_DIM),
        block_shape=(BLOCK_KERNEL_SIZE, BLOCK_HEADd_DIM, BLOCK_HEADD_DIM),
        order=(2, 1, 0),
    )

    y_ptrs = tl.make_block_ptr(
        base=Y + (y_start + pid_k) * stride_yn + pid_h * stride_yh,
        shape=(HEADD_DIM,),
        strides=(stride_yd,),
        offsets=(pid_D * BLOCK_HEADD_DIM,),
        block_shape=(BLOCK_HEADD_DIM,),
        order=(0,),
    )

    x = tl.load(x_ptrs, boundary_check=(0, 1), padding_option="zero")
    # x : [k, d]
    w = tl.load(w_ptrs, boundary_check=(0, 1, 2), padding_option="zero")
    # w: [k, d, D]

    y_d = tl.sum(tl.sum(x[:, :, None] * w, axis=0), axis=0)  # Sum over k and d
    # y_d = tl.reshape(y_d, (BLOCK_HEADD_DIM,))
    #  y_d : [D]

    tl.store(y_ptrs, y_d.to(y_ptrs.dtype.element_ty), boundary_check=(0,))


@triton.jit
def linear_compress_bwd_paralleld_kernel(
    DX,  # X's gradient pointer [total_len, num_heads, head_dim]
    DY,  # Y's gradient pointer [total_compressed_len, num_heads, head_dim]
    DW,  # weight's gradient pointer [num_heads, kernel_size, head_dim, head_dim]
    X,  # x pointer [total_len, num_heads, head_dim]
    W,  # weight matrix pointer [num_heads, kernel_size, head_dim, head_dim]
    cu_seqlens_x,  # cumulative sequence lengths before compression
    cu_seqlens_y,  # cumulative sequence lengths after compression
    stride_xn,  # stride for X's sequence dimension
    stride_xh,  # stride for X's num head dimension
    stride_xd,  # stride for X's head_dim dimension
    stride_wh,  # stride for W's num head dimension
    stride_wk,  # stride for W's kernel size  dimension
    stride_wd,  # stride for W's initial head dim dimension
    stride_wD,  # stride for W's final head dim dimension
    stride_dxn,  # stride for DX's sequence dimension
    stride_dxh,  # stride for DX's num head dimension
    stride_dxd,  # stride for DX's head_dim dimension
    stride_dwh,  # stride for DW's num head dimension
    stride_dwk,  # stride for DW's kernel size  dimension
    stride_dwd,  # stride for DW's initial head dim dimension
    stride_dwD,  # stride for DW's final head dim dimension
    stride_dyn,  # stride for DY's sequence dimension
    stride_dyh,  # stride for DY's num head dimension
    stride_dyd,  # stride for DY's head_dim dimension
    NUM_HEADS: tl.constexpr,  # total num heads
    KERNEL_SIZE: tl.constexpr,  # kernel size when calculate the output
    KERNEL_STRIDE: tl.constexpr,  # kernel stride when calculate the output
    HEADd_DIM: tl.constexpr,  # initial head dimension size
    HEADD_DIM: tl.constexpr,  # final head dimension size
    BLOCK_KERNEL_SIZE: tl.constexpr,  # Loaded kernel size when calculate the output
    BLOCK_HEADd_DIM: tl.constexpr,  # Loaded  orignal head dimension size
    BLOCK_HEADD_DIM: tl.constexpr,  # loaded final head dimension size
):
    pid_bh = tl.program_id(0)
    pid_b = pid_bh // NUM_HEADS
    pid_h = pid_bh % NUM_HEADS
    pid_k = tl.program_id(1)
    pid_D = tl.program_id(2)

    x_start = tl.load(cu_seqlens_x + pid_b)
    x_end = tl.load(cu_seqlens_x + pid_b + 1)
    x_len = x_end - x_start

    y_start = tl.load(cu_seqlens_y + pid_b)
    y_end = tl.load(cu_seqlens_y + pid_b + 1)
    y_len = y_end - y_start
    if pid_k >= y_len:
        return

    x_ptrs = tl.make_block_ptr(
        base=X + x_start * stride_xn + pid_h * stride_xh,
        shape=(x_len, HEADd_DIM),
        strides=(stride_xn, stride_xd),
        offsets=(pid_k * KERNEL_STRIDE, 0),
        block_shape=(BLOCK_KERNEL_SIZE, BLOCK_HEADd_DIM),
        order=(1, 0),
    )

    w_ptrs = tl.make_block_ptr(
        base=W + pid_h * stride_wh,
        shape=(KERNEL_SIZE, HEADd_DIM, HEADD_DIM),
        strides=(stride_wk, stride_wd, stride_wD),
        offsets=(0, 0, pid_D * BLOCK_HEADD_DIM),
        block_shape=(BLOCK_KERNEL_SIZE, BLOCK_HEADd_DIM, BLOCK_HEADD_DIM),
        order=(2, 1, 0),
    )

    # dx_ptrs = tl.make_block_ptr(
    #     base=DX + x_start * stride_dxn + pid_h * stride_dxh,
    #     shape=(x_len, HEADd_DIM),
    #     strides=(stride_dxn, stride_dxd),
    #     offsets=(pid_k * KERNEL_STRIDE, 0),
    #     block_shape=(BLOCK_KERNEL_SIZE, BLOCK_HEADd_DIM),
    #     order=(1, 0)
    # )

    # dw_ptrs = tl.make_block_ptr(
    #     base=DW + pid_h * stride_dwh,
    #     shape=(KERNEL_SIZE, HEADd_DIM, HEADD_DIM),
    #     strides=(stride_dwk, stride_dwd, stride_dwD),
    #     offsets=(0, 0, pid_D * BLOCK_HEADD_DIM),
    #     block_shape=(BLOCK_KERNEL_SIZE, BLOCK_HEADd_DIM, BLOCK_HEADD_DIM),
    #     order=(2, 1, 0)
    # )

    dy_ptrs = tl.make_block_ptr(
        base=DY + (y_start + pid_k) * stride_dyn + pid_h * stride_dyh,
        shape=(HEADD_DIM,),
        strides=(stride_dyd,),
        offsets=(pid_D * BLOCK_HEADD_DIM,),
        block_shape=(BLOCK_HEADD_DIM,),
        order=(0,),
    )

    dy = tl.load(dy_ptrs, boundary_check=(0,), padding_option="zero")
    # dy : [D, ]

    # cal dx, start
    w = tl.load(w_ptrs, boundary_check=(0, 1, 2), padding_option="zero")
    # w: [k, d, D]

    dx = tl.sum(dy[None, None, :] * w, axis=2)
    # dx: [k, d]

    off_k = tl.arange(0, BLOCK_KERNEL_SIZE)
    off_d = tl.arange(0, BLOCK_HEADd_DIM)
    off_D = tl.arange(0, BLOCK_HEADD_DIM)

    dx_ptrs = (
        DX
        + pid_h * stride_dxh
        + (x_start + pid_k * KERNEL_STRIDE + off_k[:, None]) * stride_dxn
        + off_d[None, :] * stride_dxd
    )
    tl.atomic_add(
        dx_ptrs,
        dx.to(dx_ptrs.dtype.element_ty),
        mask=(
            (off_k < x_len - pid_k * KERNEL_STRIDE)[:, None]
            & (off_d < HEADD_DIM)[None, :]
        ),
    )
    # cal dx, end

    # cal dw, start
    x = tl.load(x_ptrs, boundary_check=(0, 1), padding_option="zero")
    # x : [k, d]

    dw_ptrs = (
        DW
        + pid_h * stride_dwh
        + off_k[:, None, None] * stride_dwk
        + off_d[None, :, None] * stride_dwd
        + (pid_D * BLOCK_HEADD_DIM + off_D[None, None, :]) * stride_dwD
    )
    dw = x[:, :, None] * dy[None, None, :]
    tl.atomic_add(
        dw_ptrs,
        dw.to(dw_ptrs.dtype.element_ty),
        mask=(
            (off_k < KERNEL_SIZE)[:, None, None]
            & (off_d < HEADd_DIM)[None, :, None]
            & (pid_D * BLOCK_HEADD_DIM + off_D < HEADD_DIM)[None, None, :]
        ),
    )


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

        block_kernel_size = triton.next_power_of_2(kernel_size)
        block_head_dim = triton.next_power_of_2(head_dim)
        block_headD_dim = 32
        w = w.reshape(num_heads, kernel_size, head_dim, head_dim).contiguous()

        grid = lambda META: (
            batch_size * num_heads,
            y_seqlens.max(0)[0].item(),
            triton.cdiv(head_dim, META["BLOCK_HEADD_DIM"]),
        )

        linear_compress_fwd_paralleld_kernel[grid](
            x,
            y,
            w,
            cu_seqlens,
            y_cu_seqlens,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            w.stride(0),
            w.stride(1),
            w.stride(2),
            w.stride(3),
            y.stride(0),
            y.stride(1),
            y.stride(2),
            num_heads,
            kernel_size,
            kernel_stride,
            head_dim,
            head_dim,
            block_kernel_size,
            block_head_dim,
            block_headD_dim,
            num_warps=8,
            num_stages=3,
        )
        # save for backward
        ctx.save_for_backward(x, w, cu_seqlens, y_seqlens, y_cu_seqlens)
        # save value
        ctx.kernel_size = kernel_size
        ctx.kernel_stride = kernel_stride
        ctx.block_kernel_size = block_kernel_size
        ctx.block_headd_dim = block_head_dim
        ctx.block_headD_dim = block_headD_dim
        return y, y_cu_seqlens

    @staticmethod
    def backward(ctx, dy: torch.Tensor, *args) -> Any:
        x, w, cu_seqlens, y_seqlens, y_cu_seqlens = ctx.saved_tensors
        kernel_size = ctx.kernel_size
        kernel_stride = ctx.kernel_stride
        block_kernel_size = ctx.block_kernel_size
        block_head_dim = ctx.block_headd_dim
        block_headD_dim = ctx.block_headD_dim

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

        grid = lambda META: (
            batch_size * num_heads,
            y_seqlens.max(0)[0].item(),
            triton.cdiv(head_dim, META["BLOCK_HEADD_DIM"]),
        )

        linear_compress_bwd_paralleld_kernel[grid](
            dx,
            dy,
            dw,
            x,
            w,
            cu_seqlens,
            y_cu_seqlens,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            w.stride(0),
            w.stride(1),
            w.stride(2),
            w.stride(3),
            dx.stride(0),
            dx.stride(1),
            dx.stride(2),
            dw.stride(0),
            dw.stride(1),
            dw.stride(2),
            dw.stride(3),
            dy.stride(0),
            dy.stride(1),
            dy.stride(2),
            num_heads,
            kernel_size,
            kernel_stride,
            head_dim,
            head_dim,
            block_kernel_size,
            block_head_dim,
            block_headD_dim,
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
    """
    Wrapper function for LinearCompress.apply
    """
    y, y_cu_seqlens = LinearCompress.apply(x, w, cu_seqlens, kernel_size, kernel_stride)
    # position embedding as a bias
    if pe is not None:
        assert pe.dtype == x.dtype and pe.device == x.device
        pe = rearrange(pe, "h k d -> h (k d)")
        bias = einsum(pe, w, "h D, h D d -> h d")
        y = y + bias.unsqueeze(0)
    return y, y_cu_seqlens


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    x = torch.randn(128, 4, 64, requires_grad=True, device="cuda", dtype=torch.float16)
    w = torch.randn(
        4, 16 * 64, 64, requires_grad=True, device="cuda", dtype=torch.float16
    )
    cu_seqlens = torch.tensor([0, 32, 64, 96, 128], dtype=torch.int32, device="cuda")
    kernel_size = 16
    kernel_stride = 4
    pe = None
    torch.autograd.gradcheck(
        LinearCompress.apply,
        (x, w, cu_seqlens, kernel_size, kernel_stride, pe),
        fast_mode=False,
    )
