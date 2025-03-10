import torch


def is_hopper_gpu():
    if torch.cuda.is_available():
        device_capability = torch.cuda.get_device_capability()
        major, minor = device_capability
        return major == 9
    return False


def get_compressed_seqlens(
    cu_seqlens: torch.Tensor, kernel_size: int, kernel_stride: int
):
    # compute seqlens after compression
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    y_seqlens = torch.floor((seqlens - kernel_size) / kernel_stride).to(torch.int32) + 1
    # corner case, if sequence_length < kernel_size, no compression for this sequence
    y_seqlens[seqlens < kernel_size] = 0
    y_cu_seqlens = torch.zeros(
        y_seqlens.shape[0] + 1, dtype=torch.int32, device=cu_seqlens.device
    )
    y_cu_seqlens[1:] = torch.cumsum(y_seqlens, dim=0)
    return y_seqlens, y_cu_seqlens
