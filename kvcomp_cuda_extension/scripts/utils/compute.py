import torch
from typing import Tuple, Optional, List

def safe_cat(t1, t2, dim):
    if t1 is None:
        return t2.clone()
    if t2 is None:
        return t1.clone()
    return torch.cat([t1, t2], dim=dim)

def cut_tensor(
        buffer, new_tensor,
        block_size, recent_size,
        dim=2
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    buffer = safe_cat(buffer, new_tensor, dim)
    len_ = buffer.shape[dim]
    res_num = len_ % block_size
    to_compress_block_num = (len_ + block_size - res_num - recent_size) // block_size
    to_compress = None
    if to_compress_block_num > 0:
        to_compress = buffer[:to_compress_block_num * block_size, :, :, :]
        buffer = buffer[to_compress_block_num * block_size:, :, :, :]
    return to_compress, buffer

def quant(
        tensor: torch.Tensor,
        quant_dims: List[int],
        quant_scale_rel: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    min_ = tensor
    max_ = tensor
    for dim in quant_dims:
        min_ = min_.min(dim=dim, keepdim=True).values
        max_ = max_.max(dim=dim, keepdim=True).values
    quant_scale = (max_ - min_) * quant_scale_rel
    min_ints = (min_ / quant_scale).round_() #.to(torch.int8)
    quant_ints = (tensor / quant_scale).round_() #.to(torch.int8)
    return quant_ints - min_ints, min_ints, quant_scale