import torch
from typing import Tuple, Optional, List
from enum import Enum

class QuantMode(Enum):
    # LayerQuant = "LayerQuant"
    BlockQuant = "BlockQuant"
    ChannelQuant = "ChannelQuant"
    TokenQuant = "TokenQuant"
    VectorQuant = "VectorQuant"

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def apply_rotary_pos_emb_single(
        t: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids=None, unsqueeze_dim=1) -> torch.Tensor:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    t_embed = (t * cos) + (rotate_half(t) * sin)
    return t_embed

def safe_cat(t1, t2, dim):
    if t1 is None and t2 is None:
        return None
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
        to_compress = buffer[:, :, :to_compress_block_num * block_size, :]
        buffer = buffer[:, :, to_compress_block_num * block_size:, :]
    return to_compress, buffer

def cut_tensor_ctx_len_0(
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

def quant_ints(
        tensor: torch.Tensor,
        block_size: int,
        quant_scale_rel: float,
        quant_mode: QuantMode,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert tensor.shape[2] % block_size == 0, "Tensor shape is not divisible by block size"
    tensor = tensor.reshape(tensor.shape[0], tensor.shape[1], -1, block_size, tensor.shape[3])
    quant_dim = QUANT_DIM[quant_mode.value]
    min_val = tensor
    max_val = tensor
    for i in quant_dim:
        min_val = min_val.min(dim=i, keepdim=True).values
        max_val = max_val.max(dim=i, keepdim=True).values
    quant_scale = (max_val - min_val) * quant_scale_rel
    min_quant = (min_val / quant_scale).round()
    value_quant = (tensor/ quant_scale).round() - min_quant
    return value_quant, min_quant, quant_scale

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

def quant_error(
        error_cache: torch.Tensor,
        buffer: torch.Tensor,
        new_tensor: torch.Tensor,
        block_size: int,
        recent_size: int,
        quant_scale_rel: float,
        quant_mode: QuantMode,
) -> Tuple[torch.Tensor, torch.Tensor]:
    to_compress, in_buffer = cut_tensor(
        buffer, new_tensor,
        block_size, recent_size,
        dim=2
    )

    if to_compress is not None:
        quant_int, min_quant, quant_scale = quant_ints(
            to_compress,
            block_size,
            quant_scale_rel,
            quant_mode
        )
        to_compress = (quant_int + min_quant) * quant_scale
        to_compress = to_compress.reshape(to_compress.shape[0], to_compress.shape[1], -1, to_compress.shape[4])

    return safe_cat(error_cache, to_compress, dim=2), in_buffer

def print_quant_setting(logger):
    logger.info(QUANT_DIM)

QUANT_DIM = {
    QuantMode.BlockQuant.value: [1,3,4],
    QuantMode.ChannelQuant.value: [3],
    QuantMode.TokenQuant.value: [1,4],
    QuantMode.VectorQuant.value: [4],
}

