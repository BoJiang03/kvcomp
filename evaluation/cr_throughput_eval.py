import math
from typing import List

import torch
from sympy.physics.quantum.qubit import Qubit

from kvcomp_cuda_extension.scripts.temp import ctx_len
from models.cache.kvcomp_encode import KVCompCtx
from models.cache.kvcomp_quant import KVCompCacheConfigStatic
from utils.compute import QuantMode
from utils.config import KVCompCacheConfig

def quant_rel_to_bit_num(quant_rel: float):
    state_num = math.ceil(1 / quant_rel) + 1
    bit_num = math.ceil(math.log2(state_num))
    return bit_num

def check_error(a: torch.Tensor, b: torch.Tensor, percentage: float):
    delta = (a - b).abs()
    assert (delta < a.abs().max() * percentage).all(), "Error too large"


def k_do_transformers_mat_vec_mul_original_layout(k: torch.Tensor, q: torch.Tensor):
    q_o_t = q.unsqueeze(2)
    k_o_t = k.permute(1, 2, 0, 3)
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    k_o_t = k_o_t.transpose(2, 3)
    torch.cuda.synchronize()

    start_time.record()
    C_target = torch.matmul(q_o_t, k_o_t)
    end_time.record()
    torch.cuda.synchronize()

    C_target = C_target * 1.0
    C_target = C_target.squeeze(2).permute(2, 0, 1)
    return C_target, start_time.elapsed_time(end_time)


def v_do_transformers_mat_vec_mul_original_layout(attn_weights_t: torch.Tensor, v: torch.Tensor):
    attn_weights_t = attn_weights_t.unsqueeze(2)
    v = v.permute(1, 2, 0, 3)

    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_time.record()
    C_target = torch.matmul(attn_weights_t, v)
    end_time.record()
    torch.cuda.synchronize()

    C_target = C_target.transpose(1, 2)
    return C_target, start_time.elapsed_time(end_time)  # Return the result and the elapsed time in milliseconds


def k_get_multi_kernel_time(
        ctx: KVCompCtx,
) -> float:
    assert ctx.is_k, "k_get_multi_kernel_time only works for k"
    decompressed_k, time_ = ctx.get_only_compressed()
    # q_t = torch.empty(
    #     (ctx.batch_size, ctx.head_num, ctx.head_dim),
    #     dtype=torch.float16,
    #     device=ctx.device
    # ).uniform_(-1, 1)
    # decompressed_k = decompressed_k.view(-1, ctx.batch_size, ctx.head_num, ctx.head_dim)
    # rt_t, time__ = k_do_transformers_mat_vec_mul_original_layout(decompressed_k, q_t)
    # time = time_ + time__
    return time_

def k_get_fused_kernel_time(
        ctx: KVCompCtx,
) -> float:
    assert ctx.is_k, "k_get_fused_kernel_time only works for k"
    q_t = torch.empty(
        (ctx.batch_size, ctx.head_num, ctx.head_dim),
        dtype=torch.float16,
        device=ctx.device
    ).uniform_(-1, 1)
    decompressed_k, time = ctx.get_only_compressed()
    decompressed_k = decompressed_k.view(-1, ctx.batch_size, ctx.head_num, ctx.head_dim)
    rt_t, time_ = k_do_transformers_mat_vec_mul_original_layout(decompressed_k, q_t)
    rt_t_, time___ = ctx.mat_vec_mul(q_t)
    rt_t_ = rt_t_.permute(2, 0, 1).contiguous()
    check_error(rt_t_, rt_t, 0.01)
    assert time___ > 0, "Invalid time"
    assert time_ > 0, "Invalid time"
    return time___ - time_, time_, time___

def v_get_multi_kernel_time(
        ctx: KVCompCtx,
) -> float:
    assert not ctx.is_k, "v_get_multi_kernel_time only works for v"
    decompressed_v, time_ = ctx.get_only_compressed()
    # attn_weights_t = torch.randn(
    #     (ctx.batch_size, ctx.head_num, ctx.compressed_ctx_len),
    #     dtype=torch.float16,
    #     device=ctx.device
    # ) * 2.5
    # attn_weights_t = torch.softmax(attn_weights_t, dim=-1)
    # decompressed_v = decompressed_v.view(-1, ctx.batch_size, ctx.head_num, ctx.head_dim)
    # _, time__ = v_do_transformers_mat_vec_mul_original_layout(attn_weights_t, decompressed_v)
    return time_


def v_get_fused_kernel_time(
        ctx: KVCompCtx,
) -> float:
    assert not ctx.is_k, "v_get_fused_kernel_time only works for v"
    attn_weights_t = torch.randn(
        (ctx.batch_size, ctx.head_num, ctx.compressed_ctx_len),
        dtype=torch.float16,
        device=ctx.device
    ) * 2.5
    attn_weights_t = torch.softmax(attn_weights_t, dim=-1)
    decompressed_v, time = ctx.get_only_compressed()
    decompressed_v = decompressed_v.view(-1, ctx.batch_size, ctx.head_num, ctx.head_dim)
    rt_t, time_ = v_do_transformers_mat_vec_mul_original_layout(attn_weights_t, decompressed_v)
    rt_t_, time___ = ctx.mat_vec_mul(attn_weights_t)
    check_error(rt_t_, rt_t, 0.01)
    assert time___ > 0, "Invalid time"
    assert time_ > 0, "Invalid time"
    return time___ - time_, time_, time___


def crs_throughputs_evaluation_with_data(
        config: KVCompCacheConfig,
        key_caches: List[torch.Tensor],
        value_caches: List[torch.Tensor]
):
    KVCompCacheConfigStatic.config = config
    batch_size = key_caches[0].shape[0]
    head_num = key_caches[0].shape[1]
    head_dim = key_caches[0].shape[3]
    layer_num = len(key_caches)

    k_original_sizes = [0] * layer_num
    k_quant_sizes = [0] * layer_num
    k_encode_sizes = [0] * layer_num
    v_original_sizes = [0] * layer_num
    v_quant_sizes = [0] * layer_num
    v_encode_sizes = [0] * layer_num
    k_multi_kernel_times = [0] * layer_num
    k_fused_kernel_times = [0] * layer_num
    v_multi_kernel_times = [0] * layer_num
    v_fused_kernel_times = [0] * layer_num
    k_pytorch_mat_vec_mul_times = [0] * layer_num
    k_fused_mat_vec_mul_times = [0] * layer_num
    v_pytorch_mat_vec_mul_times = [0] * layer_num
    v_fused_mat_vec_mul_times = [0] * layer_num

    index_list = [i for i in range(10)] # for warming up
    for i in range(layer_num):
        index_list.append(i)

    for i in index_list:
        if config.k_quant_mode == QuantMode.BlockQuant:
            # for i in range(1):
            k_ctx = KVCompCtx(True, batch_size, head_num, head_dim, KVCompCacheConfigStatic.config.k_quant_scale_rel)
            v_ctx = KVCompCtx(False, batch_size, head_num, head_dim, KVCompCacheConfigStatic.config.v_quant_scale_rel)
            key_states = key_caches[i].permute(2, 0, 1, 3)
            value_states = value_caches[i].permute(2, 0, 1, 3)

            k_ctx.store(key_states)
            k_original_size_ = k_ctx.compressed_ctx_len * batch_size * head_num * head_dim * 16
            k_original_sizes[i] = k_original_size_
            quant_meta_data_size = k_ctx.quant_min_ints.numel() * 8 + k_ctx.quant_scales.numel() * 16
            encode_meta_data_size = k_ctx.block_infos.numel() * 8 + k_ctx.thread_infos.numel() * 8
            k_quant_size_ = k_ctx.compressed_ctx_len * batch_size * head_num * head_dim * quant_rel_to_bit_num(
                config.k_quant_scale_rel)
            k_quant_size_ += quant_meta_data_size
            k_quant_sizes[i] = k_quant_size_
            k_encode_size_ = k_ctx.encoded_data.numel() * 8
            k_encode_size_ += encode_meta_data_size + quant_meta_data_size
            k_encode_sizes[i] = k_encode_size_

            v_ctx.store(value_states)
            v_original_size_ = v_ctx.compressed_ctx_len * batch_size * head_num * head_dim * 16
            v_original_sizes[i] = v_original_size_
            quant_meta_data_size = v_ctx.quant_min_ints.numel() * 8 + v_ctx.quant_scales.numel() * 16
            encode_meta_data_size = v_ctx.block_infos.numel() * 8 + v_ctx.thread_infos.numel() * 8
            v_quant_size_ = v_ctx.compressed_ctx_len * batch_size * head_num * head_dim * quant_rel_to_bit_num(
                config.v_quant_scale_rel)
            v_quant_size_ += quant_meta_data_size
            v_quant_sizes[i] = v_quant_size_
            v_encode_size_ = v_ctx.encoded_data.numel() * 8
            v_encode_size_ += encode_meta_data_size + quant_meta_data_size
            v_encode_sizes[i] = v_encode_size_

            k_multi_kernel_time_ = k_get_multi_kernel_time(k_ctx)
            k_fused_kernel_time_, k_pytorch_time, k_fuse_time = k_get_fused_kernel_time(k_ctx)
            v_multi_kernel_time_ = v_get_multi_kernel_time(v_ctx)
            v_fused_kernel_time_, v_pytorch_time, v_fuse_time = v_get_fused_kernel_time(v_ctx)

            # print(f"Layer {i}: k_multi_kernel_time: {k_multi_kernel_time_}, k_fused_kernel_time: {k_fused_kernel_time_}, v_multi_kernel_time: {v_multi_kernel_time_}, v_fused_kernel_time: {v_fused_kernel_time_}")
            k_multi_kernel_times[i] = k_multi_kernel_time_
            k_fused_kernel_times[i] = k_fused_kernel_time_
            v_multi_kernel_times[i] = v_multi_kernel_time_
            v_fused_kernel_times[i] = v_fused_kernel_time_
            k_pytorch_mat_vec_mul_times[i] = k_pytorch_time
            k_fused_mat_vec_mul_times[i] = k_fuse_time
            v_pytorch_mat_vec_mul_times[i] = v_pytorch_time
            v_fused_mat_vec_mul_times[i] = v_fuse_time
        else:
            assert config.k_quant_mode == QuantMode.ChannelQuant
            assert config.v_quant_mode == QuantMode.TokenQuant
            ctx_len_ = key_caches[0].shape[2]
            k_block_num = ctx_len_ // config.k_block_size
            k_compressed_ctx_len = k_block_num * config.k_block_size
            k_quant_unit_num = k_block_num * head_num * head_dim
            k_original_sizes[i] = k_compressed_ctx_len * batch_size * head_num * head_dim * 16
            k_quant_sizes[i] = k_compressed_ctx_len * batch_size * head_num * head_dim * quant_rel_to_bit_num(config.k_quant_scale_rel)
            k_quant_sizes[i] += k_quant_unit_num * 8 + k_quant_unit_num * 16 # quant_min(assume this is int8) + quant_scale(this is float16)
            v_original_sizes[i] = ctx_len_ * batch_size * head_num * head_dim * 16
            v_quant_sizes[i] = ctx_len_ * batch_size * head_num * head_dim * quant_rel_to_bit_num(
                config.v_quant_scale_rel)
            v_quant_sizes[i] += ctx_len_ * 8 + ctx_len_ * 16 # quant_min(assume this is int8) + quant_scale(this is float16)

    rt = {
        "k_quant_crs": [k_original_size/ k_quant_size for k_original_size, k_quant_size in zip(k_original_sizes, k_quant_sizes)],
        "v_quant_crs": [v_original_size/ v_quant_size for v_original_size, v_quant_size in zip(v_original_sizes, v_quant_sizes)],
    }

    if config.k_quant_mode == QuantMode.BlockQuant:
        rt["k_huffman_crs"] = [k_original_size / k_encode_size for k_original_size, k_encode_size in zip(k_original_sizes, k_encode_sizes)]
        rt["v_huffman_crs"] = [v_original_size / v_encode_size for v_original_size, v_encode_size in zip(v_original_sizes, v_encode_sizes)]
        rt["k_time_multi_kernels"] = k_multi_kernel_times,
        rt["v_time_multi_kernels"] = v_multi_kernel_times,
        rt["k_pytorch_mat_vec_mul_times"] = k_pytorch_mat_vec_mul_times,
        rt["v_pytorch_mat_vec_mul_times"] = v_pytorch_mat_vec_mul_times,
        rt["k_fused_mat_vec_mul_times"] = k_fused_mat_vec_mul_times,
        rt["v_fused_mat_vec_mul_times"] = v_fused_mat_vec_mul_times,
        rt["k_throughput_multi_kernels"] = [k_original_size / (8 * 1024 * 1024 * 1024) / (k_multi_kernel_time / 1000) for k_original_size, k_multi_kernel_time in zip(k_original_sizes, k_multi_kernel_times)]
        rt["k_throughput_fused_kernels"] = [k_original_size / (8 * 1024 * 1024 * 1024) / (k_fused_kernel_time / 1000) if k_fused_kernel_time !=0 else float("inf") for k_original_size, k_fused_kernel_time in zip(k_original_sizes, k_fused_kernel_times)]
        rt["v_throughput_multi_kernels"] = [v_original_size / (8 * 1024 * 1024 * 1024) / (v_multi_kernel_time / 1000) for v_original_size, v_multi_kernel_time in zip(v_original_sizes, v_multi_kernel_times)]
        rt["v_throughput_fused_kernels"] = [v_original_size / (8 * 1024 * 1024 * 1024) / (v_fused_kernel_time / 1000) if v_fused_kernel_time !=0 else float("inf")  for v_original_size, v_fused_kernel_time in zip(v_original_sizes, v_fused_kernel_times)]

    return rt