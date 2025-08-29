from typing import Optional, Dict, Any, Tuple

import torch
from transformers import Cache
from utils.compute import apply_rotary_pos_emb_single, safe_cat, quant_error
import os

from utils.util import visualize_2d_tensor, get_logger, JumpOutException

class KVCompCacheConfigStatic:
    config = None
    extract_cache = None

class KVCompCachePytorchQuant(Cache):
    round_ = 0
    def __init__(self, batch_size, head_num, head_dim, layer_num):
        KVCompCachePytorchQuant.round_+=1
        super().__init__()
        self.batch_size = batch_size
        self.head_num = head_num
        self.head_dim = head_dim
        self.compressed_k_cache = [None] * layer_num
        self.compressed_v_cache = [None] * layer_num
        self.k_cache_buffer = [None] * layer_num
        self.v_cache_buffer = [None] * layer_num
        self.coss = [None] * layer_num
        self.sins = [None] * layer_num

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if KVCompCacheConfigStatic.extract_cache is not None:
            assert not KVCompCacheConfigStatic.config.enable_quant, "Disable quant to enable cache collect."
        if KVCompCacheConfigStatic.config is None:
            raise ValueError("TComCacheConfigStatic.config is not set. Please set it before using TComCache.")
        if KVCompCacheConfigStatic.config.enable_quant is False:
            return self.update_disable(key_states, value_states, layer_idx, cache_kwargs)

        cos, sin = cache_kwargs['cos'], cache_kwargs['sin']
        if KVCompCacheConfigStatic.config.enable_pre_rope:
            self.coss[layer_idx] = safe_cat(self.coss[layer_idx], cos, dim=1)
            self.sins[layer_idx] = safe_cat(self.sins[layer_idx], sin, dim=1)
        else:
            key_states = apply_rotary_pos_emb_single(key_states, cos, sin)

        self.compressed_k_cache[layer_idx], self.k_cache_buffer[layer_idx] = quant_error(
            self.compressed_k_cache[layer_idx],
            self.k_cache_buffer[layer_idx],
            key_states,
            KVCompCacheConfigStatic.config.k_block_size,
            KVCompCacheConfigStatic.config.k_recent_size,
            KVCompCacheConfigStatic.config.k_quant_scale_rel,
            KVCompCacheConfigStatic.config.k_quant_mode
        )

        self.compressed_v_cache[layer_idx], self.v_cache_buffer[layer_idx] = quant_error(
            self.compressed_v_cache[layer_idx],
            self.v_cache_buffer[layer_idx],
            value_states,
            KVCompCacheConfigStatic.config.v_block_size,
            KVCompCacheConfigStatic.config.v_recent_size,
            KVCompCacheConfigStatic.config.v_quant_scale_rel,
            KVCompCacheConfigStatic.config.v_quant_mode
        )

        if KVCompCacheConfigStatic.config.enable_pre_rope:
            return apply_rotary_pos_emb_single(safe_cat(self.compressed_k_cache[layer_idx], self.k_cache_buffer[layer_idx], dim=2), self.coss[layer_idx], self.sins[layer_idx]), \
                   safe_cat(self.compressed_v_cache[layer_idx], self.v_cache_buffer[layer_idx], dim=2)
        else:
            return safe_cat(self.compressed_k_cache[layer_idx], self.k_cache_buffer[layer_idx], dim=2), \
                    safe_cat(self.compressed_v_cache[layer_idx], self.v_cache_buffer[layer_idx], dim=2)

    def update_disable(self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cos, sin = cache_kwargs['cos'], cache_kwargs['sin']
        if KVCompCacheConfigStatic.config.enable_pre_rope:
            self.coss[layer_idx] = safe_cat(self.coss[layer_idx], cos, dim=1)
            self.sins[layer_idx] = safe_cat(self.sins[layer_idx], sin, dim=1)
        else:
            key_states = apply_rotary_pos_emb_single(key_states, cos, sin)

        self.k_cache_buffer[layer_idx] = safe_cat(self.k_cache_buffer[layer_idx], key_states, dim=2)
        self.v_cache_buffer[layer_idx] = safe_cat(self.v_cache_buffer[layer_idx], value_states, dim=2)

        if KVCompCacheConfigStatic.extract_cache is not None:
            if key_states.shape[2] != 1 and KVCompCachePytorchQuant.round_ > KVCompCacheConfigStatic.extract_cache.collect_round:
                raise JumpOutException("\nExtracting cache and new generation round has been detected, jump out to process cache.")
            if KVCompCachePytorchQuant.round_ - 1 not in KVCompCacheConfigStatic.extract_cache.key_caches:
                KVCompCacheConfigStatic.extract_cache.key_caches[KVCompCachePytorchQuant.round_ - 1] = {}
                KVCompCacheConfigStatic.extract_cache.value_caches[KVCompCachePytorchQuant.round_ - 1] = {}
            KVCompCacheConfigStatic.extract_cache.key_caches[KVCompCachePytorchQuant.round_ - 1][layer_idx - 1] = self.k_cache_buffer[layer_idx]
            KVCompCacheConfigStatic.extract_cache.value_caches[KVCompCachePytorchQuant.round_ - 1][layer_idx - 1] = self.v_cache_buffer[layer_idx]
            # print(f"collected KV cache size: {}")

        if KVCompCacheConfigStatic.config.enable_pre_rope:
            return apply_rotary_pos_emb_single(self.k_cache_buffer[layer_idx], self.coss[layer_idx], self.sins[layer_idx]), self.v_cache_buffer[layer_idx]
        else:
            return self.k_cache_buffer[layer_idx], self.v_cache_buffer[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if KVCompCacheConfigStatic.config is None:
            raise ValueError("TComCacheConfigStatic.config is not set. Please set it before using TComCache.")
        if KVCompCacheConfigStatic.config.enable_quant is False:
            if layer_idx is None:
                return self.k_cache_buffer[0].shape[2]
            else:
                if self.k_cache_buffer[layer_idx] is None:
                    return 0
                return self.k_cache_buffer[layer_idx].shape[2]

        if layer_idx is None:
            self.compressed_k_cache[0].shape[2] + self.k_cache_buffer[0].shape[2]
        else:
            if self.compressed_k_cache[layer_idx] is None and self.k_cache_buffer[layer_idx] is None:
                return 0
            return self.compressed_k_cache[layer_idx].shape[2] + self.k_cache_buffer[layer_idx].shape[2]