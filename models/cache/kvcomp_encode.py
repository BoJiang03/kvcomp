from typing import Optional, Dict, Any, Tuple, List

import torch
from transformers import Cache
from utils.compute import apply_rotary_pos_emb_single, safe_cat, QuantMode
import kvcomp_cuda
from utils.compute import cut_tensor_ctx_len_0, quant
from models.cache.kvcomp_quant import KVCompCacheConfigStatic

QUANT_MODE_MAP = {
    QuantMode.BlockQuant: [1,2,3,4],
    QuantMode.TokenQuant: [2,3,4],
}

class KVCompCtx:
    # sub_round_ = 0
    buffer_ctx_len: Dict[torch.device, int] = dict()
    block_infos_buffer: Dict[torch.device, torch.Tensor] = dict()
    thread_infos_buffer: Dict[torch.device, torch.Tensor] = dict()
    idx_offset_buffer: Dict[torch.device, torch.Tensor] = dict()
    pre_alloc_compressed_data_buffer: Dict[torch.device, torch.Tensor] = dict()
    def __init__(
            self,
            is_k: bool,
            batch_size: int,
            head_num: int,
            head_dim: int,
            quant_scale_rel: float,
            encode_codebook_path: str = None,
            decode_codebook_path: str = None
    ):
        assert KVCompCacheConfigStatic.config is not None, "KVCompCacheConfigStatic.config is None"
        assert KVCompCacheConfigStatic.config.k_quant_mode == QuantMode.BlockQuant, "k_quant_mode must be BlockQuant for cuda kernel implementation"
        assert KVCompCacheConfigStatic.config.v_quant_mode == QuantMode.TokenQuant, "v_quant_mode must be TokenQuant for cuda kernel implementation"
        assert head_dim == kvcomp_cuda.K_VEC_LEN == kvcomp_cuda.V_VEC_LEN, "head_dim must be equal to K_VEC_LEN and V_VEC_LEN for cuda kernel implementation"
        self.is_k = is_k
        self.batch_size: int = batch_size
        self.head_num: int = head_num
        self.head_dim: int = head_dim
        self.quant_scale_rel = quant_scale_rel
        self.ctx_len: int = 0
        self.compressed_ctx_len: int = 0
        self.device: Optional[torch.device] = None
        self.quant_dims = QUANT_MODE_MAP[KVCompCacheConfigStatic.config.k_quant_mode] if is_k else QUANT_MODE_MAP[KVCompCacheConfigStatic.config.v_quant_mode]
        self.block_size = KVCompCacheConfigStatic.config.k_block_size if self.is_k else KVCompCacheConfigStatic.config.v_block_size
        self.recent_size = KVCompCacheConfigStatic.config.k_recent_size if self.is_k else KVCompCacheConfigStatic.config.v_recent_size

        if self.is_k:
            assert (batch_size * head_num * head_dim) % kvcomp_cuda.K_VEC_LEN == 0, "hidden size must be divisible by VEC_LEN"
            assert self.block_size % kvcomp_cuda.K_VEC_PER_BLK == 0, "block size must be divisible by VEC_PER_BLK"
            assert self.block_size == kvcomp_cuda.K_VEC_PER_BLK, "for K, block size must be divisible by VEC_PER_BLK"
        else:
            assert (batch_size * head_num * head_dim) % kvcomp_cuda.V_VEC_PER_BLK == 0, "hidden size must be divisible by VEC_PER_BLK"
            assert self.block_size % kvcomp_cuda.V_VEC_LEN == 0, "block size must be divisible by VEC_LEN"

        self.quant_min_ints: Optional[torch.Tensor] = None
        self.quant_scales: Optional[torch.Tensor] = None
        self.encoded_data: Optional[torch.Tensor] = None
        self.block_infos: Optional[torch.Tensor] = None
        self.thread_infos: Optional[torch.Tensor] = None
        self.buffer: Optional[torch.Tensor] = None

        # temp
        self.encode_codebook: Optional[torch.Tensor] = None
        self.decode_codebook: Optional[torch.Tensor] = None

        # self.temp_: torch.Tensor = None

    def store(self, cache: torch.Tensor):
        # print(f"delta len: {cache.shape[0]}")
        # self.temp_ = safe_cat(self.temp_, cache, dim=0)
        assert cache.dtype == torch.float16, "cache must be float16"
        assert cache.shape[1] == self.batch_size, "batch size mismatch"
        assert cache.shape[2] == self.head_num, "head num mismatch"
        assert cache.shape[3] == self.head_dim, "head dim mismatch"
        if self.device is None:
            self.device = cache.device
        assert self.device == cache.device, "device mismatch"
        self.ctx_len += cache.shape[0]
        to_compress, self.buffer = cut_tensor_ctx_len_0(self.buffer, cache, self.block_size, self.recent_size, dim=0)
        if to_compress is None:
            return
        blocked_tensor = to_compress.view(-1, self.block_size, self.batch_size, self.head_num, self.head_dim)
        quant_ints, min_ints, quant_scale = quant(blocked_tensor, self.quant_dims, self.quant_scale_rel)
        block_infos_buffer_size, thread_infos_buffer_size, idx_offset_buffer_size, pre_alloc_compressed_data_size = kvcomp_cuda.calculate_buffer_size(
            quant_ints, self.is_k, self.head_num, self.head_dim)
        KVCompCtx.buffer_ctx_len[self.device] = to_compress.shape[0]
        KVCompCtx.block_infos_buffer[self.device] = torch.zeros(block_infos_buffer_size, dtype=torch.uint8,
                                                                device=self.device)
        KVCompCtx.thread_infos_buffer[self.device] = torch.zeros(thread_infos_buffer_size, dtype=torch.uint8,
                                                                 device=self.device)
        KVCompCtx.idx_offset_buffer[self.device] = torch.zeros(idx_offset_buffer_size, dtype=torch.uint8,
                                                               device=self.device)
        KVCompCtx.pre_alloc_compressed_data_buffer[self.device] = torch.zeros(pre_alloc_compressed_data_size,
                                                                              dtype=torch.uint8, device=self.device)

        min_ints = min_ints.to(torch.int8)
        quant_ints = quant_ints.to(torch.uint8)
        if self.encode_codebook is None:
            shift, encode_codebook_, decode_codebook_ = kvcomp_cuda.build_codebook([quant_ints])
            assert shift == 0, "shift must be 0"
            self.encode_codebook = encode_codebook_.to(self.device)
            self.decode_codebook = decode_codebook_.to(self.device)
        if self.is_k:
            encoded_data_, block_infos_, vec_infos_ = kvcomp_cuda.k_entropy_encode_cuda_export(
                quant_ints,
                self.encode_codebook,
                KVCompCtx.block_infos_buffer[self.device],
                KVCompCtx.thread_infos_buffer[self.device],
                KVCompCtx.idx_offset_buffer[self.device],
                KVCompCtx.pre_alloc_compressed_data_buffer[self.device],
                self.encoded_data.numel() if self.encoded_data is not None else 0,
                self.head_num,
                self.head_dim
            )
        else:
            encoded_data_, block_infos_, vec_infos_ = kvcomp_cuda.v_entropy_encode_cuda_export(
                quant_ints,
                self.encode_codebook,
                KVCompCtx.block_infos_buffer[self.device],
                KVCompCtx.thread_infos_buffer[self.device],
                KVCompCtx.idx_offset_buffer[self.device],
                KVCompCtx.pre_alloc_compressed_data_buffer[self.device],
                self.encoded_data.numel() if self.encoded_data is not None else 0,
                self.head_num,
                self.head_dim
            )

        self.encoded_data = safe_cat(self.encoded_data, encoded_data_, dim=0)
        self.block_infos = safe_cat(self.block_infos, block_infos_, dim=0)
        self.thread_infos = safe_cat(self.thread_infos, vec_infos_, dim=0)
        self.quant_min_ints = safe_cat(self.quant_min_ints, min_ints, dim=0)
        self.quant_scales = safe_cat(self.quant_scales, quant_scale, dim=0)
        self.compressed_ctx_len += quant_ints.shape[0] * quant_ints.shape[1]

        return quant_ints

    def get(self) -> Tuple[torch.Tensor, float]:
        # print(f"in .get(): {KVCompCtx.sub_round_}")
        # KVCompCtx.sub_round_+=1
        decoded_quant_ints, time_ = self.get_decode_quant_ints()
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_time.record()
        decompressed_ = (decoded_quant_ints + self.quant_min_ints) * self.quant_scales
        decompressed_ = decompressed_.flatten(0, 1)
        end_time.record()
        torch.cuda.synchronize()
        time_ += start_time.elapsed_time(end_time)
        return torch.cat([decompressed_, self.buffer], dim=0), time_

    def get_only_compressed(self) -> Tuple[torch.Tensor, float]:
        # print(f"in .get(): {KVCompCtx.sub_round_}")
        # KVCompCtx.sub_round_+=1
        decoded_quant_ints, time_ = self.get_decode_quant_ints()
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_time.record()
        decompressed_ = (decoded_quant_ints + self.quant_min_ints) * self.quant_scales
        end_time.record()
        torch.cuda.synchronize()
        time_ += start_time.elapsed_time(end_time)
        decompressed_ = decompressed_.flatten(0, 1)
        return decompressed_, time_

    def get_decode_quant_ints(self) -> Tuple[torch.Tensor, float]:
        # only for testing
        assert self.quant_min_ints is not None, "quant_min_ints is None"
        assert self.quant_scales is not None, "quant_scales is None"
        assert self.encoded_data is not None, "encoded_data is None"
        assert self.block_infos is not None, "huffman_block_info is None"
        assert self.thread_infos is not None, "huffman_thread_info is None"
        assert self.decode_codebook is not None, "decode_codebook is None"

        decoded_data = torch.empty(
            self.compressed_ctx_len // self.block_size,
            self.block_size,
            self.batch_size,
            self.head_num,
            self.head_dim,
            dtype=torch.uint8,
            device=self.device
        )
        if self.is_k:
            time_ = kvcomp_cuda.k_entropy_decode_cuda_export(
                self.encoded_data,
                self.block_infos,
                self.thread_infos,
                self.decode_codebook,
                decoded_data,
                self.head_num,
                self.head_dim
            )
        else:
            time_ = kvcomp_cuda.v_entropy_decode_cuda_export(
                self.encoded_data,
                self.block_infos,
                self.thread_infos,
                self.decode_codebook,
                decoded_data,
                self.head_num,
                self.head_dim
            )

        return decoded_data, time_

    def mat_vec_mul(self, B: torch.Tensor) -> Tuple[torch.Tensor, float]:
        assert self.quant_min_ints is not None, "quant_min_ints is None"
        assert self.quant_scales is not None, "quant_scales is None"
        assert self.encoded_data is not None, "encoded_data is None"
        assert self.block_infos is not None, "huffman_block_info is None"
        assert self.thread_infos is not None, "huffman_thread_info is None"
        assert self.decode_codebook is not None, "decode_codebook is None"
        assert B.dtype == torch.float16, "B must be float16"
        assert len(B.shape) == 3, "B must be 3D tensor"
        if self.is_k:
            assert B.shape[0] == self.batch_size, "batch size mismatch"
            assert B.shape[1] == self.head_num, "head num mismatch"
            assert B.shape[2] == self.head_dim, "head dim mismatch"
        else:
            assert B.shape[0] == self.batch_size, "batch size mismatch"
            assert B.shape[1] == self.head_num, "head num mismatch"
            assert B.shape[2] == self.compressed_ctx_len, "ctx len mismatch"

        if self.is_k:
            C = torch.empty(
                self.batch_size,
                self.head_num,
                self.compressed_ctx_len,
                dtype=torch.float16,
                device=self.device
            )
        else:
            C = torch.empty(
                self.compressed_ctx_len // kvcomp_cuda.V_VEC_LEN,
                self.batch_size,
                self.head_num,
                self.head_dim,
                dtype=torch.float16,
                device=self.device
            )

        if self.is_k:
            t_ms = kvcomp_cuda.k_decode_and_mat_vec_mul_cuda_export(
                self.decode_codebook,
                self.block_infos,
                self.thread_infos,
                self.encoded_data,
                self.quant_min_ints,
                self.quant_scales,
                B,
                C,
                self.head_num,
                self.head_dim
            )
        else:
            t_ms = kvcomp_cuda.v_decode_and_mat_vec_mul_cuda_export(
                self.decode_codebook,
                self.block_infos,
                self.thread_infos,
                self.encoded_data,
                self.quant_min_ints,
                self.quant_scales,
                B,
                C,
                self.head_num,
                self.head_dim
            )
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_time.record()
            C = C.sum(dim=0, keepdim=True)
            end_time.record()
            torch.cuda.synchronize()
            t_ms += start_time.elapsed_time(end_time)

        return C, t_ms

    def build_codebook(self, tensors: List[torch.Tensor]):
        quant_ints = None
        assert len(tensors) > 0, "tensors must not be empty"
        device_ = tensors[0].device
        for cache in tensors:
            assert cache.dtype == torch.float16, "cache must be float16"
            assert cache.shape[1] == self.batch_size, "batch size mismatch"
            assert cache.shape[2] == self.head_num, "head num mismatch"
            assert cache.shape[3] == self.head_dim, "head dim mismatch"
            blocked_len = cache.shape[0] // self.block_size * self.block_size
            cache = cache[:blocked_len]
            blocked_tensor = cache.view(-1, self.block_size, self.batch_size, self.head_num, self.head_dim)
            quant_ints_, _, _ = quant(blocked_tensor, self.quant_dims, self.quant_scale_rel)
            quant_ints_ = quant_ints_.to(torch.uint8).flatten()
            quant_ints = safe_cat(quant_ints, quant_ints_, dim=0)
            assert device_ == cache.device, "device mismatch"

        shift, encode_codebook_, decode_codebook_ = kvcomp_cuda.build_codebook([quant_ints])
        assert shift == 0, "shift must be 0"
        # print("decode_codebook size: ", decode_codebook_.numel())
        if self.device is None:
            self.device = device_
            KVCompCtx.buffer_ctx_len[self.device] = 0
        self.encode_codebook = encode_codebook_.to(self.device)
        self.decode_codebook = decode_codebook_.to(self.device)

    @property
    def shape(self):
        return torch.Size([
            self.compressed_ctx_len,
            self.batch_size,
            self.head_num,
            self.head_dim
        ])

    def get_cr(self):
        compressed_size = (
                self.block_infos.numel() +
                self.thread_infos.numel() +
                self.encoded_data.numel() +
                self.quant_min_ints.numel() +
                self.quant_scales.numel() * 2
        )
        original_size = 2 * self.batch_size * self.compressed_ctx_len * self.head_num * self.head_dim
        return original_size / compressed_size

class KVCompCacheCuda(Cache):
    round_ = 0
    def __init__(self, batch_size, head_num, head_dim, layer_num):
        super().__init__()
        KVCompCacheCuda.round_ += 1
        assert batch_size == 1, "only support batch size == 1 for now"
        self.batch_size = batch_size
        self.head_num = head_num
        self.head_dim = head_dim
        self.k_cache_buffer = [KVCompCtx(True, batch_size, head_num, head_dim, KVCompCacheConfigStatic.config.k_quant_scale_rel) for _ in range(layer_num)]
        self.v_cache_buffer = [KVCompCtx(False, batch_size, head_num, head_dim, KVCompCacheConfigStatic.config.v_quant_scale_rel) for _ in range(layer_num)]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cos, sin = cache_kwargs['cos'], cache_kwargs['sin']
        key_states = apply_rotary_pos_emb_single(key_states, cos, sin)

        key_states = key_states.permute(2,0,1,3)
        value_states = value_states.permute(2,0,1,3)

        # print(f"layer_idx: {layer_idx}")
        self.k_cache_buffer[layer_idx].store(key_states)
        self.v_cache_buffer[layer_idx].store(value_states)

        k_t = self.k_cache_buffer[layer_idx].get().permute(1,2,0,3)
        v_t = self.v_cache_buffer[layer_idx].get().permute(1,2,0,3)

        return k_t, v_t

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if layer_idx is None:
            return self.k_cache_buffer[0].shape[2]
        else:
            if self.k_cache_buffer[layer_idx] is None:
                return 0
            return self.k_cache_buffer[layer_idx].shape[2]
