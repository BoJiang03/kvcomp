import torch
from typing import Optional, List

from utils.compute import cut_tensor, quant, safe_cat
import kvcomp_cuda_extension

K_COMP_BLOCK_SIZE = 64
V_COMP_BLOCK_SIZE = 128
K_QUANT_BLOCK_SIZE = 64
V_QUANT_BLOCK_SIZE = 64
K_BUFFER_SIZE = 0
V_BUFFER_SIZE = 0
K_QUANT_DIMS = [1,2,3,4]
# K_QUANT_DIMS = [0,1,2,3,4]
V_QUANT_DIMS = [2,3,4]
# V_QUANT_DIMS = [0,1,2,3,4]

class KVComCtx:
    buffer_ctx_len: int = 0
    block_infos_buffer: Optional[torch.Tensor] = None
    thread_infos_buffer: Optional[torch.Tensor] = None
    idx_offset_buffer: Optional[torch.Tensor] = None
    pre_alloc_compressed_data_buffer: Optional[torch.Tensor] = None
    def __init__(
            self,
            is_k: bool,
            batch_size: int,
            head_num: int,
            head_dim: int,
            device: torch.device,
            quant_scale_rel: float,
            encode_codebook_path: str = None,
            decode_codebook_path: str = None
    ):
        assert batch_size == 1, "only support batch size == 1 for now"
        self.is_k = is_k
        self.batch_size: int = batch_size
        self.head_num: int = head_num
        self.head_dim: int = head_dim
        self.ctx_len: int = 0
        self.compressed_ctx_len: int = 0
        self.device: torch.device = device
        self.quant_scale_rel = quant_scale_rel
        self.quant_dims = K_QUANT_DIMS if is_k else V_QUANT_DIMS
        self.comp_block_size = K_COMP_BLOCK_SIZE if self.is_k else V_COMP_BLOCK_SIZE
        self.quant_block_size = K_QUANT_BLOCK_SIZE if self.is_k else V_QUANT_BLOCK_SIZE
        self.buffer_size = K_BUFFER_SIZE if self.is_k else V_BUFFER_SIZE

        if self.is_k:
            assert (batch_size * head_num * head_dim) % kvcomp_cuda_extension.K_VEC_LEN == 0, "hidden size must be divisible by VEC_LEN"
            assert self.comp_block_size % kvcomp_cuda_extension.K_VEC_PER_BLK == 0, "block size must be divisible by VEC_PER_BLK"
            assert self.quant_block_size == self.comp_block_size == kvcomp_cuda_extension.K_VEC_PER_BLK, "for K, we only support block size 64 for now"
        else:
            assert (batch_size * head_num * head_dim) % kvcomp_cuda_extension.V_VEC_PER_BLK == 0, "hidden size must be divisible by VEC_PER_BLK"
            assert self.comp_block_size % kvcomp_cuda_extension.V_VEC_LEN == 0, "block size must be divisible by VEC_LEN"

        self.quant_min_ints: Optional[torch.Tensor] = None
        self.quant_scales: Optional[torch.Tensor] = None
        self.encoded_data: Optional[torch.Tensor] = None
        self.block_infos: Optional[torch.Tensor] = None
        self.thread_infos: Optional[torch.Tensor] = None
        self.buffer: Optional[torch.Tensor] = None

        # temp
        self.encode_codebook: Optional[torch.Tensor] = None
        self.decode_codebook: Optional[torch.Tensor] = None

    def store(self, cache: torch.Tensor):
        assert cache.dtype == torch.float16, "cache must be float16"
        assert cache.shape[1] == self.batch_size, "batch size mismatch"
        assert cache.shape[2] == self.head_num, "head num mismatch"
        assert cache.shape[3] == self.head_dim, "head dim mismatch"
        self.ctx_len += cache.shape[0]
        to_compress, self.buffer = cut_tensor(self.buffer, cache, self.comp_block_size, self.buffer_size, dim=0)
        blocked_tensor = to_compress.view(-1, self.quant_block_size, self.batch_size, self.head_num, self.head_dim)
        quant_ints, min_ints, quant_scale = quant(blocked_tensor, self.quant_dims, self.quant_scale_rel)
        if KVComCtx.buffer_ctx_len < to_compress.shape[0]:
            block_infos_buffer_size, thread_infos_buffer_size, idx_offset_buffer_size, pre_alloc_compressed_data_size = kvcomp_cuda_extension.calculate_buffer_size(quant_ints, self.is_k, self.head_num, self.head_dim)
            KVComCtx.buffer_ctx_len = to_compress.shape[0]
            KVComCtx.block_infos_buffer = torch.empty(block_infos_buffer_size, dtype=torch.uint8, device=self.device)
            KVComCtx.thread_infos_buffer = torch.empty(thread_infos_buffer_size, dtype=torch.uint8, device=self.device)
            KVComCtx.idx_offset_buffer = torch.empty(idx_offset_buffer_size, dtype=torch.uint8, device=self.device)
            KVComCtx.pre_alloc_compressed_data_buffer = torch.empty(pre_alloc_compressed_data_size, dtype=torch.uint8, device=self.device)

        min_ints = min_ints.to(torch.int8)
        quant_ints = quant_ints.to(torch.uint8)
        if self.encode_codebook is None:
            shift, encode_codebook_, decode_codebook_ = kvcomp_cuda_extension.build_codebook([quant_ints])
            assert shift == 0, "shift must be 0"
            self.encode_codebook = encode_codebook_.to(self.device)
            self.decode_codebook = decode_codebook_.to(self.device)
        if self.is_k:
            encoded_data_, block_infos_, vec_infos_ = kvcomp_cuda_extension.k_entropy_encode_cuda_export(
                quant_ints,
                self.encode_codebook,
                KVComCtx.block_infos_buffer,
                KVComCtx.thread_infos_buffer,
                KVComCtx.idx_offset_buffer,
                KVComCtx.pre_alloc_compressed_data_buffer,
                self.encoded_data.numel() if self.encoded_data is not None else 0,
                self.head_num,
                self.head_dim
            )
        else:
            encoded_data_, block_infos_, vec_infos_ = kvcomp_cuda_extension.v_entropy_encode_cuda_export(
                quant_ints,
                self.encode_codebook,
                KVComCtx.block_infos_buffer,
                KVComCtx.thread_infos_buffer,
                KVComCtx.idx_offset_buffer,
                KVComCtx.pre_alloc_compressed_data_buffer,
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

    def get(self) -> torch.Tensor:
        decoded_quant_ints = self.get_decode_quant_ints()
        return (decoded_quant_ints + self.quant_min_ints) * self.quant_scales

    def get_decode_quant_ints(self) -> torch.Tensor:
        # only for testing
        assert self.quant_min_ints is not None, "quant_min_ints is None"
        assert self.quant_scales is not None, "quant_scales is None"
        assert self.encoded_data is not None, "encoded_data is None"
        assert self.block_infos is not None, "huffman_block_info is None"
        assert self.thread_infos is not None, "huffman_thread_info is None"
        assert self.decode_codebook is not None, "decode_codebook is None"

        decoded_data = torch.empty(
            self.compressed_ctx_len // self.quant_block_size,
            self.quant_block_size,
            self.batch_size,
            self.head_num,
            self.head_dim,
            dtype=torch.uint8,
            device=self.device
        )

        if self.is_k:
            kvcomp_cuda_extension.k_entropy_decode_cuda_export(
                self.encoded_data,
                self.block_infos,
                self.thread_infos,
                self.decode_codebook,
                decoded_data,
                self.head_num,
                self.head_dim
            )
        else:
            kvcomp_cuda_extension.v_entropy_decode_cuda_export(
                self.encoded_data,
                self.block_infos,
                self.thread_infos,
                self.decode_codebook,
                decoded_data,
                self.head_num,
                self.head_dim
            )

        return decoded_data

    def mat_vec_mul(self, B: torch.Tensor) -> torch.Tensor:
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

        # C = torch.empty(
        #     self.compressed_ctx_len,
        #     self.batch_size,
        #     self.head_num,
        #     dtype=torch.float16,
        #     device=self.device
        # )

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
                self.compressed_ctx_len // kvcomp_cuda_extension.V_VEC_LEN,
                self.batch_size,
                self.head_num,
                self.head_dim,
                dtype=torch.float16,
                device=self.device
            )

        if self.is_k:
            kvcomp_cuda_extension.k_decode_and_mat_vec_mul_cuda_export(
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
            kvcomp_cuda_extension.v_decode_and_mat_vec_mul_cuda_export(
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
            C = C.sum(dim=0, keepdim=True)

        return C

    def build_codebook(self, tensors: List[torch.Tensor]):
        quant_ints = None
        assert len(tensors) > 0, "tensors must not be empty"
        for cache in tensors:
            assert cache.dtype == torch.float16, "cache must be float16"
            assert cache.shape[1] == self.batch_size, "batch size mismatch"
            assert cache.shape[2] == self.head_num, "head num mismatch"
            assert cache.shape[3] == self.head_dim, "head dim mismatch"
            blocked_len = cache.shape[0] // self.comp_block_size * self.comp_block_size
            cache = cache[:blocked_len]
            blocked_tensor = cache.view(-1, self.quant_block_size, self.batch_size, self.head_num, self.head_dim)
            quant_ints_, _, _ = quant(blocked_tensor, self.quant_dims, self.quant_scale_rel)
            quant_ints_ = quant_ints_.to(torch.uint8).flatten()
            quant_ints = safe_cat(quant_ints, quant_ints_, dim=0)

        shift, encode_codebook_, decode_codebook_ = kvcomp_cuda_extension.build_codebook([quant_ints])
        assert shift == 0, "shift must be 0"
        print("decode_codebook size: ", decode_codebook_.numel())
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

