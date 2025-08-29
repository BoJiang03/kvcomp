from typing import List, Tuple
import torch

K_VEC_LEN: int
K_VEC_PER_BLK: int
V_VEC_LEN: int
V_VEC_PER_BLK: int
BLOCK_SIZE: int
K_BUFFER_SIZE: int
V_BUFFER_SIZE: int
K_QUANT_DIMS: List[int]
V_QUANT_DIMS: List[int]

def calculate_buffer_size(input: torch.Tensor, is_k: bool, head_num: int, head_dim: int) -> Tuple[int, int, int, int]: ...
def build_codebook(input_tensors: List[torch.Tensor]) -> Tuple[int, torch.Tensor, torch.Tensor]: ...
def k_entropy_encode_cuda_export(
        input: torch.Tensor,
        encode_code_book: torch.Tensor,
        block_infos: torch.Tensor,
        thread_infos: torch.Tensor,
        idx_offset: torch.Tensor,
        encoded_data: torch.Tensor,
        base_global_offset: int,
        head_num: int,
        head_dim: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
def v_entropy_encode_cuda_export(
        input: torch.Tensor,
        encode_code_book: torch.Tensor,
        block_infos: torch.Tensor,
        thread_infos: torch.Tensor,
        idx_offset: torch.Tensor,
        encoded_data: torch.Tensor,
        base_global_offset: int,
        head_num: int,
        head_dim: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
def k_entropy_decode_cuda_export(
        in_tensor: torch.Tensor,
        block_infos: torch.Tensor,
        thread_infos: torch.Tensor,
        decode_code_book: torch.Tensor,
        out: torch.Tensor,
        head_num: int,
        head_dim: int
) -> float: ...
def v_entropy_decode_cuda_export(
        in_tensor: torch.Tensor,
        block_infos: torch.Tensor,
        thread_infos: torch.Tensor,
        decode_code_book: torch.Tensor,
        out: torch.Tensor,
        head_num: int,
        head_dim: int
) -> float: ...
def k_decode_and_mat_vec_mul_cuda_export(
        decode_code_book: torch.Tensor,
        block_infos: torch.Tensor,
        thread_infos: torch.Tensor,
        encoded_data: torch.Tensor,
        quant_min_ints: torch.Tensor,
        quant_scales: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        head_num: int,
        head_dim: int
) -> float: ...
def v_decode_and_mat_vec_mul_cuda_export(
        decode_code_book: torch.Tensor,
        block_infos: torch.Tensor,
        thread_infos: torch.Tensor,
        encoded_data: torch.Tensor,
        quant_min_ints: torch.Tensor,
        quant_scales: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        head_num: int,
        head_dim: int
) -> float: ...
def k_mat_vec_mul_cuda_export(
        k: torch.Tensor,
        q: torch.Tensor,
        out: torch.Tensor
) -> float: ...