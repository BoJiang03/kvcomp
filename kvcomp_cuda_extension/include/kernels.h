//
// Created by tut44803 on 2/27/25.
//

#ifndef TCOM_KERNELS_H
#define TCOM_KERNELS_H

#include <cstdint>
#include <torch/torch.h>
#include <tuple>

constexpr uint8_t K_VEC_LEN = 128;
constexpr uint16_t K_VEC_PER_BLK = 64;
constexpr float K_ENCODE_PREALLOCATE_RATIO = 1;
constexpr auto K_ENCODE_PREALLOCATE_SIZE = (size_t) (K_VEC_LEN * K_ENCODE_PREALLOCATE_RATIO);
constexpr size_t K_DECODE_PREALLOCATE_SIZE = 128;

constexpr uint8_t V_VEC_LEN = 128;
constexpr uint16_t V_VEC_PER_BLK = 64;
constexpr float V_ENCODE_PREALLOCATE_RATIO = 1;
constexpr auto V_ENCODE_PREALLOCATE_SIZE = (size_t) (V_VEC_LEN * V_ENCODE_PREALLOCATE_RATIO) + 128;
// constexpr float V_DECODE_PREALLOCATE_RATIO = 0.6;
constexpr size_t V_DECODE_PREALLOCATE_SIZE = 128;

struct block_info {
    uint32_t offset;
    uint16_t byte_num;
};

struct thread_info {
    uint16_t offset;
};

union idx_offset
{
    struct
    {
        uint32_t block_offset_idx;
        uint32_t offset;
    };
    uint64_t packed;
};

struct buffer_divisions {
    uint8_t *out;
    uint8_t *block_infos;
    uint8_t *thread_infos;
    uint8_t *idx_offset;

    size_t out_size;
    size_t block_infos_size;
    size_t thread_infos_size;
    size_t idx_offset_size;
};

std::tuple<size_t, size_t, size_t, size_t> calculate_buffer_size(const torch::Tensor &in, bool is_k, size_t head_num, size_t head_dim);

// out, block_infos, thread_infos
std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> k_entropy_encode_cuda_export(
        const torch::Tensor &in,
        const torch::Tensor &encode_code_book,
        torch::Tensor &block_infos,
        torch::Tensor &thread_infos,
        torch::Tensor &idx_offset_,
        torch::Tensor &encoded_data,
        const size_t base_global_offset,
        size_t head_num,
        size_t head_dim
);

float k_entropy_decode_cuda_export(
        const torch::Tensor &in,
        const torch::Tensor &block_infos,
        const torch::Tensor &thread_infos,
        const torch::Tensor &decode_code_book,
        const torch::Tensor &out,
        size_t head_num,
        size_t head_dim
);

float v_entropy_decode_cuda_export(
        const torch::Tensor &in,
        const torch::Tensor &block_infos,
        const torch::Tensor &thread_infos,
        const torch::Tensor &decode_code_book,
        const torch::Tensor &out,
        size_t head_num,
        size_t head_dim
);

float k_decode_and_mat_vec_mul_cuda_export(
        const torch::Tensor &decode_code_book,
        const torch::Tensor &block_infos,
        const torch::Tensor &thread_infos,
        const torch::Tensor &encoded_data,
        const torch::Tensor &quant_min_ints,
        const torch::Tensor &quant_scales,
        const torch::Tensor &B,
        const torch::Tensor &C,
        size_t head_num,
        size_t head_dim
);

float v_decode_and_mat_vec_mul_cuda_export(
        const torch::Tensor &decode_code_book,
        const torch::Tensor &block_infos,
        const torch::Tensor &thread_infos,
        const torch::Tensor &encoded_data,
        const torch::Tensor &quant_min_ints,
        const torch::Tensor &quant_scales,
        const torch::Tensor &B,
        const torch::Tensor &C,
        size_t head_num,
        size_t head_dim
);

float k_mat_vec_mul_cuda_export(
        const torch::Tensor &A,
        const torch::Tensor &B,
        const torch::Tensor &C
);

std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> v_entropy_encode_cuda_export(
        const torch::Tensor &in,
        const torch::Tensor &encode_code_book,
        torch::Tensor &block_infos,
        torch::Tensor &thread_infos,
        torch::Tensor &idx_offset_,
        torch::Tensor &encoded_data,
        const size_t base_global_offset,
        size_t head_num,
        size_t head_dim
);

#endif //TCOM_KERNELS_H
