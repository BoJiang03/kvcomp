//
// Created by tut44803 on 2/26/25.
//
#include <compute.h>
#include <utils.h>

torch::Tensor safe_cat(const torch::Tensor& t1, const torch::Tensor& t2, int64_t dim) {
    if (t1.numel() == 0) return t2;
    if (t2.numel() == 0) return t1;
    return torch::cat({t1, t2}, dim);
}

std::tuple<torch::Tensor, torch::Tensor> cut_tensor(
        const torch::Tensor& buffer,
        const torch::Tensor& new_tensor,
        int64_t block_size,
        int64_t recent_size,
        int64_t dim
) {
    auto combined = safe_cat(buffer, new_tensor, dim);
    auto len = combined.size(dim);
    auto res_num = len % block_size;
    auto to_compress_block_num = (len + block_size - res_num - recent_size) / block_size;

    torch::Tensor to_compress;
    torch::Tensor remaining_buffer;

    if (to_compress_block_num > 0) {
        auto slice_point = to_compress_block_num * block_size;
        to_compress = combined.slice(0, 0, slice_point);
        remaining_buffer = combined.slice(0, slice_point);
    } else {
        to_compress = torch::Tensor();
        remaining_buffer = combined;
    }

    return std::make_tuple(to_compress, remaining_buffer);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> quant(
        const torch::Tensor& tensor,
        const std::vector<size_t>& quant_dims,
        float quant_scale_rel
) {
    auto min_vals = tensor;
    auto max_vals = tensor;

    for (auto dim : quant_dims) {
        min_vals = std::get<0>(min_vals.min(dim, true));
        max_vals = std::get<0>(max_vals.max(dim, true));
    }

    auto quant_scale = (max_vals - min_vals) * quant_scale_rel;
    auto min_ints = (min_vals / quant_scale).round();
    auto quant_ints = (tensor / quant_scale).round();

    return std::make_tuple(quant_ints - min_ints, min_ints, quant_scale);
}

void check_error_bound(
        const torch::Tensor& tensor,
        const torch::Tensor& quant_ints,
        const torch::Tensor& min_ints,
        const torch::Tensor& quant_scale
        ) {
    auto quant_ints_float = (quant_ints + min_ints) * quant_scale;
    auto error = (tensor - quant_ints_float).abs();
    tcom_assert((error < quant_scale * 0.53).all().item<bool>(), "Quantization error exceeds bound");
}
