//
// Created by tut44803 on 2/26/25.
//

#ifndef TCOM_COMPUTE_H
#define TCOM_COMPUTE_H

#include <torch/torch.h>
#include <tuple>
#include <vector>

torch::Tensor safe_cat(const torch::Tensor& t1, const torch::Tensor& t2, int64_t dim);

std::tuple<torch::Tensor, torch::Tensor> cut_tensor(
        const torch::Tensor& buffer,
        const torch::Tensor& new_tensor,
        int64_t block_size,
        int64_t recent_size,
        int64_t dim = 2
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> quant(
        const torch::Tensor& tensor,
        const std::vector<size_t>& quant_dims,
        float quant_scale_rel
);

void check_error_bound(
        const torch::Tensor& tensor,
        const torch::Tensor& quant_ints,
        const torch::Tensor& min_ints,
        const torch::Tensor& quant_scale
);

#endif //TCOM_COMPUTE_H
