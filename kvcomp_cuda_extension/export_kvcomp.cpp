//
// Created by tut44803 on 2/28/25.
//
#include <torch/extension.h>
#include <kernels.h>
#include <huffman.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.attr("K_VEC_LEN") = K_VEC_LEN;
    m.attr("K_VEC_PER_BLK") = K_VEC_PER_BLK;
    m.attr("V_VEC_LEN") = V_VEC_LEN;
    m.attr("V_VEC_PER_BLK") = V_VEC_PER_BLK;

    m.def("calculate_buffer_size", &calculate_buffer_size, "Calculate buffer size");
    m.def("build_codebook", &build_codebook, "Build Huffman codebook");
    m.def("k_entropy_encode_cuda_export", &k_entropy_encode_cuda_export, "Entropy encode using CUDA");
    m.def("k_entropy_decode_cuda_export", &k_entropy_decode_cuda_export, "Entropy decode using CUDA");
    m.def("k_decode_and_mat_vec_mul_cuda_export", &k_decode_and_mat_vec_mul_cuda_export,
          "Decode and perform matrix-vector multiplication using CUDA");
    m.def("k_mat_vec_mul_cuda_export", &k_mat_vec_mul_cuda_export, "our K Matrix-vector multiplication using CUDA");
    m.def("v_entropy_encode_cuda_export", &v_entropy_encode_cuda_export, "Entropy encode using CUDA");
    m.def("v_entropy_decode_cuda_export", &v_entropy_decode_cuda_export, "Entropy decode using CUDA");
    m.def("v_decode_and_mat_vec_mul_cuda_export", &v_decode_and_mat_vec_mul_cuda_export,
    "Decode and perform matrix-vector multiplication using CUDA");

}