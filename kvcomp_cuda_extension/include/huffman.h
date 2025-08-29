//
// Created by tut44803 on 2/26/25.
//

#ifndef TCOM_HUFFMAN_H
#define TCOM_HUFFMAN_H

#include <torch/torch.h>
#include <string>
#include <vector>
#include <map>

struct huffman_encode_code {
    uint32_t code;    // The bit pattern
    uint32_t length;   // Number of bits in the code
};

struct huffman_decode_code
{
    bool is_code = false;
    int8_t idx_or_value[2] = {-1, -1};
};

using encode_code_book_t = std::vector<huffman_encode_code>;
using decode_code_book_t = std::vector<huffman_decode_code>;
std::tuple<uint8_t, torch::Tensor, torch::Tensor> build_codebook(const std::vector<torch::Tensor>& input_tensors);

decode_code_book_t build_decode_code_book(const std::map<uint8_t, huffman_encode_code>& temp_code_book, uint8_t min,uint8_t max);
void print_encode_code_book(encode_code_book_t encode_code_book);
void print_encode_code_book(void *encode_code_book, size_t size);
void print_decode_code_book(decode_code_book_t decode_code_book);
void print_decode_code_book(void *decode_code_book, size_t size);
encode_code_book_t build_encode_code_book(const std::map<uint8_t, huffman_encode_code>& temp_code_book,uint8_t min,uint8_t max);
std::map<uint8_t, std::string> gen_huffman_codes(const torch::Tensor& counts);

#endif //TCOM_HUFFMAN_H
