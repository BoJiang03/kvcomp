//
// Created by tut44803 on 2/26/25.
//

#include <huffman.h>
#include <torch/torch.h>
#include <utils.h>

struct HuffmanNode {
    int value;
    int frequency;
    HuffmanNode* left;
    HuffmanNode* right;

    HuffmanNode(int val, int freq) : value(val), frequency(freq), left(nullptr), right(nullptr) {}

    bool operator<(const HuffmanNode& other) const {
        return frequency > other.frequency; // Min-heap
    }
};

struct CompareNodes {
    bool operator()(HuffmanNode* a, HuffmanNode* b) const {
        return a->frequency > b->frequency;
    }
};
std::string get_code_string(const huffman_encode_code &code) {
    std::string code_str;
    for (int i = 0; i < code.length; i++) {
        code_str += ((code.code >> (code.length - i - 1)) & 1) ? '1' : '0';
    }
    return code_str;
}
std::map<uint8_t, std::string> gen_huffman_codes(const torch::Tensor& counts) {
    tcom_assert(counts.numel() == 256, "Counts tensor must have 256 elements.");

    std::priority_queue<HuffmanNode*, std::vector<HuffmanNode*>, CompareNodes> pq;
    std::map<uint8_t, std::string> codes;

    // Create leaf nodes for non-zero frequencies
    for (int i = 0; i < counts.size(0); i++) {
        int freq = counts[i].item<int>();
        if (freq > 0) {
            pq.push(new HuffmanNode(i, freq));
        }
    }

    // Build Huffman tree
    while (pq.size() > 1) {
        auto left = pq.top(); pq.pop();
        auto right = pq.top(); pq.pop();

        auto parent = new HuffmanNode(-1, left->frequency + right->frequency);
        parent->left = left;
        parent->right = right;
        pq.push(parent);
    }

    // Generate codes and store root for cleanup
    HuffmanNode* root = pq.empty() ? nullptr : pq.top();
    if (root) {
        std::vector<std::pair<HuffmanNode*, std::string>> stack;
        stack.emplace_back(root, "");

        while (!stack.empty()) {
            auto [node, code] = stack.back();
            stack.pop_back();

            if (!node->left && !node->right) {
                codes[node->value] = code;
            } else {
                if (node->right) {
                    stack.emplace_back(node->right, code + "1");
                }
                if (node->left) {
                    stack.emplace_back(node->left, code + "0");
                }
            }
        }
    }

    // Cleanup: Delete the entire tree using iterative approach
    std::stack<HuffmanNode*> deleteStack;
    if (root) deleteStack.push(root);

    while (!deleteStack.empty()) {
        HuffmanNode* current = deleteStack.top();
        deleteStack.pop();

        if (current->right) deleteStack.push(current->right);
        if (current->left) deleteStack.push(current->left);
        delete current;
    }

    return codes;
}


std::tuple<uint8_t, torch::Tensor, torch::Tensor> build_codebook(const std::vector<torch::Tensor>& input_tensors) {
    auto device = input_tensors[0].device();
    for (const auto& tensor : input_tensors) {
        tcom_assert(tensor.device() == device, "All input tensors must be on the same device.");
        tcom_assert(tensor.dtype() == torch::kUInt8, "Input tensors must be of type uint8.");
    }
    auto counts = torch::zeros({256}, torch::TensorOptions().dtype(torch::kInt64).device(device));
    for (const auto& tensor : input_tensors) {
        auto tensor_counts = torch::bincount(tensor.flatten(), torch::nullopt, 256);
        counts += tensor_counts;
    }
    auto codebook = gen_huffman_codes(counts);
    uint8_t min_value = 255;
    uint8_t max_value = 0;
    std::map<uint8_t, huffman_encode_code> temp_code_book;
    for (const auto &pair : codebook) {
        auto value = (uint8_t)pair.first;
        uint32_t code = 0;
        uint8_t length = pair.second.size();

        for (char bit : pair.second) {
            code = (code << 1) | (bit - '0');
        }
        temp_code_book[value] = huffman_encode_code{code, length};
        min_value = std::min(min_value, value);
        max_value = std::max(max_value, value);
    }

    auto shift = min_value;
//    std::cout << "min value: " << (int)min_value << ", max value: " << (int)max_value << std::endl;

    // Build encode and decode codebooks
    auto encode = build_encode_code_book(temp_code_book, min_value, max_value);
    auto decode = build_decode_code_book(temp_code_book, min_value, max_value);
//    print_encode_code_book(encode);
//    std::cout << "out build function stl encode codebook: ";
//    for (int i = 0; i < 20; i++) {
//        std::cout << (int)((uint8_t*)encode.data())[i] << " ";
//    }
//    std::cout << std::endl;
    //print_decode_code_book(decode);
    torch::Tensor encode_tensor = torch::from_blob(
            encode.data(),
            {static_cast<int64_t>(encode.size() * sizeof(huffman_encode_code))},
            torch::TensorOptions().dtype(torch::kUInt8)
    ).clone();
//    std::cout << "cpu tensor encode codebook: ";
//    for (int i = 0; i < 20; i++) {
//        std::cout << (int)((uint8_t*)encode_tensor.data_ptr())[i] << " ";
//    }
//    std::cout << std::endl;

    torch::Tensor decode_tensor = torch::from_blob(
            decode.data(),
            {static_cast<int64_t>(decode.size() * sizeof(huffman_decode_code))},
            torch::TensorOptions().dtype(torch::kUInt8)
    ).clone();
    return std::make_tuple(shift, encode_tensor, decode_tensor);
}


std::string to_string(const huffman_decode_code& code) {
    std::string result = "huffman_decode_code{is_code=" + std::string(code.is_code ? "true" : "false");
    result += ", idx_or_value=[" + std::to_string(code.idx_or_value[0]) + ", " + std::to_string(code.idx_or_value[1]) + "]}";
    return result;
}

encode_code_book_t build_encode_code_book(const std::map<uint8_t, huffman_encode_code>& temp_code_book,const uint8_t min,const uint8_t max) {
    uint8_t offset = min;
    encode_code_book_t encode_code_book(max - min + 1); // Expanded size to accommodate offset
//    for (const auto &pair : temp_code_book) {
//        std::cout << "value: " << (int)pair.first << ", code: " << get_code_string(pair.second) << std::endl;
//    }
//    std::cout << "in fill codebook loop: " << std::endl;
    // Second pass: fill codebook
    for (const auto &entry : temp_code_book) {
        int adjusted_index = entry.first - offset;
        if (adjusted_index >= 0 && adjusted_index < encode_code_book.size()) {
            encode_code_book[adjusted_index] = entry.second;
//            std::cout << "adjusted_index: " << (int)adjusted_index << ", code: " << get_code_string(entry.second) << std::endl;
        }
    }
//    std::cout << "in build function stl encode codebook: ";
//    for (int i = 0; i < 20; i++) {
//        std::cout << (int)((uint8_t*)encode_code_book.data())[i] << " ";
//    }
//    std::cout << std::endl;
    return std::move(encode_code_book);
}

void print_encode_code_book(encode_code_book_t encode_code_book) {
    std::cout << "==============" << std::endl;
    std::cout << "Encode code book: " << std::endl;
    for (size_t idx = 0; idx < encode_code_book.size(); idx++) {
        std::cout << idx << ": " << get_code_string(encode_code_book[idx]) << std::endl;
    }
    std::cout << "==============" << std::endl;
}

void print_encode_code_book(void *encode_code_book, size_t size) {
    size_t code_num = size / sizeof(huffman_encode_code);
    auto *encode_code_book_ptr = static_cast<huffman_encode_code *>(encode_code_book);
    std::cout << "==============" << std::endl;
    std::cout << "Encode code book: " << std::endl;
    for (size_t idx = 0; idx < code_num; idx++) {
        std::cout << idx << ": " << get_code_string(encode_code_book_ptr[idx]) << std::endl;
    }
    std::cout << "==============" << std::endl;
}

void print_decode_code_book(decode_code_book_t decode_code_book) {
    std::cout << "==============" << std::endl;
    std::cout << "Decode code book: " << std::endl;
    for (size_t idx = 0; idx < decode_code_book.size(); idx++) {
        std::cout << idx << ": " << to_string(decode_code_book[idx]) << std::endl;
    }
    std::cout << "==============" << std::endl;
}

void print_decode_code_book(void *decode_code_book, size_t size) {
    size_t code_num = size / sizeof(huffman_decode_code);
    auto *decode_code_book_ptr = static_cast<huffman_decode_code *>(decode_code_book);
    std::cout << "==============" << std::endl;
    std::cout << "Decode code book: " << std::endl;
    for (size_t idx = 0; idx < code_num; idx++) {
        std::cout << idx << ": " << to_string(reinterpret_cast<huffman_decode_code*>(decode_code_book)[idx]) << std::endl;
    }
    std::cout << "==============" << std::endl;
}

decode_code_book_t build_decode_code_book(const std::map<uint8_t, huffman_encode_code>& temp_code_book,const uint8_t min,const uint8_t max) {
    decode_code_book_t decode_code_book;
    decode_code_book.emplace_back(huffman_decode_code{});
    decode_code_book.emplace_back(huffman_decode_code{});
    for (const auto &entry : temp_code_book) {
        int16_t idx = -1;
        int16_t last_idx = -1;
        std::string code_str = get_code_string(entry.second);
        for (char bit_char : code_str) {
            uint8_t bit = bit_char - '0';
            idx = idx == -1 ? bit : decode_code_book[idx].idx_or_value[bit];
            if (idx == -1) {
                decode_code_book.emplace_back(huffman_decode_code{});
                decode_code_book[last_idx].idx_or_value[bit] = decode_code_book.size() - 1;
                idx = decode_code_book.size() - 1;
            }
            last_idx = idx;
        }
        decode_code_book[idx].is_code = true;
        decode_code_book[idx].idx_or_value[0] = entry.first;
        decode_code_book[idx].idx_or_value[1] = entry.first;
    }
    return std::move(decode_code_book);
}