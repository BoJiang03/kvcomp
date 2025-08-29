//
// Created by tut44803 on 2/27/25.
//

#include <cuda_runtime.h>
#include <huffman.h>
#include <utils.h>
#include <kernels.h>
#include <cub/block/block_scan.cuh>

__device__ char* print_array_ints(const uint8_t *array, size_t size) {
    char *str = (char *) malloc(size * 4 + 1);  // 3 digits + space per number + null
    int pos = 0;

    for (size_t i = 0; i < size; i++) {
        uint8_t num = array[i];
        if (num >= 100) {
            str[pos++] = '0' + (num / 100);
            str[pos++] = '0' + ((num / 10) % 10);
            str[pos++] = '0' + (num % 10);
        } else if (num >= 10) {
            str[pos++] = '0' + (num / 10);
            str[pos++] = '0' + (num % 10);
        } else {
            str[pos++] = '0' + num;
        }
        str[pos++] = ' ';
    }
    str[pos] = '\0';

    return str;
}

__device__ char* print_array_bits(const uint8_t *array, size_t size) {
    char *str = (char *) malloc(size * 9 + 1);  // 8 bits + space per number + null
    int pos = 0;

    for (size_t i = 0; i < size; i++) {
        uint8_t num = array[i];
        for (int bit = 7; bit >= 0; bit--) {
            str[pos++] = ((num >> bit) & 1) ? '1' : '0';
        }
        str[pos++] = ' ';
    }
    str[pos] = '\0';

    return str;
}

__device__ void print_value_and_bits(uint8_t value, uint32_t code, uint8_t code_len) {
    char *str = (char *) malloc(code_len + 1);
    int pos = 0;
    for (int bit = code_len - 1; bit >= 0; bit--) {
        str[pos++] = ((code >> bit) & 1) ? '1' : '0';
    }
    str[pos] = '\0';
    printf("Value: %d, Code: %s\n", value, str);
    free(str);
}

typedef cub::BlockScan<uint16_t, K_VEC_PER_BLK> BlockScan;
__forceinline__ __device__ uint16_t inclusive_scan_one_warp(uint16_t value, uint16_t in_block_idx) {
    __shared__ BlockScan::TempStorage temp_storage;
    uint16_t inclusive_sum = 0;
    BlockScan(temp_storage).InclusiveSum(value, inclusive_sum);
    return inclusive_sum;
}

typedef cub::BlockScan<uint16_t, V_VEC_PER_BLK> v_BlockScan;
__forceinline__ __device__ uint16_t v_block_inclusive_scan(uint16_t value, uint16_t in_block_idx) {
    __shared__ v_BlockScan::TempStorage temp_storage;
    uint16_t inclusive_sum = 0;
    v_BlockScan(temp_storage).InclusiveSum(value, inclusive_sum);
    return inclusive_sum;
}

__forceinline__ __device__ void load_to_shared_mem(
        uint8_t *shared_mem,
        const uint8_t *global, size_t size,
        size_t thread_idx, size_t thread_num
) {
    for (; thread_idx < size; thread_idx += thread_num) {
        shared_mem[thread_idx] = global[thread_idx];
    }
}

__forceinline__ __device__ void load_to_shared_mem_half(
        __half *shared_mem,
        const __half *global, size_t size,
        size_t thread_idx, size_t thread_num
) {
    for (; thread_idx < size; thread_idx += thread_num) {
        shared_mem[thread_idx] = global[thread_idx];
    }
}

__forceinline__ __device__ void load_to_shared_mem_int8(
        int8_t *shared_mem,
        const int8_t *global, size_t size,
        size_t thread_idx, size_t thread_num
) {
    for (; thread_idx < size; thread_idx += thread_num) {
        shared_mem[thread_idx] = global[thread_idx];
    }
}

template <size_t V_LEN, size_t V_NUM>
__forceinline__ __device__ void load_2d_block_h(
        uint8_t *shared_mem,
        const uint8_t *global,
        size_t thread_idx,
        size_t stride
) {
    constexpr size_t ROUND_PER_VEC = V_LEN / V_NUM;
    for (size_t i = 0; i < V_NUM; i++) {
        #pragma unroll
        for (size_t j = 0; j < ROUND_PER_VEC; j++) {
            shared_mem[i * V_LEN + j * V_NUM + thread_idx] = global[i * stride + j * V_NUM + thread_idx];
        }
    }
    // printf("thread_idx: %d, loading vec i: %d, from global: %p to shared: %p\n", 
    //     (int)thread_idx, (int)i, global + i * stride, shared_mem + i * V_LEN);
    // reinterpret_cast<uint32_t *>(shared_mem)[i * V_LEN / 4] = reinterpret_cast<const uint32_t *>(global)[i * stride / 4];
}

template <size_t V_LEN, size_t V_NUM>
__forceinline__ __device__ void store_2d_block_h(
        const uint8_t *shared_mem,
        uint8_t *global,
        size_t thread_idx,
        size_t stride
) {

    constexpr size_t ROUND_PER_VEC = V_LEN / V_NUM;
    for (size_t i = 0; i < V_NUM; i++) {
        #pragma unroll
        for (size_t j = 0; j < ROUND_PER_VEC; j++) {
            global[i * stride + j * V_NUM + thread_idx] = shared_mem[i * V_LEN + j * V_NUM + thread_idx];
        }
    }
}

template <size_t V_LEN, size_t V_NUM>
__forceinline__ __device__ void load_2d_block_v(
        uint8_t *shared_mem,
        const uint8_t *global,
        size_t thread_idx,
        size_t stride
) {
    shared_mem += thread_idx * V_LEN;
    global += thread_idx;
    for (size_t i = 0; i < V_LEN; i++) {
        shared_mem[i] = global[i * stride];
    }
}

template <size_t V_LEN, size_t V_NUM>
__forceinline__ __device__ void store_2d_block_v(
        const uint8_t *shared_mem,
        uint8_t *global,
        size_t thread_idx,
        size_t stride
) {
    shared_mem += thread_idx * V_LEN;
    global += thread_idx;
    for (size_t i = 0; i < V_LEN; i++) {
        global[i * stride] = shared_mem[i];
    }
}

__forceinline__ __device__ void store_to_global_mem(
        uint8_t *global_mem,
        const uint8_t *shared_mem, size_t size,
        size_t thread_idx, size_t thread_num
) {
    for (; thread_idx < size; thread_idx += thread_num) {
        global_mem[thread_idx] = shared_mem[thread_idx];
    }
}

__forceinline__ __device__ uint16_t entropy_encode(
        const huffman_encode_code *encode_code_book,
        const uint8_t *input,
        uint8_t *encode_temp_buffer
) {
    uint32_t code;
    uint32_t code_len;
    uint32_t word;
    uint64_t bit_buffer = 0;
    uint16_t byte_count = 0;
    uint8_t bits_in_buffer = 0;
    uint16_t bit_count = 0;
    const uint16_t block_idx = blockIdx.x;
    const uint16_t vec_idx = threadIdx.x;
    for (uint16_t i = 0; i < K_VEC_LEN; i++) {
//         printf("encoding: %d\n", input[i]);
        auto &it = encode_code_book[input[i]];
        code = it.code;
        code_len = it.length;
//         if (vec_idx == 32 && block_idx == 0) {
//             print_value_and_bits(input[i], code, code_len);
//         }
        bit_buffer = (bit_buffer << code_len) | code;
        bit_count += code_len;
        bits_in_buffer += code_len;
        if (bits_in_buffer >= 32) {
            word = bit_buffer >> (bits_in_buffer - 32);
            word = ((word & 0xFF000000) >> 24) |
                   ((word & 0x00FF0000) >> 8) |
                   ((word & 0x0000FF00) << 8) |
                   ((word & 0x000000FF) << 24);
            *reinterpret_cast<uint32_t *>(encode_temp_buffer + byte_count) = word;
            byte_count += 4;
            bits_in_buffer -= 32;
            bit_buffer &= (1ULL << bits_in_buffer) - 1;
        }
    }

    while (bits_in_buffer > 0) {
        uint8_t byte;
        if (bits_in_buffer >= 8) {
            byte = (bit_buffer >> (bits_in_buffer - 8)) & 0xFF;
            bits_in_buffer -= 8;
        } else {
            byte = (bit_buffer << (8 - bits_in_buffer)) & 0xFF;
            bits_in_buffer = 0;
        }
        encode_temp_buffer[byte_count++] = byte;
        bit_buffer &= ((1ULL << bits_in_buffer) - 1);
    }
//    if (bit_count / 8 > 128) {
//        printf("buffer overflow, block_idx: %hu, vec_idx: %hu, bit_count: %hu, byte_count: %hu\n", block_idx, vec_idx, bit_count, byte_count);
//    }
    return bit_count;
}

__forceinline__ __device__ void entropy_decode(
    const huffman_decode_code *decode_code_book,
    const uint8_t *input,
    uint8_t *decode_temp_buffer,
    uint16_t bit_count
) {
    int16_t idx = -1;
    uint16_t write_offset = 0;
    uint8_t buffer = 0;
    uint8_t byte_num = bit_count / 8;
    uint8_t bit_remain = bit_count % 8;
    const uint16_t block_idx = blockIdx.x;
    const uint16_t vec_idx = threadIdx.x;
    for (uint16_t i = 0; i < byte_num; i++) {
        buffer = input[i];
        #pragma unroll
        for (uint8_t j = 0; j < 8; j++) {
            uint8_t bit = (buffer >> (7 - j)) & 1;
            idx = (int16_t) ((bit * (idx == -1)) + (decode_code_book[idx].idx_or_value[bit] * (idx != -1)));
            auto &code = decode_code_book[idx];
            if (code.is_code) {
                decode_temp_buffer[write_offset++] = code.idx_or_value[0];
//                 printf("block_idx: %hu, vec_idx: %hu, write_offset: %hu, decoded_data: %d\n", block_idx, vec_idx, write_offset, (int)code.idx_or_value[0]);

            }
            idx = (int16_t) ((-1 * code.is_code) + (idx * !code.is_code));
        }
    }

    if (bit_remain > 0) {
        buffer = input[byte_num];
        for (uint8_t j = 0; j < bit_remain; j++) {
            uint8_t bit = (buffer >> (7 - j)) & 1;
            idx = (int16_t) ((bit * (idx == -1)) + (decode_code_book[idx].idx_or_value[bit] * (idx != -1)));
            auto &code = decode_code_book[idx];
            if (code.is_code) {
                decode_temp_buffer[write_offset++] = code.idx_or_value[0];
//                 printf("block_idx: %hu, vec_idx: %hu, write_offset: %hu, decoded_data: %d\n", block_idx, vec_idx, write_offset, (int)code.idx_or_value[0]);

            }
            idx = (int16_t) ((-1 * code.is_code) + (idx * !code.is_code));
        }
    }

//     for (uint16_t i = 0; i < bit_count; i++) {
//         uint8_t bit = (input[decoded_bit / 8] >> (7 - (decoded_bit % 8))) & 1;
//         idx = (int16_t) ((bit * (idx == -1)) + (decode_code_book[idx].idx_or_value[bit] * (idx != -1)));
//         auto &code = decode_code_book[idx];
//         if (code.is_code) {
//         decode_temp_buffer[write_offset++] = code.idx_or_value[0];
//         }
//         idx = (int16_t) ((-1 * code.is_code) + (idx * !code.is_code));
//         decoded_bit++;
//     }
}

__forceinline__ __device__ __half entropy_decode_mat_vec_mul(
        const huffman_decode_code *decode_code_book,
        const uint8_t *input,
        const __half *B,
        int8_t min_int,
        __half quant_scale,
        uint16_t bit_count
) {
    __half result = __float2half(0.0f);  // Initialize with proper conversion
    int16_t idx = -1;
    uint16_t write_offset = 0;
    uint8_t buffer = 0;
    uint8_t byte_num = bit_count / 8;
    uint8_t bit_remain = bit_count % 8;
    const uint16_t block_idx = blockIdx.x;
    const uint16_t vec_idx = threadIdx.x;
    for (uint16_t i = 0; i < byte_num; i++) {
        buffer = input[i];
        #pragma unroll
        for (uint8_t j = 0; j < 8; j++) {
            uint8_t bit = (buffer >> (7 - j)) & 1;
//            idx = (int16_t) ((bit * (idx == -1)) + (decode_code_book[idx].idx_or_value[bit] * (idx != -1)));
            idx = idx== -1 ? bit : decode_code_book[idx].idx_or_value[bit];
            auto &code = decode_code_book[idx];
            if (code.is_code) {
                //if (block_idx == 0 && vec_idx == 0) {
                //    printf("block_idx: %hu, vec_idx: %hu, write_offset: %hu, decoded_data: %f, B: %f, dot_product: %f\n", block_idx, vec_idx, write_offset, __half2float(__hmul(__hmul(__int2half_rn(code.idx_or_value[0] + min_int), quant_scale), B[write_offset])), __half2float(B[write_offset]), __half2float(result));
                //}
                result = __hadd(result, __hmul(__hmul(__int2half_rn(code.idx_or_value[0] + min_int), quant_scale), B[write_offset++]));
                //result = __hadd(result, __hmul(__int2half_rn(code.idx_or_value[0] + min_int), quant_scale));
            }
//            idx = (int16_t) ((-1 * code.is_code) + (idx * !code.is_code));
            idx = code.is_code ? -1 : idx;
        }
    }

    if (bit_remain > 0) {
        buffer = input[byte_num];
        for (uint8_t j = 0; j < bit_remain; j++) {
            uint8_t bit = (buffer >> (7 - j)) & 1;
            idx = idx== -1 ? bit : decode_code_book[idx].idx_or_value[bit];
            auto &code = decode_code_book[idx];
            if (code.is_code) {
                //if (block_idx == 0 && vec_idx == 0) {
                //    printf("block_idx: %hu, vec_idx: %hu, write_offset: %hu, decoded_data: %f, B: %f, dot_product: %f\n", block_idx, vec_idx, write_offset, __half2float(__hmul(__hmul(__int2half_rn(code.idx_or_value[0] + min_int), quant_scale), B[write_offset])), __half2float(B[write_offset]), __half2float(result));
                //}
                result = __hadd(result, __hmul(__hmul(__int2half_rn(code.idx_or_value[0] + min_int), quant_scale), B[write_offset++]));
            }
            idx = code.is_code ? -1 : idx;
        }
    }
    //if (block_idx == 0 && vec_idx == 0) {
    //    printf("block_idx: %hu, vec_idx: %hu, dot_product: %f\n", block_idx, vec_idx, __half2float(result));
    //}
    return result;
}

__forceinline__ __device__ __half v_entropy_decode_mat_vec_mul(
        const huffman_decode_code *decode_code_book,
        const uint8_t *input,
        const __half *B,
        const int8_t *min_int,
        const __half *quant_scale,
        uint16_t bit_count
) {
    __half result = __float2half(0.0f);  // Initialize with proper conversion
    int16_t idx = -1;
    uint16_t write_offset = 0;
    uint8_t buffer = 0;
    uint8_t byte_num = bit_count / 8;
    uint8_t bit_remain = bit_count % 8;
    const uint16_t block_idx = blockIdx.x;
    const uint16_t vec_idx = threadIdx.x;
    for (uint16_t i = 0; i < byte_num; i++) {
        buffer = input[i];
        #pragma unroll
        for (uint8_t j = 0; j < 8; j++) {
            uint8_t bit = (buffer >> (7 - j)) & 1;
//            idx = (int16_t) ((bit * (idx == -1)) + (decode_code_book[idx].idx_or_value[bit] * (idx != -1)));
            idx = idx== -1 ? bit : decode_code_book[idx].idx_or_value[bit];
            auto &code = decode_code_book[idx];
            if (code.is_code) {
                //if (block_idx == 0 && vec_idx == 0) {
                //    printf("block_idx: %hu, vec_idx: %hu, write_offset: %hu, decoded_data: %f, B: %f, dot_product: %f\n", block_idx, vec_idx, write_offset, __half2float(__hmul(__hmul(__int2half_rn(code.idx_or_value[0] + min_int), quant_scale), B[write_offset])), __half2float(B[write_offset]), __half2float(result));
                //}
                result = __hadd(result, __hmul(__hmul(__int2half_rn(code.idx_or_value[0] + min_int[write_offset]), quant_scale[write_offset]), B[write_offset]));
                write_offset++;
                //result = __hadd(result, __hmul(__int2half_rn(code.idx_or_value[0] + min_int), quant_scale));
            }
//            idx = (int16_t) ((-1 * code.is_code) + (idx * !code.is_code));
            idx = code.is_code ? -1 : idx;
        }
    }

    if (bit_remain > 0) {
        buffer = input[byte_num];
        for (uint8_t j = 0; j < bit_remain; j++) {
            uint8_t bit = (buffer >> (7 - j)) & 1;
            idx = idx== -1 ? bit : decode_code_book[idx].idx_or_value[bit];
            auto &code = decode_code_book[idx];
            if (code.is_code) {
                //if (block_idx == 0 && vec_idx == 0) {
                //    printf("block_idx: %hu, vec_idx: %hu, write_offset: %hu, decoded_data: %f, B: %f, dot_product: %f\n", block_idx, vec_idx, write_offset, __half2float(__hmul(__hmul(__int2half_rn(code.idx_or_value[0] + min_int), quant_scale), B[write_offset])), __half2float(B[write_offset]), __half2float(result));
                //}
                result = __hadd(result, __hmul(__hmul(__int2half_rn(code.idx_or_value[0] + min_int[write_offset]), quant_scale[write_offset]), B[write_offset]));
                write_offset++;
            }
            idx = code.is_code ? -1 : idx;
        }
    }
    //if (block_idx == 0 && vec_idx == 0) {
    //    printf("block_idx: %hu, vec_idx: %hu, dot_product: %f\n", block_idx, vec_idx, __half2float(result));
    //}
    return result;
}

__forceinline__ __device__ void copy_array(
        const uint8_t *from,
        uint8_t *to,
        uint8_t size
) {
    for (uint8_t i = 0; i < size; i++) {
        to[i] = from[i];
    }
}

__global__ void huffman_encode(
        const huffman_encode_code *__restrict__ encode_code_book,
        const uint8_t *__restrict__ input,
        block_info *__restrict__ block_offsets,
        thread_info *__restrict__ thread_offsets,
        uint8_t *__restrict__ encoded_data,
        idx_offset *block_offset_idx_and_offset,
        const uint32_t base_global_offset,
        const size_t stride
) {
    const uint16_t block_idx = blockIdx.x;
    const uint16_t vec_idx = threadIdx.x;
    const uint16_t block_layer_idx = block_idx / (stride / K_VEC_LEN);
    const uint16_t block_in_layer_idx = block_idx % (stride / K_VEC_LEN);
    auto &block_offset = block_offsets[block_idx];
    auto &thread_offset = thread_offsets[block_idx * K_VEC_PER_BLK + vec_idx];

    __shared__ uint32_t encode_temp_buffer[K_ENCODE_PREALLOCATE_SIZE * K_VEC_PER_BLK / 4];
    __shared__ uint8_t aggregate_buffer[K_VEC_LEN * K_VEC_PER_BLK];

    load_2d_block_h<K_VEC_LEN, K_VEC_PER_BLK>(
            aggregate_buffer,
            input + K_VEC_PER_BLK * block_layer_idx * stride + K_VEC_LEN * block_in_layer_idx,
            vec_idx,
            stride
    );

    __syncthreads();

    uint16_t bit_count = entropy_encode(
            encode_code_book,
            aggregate_buffer + vec_idx * K_VEC_LEN,
            reinterpret_cast<uint8_t *>(encode_temp_buffer) + vec_idx * K_ENCODE_PREALLOCATE_SIZE
    );
    uint16_t byte_count = (bit_count + 7) / 8;
    // printf("after encode: block_idx: %d, vec_idx: %d, bit_count: %d, byte_count: %d\n",
    //     (int)block_idx, (int)vec_idx, (int)bit_count, (int)byte_count
    // );
    uint64_t accumulated_offset = inclusive_scan_one_warp(
            byte_count,
            vec_idx
    );

    uint16_t start_offset = vec_idx == 0 ? 0 : accumulated_offset - byte_count;
    thread_offset = {
            .offset=static_cast<uint16_t>((accumulated_offset << 3) + (byte_count << 3) - bit_count),
    };
    copy_array(
            reinterpret_cast<uint8_t *>(encode_temp_buffer) + vec_idx * K_ENCODE_PREALLOCATE_SIZE,
            aggregate_buffer + start_offset,
            byte_count
    );

    if (vec_idx == K_VEC_PER_BLK - 1) {
        idx_offset old{};
        old.packed = atomicAdd(reinterpret_cast<unsigned long long *>(block_offset_idx_and_offset), accumulated_offset << 32 | 1);
        block_offset = block_info{
                .offset = old.offset + base_global_offset,
                .byte_num = static_cast<uint16_t>(accumulated_offset),
        };
    }

    __syncthreads();

    store_to_global_mem(
            encoded_data + block_offset.offset - base_global_offset,
            aggregate_buffer,
            block_offset.byte_num,
            vec_idx, K_VEC_PER_BLK
    );
}

__global__ void v_huffman_encode(
        const huffman_encode_code *__restrict__ encode_code_book,
        const uint8_t *__restrict__ input,
        block_info *__restrict__ block_offsets,
        thread_info *__restrict__ thread_offsets,
        uint8_t *__restrict__ encoded_data,
        idx_offset *block_offset_idx_and_offset,
        const uint32_t base_global_offset,
        const size_t stride
) {
    const uint16_t block_idx = blockIdx.x;
    const uint16_t vec_idx = threadIdx.x;
    const uint16_t block_layer_idx = block_idx / (stride / V_VEC_PER_BLK);
    const uint16_t block_in_layer_idx = block_idx % (stride / V_VEC_PER_BLK);
    auto &block_offset = block_offsets[block_idx];
    auto &thread_offset = thread_offsets[block_idx * V_VEC_PER_BLK + vec_idx];
    __shared__ uint32_t encode_temp_buffer[V_ENCODE_PREALLOCATE_SIZE * V_VEC_PER_BLK / 4];
    __shared__ uint8_t aggregate_buffer[V_VEC_LEN * V_VEC_PER_BLK];

    // if (vec_idx == 0) {
    //     printf("block_idx: %hu, block_layer_idx: %hu, block_in_layer_idx: %hu\n", block_idx, block_layer_idx, block_in_layer_idx);
    // }
    
    load_2d_block_v<V_VEC_LEN, V_VEC_PER_BLK>(
            aggregate_buffer,
            input + V_VEC_LEN * block_layer_idx * stride + V_VEC_PER_BLK * block_in_layer_idx,
            vec_idx,
            stride
    );

    __syncthreads();

//    if (vec_idx == 22 && block_idx == 13) {
//        char * str = print_array_ints(aggregate_buffer + vec_idx * V_VEC_LEN, V_VEC_LEN);
//        printf("block_idx: %hu, vec_idx: %hu, to be encoded: %s\n", block_idx, vec_idx, str);
//    }

    uint16_t bit_count = entropy_encode(
            encode_code_book,
            aggregate_buffer + vec_idx * V_VEC_LEN,
            reinterpret_cast<uint8_t *>(encode_temp_buffer) + vec_idx * V_ENCODE_PREALLOCATE_SIZE
    );

    uint16_t byte_count = (bit_count + 7) / 8;
    //if (bit_count >= 128 * 8) {
    //    printf("overflow detected");
    //}
//    printf("bin count: %hu, byte count: %hu\n", bit_count, byte_count);
//    if (vec_idx == 22 && block_idx == 13) {
//        char * str = print_array_bits(reinterpret_cast<uint8_t *>(encode_temp_buffer) + vec_idx * K_ENCODE_PREALLOCATE_SIZE, byte_count);
//        printf("block_idx: %hu, vec_idx: %hu, encoded bits: %s\n", block_idx, vec_idx, str);
//    }

    // if (block_idx==0 && vec_idx==32) {
    //     char * str_bits = print_array_bits(reinterpret_cast<uint8_t *>(encode_temp_buffer) + vec_idx * V_ENCODE_PREALLOCATE_SIZE, byte_count);
    //     char * str_ints = print_array_ints(aggregate_buffer + vec_idx * V_VEC_LEN, V_VEC_LEN);
    //     printf("block_idx: %hu, vec_idx: %hu\nencoded ints: %s,\nencoded bits: %s\n", block_idx, vec_idx, str_ints, str_bits);
    //     free(str_bits);
    //     free(str_ints);
    // }

    uint64_t accumulated_offset = v_block_inclusive_scan(
            byte_count,
            vec_idx
    );

    // printf("v_huffman_encode: vec_idx: %hu, accumulated_offset: %llu\n",
        //    vec_idx,
        //    accumulated_offset
    // );

    uint16_t start_offset = vec_idx == 0 ? 0 : accumulated_offset - byte_count;
    thread_offset = {
            .offset=static_cast<uint16_t>((accumulated_offset << 3) + (byte_count << 3) - bit_count),
    };

    copy_array(
            reinterpret_cast<uint8_t *>(encode_temp_buffer) + vec_idx * V_ENCODE_PREALLOCATE_SIZE,
            aggregate_buffer + start_offset,
            byte_count
    );
//    if (vec_idx == 22 && block_idx == 13) {
//        char * str = print_array_bits(aggregate_buffer + start_offset, byte_count);
//        printf("block_idx: %hu, vec_idx: %hu, encoded bits: %s\n", block_idx, vec_idx, str);
//    }
    if (vec_idx == V_VEC_PER_BLK - 1) {
        idx_offset old{};
        old.packed = atomicAdd(reinterpret_cast<unsigned long long *>(block_offset_idx_and_offset), accumulated_offset << 32 | 1);
        block_offset = block_info{
                .offset = old.offset + base_global_offset,
                .byte_num = static_cast<uint16_t>(accumulated_offset),
        };
        // printf("block_idx: %hu, accumulated_offset: %hu\n",
        //        block_idx, block_offset.byte_num
        // );
    }

    __syncthreads();

    store_to_global_mem(
            encoded_data + block_offset.offset - base_global_offset,
            aggregate_buffer,
            block_offset.byte_num,
            vec_idx, V_VEC_PER_BLK
    );
}

#define BYTE_OFFSET(off) ((off) >> 3)
#define BIT_OFFSET(off) ((off) << 3)
#define MINUS_BITS(off) ((off) & 0x7)

__global__ void huffman_decode(
        const huffman_decode_code *__restrict__ decode_code_book,
        const block_info *__restrict__ block_offsets,
        const thread_info *__restrict__ thread_offsets,
        const uint8_t *__restrict__ input,
        uint8_t *__restrict__ output,
        const size_t head_num,
        const size_t stride
) {
    const uint16_t block_idx = blockIdx.x;
    const uint16_t vec_idx = threadIdx.x;
    const uint16_t block_layer_idx = block_idx / (stride / K_VEC_LEN);
    const uint16_t block_in_layer_idx = block_idx % (stride / K_VEC_LEN);
    __shared__ uint8_t aggregate_buffer[K_VEC_PER_BLK * K_DECODE_PREALLOCATE_SIZE];
    __shared__ uint8_t temp[K_VEC_PER_BLK * K_VEC_LEN];
    auto &block_offset = block_offsets[block_idx];
    auto &thread_offset = thread_offsets[block_idx * K_VEC_PER_BLK + vec_idx];

    uint16_t start_offset = vec_idx == 0 ? 0 : thread_offsets[block_idx * K_VEC_PER_BLK + vec_idx - 1].offset >> 3;

    load_to_shared_mem(
            aggregate_buffer,
            input + block_offset.offset,
            block_offset.byte_num,
            vec_idx, K_VEC_PER_BLK
    );
    
    __syncthreads();

    entropy_decode(
            decode_code_book,
            aggregate_buffer + start_offset,
            temp + vec_idx * K_VEC_LEN,
            BIT_OFFSET(BYTE_OFFSET(thread_offset.offset) - start_offset) - MINUS_BITS(thread_offset.offset)
    );
    
    __syncthreads();
    // printf("huffman_decode: after decode: block_idx: %d, vec_idx: %d, start_offset: %d, block byte_num: %d, thread byte_num: %d\n",
    //     (int)block_idx,
    //     (int)vec_idx,
    //     (int)start_offset,
    //     (int)block_offset.byte_num,
    //     (int)BYTE_OFFSET(thread_offset.offset) - start_offset
    // );
    store_2d_block_h<K_VEC_LEN, K_VEC_PER_BLK>(
            temp,
            output + K_VEC_PER_BLK * block_layer_idx * stride + K_VEC_LEN * block_in_layer_idx,
            vec_idx,
            stride
    );
}


__global__ void v_huffman_decode(
        const huffman_decode_code *__restrict__ decode_code_book,
        const block_info *__restrict__ block_offsets,
        const thread_info *__restrict__ thread_offsets,
        const uint8_t *__restrict__ input,
        uint8_t *__restrict__ output,
        const size_t head_num,
        const size_t stride
) {
    const uint16_t block_idx = blockIdx.x;
    const uint16_t vec_idx = threadIdx.x;
//     if (block_idx != 91) return;
//     if (block_idx == 91) return;
    const uint16_t block_layer_idx = block_idx / (stride / V_VEC_PER_BLK);
    const uint16_t block_in_layer_idx = block_idx % (stride / V_VEC_PER_BLK);
    __shared__ uint8_t aggregate_buffer[V_VEC_PER_BLK * V_DECODE_PREALLOCATE_SIZE];
    __shared__ uint8_t temp[V_VEC_PER_BLK * V_VEC_LEN];
    auto &block_offset = block_offsets[block_idx];
    auto &thread_offset = thread_offsets[block_idx * V_VEC_PER_BLK + vec_idx];
    uint16_t start_offset = vec_idx == 0 ? 0 : thread_offsets[block_idx * V_VEC_PER_BLK + vec_idx - 1].offset >> 3;

    load_to_shared_mem(
            aggregate_buffer,
            input + block_offset.offset,
            block_offset.byte_num,
            vec_idx, V_VEC_PER_BLK
    );

    __syncthreads();
//     if (vec_idx != 63) return;
    // if (block_idx==0 && vec_idx==32) {
    //     printf("v_huffman_decode: block_idx: %hu, block_offset.byte_num: %hu, start_offset: %hu\n",
    //         block_idx,
    //         block_offset.byte_num,
    //         start_offset
    //  );
        // char * str_bits = print_array_bits(aggregate_buffer + start_offset, BYTE_OFFSET(thread_offset.offset) - start_offset);
        // printf("in decode, block_idx: %hu, vec_idx: %hu\nto be decoded bits: %s\n", block_idx, vec_idx, str_bits);
        // free(str_bits);
        // //free(str_ints);        char * str_ints = print_array_ints(temp + vec_idx * V_VEC_LEN, V_VEC_LEN);
        // printf("after start_offset: %hu\n", start_offset);
    // }
    // __syncthreads();
//     printf("v_huffman_decode: block_idx: %hu, vec_idx: %hu, block_offset.byte_num: %hu, thread byte_num: %hu\n",
//         block_idx,
//         vec_idx,
//         block_offset.byte_num,
//         (int)(BYTE_OFFSET(thread_offset.offset) - start_offset)
//      );
    entropy_decode(
            decode_code_book,
            aggregate_buffer + start_offset,
            temp + vec_idx * V_VEC_LEN,
            BIT_OFFSET(BYTE_OFFSET(thread_offset.offset) - start_offset) - MINUS_BITS(thread_offset.offset)
    );

    __syncthreads();

    store_2d_block_v<V_VEC_LEN, V_VEC_PER_BLK>(
            temp,
            output + V_VEC_LEN * block_layer_idx * stride + V_VEC_PER_BLK * block_in_layer_idx,
            vec_idx,
            stride
    );
}

__global__ void huffman_decode_and_mat_vec_mul(
        const huffman_decode_code *__restrict__ decode_code_book,
        const block_info *__restrict__ block_offsets,
        const thread_info *__restrict__ thread_offsets,
        const uint8_t *__restrict__ input,
        const int8_t *__restrict__ quant_min_ints,
        const __half *__restrict__ quant_scales,
        const __half *__restrict__ B,
        half *__restrict__ C,
        const size_t head_dim,
        const size_t head_num,
        const size_t stride,
        const size_t ctx_len
) {
    const uint16_t block_idx = blockIdx.x;
    const uint16_t vec_idx = threadIdx.x;
    const uint16_t block_layer_idx = block_idx / (stride / K_VEC_LEN);
    const uint16_t block_in_layer_idx = block_idx % (stride / K_VEC_LEN);
    __shared__ uint8_t aggregate_buffer[K_VEC_PER_BLK * K_DECODE_PREALLOCATE_SIZE];
    __shared__ uint8_t decode_codebook[204];
    __shared__ __half B_shared[K_VEC_LEN];
    auto &block_offset = block_offsets[block_idx];
    auto &thread_offset = thread_offsets[block_idx * K_VEC_PER_BLK + vec_idx];

    uint16_t start_offset = vec_idx == 0 ? 0 : thread_offsets[block_idx * K_VEC_PER_BLK + vec_idx - 1].offset >> 3;

    load_to_shared_mem(
            aggregate_buffer,
            input + block_offset.offset,
            block_offset.byte_num,
            vec_idx, K_VEC_PER_BLK
    );
    load_to_shared_mem(
                decode_codebook,
                (const uint8_t *)decode_code_book,
                204,
                vec_idx, K_VEC_PER_BLK
        );
    load_to_shared_mem(
                (uint8_t *)B_shared,
                (const uint8_t *)(B + block_in_layer_idx * head_dim),
                K_VEC_LEN * sizeof(__half),
                vec_idx, K_VEC_PER_BLK
        );
    __syncthreads();

    __half vec_dot_product = entropy_decode_mat_vec_mul(
            (const huffman_decode_code *)decode_codebook,
            aggregate_buffer + start_offset,
            B_shared,
            quant_min_ints[block_layer_idx], // for now only one quant block
            quant_scales[block_layer_idx],
            BIT_OFFSET(BYTE_OFFSET(thread_offset.offset) - start_offset) - MINUS_BITS(thread_offset.offset)
    );
    C[block_in_layer_idx * ctx_len + block_layer_idx * K_VEC_PER_BLK + vec_idx] = vec_dot_product;
}

__global__ void v_huffman_decode_and_mat_vec_mul(
        const huffman_decode_code *__restrict__ decode_code_book,
        const block_info *__restrict__ block_offsets,
        const thread_info *__restrict__ thread_offsets,
        const uint8_t *__restrict__ input,
        const int8_t *__restrict__ quant_min_ints,
        const __half *__restrict__ quant_scales,
        const __half *__restrict__ B,
        half *__restrict__ C,
        const size_t head_dim,
        const size_t head_num,
        const size_t stride,
        const size_t ctx_len
) {
    const uint16_t block_idx = blockIdx.x;
    const uint16_t vec_idx = threadIdx.x;
    const uint16_t block_layer_idx = block_idx / (stride / V_VEC_PER_BLK);
    const uint16_t block_in_layer_idx = block_idx % (stride / V_VEC_PER_BLK);
    __shared__ uint8_t aggregate_buffer[V_VEC_PER_BLK * V_DECODE_PREALLOCATE_SIZE];
    __shared__ uint8_t decode_codebook[204];
    __shared__ __half B_shared[V_VEC_LEN];
    __shared__ int8_t s_quant_min_ints[V_VEC_LEN];
    __shared__ __half s_quant_scales[V_VEC_LEN];
    auto &block_offset = block_offsets[block_idx];
    auto &thread_offset = thread_offsets[block_idx * V_VEC_PER_BLK + vec_idx];
    uint16_t start_offset = vec_idx == 0 ? 0 : thread_offsets[block_idx * V_VEC_PER_BLK + vec_idx - 1].offset >> 3;
    
    // printf("in v_huffman_decode_and_mat_vec_mul: blcok_idx: %hu, vec_idx: %hu\n",
    //     block_idx,
    //     vec_idx
    // );

    load_to_shared_mem(
            aggregate_buffer,
            input + block_offset.offset,
            block_offset.byte_num,
            vec_idx, V_VEC_PER_BLK
    );
    load_to_shared_mem(
                decode_codebook,
                (const uint8_t *)decode_code_book,
                204,
                vec_idx, V_VEC_PER_BLK
        );
        load_to_shared_mem_int8(
                s_quant_min_ints,
                quant_min_ints + block_layer_idx * V_VEC_LEN,
                V_VEC_LEN,
                vec_idx, V_VEC_PER_BLK
        );
        load_to_shared_mem_half(
                s_quant_scales,
                quant_scales + block_layer_idx * V_VEC_LEN,
                V_VEC_LEN,
                vec_idx, V_VEC_PER_BLK
        );
    const size_t head_idx = V_VEC_PER_BLK * block_in_layer_idx / head_dim;
    const size_t attn_weights_start_idx = block_layer_idx * V_VEC_LEN;
    // if (vec_idx == 0) {
    //     printf("block_idx: %hu, block_layer_idx: %hu, block_in_layer_idx: %hu, head_idx: %lu, attn_weights_start_idx: %lu, B offset: %lu\n", block_idx, block_layer_idx, block_in_layer_idx, head_idx, attn_weights_start_idx, head_idx * ctx_len + attn_weights_start_idx);
    // }
    load_to_shared_mem(
                (uint8_t *)B_shared,
                (const uint8_t *)(B + head_idx * ctx_len + attn_weights_start_idx),
                V_VEC_LEN * sizeof(__half),
                vec_idx, V_VEC_PER_BLK
        );
    __syncthreads();
    //if (block_idx == 0 && vec_idx == 0) {
    //    printf("bit num: %hu\n", BIT_OFFSET(BYTE_OFFSET(thread_offset.offset) - start_offset) - MINUS_BITS(thread_offset.offset));
    //}
    __half vec_dot_product = v_entropy_decode_mat_vec_mul(
            (const huffman_decode_code *)decode_codebook,
            aggregate_buffer + start_offset,
            B_shared,
            s_quant_min_ints,
            s_quant_scales,
            BIT_OFFSET(BYTE_OFFSET(thread_offset.offset) - start_offset) - MINUS_BITS(thread_offset.offset)
    );
    //if (block_idx == 0 && vec_idx == 0) {
    //    printf("block_idx: %hu, vec_idx: %hu, vec_dot_product: %f\n", block_idx, vec_idx, __half2float(vec_dot_product));
    //}
    C[block_layer_idx * stride + block_in_layer_idx * V_VEC_PER_BLK + vec_idx] = vec_dot_product;
}

__host__ buffer_divisions calculate_buffer_offsets(const torch::Tensor &in, torch::Tensor &buffer, size_t head_num, size_t head_dim) {
    tcom_assert(head_dim == K_VEC_LEN, "head_dim must be equal to K_VEC_LEN");
    tcom_assert(in.numel() % head_dim == 0, "Buffer size must be divisible by head_dim");
    tcom_assert(buffer.dtype() == torch::kUInt8, "Buffer must be of type uint8_t");
    tcom_assert(buffer.is_contiguous(), "Buffer must be contiguous");

    size_t vec_num = in.numel() / head_dim;
    size_t block_num = (vec_num + K_VEC_PER_BLK - 1) / K_VEC_PER_BLK;
    auto pre_alloc_compressed_data_size = (size_t) (in.numel() * K_ENCODE_PREALLOCATE_RATIO);
    auto tensor_ptr = reinterpret_cast<uint8_t *>(buffer.data_ptr());
    buffer_divisions offsets;
    size_t offset = 0;
    offsets.block_infos = tensor_ptr + offset;
    offsets.block_infos_size = block_num * sizeof(block_info);
    offset += offsets.block_infos_size;
    offsets.thread_infos = tensor_ptr + offset;
    offsets.thread_infos_size = vec_num * sizeof(thread_info);
    offset += offsets.thread_infos_size;
    offsets.idx_offset = tensor_ptr + offset;
    offsets.idx_offset_size = sizeof(idx_offset);
    offset += offsets.idx_offset_size;
    offsets.out = tensor_ptr + offset;
    offsets.out_size = pre_alloc_compressed_data_size;
    offset += offsets.out_size;

    tcom_assert(offset <= buffer.numel(), "Buffer size is not enough");

    return offsets;
}

__host__ std::tuple<size_t, size_t, size_t, size_t> calculate_buffer_size(const torch::Tensor &in, bool is_k, size_t head_num, size_t head_dim) {
    size_t vec_num = 0;
    size_t block_num = 0;
    size_t pre_alloc_compressed_data_size = 0;

    if (is_k) {
        tcom_assert(in.size(0) * in.size(1) % K_VEC_PER_BLK == 0, "Context length must be divisible by K_VEC_PER_BLK");
        tcom_assert(in.size(2) * in.size(3) * in.size(4) % K_VEC_LEN == 0, "Hidden dim must be divisible by K_VEC_LEN");
        vec_num = in.numel() / K_VEC_LEN;
        block_num = in.numel() / K_VEC_LEN / K_VEC_PER_BLK;
        pre_alloc_compressed_data_size = (size_t) (in.numel() * K_ENCODE_PREALLOCATE_RATIO);
    } else {
        tcom_assert(in.size(0) * in.size(1) % V_VEC_LEN == 0, "Context length must be divisible by V_VEC_LEN");
        tcom_assert(in.size(2) * in.size(3) * in.size(4) % V_VEC_PER_BLK == 0, "Hidden dim must be divisible by V_VEC_PER_BLK");
        vec_num = in.numel() / V_VEC_LEN;
        block_num = in.numel() / V_VEC_LEN / V_VEC_PER_BLK;
        pre_alloc_compressed_data_size = (size_t) (in.numel() * V_ENCODE_PREALLOCATE_RATIO);
    }
    return std::make_tuple(
            block_num * sizeof(block_info),
            vec_num * sizeof(thread_info),
            sizeof(idx_offset),
            pre_alloc_compressed_data_size
    );
}

__host__ std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> k_entropy_encode_cuda_export(
        const torch::Tensor &in,
        const torch::Tensor &encode_code_book,
        torch::Tensor &block_infos,
        torch::Tensor &thread_infos,
        torch::Tensor &idx_offset_,
        torch::Tensor &encoded_data,
        const size_t base_global_offset,
        size_t head_num,
        size_t head_dim
) {
    cudaSetDevice(in.get_device());
    auto [block_infos_buffer_size, thread_infos_buffer_size, idx_offset_buffer_size, pre_alloc_compressed_data_size] = calculate_buffer_size(in, true, head_num, head_dim);
    tcom_assert(in.numel() % (K_VEC_LEN * K_VEC_PER_BLK) == 0, "Input size must be divisible by K_VEC_LEN * K_VEC_PER_BLK");
    idx_offset init_value = {0};
    cudaError_t error;

    error = cudaMemcpy(idx_offset_.mutable_data_ptr(), &init_value, sizeof(idx_offset), cudaMemcpyHostToDevice);
    tcom_assert(error == cudaSuccess, "Failed to copy init value to device");

    size_t stride = in.size(2) * in.size(3) * in.size(4);

    size_t block_num = block_infos_buffer_size / sizeof(block_info);
//     std::cout << "block_num: " << block_num << std::endl;
//     std::cout << "vec_num: " << thread_infos_buffer_size / sizeof(thread_info) << std::endl;
//     std::cout << "stride: " << stride << std::endl;
//     std::cout << "encode_code_book size: " << encode_code_book.numel() / sizeof(huffman_encode_code) << std::endl;
    huffman_encode<<<block_num, K_VEC_PER_BLK>>>(
            reinterpret_cast<const huffman_encode_code *>(encode_code_book.data_ptr()),
            reinterpret_cast<const uint8_t *>(in.data_ptr()),
            reinterpret_cast<block_info *>(block_infos.mutable_data_ptr()),
            reinterpret_cast<thread_info *>(thread_infos.mutable_data_ptr()),
            static_cast<uint8_t *>(encoded_data.mutable_data_ptr()),
            reinterpret_cast<idx_offset *>(idx_offset_.mutable_data_ptr()),
            base_global_offset,
            stride
    );
    cudaMemcpyAsync(&init_value, idx_offset_.mutable_data_ptr(), sizeof(idx_offset), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    error = cudaGetLastError();
    tcom_assert(error == cudaSuccess, std::string("Kernel launch failed: ") + cudaGetErrorString(error));

    return {
            encoded_data.slice(0, 0, init_value.offset),
            block_infos.slice(0, 0, block_infos_buffer_size),
            thread_infos.slice(0, 0, thread_infos_buffer_size)
    };
}

__host__ std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> v_entropy_encode_cuda_export(
        const torch::Tensor &in,
        const torch::Tensor &encode_code_book,
        torch::Tensor &block_infos,
        torch::Tensor &thread_infos,
        torch::Tensor &idx_offset_,
        torch::Tensor &encoded_data,
        const size_t base_global_offset,
        size_t head_num,
        size_t head_dim
) {
    cudaSetDevice(in.get_device());
    auto [block_infos_buffer_size, thread_infos_buffer_size, idx_offset_buffer_size, pre_alloc_compressed_data_size] = calculate_buffer_size(in,false, head_num, head_dim);
    idx_offset init_value = {0};
    cudaError_t error;

    error = cudaMemcpy(idx_offset_.mutable_data_ptr(), &init_value, sizeof(idx_offset), cudaMemcpyHostToDevice);
    tcom_assert(error == cudaSuccess, "Failed to copy init value to device");
    size_t stride = in.size(2) * in.size(3) * in.size(4);
    size_t block_num = block_infos_buffer_size / sizeof(block_info);
    // std::cout << "block_num: " << block_num << std::endl;
    // std::cout << "vec_num: " << thread_infos_buffer_size / sizeof(thread_info) << std::endl;
    // std::cout << "stride: " << stride << std::endl;
    v_huffman_encode<<<block_num, V_VEC_PER_BLK>>>(
            reinterpret_cast<const huffman_encode_code *>(encode_code_book.data_ptr()),
            reinterpret_cast<const uint8_t *>(in.data_ptr()),
            reinterpret_cast<block_info *>(block_infos.mutable_data_ptr()),
            reinterpret_cast<thread_info *>(thread_infos.mutable_data_ptr()),
            static_cast<uint8_t *>(encoded_data.mutable_data_ptr()),
            reinterpret_cast<idx_offset *>(idx_offset_.mutable_data_ptr()),
            base_global_offset,
            stride
    );
    cudaMemcpyAsync(&init_value, idx_offset_.mutable_data_ptr(), sizeof(idx_offset), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    error = cudaGetLastError();
    tcom_assert(error == cudaSuccess, std::string("Kernel launch failed: ") + cudaGetErrorString(error));

    return {
            encoded_data.slice(0, 0, init_value.offset),
            block_infos.slice(0, 0, block_infos_buffer_size),
            thread_infos.slice(0, 0, thread_infos_buffer_size)
    };
}

__host__ float k_entropy_decode_cuda_export(
        const torch::Tensor &in,
        const torch::Tensor &block_infos,
        const torch::Tensor &thread_infos,
        const torch::Tensor &decode_code_book,
        const torch::Tensor &out,
        size_t head_num,
        size_t head_dim
) {
    cudaSetDevice(in.get_device());
    size_t block_num = block_infos.numel() / sizeof(block_info);
    size_t thread_num = thread_infos.numel() / sizeof(thread_info);
    size_t stride = out.size(2) * out.size(3) * out.size(4);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, cudaStreamDefault);
    huffman_decode<<<block_num, K_VEC_PER_BLK>>>(
            reinterpret_cast<const huffman_decode_code *>(decode_code_book.data_ptr()),
            reinterpret_cast<const block_info *>(block_infos.data_ptr()),
            reinterpret_cast<const thread_info *>(thread_infos.data_ptr()),
            reinterpret_cast<const uint8_t *>(in.data_ptr()),
            reinterpret_cast<uint8_t *>(out.data_ptr()),
            head_num,
            stride
    );
    cudaEventRecord(stop, cudaStreamDefault);
    cudaEventSynchronize(stop);
    cudaDeviceSynchronize();
    auto error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(error));
    }
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    return elapsedTime;
}

__host__ float v_entropy_decode_cuda_export(
        const torch::Tensor &in,
        const torch::Tensor &block_infos,
        const torch::Tensor &thread_infos,
        const torch::Tensor &decode_code_book,
        const torch::Tensor &out,
        size_t head_num,
        size_t head_dim
) {
    cudaSetDevice(in.get_device());
    size_t block_num = block_infos.numel() / sizeof(block_info);
    size_t thread_num = thread_infos.numel() / sizeof(thread_info);
    size_t stride = out.size(2) * out.size(3) * out.size(4);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, cudaStreamDefault);
    v_huffman_decode<<<block_num, V_VEC_PER_BLK>>>(
            reinterpret_cast<const huffman_decode_code *>(decode_code_book.data_ptr()),
            reinterpret_cast<const block_info *>(block_infos.data_ptr()),
            reinterpret_cast<const thread_info *>(thread_infos.data_ptr()),
            reinterpret_cast<const uint8_t *>(in.data_ptr()),
            reinterpret_cast<uint8_t *>(out.data_ptr()),
            head_num,
            stride
    );
    cudaEventRecord(stop, cudaStreamDefault);
    cudaEventSynchronize(stop);
    cudaDeviceSynchronize();
    auto error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(error));
    }
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    return elapsedTime;
}

void check_available(const torch::Tensor &t) {
    t.max();
}

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
) {
    cudaSetDevice(encoded_data.get_device());
    size_t block_num = block_infos.numel() / sizeof(block_info);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, cudaStreamDefault);

    size_t ctx_len = C.size(2);
    huffman_decode_and_mat_vec_mul<<<block_num, K_VEC_PER_BLK>>>(
            reinterpret_cast<const huffman_decode_code *>(decode_code_book.data_ptr()),
            reinterpret_cast<const block_info *>(block_infos.data_ptr()),
            reinterpret_cast<const thread_info *>(thread_infos.data_ptr()),
            reinterpret_cast<const uint8_t *>(encoded_data.data_ptr()),
            reinterpret_cast<const int8_t *>(quant_min_ints.data_ptr()),
            reinterpret_cast<const __half *>(quant_scales.data_ptr()),
            reinterpret_cast<const __half *>(B.data_ptr()),
            reinterpret_cast<__half *>(C.mutable_data_ptr()),
            head_dim,
            head_num,
            head_dim * head_num,
            ctx_len
    );
    cudaEventRecord(stop, cudaStreamDefault);
    cudaEventSynchronize(stop);
    auto error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(error));
    }
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    return elapsedTime;
}

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
) {
    cudaSetDevice(encoded_data.get_device());
    size_t block_num = block_infos.numel() / sizeof(block_info);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, cudaStreamDefault);
    size_t ctx_len = B.size(2);
    v_huffman_decode_and_mat_vec_mul<<<block_num, V_VEC_PER_BLK>>>(
            reinterpret_cast<const huffman_decode_code *>(decode_code_book.data_ptr()),
            reinterpret_cast<const block_info *>(block_infos.data_ptr()),
            reinterpret_cast<const thread_info *>(thread_infos.data_ptr()),
            reinterpret_cast<const uint8_t *>(encoded_data.data_ptr()),
            reinterpret_cast<const int8_t *>(quant_min_ints.data_ptr()),
            reinterpret_cast<const __half *>(quant_scales.data_ptr()),
            reinterpret_cast<const __half *>(B.data_ptr()),
            reinterpret_cast<__half *>(C.mutable_data_ptr()),
            head_dim,
            head_num,
            head_dim * head_num,
            ctx_len
    );
    cudaEventRecord(stop, cudaStreamDefault);
    cudaEventSynchronize(stop);
    auto error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(error));
    }
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    return elapsedTime;
}

__forceinline__ __device__ void load_half_to_shared_memory(
        __half shared_mem[K_VEC_PER_BLK][K_VEC_LEN + 1],
        const __half *global, size_t size,
        size_t thread_idx, size_t thread_num
        ) {
    for (; thread_idx < size; thread_idx += thread_num) {
        shared_mem[thread_idx / K_VEC_LEN][thread_idx % K_VEC_LEN] = global[thread_idx];
    }
}

__global__ void k_mat_vec_mul(
    const __half *__restrict__ A,
    const __half *__restrict__ B,
    __half *__restrict__ C,
    const size_t head_num,
    const size_t head_dim
) {
    const uint16_t block_idx = blockIdx.x;
    const uint16_t vec_idx = threadIdx.x;
    __shared__ __half temp_buffer[K_VEC_PER_BLK][K_VEC_LEN + 1];

    load_half_to_shared_memory(
            temp_buffer,
            A + block_idx * K_VEC_PER_BLK * K_VEC_LEN,
            K_VEC_PER_BLK * K_VEC_LEN,
            vec_idx, K_VEC_PER_BLK
    );

    __syncthreads();

    __half result = __float2half(0.0f);

    #pragma unroll
    for (size_t i = 0; i < K_VEC_LEN; i++) {
        result = __hadd(result, __hmul(temp_buffer[vec_idx][i], B[(vec_idx % head_num) * head_dim + i]));
    }
    C[block_idx * K_VEC_PER_BLK + vec_idx] = result;
}

__host__ float k_mat_vec_mul_cuda_export(
    const torch::Tensor &A,
    const torch::Tensor &B,
    const torch::Tensor &C
) {
    const size_t ctx_len = A.size(0);
    const size_t batch_size = A.size(1);
    const size_t head_num = A.size(2);
    const size_t head_dim = A.size(3);
    tcom_assert(batch_size == 1, "Batch size must be 1 for now");
    tcom_assert(batch_size == B.size(0), "Batch size of A and B must be equal");
    tcom_assert(head_num == B.size(1), "Head num of A and B must be equal");
    tcom_assert(head_dim == B.size(2), "Head dim of A and B must be equal");
    tcom_assert(ctx_len == C.size(0), "Context length of A and C must be equal");
    tcom_assert(batch_size == C.size(1), "Batch size of A and C must be equal");
    tcom_assert(head_num == C.size(2), "Head num of A and C must be equal");

    tcom_assert(A.numel() % (K_VEC_LEN * K_VEC_PER_BLK) == 0, "Input size must be divisible by K_VEC_LEN * K_VEC_PER_BLK");

    tcom_assert(A.dtype() == torch::kHalf, "A must be of type half");
    tcom_assert(B.dtype() == torch::kHalf, "B must be of type half");
    tcom_assert(C.dtype() == torch::kHalf, "C must be of type half");
    tcom_assert(A.is_contiguous(), "A must be contiguous");
    tcom_assert(B.is_contiguous(), "B must be contiguous");
    tcom_assert(C.is_contiguous(), "C must be contiguous");

    cudaSetDevice(A.get_device());
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, cudaStreamDefault);

    const size_t vec_num = A.numel() / K_VEC_LEN;
    tcom_assert(vec_num % K_VEC_PER_BLK == 0, "Input vec_num size must be divisible by K_VEC_PER_BLK");
    const size_t block_num = vec_num / K_VEC_PER_BLK;
    k_mat_vec_mul<<<block_num, K_VEC_PER_BLK>>>(
        reinterpret_cast<const __half *>(A.data_ptr()),
        reinterpret_cast<const __half *>(B.data_ptr()),
        reinterpret_cast<__half *>(C.mutable_data_ptr()),
        head_num,
        head_dim
    );
    cudaEventRecord(stop, cudaStreamDefault);
    cudaEventSynchronize(stop);
    cudaDeviceSynchronize();
    auto error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(error));
    }
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    return elapsedTime;
}