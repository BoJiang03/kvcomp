import os
import torch

from models.tcom_cache import KVComCacheConfigStatic
from utils.huffman import huffman_encode
import tqdm as tqdm

def load_caches_from_folder(cache_path: str):
    k_path = os.path.join(cache_path, "k")
    v_path = os.path.join(cache_path, "v")
    
    if not (os.path.exists(k_path) and os.path.exists(v_path)):
        raise ValueError(f"Cache directories not found at {cache_path}")
    
    # Get all files and sort them by layer index
    k_files = sorted(os.listdir(k_path), key=lambda x: int(x.split('_')[0]))
    v_files = sorted(os.listdir(v_path), key=lambda x: int(x.split('_')[0]))
    
    if len(k_files) != len(v_files):
        raise ValueError("Mismatch between number of K and V cache files")
    
    k_cache_list = []
    v_cache_list = []
    
    # Load each cache file
    for k_file, v_file in zip(k_files, v_files):
        k_tensor = torch.load(os.path.join(k_path, k_file))
        v_tensor = torch.load(os.path.join(v_path, v_file))
        k_cache_list.append(k_tensor)
        v_cache_list.append(v_tensor)
    
    return k_cache_list, v_cache_list

def estimate_k_cr(k_caches, quant_scale_rel) -> float:
    original_size = 0
    compressed_size = 0
    for k_cache in tqdm.tqdm(k_caches, desc="Estimating k cache compression ratio"):
        block_num = k_cache.shape[2] // KVComCacheConfigStatic.BLOCK_SIZE
        blocked_k_cache = k_cache[:,:, :block_num * KVComCacheConfigStatic.BLOCK_SIZE, :].contiguous()
        original_size += blocked_k_cache.numel() * 2 # float16
        # max_ = blocked_k_cache.max()
        # min_ = blocked_k_cache.min()
        # quant_scale_ = (max_ - min_) * quant_scale_rel
        # quant_ints = ((blocked_k_cache - min_) / quant_scale_).round()
        quant_ints = block_channel_quant_int(blocked_k_cache, quant_scale_rel, KVComCacheConfigStatic.BLOCK_SIZE)[0].int()
        huffman_bits, codebook = huffman_encode(quant_ints)
        compressed_size += huffman_bits.numel() / 8
    return original_size / compressed_size

def estimate_v_cr(v_caches, quant_scale_rel) -> float:
    original_size = 0
    compressed_size = 0
    for v_cache in tqdm.tqdm(v_caches, desc="Estimating v cache compression ratio"):
        block_num = v_cache.shape[2] // KVComCacheConfigStatic.BLOCK_SIZE
        blocked_v_cache = v_cache[:,:, :block_num * KVComCacheConfigStatic.BLOCK_SIZE, :].contiguous()
        original_size += blocked_v_cache.numel() * 2 # float16
        quant_ints = block_token_quant_int(blocked_v_cache, quant_scale_rel, KVComCacheConfigStatic.BLOCK_SIZE)[0].int()
        huffman_bits, codebook = huffman_encode(quant_ints)
        compressed_size += huffman_bits.numel() / 8
    return original_size / compressed_size

try:
    k_caches, v_caches = load_caches_from_folder("cache")
    print(f"Loaded {len(k_caches)} layer caches")
    print(f"Estimated K cache compression ratio: {estimate_k_cr(k_caches, 0.02):.2f}")
    # print(f"Estimated V cache compression ratio: {estimate_v_cr(v_caches, 0.05):.2f}")
except ValueError as e:
    print(f"Error loading caches: {e}")

# per channel block size 64, quant scale rel 0.13: 5.33
# per layer, quant scale rel 0.01: