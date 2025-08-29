#! /usr/bin/env python
import sys
import os
# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.compute import QuantMode
from utils.config import KVCompCacheConfig
from utils.serialization import save, unified_hash
from utils.util import get_logger, block_other_logger
import numpy as np
import pickle

logger = get_logger(__file__)
block_other_logger(logger)
# register_notify()

K_BLOCK_SIZE = 64
V_BLOCK_SIZE = 128
K_RECENT_SIZE = 128
V_RECENT_SIZE = 128 + 128

quant_mode_map = {
    "kvcomp": (QuantMode.BlockQuant, QuantMode.TokenQuant),
    "kivi": (QuantMode.ChannelQuant, QuantMode.TokenQuant)
}

model_list = [
    # "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-13b-hf",
    # "microsoft/phi-4",
    # "mistralai/Ministral-8B-Instruct-2410",
]

benchmark_list = [
    "coqa",
]

block_quant_k = 0.05
channel_quant_k = 0.25
token_quant_v = 0.1
point_number = 8 + 5

def gen_quant_rels(turning_point, point_number):
    default_eb_rel = 0.001
    quant_rels = [default_eb_rel]
    delta_ = turning_point / 8
    for i in range(1, point_number):
        quant_rels.append(i * delta_)

    return quant_rels

block_quant_k_quant_scale_rels = gen_quant_rels(block_quant_k, point_number)
channel_quant_k_quant_scale_rels = gen_quant_rels(channel_quant_k, point_number)
token_quant_v_quant_scale_rels = gen_quant_rels(token_quant_v, point_number)

setting_map = {}

for model_name in model_list:
    for benchmark in benchmark_list:
        for block_quant_k_quant_rel, token_quant_v_quant_rel in zip(block_quant_k_quant_scale_rels, token_quant_v_quant_scale_rels):
            config = KVCompCacheConfig(
                model_name=model_name,
                k_quant_mode=QuantMode.BlockQuant,
                v_quant_mode=QuantMode.TokenQuant,
                enable_quant=True,
                k_block_size=K_BLOCK_SIZE,
                v_block_size=V_BLOCK_SIZE,
                k_recent_size=K_RECENT_SIZE,
                v_recent_size=V_RECENT_SIZE,
                k_quant_scale_rel=float(block_quant_k_quant_rel),
                v_quant_scale_rel=float(token_quant_v_quant_rel)
            )
            print((block_quant_k_quant_rel, token_quant_v_quant_rel))
            pair = (benchmark, config)
            if unified_hash(pair) in setting_map:
                assert pair == setting_map[unified_hash(pair)], "config with same hash but different value"
            else:
                setting_map[unified_hash(pair)] = pair

        for channel_quant_k_quant_rel, token_quant_v_quant_rel in zip(channel_quant_k_quant_scale_rels, token_quant_v_quant_scale_rels):
            config = KVCompCacheConfig(
                model_name=model_name,
                k_quant_mode=QuantMode.ChannelQuant,
                v_quant_mode=QuantMode.TokenQuant,
                enable_quant=True,
                k_block_size=K_BLOCK_SIZE,
                v_block_size=V_BLOCK_SIZE,
                k_recent_size=K_RECENT_SIZE,
                v_recent_size=V_RECENT_SIZE,
                k_quant_scale_rel=float(channel_quant_k_quant_rel),
                v_quant_scale_rel=float(token_quant_v_quant_rel)
            )
            print((channel_quant_k_quant_rel, token_quant_v_quant_rel))
            pair = (benchmark, config)
            if unified_hash(pair) in setting_map:
                assert pair == setting_map[unified_hash(pair)], "config with same hash but different value"
            else:
                setting_map[unified_hash(pair)] = pair
            
setting_path = "./data/kv_combine_setting_map.pkl"
print(setting_map)
save(setting_map, setting_path)
logger.info(f"Setting map saved to {setting_path}, total {len(setting_map)} entries")
