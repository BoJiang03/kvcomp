#! /usr/bin/env python
from utils.compute import QuantMode
from utils.config import KVCompCacheConfig
from utils.serialization import save
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

# model_list = [
#     "meta-llama/Llama-2-7b-hf",
#     "meta-llama/Llama-2-13b-hf",
#     # "meta-llama/Llama-3.1-8B",
#     # "mistralai/Ministral-8B-Instruct-2410",
#     # "microsoft/phi-4",
#     # "meta-llama/Meta-Llama-3-8B",
#     # "mistralai/Mistral-Small-24B-Base-2501",
#     # "mistralai/Mistral-Small-Instruct-2409",
# ]

model_list = [
    "meta-llama/Llama-2-7b-hf",
    # "meta-llama/Llama-2-13b-hf",
]

benchmark_list = [
    "coqa",
    "truthfulqa",
    "piqa",
    "winogrande"
]

step_map = {
    QuantMode.BlockQuant: 0.005,
    QuantMode.ChannelQuant: 0.02,
    QuantMode.TokenQuant: 0.01,
}

stop_map = {
    QuantMode.BlockQuant: 0.05,
    QuantMode.ChannelQuant: 0.20,
    QuantMode.TokenQuant: 0.1,
}

setting_map = {}

default_eb_rel = 0.001

for model_name in model_list:
    for benchmark in benchmark_list:
        for quant_mode in quant_mode_map.keys():
            k_quant_mode, v_quant_mode = quant_mode_map[quant_mode]
            k_quant_scale_rel_step = step_map[k_quant_mode]
            k_quant_scale_rel_stop = stop_map[k_quant_mode]
            for k_quant_scale_rel in np.arange(k_quant_scale_rel_step, k_quant_scale_rel_stop, k_quant_scale_rel_step):
                config = KVCompCacheConfig(
                    model_name=model_name,
                    k_quant_mode=k_quant_mode,
                    v_quant_mode=v_quant_mode,
                    enable_quant=True,
                    k_block_size=K_BLOCK_SIZE,
                    v_block_size=V_BLOCK_SIZE,
                    k_recent_size=K_RECENT_SIZE,
                    v_recent_size=V_RECENT_SIZE,
                    k_quant_scale_rel=float(k_quant_scale_rel),
                    v_quant_scale_rel=default_eb_rel
                )
                pair = (benchmark, config)
                if pair.__hash__() in setting_map:
                    assert pair == setting_map[pair.__hash__()], "config with same hash but different value"
                else:
                    setting_map[pair.__hash__()] = pair
            v_quant_scale_rel_step = step_map[v_quant_mode]
            v_quant_scale_rel_stop = stop_map[v_quant_mode]
            for v_quant_scale_rel in np.arange(v_quant_scale_rel_step, v_quant_scale_rel_stop, v_quant_scale_rel_step):
                config = KVCompCacheConfig(
                    model_name=model_name,
                    k_quant_mode=k_quant_mode,
                    v_quant_mode=v_quant_mode,
                    enable_quant=True,
                    k_block_size=K_BLOCK_SIZE,
                    v_block_size=V_BLOCK_SIZE,
                    k_recent_size=K_RECENT_SIZE,
                    v_recent_size=V_RECENT_SIZE,
                    k_quant_scale_rel=default_eb_rel,
                    v_quant_scale_rel=float(v_quant_scale_rel)
                )
                pair = (benchmark, config)
                if pair.__hash__() in setting_map:
                    assert pair == setting_map[pair.__hash__()], "config with same hash but different value"
                else:
                    setting_map[pair.__hash__()] = pair

setting_path = "./data/llama2_7b_setting_map.pkl"
save(setting_map, setting_path)
logger.info(f"Setting map saved to {setting_path}, total {len(setting_map)} entries")
