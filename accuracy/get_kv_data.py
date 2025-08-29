#! /usr/bin/env python
from evaluation.evaluation import get_collected_data
from utils.compute import QuantMode, print_quant_setting
from utils.config import KVCompCacheConfig
from utils.util import get_logger
import torch
# register_notify()
def save_to_path(list, path):
    for item, idx in zip(list, range(len(list))):
        torch.save(item, path + f"/{idx}.pt")
logger = get_logger(__file__)
# block_other_logger(logger)

print_quant_setting()
block_size = 64

config = KVCompCacheConfig(
    enable_quant=False,
    # model_name="microsoft/Phi-3.5-mini-instruct",
    model_name="meta-llama/Llama-2-7b-hf",
    enable_pre_rope=False,
    k_quant_mode=QuantMode.ChannelQuant,
    v_quant_mode=QuantMode.TokenQuant,
    k_block_size=block_size,
    v_block_size=block_size,
    k_recent_size=block_size * 2,
    v_recent_size=block_size * 2,
    k_quant_scale_rel=0.01,
    v_quant_scale_rel=0.01,
    device="cuda:1",
)
logger.info(config)
data = get_collected_data(config)
logger.info(f"{data}")
save_to_path(data.key_caches.values(), "./kv/k")
save_to_path(data.value_caches.values(), "./kv/v")