#! /usr/bin/env python
from typing import Dict, Tuple
import sys
import os
# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.evaluation import accuracy_evaluation
from utils.config import KVCompCacheConfig
from utils.serialization import load, save
from utils.util import get_logger, block_other_logger, register_notify
from tqdm import tqdm

logger = get_logger(__file__)
block_other_logger(logger)
setting_path = "./data/setting_map.pkl"
setting_map: Dict[int, Tuple[str, KVCompCacheConfig]] = load(setting_path)
logger.info(f"Setting map loaded from {setting_path} with {len(setting_map)} entries")

save_path = "./data/accuracy_result_map.pkl"
accuracy_result_map = load(save_path)

llama2_7b_accuracy_result_map = {}

for hash, pair in tqdm(setting_map.items(), total=len(setting_map), desc="Filtering"):
    if hash in accuracy_result_map and "Llama-2-7b" in pair[1].model_name:
        llama2_7b_accuracy_result_map[hash] = accuracy_result_map[hash]
    
print(f"Filtered {len(llama2_7b_accuracy_result_map)} entries")

hash_0, pair_0 = list(llama2_7b_accuracy_result_map.items())[0]
print(hash_0)
print(pair_0)