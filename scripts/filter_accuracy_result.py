#! /usr/bin/env python
from typing import Dict, Tuple
import sys
import os

from utils.compute import QuantMode

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import KVCompCacheConfig
from utils.serialization import load, save

def filter_accuracy_result_map(accuracy_result_map: Dict[str, float], setting_map: Dict[str, Tuple[str, KVCompCacheConfig]]) -> Dict[str, float]:
    filtered_map = {}
    for key, value in accuracy_result_map.items():
        if key in setting_map:
            filtered_map[key] = value
        else:
            print(f"Key {key} not found in setting map.")
    return filtered_map

setting_paths = ["./data/setting_map_llama2.pkl", "./data/setting_map_phi4_mistral.pkl", "./data/setting_map_llama2_7b.pkl", "./data/setting_map_llama2_13b_phi4_mistral.pkl"]
accuracy_result_paths = ["./data/accuracy_result_map_llama2.pkl", "./data/accuracy_result_map_phi4_mistral.pkl", "./data/accuracy_result_map_llama2_7b.pkl", "./data/accuracy_result_map_llama2_13b_phi4_mistral.pkl"]

for setting_path, accuracy_result_path in zip(setting_paths, accuracy_result_paths):
    setting_map = load(setting_path)
    accuracy_result_map = load(accuracy_result_path)
    rt = filter_accuracy_result_map(accuracy_result_map, setting_map)
    save(rt, accuracy_result_path)
    print(f"Filtered {len(rt)} items from {accuracy_result_path} based on {setting_path}.")
# This script filters the accuracy result map based on the setting map.
