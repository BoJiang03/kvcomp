#! /usr/bin/env python
from typing import Dict, Tuple
import sys
import os
from hashlib import sha256

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.serialization import load, save, unified_hash

def make_hash_value_unified(setting_map, result_map):
    hash_value_map = {}
    for key, value in setting_map.items():
        hash_value = unified_hash(value)
        hash_value_map[key] = hash_value
        # print(f"Key: {key}, Hash Value: {hash_value}")
    
    for old_hash, new_hash in hash_value_map.items():
        assert old_hash in result_map
        value = result_map[old_hash]
        result_map[new_hash] = value
        del result_map[old_hash]

        assert old_hash in setting_map
        value = setting_map[old_hash]
        setting_map[new_hash] = value
        del setting_map[old_hash]
    return setting_map, result_map

setting_paths = ["./data/setting_map_llama2.pkl", "./data/setting_map_phi4_mistral.pkl", "./data/setting_map_llama2_7b.pkl", "./data/recovered_setting_map_llama2_13b_phi4_mistral.pkl"]
accuracy_result_paths = ["./data/accuracy_result_map_llama2.pkl", "./data/accuracy_result_map_phi4_mistral.pkl", "./data/accuracy_result_map_llama2_7b.pkl", "./data/recovered_accuracy_result_map_llama2_13b_phi4_mistral.pkl"]

for setting_path, accuracy_result_path in zip(setting_paths, accuracy_result_paths):
    setting_map = load(setting_path)
    accuracy_result_map = load(accuracy_result_path)
    setting_map, accuracy_result_map = make_hash_value_unified(setting_map, accuracy_result_map)
    save(setting_map, setting_path)
    save(accuracy_result_map, accuracy_result_path)
    print(f"Unified hash values in {setting_path} and {accuracy_result_path}.")
