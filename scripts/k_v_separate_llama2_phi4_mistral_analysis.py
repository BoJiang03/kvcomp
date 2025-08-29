#! /usr/bin/env python
from typing import Dict, Tuple
import sys
import os

from utils.compute import QuantMode
import matplotlib.pyplot as plt
import numpy as np
# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import KVCompCacheConfig
from utils.serialization import load, save

def merge_map(paths):
    _map = {}
    for path in paths:
        _map_ = load(path)
        for key, value in _map_.items():
            if key not in _map:
                _map[key] = value
            else:
                assert _map[key] == value, f"Key {key} already exists with different value."
    return _map

setting_paths = ["./data/setting_map_piqa_winogrande.pkl"]
setting_map = merge_map(setting_paths)

accuracy_result_paths = ["./data/accuracy_result_map_piqa_winogrande.pkl"]
accuracy_result_map = merge_map(accuracy_result_paths)

def get_result(accuracy_result_map, setting_map, benchmark: str, model_name: str, k_quant_mode: QuantMode, is_k_var: bool):
    _base_quant_scale_rel = 0.001
    keys = setting_map.keys()
    accuracy_result = []
    for key in keys:
        setting = setting_map[key]
        if setting[0] == benchmark and setting[1].model_name == model_name and setting[1].k_quant_mode == k_quant_mode:
            if (is_k_var and setting[1].k_quant_scale_rel != _base_quant_scale_rel) or (not is_k_var and setting[1].v_quant_scale_rel != _base_quant_scale_rel):
                accuracy_result.append((setting, accuracy_result_map[key]))

    return accuracy_result

model_list = [
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-13b-hf",
    "microsoft/phi-4",
    "mistralai/Ministral-8B-Instruct-2410"
]

benchmark_list = [
    "piqa",
    "winogrande"
]

k_quant_model_list = [
    QuantMode.BlockQuant,
    QuantMode.ChannelQuant
]

is_k_var_list = [True, False]

figures_path = "./figures/k_v_separate/"

def result_to_accuracy_float(benchmark, result):
    # if benchmark == "coqa":
    #     return result[benchmark]["em,none"]
    # else:
    #     raise ValueError(f"Unknown benchmark: {benchmark}")
    _rt = result[benchmark]
    # delete "alias" key
    if "alias" in _rt:
        del _rt["alias"]
    return _rt

def draw_pic(x, y, title, save_path):
    x = np.array(x)
    y = np.array(y)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o')
    plt.title(title)
    plt.xlabel("Quantization Scale Relative")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.savefig(os.path.join(save_path, f"{title}.png"))
    plt.close()

def draw_pics(x, ys, save_path):
    y_sample = ys[0]
    y_keys = y_sample.keys()
    for y_key in y_keys:
        y = []
        for y_sample in ys:
            y.append(y_sample[y_key])
        draw_pic(x, y, y_key, save_path)

for model_name in model_list:
    for benchmark in benchmark_list:
        for k_quant_mode in k_quant_model_list:
            for is_k_var in is_k_var_list:
                rt = get_result(accuracy_result_map, setting_map, benchmark, model_name, k_quant_mode, is_k_var)
                print(f"Model: {model_name}, Benchmark: {benchmark}, K Quant Mode: {k_quant_mode}, Is K Var: {is_k_var}")
                assert len(rt) > 0, f"No result found for {model_name}, {benchmark}, {k_quant_mode}, {is_k_var}"
                assert len(rt) == 19, f"Result length is not 19 for {model_name}, {benchmark}, {k_quant_mode}, {is_k_var}"
                quant_rel_list = []
                accuracy_list = []
                for setting, accuracy in rt:
                    quant_rel_list.append(setting[1].k_quant_scale_rel if is_k_var else setting[1].v_quant_scale_rel)
                    accuracy_list.append(result_to_accuracy_float(benchmark, accuracy))
                print(f"Quant Rel List: {quant_rel_list}")
                print(f"Accuracy List: {accuracy_list}")
                # Save the results to a file
                save_path = os.path.join(figures_path,model_name,benchmark,k_quant_mode.value, "k" if is_k_var else "v")
                os.makedirs(save_path, exist_ok=True)
                # save({"quant_rels": quant_rel_list, "accuracies": accuracy_list}, save_file_path)
                draw_pics(quant_rel_list, accuracy_list, save_path)