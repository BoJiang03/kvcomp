#! /usr/bin/env python
from typing import Dict, Tuple
import sys
import os

from utils.compute import QuantMode
import matplotlib.pyplot as plt
import numpy as np
# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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

setting_paths = ["./data/kv_combine_setting_map.pkl"]
setting_map = merge_map(setting_paths)

accuracy_result_paths = ["./data/kv_combine_accuracy_result_map.pkl"]
accuracy_result_map = merge_map(accuracy_result_paths)

def get_result(accuracy_result_map, setting_map, benchmark: str, model_name: str, k_quant_mode: QuantMode):
    _base_quant_scale_rel = 0.001
    keys = setting_map.keys()
    accuracy_result = []
    for key in keys:
        setting = setting_map[key]
        if setting[0] == benchmark and setting[1].model_name == model_name and setting[1].k_quant_mode == k_quant_mode:
            accuracy_result.append((setting, accuracy_result_map[key]))

    return accuracy_result

model_list = [
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-13b-hf",
    "microsoft/phi-4",
    "mistralai/Ministral-8B-Instruct-2410"
]

benchmark_list = [
    "coqa",
]

k_quant_model_list = [
    QuantMode.BlockQuant,
    QuantMode.ChannelQuant
]

is_k_var_list = [True, False]

figures_path = "./figures/kv_combine/"

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
    # Convert x and y to numpy arrays
    x = np.array(x)
    y = np.array(y)

    # Create a figure
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(x)), y, marker='o')  # Use indices for plotting

    # Set x-axis labels as tuples
    plt.xticks(range(len(x)), [str(t) for t in x], rotation=45)

    # Add title and labels
    plt.title(title)
    plt.xlabel("Quantization Scale (Tuple)")
    plt.ylabel("Accuracy")
    plt.grid()

    # Save the plot
    plt.tight_layout()  # Adjust layout to prevent label overlap
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
            rt = get_result(accuracy_result_map, setting_map, benchmark, model_name, k_quant_mode)
            print(f"Model: {model_name}, Benchmark: {benchmark}, K Quant Mode: {k_quant_mode}")
            assert len(rt) > 0, f"No result found for {model_name}, {benchmark}, {k_quant_mode}"
            assert len(rt) == 13, f"Result length is not 13 for {model_name}, {benchmark}, {k_quant_mode}"
            quant_rel_list = []
            accuracy_list = []
            for setting, accuracy in rt:
                quant_rel_list.append((setting[1].k_quant_scale_rel, setting[1].v_quant_scale_rel))
                accuracy_list.append(result_to_accuracy_float(benchmark, accuracy))
            print(f"Quant Rel List: {quant_rel_list}")
            print(f"Accuracy List: {accuracy_list}")
            # Save the results to a file
            save_path = os.path.join(figures_path,model_name,benchmark,k_quant_mode.value)
            os.makedirs(save_path, exist_ok=True)
            # save({"quant_rels": quant_rel_list, "accuracies": accuracy_list}, save_file_path)
            draw_pics(quant_rel_list, accuracy_list, save_path)