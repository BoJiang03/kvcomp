
import os
from evaluation.evaluation import get_collected_data
from utils.util import get_logger
import matplotlib.pyplot as plt
import numpy as np
from utils.compute import cut_tensor, quant
import torch
from tqdm import tqdm
import matplotlib
from utils.serialization import load, save
from utils.compute import QuantMode
from matplotlib.font_manager import FontProperties

personal_path = '/home/tut44803/.local/share/fonts/'

font_path = personal_path + 'FoundersGrotesk-Regular.otf'
founders_reg_prop = FontProperties(fname=font_path)

def draw_single_histogram(ints_, dir_name, file_name):
    if isinstance(ints_, torch.Tensor):
        ints_ = ints_.cpu().numpy()

    os.makedirs(dir_name, exist_ok=True)
    full_path = os.path.join(dir_name, file_name + ".pdf")

    plt.figure(figsize=(5,2))
    plt.hist(ints_, bins=np.arange(ints_.min(), ints_.max() + 2) - 0.5, color="#2C708E", alpha=0.95)
    plt.yscale("log")
    plt.xlabel("Quant Ints", fontproperties=founders_reg_prop, fontsize=14)
    plt.ylabel("Frequency", fontproperties=founders_reg_prop, fontsize=14)
    plt.tight_layout()
    plt.savefig(full_path)
    plt.close()
    logger.info(f"Histogram saved to {full_path}")

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

setting_paths = ["../data/setting_map_coqa_truthfulqa_gen.pkl"]
setting_map = merge_map(setting_paths)

accuracy_result_paths = ["../data/accuracy_result_map_coqa_truthfulqa_gen.pkl"]
accuracy_result_map = merge_map(accuracy_result_paths)

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
figures_path = "../final_figure/k_v_separate/"

font_path = personal_path + 'FoundersGrotesk-Regular.otf'
founders_reg_prop = FontProperties(fname=font_path)

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

def draw_one_line_pic(x, y, title, save_path, benchmark):
    x = np.array(x)
    y = np.array(y)

    plt.figure(figsize=(6, 3))
    plt.plot(x, y, marker='o', color='#356ba0', linewidth=2)
    # Add a horizontal dashed line at y=0.97
    plt.axhline(y=0.97, color='gray', linestyle='--', alpha=0.7)

    ax = plt.gca()
    xlim = ax.get_xlim()
    # Position the text near the right edge, vertically centered on the line
    # Adjust the x-coordinate (e.g., xlim[1] * 0.95) and alignment as needed
    plt.text(xlim[1] * 0.98, 0.97, '0.97', va='center', ha='right', color='gray', fontsize=12, backgroundcolor='white', alpha=0.8)

    # Apply the specified font and increase font size for labels
    plt.xlabel("Relative Quantization Scale", fontproperties=founders_reg_prop, fontsize=16)
    plt.ylabel(f"{benchmark}[{title}]", fontproperties=founders_reg_prop, fontsize=16)
    # Optionally apply to title as well
    # plt.title(title, fontproperties=founders_reg_prop, fontsize=16)
    plt.grid(alpha=0.3, linestyle='--', linewidth=0.7)
    # Increase font size for tick labels
    plt.xticks(fontproperties=founders_reg_prop, fontsize=16)
    plt.yticks(fontproperties=founders_reg_prop, fontsize=16)
    plt.tight_layout() # Adjust layout to prevent labels overlapping
    plt.savefig(os.path.join(save_path, f"{benchmark}[{title}].pdf"))
    plt.close()

def draw_pics(x, ys, save_path, benchmark):
    y_sample = ys[0]
    y_keys = y_sample.keys()
    for y_key in y_keys:
        y = []
        for y_sample in ys:
            y.append(y_sample[y_key])
        baseline_accuracy = y[0]
        y = [y_i / baseline_accuracy for y_i in y]
        draw_one_line_pic(x, y, y_key, save_path, benchmark)

for model_name in model_list:
    for benchmark in benchmark_list:
        for k_quant_mode in k_quant_model_list:
            for is_k_var in is_k_var_list:
                rt = get_result(accuracy_result_map, setting_map, benchmark, model_name, k_quant_mode, is_k_var)
                print(f"Model: {model_name}, Benchmark: {benchmark}, K Quant Mode: {k_quant_mode}, Is K Var: {is_k_var}")
                assert len(rt) > 0, f"No result found for {model_name}, {benchmark}, {k_quant_mode}, {is_k_var}"
                # assert len(rt) == 19, f"Result length is not 19 for {model_name}, {benchmark}, {k_quant_mode}, {is_k_var}"
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
                draw_pics(quant_rel_list, accuracy_list, save_path, benchmark)