import os
from utils.util import get_logger
import matplotlib.pyplot as plt
import numpy as np
from utils.serialization import load
from utils.compute import QuantMode
from matplotlib.font_manager import FontProperties

personal_path = '/home/tut44803/.local/share/fonts/'

font_path = personal_path + 'FoundersGrotesk-Regular.otf'
founders_reg_prop = FontProperties(fname=font_path, size=16)

def get_result(accuracy_result_map, setting_map, ctx_len: int):
    _base_quant_scale_rel = 0.001
    keys = setting_map.keys()
    accuracy_result = []
    for key in keys:
        setting = setting_map[key]
        if setting[0] == ctx_len and setting[1].k_quant_mode == QuantMode.BlockQuant:
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

setting_paths = ["../data/final_kv_combine_cr_throughput_setting_map.pkl"]
setting_map = merge_map(setting_paths)

accuracy_result_paths = ["../data/final_kv_combine_cr_throughput_result_map.pkl"]
accuracy_result_map = merge_map(accuracy_result_paths)

model_list = [
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-13b-hf",
    "microsoft/phi-4",
    "mistralai/Ministral-8B-Instruct-2410"
]

k_quant_model_list = [
    QuantMode.BlockQuant,
    QuantMode.ChannelQuant
]

context_lengths = [2048 * i for i in range(1, 9)]


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

def draw_one_line_pic(x, y, title, save_path):
    x = np.array(x)
    y = np.array(y)

    y[np.isinf(y)] = np.nan

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o', color='#356ba0', linewidth=2)
    # Apply the specified font and increase font size for labels
    plt.xlabel("Relative Quantization Scale", fontproperties=founders_reg_prop, fontsize=14)
    plt.ylabel(f"{title}", fontproperties=founders_reg_prop, fontsize=14)
    # Optionally apply to title as well
    # plt.title(title, fontproperties=founders_reg_prop, fontsize=16)
    plt.grid(alpha=0.3, linestyle='--', linewidth=0.7)
    # Increase font size for tick labels
    plt.xticks(fontproperties=founders_reg_prop, fontsize=12)
    plt.yticks(fontproperties=founders_reg_prop, fontsize=12)
    plt.tight_layout() # Adjust layout to prevent labels overlapping
    plt.savefig(os.path.join(save_path, f"{title}.pdf"))
    plt.close()

def draw_two_lines_pic(x, y1, y2, title, save_path, label1="Line 1", label2="Line 2"):
    x = np.array(x)
    y1 = np.array(y1)
    y2 = np.array(y2)

    plt.figure(figsize=(6, 3))
    plt.plot(x, y1, marker='o', color='#356ba0', linewidth=2, label=label1)
    plt.plot(x, y2, marker='s', color='#d9534f', linewidth=2, label=label2)

    plt.legend(prop=founders_reg_prop, fontsize=16)

    plt.xlabel("Relative Quantization Scale", fontproperties=founders_reg_prop, fontsize=16)
    plt.ylabel(f"{title}", fontproperties=founders_reg_prop, fontsize=16)
    # plt.title(title, fontproperties=founders_reg_prop, fontsize=16)
    plt.grid(alpha=0.3, linestyle='--', linewidth=0.7)
    plt.xticks(fontproperties=founders_reg_prop, fontsize=16)
    plt.yticks(fontproperties=founders_reg_prop, fontsize=16)
    plt.tight_layout() # Adjust layout to prevent labels overlapping
    plt.savefig(os.path.join(save_path, f"kernel_time.pdf"))
    plt.close()

def draw_pics(x, ys, save_path):
    y_sample = ys[0]
    y_keys = y_sample.keys()
    for y_key in y_keys:
        y = []
        for y_sample in ys:
            y.append(y_sample[y_key])
        draw_one_line_pic(x, y, y_key, save_path)

def crs_to_cr(crs):
    total_crs = len(crs)
    compressed_size = 0
    for cr in crs:
        compressed_size += 1 / cr
    return total_crs / compressed_size

def all_to_one(result):
    for key in result.keys():
        obj = result[key]
        if isinstance(obj, tuple):
            assert len(obj) == 1
            obj = obj[0]
        if isinstance(obj, list):
            if "time" in key:
                obj = sum(obj)
            if "throughput" in key:
                obj = sum(obj) / len(obj)
        result[key] = obj
    return result

figures_path = "../final_figure/eb_single_cublas/"

for ctx_len in context_lengths:
    rt = get_result(accuracy_result_map, setting_map, ctx_len)
    assert len(rt) > 0, f"No result found for {ctx_len}"
    # assert len(rt) == 19, f"Result length is not 19 for {model_name}, {benchmark}, {k_quant_mode}, {is_k_var}"
    k_kvcomp_quant_scale_rels = []
    v_kvcomp_quant_scale_rels = []
    k_cublas_kernel_time = []
    k_single_kernel_time = []
    v_cublas_kernel_time = []
    v_single_kernel_time = []
    for setting, result in rt:
        assert len(result) == 1
        k_kvcomp_quant_scale_rels.append(setting[1].k_quant_scale_rel)
        v_kvcomp_quant_scale_rels.append(setting[1].v_quant_scale_rel)
        del result[0]["k_quant_crs"]
        del result[0]["v_quant_crs"]
        del result[0]["k_huffman_crs"]
        del result[0]["v_huffman_crs"]
        result = all_to_one(result[0])
        k_cublas_kernel_time.append(result["k_pytorch_mat_vec_mul_times"])
        v_cublas_kernel_time.append(result["v_pytorch_mat_vec_mul_times"])
        k_single_kernel_time.append(result["k_fused_mat_vec_mul_times"])
        v_single_kernel_time.append(result["v_fused_mat_vec_mul_times"])
    k_save_path = os.path.join(figures_path, "k", str(ctx_len))
    os.makedirs(k_save_path, exist_ok=True)
    v_save_path = os.path.join(figures_path, "v", str(ctx_len))
    os.makedirs(v_save_path, exist_ok=True)
    # draw_pics(kvcomp_quant_scale_rels, kvcomp_results, save_path)
    draw_two_lines_pic(k_kvcomp_quant_scale_rels, k_cublas_kernel_time, k_single_kernel_time, f"Kernel Time", k_save_path,
                       label1="cublas", label2="single_kernel")
    draw_two_lines_pic(v_kvcomp_quant_scale_rels, v_cublas_kernel_time, v_single_kernel_time, f"Kernel Time", v_save_path,
                       label1="cublas", label2="single_kernel")
