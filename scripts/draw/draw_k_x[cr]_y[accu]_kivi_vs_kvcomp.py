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

def get_result(accuracy_result_map, setting_map, ctx_len: int, k_quant_model: QuantMode):
    _base_quant_scale_rel = 0.001
    keys = setting_map.keys()
    accuracy_result = []
    for key in keys:
        setting = setting_map[key]
        if setting[0] == ctx_len and setting[1].k_quant_mode == k_quant_model:
            accuracy_result.append((setting, accuracy_result_map[key]))
    assert len(accuracy_result) == 5
    return accuracy_result[:-1]

def get_accuracy_result(accuracy_result_map, setting_map, model_name, benchmark, k_quant_model):
    keys = setting_map.keys()
    accuracy_result = []
    for key in keys:
        setting = setting_map[key]
        if setting[0] == benchmark and setting[1].model_name == model_name and setting[1].k_quant_mode == k_quant_model:
            accuracy_result.append((setting, accuracy_result_map[key]))

    assert len(accuracy_result) == 8
    return accuracy_result[3:-1]

logger = get_logger(__file__)

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

cr_throughput_setting_paths = ["../data/final_kv_combine_cr_throughput_setting_map.pkl"]
cr_throughput_setting_map = merge_map(cr_throughput_setting_paths)

cr_throughput_result_paths = ["../data/final_kv_combine_cr_throughput_result_map.pkl"]
cr_throughput_result_map = merge_map(cr_throughput_result_paths)

accruracy_setting_paths = ["../data/kv_combine_setting_map.pkl"]
accruracy_setting_map = merge_map(accruracy_setting_paths)

accruracy_result_paths = ["../data/kv_combine_accuracy_result_map.pkl"]
accruracy_result_map = merge_map(accruracy_result_paths)

model_list = [
    # "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-13b-hf",
    # "microsoft/phi-4",
    # "mistralai/Ministral-8B-Instruct-2410"
]

k_quant_model_list = [
    QuantMode.BlockQuant,
    QuantMode.ChannelQuant
]

benchmark_list = [
    "coqa",
    "gsm8k"
]

context_lengths = [2048 * i for i in range(1, 9)]

figures_path = "../final_figure/k_cr_accu/"

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

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o', color='#356ba0', linewidth=2)
    # Add a horizontal dashed line at y=0.97
    plt.axhline(y=0.97, color='gray', linestyle='--', alpha=0.7)

    ax = plt.gca()
    xlim = ax.get_xlim()
    # Position the text near the right edge, vertically centered on the line
    # Adjust the x-coordinate (e.g., xlim[1] * 0.95) and alignment as needed
    plt.text(xlim[1] * 0.98, 0.97, '0.97', va='center', ha='right', color='gray', fontsize=16, backgroundcolor='white', alpha=0.8)

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

def draw_two_lines_pic(xy1, xy2, title, save_path, label1="Line 1", label2="Line 2"):
    # Extract x and y coordinates from the input lists of pairs
    x1 = np.array([pair[0] for pair in xy1])
    y1 = np.array([pair[1] for pair in xy1])
    x2 = np.array([pair[0] for pair in xy2])
    y2 = np.array([pair[1] for pair in xy2])

    plt.figure(figsize=(6, 3))
    # Plot using the extracted coordinates
    plt.plot(x1, y1, marker='o', color='#356ba0', linewidth=2, label=label1)
    plt.plot(x2, y2, marker='s', color='#d9534f', linewidth=2, label=label2)

    plt.legend(prop=founders_reg_prop, fontsize=16)

    # Use a more general x-axis label if x represents different things for the two lines
    # Or adjust based on the specific meaning of x in your context
    plt.xlabel("Compression Ratio", fontproperties=founders_reg_prop, fontsize=16)
    plt.ylabel(f"{title}", fontproperties=founders_reg_prop, fontsize=16)
    # plt.title(title, fontproperties=founders_reg_prop, fontsize=16)
    plt.grid(alpha=0.3, linestyle='--', linewidth=0.7)
    plt.xticks(fontproperties=founders_reg_prop, fontsize=16)
    plt.yticks(fontproperties=founders_reg_prop, fontsize=16)
    plt.axhline(y=0.97, color='gray', linestyle='--', alpha=0.7)
    ax = plt.gca()
    xlim = ax.get_xlim()
    # Position the text near the right edge, vertically centered on the line
    # Adjust the x-coordinate (e.g., xlim[1] * 0.95) and alignment as needed
    plt.text(xlim[1] * 0.98, 0.97, '0.97', va='center', ha='right', color='gray', fontsize=12, backgroundcolor='white',
             alpha=0.8)
    plt.tight_layout() # Adjust layout to prevent labels overlapping
    # Ensure save_path directory exists
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"{title}.pdf"))
    plt.close()
y_key_map = {
    "exact_match_stderr,flexible-extract": "em_stderr,fe",
    "exact_match_stderr,strict-match": "em_stderr,sm",
    "exact_match,flexible-extract": "em,fe",
    "exact_match,strict-match": "em,sm",
}
def draw_two_lines_pics(xy1, xy2, title, save_path, label1="Line 1", label2="Line 2"):
    y_sample = xy1[0][1]
    y_keys = y_sample.keys()
    for y_key in y_keys:
        xy1_ = []
        xy2_ = []
        for ((x1, y_sample1),(x2,y_sample2)) in zip(xy1,xy2):
            xy1_.append((x1,y_sample1[y_key]))
            xy2_.append((x2, y_sample2[y_key]))
        max_y = 0
        for x, y in xy1_:
            if y > max_y:
                max_y = y
        for x, y in xy2_:
            if y > max_y:
                max_y = y
        for i in range(len(xy1_)):
            xy1_[i] = (xy1_[i][0], xy1_[i][1] / max_y)
        for i in range(len(xy2_)):
            xy2_[i] = (xy2_[i][0], xy2_[i][1] / max_y)
        if y_key in y_key_map:
            y_key = y_key_map[y_key]
        draw_two_lines_pic(xy1_,xy2_, f"{title}[{y_key}]", save_path, label1, label2)

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

# establish eb -> accu
for model_name in model_list:
    for benchmark in benchmark_list:
        ctx_len = 8192
        accuracy_rt = get_accuracy_result(accruracy_result_map, accruracy_setting_map, model_name, benchmark, QuantMode.BlockQuant)
        cr_throughput_rt = get_result(cr_throughput_result_map, cr_throughput_setting_map, ctx_len, QuantMode.BlockQuant)
        assert len(accuracy_rt) > 0, f"No result found for {model_name}, {benchmark}"
        assert len(cr_throughput_rt) > 0, f"No result found for {model_name}, {ctx_len}"
        assert len(accuracy_rt) == len(cr_throughput_rt)
        # assert len(rt) == 19, f"Result length is not 19 for {model_name}, {benchmark}, {k_quant_mode}, {is_k_var}"
        kvcomp_k_quant_scale_rels = []
        kvcomp_accuracys = []
        kvcomp_crs = []
        kvcomp_pairs = []
        for i, ((accuracy_setting, accuracy),(cr_throughput_setting, cr_throughput)) in enumerate(zip(accuracy_rt, cr_throughput_rt)):
            assert accuracy_setting[1] == cr_throughput_setting[1]
            kvcomp_k_quant_scale_rels.append(accuracy_setting[1].k_quant_scale_rel)
            accuracy = accuracy[benchmark]
            if "alias" in accuracy:
                del accuracy['alias']
            kvcomp_accuracys.append(accuracy)
            kvcomp_crs.append(crs_to_cr(cr_throughput[0]["k_huffman_crs"]))
            kvcomp_pairs.append((crs_to_cr(cr_throughput[0]["k_huffman_crs"]), accuracy))
        accuracy_rt = get_accuracy_result(accruracy_result_map, accruracy_setting_map, model_name, benchmark,
                                          QuantMode.ChannelQuant)
        cr_throughput_rt = get_result(cr_throughput_result_map, cr_throughput_setting_map, ctx_len,
                                      QuantMode.ChannelQuant)
        kivi_k_quant_scale_rels = []
        kivi_accuracys = []
        kivi_crs = []
        kivi_pairs = []
        for i, ((accuracy_setting, accuracy),(cr_throughput_setting, cr_throughput)) in enumerate(zip(accuracy_rt, cr_throughput_rt)):
            assert accuracy_setting[1] == cr_throughput_setting[1]
            kivi_k_quant_scale_rels.append(accuracy_setting[1].k_quant_scale_rel)
            accuracy = accuracy[benchmark]
            if "alias" in accuracy:
                del accuracy['alias']
            kivi_accuracys.append(accuracy)
            kivi_crs.append(crs_to_cr(cr_throughput[0]["k_quant_crs"]))
            kivi_pairs.append((crs_to_cr(cr_throughput[0]["k_quant_crs"]), accuracy))
        save_path = os.path.join(figures_path)
        os.makedirs(save_path, exist_ok=True)
        draw_two_lines_pics(kivi_pairs, kvcomp_pairs, f"{benchmark}", save_path, label1="kivi", label2="kvcomp")
