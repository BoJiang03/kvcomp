import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from utils.compute import cut_tensor, quant

parent_path = "dumped_cache"
model_name = "meta-llama/Llama-2-13b-hf"
benchmark = "gsm8k"
test_index = 0

k_block_size = 64
k_recent_size = 128
k_quant_rel = 0.03
v_block_size = 64
v_recent_size = 128
v_quant_rel = 0.07

cache_path = os.path.join(parent_path, model_name, benchmark, f"{test_index}")
k_path = os.path.join(cache_path, "k")
v_path = os.path.join(cache_path, "v")
k_files = os.listdir(k_path)
v_files = os.listdir(v_path)

k_files.sort()
v_files.sort()
total_layer = len(k_files)
# draw_layer = 1
# draw_layer = total_layer // 2
draw_layer = total_layer - 1

k_file = k_files[draw_layer]
v_file = v_files[draw_layer]
k = torch.load(os.path.join(k_path, k_file))
v = torch.load(os.path.join(v_path, v_file))

def block_and_quant(tensor, block_size, recent_size, quant_dims, quant_rel):
    to_quant, buffer = cut_tensor(None, tensor, block_size, recent_size)
    batch_size, head_num, seq_len, head_dim = tensor.shape
    blocked = to_quant.view(batch_size, head_num, -1, block_size, head_dim)
    return quant(blocked, quant_dims, quant_rel)

k_quant_int, k_quant_min, k_quant_scale = block_and_quant(k, k_block_size, k_recent_size, [0,1,3,4], k_quant_rel)
v_quant_int, v_quant_min, v_quant_scale = block_and_quant(v, v_block_size, v_recent_size, [0,1,4], v_quant_rel)


def draw_histogram(ints, file_name):
    ints = ints.flatten().to(torch.int)
    ints = ints.cpu().numpy()
    min_ = ints.min()
    max_ = ints.max()

    bins = np.arange(min_, max_ + 1, 1)
    hist, _ = np.histogram(ints, bins=bins)
    hist = hist / hist.sum()
    plt.bar(bins[:-1], hist, width=1)

    # Set font size for x and y axis
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Save as PDF
    plt.savefig(file_name, format='pdf')
    plt.close()

draw_histogram(k_quant_int, f"{benchmark}_k_{draw_layer}.pdf")
draw_histogram(v_quant_int, f"{benchmark}_v_{draw_layer}.pdf")