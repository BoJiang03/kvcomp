import os
from evaluation.evaluation import get_collected_data
from utils.util import get_logger
import matplotlib.pyplot as plt
import numpy as np
from utils.compute import cut_tensor, quant
import torch
from tqdm import tqdm
import matplotlib


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
    plt.xlabel("Quant Ints", fontproperties=founders_reg_prop, fontsize=12)
    plt.ylabel("Frequency", fontproperties=founders_reg_prop, fontsize=12)
    plt.tight_layout()
    plt.savefig(full_path)
    plt.close()
    logger.info(f"Histogram saved to {full_path}")

logger = get_logger(__file__)
rt = get_collected_data(logger)

test_tensor = rt.key_caches[0][-1]
def draw_k_histogram(k_tensor, dir_name, file_name):
    k_block_size = 64

    to_compress, buffer_ = cut_tensor(None, k_tensor, k_block_size, 1)
    to_compress = to_compress.view(to_compress.shape[0], to_compress.shape[1], -1, k_block_size, to_compress.shape[3])
    quant_ints, _, _ =  quant(to_compress, [0,1,3,4], 0.05)
    quant_ints = quant_ints.to(torch.int)
    ints_ = quant_ints.flatten()
    dir_name = os.path.join(dir_name, "k")
    # Pass dir_name and file_name to the drawing function
    draw_single_histogram(ints_, dir_name ,file_name)

def draw_v_histogram(v_tensor, dir_name, file_name):
    quant_ints, _, _ =  quant(v_tensor, [0,1,3], 0.1)
    quant_ints = quant_ints.to(torch.int)
    ints_ = quant_ints.flatten()
    dir_name = os.path.join(dir_name, "v")
    # Pass dir_name and file_name to the drawing function
    draw_single_histogram(ints_, dir_name ,file_name)


output_directory = "../histograms"

k_tensors = []
v_tensors = []
for i in range(len(rt.key_caches[0])):
    k_tensors.append(rt.key_caches[0][i-1])
    v_tensors.append(rt.value_caches[0][i-1])

for i, (k_tensor, v_tensor) in tqdm(enumerate(zip(k_tensors, v_tensors)), total=len(k_tensors), desc="Drawing histograms"):
    base_file_name = f"{i}"
    draw_k_histogram(k_tensor, output_directory, base_file_name)
    draw_v_histogram(v_tensor, output_directory, base_file_name)
