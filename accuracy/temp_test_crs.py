from evaluation.cr_throughput_eval import crs_throughputs_evaluation_with_data
import torch
from utils.compute import QuantMode
from utils.config import KVCompCacheConfig
import os


def load(
        root_dir: str,
        model_name: str,
        benchmark: str
):
    caches = []
    round_dir = os.path.join(root_dir, model_name, benchmark)
    round_num = len(os.listdir(round_dir))
    for round_i in range(round_num):
        caches.append(([],[]))
        k_dir = os.path.join(round_dir, str(round_i), "k")
        v_dir = os.path.join(round_dir, str(round_i), "v")
        k_files = os.listdir(k_dir)
        v_files = os.listdir(v_dir)
        for k_file in k_files:
            caches[round_i][0].append(torch.load(os.path.join(k_dir, k_file)))
        for v_file in v_files:
            caches[round_i][1].append(torch.load(os.path.join(v_dir, v_file)))
    return caches


root_dir = "dumped_cache"
config = KVCompCacheConfig(
    enable_quant=True,
    model_name="meta-llama/Llama-2-13b-hf",
    k_quant_mode=QuantMode.BlockQuant,
    v_quant_mode=QuantMode.TokenQuant,
    k_block_size=64,
    v_block_size=128,
    k_recent_size=128,
    v_recent_size=128,
    k_quant_scale_rel=0.04,
    v_quant_scale_rel=0.04
)
caches = load(root_dir, config.model_name, "gsm8k")
round_num = len(caches)

rts = []

for round_i in range(round_num):
    key_caches = caches[round_i][0]
    value_caches = caches[round_i][1]
    rt = crs_throughputs_evaluation_with_data(config, key_caches, value_caches)
    rts.append(rt)

print(rts)