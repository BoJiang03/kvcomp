#! /usr/bin/env python
from typing import Dict, Tuple
import sys
import os

import torch.cuda

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.evaluation import cr_throughput_evaluation
from utils.config import KVCompCacheConfig
from utils.serialization import load, save
from utils.util import get_logger, block_other_logger, register_notify
from tqdm import tqdm
import multiprocessing

def run_cr_evaluation(config, ctx_len, logger, result_queue):
    cr_throughput = cr_throughput_evaluation(config, ctx_len, False, logger)
    result_queue.put(cr_throughput)

register_notify()

logger = get_logger(__file__)
block_other_logger(logger)
setting_path = "./data/final_kv_combine_cr_throughput_setting_map.pkl"
setting_map: Dict[int, Tuple[int, KVCompCacheConfig]] = load(setting_path)
logger.info(f"Setting map loaded from {setting_path} with {len(setting_map)} entries")

save_path = "./data/final_kv_combine_cr_throughput_result_map.pkl"
if os.path.exists(save_path):
    cr_throughput_result_map = load(save_path)
else:
    cr_throughput_result_map = {}

result_queue = multiprocessing.Queue()

for hash, pair in tqdm(setting_map.items(), total=len(setting_map), desc="Evaluating cr_throughput"):
    if hash in cr_throughput_result_map:
        logger.info(f"Skip: {hash}")
        continue
    ctx_len, config = pair
    process = multiprocessing.Process(
        target=run_cr_evaluation,
        args=(config, ctx_len, logger, result_queue)
    )
    process.start()
    logger.info("evaluation process started")
    process.join()
    logger.info("evaluation process joined")
    cr_throughput = result_queue.get()
    cr_throughput_result_map[hash] = cr_throughput
    save(cr_throughput_result_map, save_path)
    logger.info(f"\n{hash}\n{config}\n{cr_throughput}\n\n")
    # cr_throughput = cr_throughput_evaluation(config, ctx_len, False, logger)
