#! /usr/bin/env python
from utils.compute import QuantMode, print_quant_setting
from utils.config import KVCompCacheConfig
from utils.util import register_notify, get_logger, block_other_logger
import numpy as np
from evaluation.evaluation import cr_evaluation, coqa_accuracy_evaluation, k_cr_evaluation, k_standalone_cr_evaluation, \
    v_standalone_cr_evaluation
register_notify()

logger = get_logger(__file__)
# block_other_logger(logger)

print_quant_setting()
block_size = 64
accuracy_threshold = 0.60

STEP_LEN_MAP = {
    # QuantMode.BlockQuant: 0.01,
    QuantMode.ChannelQuant: 0.05,
    # QuantMode.TokenQuant: 0.02,
}

STEP_END_MAP = {
    # QuantMode.BlockQuant: 0.15,
    QuantMode.ChannelQuant: 0.40,
    # QuantMode.TokenQuant: 0.20,
}

quant_scale_rels = []
accuracies = []
quant_crs = []
huffman_crs = []
fse_crs = []

is_pre_rope = False
quant_model = QuantMode.ChannelQuant
logger.info(f"======= is_pre_rope: {is_pre_rope}, quant_model: {quant_model.value} =======")
base_k_quant_scale_rel = 0.01
base_v_quant_scale_rel = 0.01
logger.info(f"======= begin search for best k_quant_scale_rel ========")
for k_quant_scale_rel in np.arange(0.01, STEP_END_MAP[quant_model], STEP_LEN_MAP[quant_model]).tolist():
    config = KVCompCacheConfig(
        enable_quant=True,
        model_name="meta-llama/Llama-2-7b-hf",
        enable_pre_rope=is_pre_rope,
        k_quant_mode=quant_model,
        v_quant_mode=QuantMode.TokenQuant,
        k_block_size=block_size,
        v_block_size=block_size,
        k_recent_size=block_size * 2,
        v_recent_size=block_size * 2,
        k_quant_scale_rel=k_quant_scale_rel,
        v_quant_scale_rel=base_v_quant_scale_rel,
        device="cuda:1",
    )
    logger.info(config)
    accuracy = coqa_accuracy_evaluation(config)
    quant_cr, huffman_cr, fse_cr = k_standalone_cr_evaluation(config)
    logger.info(f"coqa accuracy: {accuracy}")
    logger.info(f"quant cr: {quant_cr}, huffman cr: {huffman_cr}")
    quant_scale_rels.append(k_quant_scale_rel)
    accuracies.append(accuracy)
    quant_crs.append(quant_cr)
    huffman_crs.append(huffman_cr)
    fse_crs.append(fse_cr)

    logger.info(f"quant_scale_rels: {quant_scale_rels}")
    logger.info(f"accuracies: {accuracies}")
    logger.info(f"quant_crs: {quant_crs}")
    logger.info(f"huffman_crs: {huffman_crs}")
    logger.info(f"fse_crs: {fse_crs}")
    if accuracy < accuracy_threshold:
        logger.info(f"best k_quant_scale_rel: {k_quant_scale_rel}")
        logger.info(f"quant cr: {quant_cr}, huffman cr: {huffman_cr}")
        break

logger.info("===== final data =====")
logger.info(f"quant_scale_rels: {quant_scale_rels}")
logger.info(f"accuracies: {accuracies}")
logger.info(f"quant_crs: {quant_crs}")
logger.info(f"huffman_crs: {huffman_crs}")
logger.info(f"fse_crs: {fse_crs}")