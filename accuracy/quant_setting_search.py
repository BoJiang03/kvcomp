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
accuracy_threshold = 0.6

STEP_LEN_MAP = {
    QuantMode.BlockQuant: 0.01,
    QuantMode.ChannelQuant: 0.03,
    QuantMode.TokenQuant: 0.02,
}

STEP_END_MAP = {
    QuantMode.BlockQuant: 0.15,
    QuantMode.ChannelQuant: 0.40,
    QuantMode.TokenQuant: 0.20,
}

for is_pre_rope in [True, False]:
    for quant_model in [QuantMode.BlockQuant,QuantMode.ChannelQuant, QuantMode.TokenQuant]:
        logger.info(f"======= is_pre_rope: {is_pre_rope}, quant_model: {quant_model.value} =======")
        base_k_quant_scale_rel = 0.01
        base_v_quant_scale_rel = 0.01
        logger.info(f"======= begin search for best k_quant_scale_rel ========")
        for k_quant_scale_rel in np.arange(0.01, STEP_END_MAP[quant_model], STEP_LEN_MAP[quant_model]).tolist():
            config = KVCompCacheConfig(
                enable_quant=True,
                model_name="microsoft/Phi-3.5-mini-instruct",
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
            quant_cr, huffman_cr = k_standalone_cr_evaluation(config)
            logger.info(f"coqa accuracy: {accuracy}")
            logger.info(f"quant cr: {quant_cr}, huffman cr: {huffman_cr}")
            if accuracy < accuracy_threshold:
                logger.info(f"best k_quant_scale_rel: {k_quant_scale_rel}")
                logger.info(f"quant cr: {quant_cr}, huffman cr: {huffman_cr}")
                break

        logger.info(f"======= begin search for best v_quant_scale_rel ========")
        for v_quant_scale_rel in np.arange(0.01, STEP_END_MAP[quant_model], STEP_LEN_MAP[quant_model]).tolist():
            config = KVCompCacheConfig(
                enable_quant=True,
                model_name="microsoft/Phi-3.5-mini-instruct",
                enable_pre_rope=is_pre_rope,
                k_quant_mode=QuantMode.ChannelQuant,
                v_quant_mode=quant_model,
                k_block_size=block_size,
                v_block_size=block_size,
                k_recent_size=block_size * 2,
                v_recent_size=block_size * 2,
                k_quant_scale_rel=base_k_quant_scale_rel,
                v_quant_scale_rel=v_quant_scale_rel,
                device="cuda:1",
            )
            logger.info(config)
            accuracy = coqa_accuracy_evaluation(config)
            quant_cr, huffman_cr = v_standalone_cr_evaluation(config)
            logger.info(f"coqa accuracy: {accuracy}")
            logger.info(f"quant cr: {quant_cr}, huffman cr: {huffman_cr}")
            if accuracy < accuracy_threshold:
                logger.info(f"best v_quant_scale_rel: {v_quant_scale_rel}")
                logger.info(f"quant cr: {quant_cr}, huffman cr: {huffman_cr}")
                break
