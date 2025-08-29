#! /usr/bin/env python
from utils.compute import QuantMode, print_quant_setting
from utils.config import TComCacheConfig
from utils.util import notify_user, get_logger
import numpy as np
from evaluation.evaluation import cr_evaluation

logger = get_logger(__file__)

# MMLU: 5-shot
# IFEval: 0-shot
# GPQA: 0-shot
# HumanEval: 0-shot
# GSM8K: 8-shot
# MATH: 4-shot

print_quant_setting()

block_size = 64
for k_quant_scale_rel in np.arange(0.01, 0.08, 0.01).tolist():
    for v_quant_scale_rel in np.arange(0.01, 0.08, 0.01).tolist():
        config = TComCacheConfig(
            enable_quant=True,
            model_name="microsoft/Phi-3.5-mini-instruct",
            enable_pre_rope=True,
            k_quant_mode=QuantMode.BlockQuant,
            v_quant_mode=QuantMode.BlockQuant,
            k_block_size=block_size,
            v_block_size=block_size,
            k_recent_size=block_size * 2,
            v_recent_size=block_size * 2,
            k_quant_scale_rel=k_quant_scale_rel,
            v_quant_scale_rel=v_quant_scale_rel,
            device="cuda:0",
        )
        crs = cr_evaluation(config)
        logger.info(crs)

notify_user()