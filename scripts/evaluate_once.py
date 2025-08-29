#! /usr/bin/env python
from evaluation.evaluation import coqa_accuracy_evaluation, accuracy_evaluation, crs_throughputs_evaluation
from utils.compute import QuantMode, print_quant_setting
from utils.config import KVCompCacheConfig
from utils.util import get_logger, block_other_logger
logger = get_logger(__file__)
block_other_logger(logger)
# register_notify()

print_quant_setting(logger)
config = KVCompCacheConfig(
    model_name="meta-llama/Llama-2-7b-hf",
    # model_name="meta-llama/Llama-2-13b-hf",
    k_quant_mode=QuantMode.BlockQuant,
    v_quant_mode=QuantMode.TokenQuant,
    enable_quant=True,
    k_block_size=64,
    v_block_size=128,
    k_recent_size=128,
    v_recent_size=128 + 128,
    k_quant_scale_rel=0.04,
    v_quant_scale_rel=0.04
)

logger.info(config)
# accuracy = coqa_accuracy_evaluation(config, logger)
accuracy = accuracy_evaluation(config, "winogrande", logger)
logger.info(f"accuracy: {accuracy}")

# rts = crs_throughputs_evaluation(config, benchmark="gsm8k", logger=logger, enable_save=False, collect_round=1)
# print(rts)