#! /usr/bin/env python
from transformers import AutoTokenizer
from models.phi import Phi3ForCausalLM
from models.tcom_cache import KVComCacheConfigStatic
from utils.lm_eval_warp import LMEvalWrapper
from lm_eval import evaluator
from utils.util import notify_user, get_logger
import numpy as np

logger = get_logger(__file__)

def evaluation(block_size, k_quant_scale_rel, v_quant_scale_rel, device = "cuda:1"):
    model_name = "microsoft/Phi-3.5-mini-instruct"
    KVComCacheConfigStatic.BLOCK_SIZE = block_size
    KVComCacheConfigStatic.V_QUANT_SCALE = v_quant_scale_rel
    KVComCacheConfigStatic.K_QUANT_SCALE = k_quant_scale_rel
    logger.info(f"TComCacheConfig: {KVComCacheConfigStatic.to_string()}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = Phi3ForCausalLM.from_pretrained(model_name)
    # model = LlamaForCausalLM.from_pretrained(args.model_name,)
    # model = AutoModelForCausalLM.from_pretrained(args.model_name,)

    batch_size = 1
    lm_eval_warp = LMEvalWrapper(model, tokenizer, batch_size, device)
    results = evaluator.simple_evaluate(
        model=lm_eval_warp,
        tasks=["gsm8k"],
        # num_fewshot=0,  # Number of few-shot examples
        batch_size=batch_size,
        device=device,
    )

    return results

# MMLU: 5-shot
# IFEval: 0-shot
# GPQA: 0-shot
# HumanEval: 0-shot
# GSM8K: 8-shot
# MATH: 4-shot

# quant_scale_settings = []
# accuracies = []

block_size = 64
for k_quant_scale_rel in np.arange(0.01, 0.08, 0.01).tolist():
    for v_quant_scale_rel in np.arange(0.01, 0.08, 0.01).tolist():
        results = evaluation(block_size, k_quant_scale_rel, v_quant_scale_rel)
        # accu = results['results']['coqa']['em,none']
        # logger.info(f"accuracy: {accu}")
        # quant_scale_settings.append((k_quant_scale_rel, v_quant_scale_rel))
        # accuracies.append(accu)
        logger.info(f"quant_scale_settings: {k_quant_scale_rel, v_quant_scale_rel}")
        logger.info(f"results: {results}")

# logger.info(f"quant_scale_settings: {quant_scale_settings}")
# logger.info(f"accuracies: {accuracies}")

notify_user()