from sympy import print_tree

from evaluation.cr_throughput_eval import crs_throughputs_evaluation_with_data
from models.llama import LlamaForCausalLM
from models.cache.kvcomp_quant import (KVCompCacheConfigStatic, KVCompCachePytorchQuant)
from models.mistral import MistralForCausalLM
from models.phi3 import Phi3ForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.compute import safe_cat, QuantMode
from utils.lm_eval_warp import LMEvalWrapper
from lm_eval import evaluator
from utils.config import KVCompCacheConfig, ExtractCacheConfig
from utils.util import get_logger, JumpOutException
import os
import torch
from datasets import load_dataset

MODEL_CLASS_MAP = {
    "meta-llama/Llama-2-7b-hf": LlamaForCausalLM,
    "meta-llama/Llama-2-13b-hf": LlamaForCausalLM,
    "meta-llama/Llama-3.1-8B": LlamaForCausalLM,
    "meta-llama/Meta-Llama-3-8B": LlamaForCausalLM,
    "mistralai/Ministral-8B-Instruct-2410": MistralForCausalLM,
    "mistralai/Mistral-Small-24B-Base-2501": MistralForCausalLM,
    "mistralai/Mistral-Small-Instruct-2409": MistralForCausalLM,
    # "google/gemma-3-4b-it": Gemma3ForCausalLM, # multimodal model, not support for now
    # "google/gemma-3-12b-it": AutoModelForCausalLM, # multimodal model, not support for now
    # "Qwen/Qwen2.5-7B-Instruct": Qwen2ForCausalLM, # coqa not working, dont know why
    # "Qwen/Qwen2.5-14B-Instruct": AutoModelForCausalLM, # coqa not working, dont know why
    "microsoft/phi-4": Phi3ForCausalLM,
    "microsoft/Phi-4-mini-instruct": Phi3ForCausalLM
}

def accuracy_evaluation(
        config: KVCompCacheConfig,
        benchmark: str,
        logger
):
    logger.info(f"\n{benchmark}:\n{config}")
    KVCompCacheConfigStatic.config = config
    tokenizer = AutoTokenizer.from_pretrained(KVCompCacheConfigStatic.config.model_name)
    model_class = MODEL_CLASS_MAP.get(config.model_name)
    if model_class is None:
        raise ValueError(f"Model class not found for {config.model_name}")
    model = model_class.from_pretrained(
        KVCompCacheConfigStatic.config.model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    batch_size = 1

    lm_eval_warp = LMEvalWrapper(model, tokenizer, batch_size)

    results = evaluator.simple_evaluate(
        model=lm_eval_warp,
        tasks=[benchmark],
        # num_fewshot=0,  # Number of few-shot examples
        batch_size=batch_size,
        # device=_device,
    )

    KVCompCacheConfigStatic.config = None
    return results['results']

def coqa_accuracy_evaluation(
        config: KVCompCacheConfig,
        logger
):
    res = accuracy_evaluation(
        config,
        benchmark="coqa",
        logger=logger
    )
    return res['coqa']['em,none']


# def k_cr_evaluation(config: TComCacheConfig, ks: Iterable[torch.tensor]) -> Tuple[float, float, float]:
#     origin_size = 0
#     quant_compress_size = 0
#     huffman_compress_size = 0
#     fse_compress_size = 0
#     for k in tqdm(ks, desc="Compressing k"):
#         block_num = k.shape[2] // config.k_block_size
#         to_compress = k[..., :block_num * config.k_block_size, :]
#         origin_size += to_compress.numel() * 16
#         quant_int, min_quant, quant_scale = quant_ints(
#             to_compress,
#             config.k_block_size,
#             config.k_quant_scale_rel,
#             config.k_quant_mode
#         )
#         quant_int = quant_int.to(torch.int8)
#         state_num = quant_int.max() - quant_int.min() + 1
#         num_bits = math.ceil(math.log2(state_num))
#         quant_compress_size += to_compress.numel() * num_bits
#         codes, _ = huffman_encode(quant_int)
#         huffman_compress_size += codes.numel()
#         fse_encoded = pyfse.compress(bytes(quant_int.to("cpu").numpy()))
#         fse_compress_size += len(fse_encoded) * 8
#     return origin_size/ quant_compress_size, origin_size / huffman_compress_size, origin_size / fse_compress_size

# def v_cr_evaluation(config: TComCacheConfig, vs: Iterable[torch.tensor]) -> Tuple[float, float, float]:
#     origin_size = 0
#     quant_compress_size = 0
#     huffman_compress_size = 0
#     fse_compress_size = 0
#     for v in tqdm(vs, desc="Compressing v"):
#         block_num = v.shape[2] // config.v_block_size
#         to_compress = v[..., :block_num * config.v_block_size, :]
#         origin_size += to_compress.numel() * 16
#         quant_int, min_quant, quant_scale = quant_ints(
#             to_compress,
#             config.v_block_size,
#             config.v_quant_scale_rel,
#             config.v_quant_mode
#         )
#         quant_int = quant_int.to(torch.int8)
#         state_num = quant_int.max() - quant_int.min() + 1
#         num_bits = math.ceil(math.log2(state_num))
#         quant_compress_size += to_compress.numel() * num_bits
#         codes, _ = huffman_encode(quant_int)
#         huffman_compress_size += codes.numel()
#         fse_encoded = pyfse.compress(bytes(quant_int.to("cpu").numpy()))
#         fse_compress_size += len(fse_encoded) * 8
#
#     return origin_size / quant_compress_size, origin_size / huffman_compress_size, origin_size / fse_compress_size

def get_collected_data(
        logger,
    model_name = "meta-llama/Llama-2-13b-hf"
):
    KVCompCacheConfigStatic.config = KVCompCacheConfig(
        enable_quant=False,
        model_name=model_name,
        enable_pre_rope=False,
        k_quant_mode=QuantMode.BlockQuant,
        v_quant_mode=QuantMode.TokenQuant,
        k_block_size=64,
        v_block_size=128,
        k_recent_size=128,
        v_recent_size=256,
        k_quant_scale_rel=0.01,
        v_quant_scale_rel=0.01
    )
    KVCompCacheConfigStatic.extract_cache = ExtractCacheConfig(collect_round=1)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_class = MODEL_CLASS_MAP.get(model_name)
    if model_class is None:
        raise ValueError(f"Model class not found for {model_name}")
    model = model_class.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    batch_size = 1
    lm_eval_warp = LMEvalWrapper(model, tokenizer, batch_size)
    try:
        evaluator.simple_evaluate(
            model=lm_eval_warp,
            tasks=["gsm8k"],
            batch_size=batch_size,
        )
    except JumpOutException as e:
        logger.info(e.message)

    rt = KVCompCacheConfigStatic.extract_cache
    KVCompCacheConfigStatic.config = None
    KVCompCacheConfigStatic.extract_cache = None

    return rt

def get_ctx_len_text_from_wikitext_103_v1(
        ctx_len: int,
        tokenizer
):
    dataset = load_dataset("wikitext", "wikitext-103-v1")
    rt_text = ""
    for i in range(len(dataset['train'])):
        text = dataset['train'][i]['text']
        rt_text += text
        tokens = tokenizer(rt_text, return_tensors="pt")['input_ids']
        if tokens.shape[1] > ctx_len:
            break
    return tokenizer(rt_text, return_tensors="pt", truncation=True, max_length=ctx_len)

def cr_throughput_evaluation(
        config: KVCompCacheConfig,
        ctx_len: int,
        enable_save: bool,
        logger,
        collect_round: int = 1,
):
    logger.info(f"ctx_len: {ctx_len}")
    logger.info(config)
    logger.info(f"Created a temp config for cache collection and set its enable_quant to False.")
    KVCompCacheConfigStatic.config = KVCompCacheConfig(
        enable_quant=False,
        model_name=config.model_name,
        enable_pre_rope=config.enable_pre_rope,
        k_quant_mode=config.k_quant_mode,
        v_quant_mode=config.v_quant_mode,
        k_block_size=config.k_block_size,
        v_block_size=config.v_block_size,
        k_recent_size=config.k_recent_size,
        v_recent_size=config.v_recent_size,
        k_quant_scale_rel=config.k_quant_scale_rel,
        v_quant_scale_rel=config.v_quant_scale_rel
    )

    KVCompCacheConfigStatic.extract_cache = ExtractCacheConfig(collect_round)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model_class = MODEL_CLASS_MAP.get(config.model_name)
    if model_class is None:
        raise ValueError(f"Model class not found for {config.model_name}")
    model = model_class.from_pretrained(
        KVCompCacheConfigStatic.config.model_name,
        torch_dtype="float16",
        device_map="auto"
    )
    # batch_size = 1
    # lm_eval_warp = LMEvalWrapper(model, tokenizer, batch_size)
    inputs = get_ctx_len_text_from_wikitext_103_v1(ctx_len, tokenizer)

    with torch.no_grad():
        _ = model(**inputs)
    KVCompCachePytorchQuant.round_ = 0

    if enable_save:
        save_extract_cache(config.model_name, ctx_len, KVCompCacheConfigStatic.extract_cache, "./dumped_cache")

    rts = []

    logger.info(f"Cache Size {KVCompCacheConfigStatic.extract_cache.size() / 8 / 1024 / 1024/ 1024} GB")

    for round_i in range(collect_round):
        key_caches = list(KVCompCacheConfigStatic.extract_cache.key_caches[round_i].values())
        value_caches = list(KVCompCacheConfigStatic.extract_cache.value_caches[round_i].values())
        rts.append(crs_throughputs_evaluation_with_data(config, key_caches, value_caches))

    KVCompCacheConfigStatic.config = None
    KVCompCacheConfigStatic.extract_cache = None
    return rts

def save_extract_cache(
        model_name: str,
        ctx_len: int,
        extract_cache: ExtractCacheConfig,
        root_dir: str
):
    # check whether the directory exists
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    for round_i in range(extract_cache.collect_round):
        save_dir = os.path.join(root_dir, model_name, str(ctx_len), str(round_i))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        k_dir = os.path.join(save_dir, "k")
        v_dir = os.path.join(save_dir, "v")
        if not os.path.exists(k_dir):
            os.makedirs(k_dir)
        if not os.path.exists(v_dir):
            os.makedirs(v_dir)

        key_caches = KVCompCacheConfigStatic.extract_cache.key_caches[round_i].values()
        value_caches = KVCompCacheConfigStatic.extract_cache.value_caches[round_i].values()
        for i, key_cache in enumerate(key_caches):
            torch.save(key_cache, os.path.join(k_dir, f"{i}.pt"))
        for i, value_cache in enumerate(value_caches):
            torch.save(value_cache, os.path.join(v_dir, f"{i}.pt"))
