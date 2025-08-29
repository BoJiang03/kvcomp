import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.distributed.pipelining import pipeline, SplitPoint
from transformers import LlamaForCausalLM
model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
    model_name, low_cpu_mem_usage=True, attn_implementation="sdpa"
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)
prompts = ("I would like to", "I really like to")  # bs = 2, sending 2 per process
tokenizer.pad_token = tokenizer.eos_token
inputs = tokenizer(prompts, return_tensors="pt", padding=True)

prompts = ("I would like to", "I really like to", "The weather is pretty")  # bs = 3
inputs = tokenizer(prompts, return_tensors="pt", padding=True)
inputs = inputs.to(0)
with torch.no_grad():
    output = model(**inputs)
