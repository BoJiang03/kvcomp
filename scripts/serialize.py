from utils.compute import QuantMode
from utils.config import KVCompCacheConfig

config = KVCompCacheConfig(
    model_name="meta-llama/Llama-2-13b-hf",
    k_quant_mode=QuantMode.BlockQuant,
    v_quant_mode=QuantMode.TokenQuant,
    enable_quant=True,
    k_block_size=64,
    v_block_size=128,
    k_recent_size=128,
    v_recent_size=128,
    k_quant_scale_rel=0.04,
    v_quant_scale_rel=0.04
)

str_ = config.__str__()
restore_config = KVCompCacheConfig.from_str(str_)
print(restore_config)