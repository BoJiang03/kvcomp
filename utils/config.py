from utils.compute import QuantMode

class ExtractCacheConfig:
    def __init__(self, collect_round: int):
        self.collect_round = collect_round
        self.key_caches = {}
        self.value_caches = {}

    def size(self):
        total_size = 0
        for round_ks in self.key_caches.values():
            for k in round_ks.values():
                total_size += k.numel() * 16
        
        for round_vs in self.value_caches.values():
            for v in round_vs.values():
                total_size += v.numel() * 16

        return total_size

class KVCompCacheConfig:
    def __init__(
            self,
            model_name: str,
            k_quant_mode: QuantMode,
            v_quant_mode: QuantMode,
            k_block_size: int,
            v_block_size: int,
            k_recent_size: int,
            v_recent_size: int,
            k_quant_scale_rel: float,
            v_quant_scale_rel: float,
            enable_quant: bool = True,
            enable_pre_rope: bool = False
    ):
        self.enable_quant: bool = enable_quant
        self.model_name: str = model_name
        self.enable_pre_rope: bool = enable_pre_rope
        self.k_quant_mode: QuantMode = k_quant_mode
        self.v_quant_mode: QuantMode = v_quant_mode
        self.k_block_size: int = k_block_size
        self.v_block_size: int = v_block_size
        self.k_recent_size: int = k_recent_size
        self.v_recent_size: int = v_recent_size
        self.k_quant_scale_rel: float = k_quant_scale_rel
        self.v_quant_scale_rel: float = v_quant_scale_rel

    # to string print
    def __str__(self):
        # parse class as json
        json_ = {
            "enable_quant": self.enable_quant,
            "enable_pre_rope": self.enable_pre_rope,
            "model_name": self.model_name,
        }
        if self.enable_quant:
            json_["k_quant_mode"] = self.k_quant_mode.value
            json_["v_quant_mode"] = self.v_quant_mode.value
            json_["k_block_size"] = self.k_block_size
            json_["v_block_size"] = self.v_block_size
            json_["k_recent_size"] = self.k_recent_size
            json_["v_recent_size"] = self.v_recent_size
            json_["k_quant_scale_rel"] = self.k_quant_scale_rel
            json_["v_quant_scale_rel"] = self.v_quant_scale_rel

        return str(json_)

    @staticmethod
    def from_str(json_str: str):
        # parse json string to class
        json_ = eval(json_str)
        if json_["enable_quant"]:
            k_quant_mode = QuantMode(json_["k_quant_mode"])
            v_quant_mode = QuantMode(json_["v_quant_mode"])
            k_block_size = json_["k_block_size"]
            v_block_size = json_["v_block_size"]
            k_recent_size = json_["k_recent_size"]
            v_recent_size = json_["v_recent_size"]
            k_quant_scale_rel = json_["k_quant_scale_rel"]
            v_quant_scale_rel = json_["v_quant_scale_rel"]
        else:
            k_quant_mode = None
            v_quant_mode = None
            k_block_size = None
            v_block_size = None
            k_recent_size = None
            v_recent_size = None
            k_quant_scale_rel = None
            v_quant_scale_rel = None

        return KVCompCacheConfig(
            model_name=json_["model_name"],
            enable_quant=json_["enable_quant"],
            enable_pre_rope=json_["enable_pre_rope"],
            k_quant_mode=k_quant_mode,
            v_quant_mode=v_quant_mode,
            k_block_size=k_block_size,
            v_block_size=v_block_size,
            k_recent_size=k_recent_size,
            v_recent_size=v_recent_size,
            k_quant_scale_rel=k_quant_scale_rel,
            v_quant_scale_rel=v_quant_scale_rel,
        )

    def __eq__(self, other):
        if not isinstance(other, KVCompCacheConfig):
            return False
        if self.enable_quant != other.enable_quant:
            return False
        if self.model_name != other.model_name:
            return False
        if self.enable_pre_rope != other.enable_pre_rope:
            return False
        if self.k_quant_mode != other.k_quant_mode:
            return False
        if self.v_quant_mode != other.v_quant_mode:
            return False
        if self.k_block_size != other.k_block_size:
            return False
        if self.v_block_size != other.v_block_size:
            return False
        if self.k_recent_size != other.k_recent_size:
            return False
        if self.v_recent_size != other.v_recent_size:
            return False
        if self.k_quant_scale_rel != other.k_quant_scale_rel:
            return False
        if self.v_quant_scale_rel != other.v_quant_scale_rel:
            return False
        return True

    def __hash__(self):
        return hash((self.enable_quant, self.model_name, self.enable_pre_rope,
                     self.k_quant_mode, self.v_quant_mode, self.k_block_size,
                     self.v_block_size, self.k_recent_size, self.v_recent_size,
                     self.k_quant_scale_rel, self.v_quant_scale_rel))
    
    # for print
    def __repr__(self):
        return self.__str__()
