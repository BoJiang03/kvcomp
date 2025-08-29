import torch
import kvcomp_cuda

from models.cache.kvcomp_encode import KVCompCtx

error_data = torch.load("./k_error_prefill.pt")

ctx = KVCompCtx(False, 1, 40, 128, 0.04)
ctx.store(error_data)
# ctx_w = KVCompCtx(False, 1, 40, 128, 0.04)
# ctx_w.store(prefill_v_whole)
decompressed = ctx.get()

# quant_ints_real = torch.load("./quant_ints_real.pt")
# quant_ints_reproduce = torch.load("./quant_ints_reproduce.pt")
#
# s_r, e_r, d_r = kvcomp_cuda.build_codebook([quant_ints_reproduce])
# s_re, e_re, d_re = kvcomp_cuda.build_codebook([quant_ints_reproduce])
# s_re1, e_re1, d_re1 = kvcomp_cuda.build_codebook([quant_ints_reproduce])
# print("ok")
# for i in range(start_len, whole_len):
#     decode_v = error_data[i:i+1].contiguous()
#     ctx.store(decode_v)
#     decompressed = ctx.get()
# decompressed = ctx.get()