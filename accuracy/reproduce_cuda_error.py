import torch
import kvcomp_cuda

from models.cache.kvcomp_encode import KVCompCtx

# encoded_data = torch.load("./encoded_data.pt")
# block_infos = torch.load("./block_infos.pt")
# thread_infos = torch.load("./thread_infos.pt")
# decode_codebook = torch.load("./decode_codebook.pt")
# decoded_data = torch.load("./decoded_data.pt")
# head_num = 40
# head_dim = 128

# kvcomp_cuda.v_entropy_decode_cuda_export(
#                 encoded_data,
#                 block_infos,
#                 thread_infos,
#                 decode_codebook,
#                 decoded_data,
#                 head_num,
#                 head_dim
#             )
error_data = torch.load("./error_temp_v.pt")

whole_len = error_data.shape[0]
start_len = 952
prefill_v = error_data[:start_len].contiguous()
# prefill_v_whole = torch.load("./error_temp_v_prefill.pt")
ctx = KVCompCtx(False, 1, 40, 128, 0.04)
ctx.store(prefill_v)
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
for i in range(start_len, whole_len):
    decode_v = error_data[i:i+1].contiguous()
    ctx.store(decode_v)
    decompressed = ctx.get()
decompressed = ctx.get()