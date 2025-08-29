import torch
from kv_com_ctx import KVComCtx

k_t = torch.load("../../bin/5.pt").permute(2, 0, 1, 3).contiguous().to(torch.float16)

seq_len = 1024
batch_size = 1
head_num = 32
head_dim = 128

ctx = KVComCtx(
    is_k=True,
    batch_size=batch_size,
    head_num=head_num,
    head_dim=head_dim,
    device=k_t.device,
    quant_scale_rel=0.03
)

to_store = k_t[:seq_len, ...].contiguous()

quant_ints = ctx.store(to_store)
decoded_quant_ints = ctx.get_decode_quant_ints()

data_error = ctx.get()
data_error = ctx.get_decode_quant_ints()
data_error = ctx.get()
data_error = ctx.get()
data_error = ctx.get()
data_error = ctx.get()