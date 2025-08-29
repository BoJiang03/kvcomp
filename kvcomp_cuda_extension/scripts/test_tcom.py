import torch
from tcom_cuda import KVComCtx

k_t = torch.load("../../bin/5.pt").permute(2, 0, 1, 3).contiguous().to(torch.float16)

seq_len = 1
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

B = torch.empty(
    (batch_size, head_num, head_dim),
    dtype=torch.float16,
    device=ctx.device
).uniform_(-1, 1)

data_error = ctx.get()

start_time = torch.cuda.Event(enable_timing=True)
end_time = torch.cuda.Event(enable_timing=True)

start_time.record()
C_target = (data_error * B)
end_time.record()
torch.cuda.synchronize()

print("pytorch kernel time:", start_time.elapsed_time(end_time), "ms")

C_target = C_target.sum(-1)
# dat = ctx.get_decode_quant_ints()
C_ours = ctx.mat_vec_mul(B)
delta = (C_target - C_ours).abs()

assert (delta < C_target.abs().max() * 0.01).all(), "Error too large"
