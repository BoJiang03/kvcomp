import torch
from models.kvcomp_encode import KVCompCtx

def do_transformers_mat_vec_mul_original_layout(k: torch.Tensor, q: torch.Tensor):
    q_o_t = q.unsqueeze(2)
    k_o_t = k.permute(1,2,0,3)
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    k_o_t = k_o_t.transpose(2,3)
    torch.cuda.synchronize()

    start_time.record()
    C_target = torch.matmul(q_o_t,k_o_t)
    end_time.record()
    torch.cuda.synchronize()

    C_target = C_target * 1.0
    C_target = C_target.squeeze(2).permute(2,0,1)
    return C_target, start_time.elapsed_time(end_time)

def do_transformers_mat_vec_mul(k: torch.Tensor, q: torch.Tensor):
    q_o_t = q.unsqueeze(2).contiguous()
    k_o_t = k.permute(1,2,0,3).contiguous()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    k_o_t = k_o_t.transpose(2,3)
    torch.cuda.synchronize()

    start_time.record()
    C_target = torch.matmul(q_o_t,k_o_t)
    end_time.record()
    torch.cuda.synchronize()

    C_target = C_target * 1.0
    C_target = C_target.squeeze(2).permute(2,0,1)
    return C_target, start_time.elapsed_time(end_time)

def pytorch_mat_vec_mul(k: torch.Tensor, q: torch.Tensor):
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()
    C_target = k*q
    end_time.record()
    torch.cuda.synchronize()
    C_target = C_target.sum(dim=-1)

    return C_target, start_time.elapsed_time(end_time)

def check_decode_correctness(ctx: KVCompCtx, quant_ints: torch.Tensor):
    decoded_quant_ints = ctx.get_decode_quant_ints()
    # Find positions where tensors differ
    diff_positions = (decoded_quant_ints != quant_ints).nonzero(as_tuple=True)

    if len(diff_positions[0]) > 0:
        # Get values at those positions
        original_values = quant_ints[diff_positions]
        decoded_values = decoded_quant_ints[diff_positions]

        # Print some sample differences
        num_samples = len(diff_positions[0])
        print(f"Found {len(diff_positions[0])} differences. Showing first {num_samples}:")

        for i in range(num_samples):
            pos = tuple(p[i].item() for p in diff_positions)
            # print(f"Position {pos}: Original={original_values[i].item()}, Decoded={decoded_values[i].item()}")

        assert False, f"quant_int != decoded_quant_int, decode error in {len(diff_positions[0])} positions"
    else:
        print("Decoding is correct - tensors match exactly")

def twice_store(ctx: KVCompCtx, k_t: torch.Tensor, seq_len: int, batch_size: int, head_num: int, head_dim: int):
    to_store1 = k_t[:seq_len, :batch_size, :head_num, :head_dim].contiguous()
    to_store2 = k_t[:seq_len, :batch_size, :head_num, :head_dim].contiguous()
    ctx.build_codebook([to_store1, to_store2])
    quant_int1 = ctx.store(to_store1)
    quant_int2 = ctx.store(to_store2)
    quant_int = torch.cat([quant_int1, quant_int2], dim=0)
    check_decode_correctness(ctx, quant_int)

seq_len = 1024
batch_size = 1
head_num = 40
head_dim = 128

k_t = torch.load("../bin/temp_k.pt").to("cuda:1").contiguous().to(torch.float16)
k_t = torch.cat([k_t[:1024]] * 8, dim = 0)

q_t = torch.empty(
    (batch_size, head_num, head_dim),
    dtype=torch.float16,
    device=k_t.device
).uniform_(-1, 1)

ctx = KVCompCtx(
    is_k=True,
    batch_size=batch_size,
    head_num=head_num,
    head_dim=head_dim,
    quant_scale_rel=0.03
)

twice_store(ctx, k_t, seq_len, batch_size, head_num, head_dim)

decompressed_k = ctx.get().view(-1, batch_size, head_num, head_dim)
print("K shape:", decompressed_k.shape)
print("Q shape:", q_t.shape)
for _ in range(10):
    C_pytorch_our_layout, pytorch_mat_vec_mul_time_our_layout = do_transformers_mat_vec_mul_original_layout(decompressed_k, q_t)
    print(f"cublas mat vec mul time: {pytorch_mat_vec_mul_time_our_layout} ms")
C_pytorch_our_layout = C_pytorch_our_layout.permute(1,2,0)
for _ in range(10):
    C_ours = ctx.mat_vec_mul(q_t)

delta = (C_pytorch_our_layout - C_ours).abs()
assert (delta < C_pytorch_our_layout.abs().max() * 0.01).all(), "Error too large"
print(f"cr: {ctx.get_cr()}")
