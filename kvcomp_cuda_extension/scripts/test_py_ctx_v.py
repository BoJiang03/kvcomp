import torch
from kv_com_ctx import KVComCtx

def do_transformers_mat_vec_mul_original_layout(attn_weights_t: torch.Tensor, v: torch.Tensor):
    attn_weights_t = attn_weights_t.unsqueeze(2)
    v = v.permute(1,2,0,3)

    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_time.record()
    C_target = torch.matmul(attn_weights_t,v)
    end_time.record()
    torch.cuda.synchronize()

    C_target = C_target.transpose(1,2)
    return C_target, start_time.elapsed_time(end_time)  # Return the result and the elapsed time in milliseconds

def check_decode_correctness(ctx: KVComCtx, quant_ints: torch.Tensor):
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

def twice_store(ctx: KVComCtx, v_t: torch.Tensor, seq_len: int, batch_size: int, head_num: int, head_dim: int):
    to_store1 = v_t[:seq_len, :batch_size, :head_num, :head_dim].contiguous()
    to_store2 = v_t[:seq_len, :batch_size, :head_num, :head_dim].contiguous()
    ctx.build_codebook([to_store1, to_store2])
    quant_int1 = ctx.store(to_store1)
    quant_int2 = ctx.store(to_store2)
    quant_int = torch.cat([quant_int1, quant_int2], dim=0)
    check_decode_correctness(ctx, quant_int)

def check_error(a: torch.Tensor, b: torch.Tensor, percentage: float):
    delta = (a - b).abs()
    assert (delta < a.abs().max() * percentage).all(), "Error too large"

seq_len = 8192
batch_size = 1
head_num = 32
head_dim = 128

v_t = torch.load("../../bin/v_5.pt").to("cuda:1").permute(2, 0, 1, 3).contiguous().to(torch.float16)
v_t = torch.cat([v_t[:1024]] * 8, dim = 0)

attn_weights_t = torch.randn(
    (batch_size, head_num, seq_len * 2),
    dtype=torch.float16,
    device=v_t.device
) * 2.5
attn_weights_t = torch.softmax(attn_weights_t, dim=-1)
# attn_weights_t[...] = 1
ctx = KVComCtx(
    is_k=False,
    batch_size=batch_size,
    head_num=head_num,
    head_dim=head_dim,
    device=v_t.device,
    quant_scale_rel=0.07
)

twice_store(ctx, v_t, seq_len, batch_size, head_num, head_dim)

decompressed_v = ctx.get().view(-1, batch_size, head_num, head_dim)
print("V shape: ", decompressed_v.shape)
print("Attn weights shape: ", attn_weights_t.shape)
for _ in range(10):
    C_ours = ctx.mat_vec_mul(attn_weights_t)
for _ in range(10):
    C_target, C_target_time = do_transformers_mat_vec_mul_original_layout(attn_weights_t, decompressed_v)
    print(f"cublas time: {C_target_time} ms")
check_error(C_target, C_ours, 0.01)
print(C_target.shape)
print(f"cr: {ctx.get_cr()}")