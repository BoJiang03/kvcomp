import torch
import kvcomp_cuda_extension

def k_mat_vec_mul(k: torch.Tensor, q: torch.Tensor):
    assert k.dim() == 4
    assert q.dim() == 3
    assert k.shape[1] == q.shape[0]
    assert k.shape[2] == q.shape[1]
    assert k.shape[3] == q.shape[2]

    out = torch.empty(
        (k.shape[0], k.shape[1], k.shape[2]),
        dtype=k.dtype,
        device=k.device
    )
    time_ = kvcomp_cuda_extension.k_mat_vec_mul_cuda_export(k, q, out)
    return out, time_