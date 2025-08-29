import matplotlib.pyplot as plt
import numpy as np

ctx_len = [2048, 4096, 8192, 16384]
cublas_time = [0.21196800470352173, 0.19968000054359436, 0.22220799326896667, 0.3020800054073334]
our_fused_kernel_time = [0.198656, 0.293888, 0.456704, 0.827392]
new_our_fused_kernel_time = [0.159744, 0.220160, 0.364544, 0.634880]
new_our_fused_kernel_time_2 = [0.136192, 0.185344, 0.306176, 0.525312]

# Convert context length to MB and GB
def ctx_len_to_mb(ctx_len):
    return ctx_len * 32 * 128 * 2 / 1024 / 1024

def ctx_len_to_gb(ctx_len):
    return ctx_len * 32 * 128 * 2 / 1024 / 1024 / 1024

# Calculate data sizes in GB
data_sizes_gb = [ctx_len_to_gb(cl) for cl in ctx_len]

# Calculate time differences in seconds (ms to s)
time_diff_old = [(our - cublas) / 1000 for our, cublas in zip(our_fused_kernel_time, cublas_time)]
time_diff_new = [(our - cublas) / 1000 for our, cublas in zip(new_our_fused_kernel_time, cublas_time)]
time_diff_new_2 = [(our - cublas) / 1000 for our, cublas in zip(new_our_fused_kernel_time_2, cublas_time)]

# Calculate throughput (GB/s) = data size / time difference
throughput_old = []
throughput_new = []
throughput_new_2 = []

for size, diff_old, diff_new, diff_new_2 in zip(data_sizes_gb, time_diff_old, time_diff_new, time_diff_new_2):
    # For old kernel
    if diff_old != 0:
        if diff_old > 0:
            throughput_old.append(size / abs(diff_old))
        else:
            throughput_old.append(0)
    else:
        throughput_old.append(0)

    # For new kernel
    if diff_new != 0:
        if diff_new > 0:
            throughput_new.append(size / abs(diff_new))
        else:
            throughput_new.append(0)
    else:
        throughput_new.append(0)

    # For new kernel
    if diff_new_2 != 0:
        if diff_new_2 > 0:
            throughput_new_2.append(size / abs(diff_new_2))
        else:
            throughput_new_2.append(0)
    else:
        throughput_new_2.append(0)

# Create the throughput plot
plt.figure(figsize=(10, 6))
plt.plot(ctx_len, throughput_old, marker='o', color='green', linewidth=2, label='Old Fused Kernel')
plt.plot(ctx_len, throughput_new, marker='s', color='blue', linewidth=2, label='New Fused Kernel')
plt.plot(ctx_len, throughput_new_2, marker='s', color='brown', linewidth=2, label='New Fused Kernel 2')
plt.xlabel('Context Length')
plt.ylabel('Throughput Difference (GB/s)')
plt.title('Throughput Difference: Our Fused Kernels vs cuBLAS')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Add secondary x-axis for memory size
ax2 = plt.gca().twiny()
ctx_len_mb = [ctx_len_to_mb(cl) for cl in ctx_len]
ax2.plot(ctx_len_mb, [0]*len(ctx_len_mb), alpha=0)  # Invisible plot just to set the scale
ax2.set_xlabel('Memory Size (MB)')

plt.tight_layout()
plt.savefig("throughput.png")

# Create the performance/time comparison plot
plt.figure(figsize=(10, 6))
plt.plot(ctx_len, cublas_time, marker='o', color='red', linewidth=2, label='cuBLAS')
plt.plot(ctx_len, our_fused_kernel_time, marker='o', color='green', linewidth=2, label='Old Fused Kernel')
plt.plot(ctx_len, new_our_fused_kernel_time, marker='s', color='blue', linewidth=2, label='New Fused Kernel')
plt.plot(ctx_len, new_our_fused_kernel_time_2, marker='s', color='brown', linewidth=2, label='New Fused Kernel 2')
plt.xlabel('Context Length')
plt.ylabel('Execution Time (ms)')
plt.title('Performance Comparison: cuBLAS vs Our Fused Kernels')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Add secondary x-axis for memory size
ax2 = plt.gca().twiny()
ctx_len_mb = [ctx_len_to_mb(cl) for cl in ctx_len]
ax2.plot(ctx_len_mb, [0]*len(ctx_len_mb), alpha=0)  # Invisible plot just to set the scale
ax2.set_xlabel('Memory Size (MB)')

plt.tight_layout()
plt.savefig("performance.png")