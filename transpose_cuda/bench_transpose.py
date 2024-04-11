import math

import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load

# Load the CUDA kernel as a python module
minimal_transpose = load(name='transpose', sources=['main.cpp', 'transpose.cu'], extra_cuda_cflags=['-O2'])

# Use small model params, otherwise slower than manual attention. See caveats in README.
batch_size = 1
n_head = 1
seq_len = 4096
head_embd = 4096
# seq_len = 48 # M
# head_embd = 96 # N


q = torch.rand(batch_size, n_head, seq_len, head_embd).cuda()
print('=== profiling manual transpose ===')

# Our minimal flash attention aims to be faster than this by avoiding HBM read/writes of N^2 matrices.
def manual_transpose(q):
    y = q.transpose(-2, -1)
    return y

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    manual_result = manual_transpose(q)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

print('=== profiling minimal transpose === ')

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    minimal_transpose = minimal_transpose.forward(q)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))
print(minimal_transpose.cpu())
print(manual_result.cpu())

print('value sanity check:', torch.allclose(minimal_transpose, manual_result, rtol=0, atol=1e-02))