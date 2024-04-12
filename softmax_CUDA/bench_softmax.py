import math

import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load

# Load the CUDA kernel as a python module
minimal_softmax = load(name='softmax2', sources=['main.cpp', 'softmax2.cu'], extra_cuda_cflags=['-O2'])

# Use small model params, otherwise slower than manual attention. See caveats in README.
# batch_size = 16
# n_head = 12
# seq_len = 128
# head_embd = 64
batch_size = 4
n_head = 8
seq_len = 1200
head_embd = 2000


q = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
k = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
v = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
print('=== profiling manual transpose ===')

#print(k, F.softmax(k,dim=-1))

# Our minimal flash attention aims to be faster than this by avoiding HBM read/writes of N^2 matrices.
def manual_transpose(q):
    att = (q @ k.transpose(-2, -1) )
    soft_attn = F.softmax(att, dim=-1)

    return soft_attn

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    manual_result = manual_transpose(q)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

print('=== profiling minimal transpose === ')

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    att_cuda_kernel = (q @ k.transpose(-2, -1) )

    minimal_softmax = minimal_softmax.forward(att_cuda_kernel)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))
print(minimal_softmax.cpu())
print(manual_result.cpu())
print('attn values sanity check:', torch.allclose(minimal_softmax, manual_result, rtol=0, atol=1e-02))
torch.cuda.empty_cache()
