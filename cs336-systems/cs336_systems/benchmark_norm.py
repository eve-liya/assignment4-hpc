#!/usr/bin/env python3
import argparse
import time
import statistics
import contextlib

import torch
from torch import nn
from torch.cuda.amp import autocast

# Import your autograd Functions
from rmsnorm import RMSNormFunctionPT as rmsnorm_pt
from rmsnorm import RMSNormFunctionTriton as rmsnorm_triton

# Wrapper Modules
class RMSNormPT(nn.Module):
    def __init__(self, H, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(H))
        self.eps = eps
    def forward(self, x):
        return rmsnorm_pt(x, self.weight, self.eps)

class RMSNormTriton(nn.Module):
    def __init__(self, H, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(H))
        self.eps = eps
    def forward(self, x):
        return rmsnorm_triton(x, self.weight, self.eps)

class CompiledRMSNorm(nn.Module):
    def __init__(self, base_module):
        super().__init__()
        self.compiled = torch.compile(base_module)
    def forward(self, x):
        return self.compiled(x)

# Benchmark function
def bench_layer(module, x, do_backward=False, warmup=10, steps=1000, device='cuda'):
    # Warm-up
    dy = torch.randn_like(x)
    for _ in range(warmup):
        out = module(x)
        if do_backward:
            out.backward(dy)
            module.zero_grad()
        if device=='cuda': torch.cuda.synchronize()
    # Timed runs
    times = []
    for _ in range(steps):
        start = time.perf_counter()
        out = module(x)
        if do_backward:
            out.backward(dy)
            module.zero_grad()
        if device=='cuda': torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end-start)*1000)
    return statistics.mean(times), statistics.stdev(times)

# Main script
def main():
    p = argparse.ArgumentParser(description="Benchmark LayerNorm vs RMSNorm (PT, Triton, Compiled)")
    p.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--backward', action='store_true',
                   help='Include backward pass in timings')
    args = p.parse_args()

    device = args.device
    H_list = [1024, 2048, 4096, 8192]
    N = 50000

    results = []
    for H in H_list:
        x = torch.randn(N, H, device=device)
        # Instantiate modules
        ln = nn.LayerNorm(H, elementwise_affine=True).to(device)
        rms_pt = RMSNormPT(H).to(device)
        rms_tr = RMSNormTriton(H).to(device)
        rms_cp = CompiledRMSNorm(RMSNormPT(H).to(device)).to(device)

        # Benchmark all
        t_ln_fwd, _ = bench_layer(ln, x, False, args.warmup, args.steps, device)
        t_ln_fb, _  = bench_layer(ln, x, args.backward, args.warmup, args.steps, device)
        t_pt_fwd, _ = bench_layer(rms_pt, x, False, args.warmup, args.steps, device)
        t_pt_fb, _  = bench_layer(rms_pt, x, args.backward, args.warmup, args.steps, device)
        t_tr_fwd, _ = bench_layer(rms_tr, x, False, args.warmup, args.steps, device)
        t_tr_fb, _  = bench_layer(rms_tr, x, args.backward, args.warmup, args.steps, device)
        t_cp_fwd, _ = bench_layer(rms_cp, x, False, args.warmup, args.steps, device)
        t_cp_fb, _  = bench_layer(rms_cp, x, args.backward, args.warmup, args.steps, device)

        results.append((H,
                        t_ln_fwd, t_ln_fb,
                        t_pt_fwd, t_pt_fb,
                        t_tr_fwd, t_tr_fb,
                        t_cp_fwd, t_cp_fb))

    # Print table
    header = (" H   | LN fwd | LN fwd+bwd | PT fwd | PT fwd+bwd | TR fwd | TR fwd+bwd | CP fwd | CP fwd+bwd")
    print(header)
    print('-'*len(header))
    for row in results:
        H, *times = row
        print(f"{H:4d} | " + " | ".join(f"{t:7.2f}" for t in times))

if __name__=='__main__':
    main()