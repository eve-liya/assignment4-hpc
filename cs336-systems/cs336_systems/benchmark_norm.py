#!/usr/bin/env python3
import argparse
import time
import statistics
import contextlib

import torch
from torch import nn
from torch.cuda.amp import autocast

# Import your autograd Functions
from cs336_basics.model import RMSNorm
from rmsnorm import RMSNormFunctionPT as rmsnorm_pt
from rmsnorm import RMSNormFunctionTriton as rmsnorm_triton

# RMSNorm with PyTorch autograd
class RMSNormAutograd(nn.Module):
    def __init__(self, H, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(H))
        self.eps = eps
    def forward(self, x):
        return rmsnorm_pt.apply(x, self.weight)

class RMSNormTriton(nn.Module):
    def __init__(self, H, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(H))
        self.eps = eps
    def forward(self, x):
        return rmsnorm_triton.apply(x, self.weight)

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
    print(f"{'H':>6} | {'LayerNorm':>10} | {'NNModule':>10} | {'Autograd':>10} | {'Triton':>10} | {'Compiled':>10}")
    print("-" * 72)

    for H in H_list:
        x = torch.randn(N, H, device=device)

        # Instantiate modules
        ln_module    = nn.LayerNorm(H).to(device)
        nn_module    = RMSNorm(H).to(device)
        autograd_mod = RMSNormAutograd(H).to(device)
        triton_mod   = RMSNormTriton(H).to(device)
        compiled_mod = CompiledRMSNorm(RMSNormAutograd(H).to(device)).to(device)

        # Benchmark forward or forward+backward
        ln_t, _    = bench_layer(ln_module,    x, args.backward, args.warmup, args.steps, device)
        nn_t, _    = bench_layer(nn_module,    x, args.backward, args.warmup, args.steps, device)
        ag_t, _    = bench_layer(autograd_mod, x, args.backward, args.warmup, args.steps, device)
        tr_t, _    = bench_layer(triton_mod,   x, args.backward, args.warmup, args.steps, device)
        cp_t, _    = bench_layer(compiled_mod, x, args.backward, args.warmup, args.steps, device)

        print(f"{H:6d} | {ln_t:10.3f} | {nn_t:10.3f} | {ag_t:10.3f} | {tr_t:10.3f} | {cp_t:10.3f}")

if __name__ == "__main__":
    main()
