#!/usr/bin/env python3
import time
import torch
import statistics
import argparse
from torch import nn

# ———— Simple PyTorch RMSNorm implementation ————
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # x: [N, H]
        mean_sq = x.pow(2).mean(dim=-1, keepdim=True)
        denom   = (mean_sq + self.eps).rsqrt()
        return x * denom * self.weight

def benchmark_layer(layer, x, warmup=10, steps=1000, device="cuda"):
    # Warm-up
    for _ in range(warmup):
        _ = layer(x)
        if device=="cuda": torch.cuda.synchronize()
    # Timed runs
    times = []
    for _ in range(steps):
        start = time.perf_counter()
        _ = layer(x)
        if device=="cuda": torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end-start)*1000)  # ms
    return statistics.mean(times), statistics.stdev(times)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu","cuda"], default="cuda")
    args = parser.parse_args()

    device = args.device
    torch.manual_seed(0)

    dims = [1024, 2048, 4096, 8192]
    results = []

    for H in dims:
        # create input
        x = torch.randn(50000, H, device=device, dtype=torch.float32)

        # RMSNorm
        rms = RMSNorm(H).to(device)
        mean_rms, std_rms = benchmark_layer(rms, x, device=device)

        # LayerNorm (no bias for direct compare; add bias if you want)
        ln  = nn.LayerNorm(H).to(device)
        mean_ln, std_ln   = benchmark_layer(ln,  x, device=device)

        results.append((H, mean_rms, std_rms, mean_ln, std_ln))

    # Print table
    print(f"{'H':>6} | {'RMSNorm (ms)':>14} | {'LayerNorm (ms)':>15}")
    print("-"*42)
    for H, mr, sr, ml, sl in results:
        print(f"{H:6} | {mr:14.2f}±{sr:.2f} | {ml:15.2f}±{sl:.2f}")

if __name__=="__main__":
    main()
