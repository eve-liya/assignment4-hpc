#!/usr/bin/env python3
import argparse
import torch
from torch.profiler import profile, schedule, ProfilerActivity
import contextlib

def parse_args():
    p = argparse.ArgumentParser("Memory profiling for Transformer")
    p.add_argument("--model-size", choices=["small","medium","large","xl","2.7b"], required=True)
    p.add_argument("--mode", choices=["forward","train"], required=True,
                   help="forward = inference only; train = forward+backward+optim")
    p.add_argument("--mixed", action="store_true", help="Use mixed precision autocast")
    p.add_argument("--device", choices=["cuda","cpu"], default="cuda")
    return p.parse_args()

def build_model(size, device):
    from cs336_basics.model import Transformer
    specs = {
        "small":  (768,3072,12,12),
        "medium": (1024,4096,24,16),
        "large":  (1280,5120,36,20),
        "xl":     (1600,6400,48,25),
        "2.7b":   (2560,10240,32,32),
    }
    d_model,d_ff,n_layers,n_heads = specs[size]
    model = Transformer(d_model=d_model, d_ff=d_ff, n_layers=n_layers, n_heads=n_heads,
                        vocab_size=10000, seq_len=128).to(device)
    return model

def main():
    args = parse_args()
    device = args.device
    torch.manual_seed(0)

    model = build_model(args.model_size, device)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3) if args.mode=="train" else None
    batch = torch.randint(0,10000,(16,128), device=device)

    if device=="cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.memory._record_memory_history(max_entries=1000000)

    # set up profiler for memory
    n_steps = 3
    prof = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=0, warmup=0, active=1, repeat=n_steps),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    )
    prof.__enter__()
    for _ in range(n_steps):
        prof.step()
        # optionally autocast
        if args.mixed and device=="cuda":
            from torch.cuda.amp import autocast
            ctx = autocast()
        else:
            ctx = contextlib.nullcontext()
        with ctx:
            out = model(batch)
            if args.mode=="train":
                loss = out.sum()
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
        if device=="cuda":
            torch.cuda.synchronize()
    prof.export_memory_timeline("memory_timeline.html", device=device)
    torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
    prof.__exit__(None, None, None)

    if device=="cuda":
        peak_gb = torch.cuda.max_memory_reserved() / 1024**3
        print(f"Peak memory ({args.mode}{' + AMP' if args.mixed else ''}): {peak_gb:.2f} GB")

if __name__=="__main__":
    main()
