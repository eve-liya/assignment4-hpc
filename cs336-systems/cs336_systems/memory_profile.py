#!/usr/bin/env python3
import argparse
import torch
from torch.profiler import profile, schedule, ProfilerActivity
import contextlib
from torch.profiler import profile, record_function, ProfilerActivity
from torch.cuda.amp import autocast
import os

from cs336_basics.model import BasicsTransformerLM, RMSNorm

def run_step(model, batch, optimizer, mode, use_mixed, device):
    ctx = autocast() if use_mixed and device=="cuda" else contextlib.nullcontext()
    with ctx:
        with record_function("forward_pass"):
            out = model(batch)
        if mode == "train":
            with record_function("backward_pass"):
                loss = out.sum()
                loss.backward()
            if optimizer is not None:
                with record_function("optimizer"):
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
    return out

def parse_args():
    p = argparse.ArgumentParser("Memory profiling for Transformer")
    p.add_argument("--model-size", choices=["small","medium","large","xl","2.7b"], required=True)
    p.add_argument("--mode", choices=["forward","train"], required=True,
                   help="forward = inference only; train = forward+backward+optim")
    p.add_argument("--mixed", action="store_true", help="Use mixed precision autocast")
    p.add_argument("--device", choices=["cuda","cpu"], default="cuda")
    p.add_argument("--steps", type=int, default=5, help="Number of profiling steps")
    return p.parse_args()

def build_model(size, device):
    specs = {
        "small":  (768, 3072, 12, 12),
        "medium": (1024, 4096, 24, 16),
        "large":  (1280, 5120, 36, 20),
        "xl":     (1600, 6400, 48, 25),
        "2.7b":   (2560, 10240, 32, 32),
    }
    
    if size not in specs:
        raise ValueError(f"Unknown model size: {size}")
    
    d_model, d_ff, n_layers, n_heads = specs[size]
    model = BasicsTransformerLM(
        d_model=d_model, 
        d_ff=d_ff, 
        num_layers=n_layers, 
        num_heads=n_heads,
        vocab_size=1000, 
        context_length=64, 
        norm=RMSNorm
    ).to(device)
    return model

def main():
    args = parse_args()
    device = args.device
    torch.manual_seed(0)
    
    print(f"Using device: {device}")
    print(f"Model size: {args.model_size}")
    print(f"Mode: {args.mode}")
    print(f"Mixed precision: {args.mixed}")
    
    try:
        model = build_model(args.model_size, device)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    opt = torch.optim.Adam(model.parameters(), lr=1e-3) if args.mode=="train" else None
    batch = torch.randint(0, 1000, (16, 64), device=device)

    # Warm up
    print("Warming up...")
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
            if opt:
                opt.step()
                opt.zero_grad(set_to_none=True)
    
    if device=="cuda":
        torch.cuda.synchronize()
        
    # Enable memory tracking before profiling
    if device=="cuda":
        print("Enabling memory tracking...")
        try:
            # Try the simpler version without context parameter
            torch.cuda.memory._record_memory_history(enabled=True)
        except (AttributeError, RuntimeError, TypeError) as e:
            print(f"Memory tracking not available: {e}")
            print("Will continue without detailed memory tracking")

    # Reset peak memory stats before profiling
    if device=="cuda":
        torch.cuda.reset_peak_memory_stats()
    
    # set up profiler for memory
    n_steps = args.steps
    print(f"Profiling for {n_steps} steps...")
    
    with profile(
        activities=[ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device=="cuda" else []),
        schedule=schedule(wait=0, warmup=1, active=n_steps-1, repeat=1),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_modules=True,
    ) as prof:
        for i in range(n_steps):
            print(f"Step {i+1}/{n_steps}")
            out = run_step(model, batch, opt, args.mode, args.mixed, device)
            prof.step()
    
    print("Profiling complete.")
    
    # Generate standard profiling table output
    print("\nProfiling results summary:")
    try:
        print(prof.key_averages().table(sort_by="cuda_time_total" if device=="cuda" else "cpu_time_total", 
                                      row_limit=10))
    except Exception as e:
        print(f"Failed to print profiling table: {e}")
    
    # # Try to export Chrome trace
    # try:
    #     trace_file = f"profile_trace_{args.model_size}_{args.mode}{'_mixed' if args.mixed else ''}.json"
    #     print(f"Exporting Chrome trace to {trace_file}...")
    #     prof.export_chrome_trace(trace_file)
    #     print(f"Chrome trace exported successfully to {trace_file}")
    # except Exception as e:
    #     print(f"Failed to export Chrome trace: {e}")
    
    # # Try to export memory timeline
    # try:
    #     timeline_file = f"memory_timeline_{args.model_size}_{args.mode}{'_mixed' if args.mixed else ''}.html"
    #     print(f"Exporting memory timeline to {timeline_file}...")
    #     prof.export_memory_timeline(timeline_file, device=device)
    #     print(f"Memory timeline exported successfully to {timeline_file}")
    # except ValueError as e:
    #     print(f"Failed to export memory timeline: {e}")
    #     print("This usually means no memory events were recorded during profiling.")
    # except Exception as e:
    #     print(f"Failed to export memory timeline (unexpected error): {e}")
    
    # Try to dump memory snapshot
    if device=="cuda":
        try:
            snapshot_file = f"memory_snapshot_{args.model_size}_{args.mode}{'_mixed' if args.mixed else ''}.pickle"
            print(f"Dumping memory snapshot to {snapshot_file}...")
            torch.cuda.memory._dump_snapshot(snapshot_file)
            print(f"Memory snapshot dumped successfully to {snapshot_file}")
        except Exception as e:
            print(f"Failed to dump memory snapshot: {e}")
        
        # Disable memory tracking
        try:
            # Use simpler form without parameters
            torch.cuda.memory._record_memory_history(enabled=None)
        except (AttributeError, RuntimeError, TypeError) as e:
            try:
                # Try alternative signature
                torch.cuda.memory._record_memory_history(False)
            except Exception:
                pass

    # Report peak memory usage
    if device=="cuda":
        peak_bytes = torch.cuda.max_memory_reserved()
        peak_gb = peak_bytes / 1024**3
        allocated_gb = torch.cuda.memory_allocated() / 1024**3
        reserved_gb = torch.cuda.memory_reserved() / 1024**3
        
        print("\nMemory Usage Summary:")
        print(f"Peak memory ({args.mode}{' + AMP' if args.mixed else ''}): {peak_gb:.4f} GB")
        print(f"Currently allocated: {allocated_gb:.4f} GB")
        print(f"Currently reserved: {reserved_gb:.4f} GB")

if __name__=="__main__":
    main()