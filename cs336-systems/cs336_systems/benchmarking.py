#!/usr/bin/env python3
import argparse
import time
import statistics

import torch
from torch.profiler import profile, record_function, ProfilerActivity

from cs336_basics.model import Transformer as TransformerModel

MODEL_SPECS = {
    "small":  (768,  3072, 12, 12),
    "medium": (1024, 4096, 24, 16),
    "large":  (1280, 5120, 36, 20),
    "xl":     (1600, 6400, 48, 25),
    "2.7b":   (2560, 10240, 32, 32),
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-size", choices=MODEL_SPECS.keys(), required=True)
    p.add_argument("--vocab-size", type=int, default=10000)
    p.add_argument("--seq-len",    type=int, default=128)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--warmup",     type=int, default=1,
                   help="Number of warm-up iterations")
    p.add_argument("--steps",      type=int, default=5,
                   help="Number of timed iterations")
    p.add_argument("--mode",       choices=["forward","backward","both"],
                   default="both",
                   help="What to run")
    p.add_argument("--device",     choices=["cpu","cuda"], default="cuda")
    p.add_argument("--profile",    action="store_true",
                   help="Run PyTorch profiler (only for XL model)")
    return p.parse_args()

def make_model(args):
    d_model, d_ff, n_layers, n_heads = MODEL_SPECS[args.model_size]
    m = TransformerModel(
        d_model=d_model, d_ff=d_ff,
        n_layers=n_layers, n_heads=n_heads,
        vocab_size=args.vocab_size, seq_len=args.seq_len
    )
    return m.to(args.device)

def make_batch(args):
    return torch.randint(
        0, args.vocab_size,
        (args.batch_size, args.seq_len),
        device=args.device, dtype=torch.long
    )

def run_step(model, batch, optimizer, mode):
    with record_function("forward_pass"):
        out = model(batch)
    if mode in ("backward","both"):
        with record_function("backward_pass"):
            loss = out.sum()
            loss.backward()
    if optimizer is not None:
        with record_function("optimizer"):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

def main():
    args = parse_args()
    torch.manual_seed(0)

    model = make_model(args)
    batch = make_batch(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) if args.mode in ("backward","both") else None

    # warm-up
    for _ in range(args.warmup):
        run_step(model, batch, optimizer, args.mode)
        if args.device=="cuda":
            torch.cuda.synchronize()

    if args.profile:
        assert args.model_size=="xl", "--profile only supported for XL"
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True,
            profile_memory=False
        ) as prof:
            for _ in range(args.steps):
                run_step(model, batch, optimizer, args.mode)
                prof.step()
                if args.device=="cuda":
                    torch.cuda.synchronize()

        # output stacks for flame graph
        prof.export_stacks("xl_profiler_stacks.txt", "self_cuda_time_total")
        # print top 50 operators by CPU time
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=50))
    else:
        # normal timing
        times = []
        for _ in range(args.steps):
            start = time.perf_counter()
            run_step(model, batch, optimizer, args.mode)
            if args.device=="cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)

        mean_t = statistics.mean(times)
        std_t  = statistics.stdev(times) if len(times)>1 else 0.0
        print(f"{args.model_size:>6} | mode={args.mode:<7}"
              f" | mean {mean_t:.4f}s ± {std_t:.4f}s over {args.steps} runs")

if __name__=="__main__":
    main()
