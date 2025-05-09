#!/usr/bin/env python3
import argparse
import time
import statistics

import torch

from cs336_basics.model import Transformer as TransformerModel

MODEL_SPECS = {
    # d_model, d_ff, n_layers, n_heads
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
                   help="What to time")
    p.add_argument("--device",     choices=["cpu","cuda"], default="cuda")
    return p.parse_args()

def make_model(args):
    d_model, d_ff, n_layers, n_heads = MODEL_SPECS[args.model_size]
    model = TransformerModel(
        d_model=d_model,
        d_ff=d_ff,
        n_layers=n_layers,
        n_heads=n_heads,
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
    )
    return model.to(args.device)

def make_batch(args):
    # assumes your model takes token IDs as input
    return torch.randint(
        low=0, high=args.vocab_size,
        size=(args.batch_size, args.seq_len),
        device=args.device,
        dtype=torch.long,
    )

def run_step(model, batch, mode):
    out = model(batch)
    if mode in ("backward","both"):
        loss = out.sum()
        loss.backward()
    return out

def main():
    args = parse_args()
    torch.manual_seed(0)
    model = make_model(args)
    batch = make_batch(args)

    # zero gradients
    def zero_grad():
        for p in model.parameters():
            if p.grad is not None:
                p.grad = None

    # warm-up
    for _ in range(args.warmup):
        zero_grad()
        out = run_step(model, batch, args.mode)
        if args.device=="cuda":
            torch.cuda.synchronize()

    # timed runs
    times = []
    for _ in range(args.steps):
        zero_grad()
        start = time.perf_counter()
        out = run_step(model, batch, args.mode)
        if args.device=="cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)

    mean_t = statistics.mean(times)
    std_t  = statistics.stdev(times) if len(times)>1 else 0.0

    print(f"{args.model_size:>6} | mode={args.mode:<7} "
          f"| mean {mean_t:.4f}s ± {std_t:.4f}s over {args.steps} runs")

if __name__=="__main__":
    main()
