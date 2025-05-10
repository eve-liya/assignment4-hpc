#!/usr/bin/env python3
import argparse
import time
import statistics
import contextlib
from torch.cuda.amp import autocast


import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
import cs336_basics.model as model
from rmsnorm import RMSNormFunctionPT as rmsnorm_pt
from rmsnorm import RMSNormFunctionTriton as rmsnorm_triton

MODEL_SPECS = {
    "small":  (768,  3072, 12, 12),
    "medium": (1024, 4096, 24, 16),
    "large":  (1280, 5120, 36, 20),
    "xl":     (1600, 6400, 48, 25),
    "2.7b":   (2560, 10240, 32, 32),
}

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

NORM_IMPLS = {
    "triton": RMSNormTriton,
    "pytorch": model.RMSNorm,
    "layernorm": nn.LayerNorm,
    "ptautograd": RMSNormAutograd,
    "compiled": lambda H: torch.compile(RMSNormAutograd(H)),
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-size", choices=MODEL_SPECS.keys(), required=True)
    p.add_argument("--vocab-size", type=int, default=1000)
    p.add_argument("--seq-len",    type=int, default=64)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--warmup",     type=int, default=1,
                   help="Number of warm-up iterations")
    p.add_argument("--norm",       choices=NORM_IMPLS.keys(), default="pytorch",
                     help="Normalization implementation")
    p.add_argument("--steps",      type=int, default=5,
                   help="Number of timed iterations")
    p.add_argument("--mode",       choices=["forward","backward","both"],
                   default="both",
                   help="What to run")
    p.add_argument("--device",     choices=["cpu","cuda"], default="cuda")
    p.add_argument("--profile",    action="store_true",
                   help="Run PyTorch profiler")
    p.add_argument("--profile-memory", action="store_true",
                     help="Run PyTorch profiler with memory tracking"),
    p.add_argument("--compile", action="store_true"),
    p.add_argument("--mixed", action="store_true",
               help="Enable torch.autocast mixed precision")

    return p.parse_args()

def make_model(args):
    d_model, d_ff, n_layers, n_heads = MODEL_SPECS[args.model_size]
    m = model.BasicsTransformerLM(
        d_model=d_model, d_ff=d_ff,
        num_layers=n_layers, num_heads=n_heads,
        vocab_size=args.vocab_size, context_length=args.seq_len, 
        norm=NORM_IMPLS[args.norm],
    )
    if args.compile:
        m = torch.compile(m)
    return m.to(args.device)

def make_batch(args):
    return torch.randint(
        0, args.vocab_size,
        (args.batch_size, args.seq_len),
        device=args.device, dtype=torch.long
    )

def run_step(model, batch, optimizer, mode, use_mixed, device):
    ctx = autocast() if use_mixed and device=="cuda" else contextlib.nullcontext()
    with ctx, record_function("forward_pass"):
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
        run_step(model, batch, optimizer, args.mode, args.mixed, args.device)
        if args.device=="cuda":
            torch.cuda.synchronize()

    if args.profile:
        from torch._C._profiler import _ExperimentalConfig
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            experimental_config=_ExperimentalConfig(verbose=True),
            record_shapes=True,
            with_stack=True,
            profile_memory=False
        ) as prof:

            for _ in range(args.steps):
                run_step(model, batch, optimizer, args.mode, args.mixed, args.device)
                prof.step()
                if args.device=="cuda":
                    torch.cuda.synchronize()

        # output stacks for flame graph
        prof.export_stacks("l_profiler_stacks.txt", "self_cuda_time_total")
        # print top 50 operators by CPU time
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=50))
    elif args.profile_memory:
        from torch._C._profiler import _ExperimentalConfig
        torch.cuda.memory._record_memory_history(max_entries=1000000)
        n_steps = 3
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=n_steps),
            experimental_config=_ExperimentalConfig(verbose=True),
            record_shapes=True,
            with_stack=True,
            profile_memory=True
        ) as prof:
            for _ in range(n_steps):
                run_step(model, batch, optimizer, args.mode, args.mixed, args.device)
                prof.step()
                if args.device=="cuda":
                    torch.cuda.synchronize()
            prof.export_memory_timeline("timeline.html", device=args.device)

        torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)
    else:
        # normal timing
        times = []
        for _ in range(args.steps):
            start = time.perf_counter()
            run_step(model, batch, optimizer, args.mode, args.mixed, args.device)
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
