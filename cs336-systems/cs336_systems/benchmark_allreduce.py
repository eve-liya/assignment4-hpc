#!/usr/bin/env python3
import os
import time
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pandas as pd

def run_allreduce(rank, world_size, args, return_dict):
    # set up master address/port
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = args.port
    dist.init_process_group(
        backend=args.backend,
        rank=rank,
        world_size=world_size,
        timeout=torch.distributed.timedelta(seconds=30),
    )

    # choose device
    if args.device == "gpu":
        torch.cuda.set_device(rank % torch.cuda.device_count())
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")

    # Tensor for this rank
    x = torch.ones(args.elements, dtype=torch.float32, device=dev)
    # Warm-up
    for _ in range(args.warmup):
        dist.all_reduce(x, async_op=False)

    # Timed runs
    times = []
    for _ in range(args.iters):
        start = time.perf_counter()
        dist.all_reduce(x, async_op=False)
        if dev.type=="cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)

    # Store mean time
    return_dict[rank] = sum(times) / len(times)
    dist.destroy_process_group()

def benchmark_config(backend, device, elements, world_size, args):
    manager = mp.Manager()
    return_dict = manager.dict()
    mp.spawn(
        run_allreduce,
        args=(world_size, argparse.Namespace(backend=backend,
                                              device=device,
                                              elements=elements,
                                              warmup=args.warmup,
                                              iters=args.iters,
                                              port=args.port), return_dict),
        nprocs=world_size,
        join=True
    )
    # average across ranks
    avg_time = sum(return_dict.values()) / world_size
    return avg_time

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backends", nargs="+",
                   choices=["gloo","nccl"], default=["gloo","nccl"])
    p.add_argument("--devices", nargs="+",
                   choices=["cpu","gpu"], default=["cpu","gpu"])
    p.add_argument("--sizes", nargs="+", type=int,
                   help="Tensor element counts (e.g. 512*1024,1*1024*1024,...)",
                   required=True)
    p.add_argument("--procs", nargs="+", type=int, default=[2,4,6],
                   help="Number of processes to spawn")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters",  type=int, default=20)
    p.add_argument("--port",   type=str, default="29500")
    p.add_argument("--out",    type=str, default="allreduce_results.csv")
    args = p.parse_args()

    rows = []
    for backend in args.backends:
        for device in args.devices:
            # NCCL only works for GPU
            if backend=="nccl" and device=="cpu":
                continue
            for size in args.sizes:
                for world_size in args.procs:
                    t = benchmark_config(backend, device, size, world_size, args)
                    rows.append({
                        "backend": backend,
                        "device":  device,
                        "elements": size,
                        "processes": world_size,
                        "time_s": t
                    })
                    print(f"{backend}/{device} | size={size} | procs={world_size} -> {t:.4f}s")

    # save results
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\nResults written to {args.out}")

if __name__=="__main__":
    main()
