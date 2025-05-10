#!/usr/bin/env python3
import os
import time
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_basics.model import Transformer as TransformerModel

MODEL_SPECS = {
    "small":  (768,  3072, 12, 12),
    "medium": (1024, 4096, 24, 16),
    "large":  (1280, 5120, 36, 20),
    "xl":     (1600, 6400, 48, 25),
    "2.7b":   (2560,10240, 32, 32),
}

def make_model(size, device):
    d_model, d_ff, n_layers, n_heads = MODEL_SPECS[size]
    m = TransformerModel(
        d_model=d_model, d_ff=d_ff,
        n_layers=n_layers, n_heads=n_heads,
        vocab_size=10000, seq_len=128
    ).to(device)
    return m

def run_step(model, batch, optimizer, mode):
    # forward
    out = model(batch)
    if mode in ("backward","both"):
        out.sum().backward()
    if optimizer:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

def ddp_worker(rank, world_size, args):
    # 1) init process group
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = str(args.master_port)
    dist.init_process_group("nccl" if args.backend=="nccl" else "gloo",
                            rank=rank, world_size=world_size)
    torch.cuda.set_device(rank % torch.cuda.device_count())
    device = f"cuda:{rank % torch.cuda.device_count()}"

    # 2) build model & broadcast params
    model = make_model(args.model_size, device)
    for p in model.parameters():
        dist.broadcast(p.data, src=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) if args.mode!="forward" else None

    # 3) prepare data shard
    total_batch = args.batch_size * world_size
    X = torch.randint(0, 10000, (total_batch,128), device=device)
    shard = X[rank*args.batch_size : (rank+1)*args.batch_size]

    # 4) timing loop
    # we separately time communication by wrapping the all-reduce
    comm_time = 0.0
    total_time = 0.0
    for _ in range(args.steps):
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        out = model(shard)
        if args.mode in ("backward","both"):
            out.sum().backward()
            # measure all-reduce time
            t0 = time.perf_counter()
            for p in model.parameters():
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad /= world_size
            comm_time += (time.perf_counter() - t0)
        if optimizer:
            optimizer.step()
        # sync to get accurate total time
        torch.cuda.synchronize(device)
        total_time += (time.perf_counter() - start)

    avg_total = total_time / args.steps
    avg_comm  = comm_time  / args.steps
    # gather results at rank 0
    results = torch.tensor([avg_total, avg_comm], device=device)
    gather = [torch.zeros_like(results) for _ in range(world_size)] if rank==0 else None
    dist.gather(results, gather_list=gather, dst=0)
    if rank==0:
        # average across ranks
        gathered = torch.stack(gather)  # shape (world_size, 2)
        avg = gathered.mean(dim=0).tolist()
        print(f"{args.mode.upper()} | {args.setup_desc} | total={avg[0]*1000:7.1f} ms | comm={avg[1]*1000:7.1f} ms | ratio={avg[1]/avg[0]:5.2%}")
    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", required=True, choices=MODEL_SPECS)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps",      type=int, default=20)
    parser.add_argument("--mode",       choices=["forward","backward","both"], default="both")
    parser.add_argument("--backend",    choices=["gloo","nccl"], default="nccl")
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", type=int,   default=29500)
    parser.add_argument("--setup-desc", help="Description for logging")
    args = parser.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    # single-node multi-GPU: WORLD_SIZE=2, launch with torchrun
    # multi-node: export WORLD_SIZE=2 and MASTER_ADDR/PORT, then torchrun --nnodes=2 ...
    mp.spawn(ddp_worker, args=(world_size,args), nprocs=world_size, join=True)

if __name__=="__main__":
    main()
