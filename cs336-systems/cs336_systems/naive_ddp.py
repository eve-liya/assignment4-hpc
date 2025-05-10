#!/usr/bin/env python3
import os
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim


class ToyModel(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=32, output_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    def forward(self, x):
        return self.net(x)

def single_process_train(args):
    torch.manual_seed(0)
    model = ToyModel().to("cpu")
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    total_batch = args.batch_size * args.world_size
    # random data: X shape (total_batch, input_dim), Y same
    X = torch.randn(total_batch, 10)
    Y = torch.randn(total_batch, 5)

    model.train()
    for _ in range(args.epochs):
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, Y)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), "single.pth")
    print("✅ Single-process training complete, checkpoint saved to single.pth")

def ddp_worker(rank, world_size, args):
    # 3.1) init process group
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(args.backend, rank=rank, world_size=world_size)

    torch.manual_seed(0)
    device = "cpu"
    model = ToyModel().to(device)

    # 3.2) broadcast initial params from rank 0
    for p in model.parameters():
        dist.broadcast(p.data, src=0)

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # same random data
    total_batch = args.batch_size * world_size
    X = torch.randn(total_batch, 10)
    Y = torch.randn(total_batch, 5)

    # shard the data
    per_rank = args.batch_size
    start = rank * per_rank
    end   = start + per_rank
    x_part = X[start:end].to(device)
    y_part = Y[start:end].to(device)

    model.train()
    for _ in range(args.epochs):
        optimizer.zero_grad()
        out = model(x_part)
        loss = criterion(out, y_part)
        loss.backward()

        if args.naive:
            # 3.3) naive all-reduce each grad tensor
            for p in model.parameters():
                dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
                p.grad.data /= world_size
        else:
            grads = [p.grad.data for p in model.parameters()]
            flat  = torch._utils._flatten_dense_tensors(grads)
            dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            flat.div_(world_size)
            # unpack back into each gradient buffer
            for buf, synced in zip(grads,
                                torch._utils._unflatten_dense_tensors(flat, grads)):
                buf.copy_(synced)
        optimizer.step()

    # rank 0 writes checkpoint
    if rank == 0:
        torch.save(model.state_dict(), "ddp.pth")
        print("✅ DDP training complete on rank 0, checkpoint saved to ddp.pth")

    dist.destroy_process_group()


def verify_match():
    single_sd = torch.load("single.pth", map_location="cpu")
    ddp_sd    = torch.load("ddp.pth",    map_location="cpu")

    all_close = True
    for k in single_sd:
        if not torch.allclose(single_sd[k], ddp_sd[k], atol=1e-6):
            print(f"MISMATCH in {k}")
            all_close = False
            break

    if all_close:
        print("SUCCESS: single-process and DDP weights match!")


def main():
    parser = argparse.ArgumentParser(description="Naïve DDP correctness test")
    parser.add_argument("--world-size", type=int, default=2,
                        help="Number of processes / devices")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Per-process batch size")
    parser.add_argument("--backend", choices=["gloo","nccl"], default="gloo")
    parser.add_argument("--naive", action="store_true",
                        help="Run the naïve DDP example")
    args = parser.parse_args()

    # 1) single-process run & checkpoint
    single_process_train(args)

    # 2) spawn DDP workers
    mp.spawn(ddp_worker,
             args=(args.world_size, args),
             nprocs=args.world_size,
             join=True)

    # 3) verify
    verify_match()

if __name__ == "__main__":
    main()
