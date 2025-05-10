import torch
import torch.distributed as dist
import torch.nn as nn

class DDP(nn.Module):
    def __init__(self, module: torch.nn.Module):
        super(DDP, self).__init__()
        self.module = module
        self.back_handles = []
        self.post_acc_handles = []
        def transform_grad(param):
            with torch.no_grad():
                param.grad.data /= dist.get_world_size()
            handle = dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM, async_op=True)
            self.back_handles.append(handle)

        for parameter in self.module.parameters():
            dist.broadcast(parameter.data, 0)
            if parameter.requires_grad:
                self.post_acc_handles.append(
                    parameter.register_post_accumulate_grad_hook(transform_grad)
                )

    def __del__(self):
        for handle in self.post_acc_handles:
            handle.remove()
    
    def forward(self, *inputs, **kwargs):
        return self.module.forward(*inputs, **kwargs)
    
    def finish_gradient_synchronization(self):
        while self.back_handles:
            self.back_handles.pop().wait()