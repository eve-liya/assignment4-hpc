import torch
from torch import nn
from torch.autograd import Function
import triton
import triton.language as tl

def compute_rmsnorm_backward_g(grad_out, x, g):
    xrms = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)
    dims = tuple(range(0, len(x.shape) - 1))
    return (xrms * grad_out).sum(dims)


def compute_rmsnorm_backward_x(grad_out, x, g):
    x_shape = x.shape
    d = x.shape[-1]
    x = x.view(-1, d)
    grad_out = grad_out.view(-1, d)

    gj = g[None, :]
    ms = x.pow(2).mean(-1, keepdim=True) + 1e-5

    gxgrad = (x * gj * grad_out).sum(-1, keepdim=True)

    out = (gj * grad_out - x * gxgrad / (d * ms)) * torch.rsqrt(ms)
    return out.view(*x_shape)


# ---------------- PyTorch RMSNorm Autograd Function ----------------
class RMSNormFunctionPT(Function):
    @staticmethod
    def forward(ctx, x, weight, eps=1e-6):
        """
        Pure PyTorch RMSNorm forward.
        x: Tensor[..., H]
        weight: Tensor[H]
        """
        ctx.save_for_backward(x, weight)
        ctx.eps = eps
        # Compute variance (mean of squares)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        inv = (var + eps).rsqrt()
        out = x * inv * weight
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        eps = ctx.eps
        H = x.shape[-1]
        # Recompute inv
        var = x.pow(2).mean(dim=-1, keepdim=True)
        inv = (var + eps).rsqrt()
        # grad w.r.t. weight (g)
        grad_g = (grad_output * x * inv).sum(dim=0)
        # grad w.r.t. input x
        term1 = grad_output * weight * inv
        dot = (grad_output * x * weight).sum(dim=-1, keepdim=True)
        term2 = x * weight * inv.pow(3) * (dot / H)
        grad_x = term1 - term2
        return grad_x, grad_g, None

# Alias
rmsnorm_pt = RMSNormFunctionPT.apply


# ---------------- Triton RMSNorm Forward Kernel ----------------
@triton.jit
def _rmsnorm_fwd_kernel(
    x_ptr, weight_ptr, out_ptr,
    N, H, eps,
    stride_n, stride_h,
    BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < H
    # Load x row and weight
    x_row = tl.load(x_ptr + row * stride_n + offsets * stride_h, mask=mask, other=0.0)
    w = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    # Compute inv
    sum_sq = tl.sum(x_row * x_row, axis=0)
    inv = tl.math.rsqrt(sum_sq / H + eps)
    # Compute output
    out_row = x_row * inv * w
    tl.store(out_ptr + row * stride_n + offsets * stride_h, out_row, mask=mask)


# ---------------- Triton RMSNorm Backward Kernel ----------------
@triton.jit
def _rmsnorm_backward_kernel(
    x_ptr, weight_ptr, grad_out_ptr, grad_x_ptr, partial_g_ptr,
    N, H, eps,
    stride_n, stride_h,
    BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < H
    # Pointers
    base = row * stride_n
    x_row = tl.load(x_ptr + base + offsets * stride_h, mask=mask, other=0.0)
    w = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    grad_out = tl.load(grad_out_ptr + base + offsets * stride_h, mask=mask, other=0.0)
    # Compute inv and inv^3
    sum_sq = tl.sum(x_row * x_row, axis=0)
    inv = tl.math.rsqrt(sum_sq / H + eps)
    inv3 = inv * inv * inv
    # grad_x = grad_out * w * inv - x * w * inv^3 * (sum(grad_out * x * w)/H)
    term1 = grad_out * w * inv
    dot = tl.sum(grad_out * x_row * w, axis=0)
    term2 = x_row * w * inv3 * (dot / H)
    grad_x_row = term1 - term2
    tl.store(grad_x_ptr + base + offsets * stride_h, grad_x_row, mask=mask)
    # partial grad_g = grad_out * x * inv
    partial_g = grad_out * x_row * inv
    tl.store(partial_g_ptr + base + offsets * stride_h, partial_g, mask=mask)


# ---------------- Triton Autograd Function ----------------
class RMSNormFunctionTriton(Function):
    @staticmethod
    def forward(ctx, x, weight, eps=1e-6):
        ctx.save_for_backward(x, weight)
        ctx.eps = eps
        orig_shape = x.shape
        H = orig_shape[-1]
        x_flat = x.contiguous().view(-1, H)
        N = x_flat.size(0)
        out_flat = torch.empty_like(x_flat)
        BLOCK_SIZE = triton.next_power_of_2(H)
        _rmsnorm_fwd_kernel[(N,)](
            x_flat, weight, out_flat,
            N, H, eps,
            x_flat.stride(0), x_flat.stride(1),
            BLOCK_SIZE=BLOCK_SIZE
        )
        return out_flat.view(orig_shape)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        eps = ctx.eps
        orig_shape = x.shape
        H = orig_shape[-1]
        x_flat = x.contiguous().view(-1, H)
        grad_out_flat = grad_output.contiguous().view(-1, H)
        N = x_flat.size(0)
        grad_x_flat = torch.empty_like(x_flat)
        partial_g = torch.empty_like(x_flat)
        BLOCK_SIZE = triton.next_power_of_2(H)
        _rmsnorm_backward_kernel[(N,)](
            x_flat, weight,
            grad_out_flat,
            grad_x_flat,
            partial_g,
            N, H, eps,
            x_flat.stride(0), x_flat.stride(1),
            BLOCK_SIZE=BLOCK_SIZE
        )
        grad_g = partial_g.sum(dim=0)
        grad_x = grad_x_flat.view(orig_shape)
        return grad_x, grad_g, None

# Alias
rmsnorm_triton = RMSNormFunctionTriton.apply
