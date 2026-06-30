# """Simple, minimal implementation of Mamba in one file of PyTorch with parallel selective scan.
#
# Suggest reading the following before/while reading the code:
#     [1] Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Albert Gu and Tri Dao)
#         https://arxiv.org/abs/2312.00752
#     [2] The Annotated S4 (Sasha Rush and Sidd Karamcheti)
#         https://srush.github.io/annotated-s4
#
# Glossary:
#     b: batch size                       (`B` in Mamba paper [1] Algorithm 2)
#     l: sequence length                  (`L` in [1] Algorithm 2)
#     d or d_model: hidden dim
#     n or d_state: latent state dim      (`N` in [1] Algorithm 2)
#     expand: expansion factor            (`E` in [1] Section 3.4)
#     d_in or d_inner: d * expand         (`D` in [1] Algorithm 2)
#     A, B, C, D: state space parameters  (See any state space representation formula)
#                                         (B, C are input-dependent (aka selective, a key innovation in Mamba); A, D are not)
#     Δ or delta: input-dependent step size
#     dt_rank: rank of Δ                  (See [1] Section 3.6 "Parameterization of ∆")
#
# """
# from __future__ import annotations
#
# from typing import Union
#
# import math
# import json
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from dataclasses import dataclass
# from einops import rearrange, repeat, einsum
# import einops
#
#
# @dataclass
# class ModelArgs:
#     d_model: int
#     d_inner: int
#     n_layer: int
#     size: int
#     d_state: int = 16
#     expand: int = 2
#     dt_rank = math.ceil(2048 / 16)
#     d_conv: int = 4
#     conv_bias: bool = True
#     bias: bool = False
#
# class Mamba(nn.Module):
#     def __init__(self, args: ModelArgs):
#         """Full Mamba model."""
#         super().__init__()
#         self.args = args
#         self.layers = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])
#         self.norm_f = RMSNorm(args.d_model)
#
#     def forward(self, input_ids):
#         """
#         Args:
#             input_ids (long tensor): shape (b, l)    (See Glossary at top for definitions of b, l, d_in, n...)
#
#         Returns:
#             logits: shape (b, l, vocab_size)
#         """
#         x = input_ids
#         T = input_ids.shape[1]
#         x = self.norm_f(x)
#         for layer in self.layers:
#             x = layer(x)
#         return x
#
#
# class ResidualBlock(nn.Module):
#     def __init__(self, args: ModelArgs):
#         """Simple block wrapping Mamba block with normalization and residual connection."""
#         super().__init__()
#         self.args = args
#         self.mixer = MambaBlock(args)
#         self.norm = RMSNorm(args.d_model)
#
#     def forward(self, x):
#         """
#         Args:
#             x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)
#
#         Returns:
#             output: shape (b, l, d)
#         """
#         output = self.mixer(self.norm(x)) + x
#         return output
#
#
# class MambaBlock(nn.Module):
#     def __init__(self, args: ModelArgs):
#         """A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper [1]."""
#         super().__init__()
#         self.args = args
#         self.in_proj = nn.Linear(args.d_model, args.d_inner * 3, bias=args.bias)
#
#         # Multi-scale temporal convolutions
#         self.temporal_conv = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv1d(in_channels=args.d_inner,
#                           out_channels=args.d_inner,
#                           bias=args.conv_bias,
#                           kernel_size=k,
#                           groups=args.d_inner,
#                           padding="same"),
#                 nn.BatchNorm1d(args.d_inner),
#                 nn.GELU()
#             ) for k in [3, 5, 7, 9]
#         ])
#
#         # Dropout
#         if tuple(map(int, torch.__version__.split('.')[:2])) == (1, 11):
#             print("WARNING: Dropout is bugged in PyTorch 1.11. Results may be worse.")
#             self.dropout_fn = nn.Dropout(0.2)
#         elif tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 12):
#             self.dropout_fn = nn.Dropout1d(0.2)
#         else:
#             self.dropout_fn = nn.Dropout2d(0.2)
#
#         # x_proj takes in `x` and outputs the input-specific Δ, B, C
#         self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)
#
#         # dt_proj projects Δ from dt_rank to d_in
#         self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)
#
#         A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner)
#         self.A_log = nn.Parameter(torch.log(A))
#         self.D = nn.Parameter(torch.ones(args.d_inner))
#         self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)
#         self.norm_f = nn.LayerNorm(args.d_model)
#         self.linear = nn.Linear(2048, 512)
#
#     def forward(self, x):
#         """Mamba block forward. This looks the same as Figure 3 in Section 3.4 in the Mamba paper [1]."""
#         (b, l, d) = x.shape
#
#         x_and_res = self.in_proj(x)  # (b, l, 3 * d_in)
#         x, z, res = x_and_res.split(split_size=[self.args.d_inner, self.args.d_inner, self.args.d_inner], dim=-1)
#
#         # Forward path
#         multi_scale = []
#         for conv in self.temporal_conv:
#             multi_scale.append(conv(x.transpose(1, 2)))
#         x_local = torch.cat(multi_scale, dim=1)  # [B, 4*d_inner, T]
#         x_fwd = x_local.transpose(1, 2)  # [B, T, 4*d_inner]
#         x_fwd = F.silu(x_fwd)
#         y_fwd = self.ssm(res)  # Forward SSM
#         x_fwd = self.linear(x_fwd)
#         y_fwd = y_fwd * x_fwd
#
#         # 替换为“时间反转后再进入 backward conv”的版本
#         multi_scale1 = []
#         # 将时间维度反转后输入卷积块
#         x_rev = x.transpose(1, 2).flip(dim=-1)  # (B, d_inner, L)
#         for conv in self.temporal_conv:
#             multi_scale1.append(conv(x_rev))
#         x_local1 = torch.cat(multi_scale1, dim=1)  # [B, 4*d_inner, L]
#         x_bwd = x_local1.transpose(1, 2)  # [B, L, 4*d_inner]
#         x_bwd = x_bwd.flip(dim=1)  # 还原到原始时间顺序
#         x_bwd = F.silu(x_bwd)
#         y_bwd = self.ssm(res)  # Backward SSM
#         x_bwd = self.linear(x_bwd)
#         y_bwd = y_bwd * x_bwd
#         # Combine forward and backward paths
#         y = y_fwd + y_bwd
#
#         # Activation branch
#         z = F.silu(z)  # Activation for gating
#         y = y * z  # Gating mechanism
#
#         # Output projection
#         output = self.out_proj(y)  # (b,l,d_in) -> (b,l,d)
#
#         return output
#
#     def ssm(self, x):
#         """Runs the SSM. See:
#             - Algorithm 2 in Section 3.2 in the Mamba paper [1]
#             - run_SSM(A, B, C, u) in The Annotated S4 [2]
#         """
#         (d_in, n) = self.A_log.shape
#
#         # Compute ∆ A B C D, the state space parameters.
#         A = -torch.exp(self.A_log.float())  # shape (d_in, n)
#         D = self.D.float()
#
#         x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)
#
#         (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, n, n], dim=-1)
#         delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)
#
#         y = self.selective_scan_parallel(x, delta, A, B, C, D)
#
#         return y
#
#     def selective_scan_parallel(self, u, delta, A, B, C, D):
#         """
#         Parallel implementation of selective scan algorithm using cumulative operations.
#
#         This replaces the sequential loop with parallel cumulative product and sum operations.
#         The key insight is that the recurrence relation:
#             x_t = A_t * x_{t-1} + B_t * u_t
#         can be solved in closed form using cumulative products.
#
#         Args:
#             u: shape (b, l, d_in)
#             delta: shape (b, l, d_in)
#             A: shape (d_in, n)
#             B: shape (b, l, n)
#             C: shape (b, l, n)
#             D: shape (d_in,)
#
#         Returns:
#             output: shape (b, l, d_in)
#         """
#         (b, l, d_in) = u.shape
#         n = A.shape[1]
#
#         # Discretize continuous parameters (A, B)
#         deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
#         deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')
#
#         # Parallel selective scan using cumulative operations
#         # The recurrence x_t = deltaA_t * x_{t-1} + deltaB_u_t can be solved as:
#         # x_t = (∏_{i=0}^{t} deltaA_i) * (∑_{k=0}^{t} deltaB_u_k / (∏_{i=0}^{k} deltaA_i))
#
#         # Step 1: Compute cumulative products S_t = ∏_{i=0}^{t} deltaA_i
#         # Add small epsilon to avoid numerical issues
#         S = torch.cumprod(deltaA + 1e-12, dim=1)  # (b, l, d_in, n)
#
#         # Step 2: Compute normalization factors
#         # We need 1/S_t for each position
#         S_inv = 1.0 / (S + 1e-12)  # (b, l, d_in, n)
#
#         # Step 3: Compute normalized terms c_t = deltaB_u_t / S_t
#         c = deltaB_u * S_inv  # (b, l, d_in, n)
#
#         # Step 4: Compute cumulative sum of normalized terms
#         scanned = torch.cumsum(c, dim=1)  # (b, l, d_in, n)
#
#         # Step 5: Multiply by cumulative products to get final states
#         x = S * scanned  # (b, l, d_in, n)
#
#         # Step 6: Compute output y_t = C_t * x_t
#         # C: (b, l, n), x: (b, l, d_in, n) -> y: (b, l, d_in)
#         y = torch.sum(x * C[:, :, None, :], dim=-1)  # (b, l, d_in)
#
#         # Step 7: Add skip connection
#         y = y + u * D
#
#         return y
#
#
# def drop_path(x, drop_prob: float = 0., training: bool = False):
#     if drop_prob == 0. or not training:
#         return x
#     keep_prob = 1 - drop_prob
#     shape = (x.shape[0],) + (1,) * (x.ndim - 1)
#     random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
#     random_tensor.floor_()
#     output = x.div(keep_prob) * random_tensor
#     return output
#
#
# class DropPath(nn.Module):
#     """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""
#
#     def __init__(self, drop_prob=None):
#         super(DropPath, self).__init__()
#         self.drop_prob = drop_prob
#
#     def forward(self, x):
#         return drop_path(x, self.drop_prob, self.training)
#
#
# class RMSNorm(nn.Module):
#     def __init__(self, d_model: int, eps: float = 1e-5):
#         super().__init__()
#         self.eps = eps
#         self.weight = nn.Parameter(torch.ones(d_model))
#
#     def forward(self, x):
#         output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
#         return output



# # -*- coding: utf-8 -*-
# """
# Simple, minimal implementation of Mamba in one file of PyTorch with parallel selective scan.
# Modified with dual-path friendly fusion for scene boundary detection.
#
# References:
# [1] Mamba: Linear-Time Sequence Modeling with Selective State Spaces
# [2] The Annotated S4
# """
# from __future__ import annotations
# from dataclasses import dataclass
# from typing import Union
# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import repeat, einsum
#
#
# # ----------------------------- Config -----------------------------
#
# @dataclass
# class ModelArgs:
#     d_model: int
#     d_inner: int
#     n_layer: int
#     size: int
#     d_state: int = 16
#     expand: int = 2
#     dt_rank: int = math.ceil(2048 / 16)
#     d_conv: int = 4
#     conv_bias: bool = True
#     bias: bool = False
#     # Optional knobs (will be read via getattr)
#     bidirectional: bool = True
#     kernels: tuple = (3, 5, 7, 9)
#     norm_groups: int = 32
#     dropout: float = 0.2
#
#
# # ----------------------------- Utilities -----------------------------
#
# def drop_path(x, drop_prob: float = 0., training: bool = False):
#     if drop_prob == 0. or not training:
#         return x
#     keep_prob = 1 - drop_prob
#     shape = (x.shape[0],) + (1,) * (x.ndim - 1)
#     random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
#     random_tensor.floor_()
#     output = x.div(keep_prob) * random_tensor
#     return output
#
#
# class DropPath(nn.Module):
#     def __init__(self, drop_prob=None):
#         super().__init__()
#         self.drop_prob = drop_prob
#
#     def forward(self, x):
#         return drop_path(x, self.drop_prob, self.training)
#
#
# class RMSNorm(nn.Module):
#     def __init__(self, d_model: int, eps: float = 1e-5):
#         super().__init__()
#         self.eps = eps
#         self.weight = nn.Parameter(torch.ones(d_model))
#
#     def forward(self, x):
#         # x: [B, T, D]
#         norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
#         return x * norm * self.weight
#
#
# # ----------------------------- Model -----------------------------
#
# class Mamba(nn.Module):
#     def __init__(self, args: ModelArgs):
#         super().__init__()
#         self.args = args
#         self.layers = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])
#         self.norm_f = RMSNorm(args.d_model)
#
#     def forward(self, input_ids):
#         # input_ids: [B, T, D]
#         x = self.norm_f(input_ids)
#         for layer in self.layers:
#             x = layer(x)
#         return x
#
#
# class ResidualBlock(nn.Module):
#     def __init__(self, args: ModelArgs):
#         super().__init__()
#         self.args = args
#         self.mixer = MambaBlock(args)
#         self.norm = RMSNorm(args.d_model)
#
#     def forward(self, x):
#         # x: [B, T, D]
#         return self.mixer(self.norm(x)) + x
#
#
# class MambaBlock(nn.Module):
#     """
#     Improved dual-path friendly Mamba block for boundary detection:
#     - Multi-scale depthwise conv -> concat -> Linear(4d->d) cross-scale mixing (+Dropout)
#     - True backward scan optional; learnable forward/backward weights (w_f, w_b)
#     - Residual-style multiplicative fusion: y = y + y * x_local
#     - Unified gating at the end and a post LayerNorm for stability
#     """
#
#     def __init__(self, args: ModelArgs):
#         super().__init__()
#         self.args = args
#         d_in = args.d_model
#         d_inner = args.d_inner
#         self.bidirectional = getattr(args, "bidirectional", True)
#
#         # In-projection into three streams: local conv branch, gate, and SSM input
#         self.in_proj = nn.Linear(d_in, d_inner * 3, bias=args.bias)
#
#         # Multi-scale depthwise conv stack (operate on [B, C=d_inner, T])
#         ks = getattr(args, "kernels", (3, 5, 7, 9))
#         self.temporal_conv = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv1d(
#                     in_channels=d_inner,
#                     out_channels=d_inner,
#                     kernel_size=k,
#                     groups=d_inner,
#                     padding="same",
#                     bias=args.conv_bias
#                 ),
#                 # 可按需在小 batch 下改为 LayerNorm；这里沿用 BN 习惯，与旧版接近
#                 nn.BatchNorm1d(d_inner),
#                 nn.GELU()
#             ) for k in ks
#         ])
#
#         # Cross-scale linear remixing: [B,T, len(ks)*d_inner] -> [B,T,d_inner]
#         self.mix_linear = nn.Linear(d_inner * len(ks), d_inner, bias=True)
#         nn.init.xavier_uniform_(self.mix_linear.weight)
#         nn.init.zeros_(self.mix_linear.bias)
#
#         self.dropout = nn.Dropout(getattr(args, "dropout", 0.2))
#
#         # SSM parameterization (same as before)
#         self.x_proj = nn.Linear(d_inner, args.dt_rank + args.d_state * 2, bias=False)
#         self.dt_proj = nn.Linear(args.dt_rank, d_inner, bias=True)
#
#         A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=d_inner)
#         self.A_log = nn.Parameter(torch.log(A))
#         self.D = nn.Parameter(torch.ones(d_inner))
#
#         # Learnable weights for forward/backward contributions
#         self.w_f = nn.Parameter(torch.tensor(1.0))
#         self.w_b = nn.Parameter(torch.tensor(0.5))
#
#         # Stabilization and output projection
#         self.post_ln = nn.LayerNorm(d_inner)
#         self.out_proj = nn.Linear(d_inner, d_in, bias=args.bias)
#
#     # ------------------------------ SSM core ------------------------------
#
#     def ssm(self, x):
#         """
#         Run SSM on input x: [B, T, d_inner]
#         """
#         (d_inner, n) = self.A_log.shape
#         A = -torch.exp(self.A_log.float())      # [d_inner, n]
#         D = self.D.float()                      # [d_inner]
#
#         x_dbl = self.x_proj(x)                  # [B, T, dt_rank + 2*n]
#         (delta, B, C) = x_dbl.split(
#             split_size=[self.args.dt_rank, n, n], dim=-1
#         )
#         delta = F.softplus(self.dt_proj(delta))  # [B, T, d_inner]
#
#         y = self.selective_scan_parallel(x, delta, A, B, C, D)  # [B, T, d_inner]
#         return y
#
#     def selective_scan_parallel(self, u, delta, A, B, C, D):
#         """
#         Parallel selective scan using cumulative products/sums.
#
#         u:     [B, T, d]
#         delta: [B, T, d]
#         A:     [d, n]
#         B:     [B, T, n]
#         C:     [B, T, n]
#         D:     [d]
#         """
#         (b, l, d_in) = u.shape
#         n = A.shape[1]
#
#         # Discretization
#         deltaA = torch.exp(einsum(delta, A, 'b l d, d n -> b l d n'))         # [B, T, d, n]
#         deltaB_u = einsum(delta, B, u, 'b l d, b l n, b l d -> b l d n')      # [B, T, d, n]
#
#         # Parallel scan
#         eps = 1e-12
#         S = torch.cumprod(deltaA + eps, dim=1)                                # [B, T, d, n]
#         x = S * torch.cumsum(deltaB_u / (S + eps), dim=1)                     # [B, T, d, n]
#
#         # Output combine with C and skip via D
#         y = torch.sum(x * C[:, :, None, :], dim=-1)                           # [B, T, d]
#         y = y + u * D                                                         # [B, T, d]
#         return y
#
#     # ------------------------------ Forward ------------------------------
#
#     def forward(self, x):
#         """
#         x: [B, T, d_model]
#         returns: [B, T, d_model]
#         """
#         B, T, _ = x.shape
#
#         # In projection -> three streams
#         x_and_res = self.in_proj(x)                     # [B, T, 3*d_inner]
#         x_feat, z, res = x_and_res.chunk(3, dim=-1)     # [B, T, d_inner] each
#
#         # Multi-scale local conv once (on [B, C, T])
#         xf = x_feat.transpose(1, 2)                     # [B, d_inner, T]
#         ms = [conv(xf) for conv in self.temporal_conv]  # list of [B, d_inner, T]
#         x_local = torch.cat(ms, dim=1).transpose(1, 2)  # [B, T, len(ks)*d_inner]
#         x_local = F.silu(x_local)
#         x_local = self.mix_linear(x_local)              # [B, T, d_inner]
#         x_local = self.dropout(x_local)
#
#         # SSM forward (+ optional backward with flip)
#         y_fwd = self.ssm(res)                           # [B, T, d_inner]
#         if self.bidirectional:
#             y_bwd = self.ssm(res.flip(1)).flip(1)       # [B, T, d_inner]
#             y = self.w_f * y_fwd + self.w_b * y_bwd
#         else:
#             y = y_fwd
#
#         # Residual-style multiplicative fusion to avoid energy collapse
#         y = y + y * x_local                             # [B, T, d_inner]
#
#         # Unified gating at the end
#         y = y * F.silu(z)
#
#         # Stabilize distribution
#         y = self.post_ln(y)
#
#         # Project back to model dimension
#         out = self.out_proj(y)                          # [B, T, d_model]
#         return out


# -*- coding: utf-8 -*-
"""
Simple, minimal implementation of Mamba in one file of PyTorch with parallel selective scan.
Modified with dual-path friendly fusion for scene boundary detection.

This version includes:
- Step 1: Sigmoid-constrained bidirectional weights (w_f, w_b) for stability
- Step 2: Post-SSM lightweight depthwise smoothing (kernel=3, mean init)

References:
[1] Mamba: Linear-Time Sequence Modeling with Selective State Spaces
[2] The Annotated S4
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, einsum


# ----------------------------- Config -----------------------------

@dataclass
class ModelArgs:
    d_model: int
    d_inner: int
    n_layer: int
    size: int
    d_state: int = 16
    expand: int = 2
    dt_rank: int = math.ceil(2048 / 16)
    d_conv: int = 4
    conv_bias: bool = True
    bias: bool = False
    # Optional knobs (will be read via getattr)
    bidirectional: bool = True
    kernels: tuple = (3, 5, 7, 9)
    norm_groups: int = 32
    dropout: float = 0.2


# ----------------------------- Utilities -----------------------------

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        # x: [B, T, D]
        var = x.pow(2).mean(-1, keepdim=True)
        x_norm = x * torch.rsqrt(var + self.eps)
        return x_norm * self.weight  # [B, T, D]


# ----------------------------- Model -----------------------------

class Mamba(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.layers = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])
        self.norm_f = RMSNorm(args.d_model)

    def forward(self, input_ids):
        # input_ids: [B, T, D]
        x = self.norm_f(input_ids)
        for layer in self.layers:
            x = layer(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.mixer = MambaBlock(args)
        self.norm = RMSNorm(args.d_model)

    def forward(self, x):
        # x: [B, T, D]
        return self.mixer(self.norm(x)) + x


class MambaBlock(nn.Module):
    """
    Improved dual-path friendly Mamba block for boundary detection:
    - Multi-scale depthwise conv -> concat -> Linear(4d->d) cross-scale mixing (+Dropout)
    - True backward scan optional; learnable forward/backward weights (sigmoid constrained)
    - Post-SSM lightweight depthwise smoothing (kernel=3, mean init)
    - Residual-style multiplicative fusion: y = y + y * x_local
    - Unified gating at the end and a post LayerNorm for stability
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        d_in = args.d_model
        d_inner = args.d_inner
        self.bidirectional = getattr(args, "bidirectional", True)

        # In-projection into three streams: local conv branch, gate, and SSM input
        self.in_proj = nn.Linear(d_in, d_inner * 3, bias=args.bias)

        # Multi-scale depthwise conv stack (operate on [B, C=d_inner, T])
        ks = getattr(args, "kernels", (3, 5, 7, 9))
        self.temporal_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels=d_inner,
                    out_channels=d_inner,
                    kernel_size=k,
                    groups=d_inner,
                    padding="same",
                    bias=args.conv_bias
                ),
                nn.BatchNorm1d(d_inner),
                nn.GELU()
            ) for k in ks
        ])

        # Cross-scale linear remixing: [B,T, len(ks)*d_inner] -> [B,T,d_inner]
        self.mix_linear = nn.Linear(d_inner * len(ks), d_inner, bias=True)
        nn.init.xavier_uniform_(self.mix_linear.weight)
        nn.init.zeros_(self.mix_linear.bias)

        self.dropout = nn.Dropout(getattr(args, "dropout", 0.2))

        # SSM parameterization
        self.x_proj = nn.Linear(d_inner, args.dt_rank + args.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(args.dt_rank, d_inner, bias=True)

        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))

        # Step 1: Learnable weights for forward/backward contributions (sigmoid-constrained)
        self.w_f_raw = nn.Parameter(torch.tensor(0.0))    # sigmoid(0.0) ~ 0.5
        self.w_b_raw = nn.Parameter(torch.tensor(-0.7))   # sigmoid(-0.7) ~ 0.33

        # Step 2: Post-SSM lightweight smoothing (depthwise conv, kernel=3, init as mean filter)
        self.post_ssm_smooth = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=3,
            padding=1,
            groups=d_inner,
            bias=False
        )
        with torch.no_grad():
            self.post_ssm_smooth.weight.zero_()
            # depthwise: weight shape [d_inner, 1, 3]
            self.post_ssm_smooth.weight[:, 0, 0] = 1.0 / 3.0
            self.post_ssm_smooth.weight[:, 0, 1] = 1.0 / 3.0
            self.post_ssm_smooth.weight[:, 0, 2] = 1.0 / 3.0

        # Stabilization and output projection
        self.post_ln = nn.LayerNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, d_in, bias=args.bias)

    # ------------------------------ SSM core ------------------------------

    def ssm(self, x):
        """
        Run SSM on input x: [B, T, d_inner]
        """
        (d_inner, n) = self.A_log.shape
        A = -torch.exp(self.A_log.float())      # [d_inner, n]
        D = self.D.float()                      # [d_inner]

        x_dbl = self.x_proj(x)                  # [B, T, dt_rank + 2*n]
        (delta, B, C) = x_dbl.split(
            split_size=[self.args.dt_rank, n, n], dim=-1
        )
        delta = F.softplus(self.dt_proj(delta))  # [B, T, d_inner]

        y = self.selective_scan_parallel(x, delta, A, B, C, D)  # [B, T, d_inner]
        return y

    def selective_scan_parallel(self, u, delta, A, B, C, D):
        """
        Parallel selective scan using cumulative products/sums.

        u:     [B, T, d]
        delta: [B, T, d]
        A:     [d, n]
        B:     [B, T, n]
        C:     [B, T, n]
        D:     [d]
        """
        (b, l, d_in) = u.shape
        n = A.shape[1]

        # Discretization
        deltaA = torch.exp(einsum(delta, A, 'b l d, d n -> b l d n'))         # [B, T, d, n]
        deltaB_u = einsum(delta, B, u, 'b l d, b l n, b l d -> b l d n')      # [B, T, d, n]

        # Parallel scan
        eps = 1e-12
        S = torch.cumprod(deltaA + eps, dim=1)                                # [B, T, d, n]
        x = S * torch.cumsum(deltaB_u / (S + eps), dim=1)                     # [B, T, d, n]

        # Output combine with C and skip via D
        y = torch.sum(x * C[:, :, None, :], dim=-1)                           # [B, T, d]
        y = y + u * D                                                         # [B, T, d]
        return y

    # ------------------------------ Forward ------------------------------

    def forward(self, x):
        """
        x: [B, T, d_model]
        returns: [B, T, d_model]
        """
        B, T, _ = x.shape

        # In projection -> three streams
        x_and_res = self.in_proj(x)                     # [B, T, 3*d_inner]
        x_feat, z, res = x_and_res.chunk(3, dim=-1)     # [B, T, d_inner] each

        # Multi-scale local conv once (on [B, C, T])
        xf = x_feat.transpose(1, 2)                     # [B, d_inner, T]
        ms = [conv(xf) for conv in self.temporal_conv]  # list of [B, d_inner, T]
        x_local = torch.cat(ms, dim=1).transpose(1, 2)  # [B, T, len(ks)*d_inner]
        x_local = F.silu(x_local)
        x_local = self.mix_linear(x_local)              # [B, T, d_inner]
        x_local = self.dropout(x_local)

        # SSM forward (+ optional backward with flip)
        y_fwd = self.ssm(res)                           # [B, T, d_inner]
        if self.bidirectional:
            y_bwd = self.ssm(res.flip(1)).flip(1)       # [B, T, d_inner]
            # Step 1: sigmoid-constrained weights
            w_f = torch.sigmoid(self.w_f_raw)
            w_b = torch.sigmoid(self.w_b_raw)
            y = w_f * y_fwd + w_b * y_bwd
        else:
            y = y_fwd

        # Step 2: lightweight smoothing between SSM and multiplicative fusion
        y = y.transpose(1, 2)                           # [B, d_inner, T]
        y = self.post_ssm_smooth(y)                     # depthwise conv
        y = y.transpose(1, 2)                           # [B, T, d_inner]

        # Residual-style multiplicative fusion to avoid energy collapse
        y = y + y * x_local                             # [B, T, d_inner]

        # Unified gating at the end
        y = y * F.silu(z)

        # Stabilize distribution
        y = self.post_ln(y)

        # Project back to model dimension
        out = self.out_proj(y)                          # [B, T, d_model]
        return out
