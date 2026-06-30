# networks/subnet.py
"""
Subnet 层：基于 WSN (Winning SubNetworks) 思想
- 每层维护一个可学习的 score
- 根据 sparsity 取 top-k 形成二值 mask
- forward 时用 weight * mask
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GetSubnet(torch.autograd.Function):
    """从 score 取 top-k 形成 mask（forward），梯度直通（backward）"""
    @staticmethod
    def forward(ctx, scores, sparsity):
        # sparsity = 保留比例 (0~1)
        k = int(sparsity * scores.numel())
        if k < 1:
            k = 1
        out = torch.zeros_like(scores)
        if k >= scores.numel():
            out.fill_(1.0)
            return out
        # 取 score 最大的 k 个
        _, idx = scores.flatten().topk(k)
        flat = out.flatten()
        flat[idx] = 1.0
        return flat.view_as(scores)

    @staticmethod
    def backward(ctx, g):
        # 梯度直通
        return g, None


class SubnetLinear(nn.Linear):
    """
    带 mask 的 Linear：
    - 训练时：用自身 score 生成 mask（如果外部没传 mask）
    - 测试时：用外部传入的 mask
    """
    def __init__(self, in_features, out_features, bias=True, sparsity=0.5):
        super().__init__(in_features, out_features, bias=bias)
        self.sparsity = sparsity
        # 每个权重对应一个 score（可学习）
        self.weight_score = nn.Parameter(torch.empty_like(self.weight))
        nn.init.kaiming_uniform_(self.weight_score, a=math.sqrt(5))

    def get_mask(self):
        """返回当前任务的 binary mask"""
        with torch.no_grad():
            mask = GetSubnet.apply(self.weight_score.abs(), self.sparsity)
        return mask.detach()

    def forward(self, x, weight_mask=None, mode='train'):
        if weight_mask is None:
            # 训练阶段：从 score 算 mask
            mask = GetSubnet.apply(self.weight_score.abs(), self.sparsity)
        else:
            # 测试阶段：用外部 mask
            mask = weight_mask
        w = self.weight * mask
        return F.linear(x, w, self.bias)


class SubnetConv2d(nn.Conv2d):
    """同上，Conv2d 版本（备用）"""
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 sparsity=0.5):
        super().__init__(in_channels, out_channels, kernel_size,
                         stride, padding, dilation, groups, bias)
        self.sparsity = sparsity
        self.weight_score = nn.Parameter(torch.empty_like(self.weight))
        nn.init.kaiming_uniform_(self.weight_score, a=math.sqrt(5))

    def get_mask(self):
        with torch.no_grad():
            mask = GetSubnet.apply(self.weight_score.abs(), self.sparsity)
        return mask.detach()

    def forward(self, x, weight_mask=None, mode='train'):
        if weight_mask is None:
            mask = GetSubnet.apply(self.weight_score.abs(), self.sparsity)
        else:
            mask = weight_mask
        w = self.weight * mask
        return F.conv2d(x, w, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
