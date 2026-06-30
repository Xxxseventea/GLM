import torch
import torch.nn as nn
import torch.nn.functional as F
class FrameWiseHead(nn.Module):
    """
    无 Query 的 Proposal-free 头部（K=1）：
    - 输入: x (B, L, N)
    - 输出:
        score_logit: (B,)      是否有边界（分类logit）
        center:      (B,)      边界在当前窗口内的相对位置 ∈ [0,1]
    - 时序聚合:
        默认用 avg pooling（建议换成 1D Conv/Transformer 以提升定位）
    """
    def __init__(self, in_features: int, hidden: int = 256, dropout: float = 0.1, agg: str = "avg"):
        super().__init__()
        self.agg = agg
        self.proj = nn.Linear(in_features, hidden)
        self.dropout = nn.Dropout(dropout)
        self.cls = nn.Linear(hidden, 1)
        self.reg = nn.Linear(hidden, 1)

        # 可选：简单的时序卷积替代 avg 池化（注：需把 (B,L,N) 转换为 (B,N,L) 再 Conv1d）
        self.use_conv = False
        if self.use_conv:
            self.temporal = nn.Sequential(
                nn.Conv1d(in_features, hidden, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),  # (B,hidden,1) -> 全局聚合
            )

    def forward(self, x: torch.Tensor):
        B, L, N = x.shape

        if getattr(self, 'use_conv', False):
            # (B,L,N) -> (B,N,L)
            xc = x.transpose(1, 2)
            h = self.temporal(xc).squeeze(-1)  # (B,hidden)
        else:
            # 简单的时序平均
            xt = x.mean(dim=1)                 # (B, N)
            h = F.relu(self.proj(xt))          # (B, hidden)

        h = self.dropout(h)
        score_logit = self.cls(h).squeeze(1)   # (B,)
        center_raw = self.reg(h).squeeze(1)    # (B,)
        center = torch.sigmoid(center_raw)     # 映射到 [0,1]
        return score_logit, center