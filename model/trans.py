import torch
import torch.nn as nn
import torch.nn.functional as F


class FrameWiseHead(nn.Module):
    """
    无 Query 的 Proposal-free 头部（K=1）：
    - 输入: x (B, L, N)
    - 输出:
        score_logit: (B,)      是否有边界（分类 logit）
        center:      (B,)      边界在当前窗口内的相对位置 ∈ [0,1]
    """
    def __init__(self, in_features: int, hidden: int = 256,
                 dropout: float = 0.1, agg: str = "avg"):
        super().__init__()
        self.agg = agg
        self.proj = nn.Linear(in_features, hidden)
        self.dropout = nn.Dropout(dropout)
        self.cls = nn.Linear(hidden, 1)
        self.reg = nn.Linear(hidden, 1)

        self.use_conv = False
        if self.use_conv:
            self.temporal = nn.Sequential(
                nn.Conv1d(in_features, hidden, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )

    def forward(self, x: torch.Tensor):
        B, L, N = x.shape

        if getattr(self, 'use_conv', False):
            xc = x.transpose(1, 2)
            h = self.temporal(xc).squeeze(-1)  # (B,hidden)
        else:
            xt = x.mean(dim=1)                 # (B,N)
            h = F.relu(self.proj(xt))          # (B,hidden)

        h = self.dropout(h)
        score_logit = self.cls(h).squeeze(1)   # (B,)
        center_raw = self.reg(h).squeeze(1)    # (B,)
        center = torch.sigmoid(center_raw)     # [0,1]
        return score_logit, center


class TransformerModel(nn.Module):
    """
    纯 Transformer 版本：
    - 输入: x [B, 21, 2048]
    - 主干: 标准 TransformerEncoder
    - 输出: FrameWiseHead(score_logit, center)
    """
    def __init__(
        self,
        in_dim: int = 2048,
        d_model: int = 512,
        n_heads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_len: int = 21,
    ):
        super().__init__()

        self.d_model = d_model
        self.max_len = max_len

        # 输入线性映射 2048 -> d_model
        self.input_proj = nn.Linear(in_dim, d_model)

        # 学习型位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,    # 输入输出都是 (B, L, D)
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.norm_out = nn.LayerNorm(d_model)

        self.head = FrameWiseHead(
            in_features=d_model,
            hidden=256,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor):
        """
        x: [B, 21, 2048]
        """
        B, L, C = x.shape
        assert L <= self.max_len, f"L={L} 超过 max_len={self.max_len}"

        # 1) 线性映射
        x = self.input_proj(x)                 # (B,L,d_model)

        # 2) 加位置编码
        pos = self.pos_embed[:, :L, :]        # (1,L,d_model)
        x = x + pos

        # 3) Transformer 编码
        x = self.transformer(x)               # (B,L,d_model)
        x = self.norm_out(x)                  # (B,L,d_model)

        # 4) 检测头
        score_logit, center = self.head(x)    # (B,), (B,)
        return score_logit, center


if __name__ == "__main__":
    B = 2
    x = torch.randn(B, 21, 2048)
    model = SimpleTransformerSceneModel()
    score_logit, center = model(x)
    print(score_logit.shape, center.shape)  # torch.Size([2]) torch.Size([2])