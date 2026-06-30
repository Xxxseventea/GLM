import torch
import torch.nn as nn
import torch.nn.functional as F


class FrameWiseHead(nn.Module):
    """
    你给的头，原样照搬
    - 输入: x (B, L, N)
    - 输出:
        score_logit: (B,)      是否有边界（分类logit）
        center:      (B,)      边界在当前窗口内的相对位置 ∈ [0,1]
    """
    def __init__(self, in_features: int, hidden: int = 256, dropout: float = 0.1, agg: str = "avg"):
        super().__init__()
        self.agg = agg
        self.proj = nn.Linear(in_features, hidden)
        self.dropout = nn.Dropout(dropout)
        self.cls = nn.Linear(hidden, 1)
        self.reg = nn.Linear(hidden, 1)

        # 可选：简单的时序卷积替代 avg 池化
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


class ContextAwareTransformer(nn.Module):
    """
    Context-Aware Transformer
    - 输入: x [B, 21, 2048]
    - 使用多层 TransformerEncoder 在时间维上做上下文建模
    - 输出: FrameWiseHead(score_logit, center)
    """

    def __init__(
        self,
        in_dim: int = 2048,
        d_model: int = 512,
        n_heads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_proj = nn.Linear(in_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,  # 这样就可以用 (B, L, D)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 可学习的位置编码：长度固定为 21（你的时间长度）
        self.pos_embed = nn.Parameter(torch.zeros(1, 21, d_model))

        # 输出头：把上下文编码后的特征传给你的 FrameWiseHead
        # 注意：FrameWiseHead 的 in_features 要等于 d_model
        self.head = FrameWiseHead(in_features=d_model, hidden=256, dropout=dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor):
        """
        x: [B, 21, 2048]
        返回:
            score_logit: (B,)
            center:      (B,)
        """
        B, L, C = x.shape
        assert L == 21, f"当前实现假定 L=21，实际 L={L}"
        # 1) 映射到 Transformer 维度
        feat = self.input_proj(x)  # [B, 21, d_model]

        # 2) 加上位置编码
        feat = feat + self.pos_embed[:, :L, :]  # [B, 21, d_model]

        # 3) 时序上下文建模
        ctx = self.encoder(feat)  # [B, 21, d_model]

        # 4) 使用你的 FrameWiseHead
        score_logit, center = self.head(ctx)  # (B,), (B,)

        return score_logit, center


if __name__ == "__main__":
    B = 8
    x = torch.randn(B, 21, 2048)
    model = ContextAwareTransformer(
        in_dim=2048,
        d_model=512,
        n_heads=8,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
    )
    score_logit, center = model(x)
    print("score_logit:", score_logit.shape)  # (B,)
    print("center:", center.shape)            # (B,)