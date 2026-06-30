import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================
# 你的头：原样
# =========================================
class FrameWiseHead(nn.Module):
    """
    无 Query 的 Proposal-free 头部（K=1）：
    - 输入: x (B, L, N)
    - 输出:
        score_logit: (B,)      是否有边界（分类logit）
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


# =========================================
# 用库里的 S4 做 Gated SSM
# =========================================
# 根据你实际的库修改这一行：
from model.s4.trans4mer_s4 import S4   # 如果你的包叫别的，比如 s4torch.s4nd，就换成对应的


class GatedS4(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.proj_u = nn.Linear(d_model, d_model)
        self.proj_v = nn.Linear(d_model, d_model)

        # 根据你自己的路径导入
        from model.s4.trans4mer_s4 import S4
        self.ssm = S4(d_model)

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        """
        x: (B, L, D)  ->  S4: (B, D, L)  ->  out: (B, L, D)
        """
        # 1) 线性 + 激活
        u = F.gelu(self.proj_u(x))       # (B,L,D)
        v = F.gelu(self.proj_v(x))       # (B,L,D)

        # 2) 调整维度喂给 S4
        u_ssm_in = u.transpose(1, 2)     # (B,D,L)

        # 3) 调用 S4：注意这里解包 tuple
        s4_out = self.ssm(u_ssm_in)
        if isinstance(s4_out, tuple):
            y_ssm, _ = s4_out            # 忽略 state
        else:
            y_ssm = s4_out               # 有的实现只返回 y

        # 4) 再转回 (B,L,D)
        if y_ssm.dim() == 3:
            h = y_ssm.transpose(1, 2)    # (B,L,D)
        elif y_ssm.dim() == 4:
            # 万一你的 S4 返回 (B,C,H,L)，这里假设 C=1，自己按需要改
            B, C, H, L = y_ssm.shape
            assert C == 1, f"unexpected C={C} in S4 output"
            h = y_ssm[:, 0].transpose(1, 2)  # (B,L,H)
        else:
            raise RuntimeError(f"unexpected S4 output shape: {y_ssm.shape}")

        # 5) gating + 输出
        y_out = self.out_proj(h * v)     # (B,L,D)
        y_out = self.dropout(y_out)
        return y_out


# =========================================
# S4A Block: Intra self-attn + Inter S4
# =========================================
class S4ABlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int,
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()

        # Intra: Multi-head self-attention + MLP
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,   # (B,L,D)
        )
        self.norm2 = nn.LayerNorm(d_model)
        mlp_hidden = int(d_model * mlp_ratio)
        self.mlp1 = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, d_model),
            nn.Dropout(dropout),
        )

        # Inter: Gated S4 + MLP
        self.norm3 = nn.LayerNorm(d_model)
        self.gs4 = GatedS4(d_model=d_model, dropout=dropout)
        self.norm4 = nn.LayerNorm(d_model)
        self.mlp2 = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor):
        # ----- Intra (self-attention) -----
        x_in = x
        x_norm = self.norm1(x_in)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)  # (B,L,D)
        x = x_in + attn_out

        x_in2 = x
        x_norm2 = self.norm2(x_in2)
        x = x_in2 + self.mlp1(x_norm2)

        # ----- Inter (S4) -----
        z_in = x
        z_norm = self.norm3(z_in)
        z = z_in + self.gs4(z_norm)

        z_in2 = z
        z_norm2 = self.norm4(z_in2)
        z = z_in2 + self.mlp2(z_norm2)

        return z


# =========================================
# 整体模型：输入 [B,21,2048]，输出你的头
# =========================================
class S4SceneModel(nn.Module):
    """
    状态空间模型版本的场景边界检测：
    - 输入: x [B, 21, 2048]
    - 主干: 若干层 S4ABlock
    - 输出: FrameWiseHead(score_logit, center)
    """
    def __init__(
        self,
        in_dim: int = 2048,
        d_model: int = 512,
        n_heads: int = 8,
        num_layers: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_proj = nn.Linear(in_dim, d_model)

        # 这里直接固定 21 个位置编码；如果之后窗口变，可以改成 max_len
        self.pos_embed = nn.Parameter(torch.zeros(1, 21, d_model))

        self.blocks = nn.ModuleList([
            S4ABlock(d_model, n_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

        self.norm_out = nn.LayerNorm(d_model)

        # 检测头：使用你给的 Head
        self.head = FrameWiseHead(
            in_features=d_model,
            hidden=256,
            dropout=dropout,
        )

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor):
        """
        x: [B, 21, 2048]
        返回:
            score_logit: (B,)
            center:      (B,)
        """
        B, L, C = x.shape
        assert L == 21, f"当前实现假定 L=21，实际 L={L}"

        # 1) 投影到 d_model
        x = self.input_proj(x)                 # (B,21,d_model)

        # 2) 加位置编码
        x = x + self.pos_embed[:, :L, :]       # (B,21,d_model)

        # 3) 多层 S4ABlock
        for blk in self.blocks:
            x = blk(x)                         # (B,21,d_model)

        x = self.norm_out(x)                   # (B,21,d_model)

        # 4) 你的 FrameWiseHead
        score_logit, center = self.head(x)     # (B,), (B,)
        return score_logit, center


# =========================================
# 简单自测
# =========================================
if __name__ == "__main__":
    B = 2
    x = torch.randn(B, 21, 2048)
    model = S4SceneModel(
        in_dim=2048,
        d_model=512,
        n_heads=8,
        num_layers=3,
        mlp_ratio=4.0,
        dropout=0.1,
    )
    score_logit, center = model(x)
    print("score_logit:", score_logit.shape)
    print("center:", center.shape)