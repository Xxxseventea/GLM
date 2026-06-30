from typing import Tuple, Dict, List

import einops
import torch
from torch import nn
import torch.nn.functional as F
from model.backbone.mymamba import Mamba,ModelArgs
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math
from torch.nn import Linear

from model.detector.event_detector import FrameWiseHead
from model.detector.scene_detector import MlpHead


class LocalUncertaintyAwareGraphAttentionLite(nn.Module):
    """
    简约不减效版：带不确定性感知的局部图注意力
    - 局部窗口: unfold
    - 相似度: 余弦 + 温度
    - 不确定性: log-variance 或 confidence 门控
    - 邻居抑制: 动态有效邻居数 soft-count
    - 聚合: 轻量多头注意力
    I/O:
        forward(x, place=None):
            x: (B, T, Cx)
            place: (B, T, Cp) or None
        return:
            out: detector 输出 (形状取决于 LatentDetector)
    """

    def __init__(
        self,
        dim_in: int,                    # Cx: 原始特征维度（不含 place）
        place_dim: int = 0,             # Cp: 位置特征维度（若 place 不传，可置 0）
        num_heads: int = 8,
        window_size: int = 9,           # 奇数
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        use_relative_pos_bias: bool = True,
        uncertainty_mode: str = "variance",  # "variance" | "confidence"
        clamp_logvar: tuple = (-6.0, 2.0),
        neighbor_temp: float = 0.5,
        sim_temperature: float = 0.07,
        use_residual: bool = True,
        norm: str = "ln",               # "ln" | "bn"
        enable_refine_in_train: bool = False,
        size : int = 21
    ):
        super().__init__()
        assert window_size % 2 == 1, "window_size should be odd"
        assert uncertainty_mode in ["variance", "confidence"]

        C = dim_in + 128
        assert C % num_heads == 0, "channels must be divisible by num_heads"
        self.enable_refine_in_train = enable_refine_in_train
        self.C = C
        self.num_heads = num_heads
        self.head_dim = C // num_heads
        self.W = window_size
        # 归一化
        if norm == "ln":
            self.norm = nn.LayerNorm(C)
        elif norm == "bn":
            self.norm = nn.BatchNorm1d(C)
        else:
            raise ValueError("norm must be 'ln' or 'bn'")

        # QKV 投影（保持维度 C）
        self.q_proj = nn.Linear(C, C, bias=True)
        self.k_proj = nn.Linear(C, C, bias=True)
        self.v_proj = nn.Linear(C, C, bias=True)

        # 不确定性：逐头预测 H
        self.uncert_proj = nn.Linear(C, num_heads)

        # 相对位置偏置
        self.use_relative_pos_bias = use_relative_pos_bias
        if use_relative_pos_bias:
            self.rel_pos_bias = nn.Parameter(torch.zeros(num_heads, window_size))
            nn.init.trunc_normal_(self.rel_pos_bias, std=0.02)
        else:
            self.register_parameter("rel_pos_bias", None)

        # 输出投影
        self.out_proj = nn.Linear(C, C)
        self.proj_drop = nn.Dropout(dropout)
        self.attn_drop = nn.Dropout(attn_dropout)

        self.use_residual = use_residual
        self.uncertainty_mode = uncertainty_mode
        self.clamp_logvar = clamp_logvar
        self.neighbor_temp = neighbor_temp
        self.sim_temperature = sim_temperature

        # 你的检测头
        self.pos = PositionEmbedding(size=size)
        self.mamba = LSUBoundaryTendencyDetector(size=size)
        #
        self.detector = MlpHead(in_dim = 2176,hid_dim = 512)
    @torch.no_grad()
    def _local_unfold(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C) -> (B, T, W, C)
        使用 unfold 保证维度与边界更稳健
        """
        B, T, C = x.shape
        h = self.W // 2
        x_pad = F.pad(x.transpose(1, 2), (h, h), mode="reflect").transpose(1, 2)  # (B, T+2h, C)
        x_unf = x_pad.transpose(1, 2).unfold(dimension=-1, size=self.W, step=1)   # (B, C, T, W)
        x_win = x_unf.permute(0, 2, 3, 1).contiguous()                              # (B, T, W, C)
        return x_win

    def _maybe_norm(self, x: torch.Tensor) -> torch.Tensor:
        # 支持 BN 或 LN
        if isinstance(self.norm, nn.BatchNorm1d):
            return self.norm(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(x)

    def forward(self, x: torch.Tensor, task_name):
        """
        x: (B, T, Cx); place: (B, T, Cp) or None
        return: detector(x'): 形状由 LatentDetector 决定
        """
        B, T, Cx = x.shape
        x = self.pos(x)
        xn = self._maybe_norm(x)  # (B,T,C)

        # QKV
        q = self.q_proj(xn)  # (B,T,C)
        k = self.k_proj(xn)
        v = self.v_proj(xn)

        # 局部窗口（一次性拿到）
        k_win = self._local_unfold(k)  # (B,T,W,C)
        v_win = self._local_unfold(v)  # (B,T,W,C)

        # 多头拆分
        def split_q(t):  # (B,T,C) -> (B,H,T,D)
            return t.view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        def split_kwv(t):  # (B,T,W,C) -> (B,H,T,W,D)
            return t.view(B, T, self.W, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)

        qh = split_q(q)            # (B,H,T,D)
        kh = split_kwv(k_win)      # (B,H,T,W,D)
        vh = split_kwv(v_win)      # (B,H,T,W,D)

        # 余弦相似度 + 温度
        qn = F.normalize(qh, dim=-1)
        kn = F.normalize(kh, dim=-1)
        sim = (qn.unsqueeze(3) * kn).sum(-1) / self.sim_temperature  # (B,H,T,W)
        if self.use_relative_pos_bias and self.rel_pos_bias is not None:
            sim = sim + self.rel_pos_bias.view(1, self.num_heads, 1, self.W)
        attn_logits = sim
        attn = F.softmax(attn_logits, dim=-1)
        attn = self.attn_drop(attn)
        # 聚合
        out = torch.einsum("bhtw,bhtwd->bhtd", attn, vh)                        # (B,H,T,D)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, self.C)          # (B,T,C)

        # 输出投影 + 残差
        out = self.proj_drop(self.out_proj(out))                                # (B,T,C)
        # 序列表征（供两分支共用）
        seq_feat = self.mamba(out)  # (B, T, C)

        # 调用 detector（可能是 MlpHead，也可能是 nn.Identity）
        # out_det = self.detector[task_name](seq_feat)
        out_det = self.detector(seq_feat)

        # ---- 兼容不同类型的 detector 输出 ----
        if isinstance(out_det, tuple):
            # MlpHead 一般返回 (score_logit, center)
            if len(out_det) == 2:
                score_logit, center = out_det
            else:
                score_logit, center = out_det[0], None
        else:
            # nn.Identity() 或其他单输出层，返回一个 tensor
            score_logit, center = out_det, None

        return score_logit, center

    def features_last(self, x: torch.Tensor):

        """
        x: (B, T, Cx); place: (B, T, Cp) or None
        return: detector(x'): 形状由 LatentDetector 决定
        """
        B, T, Cx = x.shape
        x = self.pos(x)
        xn = self._maybe_norm(x)  # (B,T,C)

        # QKV
        q = self.q_proj(xn)  # (B,T,C)
        k = self.k_proj(xn)
        v = self.v_proj(xn)

        # 局部窗口（一次性拿到）
        k_win = self._local_unfold(k)  # (B,T,W,C)
        v_win = self._local_unfold(v)  # (B,T,W,C)

        # 多头拆分
        def split_q(t):  # (B,T,C) -> (B,H,T,D)
            return t.view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        def split_kwv(t):  # (B,T,W,C) -> (B,H,T,W,D)
            return t.view(B, T, self.W, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)

        qh = split_q(q)            # (B,H,T,D)
        kh = split_kwv(k_win)      # (B,H,T,W,D)
        vh = split_kwv(v_win)      # (B,H,T,W,D)

        # 余弦相似度 + 温度
        qn = F.normalize(qh, dim=-1)
        kn = F.normalize(kh, dim=-1)
        sim = (qn.unsqueeze(3) * kn).sum(-1) / self.sim_temperature  # (B,H,T,W)
        if self.use_relative_pos_bias and self.rel_pos_bias is not None:
            sim = sim + self.rel_pos_bias.view(1, self.num_heads, 1, self.W)
        attn_logits = sim
        attn = F.softmax(attn_logits, dim=-1)
        attn = self.attn_drop(attn)
        # 聚合
        out = torch.einsum("bhtw,bhtwd->bhtd", attn, vh)                        # (B,H,T,D)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, self.C)          # (B,T,C)

        # 输出投影 + 残差
        out = self.proj_drop(self.out_proj(out))                                # (B,T,C)
        # 序列表征（供两分支共用）
        seq_feat = self.mamba(out)  # (B, T, C)

        return seq_feat.mean(dim=1)


class LocalUncertaintyAwareGraphAttentionLiteNoMamba(LocalUncertaintyAwareGraphAttentionLite):
    """
    继承自 LocalUncertaintyAwareGraphAttentionLite
    唯一改动：forward 中删除 mamba，直接将 out 输入 detector
    """

    def forward(self, x: torch.Tensor):
        """
        x: (B, T, Cx)
        return:
            score_logit, center
        """
        B, T, Cx = x.shape

        x = self.pos(x)
        xn = self._maybe_norm(x)  # (B,T,C)

        # QKV
        q = self.q_proj(xn)  # (B,T,C)
        k = self.k_proj(xn)
        v = self.v_proj(xn)

        # 局部窗口
        k_win = self._local_unfold(k)  # (B,T,W,C)
        v_win = self._local_unfold(v)  # (B,T,W,C)

        # 多头拆分
        def split_q(t):
            # (B,T,C) -> (B,H,T,D)
            return t.view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        def split_kwv(t):
            # (B,T,W,C) -> (B,H,T,W,D)
            return t.view(B, T, self.W, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)

        qh = split_q(q)       # (B,H,T,D)
        kh = split_kwv(k_win) # (B,H,T,W,D)
        vh = split_kwv(v_win) # (B,H,T,W,D)

        # 余弦相似度 + 温度
        qn = F.normalize(qh, dim=-1)
        kn = F.normalize(kh, dim=-1)
        sim = (qn.unsqueeze(3) * kn).sum(-1) / self.sim_temperature  # (B,H,T,W)

        if self.use_relative_pos_bias and self.rel_pos_bias is not None:
            sim = sim + self.rel_pos_bias.view(1, self.num_heads, 1, self.W)

        attn = F.softmax(sim, dim=-1)
        attn = self.attn_drop(attn)

        # 聚合
        out = torch.einsum("bhtw,bhtwd->bhtd", attn, vh)               # (B,H,T,D)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, self.C) # (B,T,C)

        # 输出投影
        out = self.proj_drop(self.out_proj(out))  # (B,T,C)

        # 这里不经过 mamba，直接送入 detector
        out_det = self.detector(out)

        # 兼容不同 detector 输出
        if isinstance(out_det, tuple):
            if len(out_det) == 2:
                score_logit, center = out_det
            else:
                score_logit, center = out_det[0], None
        else:
            score_logit, center = out_det, None

        return score_logit, center

class PositionEmbedding(nn.Module):
    def __init__(self, size, dim=128):
        super().__init__()
        self.size = size
        self.pe = nn.Embedding(size, dim)
        self.pos_ids = torch.arange(size, dtype=torch.long, device='cuda:0')

    def forward(self, x):
        pos_ids = einops.repeat(self.pos_ids, 'n -> b n', b=len(x))

        embeddings = torch.cat([x, self.pe(pos_ids)], dim=-1)
        return embeddings



    ## 显式相似度计算
class LSUBoundaryTendencyDetector(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=512, pos_dim=128, size=21):
        super(LSUBoundaryTendencyDetector, self).__init__()
        # 🔧 修复1: 正确计算TSM位置编码后的维度
        # TSMPositionEmbedding 会添加 3 * (dim//2) = 3 * 64 = 192 维度
        tsm_pos_dim = 3 * (pos_dim // 2)  # 192
        self.d_dim = input_dim + tsm_pos_dim  # 2048 + 192 = 2240
        self.d_dim = input_dim + pos_dim  # 2048 + 128 = 2176
        self.mambaArg = ModelArgs(
            d_model=self.d_dim,  # 2176
            d_inner=hidden_dim,  # 512
            n_layer=2,
            size = size
        )
        self.hidden_dim = hidden_dim
        self.mamba = Mamba(self.mambaArg)
        self.final_fusion = nn.Linear(self.d_dim + 256 + self.d_dim, self.d_dim)  # (2176 + 256 + 2176) -> 2176
    def forward(self, x):
        # x shape: (batch_size, sequence_length, 2048)
        batch_size, seq_len, _ = x.size()
        # 计算局部对比特征
        lstm_out = self.mamba(x)  # (batch_size, seq_len, 2176)
        return lstm_out

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class TransformerBackbone(nn.Module):
#     """
#     最小版 Transformer Encoder Backbone
#     - 输入: x [B, T, 2048]
#     - 输出: h [B, T, d_model]
#     """
#     def __init__(self, in_dim=2048, d_model=512, nhead=8, num_layers=4, dim_feedforward=2048, dropout=0.1, max_len=1024):
#         super().__init__()
#         self.in_proj = nn.Linear(in_dim, d_model)
#         # 可学习位置编码（最简）
#         self.pos_emb = nn.Embedding(max_len, d_model)
#
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model, nhead=nhead,
#             dim_feedforward=dim_feedforward,
#             dropout=dropout,
#             batch_first=True,   # 让输入输出都是 [B, T, C]
#             activation='gelu'
#         )
#         self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#         self.ln_out = nn.LayerNorm(d_model)
#
#     def forward(self, x):  # x: [B, T, 2048]
#         B, T, _ = x.shape
#         h = self.in_proj(x)  # [B, T, d_model]
#
#         # 位置编码
#         pos_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)  # [B, T]
#         h = h + self.pos_emb(pos_ids)
#
#         # Transformer 编码
#         h = self.encoder(h)           # [B, T, d_model]
#         h = self.ln_out(h)            # [B, T, d_model]
#         return h
#
#
# import torch
# import torch.nn as nn
#
# class FrameWiseHead(nn.Module):
#     """
#     仅对窗口中间步做二分类预测：
#     输入: x (B, T, N)
#     输出: logits (B,)  未过 sigmoid
#     """
#     def __init__(self, in_features: int, hidden: int = 256, dropout: float = 0.1):
#         super().__init__()
#         self.proj = nn.Linear(in_features, hidden)
#         self.dropout = nn.Dropout(dropout)
#         self.cls = nn.Linear(hidden, 1)
#         self.act = nn.Sigmoid()
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         x: (B, T, N)
#         """
#         B, T, N = x.shape
#         mid = T // 2                       # 中间位置（若 T 为偶数，请确保你的数据构造保证中心定义）
#         xc = x[:, mid, :]                  # (B, N)
#
#         h = torch.relu(self.proj(xc))      # (B, H)
#         h = self.dropout(h)
#         logits = self.cls(h).squeeze(-1)   # (B,)
#         logits = self.act(logits)
#         return logits
#
#
# class Model(nn.Module):
#     """
#     拼装：Transformer Backbone + Detector
#     - forward 输出 [B, T] 的逐步 logits
#     """
#     def __init__(self, in_dim=2048, d_model=512, nhead=8, num_layers=4, dim_feedforward=2048, dropout=0.1, max_len=1024):
#         super().__init__()
#         self.backbone = TransformerBackbone(
#             in_dim=in_dim, d_model=d_model, nhead=nhead,
#             num_layers=num_layers, dim_feedforward=dim_feedforward,
#             dropout=dropout, max_len=max_len
#         )
#         self.detector = FrameWiseHead(in_features=d_model)
#
#     def forward(self, x):  # x: [B, T, 2048]
#         h = self.backbone(x)          # [B, T, d_model]
#         logits = self.detector(h)     # [B, T]
#         return logits