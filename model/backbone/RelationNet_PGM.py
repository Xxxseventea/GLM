# model/backbone/RelationNet.py
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbone.subnet import SubnetLinear
from model.backbone.mymamba import Mamba, ModelArgs
from model.detector.event_detector import FrameWiseHead
from model.detector.scene_detector import MlpHead


class PositionEmbedding(nn.Module):
    def __init__(self, size, dim=128):
        super().__init__()
        self.size = size
        self.pe = nn.Embedding(size, dim)

    def forward(self, x):
        B, T, _ = x.shape
        pos_ids = torch.arange(T, dtype=torch.long, device=x.device)
        pos_ids = einops.repeat(pos_ids, 'n -> b n', b=B)
        embeddings = torch.cat([x, self.pe(pos_ids)], dim=-1)
        return embeddings


class LSUBoundaryTendencyDetector(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=512, pos_dim=128, size=21):
        super().__init__()
        self.d_dim = input_dim + pos_dim  # 2176
        self.mambaArg = ModelArgs(
            d_model=self.d_dim,
            d_inner=hidden_dim,
            n_layer=2,
            size=size,
        )
        self.mamba = Mamba(self.mambaArg)

    def forward(self, x):
        return self.mamba(x)


class LocalUncertaintyAwareGraphAttentionLite(nn.Module):
    """
    改造版：
    - q/k/v/out_proj 使用 SubnetLinear（支持 mask）
    - last 是 ModuleDict，包含 'scene' 和 'event' 两个 head
    - forward 增加 task_name / mask / mode 参数
    """

    def __init__(
        self,
        dim_in: int = 2048,
        num_heads: int = 8,
        window_size: int = 9,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        sparsity: float = 0.5,
        sim_temperature: float = 0.07,
        size: int = 21,
        use_relative_pos_bias: bool = True,
    ):
        super().__init__()
        assert window_size % 2 == 1
        C = dim_in + 128
        assert C % num_heads == 0

        self.C = C
        self.num_heads = num_heads
        self.head_dim = C // num_heads
        self.W = window_size
        self.sim_temperature = sim_temperature
        self.use_relative_pos_bias = use_relative_pos_bias

        self.norm = nn.LayerNorm(C)

        # ⭐ Subnet Linear
        self.q_proj = SubnetLinear(C, C, bias=True, sparsity=sparsity)
        self.k_proj = SubnetLinear(C, C, bias=True, sparsity=sparsity)
        self.v_proj = SubnetLinear(C, C, bias=True, sparsity=sparsity)
        self.out_proj = SubnetLinear(C, C, bias=True, sparsity=sparsity)

        if use_relative_pos_bias:
            self.rel_pos_bias = nn.Parameter(torch.zeros(num_heads, window_size))
            nn.init.trunc_normal_(self.rel_pos_bias, std=0.02)

        self.proj_drop = nn.Dropout(dropout)
        self.attn_drop = nn.Dropout(attn_dropout)

        # 位置 + 序列建模（mamba 这里不参与 mask，简化版）
        self.pos = PositionEmbedding(size=size)
        self.mamba = LSUBoundaryTendencyDetector(size=size)

        # ⭐ 两个任务头
        self.last = nn.ModuleDict({
            'event': FrameWiseHead(in_features=2176),
            'scene': MlpHead(in_dim=2176, hid_dim=512),
        })

    @torch.no_grad()
    def _local_unfold(self, x):
        B, T, C = x.shape
        h = self.W // 2
        x_pad = F.pad(x.transpose(1, 2), (h, h), mode="reflect").transpose(1, 2)
        x_unf = x_pad.transpose(1, 2).unfold(dimension=-1, size=self.W, step=1)
        x_win = x_unf.permute(0, 2, 3, 1).contiguous()
        return x_win

    def forward(self, x, task_name, mask=None, mode='train'):
        """
        x: (B, T, 2048)
        task_name: 'scene' | 'event'
        mask: dict[str -> tensor] | None
        mode: 'train' | 'test' | 'valid'
        """
        B, T, _ = x.shape
        x = self.pos(x)            # (B, T, 2176)
        xn = self.norm(x)          # (B, T, 2176)

        def get_m(name):
            if mask is None:
                return None
            return mask.get(name, None)

        q = self.q_proj(xn, weight_mask=get_m('q_proj.weight'), mode=mode)
        k = self.k_proj(xn, weight_mask=get_m('k_proj.weight'), mode=mode)
        v = self.v_proj(xn, weight_mask=get_m('v_proj.weight'), mode=mode)

        k_win = self._local_unfold(k)
        v_win = self._local_unfold(v)

        def split_q(t):
            return t.view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        def split_kwv(t):
            return t.view(B, T, self.W, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)

        qh = split_q(q)
        kh = split_kwv(k_win)
        vh = split_kwv(v_win)

        qn_ = F.normalize(qh, dim=-1)
        kn_ = F.normalize(kh, dim=-1)
        sim = (qn_.unsqueeze(3) * kn_).sum(-1) / self.sim_temperature

        if self.use_relative_pos_bias:
            sim = sim + self.rel_pos_bias.view(1, self.num_heads, 1, self.W)

        attn = F.softmax(sim, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.einsum("bhtw,bhtwd->bhtd", attn, vh)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, self.C)

        out = self.out_proj(out, weight_mask=get_m('out_proj.weight'), mode=mode)
        out = self.proj_drop(out)

        seq_feat = self.mamba(out)  # (B, T, 2176)

        # 任务头
        out_det = self.last[task_name](seq_feat)
        if isinstance(out_det, tuple):
            score_logit, center = out_det
        else:
            score_logit, center = out_det, None

        return seq_feat, (score_logit, center)

    def get_masks(self, task_id=None):
        """
        返回当前 task 的 binary mask（共享层），不包含任务头
        """
        masks = {}
        for name, module in self.named_modules():
            if isinstance(module, SubnetLinear):
                # 跳过 head 内部
                if name.startswith('last.'):
                    continue
                masks[name + '.weight'] = module.get_mask()
        return masks
