import torch
import torch.nn as nn
import torch.nn.functional as F


# ===== 你的 MlpHead：原样保留，不改 =====
class MlpHead(nn.Module):
    def __init__(self, in_dim, hid_dim=512, out_dim=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, hid_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(.1),
            nn.Linear(hid_dim, out_dim, bias=True),
        )
        self.act = nn.Sigmoid()

    def forward(self, x):
        # x shape: [b t d] where t means the number of views
        x1 = self.model(x[:, 21 // 2])   # 取中间帧
        x = self.act(x1)
        return x1, x


# ===== PC 模型：构造 pair 特征，再交给你的 MlpHead =====
class PCModel(nn.Module):
    """
    Pairwise Contrast (PC) 模型：
    - 输入: x [B, T, C]，例如 [B, 21, 2048]
    - 在时间维上做前/后窗口平均，得到 before / after 特征
    - 拼接成 pair 特征 [B, T, 2C]
    - 把 [B, T, 2C] 交给你原来的 MlpHead（里面自己取中间帧）
    """

    def __init__(self,
                 feat_dim=2048,      # C
                 window_size=5,      # 前后各取多少帧做平均
                 mlp_hidden=512):
        super().__init__()
        self.window_size = window_size
        self.feat_dim = feat_dim

        # 这里的 in_dim = 2 * feat_dim，因为要拼接 before + after
        self.head = MlpHead(in_dim=feat_dim * 2,
                            hid_dim=mlp_hidden,
                            out_dim=1)

        # depthwise conv 的平均权重，用于时间滑动平均
        avg_weight = torch.ones(feat_dim, 1, window_size) / window_size
        self.register_buffer("avg_weight", avg_weight)

    def _avg_pool_1d(self, x):
        """
        对时间做滑动平均。
        x: [B, T, C]
        return: [B, T, C]
        """
        B, T, C = x.shape
        p = self.window_size

        # 手动在时间维复制 padding（左边复制第 0 帧，右边复制最后一帧）
        left = x[:, :1, :].expand(B, p, C)  # [B, p, C]
        right = x[:, -1:, :].expand(B, p, C)  # [B, p, C]
        x_padded = torch.cat([left, x, right], dim=1)  # [B, T+2p, C]

        # Conv1d 需要 [B, C, T]
        xp = x_padded.transpose(1, 2)  # [B, C, T+2p]

        out = F.conv1d(xp, self.avg_weight, bias=None, groups=C)  # [B, C, T+p+1]
        out = out[:, :, :T]  # 对齐到原始 T
        out = out.transpose(1, 2)  # [B, T, C]
        return out

    def forward(self, x):
        """
        x: [B, T, C]，例如 [B, 21, 2048]
        返回:
            logits: [B, 1]  -- 你 MlpHead 的输出（中间帧）
            prob:   [B, 1]  -- Sigmoid 后的概率
        """
        B, T, C = x.shape
        assert C == self.feat_dim, f"feat_dim mismatch, got {C}, expected {self.feat_dim}"
        assert T == 21, f"MlpHead 里写死了 21//2，这里 T 也必须是 21，当前是 {T}"

        # before: 每个时间步左侧 window_size 帧平均
        before_avg = self._avg_pool_1d(x)          # [B, T, C]

        # after: 每个时间步右侧 window_size 帧平均（用时间翻转实现）
        x_rev = torch.flip(x, dims=[1])            # [B, T, C]
        after_avg_rev = self._avg_pool_1d(x_rev)   # [B, T, C]
        after_avg = torch.flip(after_avg_rev, dims=[1])  # [B, T, C]

        # 拼接 pair 特征: [B, T, 2C]
        pair_feat = torch.cat([before_avg, after_avg], dim=-1)  # [B, 21, 2C]

        # 直接交给你原来的 MlpHead（内部会取 pair_feat[:, 21//2]）
        logits, prob = self.head(pair_feat)  # [B, 1], [B, 1]
        return logits, prob

