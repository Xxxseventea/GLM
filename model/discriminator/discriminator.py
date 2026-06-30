from torch import nn


class TemporalDiscriminator(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        # in_dim 是 D。假设你要输出 1 维域 logit。
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1)
        )

    def forward(self, feats):
        # feats: [B,D] or [B,T,D]
        if feats.dim() == 3:
            # 如果是时序特征 => 平均时间维
            feats = feats.mean(dim=1)
        # feats: [B,D]
        out = self.fc(feats)
        return out