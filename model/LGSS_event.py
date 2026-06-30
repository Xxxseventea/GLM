import torch
import torch.nn as nn
import torch.nn.functional as F

class Cos(nn.Module):
    def __init__(self, shot_num=20, sim_channel=512):
        super(Cos, self).__init__()
        self.shot_num = shot_num
        self.channel = sim_channel
        self.conv1 = nn.Conv2d(1, self.channel, kernel_size=(self.shot_num//2, 1))

    def forward(self, x):  # [batch_size, seq_len, shot_num, feat_dim]
        x = x.view(-1, 1, x.shape[2], x.shape[3])
        part1, part2 = torch.split(x, [self.shot_num//2]*2, dim=2)
        # batch_size*seq_len, 1, [self.shot_num//2], feat_dim
        part1 = self.conv1(part1).squeeze()
        part2 = self.conv1(part2).squeeze()
        x = F.cosine_similarity(part1, part2, dim=2)  # batch_size,channel
        return x
    

class BNet(nn.Module):
    def __init__(self, shot_num=20, sim_channel=512):
        super(BNet, self).__init__()
        self.shot_num = shot_num
        self.channel = sim_channel
        self.conv1 = nn.Conv2d(1, self.channel, kernel_size=(shot_num, 1))
        self.max3d = nn.MaxPool3d(kernel_size=(self.channel, 1, 1))
        self.cos = Cos(shot_num, sim_channel)

    def forward(self, x):  # [batch_size, seq_len, shot_num, feat_dim]
        context = x.view(x.shape[0]*x.shape[1], 1, -1, x.shape[-1])
        context = self.conv1(context)  # batch_size*seq_len,512,1,feat_dim
        context = self.max3d(context)  # batch_size*seq_len,1,1,feat_dim
        context = context.squeeze()
        sim = self.cos(x)
        bound = torch.cat((context, sim), dim=1)
        return bound


class LGSSEventDet(nn.Module):
    """Event detection with LGSS BNet + FC classifier.

    Input:  [B, 21, 2048]  -- drops the last frame inside forward.
    Output: [B, 2]         -- logits for non-event / event.

    Required cfg fields:
        cfg.shot_num             = 20           # after dropping 1 frame
        cfg.model.place_feat_dim = 2048
        cfg.model.sim_channel    = 512
    """

    def __init__(self, shot_num=20, place_feat_dim=2048, sim_channel=512):
        super().__init__()
        self.shot_num = shot_num
        self.feat_dim = place_feat_dim
        self.sim_channel = sim_channel

        self.bnet = BNet(shot_num, sim_channel)
        self.fc1 = nn.Linear(self.feat_dim + self.sim_channel, 100)
        self.fc2 = nn.Linear(100, 1)
        self.reg = nn.Linear(100, 1)

    def forward(self, x, task_name=None):  # x: [B, 21, 2048]
        x = x[:, :self.shot_num, :]        # drop last frame -> [B, 20, 2048]
        x = x.unsqueeze(1)                 # [B, 1, 20, 2048]  (seq_len=1)

        # BNet uses .squeeze() internally which collapses B=1 to a 1D tensor.
        # Duplicate the batch so squeeze keeps a batch axis, then trim back.
        B = x.shape[0]
        if B == 1:
            x = x.repeat(2, 1, 1, 1)

        feat = self.bnet(x)                # [B, feat_dim + sim_channel]
        feat = F.relu(self.fc1(feat))
        out = self.fc2(feat)

        center_raw = self.reg(feat).squeeze(1)    # (B,)
        center = torch.sigmoid(center_raw)     # 映射到 [0,1]
        if B == 1:
            out = out[:1]
        return out, center
