import torch
import torch.nn as nn
import torch.nn.functional as F


class SpottingHead(nn.Module):
    def __init__(self, input_dim=2176, time_steps=24, num_detections=9):
        super(SpottingHead, self).__init__()

        self.input_dim = input_dim
        self.time_steps = time_steps
        self.num_detections = num_detections

        # 初始特征处理
        self.conv_initial = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1, input_dim))

        # 时序卷积层
        self.kernel_spot_size = 3

        # 第一层
        self.pad_spot_1 = nn.ZeroPad2d(
            (0, 0, (self.kernel_spot_size - 1) // 2, self.kernel_spot_size - 1 - (self.kernel_spot_size - 1) // 2))
        self.conv_spot_1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(self.kernel_spot_size, 1))

        # 第二层
        self.pad_spot_2 = nn.ZeroPad2d(
            (0, 0, (self.kernel_spot_size - 1) // 2, self.kernel_spot_size - 1 - (self.kernel_spot_size - 1) // 2))
        self.conv_spot_2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(self.kernel_spot_size, 1))

        # 🔑 使用自适应池化，输出固定尺寸
        self.adaptive_pool = nn.AdaptiveMaxPool2d((2, 1))  # 输出时间维度固定为2

        # 固定最终特征维度
        final_feature_dim = 16 * 2

        # 置信度分支
        self.conv_conf = nn.Conv2d(
            in_channels=final_feature_dim,
            out_channels=self.num_detections * 2,
            kernel_size=(1, 1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        batch_size = x.shape[0]

        # 转换输入格式
        x = x.unsqueeze(1)

        # 初始特征提取
        x = F.relu(self.conv_initial(x))  # [batch, 64, time, 1]

        # 卷积层
        conv_spot_1 = F.relu(self.conv_spot_1(self.pad_spot_1(x)))
        conv_spot_2 = F.relu(self.conv_spot_2(self.pad_spot_2(conv_spot_1)))

        # 🔑 使用自适应池化替代多次固定池化
        conv_spot_2_pooled = self.adaptive_pool(conv_spot_2)

        # reshape
        spotting_reshaped = conv_spot_2_pooled.view(batch_size, -1, 1, 1)

        # 置信度预测
        conf_pred = torch.sigmoid(
            self.conv_conf(spotting_reshaped).view(
                batch_size, self.num_detections, 2
            )
        )

        return conf_pred