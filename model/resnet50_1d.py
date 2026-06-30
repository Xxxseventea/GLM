import torch
import torch.nn as nn
import torch.nn.functional as F


# ========== Same padding 的 1D conv（和原码风格一致） ==========
class MyConv1dPadSame(nn.Module):
    """Conv1d with 'same' padding (for stride=1) / 'same'-like for stride>1."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, bias=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=0, groups=groups, bias=bias)

    def forward(self, x):
        # 动态 SAME padding
        L_in = x.shape[-1]
        L_out = (L_in + self.stride - 1) // self.stride
        pad = max((L_out - 1) * self.stride + self.kernel_size - L_in, 0)
        x = F.pad(x, [pad // 2, pad - pad // 2])
        return self.conv(x)


class MyMaxPool1dPadSame(nn.Module):
    def __init__(self, kernel_size, stride=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pool = nn.MaxPool1d(kernel_size, stride=stride, padding=0)

    def forward(self, x):
        L_in = x.shape[-1]
        L_out = (L_in + self.stride - 1) // self.stride
        pad = max((L_out - 1) * self.stride + self.kernel_size - L_in, 0)
        x = F.pad(x, [pad // 2, pad - pad // 2])
        return self.pool(x)


# ========== Bottleneck（1x1 -> 3x3 -> 1x1，expansion=4）==========
class Bottleneck1D(nn.Module):
    expansion = 4

    def __init__(self, in_channels, planes, kernel_size=3, stride=1,
                 groups=1, use_bn=True, use_do=False, do_p=0.1):
        """
        in_channels : 输入通道
        planes      : 中间层通道（瓶颈通道），最终输出 planes * 4
        stride      : 只作用在 3x3 的 conv 上（标准 ResNet 做法）
        """
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = planes * self.expansion
        self.stride       = stride
        self.use_bn       = use_bn
        self.use_do       = use_do

        self.do1 = nn.Dropout(do_p) if use_do else nn.Identity()
        self.do2 = nn.Dropout(do_p) if use_do else nn.Identity()
        self.do3 = nn.Dropout(do_p) if use_do else nn.Identity()

        # 1x1 降维
        self.conv1 = MyConv1dPadSame(in_channels, planes, kernel_size=1, stride=1)
        self.bn1   = nn.BatchNorm1d(planes) if use_bn else nn.Identity()

        # 3x3 主卷积（下采样发生在这里）
        self.conv2 = MyConv1dPadSame(planes, planes, kernel_size=kernel_size,
                                     stride=stride, groups=groups)
        self.bn2   = nn.BatchNorm1d(planes) if use_bn else nn.Identity()

        # 1x1 升维
        self.conv3 = MyConv1dPadSame(planes, self.out_channels, kernel_size=1, stride=1)
        self.bn3   = nn.BatchNorm1d(self.out_channels) if use_bn else nn.Identity()

        # shortcut
        if stride != 1 or in_channels != self.out_channels:
            self.downsample = nn.Sequential(
                MyConv1dPadSame(in_channels, self.out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(self.out_channels) if use_bn else nn.Identity()
            )
        else:
            self.downsample = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.downsample(x)

        out = self.conv1(x); out = self.bn1(out); out = self.relu(out); out = self.do1(out)
        out = self.conv2(out); out = self.bn2(out); out = self.relu(out); out = self.do2(out)
        out = self.conv3(out); out = self.bn3(out); out = self.do3(out)

        out = out + identity
        out = self.relu(out)
        return out


# ========== ResNet-50 1D ==========
class ResNet50_1D(nn.Module):
    """
    严格的 1D 版 ResNet-50：Bottleneck + layers=[3, 4, 6, 3]

    Input  : x  [B, in_channels, L]
    Output : out [B, n_classes]
    """
    def __init__(self,
                 in_channels=1,
                 n_classes=2,
                 base_filters=64,
                 kernel_size=3,
                 groups=1,
                 use_bn=True,
                 use_do=False,
                 do_p=0.1,
                 # 下采样控制：strides_per_stage 对应 layer1~layer4 的 stride
                 # 标准 ResNet-50: [1, 2, 2, 2] + stem 下采样 4 倍 = 总 32 倍
                 # 短序列场景可把后面几段改成 1，例如 [1,1,1,1]
                 strides_per_stage=(1, 2, 2, 2),
                 stem_stride=2,
                 stem_pool=True,
                 verbose=False):
        super().__init__()
        assert len(strides_per_stage) == 4
        self.verbose = verbose
        self.kernel_size = kernel_size
        self.groups = groups
        self.use_bn = use_bn
        self.use_do = use_do
        self.do_p  = do_p

        layers_cfg = [3, 4, 6, 3]        # ResNet-50 标配
        planes_cfg = [base_filters * m for m in (1, 2, 4, 8)]  # 64,128,256,512

        # ---- stem ----
        self.stem_conv = MyConv1dPadSame(in_channels, base_filters,
                                         kernel_size=7, stride=stem_stride)
        self.stem_bn   = nn.BatchNorm1d(base_filters) if use_bn else nn.Identity()
        self.stem_relu = nn.ReLU(inplace=True)
        self.stem_pool = MyMaxPool1dPadSame(kernel_size=3, stride=2) if stem_pool else nn.Identity()

        # ---- 4 个 stage ----
        self.in_c = base_filters
        self.layer1 = self._make_layer(planes_cfg[0], layers_cfg[0], stride=strides_per_stage[0])
        self.layer2 = self._make_layer(planes_cfg[1], layers_cfg[1], stride=strides_per_stage[1])
        self.layer3 = self._make_layer(planes_cfg[2], layers_cfg[2], stride=strides_per_stage[2])
        self.layer4 = self._make_layer(planes_cfg[3], layers_cfg[3], stride=strides_per_stage[3])

        # ---- head ----
        self.final_bn   = nn.BatchNorm1d(self.in_c) if use_bn else nn.Identity()
        self.final_relu = nn.ReLU(inplace=True)
        self.avgpool    = nn.AdaptiveAvgPool1d(1)
        self.dense      = nn.Linear(self.in_c, n_classes)
        self.reg        = nn.Linear(self.in_c, n_classes)

        self._init_weights()

    def _make_layer(self, planes, n_blocks, stride):
        blocks = []
        # 第一个 block 负责通道扩张 / 下采样
        blocks.append(Bottleneck1D(self.in_c, planes, kernel_size=self.kernel_size,
                                   stride=stride, groups=self.groups,
                                   use_bn=self.use_bn, use_do=self.use_do, do_p=self.do_p))
        self.in_c = planes * Bottleneck1D.expansion
        # 剩余 block 保持尺寸
        for _ in range(1, n_blocks):
            blocks.append(Bottleneck1D(self.in_c, planes, kernel_size=self.kernel_size,
                                       stride=1, groups=self.groups,
                                       use_bn=self.use_bn, use_do=self.use_do, do_p=self.do_p))
        return nn.Sequential(*blocks)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.transpose(1, 2).contiguous()
        if self.verbose: print('input  :', x.shape)
        x = self.stem_conv(x)
        if self.use_bn: x = self.stem_bn(x)
        x = self.stem_relu(x)
        x = self.stem_pool(x)
        if self.verbose: print('stem   :', x.shape)

        x = self.layer1(x);  self.verbose and print('layer1 :', x.shape)
        x = self.layer2(x);  self.verbose and print('layer2 :', x.shape)
        x = self.layer3(x);  self.verbose and print('layer3 :', x.shape)
        x = self.layer4(x);  self.verbose and print('layer4 :', x.shape)

        x = self.final_bn(x)
        x = self.final_relu(x)
        x = self.avgpool(x).flatten(1)
        if self.verbose: print('pool   :', x.shape)
        out = self.dense(x)
        center_raw = self.reg(x).squeeze(1)    # (B,)
        center = torch.sigmoid(center_raw)     # 映射到 [0,1]
        if self.verbose: print('dense  :', out.shape)
        return out, center


# ========== 工厂函数 ==========
def resnet50_1d(in_channels=2048, n_classes=2, short_seq=False, **kwargs):
    """
    in_channels : 输入通道（如 shot 特征是 2048）
    n_classes   : 输出类别
    short_seq   : True 时自动关闭 stem 下采样，所有 stage stride=1，适合 T<=32 的短序列
    """
    if short_seq:
        kwargs.setdefault('stem_stride', 1)
        kwargs.setdefault('stem_pool',   False)
        kwargs.setdefault('strides_per_stage', (1, 1, 1, 1))
    return ResNet50_1D(in_channels=in_channels, n_classes=n_classes, **kwargs)


# ========== 测试 ==========
if __name__ == '__main__':
    # # 1) 标准版（适合长序列，比如原始波形 L>=224）
    # T = 21
    # in_channel = 2048
    # net = resnet50_1d(in_channels=in_channel, n_classes=1, verbose=True)
    # x = torch.randn(2, T, in_channel)
    # print('out:', net(x).shape)
    # print('params:', sum(p.numel() for p in net.parameters()) / 1e6, 'M')
    # print('-' * 60)

    # 2) 短序列版（适合你 T=21 的 shot 特征场景）
    net = resnet50_1d(in_channels=2048, n_classes=1, short_seq=True, verbose=True)
    x = torch.randn(2, 21, 2048)     # [B, C, T]
    print('out:', net(x).shape)
    print('params:', sum(p.numel() for p in net.parameters()) / 1e6, 'M')